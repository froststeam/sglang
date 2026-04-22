import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch.cuda
import torch.distributed
from torch.distributed import P2POp, ProcessGroup

from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._rebalance_async = self._server_args.enable_eplb_rebalance_async
        self._disable_eplb_warmup = self._server_args.disable_eplb_warmup
        self._batch_isend_irecv_warmuped = False
        self._src_rank = 0
        self._world_size = self._model_runner.tp_group.world_size
        self._local_rank = self._model_runner.tp_rank
        self._num_gpu_per_node = (
            self._model_runner.tp_size
            * self._model_runner.pp_size
            // self._server_args.nnodes
        )
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_experts_per_chunk = (
            self._server_args.eplb_rebalance_experts_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        if not self._disable_eplb_warmup and not self._batch_isend_irecv_warmuped:
            self._warmup()
        next(self._main_generator)
        self._step_counter += 1

    def _warmup(self):
        logger.info("Warmup batch_isend_irecv...")
        moe_ep_size = self._model_runner.moe_ep_size
        moe_ep_rank = self._model_runner.moe_ep_rank
        device = self._model_runner.device
        fake_weight = torch.empty((moe_ep_size, 2), device=device, dtype=torch.int8)
        fake_buffer = torch.empty((moe_ep_size, 2), device=device, dtype=torch.int8)
        self._warmup_batch_isend_irecv(
            moe_ep_rank, moe_ep_size, fake_weight, fake_buffer
        )
        self._batch_isend_irecv_warmuped = True
        logger.info("Warmup batch_isend_irecv done.")

    def _warmup_batch_isend_irecv(
        self, moe_ep_rank, moe_ep_size, fake_weight, fake_buffer
    ):
        """
        Warmup for `torch.distributed.batch_isend_irecv()` between all send-recv ranks.
        The first call to `torch.distributed.batch_isend_irecv()` is often slow for reasons
        that are not about the send/recv itself, but about one-time setup that happens
        the first time you do any point-to-point distributed work in that process,
        such as connection establishment, creation of communicators/channel setup.

        An example for EP8:
        Send(Recv):
        rank-0: send to (recv from)   1 2 3 4 5 6 7
        rank-1: send to (recv from)     2 3 4 5 6 7
        rank-2: send to (recv from)       3 4 5 6 7
        rank-3: send to (recv from)         4 5 6 7
        rank-4: send to (recv from)           5 6 7
        rank-5: send to (recv from)             6 7
        rank-6: send to (recv from)               7
        Recv(Send):
        rank-1: recv from (send to)   0
        rank-2: recv from (send to)   0 1
        rank-3: recv from (send to)   0 1 2
        rank-4: recv from (send to)   0 1 2 3
        rank-5: recv from (send to)   0 1 2 3 4
        rank-6: recv from (send to)   0 1 2 3 4 5
        rank-7: recv from (send to)   0 1 2 3 4 5 6

        Parameters:
            moe_ep_rank: rank in EP group
            moe_ep_size: EP group size
            fake_weight: [moe_ep_size, 2], the fake weight of each rank
            fake_buffer: [moe_ep_size, 2], the fake buffer of each rank
        """
        p2p_ops = []
        # Prepare send(recv)
        for dst_or_src_rank in range(moe_ep_rank + 1, moe_ep_size):
            # Send
            p2p_ops.append(
                P2POp(
                    op=torch.distributed.isend,
                    tensor=fake_weight[dst_or_src_rank][0],
                    peer=get_moe_ep_group().ranks[dst_or_src_rank],
                    group=get_moe_ep_group().device_group,
                )
            )
            # Recv
            p2p_ops.append(
                P2POp(
                    op=torch.distributed.irecv,
                    tensor=fake_buffer[dst_or_src_rank][1],
                    peer=get_moe_ep_group().ranks[dst_or_src_rank],
                    group=get_moe_ep_group().device_group,
                )
            )
        # Prepare recv(send)
        for src_or_dst_rank in range(0, moe_ep_rank):
            # Recv
            p2p_ops.append(
                P2POp(
                    op=torch.distributed.irecv,
                    tensor=fake_buffer[src_or_dst_rank][0],
                    peer=get_moe_ep_group().ranks[src_or_dst_rank],
                    group=get_moe_ep_group().device_group,
                )
            )
            # Send
            p2p_ops.append(
                P2POp(
                    op=torch.distributed.isend,
                    tensor=fake_weight[src_or_dst_rank][1],
                    peer=get_moe_ep_group().ranks[src_or_dst_rank],
                    group=get_moe_ep_group().device_group,
                )
            )

        torch.get_device_module().synchronize()
        time_start = time.time()
        reqs = torch.distributed.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()
        torch.get_device_module().synchronize()
        time_end = time.time()
        msg = f" time={time_end - time_start:.3f}s"
        logger.info(f"Warmup batch_isend_irecv cost: {msg}")

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            self.initial_rebalance()

            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def initial_rebalance(self):
        self._step_counter = 0
        if self._rebalance_async:
            self._stop_transfer = False
            self._begin_transfer_step = None
            self._compute_ongoing = False
            self._tp_sync_ongoing = False
            self._rebalance_result = None

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()
        if self._rebalance_async:
            # async: overlap eplb computing with model_runner-forward
            yield from self.compute()
            if self._stop_transfer:
                return
            yield from self.tp_sync()
            # ensure all the tp enter transfer stage at the same forward-pass
            while (
                self._begin_transfer_step is None
                or self._step_counter < self._begin_transfer_step
            ):
                yield
        else:
            # synchronous mode: directly run on the main thread
            dump_record_output = get_global_expert_distribution_recorder().dump_record(
                output_mode="object"
            )
            logical_count = dump_record_output["logical_count"]
            average_utilization_rate_over_window = dump_record_output[
                "average_utilization_rate_over_window"
            ]
            # Check whether rebalancing is needed
            if not self._check_rebalance_needed(average_utilization_rate_over_window):
                return
            self._rebalance_result = ExpertLocationMetadata.init_by_eplb(
                self._server_args, self._model_runner.model_config, logical_count
            )
        # ------------- real parameter transfer --------------------

        yield from self.transfer_parameter()

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def compute(self):
        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]
        # Check whether rebalancing is needed
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            self._stop_transfer = True
            return
        # use rank-src broadcast to make `logical_count_sum` identical
        logical_count_sum = logical_count.sum(dim=0).clone()
        self._model_runner.tp_group.broadcast(logical_count_sum, src=self._src_rank)
        yield
        torch.cuda.synchronize()
        self._compute_ongoing = True
        self._compute_thread = threading.Thread(
            target=self._compute_expert_metadata, args=(logical_count_sum,), daemon=True
        )
        self._compute_thread.start()
        yield
        # spin until the computation completes
        while self._compute_ongoing:
            yield

    def _compute_expert_metadata(self, logical_count):
        torch.cuda.set_device(self._local_rank % self._num_gpu_per_node)
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )
        self._rebalance_result = expert_location_metadata
        self._compute_ongoing = False

    def tp_sync(self):
        """Barrier-like handshake among all TP ranks.
        Each rank sends a step counter to every other rank and
        waits for all peers' counters.  When all messages are completed,
        we know that the new expert mapping is globally visible and can
        be applied safely.
        """
        self._send_signal_step = self._step_counter
        send_works, recv_works, buffer_kept = self._gen_rec_send_works()
        self._tp_sync_ongoing = True
        self._tp_sync_thread = threading.Thread(
            target=self._wait_for_tp_sync_signals,
            args=(
                send_works,
                recv_works,
            ),
            daemon=True,
        )
        self._tp_sync_thread.start()
        yield
        while self._tp_sync_ongoing:
            yield
        # the largest counter among all ranks indicates the earliest
        # forward-pass index after which the new mapping becomes valid
        max_tensor = torch.stack(buffer_kept).max()
        # Add a buffer of 2 steps to ensure all ranks see the new step before acting on it.
        self._begin_transfer_step = max(max_tensor.item(), self._send_signal_step) + 2

    def _gen_rec_send_works(self):
        # Using the dedicated CPU(Gloo) process-group avoids touching NCCL streams and
        # therefore never blocks the GPU-side decoding kernels that are running in parallel.
        group = self._model_runner.tp_group.cpu_group
        send_works = []
        recv_buffer_kept = []
        recv_works = []
        for src in range(self._world_size):
            if src == self._local_rank:
                continue
            w, t = self._recv_single_signal(src, group)
            recv_works.append(w)
            recv_buffer_kept.append(t)
        for dst in range(self._world_size):
            if dst == self._local_rank:
                continue
            w, t = self._send_single_signal(dst, self._send_signal_step, group)
            send_works.append(w)
        return send_works, recv_works, recv_buffer_kept

    def _send_single_signal(
        self,
        dst_rank_in_group: int,
        value: int,
        group: Optional[ProcessGroup] = None,
    ) -> Tuple[Optional[torch.distributed.Work], torch.Tensor]:
        signal_tensor = torch.tensor([value], dtype=torch.long)
        w = torch.distributed.isend(
            signal_tensor,
            dst=self._model_runner.tp_group.ranks[dst_rank_in_group],
            group=group,
        )
        return w, signal_tensor

    def _recv_single_signal(
        self,
        src_rank_in_group: int,
        group: Optional[ProcessGroup] = None,
    ) -> Tuple[Optional[torch.distributed.Work], torch.Tensor]:
        signal_tensor = torch.empty(1, dtype=torch.long)
        w = torch.distributed.irecv(
            signal_tensor,
            src=self._model_runner.tp_group.ranks[src_rank_in_group],
            group=group,
        )
        return w, signal_tensor

    def _wait_for_tp_sync_signals(self, send_works, recv_works):
        works = send_works + recv_works
        for w in works:
            w.wait()
        self._tp_sync_ongoing = False

    def transfer_parameter(self):
        expert_location_metadata = self._rebalance_result
        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            yield from self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
                rebalance_experts_per_chunk=self._rebalance_experts_per_chunk,
            )

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        if average_utilization_rate_over_window is None:
            return True

        if (
            average_utilization_rate_over_window
            > self._server_args.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {self._server_args.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
