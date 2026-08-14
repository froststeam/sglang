from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_utils import (
    resolve_dflash_verify_mask_policy,
    top_k_renorm_prob,
    top_p_renorm_prob,
)
from sglang.srt.speculative.dspark_info import DSparkDraftInput, DSparkVerifyInput
from sglang.srt.speculative.dspark_components.dspark_ragged import (
    DSparkRaggedPlanner,
    RaggedVerifyMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


class DSparkWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device
        self.verify_token_num = int(server_args.speculative_num_draft_tokens)
        self.block_size = self.verify_token_num - 1
        self.confidence_threshold = float(
            server_args.speculative_dspark_confidence_threshold
        )
        self._logged_first_verify = False
        self._warned_non_greedy = False

        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_server_args.context_length = target_worker.model_runner.model_config.context_len
        # Keep speculative_num_draft_tokens as the global verify window (gamma + 1).
        # Draft width gamma is derived via
        # SpeculativeAlgorithm.get_num_tokens_per_req_for_target_verify(...).

        target_model = self.target_worker.model_runner.model
        self.target_embed_tokens = self._resolve_target_embed_tokens(target_model)
        target_lm_head = getattr(target_model, "lm_head", None)
        if target_lm_head is None:
            raise RuntimeError("DSPARK requires the target model to expose lm_head.")

        def attach_target_vocab(draft_model):
            draft_model.attach_shared_modules(
                embed_tokens=self.target_embed_tokens,
                lm_head=target_lm_head,
            )

        saved_server_args = get_global_server_args()
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=target_req_to_token_pool,
            token_to_kv_pool_allocator=target_token_to_kv_pool_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
            model_load_callback=attach_target_vocab,
        )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model

        self.ragged_planner = DSparkRaggedPlanner(
            worker=self, gamma=self.block_size, server_args=server_args
        )

        model_block_size = getattr(self.draft_model, "block_size", None)
        if model_block_size is not None and int(model_block_size) != self.block_size:
            logger.warning(
                "DSPARK block size mismatch: using gamma=%s from "
                "speculative_num_draft_tokens=%s but draft config block_size=%s.",
                self.block_size,
                self.verify_token_num,
                model_block_size,
            )
        if self.confidence_threshold != 0.0:
            logger.warning(
                "DSPARK v1/no-overlap currently uses fixed full-block proposals; confidence_threshold=%s is ignored.",
                self.confidence_threshold,
            )

        if self.tp_rank == 0:
            logger.info(
                "Initialized DSPARK v1/no-overlap worker. model=%s, block_size=%s, verify_token_num=%s, confidence_threshold=%s",
                server_args.speculative_draft_model_path,
                self.block_size,
                self.verify_token_num,
                self.confidence_threshold,
            )

        self._pos_offsets = torch.arange(
            self.verify_token_num, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None
        self._draft_block_positions_buf: Optional[torch.Tensor] = None
        self._verify_tokens_buf: Optional[torch.Tensor] = None
        self._verify_positions_buf: Optional[torch.Tensor] = None
        self._draft_block_end_buf: Optional[torch.Tensor] = None
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None
        self._draft_block_spec_info = DSparkVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    @staticmethod
    def _resolve_target_embed_tokens(target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    def clear_cache_pool(self):
        pass

    def _ensure_buffers(self, bs: int) -> None:
        cap = 0 if self._draft_block_ids_buf is None else int(self._draft_block_ids_buf.shape[0])
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        self._draft_block_ids_buf = torch.empty(
            (new_cap, self.block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, self.block_size), dtype=torch.int64, device=device
        )
        self._verify_tokens_buf = torch.empty(
            (new_cap, self.verify_token_num), dtype=torch.long, device=device
        )
        self._verify_positions_buf = torch.empty(
            (new_cap, self.verify_token_num), dtype=torch.int64, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        pos2d: torch.Tensor,
        mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if pos2d.ndim != 2:
            raise RuntimeError(
                f"{context} expected 2D positions, got shape={tuple(pos2d.shape)}."
            )
        if mask.shape != pos2d.shape:
            raise RuntimeError(
                f"{context} mask/position shape mismatch: {tuple(mask.shape)} vs {tuple(pos2d.shape)}."
            )

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        table_width = int(req_to_token.shape[1])
        if table_width <= 0:
            if bool(mask.any().item()):
                raise RuntimeError(
                    f"{context} req_to_token table is empty but gather mask is non-empty."
                )
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        safe_pos2d = pos2d.masked_fill(~mask, 0)
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DSparkDraftInput,
    ) -> None:
        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError("DSPARK draft state missing target_hidden context features.")
        if draft_input.ctx_lens.numel() != bs:
            raise RuntimeError(
                f"DSPARK ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError(
                f"DSPARK draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.target_hidden = draft_input.target_hidden[:0]
            return

        target_req_to_token = batch.req_to_token_pool.req_to_token
        req_pool_indices = batch.req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens
        if ctx_lens.dtype != torch.int32:
            ctx_lens = ctx_lens.to(torch.int32)
        if ctx_lens.device != device:
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)

        if bs == 1:
            max_ctx = int(total_ctx)
            if max_ctx <= self._pos_offsets.numel():
                r = self._pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = ctx_start[:, None] + r[None, :]
            ctx_cache_loc = target_req_to_token[req_pool_indices[:, None], pos2d].reshape(-1).to(torch.int64)
            ctx_positions = pos2d.reshape(-1)
        else:
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(self.verify_token_num)
            if max_ctx <= 0:
                raise RuntimeError(f"DSPARK invalid max_ctx={max_ctx} for KV append.")

            if max_ctx <= self._pos_offsets.numel():
                r = self._pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]
            pos2d = ctx_start[:, None] + r
            mask = r < ctx_lens[:, None]
            ctx_cache_loc = self._gather_req_to_token_masked(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                pos2d=pos2d,
                mask=mask,
                context="DSPARK target hidden KV append",
            )
            ctx_positions = pos2d[mask]

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(draft_input.target_hidden)
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    f"DSPARK ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )
            for layer in self.draft_model.layers:
                attn = layer.self_attn
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(ctx_positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                    attn.attn,
                    ctx_cache_loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

        draft_input.draft_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]

    def _greedy_sample_from_vocab_parallel_logits(
        self,
        *,
        local_logits: torch.Tensor,
        lm_head,
    ) -> torch.Tensor:
        if local_logits.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=local_logits.device)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        if not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DSPARK greedy sampling requires a vocab-parallel head with `shard_indices`."
            )

        shard = lm_head.shard_indices
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)
        num_tokens = int(local_logits.shape[0])

        if num_org > 0:
            local_max, local_arg = torch.max(local_logits[:, :num_org], dim=-1)
        else:
            local_max = torch.full(
                (num_tokens,),
                torch.finfo(local_logits.dtype).min,
                dtype=local_logits.dtype,
                device=local_logits.device,
            )
            local_arg = torch.zeros((num_tokens,), dtype=torch.int64, device=local_logits.device)

        if num_added > 0:
            added_slice_start = num_org_padded
            added_slice_end = num_org_padded + num_added
            added_max, added_arg = torch.max(
                local_logits[:, added_slice_start:added_slice_end], dim=-1
            )
            use_added = added_max > local_max
            local_max = torch.where(use_added, added_max, local_max)
            local_arg = torch.where(
                use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
            )

        if num_added == 0:
            global_ids = local_arg + org_vocab_start
        else:
            global_ids = torch.empty(
                (num_tokens,), dtype=torch.int64, device=local_logits.device
            )
            is_base = local_arg < num_org
            global_ids[is_base] = org_vocab_start + local_arg[is_base]
            global_ids[~is_base] = added_vocab_start + (
                local_arg[~is_base] - num_org_padded
            )

        if tp_size == 1:
            return global_ids.to(torch.long)

        needed = tp_size * num_tokens
        if (
            self._draft_greedy_gather_cap < needed
            or self._draft_greedy_gathered_max_buf is None
            or self._draft_greedy_gathered_ids_buf is None
            or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
            or self._draft_greedy_gathered_max_buf.device != local_logits.device
        ):
            self._draft_greedy_gathered_max_buf = torch.empty(
                (needed,), dtype=local_max.dtype, device=local_logits.device
            )
            self._draft_greedy_gathered_ids_buf = torch.empty(
                (needed,), dtype=global_ids.dtype, device=local_logits.device
            )
            self._draft_greedy_gather_cap = needed

        if (
            self._draft_greedy_index_cap < num_tokens
            or self._draft_greedy_best_rank_buf is None
            or self._draft_greedy_rank_index_buf is None
            or self._draft_greedy_selected_ids_buf is None
            or self._draft_greedy_best_rank_buf.device != local_logits.device
            or self._draft_greedy_selected_ids_buf.device != local_logits.device
        ):
            self._draft_greedy_best_rank_buf = torch.empty(
                (num_tokens,), dtype=torch.int64, device=local_logits.device
            )
            self._draft_greedy_rank_index_buf = torch.empty(
                (1, num_tokens), dtype=torch.int64, device=local_logits.device
            )
            self._draft_greedy_selected_ids_buf = torch.empty(
                (1, num_tokens), dtype=torch.int64, device=local_logits.device
            )
            self._draft_greedy_index_cap = num_tokens

        gathered_max = self._draft_greedy_gathered_max_buf[:needed]
        gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]
        tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
        tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
        gathered_max = gathered_max.view(tp_size, num_tokens)
        gathered_ids = gathered_ids.view(tp_size, num_tokens)

        best_rank = self._draft_greedy_best_rank_buf[:num_tokens]
        torch.argmax(gathered_max, dim=0, out=best_rank)
        rank_index = self._draft_greedy_rank_index_buf[:, :num_tokens]
        rank_index[0].copy_(best_rank)
        selected_ids = self._draft_greedy_selected_ids_buf[:, :num_tokens]
        torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
        return selected_ids.view(-1).to(torch.long)

    def _sample_draft_tokens(
        self,
        *,
        draft_hidden: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        sampling_info,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bs, proposal_len = draft_hidden.shape[:2]
        draft_tokens = torch.empty(
            (bs, proposal_len), dtype=torch.long, device=draft_hidden.device
        )
        prev_tokens = first_prev_tokens.to(torch.long)
        lm_head = self.draft_model.lm_head
        # The draft block has already produced every hidden state.  The base
        # LM head is row-wise independent, so batch its seven projections and
        # leave only the Markov correction and argmax in the sequential loop.
        base_logits = self.draft_model.compute_logits(
            draft_hidden.reshape(bs * proposal_len, -1)
        ).view(bs, proposal_len, -1)
        is_all_greedy = sampling_info is None or sampling_info.is_all_greedy
        tp_size = int(get_tp_group().world_size)
        vocab_size = int(self.draft_model.config.vocab_size)
        draft_probs = [] if not is_all_greedy else None
        confidence_values = [] if self.ragged_planner.enabled else None
        greedy_mask = (
            None if sampling_info is None else (sampling_info.top_ks <= 1).view(-1)
        )
        for step in range(proposal_len):
            step_hidden = draft_hidden[:, step, :]
            step_logits = self.draft_model.apply_step_logits(
                base_logits[:, step, :],
                prev_tokens=prev_tokens,
                hidden_states=step_hidden,
            )
            if confidence_values is not None:
                confidence_values.append(
                    self.draft_model.predict_confidence_step(
                        step_hidden, prev_tokens=prev_tokens
                    )
                )
            if is_all_greedy:
                step_tokens = self._greedy_sample_from_vocab_parallel_logits(
                    local_logits=step_logits,
                    lm_head=lm_head,
                )
            else:
                if tp_size != 1:
                    step_logits = self.draft_model.gather_vocab_logits(step_logits)
                else:
                    # TP=1 local logits are target-vocabulary logits; discard any
                    # padding rows before constructing the proposal distribution q.
                    step_logits = step_logits[:, :vocab_size]
                probs = F.softmax(
                    step_logits.float()
                    / sampling_info.temperatures.to(torch.float32).clamp_min(1e-5),
                    dim=-1,
                )
                if sampling_info.need_top_k_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                if sampling_info.need_top_p_sampling:
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                assert greedy_mask is not None
                step_tokens = torch.where(
                    greedy_mask, torch.argmax(step_logits, dim=-1), sampled_tokens
                )
                draft_probs.append(probs)
            draft_tokens[:, step].copy_(step_tokens)
            prev_tokens = step_tokens
        return (
            draft_tokens,
            None if draft_probs is None else torch.stack(draft_probs, dim=1).contiguous(),
            (
                None
                if confidence_values is None
                else torch.stack(confidence_values, dim=1).contiguous()
            ),
        )

    def _prepare_for_speculative_decoding(
        self,
        batch: ScheduleBatch,
        draft_input: DSparkDraftInput,
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return
        if batch.has_grammar:
            raise RuntimeError(
                "Invariant broken: DSPARK batch has grammar constraints, but scheduler should have rejected this request."
            )
        if (
            batch.sampling_info is not None
            and not batch.sampling_info.is_all_greedy
            and not self._warned_non_greedy
            and self.tp_rank == 0
        ):
            logger.info(
                "DSPARK v1/no-overlap is using probability draft sampling and "
                "classic rejection verification."
            )
            self._warned_non_greedy = True

        bs = batch.batch_size()
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        self._ensure_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._verify_tokens_buf is not None
        assert self._verify_positions_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self.draft_model.mask_token_id))
        block_ids[:, 0].copy_(draft_input.bonus_tokens.to(torch.long))

        target_prefix_lens = batch.seq_lens
        draft_prefix_lens = draft_input.draft_seq_lens
        if draft_prefix_lens.dtype != torch.int32:
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        block_positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            target_prefix_lens.unsqueeze(1),
            self._pos_offsets[: self.block_size],
            out=block_positions_2d,
        )
        block_positions = block_positions_2d.reshape(-1)

        block_start = draft_prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, int(self.block_size), out=block_end)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = allocator.backup_state()
        try:
            if self.page_size == 1:
                block_cache_loc = allocator.alloc(bs * self.block_size)
            else:
                block_end_cpu = seq_lens_cpu + int(self.block_size)
                last_loc = get_last_loc(
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    block_start,
                )
                block_cache_loc = allocator.alloc_extend(
                    block_start,
                    seq_lens_cpu,
                    block_end,
                    block_end_cpu,
                    last_loc,
                    bs * self.block_size,
                )
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DSPARK draft OOM when allocating {bs * self.block_size} block tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=draft_prefix_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=int(draft_prefix_lens.sum().item()),
                seq_lens_cpu=seq_lens_cpu,
                positions=block_positions,
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                spec_algorithm=SpeculativeAlgorithm.DSPARK,
                spec_info=self._draft_block_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DSPARK draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        draft_tokens, draft_probs, confidence = self._sample_draft_tokens(
            draft_hidden=draft_hidden,
            first_prev_tokens=draft_input.bonus_tokens,
            sampling_info=batch.sampling_info,
        )

        verify_tokens = self._verify_tokens_buf[:bs]
        verify_tokens[:, 0].copy_(draft_input.bonus_tokens.to(torch.long))
        verify_tokens[:, 1:].copy_(draft_tokens)

        verify_positions_2d = self._verify_positions_buf[:bs]
        torch.add(
            target_prefix_lens.unsqueeze(1), self._pos_offsets, out=verify_positions_2d
        )

        verify_input = DSparkVerifyInput(
            draft_token=verify_tokens.reshape(-1),
            positions=verify_positions_2d.reshape(-1),
            draft_token_num=self.verify_token_num,
            draft_probs=draft_probs,
            confidence_threshold=self.confidence_threshold,
        )
        verify_input.confidence = confidence
        if self.ragged_planner.enabled:
            planned_lens = self.ragged_planner.plan(confidence, bs)
            if self.ragged_planner.mode == RaggedVerifyMode.COMPACT:
                verify_input.verify_lens = planned_lens
                verify_input.accept_lens = planned_lens
                verify_input.graph_num_tokens = self.ragged_planner.graph_num_tokens(
                    planned_lens, bs
                )
                verify_input.graph_slots = self.ragged_planner.graph_slots
            else:
                verify_input.accept_lens = planned_lens
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        verify_input.prepare_for_verify(
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return
        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=None,
            model=self.target_worker.model_runner.model,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "Invariant broken: DSPARK batch requested return_logprob, but scheduler should have rejected this request."
            )

        if isinstance(batch, ModelWorkerBatch):
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DSPARK requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DSpark layers-to-capture configured."
                )
            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DSPARK expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(
                model_worker_batch.extend_seq_lens
            )
            draft_input = DSparkDraftInput(
                bonus_tokens=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=_to_int32_device_tensor(
                    model_worker_batch.extend_prefix_lens
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_correct_drafts=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        draft_input = batch.spec_info
        if not isinstance(draft_input, DSparkDraftInput):
            raise RuntimeError(
                "DSPARK decode requires DSparkDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DSparkVerifyInput)
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        (
            new_bonus_tokens,
            commit_lens,
            next_target_hidden,
            num_correct_drafts_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        draft_input.bonus_tokens = new_bonus_tokens
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_correct_drafts = sum(num_correct_drafts_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DSPARK verify completed. num_correct_drafts_per_req=%s",
                num_correct_drafts_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_bonus_tokens,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
