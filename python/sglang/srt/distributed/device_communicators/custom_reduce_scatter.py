# SPDX-License-Identifier: Apache-2.0

import ctypes
import logging
import os
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from torch.distributed import ProcessGroup

_is_musa = (
    hasattr(torch, "musa")
    and hasattr(torch.version, "musa")
    and torch.version.musa is not None
)

logger = logging.getLogger(__name__)


def _env_flag(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value.lower() in ("1", "true", "yes", "on")
    return default


def _use_graph_registered_input() -> bool:
    # Keep the original AR switches as compatibility aliases because the
    # registered RS path was originally hosted by the all-reduce communicator.
    # The RS-specific switches take precedence when both are present.
    return _env_flag(
        (
            "SGLANG_MUSA_CUSTOM_RS_GRAPH_REGISTERED_INPUT",
            "SGL_CUSTOM_RS_GRAPH_REGISTERED_INPUT",
            "SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT",
            "SGL_CUSTOM_AR_GRAPH_REGISTERED_INPUT",
        ),
        True,
    )


def _use_jit_reduce_scatter() -> bool:
    if not _is_musa:
        return False
    value = os.environ.get("SGLANG_MUSA_USE_JIT_REDUCE_SCATTER")
    return value is not None and value.lower() in ("1", "true", "yes", "y", "on")


def _create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup] = None
) -> List[int]:
    lib = CudaRTLibrary()
    pointer = lib.cudaMalloc(size_in_bytes)
    lib.cudaMemset(pointer, 0, size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=group)

    pointers: List[int] = []
    for index, peer_handle in enumerate(handles):
        if index == rank:
            pointers.append(pointer.value)
        else:
            pointers.append(lib.cudaIpcOpenMemHandle(peer_handle).value)
    return pointers


def _free_shared_buffer(
    pointers: List[int], group: Optional[ProcessGroup] = None
) -> None:
    rank = dist.get_rank(group=group)
    CudaRTLibrary().cudaFree(ctypes.c_void_p(pointers[rank]))


class MusaJitCustomReduceScatter:
    """Node-local MUSA reduce-scatter using the fixed d3 kernel."""

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CRS_SIZE = 512 * 1024 * 1024
    _RANK_DATA_WIDTH = 8
    requires_graph_capture_registration_recapture = False

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: Optional[int] = None,
    ) -> None:
        self.disabled = True
        self.group = group
        self.max_size = int(
            os.environ.get(
                "SGLANG_CUSTOM_RS_MAX_SIZE_BYTES",
                max_size if max_size is not None else self._MAX_CRS_SIZE,
            )
        )
        self._launcher = None
        self._registered_launcher = None
        self._IS_CAPTURING = False
        self._graph_registered_input_enabled = False
        self._graph_inputs: list[torch.Tensor] = []
        self._graph_registered_input_sequence: list[
            tuple[tuple[object, ...], torch.Tensor, torch.Tensor]
        ] = []
        self._graph_registered_sequence_signature: tuple[
            tuple[tuple[object, ...], tuple[int, ...]], ...
        ] = ()
        self._graph_registered_cursor = 0
        self._graph_registered_miss = False
        # Allocate only as many RankData slots as one graph actually uses.
        # Completed pools stay referenced for the lifetime of captured graphs.
        self._graph_rank_data_pools: list[torch.Tensor] = []
        self._graph_capture_rank_data_slots: Optional[torch.Tensor] = None
        self._rank_data_cache: dict[int, torch.Tensor] = {}
        self._last_input_ptr: Optional[int] = None
        self._last_rank_data: Optional[torch.Tensor] = None
        self._opened_ipc_ptrs: dict[bytes, int] = {}
        self._musa_lib = None
        self._mu_pointer_get_attribute = None

        if not _is_musa:
            return

        if isinstance(device, int):
            device = torch.device(f"musa:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            return

        from sglang.srt.hardware_backend.musa.jit_kernel.csrc import reduce_scatter

        self._jit_rs = reduce_scatter
        reduce_scatter.ensure_compiled(self.world_size)
        self.meta_ptrs = _create_shared_buffer(
            reduce_scatter.meta_size(self.world_size), group=group
        )
        self.buffer_ptrs = _create_shared_buffer(self.max_size, group=group)
        self.rank_data = torch.tensor(
            self.buffer_ptrs + [0] * (8 - self.world_size),
            dtype=torch.int64,
            device="cpu",
        )
        self.signal_ptrs_cpu = torch.tensor(
            self.meta_ptrs, dtype=torch.int64, device="cpu"
        )
        self._graph_registered_input_enabled = _use_graph_registered_input()
        self.requires_graph_capture_registration_recapture = (
            self._graph_registered_input_enabled
        )
        self.disabled = False

    @contextmanager
    def capture(self):
        try:
            self._graph_inputs.clear()
            # Graph runners that do not implement the registration/recapture
            # lifecycle must capture the safe staging path instead of reusing
            # registered pointers from an earlier graph.
            self._graph_registered_input_sequence.clear()
            self._graph_registered_sequence_signature = ()
            self._graph_registered_cursor = 0
            self._graph_registered_miss = False
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self._graph_registered_input_enabled:
                self._graph_inputs.clear()

    def _get_base_ptr_and_offset(self, inp: torch.Tensor) -> tuple[int, int]:
        ptr_value = int(inp.data_ptr())
        if self._mu_pointer_get_attribute is None:
            self._musa_lib = ctypes.CDLL("libmusa.so")
            self._mu_pointer_get_attribute = self._musa_lib.muPointerGetAttribute
            self._mu_pointer_get_attribute.restype = ctypes.c_int
            self._mu_pointer_get_attribute.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_ulonglong,
            ]
        base_ptr = ctypes.c_void_p()
        err = self._mu_pointer_get_attribute(
            ctypes.byref(base_ptr),
            11,  # MU_POINTER_ATTRIBUTE_RANGE_START_ADDR
            ctypes.c_ulonglong(ptr_value),
        )
        if err != 0:
            raise RuntimeError(f"muPointerGetAttribute failed: {err}")
        base_value = int(base_ptr.value)
        return base_value, ptr_value - base_value

    def _local_ipc_record_for_input(self, inp: torch.Tensor) -> tuple[int, bytes, int]:
        ptr_value = int(inp.data_ptr())
        base_value, offset = self._get_base_ptr_and_offset(inp)
        handle = CudaRTLibrary().cudaIpcGetMemHandle(ctypes.c_void_p(base_value))
        return ptr_value, bytes(handle), offset

    def _rank_data_from_ipc_records(
        self, ptr_value: int, records: list[tuple[int, bytes, int]]
    ) -> torch.Tensor:
        lib = CudaRTLibrary()
        ptrs: List[int] = []
        for index, (_, handle, offset) in enumerate(records):
            if index == self.rank:
                ptrs.append(ptr_value)
                continue
            opened_base = self._opened_ipc_ptrs.get(handle)
            if opened_base is None:
                ipc_handle = cudaIpcMemHandle_t.from_buffer_copy(handle)
                opened_base = lib.cudaIpcOpenMemHandle(ipc_handle).value
                self._opened_ipc_ptrs[handle] = opened_base
            ptrs.append(opened_base + int(offset))
        ptrs += [0] * (self._RANK_DATA_WIDTH - self.world_size)
        return torch.tensor(ptrs, dtype=torch.int64, device="cpu")

    @staticmethod
    def _rank_data_ptr_tuple(rank_data: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(value) for value in rank_data.tolist())

    @staticmethod
    def _graph_input_signature(inp: torch.Tensor) -> tuple[object, ...]:
        return (
            tuple(int(dim) for dim in inp.shape),
            str(inp.dtype),
            int(inp.numel()),
            int(inp.element_size()),
        )

    def _record_graph_input(self, inp: torch.Tensor) -> None:
        # Preserve call order, including duplicate pointers. The captured graph
        # consumes one stable RankData slot per collective invocation.
        self._graph_inputs.append(inp)

    def _rank_data_for_registered_input(
        self, inp: torch.Tensor
    ) -> Optional[torch.Tensor]:
        is_graph_launch = (
            self._IS_CAPTURING
            and torch.get_device_module().is_current_stream_capturing()
        )
        if not is_graph_launch:
            return None

        self._record_graph_input(inp)
        signature = self._graph_input_signature(inp)
        cursor = self._graph_registered_cursor
        self._graph_registered_cursor += 1
        if cursor < len(self._graph_registered_input_sequence):
            cached_signature, rank_data, rank_data_slot = (
                self._graph_registered_input_sequence[cursor]
            )
            if cached_signature == signature and self._rank_data_ptr_tuple(rank_data)[
                self.rank
            ] == int(inp.data_ptr()):
                return rank_data_slot
        if self._graph_registered_input_sequence:
            self._graph_registered_miss = True
        return None

    def _graph_object_broadcast_device(self) -> Union[str, torch.device]:
        try:
            backend = str(dist.get_backend(group=self.group)).lower()
        except Exception:
            backend = ""
        if "gloo" in backend:
            return "cpu"
        return self.device

    def register_graph_buffers(self) -> int:
        if not self._graph_registered_input_enabled:
            self._graph_inputs.clear()
            self._graph_registered_input_sequence.clear()
            self._graph_registered_sequence_signature = ()
            self._graph_registered_cursor = 0
            return 0

        entries = []
        try:
            for inp in tuple(self._graph_inputs):
                signature = self._graph_input_signature(inp)
                ptr_value, handle, offset = self._local_ipc_record_for_input(inp)
                entries.append((ptr_value, handle, offset, signature))

            all_entries = [None for _ in range(self.world_size)]
            all_entries[self.rank] = entries
            ranks = dist.get_process_group_ranks(group=self.group)
            broadcast_device = self._graph_object_broadcast_device()
            for index, rank in enumerate(ranks):
                holder = [all_entries[index]]
                dist.broadcast_object_list(
                    holder,
                    src=rank,
                    group=self.group,
                    device=broadcast_device,
                )
                all_entries[index] = holder[0]

            entry_counts = [len(rank_entries or ()) for rank_entries in all_entries]
            if len(set(entry_counts)) != 1:
                raise RuntimeError(
                    "MUSA JIT custom reduce-scatter graph input registration "
                    f"mismatch: rank entry counts are {entry_counts}."
                )

            if entries and (
                self._graph_capture_rank_data_slots is None
                or self._graph_capture_rank_data_slots.size(0) != len(entries)
            ):
                # A previous attempt with a different entry count is discarded
                # before recapture. Keep its slots alive until communicator
                # teardown because graph destruction may lag behind Python's
                # reference update and the allocator must not reuse the address.
                if self._graph_capture_rank_data_slots is not None:
                    self._graph_rank_data_pools.append(
                        self._graph_capture_rank_data_slots
                    )
                self._graph_capture_rank_data_slots = torch.empty(
                    (len(entries), self._RANK_DATA_WIDTH),
                    dtype=torch.int64,
                    device=self.device,
                )
            elif not entries:
                if self._graph_capture_rank_data_slots is not None:
                    self._graph_rank_data_pools.append(
                        self._graph_capture_rank_data_slots
                    )
                self._graph_capture_rank_data_slots = None

            new_sequence = []
            new_sequence_signature = []
            for index, (ptr_value, _, _, signature) in enumerate(entries):
                rank_records = []
                rank_signatures = []
                for rank_entries in all_entries:
                    peer_ptr, peer_handle, peer_offset, peer_signature = rank_entries[
                        index
                    ]
                    rank_records.append((peer_ptr, peer_handle, peer_offset))
                    rank_signatures.append(peer_signature)

                if any(
                    peer_signature != signature for peer_signature in rank_signatures
                ):
                    raise RuntimeError(
                        "MUSA JIT custom reduce-scatter graph input registration "
                        f"order mismatch at entry {index}: {rank_signatures}."
                    )

                rank_data = self._rank_data_from_ipc_records(ptr_value, rank_records)
                rank_data_slot = self._graph_capture_rank_data_slots[index]
                rank_data_slot.copy_(
                    rank_data.to(device=self.device, non_blocking=False)
                )
                new_sequence.append((signature, rank_data, rank_data_slot))
                new_sequence_signature.append(
                    (signature, self._rank_data_ptr_tuple(rank_data))
                )
                self._rank_data_cache[ptr_value] = rank_data
                self._last_input_ptr = ptr_value
                self._last_rank_data = rank_data

            new_sequence_signature_tuple = tuple(new_sequence_signature)
            registered = int(
                self._graph_registered_sequence_signature
                != new_sequence_signature_tuple
                or self._graph_registered_miss
            )
            self._graph_registered_sequence_signature = new_sequence_signature_tuple
            if entries:
                torch.get_device_module().synchronize()
            self._graph_registered_input_sequence = new_sequence
            self._graph_registered_cursor = 0
            self._graph_registered_miss = False
            return registered
        finally:
            self._graph_inputs.clear()

    def begin_graph_capture_registration(self) -> None:
        if not self._graph_registered_input_enabled:
            return
        self._graph_registered_input_sequence.clear()
        self._graph_registered_sequence_signature = ()
        self._graph_registered_cursor = 0
        self._graph_registered_miss = False
        self._graph_capture_rank_data_slots = None

    def end_graph_capture_registration(self) -> None:
        if not self._graph_registered_input_enabled:
            return
        if self._graph_capture_rank_data_slots is not None:
            self._graph_rank_data_pools.append(self._graph_capture_rank_data_slots)
        self._graph_capture_rank_data_slots = None
        self._graph_registered_input_sequence.clear()
        self._graph_registered_sequence_signature = ()
        self._graph_registered_cursor = 0
        self._graph_registered_miss = False
        self._graph_inputs.clear()

    def prepare_graph_capture(self) -> None:
        if self._graph_registered_input_enabled:
            self._graph_registered_cursor = 0
            self._graph_registered_miss = False

    def prepare_graph_replay(self) -> None:
        if self._graph_registered_input_enabled:
            self._graph_registered_cursor = 0

    @staticmethod
    def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()

    def custom_rs_reason(self, output: torch.Tensor, inp: torch.Tensor) -> str:
        if self.disabled:
            return "communicator_disabled"
        if output.numel() == 0 or inp.numel() == 0:
            return "empty_tensor"
        if output.layout != torch.strided or inp.layout != torch.strided:
            return "non_strided_layout"
        if output.device != self.device or inp.device != self.device:
            return "device_mismatch"
        if output.dtype != inp.dtype:
            return "dtype_mismatch"
        if output.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            return "unsupported_dtype"
        if not output.is_contiguous() or not inp.is_contiguous():
            return "noncontiguous_tensor"
        if output.ndim == 0 or inp.ndim != output.ndim:
            return "invalid_dimensions"
        expected_shape = (output.shape[0] * self.world_size, *output.shape[1:])
        if tuple(inp.shape) != expected_shape:
            return "shape_mismatch"

        output_bytes = output.numel() * output.element_size()
        input_bytes = inp.numel() * inp.element_size()
        if output_bytes % 16 != 0:
            return "unaligned_output_size"
        if input_bytes > self.max_size:
            return "size_exceeds_max"
        # The staging path can tolerate an unaligned input, but graph replay can
        # switch to the registered kernel, which reads peer inputs as 16-byte
        # packs. Keep the shared eligibility predicate safe for both paths.
        if int(output.data_ptr()) % 16 != 0 or int(inp.data_ptr()) % 16 != 0:
            return "unaligned_pointer"
        if self._shares_storage(output, inp):
            shard_offset = self.rank * output_bytes
            if output.data_ptr() != inp.data_ptr() + shard_offset:
                return "invalid_storage_alias"
        return "eligible"

    def should_custom_rs(self, output: torch.Tensor, inp: torch.Tensor) -> bool:
        return self.custom_rs_reason(output, inp) == "eligible"

    def _launch_registered(self, rank_data: torch.Tensor, output: torch.Tensor) -> None:
        if self._registered_launcher is None:
            self._registered_launcher = self._jit_rs.launch_registered_d3_func(
                self.world_size
            )
        self._registered_launcher(
            rank_data,
            self.signal_ptrs_cpu,
            output,
            self.meta_ptrs[self.rank],
            self.rank,
            self.world_size,
        )

    def _launch_unregistered(self, output: torch.Tensor, inp: torch.Tensor) -> None:
        if self._launcher is None:
            # d3 is the chunk-aware rank ordering from e53bcb9d.
            self._launcher = self._jit_rs.launch_d3_func(self.world_size)
        self._launcher(
            self.rank_data,
            self.signal_ptrs_cpu,
            inp,
            output,
            self.meta_ptrs[self.rank],
            self.buffer_ptrs[self.rank],
            self.max_size,
            self.rank,
            self.world_size,
        )

    def custom_reduce_scatter(
        self, output: torch.Tensor, inp: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.should_custom_rs(output, inp):
            return None

        is_graph_launch = (
            self._IS_CAPTURING
            and torch.get_device_module().is_current_stream_capturing()
        )
        if is_graph_launch and self._graph_registered_input_enabled:
            rank_data = self._rank_data_for_registered_input(inp)
            if rank_data is not None:
                self._launch_registered(rank_data, output)
            else:
                # The first capture and any sequence miss keep the original
                # staging path. The runner registers inputs and recaptures.
                self._launch_unregistered(output, inp)
        else:
            # Eager, warmup, disabled registration, and ordinary graph capture
            # retain the established staging-buffer implementation.
            self._launch_unregistered(output, inp)
        return output

    def close(self) -> None:
        if not self.disabled and dist.is_initialized():
            lib = CudaRTLibrary()
            for ptr in self._opened_ipc_ptrs.values():
                lib.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
            _free_shared_buffer(self.buffer_ptrs, group=self.group)
            _free_shared_buffer(self.meta_ptrs, group=self.group)
        self._opened_ipc_ptrs.clear()
        self._rank_data_cache.clear()
        self._last_input_ptr = None
        self._last_rank_data = None
        self._graph_inputs.clear()
        self._graph_registered_input_sequence.clear()
        self._graph_registered_sequence_signature = ()
        self._graph_rank_data_pools.clear()
        self._graph_capture_rank_data_slots = None
        self.requires_graph_capture_registration_recapture = False
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def dispatch_custom_reduce_scatter():
    if _use_jit_reduce_scatter():
        logger.debug("[RS] Using MusaJitCustomReduceScatter d3 (JIT-compiled)")
        return MusaJitCustomReduceScatter
    return None
