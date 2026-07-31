# SPDX-License-Identifier: Apache-2.0

import ctypes
import logging
import os
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

_is_musa = (
    hasattr(torch, "musa")
    and hasattr(torch.version, "musa")
    and torch.version.musa is not None
)

logger = logging.getLogger(__name__)


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
        self.disabled = False

    @contextmanager
    def capture(self):
        # The d3 path copies into pre-registered IPC buffers, so graph capture
        # needs no dynamic input registration.
        yield

    @staticmethod
    def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()

    def should_custom_rs(self, output: torch.Tensor, inp: torch.Tensor) -> bool:
        if self.disabled or output.numel() == 0 or inp.numel() == 0:
            return False
        if output.layout != torch.strided or inp.layout != torch.strided:
            return False
        if output.device != self.device or inp.device != self.device:
            return False
        if output.dtype != inp.dtype or output.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            return False
        if not output.is_contiguous() or not inp.is_contiguous():
            return False
        if output.ndim == 0 or inp.ndim != output.ndim:
            return False
        expected_shape = (output.shape[0] * self.world_size, *output.shape[1:])
        if tuple(inp.shape) != expected_shape:
            return False

        output_bytes = output.numel() * output.element_size()
        input_bytes = inp.numel() * inp.element_size()
        if output_bytes % 16 != 0 or input_bytes > self.max_size:
            return False
        if int(output.data_ptr()) % 16 != 0:
            return False
        if self._shares_storage(output, inp):
            shard_offset = self.rank * output_bytes
            if output.data_ptr() != inp.data_ptr() + shard_offset:
                return False
        return True

    def custom_reduce_scatter(
        self, output: torch.Tensor, inp: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.should_custom_rs(output, inp):
            return None
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
        return output

    def close(self) -> None:
        if not self.disabled and dist.is_initialized():
            _free_shared_buffer(self.buffer_ptrs, group=self.group)
            _free_shared_buffer(self.meta_ptrs, group=self.group)
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
