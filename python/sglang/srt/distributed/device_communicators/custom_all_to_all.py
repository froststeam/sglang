# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import logging
import math
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from sglang.srt.distributed.device_communicators.custom_all_gather import (
    _create_shared_buffer,
    _free_shared_buffer,
)

logger = logging.getLogger(__name__)

_is_musa = (
    hasattr(torch, "musa")
    and hasattr(torch.version, "musa")
    and torch.version.musa is not None
)


class MusaJitCustomAllToAll:
    """Single-node equal-split All-to-All specialized for Qwen-Image."""

    _SUPPORTED_WORLD_SIZES = (2, 4, 8)
    _SLOTS = 4
    _MAX_SLOT_BYTES = 64 * 1024 * 1024
    _QWEN_GLOBAL_HEADS = 24
    _QWEN_HEAD_DIM = 128

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
    ) -> None:
        self.disabled = True
        self.group = group
        self._opened_ipc_ptrs: list[int] = []
        self.meta_ptrs: list[int] = []
        self.output_storage: Optional[torch.Tensor] = None

        if not _is_musa:
            return
        if isinstance(device, int):
            device = torch.device(f"musa:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            return

        from sglang.srt.hardware_backend.musa.jit_kernel.csrc import alltoall

        self._jit_a2a = alltoall
        try:
            alltoall.ensure_compiled()
            self.meta_ptrs = _create_shared_buffer(alltoall.meta_size(), group=group)
            self.signal_ptrs_cpu = torch.tensor(
                self.meta_ptrs, dtype=torch.int64, device="cpu"
            )

            max_elements = self._MAX_SLOT_BYTES // 2
            self.output_storage = torch.empty(
                self._SLOTS * max_elements,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._max_elements = max_elements
            self.output_ptrs = self._share_output_storage()
            self.rank_data = torch.tensor(
                self.output_ptrs + [0] * (8 - self.world_size),
                dtype=torch.int64,
                device="cpu",
            )
            self._slot = 0
            self.disabled = False
        except Exception:
            self._release_resources()
            raise

    def _share_output_storage(self) -> list[int]:
        assert self.output_storage is not None
        lib = CudaRTLibrary()
        local_ptr = int(self.output_storage.data_ptr())
        handle = lib.cudaIpcGetMemHandle(ctypes.c_void_p(local_ptr))
        handles: list[Optional[cudaIpcMemHandle_t]] = [None] * self.world_size
        dist.all_gather_object(handles, handle, group=self.group)
        pointers: list[int] = []
        for rank, peer_handle in enumerate(handles):
            if rank == self.rank:
                pointers.append(local_ptr)
            else:
                assert peer_handle is not None
                pointer = lib.cudaIpcOpenMemHandle(peer_handle).value
                self._opened_ipc_ptrs.append(pointer)
                pointers.append(pointer)
        return pointers

    def should_custom_a2a(self, input_: torch.Tensor) -> bool:
        if self.disabled or input_.numel() == 0:
            return False
        if input_.device != self.device or input_.dtype != torch.bfloat16:
            return False
        if not input_.is_contiguous():
            return False
        input_nbytes = input_.numel() * input_.element_size()
        return (
            input_nbytes <= self._MAX_SLOT_BYTES
            and input_nbytes % (self.world_size * 16) == 0
            and int(input_.data_ptr()) % 16 == 0
        )

    def custom_all_to_all(self, input_: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.should_custom_a2a(input_):
            return None
        output, slot = self._next_output(input_.shape, input_.numel())
        self._jit_a2a.launch(
            self.rank_data,
            self.signal_ptrs_cpu,
            input_,
            output,
            self.meta_ptrs[self.rank],
            self._MAX_SLOT_BYTES,
            slot,
            self._SLOTS,
            self.rank,
            self.world_size,
        )
        return output

    def custom_ulysses(
        self, input_: torch.Tensor, head_dim: int, input_layout: bool
    ) -> Optional[torch.Tensor]:
        """Fuse Qwen-Image All-to-All with its surrounding layout copies."""
        if head_dim != 2 or input_.ndim != 4 or not self.should_custom_a2a(input_):
            return None
        batch, sequence, heads, dim = input_.shape
        if input_layout:
            if (
                batch != 1
                or heads != self._QWEN_GLOBAL_HEADS
                or dim != self._QWEN_HEAD_DIM
            ):
                return None
            local_sequence = sequence
            global_heads = heads
            output_shape = (
                batch,
                sequence * self.world_size,
                heads // self.world_size,
                dim,
            )
        else:
            if (
                batch != 1
                or heads * self.world_size != self._QWEN_GLOBAL_HEADS
                or dim != self._QWEN_HEAD_DIM
                or sequence % self.world_size != 0
            ):
                return None
            local_sequence = sequence // self.world_size
            global_heads = heads * self.world_size
            output_shape = (batch, local_sequence, global_heads, dim)

        output, slot = self._next_output(output_shape, input_.numel())
        self._jit_a2a.launch_ulysses(
            self.rank_data,
            self.signal_ptrs_cpu,
            input_,
            output,
            self.meta_ptrs[self.rank],
            self._MAX_SLOT_BYTES,
            slot,
            self._SLOTS,
            local_sequence,
            self.rank,
            self.world_size,
            input_layout,
        )
        return output

    def custom_qkv_ulysses(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        inputs = (query, key, value)
        if any(not self.should_custom_a2a(tensor) for tensor in inputs):
            return None
        if not (query.shape == key.shape == value.shape):
            return None
        batch, local_sequence, heads, dim = query.shape
        if batch != 1 or heads != self._QWEN_GLOBAL_HEADS or dim != self._QWEN_HEAD_DIM:
            return None

        output_shape = (
            batch,
            local_sequence * self.world_size,
            heads // self.world_size,
            dim,
        )
        outputs_and_slots = [
            self._next_output(output_shape, query.numel()) for _ in range(3)
        ]
        outputs = tuple(item[0] for item in outputs_and_slots)
        slots = tuple(item[1] for item in outputs_and_slots)
        self._jit_a2a.launch_qkv_ulysses(
            self.rank_data,
            self.signal_ptrs_cpu,
            query,
            key,
            value,
            *outputs,
            self.meta_ptrs[self.rank],
            self._MAX_SLOT_BYTES,
            *slots,
            self._SLOTS,
            local_sequence,
            self.rank,
            self.world_size,
        )
        return outputs

    def custom_ulysses_prefix_output(
        self, prefix: torch.Tensor, sharded: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.should_custom_a2a(prefix) or not self.should_custom_a2a(sharded):
            return None
        local_heads = self._QWEN_GLOBAL_HEADS // self.world_size
        if (
            prefix.ndim != 4
            or sharded.ndim != 4
            or prefix.shape[0] != 1
            or sharded.shape[0] != 1
            or prefix.shape[2:] != (local_heads, self._QWEN_HEAD_DIM)
            or sharded.shape[2:] != (local_heads, self._QWEN_HEAD_DIM)
            or sharded.shape[1] % self.world_size != 0
        ):
            return None
        prefix_sequence = prefix.shape[1]
        local_sequence = sharded.shape[1] // self.world_size
        output_shape = (
            1,
            prefix_sequence + local_sequence,
            self._QWEN_GLOBAL_HEADS,
            self._QWEN_HEAD_DIM,
        )
        output_numel = math.prod(output_shape)
        if output_numel * sharded.element_size() > self._MAX_SLOT_BYTES:
            return None
        output, slot = self._next_output(output_shape, output_numel)
        self._jit_a2a.launch_ulysses_prefix_output(
            self.rank_data,
            self.signal_ptrs_cpu,
            prefix,
            sharded,
            output,
            self.meta_ptrs[self.rank],
            self._MAX_SLOT_BYTES,
            slot,
            self._SLOTS,
            prefix_sequence,
            local_sequence,
            self.rank,
            self.world_size,
        )
        return output

    def _next_output(
        self, shape: torch.Size | tuple[int, ...], numel: int
    ) -> tuple[torch.Tensor, int]:
        if numel <= 0 or numel > self._max_elements:
            raise ValueError(
                f"Custom All-to-All output has {numel} elements; "
                f"expected 1..{self._max_elements}"
            )
        assert self.output_storage is not None
        slot = self._slot
        self._slot = (slot + 1) % self._SLOTS
        start = slot * self._max_elements
        output = self.output_storage.narrow(0, start, numel).view(shape)
        return output, slot

    def _release_resources(self) -> None:
        if (
            not self._opened_ipc_ptrs
            and not self.meta_ptrs
            and self.output_storage is None
        ):
            return
        lib = CudaRTLibrary()
        for pointer in self._opened_ipc_ptrs:
            lib.cudaIpcCloseMemHandle(ctypes.c_void_p(pointer))
        self._opened_ipc_ptrs.clear()
        if self.meta_ptrs:
            if dist.is_initialized():
                _free_shared_buffer(self.meta_ptrs, group=self.group)
            else:
                lib.cudaFree(ctypes.c_void_p(self.meta_ptrs[self.rank]))
            self.meta_ptrs = []
        self.output_storage = None

    def close(self) -> None:
        if not self.disabled:
            torch.musa.synchronize(self.device)
        self._release_resources()
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def dispatch_custom_alltoall():
    if _is_musa:
        return MusaJitCustomAllToAll
    return None
