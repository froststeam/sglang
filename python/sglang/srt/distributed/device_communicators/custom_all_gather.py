# SPDX-License-Identifier: Apache-2.0

import ctypes
import logging
import os
import weakref
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)

_is_musa = (
    hasattr(torch, "musa")
    and hasattr(torch.version, "musa")
    and torch.version.musa is not None
)

logger = logging.getLogger(__name__)


def _use_jit_all_gather() -> bool:
    return _is_musa and _env_bool(("SGLANG_MUSA_USE_JIT_ALL_GATHER",), False)


def _env_int(names: tuple[str, ...]) -> Optional[int]:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return None


def _env_bool(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value.lower() in ("1", "true", "yes", "y", "on")
    return default


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


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
    for i, h in enumerate(handles):
        if i == rank:
            pointers.append(pointer.value)
        else:
            pointers.append(lib.cudaIpcOpenMemHandle(h).value)
    return pointers


def _free_shared_buffer(
    pointers: List[int], group: Optional[ProcessGroup] = None
) -> None:
    rank = dist.get_rank(group=group)
    CudaRTLibrary().cudaFree(ctypes.c_void_p(pointers[rank]))


class MusaJitCustomAllGather:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CAG_SIZE = 512 * 1024 * 1024
    _MULTI_RANK_FAST_SIZE = 2 * 1024 * 1024

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: Optional[int] = None,
    ) -> None:
        self.disabled = True
        self.group = group
        requested_max_size = max_size
        self.max_size = self._MAX_CAG_SIZE
        self._IS_CAPTURING = False
        self._rank_data_cache: dict[int, torch.Tensor] = {}
        self._rank_data_refs: dict[int, weakref.ReferenceType] = {}
        self._opened_ipc_ptrs: dict[bytes, int] = {}
        self._last_input_ptr: Optional[int] = None
        self._last_rank_data: Optional[torch.Tensor] = None
        self._graph_inputs: dict[int, torch.Tensor] = {}
        self._registered_launcher = None
        self._unregistered_launcher = None

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
        self.max_size = self._resolve_max_size(requested_max_size)

        from sglang.srt.hardware_backend.musa.jit_kernel.csrc import allgather

        self._jit_ag = allgather
        allgather.ensure_compiled(self.world_size)
        self.meta_ptrs = _create_shared_buffer(
            allgather.meta_size(self.world_size), group=group
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

    def _resolve_max_size(self, requested_max_size: Optional[int]) -> int:
        env_max_size = _env_int(
            ("SGLANG_CUSTOM_AG_MAX_SIZE_BYTES", "SGL_CUSTOM_AG_MAX_SIZE_BYTES")
        )
        if env_max_size is not None:
            return env_max_size
        if requested_max_size is not None:
            return int(requested_max_size)
        if self.world_size in (2, 4, 8):
            return self._MAX_CAG_SIZE
        return self._MULTI_RANK_FAST_SIZE

    @contextmanager
    def capture(self):
        try:
            self._graph_inputs.clear()
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            try:
                if not self.disabled:
                    self.register_graph_buffers()
            finally:
                self._graph_inputs.clear()

    def should_custom_ag(self, output: torch.Tensor, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.numel() == 0:
            return False
        if inp.device.type != "musa" or output.device.type != "musa":
            return False
        if inp.device != output.device or inp.device != self.device:
            return False
        if inp.dtype != output.dtype:
            return False
        if output.numel() != inp.numel() * self.world_size:
            return False
        if not _is_weak_contiguous(inp) or not output.is_contiguous():
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0 or inp_size > self.max_size:
            return False
        if int(inp.data_ptr()) % 16 != 0 or int(output.data_ptr()) % 16 != 0:
            return False
        return True

    def _get_base_ptr_and_offset(self, inp: torch.Tensor) -> tuple[int, int]:
        ptr_value = int(inp.data_ptr())
        musa = ctypes.CDLL("libmusa.so")
        mu_pointer_get_attribute = musa.muPointerGetAttribute
        mu_pointer_get_attribute.restype = ctypes.c_int
        mu_pointer_get_attribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]
        base_ptr = ctypes.c_void_p()
        err = mu_pointer_get_attribute(
            ctypes.byref(base_ptr),
            11,  # MU_POINTER_ATTRIBUTE_RANGE_START_ADDR
            ctypes.c_ulonglong(ptr_value),
        )
        if err != 0:
            raise RuntimeError(f"muPointerGetAttribute failed: {err}")
        base_value = int(base_ptr.value)
        return base_value, ptr_value - base_value

    def _gather_ipc_meta(self, shard_data: tuple[bytes, int]):
        handle, offset = shard_data
        handle_tensor = torch.tensor(list(handle), dtype=torch.uint8, device="cpu")
        offset_tensor = torch.tensor([offset], dtype=torch.int64, device="cpu")
        handle_list = [
            torch.empty_like(handle_tensor) for _ in range(self.world_size)
        ]
        offset_list = [
            torch.empty_like(offset_tensor) for _ in range(self.world_size)
        ]
        dist.all_gather(handle_list, handle_tensor, group=self.group)
        dist.all_gather(offset_list, offset_tensor, group=self.group)
        handles = [bytes(t.tolist()) for t in handle_list]
        offsets = [int(t.item()) for t in offset_list]
        return handles, offsets

    def _cached_rank_data_for_input(self, inp: torch.Tensor) -> Optional[torch.Tensor]:
        ptr_value = int(inp.data_ptr())
        cached = self._rank_data_cache.get(ptr_value)
        if cached is None:
            return None
        ref = self._rank_data_refs.get(ptr_value)
        tensor = None if ref is None else ref()
        if tensor is None or int(tensor.data_ptr()) != ptr_value:
            self._rank_data_cache.pop(ptr_value, None)
            self._rank_data_refs.pop(ptr_value, None)
            if self._last_input_ptr == ptr_value:
                self._last_input_ptr = None
                self._last_rank_data = None
            return None
        return cached

    def _rank_data_for_input(self, inp: torch.Tensor) -> torch.Tensor:
        ptr_value = int(inp.data_ptr())
        cached = self._cached_rank_data_for_input(inp)
        if cached is not None:
            self._last_input_ptr = ptr_value
            self._last_rank_data = cached
            return cached

        base_value, offset = self._get_base_ptr_and_offset(inp)
        lib = CudaRTLibrary()
        handle = lib.cudaIpcGetMemHandle(ctypes.c_void_p(base_value))
        handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
        ptrs: List[int] = []
        for i, (h, off) in enumerate(zip(handles, offsets)):
            if i == self.rank:
                ptrs.append(ptr_value)
            else:
                opened_base = self._opened_ipc_ptrs.get(h)
                if opened_base is None:
                    ipc_handle = cudaIpcMemHandle_t.from_buffer_copy(h)
                    opened_base = lib.cudaIpcOpenMemHandle(ipc_handle).value
                    self._opened_ipc_ptrs[h] = opened_base
                ptrs.append(opened_base + int(off))
        ptrs += [0] * (8 - self.world_size)
        rank_data = torch.tensor(ptrs, dtype=torch.int64, device="cpu")
        self._rank_data_cache[ptr_value] = rank_data
        self._rank_data_refs[ptr_value] = weakref.ref(inp)
        self._last_input_ptr = ptr_value
        self._last_rank_data = rank_data
        return rank_data

    def _record_graph_input(self, inp: torch.Tensor) -> None:
        self._graph_inputs.setdefault(int(inp.data_ptr()), inp)

    def register_graph_buffers(self) -> None:
        # Do not call should_custom_ag here because output is not available after
        # capture exits. The runtime path validates before recording inputs.
        for inp in tuple(self._graph_inputs.values()):
            self._rank_data_for_input(inp)

    def _launch_registered(
        self, rank_data: torch.Tensor, output: torch.Tensor, inp: torch.Tensor
    ):
        if self._registered_launcher is None:
            self._registered_launcher = self._jit_ag.launch_registered_func(
                self.world_size
            )
        self._registered_launcher(
            rank_data,
            self.signal_ptrs_cpu,
            output,
            self.meta_ptrs[self.rank],
            inp.numel() * inp.element_size(),
            self.rank,
            self.world_size,
        )

    def _launch_unregistered(self, output: torch.Tensor, inp: torch.Tensor):
        if self._unregistered_launcher is None:
            self._unregistered_launcher = self._jit_ag.launch_unregistered_func(
                self.world_size
            )
        self._unregistered_launcher(
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

    def prepare_registered_input(self, inp: torch.Tensor) -> bool:
        # Registered input is kept as an explicit experimental path for
        # benchmark/test and graph-captured stable buffers. It is not enabled
        # for the normal eager path because dynamic activations do not
        # guarantee stable data_ptr cache hits.
        if self.disabled:
            return False
        if inp.numel() == 0 or inp.device.type != "musa" or inp.device != self.device:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0 or inp_size > self.max_size:
            return False
        if int(inp.data_ptr()) % 16 != 0:
            return False
        self._rank_data_for_input(inp)
        return True

    def custom_all_gather_registered(
        self, output: torch.Tensor, inp: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.should_custom_ag(output, inp):
            return None
        rank_data = self._cached_rank_data_for_input(inp)
        if rank_data is None:
            return None
        self._launch_registered(rank_data, output, inp)
        return output

    def custom_all_gather(
        self, output: torch.Tensor, inp: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.should_custom_ag(output, inp):
            return None

        is_graph_launch = (
            self._IS_CAPTURING
            and torch.get_device_module().is_current_stream_capturing()
        )
        if self._IS_CAPTURING:
            self._record_graph_input(inp)
            if is_graph_launch:
                rank_data = self._cached_rank_data_for_input(inp)
                if rank_data is not None:
                    self._launch_registered(rank_data, output, inp)
                else:
                    self._launch_unregistered(output, inp)
            else:
                self._rank_data_for_input(inp)
                self._launch_unregistered(output, inp)
        else:
            # Normal eager currently uses the unregistered path. Registered
            # eager remains disabled until end-to-end stable-buffer hit rates
            # prove the extra registration path is worth enabling.
            self._launch_unregistered(output, inp)
        return output

    def close(self):
        if not self.disabled and dist.is_initialized():
            lib = CudaRTLibrary()
            for ptr in self._opened_ipc_ptrs.values():
                lib.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
            self._opened_ipc_ptrs.clear()
            self._rank_data_cache.clear()
            self._rank_data_refs.clear()
            self._graph_inputs.clear()
            _free_shared_buffer(self.buffer_ptrs, group=self.group)
            _free_shared_buffer(self.meta_ptrs, group=self.group)
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def dispatch_custom_allgather():
    if _use_jit_all_gather():
        logger.debug("[AG] Using MusaJitCustomAllGather (JIT-compiled)")
        return MusaJitCustomAllGather
    return None
