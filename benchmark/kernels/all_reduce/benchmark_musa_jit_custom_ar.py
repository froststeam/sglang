#!/usr/bin/env python3
"""Benchmark MUSA JIT custom all-reduce against torch and SGL custom AR.

Example:
    PYTHONPATH=/path/to/sglang/python:/path/to/sglang/sgl-kernel/python \
    SGLANG_MUSA_JIT_CACHE_DIR=/path/to/.cache/sglang_musa_jit \
    MUSA_VISIBLE_DEVICES=0,1,2,3 \
    python benchmark_musa_jit_custom_ar.py \
      --world-size 4 \
      --bytes 536870912 \
      --dtype bf16 \
      --include-aot \
      --include-dispatch
"""

from __future__ import annotations

import argparse
import ctypes
import socket
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_musa  # noqa: F401

from sglang.srt.distributed import (
    destroy_distributed_environment,
    init_distributed_environment,
)


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sync_device() -> None:
    torch.get_device_module().synchronize()


def _event_ms(fn, iters: int, warmup: int, group) -> float:
    dist.barrier(group=group)
    for _ in range(warmup):
        fn()
    _sync_device()
    dist.barrier(group=group)

    dev = torch.get_device_module()
    start = dev.Event(enable_timing=True)
    end = dev.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync_device()
    dist.barrier(group=group)
    return start.elapsed_time(end) / iters


def _busbw_gbs(bytes_per_rank: int, world_size: int, ms: float) -> float:
    # NCCL-style allreduce bus bandwidth factor.
    factor = 2 * (world_size - 1) / world_size
    return bytes_per_rank * factor / (ms / 1e3) / 1e9


def _algbw_gbs(bytes_per_rank: int, ms: float) -> float:
    return bytes_per_rank / (ms / 1e3) / 1e9


def _all_ranks_ok(ok: bool, group) -> bool:
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _is_close(out: torch.Tensor | None, ref: torch.Tensor) -> bool:
    if out is None:
        return False
    return bool(torch.allclose(out, ref, rtol=1e-2, atol=1e-2))


def _make_ipc_ptrs(size_bytes: int, group) -> List[int]:
    from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary()
    ptr = lib.cudaMalloc(size_bytes)
    lib.cudaMemset(ptr, 0, size_bytes)
    handle = lib.cudaIpcGetMemHandle(ptr)
    handles = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(handles, handle, group=group)

    rank = dist.get_rank(group=group)
    ptrs: List[int] = []
    for i, handle_i in enumerate(handles):
        if i == rank:
            ptrs.append(ptr.value)
        else:
            ptrs.append(lib.cudaIpcOpenMemHandle(handle_i).value)
    return ptrs


def _make_ipc_ptrs_from_tensor(tensor: torch.Tensor, group) -> List[int]:
    from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary()
    ptr_value = int(tensor.data_ptr())
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
    offset = ptr_value - base_value
    handle = lib.cudaIpcGetMemHandle(ctypes.c_void_p(base_value))
    handles = [None] * dist.get_world_size(group=group)
    offsets = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(handles, handle, group=group)
    dist.all_gather_object(offsets, offset, group=group)

    rank = dist.get_rank(group=group)
    ptrs: List[int] = []
    for i, handle_i in enumerate(handles):
        if i == rank:
            ptrs.append(ptr_value)
        else:
            ptrs.append(lib.cudaIpcOpenMemHandle(handle_i).value + int(offsets[i]))
    return ptrs


def _free_own_ipc_ptr(ptrs: List[int], group) -> None:
    from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    rank = dist.get_rank(group=group)
    CudaRTLibrary().cudaFree(ctypes.c_void_p(ptrs[rank]))


@dataclass
class JitCAR:
    rank: int
    world_size: int
    meta_ptrs: List[int]
    buffer_ptrs: List[int]
    max_size: int
    rank_data: torch.Tensor
    signal_ptrs_cpu: torch.Tensor

    @classmethod
    def create(
        cls,
        inp: torch.Tensor,
        max_size: int,
        group,
        registered: bool,
    ) -> "JitCAR":
        from sglang.srt.hardware_backend.musa.jit_kernel.csrc import allreduce as jit_ar

        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        meta_ptrs = _make_ipc_ptrs(jit_ar.meta_size(world_size) + max_size, group)
        buffer_ptrs = _make_ipc_ptrs(max_size, group)
        data_ptrs = (
            _make_ipc_ptrs_from_tensor(inp, group) if registered else buffer_ptrs
        )
        rank_data = torch.tensor(data_ptrs + [0] * (8 - world_size), dtype=torch.int64)
        signal_ptrs_cpu = torch.tensor(meta_ptrs, dtype=torch.int64)
        return cls(
            rank,
            world_size,
            meta_ptrs,
            buffer_ptrs,
            max_size,
            rank_data,
            signal_ptrs_cpu,
        )

    def all_reduce_registered(self, inp: torch.Tensor, shot: int) -> torch.Tensor:
        from sglang.srt.hardware_backend.musa.jit_kernel.csrc import allreduce as jit_ar

        out = torch.empty_like(inp)
        jit_ar.launch_registered(
            self.rank_data,
            self.signal_ptrs_cpu,
            out,
            self.meta_ptrs[self.rank],
            self.rank,
            self.world_size,
            shot,
        )
        return out

    def all_reduce_eager(self, inp: torch.Tensor, shot: int) -> torch.Tensor:
        from sglang.srt.hardware_backend.musa.jit_kernel.csrc import allreduce as jit_ar

        out = torch.empty_like(inp)
        jit_ar.launch_unregistered(
            self.rank_data,
            self.signal_ptrs_cpu,
            inp,
            out,
            self.meta_ptrs[self.rank],
            self.buffer_ptrs[self.rank],
            self.max_size,
            self.rank,
            self.world_size,
            shot,
        )
        return out

    def close(self, group) -> None:
        dist.barrier(group=group)
        _free_own_ipc_ptr(self.buffer_ptrs, group)
        _free_own_ipc_ptr(self.meta_ptrs, group)


def _worker(rank: int, args, port: int) -> None:
    torch.musa.set_device(rank)
    init_distributed_environment(
        backend=args.backend,
        world_size=args.world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    group = dist.group.WORLD
    cpu_group = dist.new_group(list(range(args.world_size)), backend="gloo")
    device = torch.device(f"musa:{rank}")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    numel = args.bytes // torch.empty((), dtype=dtype).element_size()
    inp = torch.full((numel,), rank + 1, dtype=dtype, device=device)
    ref = inp.clone()
    dist.all_reduce(ref, group=group)

    results = []
    max_size = max(args.bytes, 16 * 1024 * 1024)

    def add_result(name: str, ms: float | None) -> None:
        if ms is None:
            results.append((name, None, None, None))
        else:
            results.append(
                (
                    name,
                    ms,
                    _algbw_gbs(args.bytes, ms),
                    _busbw_gbs(args.bytes, args.world_size, ms),
                )
            )

    torch_inp = inp.clone()
    add_result(
        "torch_dist",
        _event_ms(
            lambda: dist.all_reduce(torch_inp, group=group),
            args.iters,
            args.warmup,
            group,
        ),
    )

    if args.include_aot:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        aot = CustomAllreduce(cpu_group, device, max_size=max_size)
        if not aot.disabled and aot.should_custom_ar(inp):
            out = aot.custom_all_reduce(inp)
            if _all_ranks_ok(_is_close(out, ref), cpu_group):
                add_result(
                    "aot_custom_ar",
                    _event_ms(
                        lambda: aot.custom_all_reduce(inp),
                        args.iters,
                        args.warmup,
                        group,
                    ),
                )
            else:
                add_result("aot_custom_ar_FAIL", None)
        else:
            _all_ranks_ok(False, cpu_group)
            add_result("aot_custom_ar_SKIP", None)

    if args.include_dispatch:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            dispatch_custom_allreduce,
        )

        DispatchCustomAllreduce = dispatch_custom_allreduce()
        dispatch_ar = DispatchCustomAllreduce(cpu_group, device, max_size=max_size)
        if not dispatch_ar.disabled and dispatch_ar.should_custom_ar(inp):
            out = dispatch_ar.custom_all_reduce(inp)
            if _all_ranks_ok(_is_close(out, ref), cpu_group):
                add_result(
                    "dispatch_custom_ar",
                    _event_ms(
                        lambda: dispatch_ar.custom_all_reduce(inp),
                        args.iters,
                        args.warmup,
                        group,
                    ),
                )
            else:
                add_result("dispatch_custom_ar_FAIL", None)
        else:
            _all_ranks_ok(False, cpu_group)
            add_result("dispatch_custom_ar_SKIP", None)
        if hasattr(dispatch_ar, "close"):
            dispatch_ar.close()

    # Eager mode uses pre-registered staging buffers, matching production custom
    # AR. Registered mode microbenches graph-style direct input pointers.
    jit = JitCAR.create(
        inp=inp,
        max_size=max_size,
        group=cpu_group,
        registered=args.jit_registered,
    )
    jit_fn = jit.all_reduce_registered if args.jit_registered else jit.all_reduce_eager
    for shot in args.shots:
        out = jit_fn(inp, shot)
        if _all_ranks_ok(_is_close(out, ref), cpu_group):
            name = (
                f"jit_reg_shot{shot}"
                if args.jit_registered
                else f"jit_eager_shot{shot}"
            )
            add_result(
                name,
                _event_ms(
                    lambda shot=shot: jit_fn(inp, shot),
                    args.iters,
                    args.warmup,
                    group,
                ),
            )
        else:
            name = (
                f"jit_reg_shot{shot}_FAIL"
                if args.jit_registered
                else f"jit_eager_shot{shot}_FAIL"
            )
            add_result(name, None)
    jit.close(cpu_group)

    if rank == 0:
        print(
            f"bytes_per_rank={args.bytes} dtype={args.dtype} "
            f"world_size={args.world_size}"
        )
        print(f"{'name':<18} {'lat_ms':>10} {'algBW_GB/s':>12} {'busBW_GB/s':>12}")
        for name, ms, algbw, busbw in results:
            if ms is None:
                print(f"{name:<18} {'NA':>10} {'NA':>12} {'NA':>12}")
            else:
                print(f"{name:<18} {ms:10.4f} {algbw:12.2f} {busbw:12.2f}")

    dist.destroy_process_group(cpu_group)
    destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--bytes", type=int, default=163_840_000)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--backend", default="mccl")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--shots", type=int, nargs="+", default=[2])
    parser.add_argument("--include-aot", action="store_true")
    parser.add_argument("--include-dispatch", action="store_true")
    parser.add_argument("--jit-registered", action="store_true")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    mp.spawn(_worker, args=(args, _open_port()), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
