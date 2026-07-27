from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit


@dataclass(frozen=True)
class CompileConfig:
    threads: int
    blocks: int
    max_blocks: int
    atomic_barrier: int
    dynamic_blocks: int


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def _has_env(names: tuple[str, ...]) -> bool:
    return any(name in os.environ for name in names)


def _default_threads_blocks(world_size: int) -> tuple[int, int]:
    if world_size == 2:
        return 512, 48
    if world_size == 4:
        return 512, 80
    if world_size == 8:
        return 512, 64
    return 512, 48


def _compile_config(world_size: int) -> CompileConfig:
    default_threads, default_blocks = _default_threads_blocks(int(world_size))
    block_env_names = ("SGLANG_CUSTOM_AG_BLOCKS", "SGL_CUSTOM_AG_BLOCKS")
    threads = _env_int(
        ("SGLANG_CUSTOM_AG_THREADS", "SGL_CUSTOM_AG_THREADS"),
        default_threads,
    )
    blocks = _env_int(
        block_env_names,
        default_blocks,
    )
    max_blocks = _env_int(
        ("SGLANG_CUSTOM_AG_MAX_BLOCKS", "SGL_CUSTOM_AG_MAX_BLOCKS"),
        max(120, blocks),
    )
    atomic_barrier = _env_int(
        ("SGLANG_CUSTOM_AG_ATOMIC_BARRIER", "SGL_CUSTOM_AG_ATOMIC_BARRIER"),
        1,
    )
    dynamic_blocks = _env_int(
        ("SGLANG_CUSTOM_AG_DYNAMIC_BLOCKS", "SGL_CUSTOM_AG_DYNAMIC_BLOCKS"),
        0 if _has_env(block_env_names) else 1,
    )
    return CompileConfig(
        threads=threads,
        blocks=blocks,
        max_blocks=max_blocks,
        atomic_barrier=atomic_barrier,
        dynamic_blocks=dynamic_blocks,
    )


def _compile_name(config: CompileConfig) -> str:
    return (
        f"sglang_musa_custom_all_gather_t{config.threads}_b{config.blocks}"
        f"_mb{config.max_blocks}_ab{config.atomic_barrier}"
        f"_db{config.dynamic_blocks}"
    )


def _musa_cflags(config: CompileConfig) -> tuple[str, ...]:
    return (
        "-Wno-error=address-of-temporary",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-D__MUSA_ARCH_LIST__=310",
        f"-DSGL_CUSTOM_AG_THREADS={config.threads}",
        f"-DSGL_CUSTOM_AG_BLOCKS={config.blocks}",
        f"-DSGL_CUSTOM_AG_MAX_BLOCKS={config.max_blocks}",
        f"-DSGL_CUSTOM_AG_ATOMIC_BARRIER={config.atomic_barrier}",
        f"-DSGL_CUSTOM_AG_DYNAMIC_BLOCKS={config.dynamic_blocks}",
        "-mllvm",
        "-mtgpu-opt-level=1",
        "-mllvm",
        "-mtgpu-load-store-opt=1",
        "-mllvm",
        "-mtgpu-fold-global-ldst=1",
    )


@functools.lru_cache(maxsize=8)
def _custom_ag_module(world_size: int):
    config = _compile_config(int(world_size))
    return load_musa_jit(
        _compile_name(config),
        ("distributed/custom_all_gather.mu",),
        extra_musa_cflags=_musa_cflags(config),
    )


def ensure_compiled(world_size: int) -> None:
    _custom_ag_module(int(world_size))


def meta_size(world_size: int = 8) -> int:
    world_size = int(world_size)
    config = _compile_config(world_size)

    def align(value: int, alignment: int = 128) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    flag_bytes = 4
    max_ranks = 8
    offset = 0
    offset = align(offset) + flag_bytes * config.max_blocks * max_ranks
    offset = align(offset) + 2 * flag_bytes * config.max_blocks * max_ranks
    return align(offset)


def launch_registered_func(world_size: int):
    return _custom_ag_module(int(world_size)).sgl_musa_custom_ag_launch


def launch_unregistered_func(world_size: int):
    return _custom_ag_module(int(world_size)).sgl_musa_custom_ag_launch_unregistered


def launch_registered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    out: torch.Tensor,
    self_signal_ptr: int,
    input_nbytes: int,
    rank: int,
    world_size: int,
) -> None:
    _custom_ag_module(int(world_size)).sgl_musa_custom_ag_launch(
        rank_data,
        signal_ptrs_cpu,
        out,
        int(self_signal_ptr),
        int(input_nbytes),
        int(rank),
        int(world_size),
    )


def launch_unregistered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    out: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
) -> None:
    _custom_ag_module(int(world_size)).sgl_musa_custom_ag_launch_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        out,
        int(self_signal_ptr),
        int(self_buffer_ptr),
        int(max_size_bytes),
        int(rank),
        int(world_size),
    )
