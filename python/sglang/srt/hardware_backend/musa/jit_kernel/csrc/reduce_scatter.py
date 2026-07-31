from __future__ import annotations

import functools
import os
from dataclasses import dataclass, replace

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit


@dataclass(frozen=True)
class CompileConfig:
    threads: int
    blocks: int
    max_blocks: int
    dynamic_blocks: int


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def _compile_config(world_size: int) -> CompileConfig:
    default_blocks = 80 if world_size in (4, 8) else 56
    blocks = _env_int(
        ("SGLANG_CUSTOM_RS_BLOCKS", "SGL_CUSTOM_RS_BLOCKS"), default_blocks
    )
    max_blocks = _env_int(
        ("SGLANG_CUSTOM_RS_MAX_BLOCKS", "SGL_CUSTOM_RS_MAX_BLOCKS"), max(120, blocks)
    )
    return CompileConfig(
        threads=_env_int(("SGLANG_CUSTOM_RS_THREADS", "SGL_CUSTOM_RS_THREADS"), 512),
        blocks=blocks,
        max_blocks=max_blocks,
        dynamic_blocks=_env_int(
            ("SGLANG_CUSTOM_RS_DYNAMIC_BLOCKS", "SGL_CUSTOM_RS_DYNAMIC_BLOCKS"), 1
        ),
    )


def _compile_name(config: CompileConfig) -> str:
    return (
        "sglang_musa_custom_reduce_scatter_v7"
        f"_t{config.threads}_b{config.blocks}"
        f"_mb{config.max_blocks}_db{config.dynamic_blocks}"
    )


def _musa_cflags(config: CompileConfig) -> tuple[str, ...]:
    return (
        "-Wno-error=address-of-temporary",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-D__MUSA_ARCH_LIST__=310",
        f"-DSGL_CUSTOM_RS_THREADS={config.threads}",
        f"-DSGL_CUSTOM_RS_BLOCKS={config.blocks}",
        f"-DSGL_CUSTOM_RS_MAX_BLOCKS={config.max_blocks}",
        f"-DSGL_CUSTOM_RS_DYNAMIC_BLOCKS={config.dynamic_blocks}",
        "-mllvm",
        "-mtgpu-opt-level=1",
        "-mllvm",
        "-mtgpu-load-store-opt=1",
        "-mllvm",
        "-mtgpu-fold-global-ldst=1",
    )


@functools.lru_cache(maxsize=8)
def _custom_rs_module(world_size: int):
    config = _compile_config(int(world_size))
    try:
        return load_musa_jit(
            _compile_name(config),
            ("distributed/custom_reduce_scatter.mu",),
            extra_musa_cflags=_musa_cflags(config),
        )
    except Exception:
        return load_musa_jit(
            _compile_name(replace(config, dynamic_blocks=0)),
            ("distributed/custom_reduce_scatter.mu",),
            extra_musa_cflags=_musa_cflags(replace(config, dynamic_blocks=0)),
        )


def ensure_compiled(world_size: int) -> None:
    _custom_rs_module(int(world_size))


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
    offset = align(offset) + flag_bytes * config.max_blocks
    return align(offset)


def launch_d3_func(world_size: int):
    return _custom_rs_module(
        int(world_size)
    ).sgl_musa_custom_rs_launch_unregistered_chunked
