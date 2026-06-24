from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit


@cache_once
def _memory_module():
    return load_musa_jit(
        "sglang_musa_memory_store_cache",
        ("memory/store_cache.mu",),
        extra_musa_cflags=(
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
        ),
    )


def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    _memory_module().sgl_musa_store_cache(k, v, k_cache, v_cache, indices)
