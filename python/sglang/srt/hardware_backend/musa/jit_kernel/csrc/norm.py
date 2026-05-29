from __future__ import annotations

from pathlib import Path

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _norm_module():
    import tilelang

    tilelang_dir = Path(tilelang.__file__).resolve().parent
    return load_musa_jit(
        "sglang_musa_norm",
        ("norm/rmsnorm.mu",),
        extra_musa_cflags=(
            f"-I{(tilelang_dir / 'src').resolve()}",
            f"-I{(tilelang_dir / '3rdparty' / 'mutlass' / 'include').resolve()}",
            "-Wno-error=address-of-temporary",
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-D__MUSA_ARCH_LIST__=310",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
            "-mllvm",
            "-mtgpu-load-cluster-mutation=1",
            "-mllvm",
            "-mtgpu-store-cluster-mutation=1",
            "-mllvm",
            "-mtgpu-memory-sched-mutation=1",
            "-mllvm",
            "-mtgpu-alloc-shared-memory-from-zero=1",
        ),
    )


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    _ = enable_pdl
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm_custom(input, weight, out, float(eps), False)
    return out


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    _ = enable_pdl
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm_custom(input, weight, out, float(eps), True)
    return out


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    _ = enable_pdl
    _fused_add_rmsnorm_custom(input, residual, weight, float(eps), False)


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    _ = enable_pdl
    _fused_add_rmsnorm_custom(input, residual, weight, float(eps), True)


@register_custom_op(
    op_name="musa_csrc_rmsnorm",
    mutates_args=["out"],
)
def _rmsnorm_custom(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    _norm_module().sgl_musa_rmsnorm(input, weight, out, float(eps), bool(gemma))


@register_custom_op(
    op_name="musa_csrc_fused_add_rmsnorm",
    mutates_args=["input", "residual"],
)
def _fused_add_rmsnorm_custom(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    _norm_module().sgl_musa_fused_add_rmsnorm(
        input, residual, weight, float(eps), bool(gemma)
    )
