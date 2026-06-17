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


def fused_qk_rmsnorm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _fused_qk_rmsnorm_mrope_custom(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
    )
    return q_out, k_out


def fused_qk_rmsnorm_mrope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _fused_qk_rmsnorm_mrope_cache_custom(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_cache,
        v_cache,
        indices,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
    )
    return q_out


def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    _norm_module().sgl_musa_store_cache(k, v, k_cache, v_cache, indices)


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


@register_custom_op(
    op_name="musa_csrc_fused_qk_rmsnorm_mrope",
    mutates_args=["q_out", "k_out"],
)
def _fused_qk_rmsnorm_mrope_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
) -> None:
    _norm_module().sgl_musa_fused_qk_rmsnorm_mrope(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
    )


@register_custom_op(
    op_name="musa_csrc_fused_qk_rmsnorm_mrope_cache",
    mutates_args=["q_out", "k_cache", "v_cache"],
)
def _fused_qk_rmsnorm_mrope_cache_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
) -> None:
    _norm_module().sgl_musa_fused_qk_rmsnorm_mrope_cache(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_cache,
        v_cache,
        indices,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
    )
