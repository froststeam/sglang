from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op


def _jit_csrc_dir() -> Path:
    return Path(__file__).resolve().parent


@cache_once
def _moe_gemv_bf16_module():
    return load_musa_jit(
        "sglang_musa_moe_gemv_bf16",
        ("gemm/moe_gemv_bf16.mu",),
        extra_musa_cflags=(
            f"-I{_jit_csrc_dir()}",
            f"-I{_jit_csrc_dir() / 'include'}",
            "-mtgpu",
            "-Od3",
            "-fmusa-flush-denormals-to-zero",
            "-fno-strict-aliasing",
            "-Wno-macro-redefined",
        ),
    )


@cache_once
def _moe_gemv_fp8_module():
    return load_musa_jit(
        "sglang_musa_moe_gemv_fp8",
        ("gemm/moe_gemv_fp8.mu",),
        extra_musa_cflags=(
            f"-I{_jit_csrc_dir()}",
            f"-I{_jit_csrc_dir() / 'include'}",
            "-mtgpu",
            "-Od3",
            "-fmusa-flush-denormals-to-zero",
            "-fno-strict-aliasing",
            "-Wno-macro-redefined",
        ),
    )


@cache_once
def _moe_gemv_int4_module():
    return load_musa_jit(
        "sglang_musa_moe_gemv_int4",
        ("gemm/moe_gemv_int4.mu",),
        extra_musa_cflags=(
            f"-I{_jit_csrc_dir()}",
            f"-I{_jit_csrc_dir() / 'include'}",
            "-mtgpu",
            "-Od3",
            "-fmusa-flush-denormals-to-zero",
            "-fno-strict-aliasing",
            "-Wno-macro-redefined",
        ),
    )


@cache_once
def _gemv_module():
    return load_musa_jit(
        "sglang_musa_gemv",
        ("gemm/gemv.mu",),
        extra_musa_cflags=(
            f"-I{_jit_csrc_dir()}",
            f"-I{_jit_csrc_dir() / 'include'}",
            "-mtgpu",
            "-Od3",
            "-fmusa-flush-denormals-to-zero",
            "-fno-strict-aliasing",
            "-Wno-macro-redefined",
        ),
    )


def _empty_scale_arg(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), device=x.device, dtype=torch.float32)


def _select_moe_gemv_module(B: torch.Tensor, use_int4_w4a16: bool):
    if use_int4_w4a16:
        return _moe_gemv_int4_module()
    if B.dtype == torch.float8_e4m3fn:
        return _moe_gemv_fp8_module()
    return _moe_gemv_bf16_module()


def _musa_fused_moe_gemv_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    topk: int,
    use_int4_w4a16: bool,
    use_swigelu: bool,
) -> None:
    has_a_scale = A_scale is not None
    has_b_scale = B_scale is not None
    _select_moe_gemv_module(B, use_int4_w4a16).sgl_musa_fused_moe_gemv(
        A,
        B,
        C,
        A_scale if has_a_scale else _empty_scale_arg(A),
        B_scale if has_b_scale else _empty_scale_arg(A),
        topk_weights,
        topk_ids,
        bool(has_a_scale),
        bool(has_b_scale),
        bool(mul_routed_weight),
        int(topk),
        bool(use_int4_w4a16),
        bool(use_swigelu),
    )


def _musa_fused_gemv_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    use_int4_w4a16: bool,
    use_swigelu: bool,
    use_silu: bool,
) -> None:
    has_b_scale = B_scale is not None
    _gemv_module().sgl_musa_fused_gemv(
        A,
        B,
        C,
        B_scale if has_b_scale else _empty_scale_arg(A),
        bool(has_b_scale),
        bool(use_int4_w4a16),
        bool(use_swigelu),
        bool(use_silu),
    )


@register_custom_op(
    op_name="musa_fused_moe_gemv",
    mutates_args=["C"],
)
def _musa_fused_moe_gemv_custom(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    topk: int,
    use_int4_w4a16: bool,
    use_swigelu: bool,
) -> None:
    _musa_fused_moe_gemv_impl(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        topk_ids,
        mul_routed_weight,
        topk,
        use_int4_w4a16,
        use_swigelu,
    )


@register_custom_op(
    op_name="musa_fused_gemv",
    mutates_args=["C"],
)
def _musa_fused_gemv_custom(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    use_int4_w4a16: bool,
    use_swigelu: bool,
    use_silu: bool,
) -> None:
    _musa_fused_gemv_impl(
        A,
        B,
        C,
        B_scale,
        use_int4_w4a16,
        use_swigelu,
        use_silu,
    )


def musa_fused_moe_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    topk: int,
    use_int4_w4a16: bool,
    use_swigelu: bool,
) -> None:
    _musa_fused_moe_gemv_custom(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        topk_ids,
        mul_routed_weight,
        topk,
        use_int4_w4a16,
        use_swigelu,
    )


def musa_fused_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    use_int4_w4a16: bool,
    use_swigelu: bool,
    use_silu: bool = False,
) -> None:
    _musa_fused_gemv_custom(
        A,
        B,
        C,
        B_scale,
        use_int4_w4a16,
        use_swigelu,
        use_silu,
    )


def musa_linear_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_silu: bool = False,
) -> torch.Tensor:
    input_shape = A.shape
    if A.dim() != 2:
        A = A.reshape(-1, input_shape[-1])
    C = torch.empty(
        (A.shape[0], B.shape[0]),
        device=A.device,
        dtype=A.dtype,
    )
    musa_fused_gemv(
        A,
        B,
        C,
        B_scale,
        False,
        False,
        use_silu,
    )
    output = C
    if bias is not None:
        output = output + bias
    if len(input_shape) != 2:
        output = output.view(*input_shape[:-1], B.shape[0])
    return output


__all__ = ["musa_fused_gemv", "musa_fused_moe_gemv", "musa_linear_gemv"]
