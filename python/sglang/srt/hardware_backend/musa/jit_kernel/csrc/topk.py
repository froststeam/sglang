from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _topk_module():
    return load_musa_jit(
        "sglang_musa_topk_gating",
        ("topk/topk_gating.mu",),
    )


def _topk_softmax_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    _topk_module().sgl_musa_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        float(moe_softcapping),
        bias_arg,
        bool(has_correction_bias),
    )


def _topk_sigmoid_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    _topk_module().sgl_musa_topk_sigmoid(
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        bias_arg,
        bool(has_correction_bias),
    )


@register_custom_op(
    op_name="musa_topk_softmax",
    mutates_args=["topk_weights", "topk_ids"],
)
def _topk_softmax_custom(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    _topk_softmax_impl(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
    )


@register_custom_op(
    op_name="musa_topk_sigmoid",
    mutates_args=["topk_weights", "topk_ids"],
)
def _topk_sigmoid_custom(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    _topk_sigmoid_impl(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """sgl_kernel-compatible top-k softmax entry point."""
    _topk_softmax_custom(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    _topk_sigmoid_custom(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias,
    )
