from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
import triton

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.activation import (  # noqa: F401
    act_and_mul,
    act_and_mul_masked,
    act_and_mul_masked_post_quant_fwd,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _topk_module():
    return load_musa_jit(
        "sglang_musa_topk_gating",
        ("moe/topk_gating.mu",),
    )


def _topk_softmax_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    has_shared_experts = (
        shared_expert_gate_output is not None and num_fused_shared_experts > 0
    )
    shared_arg = (
        shared_expert_gate_output if has_shared_experts else topk_weights.reshape(-1)
    )
    _topk_module().sgl_musa_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        float(moe_softcapping),
        bias_arg,
        bool(has_correction_bias),
        shared_arg,
        int(num_fused_shared_experts),
        bool(has_shared_experts),
    )


def _topk_sigmoid_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    has_shared_experts = (
        shared_expert_gate_output is not None and num_fused_shared_experts > 0
    )
    shared_arg = (
        shared_expert_gate_output if has_shared_experts else topk_weights.reshape(-1)
    )
    _topk_module().sgl_musa_topk_sigmoid(
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        bias_arg,
        bool(has_correction_bias),
        shared_arg,
        int(num_fused_shared_experts),
        bool(has_shared_experts),
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
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    _topk_softmax_impl(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        shared_expert_gate_output=shared_expert_gate_output,
        num_fused_shared_experts=num_fused_shared_experts,
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
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    _topk_sigmoid_impl(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias=correction_bias,
        shared_expert_gate_output=shared_expert_gate_output,
        num_fused_shared_experts=num_fused_shared_experts,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    """sgl_kernel-compatible top-k softmax entry point."""
    _topk_softmax_custom(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        shared_expert_gate_output=shared_expert_gate_output,
        num_fused_shared_experts=num_fused_shared_experts,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    _topk_sigmoid_custom(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias=correction_bias,
        shared_expert_gate_output=shared_expert_gate_output,
        num_fused_shared_experts=num_fused_shared_experts,
    )


@cache_once
def _moe_sum_reduce_module():
    return load_musa_jit(
        "sglang_musa_moe_sum_reduce",
        ("moe/moe_sum_reduce.mu",),
    )


def _moe_sum_reduce_impl(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float = 0.0,
) -> None:
    _moe_sum_reduce_module().sgl_musa_moe_sum_reduce(
        input_tensor,
        output_tensor,
        float(routed_scaling_factor),
    )


@register_custom_op(
    op_name="musa_moe_sum_reduce",
    mutates_args=["output_tensor"],
)
def _moe_sum_reduce_custom(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float = 0.0,
) -> None:
    _moe_sum_reduce_impl(input_tensor, output_tensor, routed_scaling_factor)


def moe_sum_reduce(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float = 0.0,
) -> None:
    _moe_sum_reduce_custom(input_tensor, output_tensor, routed_scaling_factor)


@cache_once
def _moe_align_block_size_module():
    return load_musa_jit(
        "sglang_musa_moe_align_block_size",
        ("moe/moe_align_block_size.mu",),
        extra_musa_cflags=(
            "-mtgpu",
            "-Od3",
            "-fmusa-flush-denormals-to-zero",
            "-fno-strict-aliasing",
        ),
    )


def _moe_align_block_size_impl(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    large_threshold = int(os.getenv("SGLANG_MUSA_MOE_ALIGN_LARGE_THRESHOLD", "16384"))
    chunked_direct_threshold = int(
        os.getenv("SGLANG_MUSA_MOE_ALIGN_CHUNKED_DIRECT_THRESHOLD", "1")
    )
    chunked_direct_limit = int(
        os.getenv("SGLANG_MUSA_MOE_ALIGN_CHUNKED_DIRECT_LIMIT", "8193")
    )
    use_chunked_direct = (
        chunked_direct_threshold <= topk_ids.size(0) < chunked_direct_limit
        and int(num_experts) <= 257
    )
    if use_chunked_direct:
        chunk_size = int(os.getenv("SGLANG_MUSA_MOE_ALIGN_DIRECT_CHUNK", "4096"))
        num_chunks = triton.cdiv(topk_ids.numel(), chunk_size)
        workspace = torch.empty(
            (num_chunks * int(num_experts),),
            device=topk_ids.device,
            dtype=torch.int32,
        )
        _moe_align_block_size_module().sgl_musa_moe_align_block_size_chunked_direct(
            topk_ids,
            int(num_experts),
            int(block_size),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            workspace,
            chunk_size,
            bool(pad_sorted_token_ids),
        )
        return

    if topk_ids.size(0) >= large_threshold:
        chunk_size = int(os.getenv("SGLANG_MUSA_MOE_ALIGN_CHUNK", "4096"))
        num_chunks = triton.cdiv(topk_ids.numel(), chunk_size)
        workspace = torch.empty(
            (2 * (num_chunks + 1) * int(num_experts),),
            device=topk_ids.device,
            dtype=torch.int32,
        )
        _moe_align_block_size_module().sgl_musa_moe_align_block_size_large_sort(
            topk_ids,
            int(num_experts),
            int(block_size),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            workspace,
            chunk_size,
            bool(pad_sorted_token_ids),
        )
        return

    _moe_align_block_size_module().sgl_musa_moe_align_block_size(
        topk_ids,
        int(num_experts),
        int(block_size),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        bool(pad_sorted_token_ids),
    )


@register_custom_op(
    op_name="musa_moe_align_block_size",
    mutates_args=[
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_pad",
        "cumsum_buffer",
    ],
)
def _moe_align_block_size_custom(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    _moe_align_block_size_impl(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    _moe_align_block_size_custom(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


def deep_gemm_contig_preprocess(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    m_indices: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids_for_combine: torch.Tensor,
    num_local_experts: int,
    use_fp8_quant: bool,
    block_m: int,
) -> None:
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")

    # XXX (MUSA): Workspace is private to the contiguous DeepGEMM preprocess op.
    counts = torch.empty(
        (num_local_experts,), device=hidden_states.device, dtype=torch.int32
    )
    cursor = torch.empty_like(counts)

    from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.deep_gemm_contig_preprocess import (
        can_use_bf16_tilelang,
        can_use_fp8_tilelang,
        deep_gemm_contig_preprocess_bf16_tilelang,
        deep_gemm_contig_preprocess_fp8_tilelang,
    )

    if use_fp8_quant:
        if not can_use_fp8_tilelang(
            hidden_states, topk_ids, num_local_experts, use_fp8_quant
        ):
            raise RuntimeError(
                "MUSA DeepGEMM fp8 contiguous preprocess now requires the "
                "TileLang implementation: expected hidden_size to be a "
                "supported multiple of 128, topk=8, and num_local_experts>0."
            )
        deep_gemm_contig_preprocess_fp8_tilelang(
            hidden_states,
            topk_ids,
            output,
            output_scale,
            m_indices,
            src2dst,
            topk_ids_for_combine,
            counts,
            cursor,
            num_local_experts,
            block_m,
        )
        return

    if not can_use_bf16_tilelang(
        hidden_states, topk_ids, num_local_experts, use_fp8_quant
    ):
        raise RuntimeError(
            "MUSA DeepGEMM bf16 contiguous preprocess now requires the "
            "TileLang implementation: expected hidden_size to be a supported "
            "multiple of 128, topk<=16, and num_local_experts>0."
        )
    deep_gemm_contig_preprocess_bf16_tilelang(
        hidden_states,
        topk_ids,
        output,
        m_indices,
        src2dst,
        topk_ids_for_combine,
        counts,
        cursor,
        num_local_experts,
        block_m,
    )


def deep_gemm_ep_preprocess(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    src2dst: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor | None,
    num_local_experts: int,
    use_fp8_quant: bool,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.deep_gemm_ep_preprocess import (
        can_use_ep_bf16_tilelang,
        can_use_ep_fp8_tilelang,
        deep_gemm_ep_preprocess_bf16_tilelang,
        deep_gemm_ep_preprocess_fp8_tilelang,
    )

    if use_fp8_quant:
        if output_scale is None:
            raise ValueError("output_scale must be provided for fp8 EP preprocess")
        if not can_use_ep_fp8_tilelang(hidden_states, topk_ids, num_local_experts):
            raise RuntimeError(
                "MUSA DeepGEMM fp8 EP preprocess requires hidden_size to be a "
                "supported multiple of 128, topk<=16, int32 topk_ids, and "
                "num_local_experts>0."
            )
        deep_gemm_ep_preprocess_fp8_tilelang(
            topk_ids,
            hidden_states,
            masked_m,
            src2dst,
            output,
            output_scale,
            num_local_experts,
        )
        return

    if not can_use_ep_bf16_tilelang(hidden_states, topk_ids, num_local_experts):
        raise RuntimeError(
            "MUSA DeepGEMM bf16 EP preprocess requires hidden_size to be a "
            "supported multiple of 128, topk<=16, int32 topk_ids, and "
            "num_local_experts>0."
        )
    deep_gemm_ep_preprocess_bf16_tilelang(
        topk_ids,
        hidden_states,
        masked_m,
        src2dst,
        output,
        num_local_experts,
    )


@cache_once
def _fused_share_gate_sigmoid_mul_module():
    return load_musa_jit(
        "sglang_musa_fused_share_gate_sigmoid_mul",
        ("moe/fused_share_gate_sigmoid_mul.mu",),
    )


def _can_use_fused_share_gate_sigmoid_mul(
    hidden_state: torch.Tensor,
    share_gate_weight: torch.Tensor,
    share_expert_output: torch.Tensor,
) -> bool:
    return (
        hidden_state.is_contiguous()
        and share_gate_weight.is_contiguous()
        and share_expert_output.is_contiguous()
        and hidden_state.dim() == 2
        and share_gate_weight.dim() == 2
        and share_expert_output.dim() == 2
        and share_gate_weight.size(0) == 1
        and hidden_state.shape == share_expert_output.shape
        and hidden_state.size(1) == share_gate_weight.size(1)
        and hidden_state.size(1) % 8 == 0
        and hidden_state.dtype in (torch.float16, torch.bfloat16)
        and share_gate_weight.dtype == hidden_state.dtype
        and share_expert_output.dtype == hidden_state.dtype
    )


def _fused_share_gate_sigmoid_mul_impl(
    hidden_state: torch.Tensor,
    share_gate_weight: torch.Tensor,
    share_expert_output: torch.Tensor,
    output: torch.Tensor,
) -> None:
    _fused_share_gate_sigmoid_mul_module().sgl_musa_fused_share_gate_sigmoid_mul(
        output,
        hidden_state,
        share_gate_weight,
        share_expert_output,
    )


@register_custom_op(
    op_name="musa_fused_share_gate_sigmoid_mul",
    mutates_args=["output"],
)
def _fused_share_gate_sigmoid_mul_custom(
    hidden_state: torch.Tensor,
    share_gate_weight: torch.Tensor,
    share_expert_output: torch.Tensor,
    output: torch.Tensor,
) -> None:
    _fused_share_gate_sigmoid_mul_impl(
        hidden_state,
        share_gate_weight,
        share_expert_output,
        output,
    )


def fused_share_gate_sigmoid_mul(
    hidden_state: torch.Tensor,
    share_gate_weight: torch.Tensor,
    share_expert_output: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(share_expert_output)
    if _can_use_fused_share_gate_sigmoid_mul(
        hidden_state, share_gate_weight, share_expert_output
    ):
        _fused_share_gate_sigmoid_mul_custom(
            hidden_state,
            share_gate_weight,
            share_expert_output,
            output,
        )
        return output

    gate = F.linear(hidden_state, share_gate_weight)
    return torch.mul(torch.sigmoid(gate), share_expert_output, out=output)
