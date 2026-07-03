from __future__ import annotations

import os

import torch

from ..kernels.moe_prefill_kernels import (
    _tilelang_moe_deepgemm_compact_quant_scatter_kernel,
    _tilelang_moe_deepgemm_static_cap_quant_scatter_kernel,
    _tilelang_moe_deepgemm_static_cap_src2dst_quant_scatter_kernel,
    _tilelang_moe_post_combine_kernel,
    _tilelang_moe_post_combine_src2dst_cached_kernel,
    _tilelang_moe_post_combine_src2dst_kernel,
)


def _dtype_name(dtype: torch.dtype) -> str | None:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return None


def _uint8_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8).reshape(tensor.shape)


def try_moe_deepgemm_compact_quant_scatter_tilelang_musa(
    hidden_states: torch.Tensor,
    compact_input: torch.Tensor,
    compact_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    offsets: torch.Tensor,
    route_ranks: torch.Tensor,
    src2dst: torch.Tensor,
    num_local_experts: int,
    topk: int,
    group_size: int,
) -> bool:
    if hidden_states.device.type != "musa":
        return False
    if hidden_states.dim() != 2 or compact_input.dim() != 2 or compact_scale.dim() != 2:
        return False
    if topk_ids.dim() != 2 or route_ranks.numel() != topk_ids.numel() or src2dst.shape != topk_ids.shape:
        return False
    if topk_ids.shape[1] != topk or hidden_states.shape[1] != compact_input.shape[1]:
        return False
    if hidden_states.shape[1] % group_size != 0 or group_size != 128:
        return False
    if compact_scale.shape != (compact_input.shape[0], hidden_states.shape[1] // group_size):
        return False
    if not (
        hidden_states.is_contiguous()
        and compact_input.is_contiguous()
        and compact_scale.is_contiguous()
        and topk_ids.is_contiguous()
        and offsets.is_contiguous()
        and route_ranks.is_contiguous()
        and src2dst.is_contiguous()
    ):
        return False
    if topk_ids.dtype != torch.int32 or route_ranks.dtype != torch.int32 or src2dst.dtype != torch.int32:
        return False
    if offsets.dtype != torch.int32 or offsets.numel() < num_local_experts:
        return False
    if compact_input.dtype != torch.float8_e4m3fn or compact_scale.dtype != torch.float32:
        return False

    input_dtype = _dtype_name(hidden_states.dtype)
    if input_dtype not in {"bfloat16", "float16", "float32"}:
        return False
    if compact_input.numel() == 0:
        return True

    try:
        kernel = _tilelang_moe_deepgemm_compact_quant_scatter_kernel(
            hidden_states.shape[1], topk, input_dtype, group_size=group_size
        )
        kernel(
            hidden_states,
            _uint8_view(compact_input),
            compact_scale,
            topk_ids,
            offsets,
            route_ranks.reshape_as(topk_ids),
            src2dst,
            int(num_local_experts),
        )
    except Exception as exc:
        if os.environ.get("SGLANG_DSV4_MUSA_MOE_DEBUG") == "1":
            print(
                "compact quant scatter TileLang miss: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return False
    return True


def try_moe_deepgemm_static_cap_quant_scatter_tilelang_musa(
    hidden_states: torch.Tensor,
    compact_input: torch.Tensor,
    compact_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    route_ranks: torch.Tensor,
    src2dst: torch.Tensor,
    overflow_flag: torch.Tensor,
    num_local_experts: int,
    cap_per_expert: int,
    topk: int,
    group_size: int,
) -> bool:
    if hidden_states.device.type != "musa":
        return False
    if cap_per_expert <= 0 or num_local_experts <= 0:
        return False
    if hidden_states.dim() != 2 or compact_input.dim() != 2 or compact_scale.dim() != 2:
        return False
    if topk_ids.dim() != 2 or route_ranks.numel() != topk_ids.numel() or src2dst.shape != topk_ids.shape:
        return False
    if topk_ids.shape[1] != topk or hidden_states.shape[1] != compact_input.shape[1]:
        return False
    if compact_input.shape[0] != int(num_local_experts) * int(cap_per_expert):
        return False
    if hidden_states.shape[1] % group_size != 0 or group_size != 128:
        return False
    if compact_scale.shape != (compact_input.shape[0], hidden_states.shape[1] // group_size):
        return False
    if overflow_flag.shape != (1,) or overflow_flag.dtype != torch.int32:
        return False
    if not (
        hidden_states.is_contiguous()
        and compact_input.is_contiguous()
        and compact_scale.is_contiguous()
        and topk_ids.is_contiguous()
        and route_ranks.is_contiguous()
        and src2dst.is_contiguous()
        and overflow_flag.is_contiguous()
    ):
        return False
    if topk_ids.dtype != torch.int32 or route_ranks.dtype != torch.int32 or src2dst.dtype != torch.int32:
        return False
    if compact_input.dtype != torch.float8_e4m3fn or compact_scale.dtype != torch.float32:
        return False

    input_dtype = _dtype_name(hidden_states.dtype)
    if input_dtype not in {"bfloat16", "float16", "float32"}:
        return False
    if compact_input.numel() == 0:
        return True

    try:
        kernel = _tilelang_moe_deepgemm_static_cap_quant_scatter_kernel(
            hidden_states.shape[1], topk, input_dtype, group_size=group_size
        )
        kernel(
            hidden_states,
            _uint8_view(compact_input),
            compact_scale,
            topk_ids,
            route_ranks.reshape_as(topk_ids),
            src2dst,
            overflow_flag,
            int(num_local_experts),
            int(cap_per_expert),
        )
    except Exception as exc:
        if os.environ.get("SGLANG_DSV4_MUSA_MOE_DEBUG") == "1":
            print(
                "static-cap quant scatter TileLang miss: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return False
    return True


def try_moe_deepgemm_static_cap_src2dst_quant_scatter_tilelang_musa(
    hidden_states: torch.Tensor,
    compact_input: torch.Tensor,
    compact_scale: torch.Tensor,
    src2dst: torch.Tensor,
    topk: int,
    group_size: int,
    groups_per_cta: int = 8,
) -> bool:
    if hidden_states.device.type != "musa":
        return False
    if hidden_states.dim() != 2 or compact_input.dim() != 2 or compact_scale.dim() != 2:
        return False
    if src2dst.dim() != 2 or src2dst.shape[0] != hidden_states.shape[0]:
        return False
    if src2dst.shape[1] != topk or hidden_states.shape[1] != compact_input.shape[1]:
        return False
    if hidden_states.shape[1] % group_size != 0 or group_size != 128:
        return False
    if compact_scale.shape != (compact_input.shape[0], hidden_states.shape[1] // group_size):
        return False
    if not (
        hidden_states.is_contiguous()
        and compact_input.is_contiguous()
        and compact_scale.is_contiguous()
        and src2dst.is_contiguous()
    ):
        return False
    if src2dst.dtype != torch.int32:
        return False
    if compact_input.dtype != torch.float8_e4m3fn or compact_scale.dtype != torch.float32:
        return False

    input_dtype = _dtype_name(hidden_states.dtype)
    if input_dtype not in {"bfloat16", "float16", "float32"}:
        return False
    if compact_input.numel() == 0:
        return True

    try:
        kernel = _tilelang_moe_deepgemm_static_cap_src2dst_quant_scatter_kernel(
            hidden_states.shape[1],
            topk,
            input_dtype,
            group_size=group_size,
            groups_per_cta=groups_per_cta,
        )
        kernel(hidden_states, _uint8_view(compact_input), compact_scale, src2dst)
    except Exception as exc:
        if os.environ.get("SGLANG_DSV4_MUSA_MOE_DEBUG") == "1":
            print(
                "static-cap src2dst quant scatter TileLang miss: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return False
    return True


def try_moe_deepgemm_compact_src2dst_quant_scatter_tilelang_musa(
    hidden_states: torch.Tensor,
    compact_input: torch.Tensor,
    compact_scale: torch.Tensor,
    src2dst: torch.Tensor,
    topk: int,
    group_size: int,
) -> bool:
    return try_moe_deepgemm_static_cap_src2dst_quant_scatter_tilelang_musa(
        hidden_states,
        compact_input,
        compact_scale,
        src2dst,
        topk,
        group_size,
        groups_per_cta=16,
    )


def try_moe_post_combine_tilelang_musa(
    down_output: torch.Tensor,
    output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
    allow_slow_shape: bool = False,
) -> bool:
    if down_output.device.type != "musa":
        return False
    if down_output.dim() != 2 or output.dim() != 2 or output.shape[1] != down_output.shape[1]:
        return False
    if topk_ids.dim() != 2 or src2dst.shape != topk_ids.shape or topk_weights.shape != topk_ids.shape:
        return False
    if topk_ids.shape != (output.shape[0], topk):
        return False
    if not (
        down_output.is_contiguous()
        and output.is_contiguous()
        and src2dst.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        return False
    if topk_ids.dtype != torch.int32 or src2dst.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        return False
    # m8192 is slower than the Triton scatter-add on the current MUSA stack;
    # keep auto-dispatch conservative and use TileLang only for measured wins.
    if not allow_slow_shape and output.shape[0] > 4096:
        return False

    input_dtype = _dtype_name(down_output.dtype)
    output_dtype = _dtype_name(output.dtype)
    if input_dtype not in {"bfloat16", "float16"} or output_dtype not in {"bfloat16", "float16"}:
        return False

    try:
        kernel = _tilelang_moe_post_combine_kernel(
            down_output.shape[1], topk, input_dtype, output_dtype
        )
        kernel(down_output, output, src2dst, topk_ids, topk_weights)
    except Exception:
        return False
    return True


def try_moe_post_combine_src2dst_tilelang_musa(
    down_output: torch.Tensor,
    output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
    block_h: int = 1024,
) -> bool:
    if down_output.device.type != "musa":
        return False
    if down_output.dim() != 2 or output.dim() != 2 or output.shape[1] != down_output.shape[1]:
        return False
    if src2dst.dim() != 2 or topk_weights.shape != src2dst.shape:
        return False
    if src2dst.shape != (output.shape[0], topk):
        return False
    if not (
        down_output.is_contiguous()
        and output.is_contiguous()
        and src2dst.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        return False
    if src2dst.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        return False

    input_dtype = _dtype_name(down_output.dtype)
    output_dtype = _dtype_name(output.dtype)
    if input_dtype not in {"bfloat16", "float16"} or output_dtype not in {"bfloat16", "float16"}:
        return False

    try:
        kernel = _tilelang_moe_post_combine_src2dst_kernel(
            down_output.shape[1], topk, input_dtype, output_dtype, block_h=block_h
        )
        kernel(down_output, output, src2dst, topk_weights)
    except Exception:
        return False
    return True


def try_moe_post_combine_src2dst_cached_tilelang_musa(
    down_output: torch.Tensor,
    output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
    block_h: int = 1024,
) -> bool:
    if down_output.device.type != "musa":
        return False
    if down_output.dim() != 2 or output.dim() != 2 or output.shape[1] != down_output.shape[1]:
        return False
    if src2dst.dim() != 2 or topk_weights.shape != src2dst.shape:
        return False
    if src2dst.shape != (output.shape[0], topk):
        return False
    if not (
        down_output.is_contiguous()
        and output.is_contiguous()
        and src2dst.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        return False
    if src2dst.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        return False

    input_dtype = _dtype_name(down_output.dtype)
    output_dtype = _dtype_name(output.dtype)
    if input_dtype not in {"bfloat16", "float16"} or output_dtype not in {"bfloat16", "float16"}:
        return False

    try:
        kernel = _tilelang_moe_post_combine_src2dst_cached_kernel(
            down_output.shape[1], topk, input_dtype, output_dtype, block_h=block_h
        )
        kernel(down_output, output, src2dst, topk_weights)
    except Exception as exc:
        if os.environ.get("SGLANG_DSV4_MUSA_MOE_DEBUG") == "1":
            print(
                "post-combine cached src2dst TileLang miss: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return False
    return True


__all__ = [
    "try_moe_deepgemm_compact_quant_scatter_tilelang_musa",
    "try_moe_deepgemm_compact_src2dst_quant_scatter_tilelang_musa",
    "try_moe_deepgemm_static_cap_quant_scatter_tilelang_musa",
    "try_moe_deepgemm_static_cap_src2dst_quant_scatter_tilelang_musa",
    "try_moe_post_combine_src2dst_cached_tilelang_musa",
    "try_moe_post_combine_tilelang_musa",
    "try_moe_post_combine_src2dst_tilelang_musa",
]
