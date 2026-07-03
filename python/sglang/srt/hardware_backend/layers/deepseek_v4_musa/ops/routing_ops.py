import importlib
import logging
import os
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F

from ..kernels.routing_kernels import (
    _tilelang_hash_topk_kernel,
    _tilelang_hash_topk_warp_block_kernel,
    _tilelang_hash_topk_warp_kernel,
    _tilelang_mask_topk_ids_int32_kernel,
    _tilelang_mask_topk_ids_int64_kernel,
    _tilelang_moe_fused_gate_kernel,
    _tilelang_topk_ids_logical_to_physical_static_int32_kernel,
    _tilelang_topk_ids_logical_to_physical_static_int64_kernel,
)
from .ops_common import _debug_musa_allow_torch_fallback, _debug_musa_torch_fallback, _has_musa_tensor

_ROUTING_TRACE_ENV = "SGLANG_DEEPSEEK_V4_MUSA_ROUTING_TRACE_DISPATCH"


def _routing_trace(message: str) -> None:
    if os.environ.get(_ROUTING_TRACE_ENV) == "1":
        print(f"ROUTING_DISPATCH {message}", flush=True)


def _sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x).sqrt()

def _hash_topk_input_dtype_name(dtype: torch.dtype) -> Optional[str]:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return None


def _hash_topk_tid2eid_dtype_name(dtype: torch.dtype) -> Optional[str]:
    if dtype == torch.int64:
        return "int64"
    if dtype == torch.int32:
        return "int32"
    return None


def _try_tilelang_hash_topk_musa(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if router_logits.device.type != "musa":
        return None
    input_dtype = _hash_topk_input_dtype_name(router_logits.dtype)
    if input_dtype is None or router_logits.dim() != 2:
        return None
    input_ids_dtype = _hash_topk_tid2eid_dtype_name(input_ids.dtype)
    tid2eid_dtype = _hash_topk_tid2eid_dtype_name(tid2eid.dtype)
    if input_ids.dim() != 1 or tid2eid.dim() != 2:
        return None
    if input_ids_dtype is None or tid2eid_dtype is None:
        return None
    if input_ids.shape[0] != router_logits.shape[0]:
        return None
    if num_fused_shared_experts < 0 or routed_scaling_factor == 0:
        return None

    topk = tid2eid.shape[1]
    output_topk = topk + num_fused_shared_experts
    routed_scores = torch.empty((router_logits.shape[0], output_topk), dtype=torch.float32, device=router_logits.device)
    routed_ids = torch.empty((router_logits.shape[0], output_topk), dtype=torch.int64, device=router_logits.device)

    try:
        if output_topk <= 32:
            kernel = _tilelang_hash_topk_warp_kernel(
                topk,
                num_fused_shared_experts,
                input_dtype=input_dtype,
                input_ids_dtype=input_ids_dtype,
                tid2eid_dtype=tid2eid_dtype,
            )
        elif os.environ.get("SGLANG_OPT_HASH_TOPK_WARP_BLOCK", "0") == "1" and output_topk <= 128:
            kernel = _tilelang_hash_topk_warp_block_kernel(
                topk,
                num_fused_shared_experts,
                threads=128,
                input_dtype=input_dtype,
                input_ids_dtype=input_ids_dtype,
                tid2eid_dtype=tid2eid_dtype,
            )
        elif os.environ.get("SGLANG_OPT_HASH_TOPK_WARP_BLOCK", "0") == "1" and output_topk <= 256:
            kernel = _tilelang_hash_topk_warp_block_kernel(
                topk,
                num_fused_shared_experts,
                threads=256,
                input_dtype=input_dtype,
                input_ids_dtype=input_ids_dtype,
                tid2eid_dtype=tid2eid_dtype,
            )
        else:
            if topk <= 8:
                threads = 128
            else:
                threads = 256
            kernel = _tilelang_hash_topk_kernel(
                topk,
                num_fused_shared_experts,
                threads=threads,
                input_dtype=input_dtype,
                input_ids_dtype=input_ids_dtype,
                tid2eid_dtype=tid2eid_dtype,
            )
        kernel(
            router_logits,
            input_ids,
            tid2eid,
            routed_scores,
            routed_ids,
            1.0 / routed_scaling_factor,
        )
    except Exception:
        return None
    return routed_scores, routed_ids

def _tilelang_hash_topk_miss_reason(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> str:
    if router_logits.device.type != "musa":
        return f"router_logits is on {router_logits.device}, not MUSA"
    if _hash_topk_input_dtype_name(router_logits.dtype) is None:
        return f"unsupported router_logits dtype={router_logits.dtype}"
    if router_logits.dim() != 2:
        return f"router_logits must be 2D, got shape={tuple(router_logits.shape)}"
    if input_ids.dim() != 1:
        return f"input_ids must be 1D, got shape={tuple(input_ids.shape)}"
    if tid2eid.dim() != 2:
        return f"tid2eid must be 2D, got shape={tuple(tid2eid.shape)}"
    if input_ids.dtype not in (torch.int32, torch.int64):
        return f"input_ids must be int32 or int64, got {input_ids.dtype}"
    if _hash_topk_tid2eid_dtype_name(tid2eid.dtype) is None:
        return f"tid2eid must be int32 or int64, got {tid2eid.dtype}"
    if input_ids.shape[0] != router_logits.shape[0]:
        return f"input_ids rows {input_ids.shape[0]} != router_logits rows {router_logits.shape[0]}"
    if num_fused_shared_experts < 0:
        return f"num_fused_shared_experts must be >= 0, got {num_fused_shared_experts}"
    if routed_scaling_factor == 0:
        return "routed_scaling_factor must be non-zero"
    return "TileLang kernel launch failed"

def _normalize_topk_weights_musa(topk_weights: torch.Tensor) -> torch.Tensor:
    if topk_weights.device.type == "musa":
        try:
            tile_moe = importlib.import_module("tile_kernels.moe")
            _, normalized = tile_moe.normalize_weight(topk_weights.contiguous().to(torch.float32))
            return normalized
        except Exception:
            pass
    denominator = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return topk_weights / denominator


def _try_tilelang_moe_fused_gate_musa(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    scoring_func: str = "sqrtsoftplus",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if gating_output.device.type != "musa":
        _routing_trace("moe_fused_gate status=miss reason=non_musa")
        return None
    if gating_output.dtype != torch.float32 or correction_bias.dtype != torch.float32:
        _routing_trace(
            "moe_fused_gate status=miss reason=dtype "
            f"gating_dtype={gating_output.dtype} bias_dtype={correction_bias.dtype}"
        )
        return None
    if gating_output.dim() != 2 or correction_bias.dim() != 1:
        _routing_trace(
            "moe_fused_gate status=miss reason=rank "
            f"gating_dim={gating_output.dim()} bias_dim={correction_bias.dim()}"
        )
        return None
    if not gating_output.is_contiguous() or not correction_bias.is_contiguous():
        _routing_trace(
            "moe_fused_gate status=miss reason=non_contiguous "
            f"gating_stride={gating_output.stride()} bias_stride={correction_bias.stride()}"
        )
        return None
    if gating_output.shape[1] != correction_bias.shape[0]:
        _routing_trace(
            "moe_fused_gate status=miss reason=shape "
            f"gating_shape={tuple(gating_output.shape)} bias_shape={tuple(correction_bias.shape)}"
        )
        return None
    if scoring_func != "sqrtsoftplus":
        _routing_trace(f"moe_fused_gate status=miss reason=scoring_func value={scoring_func}")
        return None
    if num_fused_shared_experts != 0:
        _routing_trace(
            "moe_fused_gate status=miss reason=shared_experts "
            f"value={num_fused_shared_experts}"
        )
        return None
    if not renormalize or apply_routed_scaling_factor_on_output:
        _routing_trace(
            "moe_fused_gate status=miss reason=normalize_flags "
            f"renormalize={renormalize} apply_scale={apply_routed_scaling_factor_on_output}"
        )
        return None
    if topk <= 0 or topk > gating_output.shape[1]:
        _routing_trace(f"moe_fused_gate status=miss reason=topk topk={topk}")
        return None

    # Current TileLang fused gate is beneficial on the measured decode and
    # small/paged prefill shapes, but it loses to torch on long-prefill rows.
    # Keep the optimization on shapes measured positive.
    if gating_output.shape[0] > 2048:
        _routing_trace(
            "moe_fused_gate status=miss reason=row_threshold "
            f"rows={gating_output.shape[0]}"
        )
        return None

    num_experts = gating_output.shape[1]
    threads = 256 if num_experts > 128 else 128
    topk_weights = torch.empty((gating_output.shape[0], topk), dtype=torch.float32, device=gating_output.device)
    topk_ids = torch.empty((gating_output.shape[0], topk), dtype=torch.int32, device=gating_output.device)

    try:
        kernel = _tilelang_moe_fused_gate_kernel(num_experts, topk, threads=threads)
        kernel(gating_output, correction_bias, topk_weights, topk_ids)
    except Exception as exc:
        _routing_trace(f"moe_fused_gate status=miss reason=kernel_exception error={exc!r}")
        return None
    _routing_trace(
        "moe_fused_gate status=hit "
        f"shape={tuple(gating_output.shape)} topk={topk} threads={threads}"
    )
    return topk_weights, topk_ids


def tilelang_moe_fused_gate_musa(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    scoring_func: str = "sqrtsoftplus",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    return _try_tilelang_moe_fused_gate_musa(
        gating_output,
        correction_bias,
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )


def _routing_torch_fallback_guard(
    op_name: str,
    *tensors: Optional[torch.Tensor],
    detail: Optional[str] = None,
) -> None:
    if not _has_musa_tensor(*tensors):
        return
    message = (
        f"DeepSeekV4 MUSA {op_name} has no torch fallback by default; "
        "torch fallback is disabled on MUSA"
    )
    if detail:
        message = f"{message}; {detail}"
    if not _debug_musa_allow_torch_fallback():
        raise NotImplementedError(message)
    _debug_musa_torch_fallback(f"DeepSeekV4 MUSA {op_name} using torch fallback after TileLang miss")

def hash_topk_musa(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func != "sqrtsoftplus":
        raise NotImplementedError(f"Unsupported DeepSeekV4 hash topk scoring_func={scoring_func!r}")

    tilelang_result = _try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts,
        routed_scaling_factor,
    )
    if tilelang_result is not None:
        routed_scores, routed_ids = tilelang_result
        return tilelang_result

    _routing_torch_fallback_guard(
        "hash_topk",
        router_logits,
        input_ids,
        tid2eid,
        detail=_tilelang_hash_topk_miss_reason(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts,
            routed_scaling_factor,
        ),
    )
    routed_ids = tid2eid[input_ids.long()].to(torch.int64)
    routed_scores = _sqrtsoftplus(router_logits).gather(1, routed_ids).to(torch.float32)
    routed_scores = _normalize_topk_weights_musa(routed_scores)

    if num_fused_shared_experts == 0:
        return routed_scores, routed_ids

    num_tokens = router_logits.shape[0]
    num_routed_experts = router_logits.shape[1]
    shared_ids = torch.arange(
        num_routed_experts,
        num_routed_experts + num_fused_shared_experts,
        dtype=torch.int64,
        device=router_logits.device,
    ).expand(num_tokens, -1)
    shared_weights = torch.full(
        (num_tokens, num_fused_shared_experts),
        1.0 / routed_scaling_factor,
        dtype=torch.float32,
        device=router_logits.device,
    )
    return (
        torch.cat([routed_scores, shared_weights], dim=1),
        torch.cat([routed_ids, shared_ids], dim=1),
    )

def _try_tilelang_mask_topk_ids_musa(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor) -> bool:
    if topk_ids.device.type != "musa" or topk_ids.dtype not in (torch.int32, torch.int64):
        return False
    if topk_ids.dim() != 2:
        raise NotImplementedError(
            "DeepSeekV4 MUSA mask_topk_ids TileLang path requires a 2D int32/int64 topk_ids tensor"
        )
    if not topk_ids.is_contiguous():
        raise NotImplementedError(
            "DeepSeekV4 MUSA mask_topk_ids TileLang path requires contiguous int32/int64 topk_ids"
        )
    if num_token_non_padded.numel() != 1:
        raise NotImplementedError(
            "DeepSeekV4 MUSA mask_topk_ids TileLang path requires a single num_token_non_padded value"
        )
    if num_token_non_padded.device != topk_ids.device or num_token_non_padded.dtype != torch.int32:
        raise NotImplementedError(
            "DeepSeekV4 MUSA mask_topk_ids TileLang path requires MUSA int32 num_token_non_padded"
        )

    try:
        kernel_factory = (
            _tilelang_mask_topk_ids_int32_kernel
            if topk_ids.dtype == torch.int32
            else _tilelang_mask_topk_ids_int64_kernel
        )
        kernel = kernel_factory(topk_ids.shape[1])
        num_token_non_padded_storage = num_token_non_padded.as_strided(
            (1,),
            (1,),
            storage_offset=num_token_non_padded.storage_offset(),
        )
        kernel(topk_ids, num_token_non_padded_storage)
    except Exception as exc:
        raise NotImplementedError(
            "DeepSeekV4 MUSA mask_topk_ids TileLang launch failed for supported int64 input"
        ) from exc
    return True

def mask_topk_ids_musa(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor) -> torch.Tensor:
    if _try_tilelang_mask_topk_ids_musa(topk_ids, num_token_non_padded):
        return topk_ids
    _routing_torch_fallback_guard("mask_topk_ids", topk_ids, num_token_non_padded)
    row_ids = torch.arange(topk_ids.shape[0], device=topk_ids.device).view(-1, 1)
    valid_rows = num_token_non_padded.to(device=topk_ids.device, dtype=row_ids.dtype).view(1, 1)
    topk_ids.masked_fill_(row_ids >= valid_rows, -1)
    return topk_ids

def _try_tilelang_topk_ids_logical_to_physical_static_musa(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
) -> bool:
    if topk_ids.device.type != "musa" or topk_ids.dtype not in (torch.int32, torch.int64):
        return False
    if (
        logical_to_physical_map.device != topk_ids.device
        or logical_to_physical_map.dtype not in (torch.int32, torch.int64)
    ):
        return False
    if topk_ids.dtype == torch.int32 and logical_to_physical_map.dtype == torch.int64:
        logical_to_physical_map = logical_to_physical_map.to(torch.int32)
    if topk_ids.dim() != 2 or logical_to_physical_map.dim() != 1:
        raise NotImplementedError(
            "DeepSeekV4 MUSA static expert mapping requires 2D int32/int64 topk_ids and 1D int32/int64 map"
        )
    if not topk_ids.is_contiguous() or not logical_to_physical_map.is_contiguous():
        raise NotImplementedError(
            "DeepSeekV4 MUSA static expert mapping requires contiguous int32/int64 tensors"
        )
    try:
        if topk_ids.dtype == torch.int64:
            map_dtype = "int64" if logical_to_physical_map.dtype == torch.int64 else "int32"
            kernel = _tilelang_topk_ids_logical_to_physical_static_int64_kernel(
                topk_ids.shape[1],
                map_dtype,
            )
        else:
            kernel = _tilelang_topk_ids_logical_to_physical_static_int32_kernel(topk_ids.shape[1])
        kernel(topk_ids, logical_to_physical_map)
    except Exception as exc:
        raise NotImplementedError(
            "DeepSeekV4 MUSA static expert mapping TileLang launch failed for supported int32/int64 input"
        ) from exc
    return True

def topk_ids_logical_to_physical_static_musa(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
) -> torch.Tensor:
    if _try_tilelang_topk_ids_logical_to_physical_static_musa(topk_ids, logical_to_physical_map):
        return topk_ids
    _routing_torch_fallback_guard("topk_ids_logical_to_physical_static", topk_ids, logical_to_physical_map)
    topk_ids.copy_(logical_to_physical_map[topk_ids])
    return topk_ids

__all__ = [
    '_sqrtsoftplus',
    '_try_tilelang_hash_topk_musa',
    '_try_tilelang_moe_fused_gate_musa',
    'tilelang_moe_fused_gate_musa',
    '_normalize_topk_weights_musa',
    'hash_topk_musa',
    '_try_tilelang_mask_topk_ids_musa',
    'mask_topk_ids_musa',
    '_try_tilelang_topk_ids_logical_to_physical_static_musa',
    'topk_ids_logical_to_physical_static_musa',
]
