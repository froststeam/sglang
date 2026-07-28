from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op

MOE_GEMV_CONFIG_ABI = 6


class MoeGemvConfig(NamedTuple):
    gate_config_id: int
    down_config_id: int
    down_reduce_config_id: int


_DEFAULT_MOE_GEMV_CONFIG = MoeGemvConfig(-1, -1, -1)


@lru_cache(maxsize=1)
def _load_moe_gemv_configs() -> dict[tuple[object, ...], MoeGemvConfig]:
    path = os.getenv("SGLANG_MUSA_MOE_GEMV_CONFIG")
    if not path:
        return {}
    try:
        records = json.loads(Path(path).read_text())
        capability = tuple(torch.get_device_module("musa").get_device_capability())
    except (
        AssertionError,
        AttributeError,
        OSError,
        RuntimeError,
        json.JSONDecodeError,
    ):
        return {}
    if not isinstance(records, list):
        return {}

    configs = {}
    for record in records:
        try:
            if (
                record["status"] != "ok"
                or record["validation_passed"] is not True
                or int(record["config_abi"]) != MOE_GEMV_CONFIG_ABI
                or tuple(record["device_capability"]) != capability
            ):
                continue
            key = (
                str(record["dtype"]),
                int(record["tokens"]),
                int(record["hidden"]),
                int(record["intermediate"]),
                int(record["experts"]),
                int(record["routed_topk"]),
                int(record["shared_experts"]),
            )
            configs[key] = MoeGemvConfig(
                int(record["gate_config_id"]),
                int(record["down_config_id"]),
                int(record["down_reduce_config_id"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return configs


def get_moe_gemv_config(
    dtype: str,
    tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    routed_topk: int,
    shared_experts: int,
) -> MoeGemvConfig:
    if tokens > 32:
        return _DEFAULT_MOE_GEMV_CONFIG
    key = (dtype, tokens, hidden, intermediate, experts, routed_topk, shared_experts)
    return _load_moe_gemv_configs().get(key, _DEFAULT_MOE_GEMV_CONFIG)


def _jit_csrc_dir() -> Path:
    return Path(__file__).resolve().parent


def _jit_flags() -> tuple[str, ...]:
    return (
        f"-I{_jit_csrc_dir()}",
        f"-I{_jit_csrc_dir() / 'include'}",
        "-mtgpu",
        "-Od3",
        "-fmusa-flush-denormals-to-zero",
        "-fno-strict-aliasing",
        "-mllvm",
        "-mtgpu-load-store-opt=1",
        "-mllvm",
        "-mtgpu-memory-sched-mutation=1",
        "-Wno-macro-redefined",
    )


@cache_once
def _gemv_module():
    return load_musa_jit(
        "sglang_musa_gemv",
        ("gemv/gemv.mu",),
        extra_musa_cflags=_jit_flags(),
    )


def _empty_scale_arg(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), device=x.device, dtype=torch.float32)


@register_custom_op(
    op_name="musa_gemv_out",
    mutates_args=["C"],
)
def _musa_gemv_out_custom(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    use_int4_w4a16: bool,
    fuse_swiglu: bool,
    fuse_silu: bool,
    config_id: int,
) -> None:
    has_b_scale = B_scale is not None
    has_bias = bias is not None
    _gemv_module().sgl_musa_gemv(
        A,
        B,
        C,
        B_scale if has_b_scale else _empty_scale_arg(A),
        bias if has_bias else _empty_scale_arg(A),
        bool(has_b_scale),
        bool(has_bias),
        bool(use_int4_w4a16),
        bool(fuse_swiglu),
        bool(fuse_silu),
        int(config_id),
    )


def musa_gemv_out(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    use_int4_w4a16: bool,
    fuse_swiglu: bool,
    fuse_silu: bool = False,
    config_id: int = -1,
) -> None:
    """Run dense GEMV into a caller-provided output tensor."""
    _musa_gemv_out_custom(
        A,
        B,
        C,
        B_scale,
        bias,
        use_int4_w4a16,
        fuse_swiglu,
        fuse_silu,
        config_id,
    )


def musa_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    fuse_swiglu: bool = False,
    fuse_silu: bool = False,
    config_id: int = -1,
) -> torch.Tensor:
    """Run dense BF16 or block-FP8 GEMV and allocate its output."""
    input_shape = A.shape
    if A.dim() != 2:
        A = A.reshape(-1, input_shape[-1])
    if fuse_swiglu and fuse_silu:
        raise ValueError("fused SwiGLU and SiLU cannot both be enabled")
    if fuse_swiglu and bias is not None:
        raise ValueError("fused SwiGLU GEMV does not support bias")
    output_n = B.shape[0] // 2 if fuse_swiglu else B.shape[0]
    if fuse_swiglu and B.shape[0] % 2 != 0:
        raise ValueError("fused SwiGLU GEMV requires an even weight N")
    C = torch.empty((A.shape[0], output_n), device=A.device, dtype=A.dtype)
    musa_gemv_out(
        A,
        B,
        C,
        B_scale,
        bias if fuse_silu else None,
        False,
        fuse_swiglu,
        fuse_silu,
        config_id,
    )
    output = C if bias is None or fuse_silu else C + bias
    if len(input_shape) != 2:
        output = output.view(*input_shape[:-1], output_n)
    return output


@cache_once
def _moe_gemv_module():
    return load_musa_jit(
        "sglang_musa_moe_gemv",
        ("gemv/moe/moe_gemv.mu",),
        extra_musa_cflags=_jit_flags(),
    )


@cache_once
def _moe_gemv_w8ax_block128_module():
    return load_musa_jit(
        "sglang_musa_moe_gemv_w8ax_block128",
        ("gemv/moe/moe_gemv_w8ax_block128.mu",),
        extra_musa_cflags=_jit_flags(),
    )


@cache_once
def _moe_gemv_w4a16_module():
    return load_musa_jit(
        "sglang_musa_moe_gemv_w4a16",
        ("gemv/moe/moe_gemv_w4a16.mu",),
        extra_musa_cflags=_jit_flags(),
    )


@cache_once
def _moe_down_reduce_module():
    return load_musa_jit(
        "sglang_musa_moe_down_reduce",
        ("gemv/moe/moe_down_reduce.mu",),
        extra_musa_cflags=_jit_flags(),
    )


def _select_moe_gemv_module(B: torch.Tensor, use_int4_w4a16: bool):
    if use_int4_w4a16:
        return _moe_gemv_w4a16_module()
    if B.dtype == torch.float8_e4m3fn:
        return _moe_gemv_w8ax_block128_module()
    return _moe_gemv_module()


@register_custom_op(op_name="musa_moe_gemv", mutates_args=["C"])
def _musa_moe_gemv_custom(
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
    fuse_swiglu: bool,
    config_id: int,
) -> None:
    has_a_scale = A_scale is not None
    has_b_scale = B_scale is not None
    _select_moe_gemv_module(B, use_int4_w4a16).sgl_musa_moe_gemv(
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
        bool(fuse_swiglu),
        int(config_id),
    )


@register_custom_op(op_name="musa_moe_down_reduce", mutates_args=["C"])
def _musa_moe_down_reduce_custom(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    config_id: int,
) -> None:
    _moe_down_reduce_module().sgl_musa_moe_down_reduce(
        A,
        B,
        C,
        B_scale if B_scale is not None else _empty_scale_arg(A),
        topk_weights,
        topk_ids,
        int(topk),
        float(routed_scaling_factor),
        int(config_id),
    )


def musa_moe_gemv(
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
    fuse_swiglu: bool,
    config_id: int = -1,
) -> None:
    """Run routed MoE GEMV into a caller-provided output tensor."""
    _musa_moe_gemv_custom(
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
        fuse_swiglu,
        config_id,
    )


def musa_moe_down_reduce(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    routed_scaling_factor: float = 1.0,
    config_id: int = -1,
) -> None:
    """Fuse MoE down GEMV with top-k route reduction."""
    _musa_moe_down_reduce_custom(
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        topk,
        routed_scaling_factor,
        config_id,
    )


__all__ = [
    "MOE_GEMV_CONFIG_ABI",
    "MoeGemvConfig",
    "get_moe_gemv_config",
    "musa_gemv",
    "musa_gemv_out",
    "musa_moe_down_reduce",
    "musa_moe_gemv",
]
