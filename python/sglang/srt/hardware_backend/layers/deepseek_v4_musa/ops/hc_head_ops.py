import logging
import os
from dataclasses import dataclass

import torch

from ..kernels.hc_head_kernels import (
    _tilelang_hc_head_fused_splitk_warp_kernel,
    _tilelang_hc_head_linear_splitk_kernel,
    _tilelang_hc_head_linear_splitk_warp_kernel,
)
from .ops_common import _has_musa_tensor
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

__all__ = [
    "HcHeadTileLangConfig",
    "hc_head_tilelang",
    "hc_head_linear_tilelang",
    "try_hc_head_tilelang",
    "try_hc_head_linear_tilelang",
]


@dataclass(frozen=True)
class HcHeadTileLangConfig:
    impl: str = "auto"
    split_k: int = 0
    token_block: int = 16
    hidden_block: int = 0


def _get_split_k(hidden_size: int, num_tokens: int, config: HcHeadTileLangConfig) -> int:
    split_k = config.split_k
    if split_k <= 0:
        # Small-M decode needs K-parallelism; prefill already has enough M CTAs,
        # so lower split-K reduces partial writes and the second-stage reduce.
        if hidden_size == 16384:
            split_k = 128 if num_tokens <= 64 else 16
        else:
            split_k = 32 if num_tokens <= 64 else 8
    if split_k <= 0 or hidden_size % split_k != 0:
        raise ValueError(
            f"HC HEAD split_k must divide hidden_size, "
            f"got hidden_size={hidden_size}, split_k={split_k}"
        )
    return split_k


def _get_hidden_block(hidden_size: int, split_k: int, config: HcHeadTileLangConfig) -> int:
    hidden_block = config.hidden_block
    split_size = hidden_size // split_k
    if hidden_block <= 0:
        hidden_block = min(128, split_size)
    if hidden_block <= 0 or split_size % hidden_block != 0:
        raise ValueError(
            f"HC HEAD hidden_block must divide split_size, "
            f"got split_size={split_size}, hidden_block={hidden_block}"
        )
    return hidden_block


def _get_token_block(config: HcHeadTileLangConfig) -> int:
    token_block = config.token_block
    if token_block not in (8, 16, 32):
        raise ValueError(
            "HC HEAD token_block must be one of 8, 16, or 32, "
            f"got {token_block}"
        )
    return token_block


def hc_head_linear_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: HcHeadTileLangConfig | None = None,
) -> torch.Tensor:
    """Compute the `hc_head` linear projection with TileLang split-K."""
    config = config or HcHeadTileLangConfig()
    if not _has_musa_tensor(x, weight):
        raise RuntimeError("hc_head_linear_tilelang only supports MUSA tensors")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError(
            "hc_head_linear_tilelang expects float32 x and weight, "
            f"got x={x.dtype}, weight={weight.dtype}"
        )
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(
            f"hc_head_linear_tilelang expects 2D tensors, got {x.shape=} {weight.shape=}"
        )
    num_tokens, hidden_size = x.shape
    hc_mult, weight_hidden = weight.shape
    if hidden_size != weight_hidden:
        raise ValueError(
            f"hc_head_linear_tilelang shape mismatch: x={tuple(x.shape)}, "
            f"weight={tuple(weight.shape)}"
        )
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError(
            "hc_head_linear_tilelang requires contiguous x and weight, "
            f"got x_stride={tuple(x.stride())}, weight_stride={tuple(weight.stride())}"
        )
    if hc_mult > 32:
        raise ValueError(f"hc_head_linear_tilelang supports hc_mult <= 32, got {hc_mult}")
    if num_tokens == 0:
        return torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=x.device)

    split_k = _get_split_k(hidden_size, num_tokens, config)
    token_block = _get_token_block(config)
    hidden_block = _get_hidden_block(hidden_size, split_k, config)
    partial = torch.empty(
        split_k, num_tokens, hc_mult, dtype=torch.float32, device=x.device
    )
    out = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=x.device)
    impl = config.impl.strip().lower()
    if impl in ("", "auto", "warp") and hc_mult == 4 and hidden_size // split_k % 32 == 0:
        selected_impl = "warp"
        stage0, stage1 = _tilelang_hc_head_linear_splitk_warp_kernel(
            hidden_size,
            hc_mult,
            split_k,
        )
    elif impl in ("shared", "scalar"):
        selected_impl = "shared"
        stage0, stage1 = _tilelang_hc_head_linear_splitk_kernel(
            hidden_size,
            hc_mult,
            split_k,
            token_block=token_block,
            hidden_block=hidden_block,
        )
    else:
        raise ValueError(
            "HC HEAD TileLang impl must be one of auto, warp, shared, "
            f"or scalar, got {impl!r}"
        )
    if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_HC_HEAD_TRACE_DISPATCH") == "1":
        logger.info(
            "HC_HEAD_DISPATCH status=hit impl=%s shape=%s weight_shape=%s "
            "split_k=%d token_block=%d hidden_block=%d",
            selected_impl,
            tuple(x.shape),
            tuple(weight.shape),
            split_k,
            token_block,
            hidden_block,
        )
    stage0(x, weight, partial)
    stage1(partial, out)
    return out


def hc_head_tilelang(
    residual: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    config: HcHeadTileLangConfig | None = None,
) -> torch.Tensor:
    """Compute the full DeepSeek V4 `hc_head` with TileLang split-K."""
    config = config or HcHeadTileLangConfig()
    if not _has_musa_tensor(residual, weight, hc_scale, hc_base):
        raise RuntimeError("hc_head_tilelang only supports MUSA tensors")
    if residual.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise TypeError(
            "hc_head_tilelang expects bfloat16 residual and float32 weight, "
            f"got residual={residual.dtype}, weight={weight.dtype}"
        )
    if hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise TypeError(
            "hc_head_tilelang expects float32 hc_scale and hc_base, "
            f"got hc_scale={hc_scale.dtype}, hc_base={hc_base.dtype}"
        )
    if residual.dim() != 3 or weight.dim() != 2:
        raise ValueError(
            "hc_head_tilelang expects residual [tokens,hc_mult,hidden] and "
            f"weight [hc_mult,hc_mult*hidden], got {residual.shape=} {weight.shape=}"
        )
    num_tokens, hc_mult, hidden_size = residual.shape
    weight_hc_mult, flat_hidden = weight.shape
    if num_tokens == 0:
        return torch.empty(num_tokens, hidden_size, dtype=residual.dtype, device=residual.device)
    if weight_hc_mult != hc_mult or flat_hidden != hc_mult * hidden_size:
        raise ValueError(
            "hc_head_tilelang shape mismatch: "
            f"residual={tuple(residual.shape)}, weight={tuple(weight.shape)}"
        )
    if hc_mult != 4:
        raise ValueError(f"hc_head_tilelang currently supports hc_mult=4, got {hc_mult}")
    if not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError(
            "hc_head_tilelang requires contiguous residual and weight, "
            f"got residual_stride={tuple(residual.stride())}, "
            f"weight_stride={tuple(weight.stride())}"
        )

    split_k = _get_split_k(flat_hidden, num_tokens, config)
    partial = torch.empty(
        split_k, num_tokens, hc_mult + 1, dtype=torch.float32, device=residual.device
    )
    out = torch.empty(num_tokens, hidden_size, dtype=residual.dtype, device=residual.device)
    stage0, stage1 = _tilelang_hc_head_fused_splitk_warp_kernel(
        hidden_size,
        hc_mult,
        split_k,
        norm_eps,
        hc_eps,
    )
    if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_HC_HEAD_TRACE_DISPATCH") == "1":
        logger.info(
            "HC_HEAD_DISPATCH status=hit impl=fused_warp shape=%s weight_shape=%s "
            "split_k=%d",
            tuple(residual.shape),
            tuple(weight.shape),
            split_k,
        )
    stage0(residual, weight, partial)
    stage1(residual, partial, hc_scale, hc_base, out)
    return out


def try_hc_head_tilelang(
    residual: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
) -> torch.Tensor | None:
    if not envs.SGLANG_OPT_HC_HEAD_TILELANG.get():
        return None
    if not _has_musa_tensor(residual, weight, hc_scale, hc_base):
        return None
    try:
        return hc_head_tilelang(residual, weight, hc_scale, hc_base, norm_eps, hc_eps)
    except Exception:
        if envs.SGLANG_OPT_HC_HEAD_TILELANG_STRICT.get():
            raise
        return None


def try_hc_head_linear_tilelang(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    if not envs.SGLANG_OPT_HC_HEAD_TILELANG.get():
        return None
    if not _has_musa_tensor(x, weight):
        return None
    try:
        return hc_head_linear_tilelang(x, weight)
    except Exception:
        if envs.SGLANG_OPT_HC_HEAD_TILELANG_STRICT.get():
            raise
        return None
