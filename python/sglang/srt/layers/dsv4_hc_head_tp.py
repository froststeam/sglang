"""Tensor-parallel sharding for DeepSeek V4 hc_head (row-parallel on hc_dim)."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.utils.common import log_info_on_rank0

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


def is_dsv4_hc_head_tp_enabled() -> bool:
    return envs.SGLANG_DSV4_HC_HEAD_TP.get()


def hc_head_tp_local_hc_dim(hc_dim: int, tp_size: int, enabled: bool) -> int:
    if enabled and tp_size > 1:
        assert hc_dim % tp_size == 0, (
            f"hc_dim={hc_dim} (hc_mult * hidden_size) must be divisible by "
            f"tp_size={tp_size} when SGLANG_DSV4_HC_HEAD_TP=1"
        )
        return hc_dim // tp_size
    return hc_dim


def _hc_head_fn_tp_weight_loader(
    param: Parameter,
    loaded_weight: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> None:
    shard_cols = param.data.shape[1]
    start = tp_rank * shard_cols
    param.data.copy_(loaded_weight.narrow(1, start, shard_cols))


def setup_dsv4_hc_head_params(
    module: nn.Module,
    config: PretrainedConfig,
) -> None:
    """Create hc_head_* parameters; shard hc_head_fn on hc_dim by TP when env is on."""
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    hc_mult = config.hc_mult
    hc_dim = hc_mult * config.hidden_size

    tp_enabled = is_dsv4_hc_head_tp_enabled() and tp_size > 1
    local_hc_dim = hc_head_tp_local_hc_dim(hc_dim, tp_size, tp_enabled)

    module.hc_head_tp_enabled = tp_enabled
    module.hc_head_tp_rank = tp_rank
    module.hc_head_tp_size = tp_size
    module.hc_head_tp_hc_dim = hc_dim
    module.hc_head_tp_local_hc_dim = local_hc_dim

    module.hc_head_fn = nn.Parameter(
        torch.empty(hc_mult, local_hc_dim, dtype=torch.float32)
    )
    # base/scale stay full on every rank (small; mixes are all_reduced).
    module.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
    module.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    if tp_enabled:
        module.hc_head_fn.weight_loader = partial(
            _hc_head_fn_tp_weight_loader, tp_rank=tp_rank, tp_size=tp_size
        )
        log_info_on_rank0(
            logger,
            f"DSV4 hc_head row-TP enabled: hc_mult={hc_mult} hc_dim={hc_dim} "
            f"tp_size={tp_size} local_hc_dim={local_hc_dim}",
        )


def hc_head_forward(
    module: nn.Module,
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    hc_eps: float,
    use_fused: bool = True,
    use_tilelang: bool = False,
) -> torch.Tensor:
    """hc_head forward with optional row-TP on the flattened hc_dim axis."""
    if getattr(module, "hc_head_tp_enabled", False):
        return _hc_head_forward_tp(
            module, x, hc_fn, hc_scale, hc_base, norm_eps=norm_eps, hc_eps=hc_eps
        )

    dtype = x.dtype
    if use_tilelang and envs.SGLANG_OPT_HC_HEAD_TILELANG.get():
        from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops import (
            try_hc_head_tilelang,
        )

        y = try_hc_head_tilelang(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
        if y is not None:
            return y.to(dtype)

    if use_fused and x.numel() > 0:
        from sglang.srt.layers.mhc_head import fused_hc_head

        return fused_hc_head(
            x.contiguous(),
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )

    return _hc_head_forward_torch(
        x, hc_fn, hc_scale, hc_base, norm_eps=norm_eps, hc_eps=hc_eps
    )


def _hc_head_forward_tp(
    module: nn.Module,
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = x.size(), x.dtype
    if x.numel() == 0:
        return x.squeeze(1).to(dtype) if x.dim() > 2 else x.to(dtype)

    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)

    start = module.hc_head_tp_rank * module.hc_head_tp_local_hc_dim
    end = start + module.hc_head_tp_local_hc_dim
    mixes = F.linear(x_flat[:, start:end], hc_fn) * rsqrt
    if module.hc_head_tp_size > 1:
        mixes = tensor_model_parallel_all_reduce(mixes)

    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def _hc_head_forward_torch(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)
