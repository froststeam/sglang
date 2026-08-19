from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)
_SUPPORTED_HIDDEN_SIZE = 3072
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@cache_once
def _mul_add_module():
    return load_musa_jit(
        "sglang_musa_diffusion_mul_add",
        ("diffusion/mul_add.mu",),
    )


@cache_once
def _layernorm_scale_shift_module():
    return load_musa_jit(
        "sglang_musa_diffusion_layernorm_scale_shift",
        ("diffusion/layernorm_scale_shift.mu",),
    )


@cache_once
def _qknorm_rope_module():
    return load_musa_jit(
        "sglang_musa_diffusion_qknorm_rope",
        ("diffusion/qknorm_rope.mu",),
    )


def _can_use(
    hidden_size: int,
    dtype: torch.dtype,
    module_loader: Callable,
    name: str,
) -> bool:
    if hidden_size != _SUPPORTED_HIDDEN_SIZE or dtype not in _SUPPORTED_DTYPES:
        return False
    try:
        module_loader()
        return True
    except Exception as exc:
        logger.warning("Failed to load MUSA fused %s: %s", name, exc)
        return False


@torch.compiler.assume_constant_result
@cache_once
def can_use_musa_mul_add(hidden_size: int, dtype: torch.dtype) -> bool:
    return _can_use(hidden_size, dtype, _mul_add_module, "MulAdd")


@torch.compiler.assume_constant_result
@cache_once
def can_use_musa_layernorm_scale_shift(hidden_size: int, dtype: torch.dtype) -> bool:
    return _can_use(
        hidden_size,
        dtype,
        _layernorm_scale_shift_module,
        "LayerNorm+ScaleShift",
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_musa_qknorm_rope(dtype: torch.dtype) -> bool:
    if dtype not in _SUPPORTED_DTYPES:
        return False
    try:
        _qknorm_rope_module()
        return True
    except Exception as exc:
        logger.warning("Failed to load MUSA fused QKNorm+RoPE: %s", exc)
        return False


@register_custom_op(mutates_args=["q", "k"])
def musa_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> None:
    _qknorm_rope_module().sgl_musa_diffusion_qknorm_rope(
        q, k, q_weight, k_weight, cos_sin_cache, positions, eps
    )


@register_custom_op(mutates_args=[], out_shape="a")
def musa_mul_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    if a.ndim != 3 or b.ndim != 3 or c.ndim != 3:
        raise ValueError("expected a/b/c to be 3D")
    if a.shape != c.shape:
        raise ValueError("a and c must have the same shape")
    if b.shape != (a.shape[0], 1, a.shape[-1]):
        raise ValueError("b must have shape [B, 1, D]")

    a_2d = a.reshape(-1, a.shape[-1])
    b_2d = b.reshape(a.shape[0], a.shape[-1]).contiguous()
    c_2d = c.reshape_as(a_2d)
    output = torch.empty_like(a_2d)
    _mul_add_module().sgl_musa_diffusion_mul_add(a_2d, b_2d, c_2d, output)
    return output.view_as(a)


def _flatten_modulation_inputs(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError("expected x to be 3D")
    valid_shapes = ((x.shape[0], x.shape[-1]), (x.shape[0], 1, x.shape[-1]))
    if scale.shape != shift.shape or scale.shape not in valid_shapes:
        raise ValueError("scale/shift must have shape [B, D] or [B, 1, D]")
    return (
        x.reshape(-1, x.shape[-1]),
        scale.reshape(x.shape[0], x.shape[-1]).contiguous(),
        shift.reshape(x.shape[0], x.shape[-1]).contiguous(),
    )


def _fake_layernorm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(mutates_args=[], fake_impl=_fake_layernorm_scale_shift)
def musa_layernorm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_2d, scale_2d, shift_2d = _flatten_modulation_inputs(x, scale, shift)
    output = torch.empty_like(x_2d)
    _layernorm_scale_shift_module().sgl_musa_diffusion_layernorm_scale_shift(
        x_2d, scale_2d, shift_2d, output, eps
    )
    return output.view_as(x)


def _fake_scale_residual_layernorm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)


@register_custom_op(
    mutates_args=[], fake_impl=_fake_scale_residual_layernorm_scale_shift
)
def musa_scale_residual_layernorm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.shape != x.shape:
        raise ValueError("residual and x must have the same shape")
    if gate.shape != (x.shape[0], 1, x.shape[-1]):
        raise ValueError("gate must have shape [B, 1, D]")

    x_2d, scale_2d, shift_2d = _flatten_modulation_inputs(x, scale, shift)
    residual_2d = residual.reshape_as(x_2d)
    gate_2d = gate.reshape(x.shape[0], x.shape[-1]).contiguous()
    output = torch.empty_like(x_2d)
    residual_output = torch.empty_like(x_2d)
    module = _layernorm_scale_shift_module()
    module.sgl_musa_diffusion_scale_residual_layernorm_scale_shift(
        x_2d,
        residual_2d,
        gate_2d,
        scale_2d,
        shift_2d,
        output,
        residual_output,
        eps,
    )
    return output.view_as(x), residual_output.view_as(x)


def _flatten_select01_inputs(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    x_2d, scale0_2d, shift0_2d = _flatten_modulation_inputs(x, scale0, shift0)
    expected_shape = scale0.shape
    if any(tensor.shape != expected_shape for tensor in (gate0, scale1, shift1, gate1)):
        raise ValueError("all Select01 modulation tensors must have the same shape")
    if index.shape != x.shape[:2] or index.dtype != torch.int32:
        raise ValueError("index must have shape [B, L] and dtype int32")
    modulation = (
        scale0_2d,
        shift0_2d,
        gate0.reshape(x.shape[0], x.shape[-1]).contiguous(),
        scale1.reshape(x.shape[0], x.shape[-1]).contiguous(),
        shift1.reshape(x.shape[0], x.shape[-1]).contiguous(),
        gate1.reshape(x.shape[0], x.shape[-1]).contiguous(),
    )
    return x_2d, modulation, index.reshape(-1).contiguous()


def _fake_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


@register_custom_op(
    mutates_args=[], fake_impl=_fake_layernorm_scale_shift_gate_select01
)
def musa_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_2d, modulation, index_1d = _flatten_select01_inputs(
        x, scale0, shift0, gate0, scale1, shift1, gate1, index
    )
    output = torch.empty_like(x_2d)
    gate_output = torch.empty_like(x_2d)
    _layernorm_scale_shift_module().sgl_musa_diffusion_layernorm_scale_shift_gate_select01(
        x_2d, *modulation, index_1d, output, gate_output, eps
    )
    return output.view_as(x), gate_output.view_as(x)


def _fake_residual_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)


@register_custom_op(
    mutates_args=[], fake_impl=_fake_residual_layernorm_scale_shift_gate_select01
)
def musa_residual_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual.shape != x.shape or residual_gate.shape != x.shape:
        raise ValueError("residual and residual_gate must have the same shape as x")
    x_2d, modulation, index_1d = _flatten_select01_inputs(
        x, scale0, shift0, gate0, scale1, shift1, gate1, index
    )
    residual_2d = residual.reshape_as(x_2d).contiguous()
    residual_gate_2d = residual_gate.reshape_as(x_2d).contiguous()
    output = torch.empty_like(x_2d)
    residual_output = torch.empty_like(x_2d)
    gate_output = torch.empty_like(x_2d)
    module = _layernorm_scale_shift_module()
    module.sgl_musa_diffusion_residual_layernorm_scale_shift_gate_select01(
        x_2d,
        residual_2d,
        residual_gate_2d,
        *modulation,
        index_1d,
        output,
        residual_output,
        gate_output,
        eps,
    )
    return (
        output.view_as(x),
        residual_output.view_as(x),
        gate_output.view_as(x),
    )
