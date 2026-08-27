from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _act_and_mul_module():
    return load_musa_jit(
        "sglang_musa_act_and_mul",
        ("activation/act_and_mul.mu",),
    )


@cache_once
def _act_and_mul_masked_module():
    return load_musa_jit(
        "sglang_musa_act_and_mul_masked",
        ("activation/act_and_mul_masked.mu",),
    )


@cache_once
def _sigmoid_mul_module():
    return load_musa_jit(
        "sglang_musa_sigmoid_mul",
        ("activation/sigmoid_mul.mu",),
    )


def _sigmoid_mul_impl(
    gate: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
) -> None:
    _sigmoid_mul_module().sgl_musa_sigmoid_mul(gate, value, output)


@register_custom_op(
    op_name="musa_sigmoid_mul",
    mutates_args=["output"],
)
def _sigmoid_mul_custom(
    gate: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
) -> None:
    _sigmoid_mul_impl(gate, value, output)


def sigmoid_mul(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Compute ``sigmoid(gate) * value`` in one MUSA JIT kernel."""
    if (
        gate.device.type != "musa"
        or value.device != gate.device
        or gate.shape != value.shape
        or gate.dtype != value.dtype
        or gate.dtype not in (torch.float16, torch.bfloat16)
        or not gate.is_contiguous()
        or not value.is_contiguous()
    ):
        return torch.sigmoid(gate) * value
    output = torch.empty_like(value)
    _sigmoid_mul_custom(gate, value, output)
    return output


def _activation_type_id(activation: str) -> int:
    if activation == "silu":
        return 0
    if activation == "gelu":
        return 1
    if activation == "gelu_tanh":
        return 2
    raise ValueError(f"Unsupported activation: {activation}")


def _act_and_mul_impl(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_step: int,
    activation_type: int,
    skip_expert_check: int,
) -> None:
    _act_and_mul_module().sgl_musa_act_and_mul(
        gateup_output,
        down_input,
        expert_ids,
        int(expert_step),
        int(activation_type),
        int(skip_expert_check),
    )


def _act_and_mul_masked_impl(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    masked_m: torch.Tensor,
    activation_type: int,
) -> None:
    _act_and_mul_masked_module().sgl_musa_act_and_mul_masked(
        gateup_output,
        down_input,
        masked_m,
        int(activation_type),
    )


@register_custom_op(
    op_name="musa_act_and_mul",
    mutates_args=["down_input"],
)
def _act_and_mul_custom(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_step: int,
    activation_type: int,
    skip_expert_check: int,
) -> None:
    _act_and_mul_impl(
        gateup_output,
        down_input,
        expert_ids,
        expert_step,
        activation_type,
        skip_expert_check,
    )


@register_custom_op(
    op_name="musa_act_and_mul_masked",
    mutates_args=["down_input"],
)
def _act_and_mul_masked_custom(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    masked_m: torch.Tensor,
    activation_type: int,
) -> None:
    _act_and_mul_masked_impl(
        gateup_output,
        down_input,
        masked_m,
        activation_type,
    )


def act_and_mul(
    gateup_output: torch.Tensor,
    down_input: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    topk_ids: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    down_moe_use_tma: bool = False,
    activation: str = "silu",
    filter_expert: bool = False,
) -> Optional[torch.Tensor]:
    """MUSA JIT gated activation-and-multiply op."""
    return_output = down_input is None
    if down_input is None:
        down_input = torch.empty(
            (gateup_output.shape[0], gateup_output.shape[1] // 2),
            dtype=gateup_output.dtype,
            device=gateup_output.device,
        )

    if down_moe_use_tma:
        if expert_ids is None:
            raise ValueError("expert_ids is required when down_moe_use_tma=True")
        selected_expert_ids = expert_ids
        expert_step = int(config["BLOCK_SIZE_M"]) if config is not None else 1
        skip_expert_check = 0
    else:
        selected_expert_ids = (
            topk_ids.view(-1)
            if topk_ids is not None
            else torch.empty((0,), dtype=torch.int32, device=gateup_output.device)
        )
        expert_step = 1
        skip_expert_check = 0 if filter_expert and topk_ids is not None else 1

    _act_and_mul_custom(
        gateup_output,
        down_input,
        selected_expert_ids,
        expert_step,
        _activation_type_id(activation),
        skip_expert_check,
    )
    if return_output:
        return down_input
    return None


def act_and_mul_masked(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    masked_m: torch.Tensor,
    activation: str = "silu",
) -> None:
    """MUSA JIT masked gated activation-and-multiply op for [E, M, 2N]."""
    _act_and_mul_masked_custom(
        gateup_output,
        down_input,
        masked_m,
        _activation_type_id(activation),
    )


@cache_once
def _act_and_mul_post_quant_module():
    return load_musa_jit(
        "sglang_musa_act_and_mul_post_quant",
        ("activation/act_and_mul_post_quant.mu",),
    )


def _act_and_mul_masked_post_quant_impl(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    activation_type: int,
    swiglu_limit: float,
    swizzle: bool,
    expected_m: int,
) -> None:
    _act_and_mul_post_quant_module().sgl_musa_act_and_mul_masked_post_quant(
        input,
        output,
        output_scale,
        masked_m,
        int(activation_type),
        float(swiglu_limit),
        bool(swizzle),
        int(expected_m),
    )


@register_custom_op(
    op_name="musa_act_and_mul_masked_post_quant",
    mutates_args=["output", "output_scale"],
)
def _act_and_mul_masked_post_quant_custom(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    activation_type: int,
    swiglu_limit: float,
    swizzle: bool,
    expected_m: int,
) -> None:
    _act_and_mul_masked_post_quant_impl(
        input,
        output,
        output_scale,
        masked_m,
        activation_type,
        swiglu_limit,
        swizzle,
        expected_m,
    )


def act_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
    expected_m: int = 0,
) -> None:
    _act_and_mul_masked_post_quant_custom(
        input,
        output,
        output_scale,
        masked_m,
        _activation_type_id(activation),
        0.0 if swiglu_limit is None else float(swiglu_limit),
        swizzle,
        expected_m,
    )


def act_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
    expected_m: int = 0,
) -> None:
    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0
    assert activation in ("silu", "gelu", "gelu_tanh")
    if swiglu_limit is not None and activation != "silu":
        raise ValueError("swiglu_limit is only supported with silu activation.")

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    if input.dtype != torch.bfloat16 or quant_group_size != 128 or scale_ue8m0:
        raise ValueError(
            "MUSA act_and_mul masked post-quant requires bf16 input, "
            "quant_group_size=128, and scale_ue8m0=False."
        )

    act_and_mul_masked_post_quant(
        input,
        output,
        output_scale,
        masked_m,
        activation=activation,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
        expected_m=expected_m,
    )
