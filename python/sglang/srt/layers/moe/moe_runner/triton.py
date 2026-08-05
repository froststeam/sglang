from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_musa

_is_musa = is_musa()
_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS = int(
    os.getenv("SGLANG_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS", "32")
)
_MUSA_MOE_GEMV_BLOCK_KS: tuple[int, ...] = (4, 8, 16, 32)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


def _run_musa_moe_gemv_swiglu(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    if not _can_run_musa_moe_gemv_swiglu(hidden_states, quant_info, runner_config):
        raise RuntimeError(
            "MUSA MoE GEMV SwiGLU only supports local BF16 MoE or FP8 block-"
            "quantized MoE with BF16 activations and FP8 weight scales."
        )

    from sglang.srt.hardware_backend.musa.jit_kernel import moe_sum_reduce
    from sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemv import (
        get_moe_gemv_config,
        musa_moe_down_reduce,
        musa_moe_gemv,
    )

    topk = topk_ids.shape[1]
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_ids = topk_ids.contiguous()
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)
    topk_weights = topk_weights.contiguous()

    num_tokens = hidden_states.shape[0]
    w13 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    inter_size = w13.shape[1] // 2
    use_fp8 = quant_info.use_fp8_w8a8
    shared_experts = runner_config.num_fused_shared_experts or 0
    gemv_config = get_moe_gemv_config(
        dtype="fp8" if use_fp8 else "bf16",
        tokens=num_tokens,
        hidden=hidden_states.shape[1],
        intermediate=inter_size,
        experts=w13.shape[0] - shared_experts,
        routed_topk=topk - shared_experts,
        shared_experts=shared_experts,
    )
    # The table is optional and exact-shape keyed.  A missing entry returns
    # (-1, -1, -1), preserving the in-kernel dispatch and the regular
    # down-GEMV + sum-reduce path.

    intermediate = torch.empty(
        (num_tokens, topk, inter_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    musa_moe_gemv(
        hidden_states,
        w13,
        intermediate,
        None,
        quant_info.w13_scale if use_fp8 else None,
        topk_weights,
        topk_ids,
        False,
        topk,
        False,
        True,
        gemv_config.gate_config_id,
    )

    routed_scaling_factor = runner_config.routed_scaling_factor
    routed_scaling_factor = (
        1.0 if routed_scaling_factor is None else routed_scaling_factor
    )
    out_hidden_states = (
        hidden_states if runner_config.inplace else torch.empty_like(hidden_states)
    )
    if gemv_config.down_reduce_config_id >= 0:
        musa_moe_down_reduce(
            intermediate,
            w2,
            out_hidden_states,
            quant_info.w2_scale if use_fp8 else None,
            topk_weights,
            topk_ids,
            topk,
            routed_scaling_factor,
            gemv_config.down_reduce_config_id,
        )
        return out_hidden_states

    routed_output = torch.empty(
        (num_tokens, topk, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    flat_tokens = num_tokens * topk
    musa_moe_gemv(
        intermediate.view(flat_tokens, inter_size),
        w2,
        routed_output.view(flat_tokens, 1, w2.shape[1]),
        None,
        quant_info.w2_scale if use_fp8 else None,
        topk_weights.view(flat_tokens, 1),
        topk_ids.view(flat_tokens, 1),
        True,
        1,
        False,
        False,
        gemv_config.down_config_id,
    )

    if topk == 1 and routed_scaling_factor == 1.0:
        out_hidden_states.copy_(routed_output[:, 0])
    else:
        moe_sum_reduce(routed_output, out_hidden_states, routed_scaling_factor)
    return out_hidden_states


def _can_run_musa_moe_gemv_swiglu_quant_info(
    hidden_states: torch.Tensor,
    quant_info: TritonMoeQuantInfo,
) -> bool:
    if hidden_states.dtype != torch.bfloat16:
        return False

    if (
        quant_info.w13_weight.dtype == torch.bfloat16
        and quant_info.w2_weight.dtype == torch.bfloat16
    ):
        return (
            quant_info.w13_scale is None
            and quant_info.w2_scale is None
            and not quant_info.use_fp8_w8a8
            and not quant_info.use_int8_w8a8
            and not quant_info.use_int8_w8a16
            and not quant_info.use_int4_w4a16
        )

    if not (
        quant_info.use_fp8_w8a8
        and quant_info.w13_weight.dtype == torch.float8_e4m3fn
        and quant_info.w2_weight.dtype == torch.float8_e4m3fn
        and quant_info.w13_scale is not None
        and quant_info.w2_scale is not None
        and quant_info.w13_scale.is_contiguous()
        and quant_info.w2_scale.is_contiguous()
        and quant_info.block_shape is not None
        and len(quant_info.block_shape) == 2
        and tuple(quant_info.block_shape) == (128, 128)
    ):
        return False

    return (
        not quant_info.use_int8_w8a8
        and not quant_info.use_int8_w8a16
        and not quant_info.use_int4_w4a16
    )


def _can_run_musa_moe_gemv_swiglu(
    hidden_states: torch.Tensor,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> bool:
    if _is_musa and envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
        return False
    if runner_config.musa_moe_gemv_enabled is False:
        return False

    hidden_size = hidden_states.shape[1]
    element_size = quant_info.w13_weight.element_size()
    if element_size <= 0:
        return False
    vlen = 128 // (element_size * 8)
    if vlen <= 0:
        return False
    if not any(
        hidden_size % (block_k * vlen) == 0 for block_k in _MUSA_MOE_GEMV_BLOCK_KS
    ):
        return False

    return (
        _is_musa
        and (
            runner_config.musa_moe_gemv_enabled is True
            or (
                _MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS > 0
                and hidden_states.shape[0] <= _MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS
            )
        )
        and hidden_states.shape[0] > 0
        and hidden_states.is_contiguous()
        and runner_config.num_local_experts == quant_info.w13_weight.shape[0]
        and runner_config.num_local_experts == quant_info.w2_weight.shape[0]
        and runner_config.activation == "silu"
        and runner_config.swiglu_limit is None
        and runner_config.gemm1_alpha is None
        and runner_config.gemm1_clamp_limit is None
        and not runner_config.no_combine
        and not runner_config.apply_router_weight_on_input
        and runner_config.is_gated
        and quant_info.b13 is None
        and quant_info.b2 is None
        and quant_info.w13_weight.is_contiguous()
        and quant_info.w2_weight.is_contiguous()
        and quant_info.w13_zp is None
        and quant_info.w2_zp is None
        and not quant_info.per_channel_quant
        and _can_run_musa_moe_gemv_swiglu_quant_info(hidden_states, quant_info)
    )


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonRunnerOutput:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )

        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )

        if hooks is None and _can_run_musa_moe_gemv_swiglu(
            runner_input.hidden_states,
            quant_info,
            self.config,
        ):
            return TritonRunnerOutput(
                hidden_states=_run_musa_moe_gemv_swiglu(
                    runner_input.hidden_states,
                    runner_input.topk_weights,
                    runner_input.topk_ids,
                    quant_info,
                    self.config,
                )
            )

        out = _fused_moe_kernel_sequence(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            runner_input.sorted_token_ids,
            runner_input.expert_ids,
            runner_input.num_tokens_post_padded,
            running_state["config"],
            running_state.get("down_config"),
            running_state.get("down_moe_use_tma", False),
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            activation=self.config.activation,
            is_gated=self.config.is_gated,
            no_combine=self.config.no_combine,
            inplace=self.config.inplace,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            routed_scaling_factor=self.config.routed_scaling_factor,
            gemm1_alpha=self.config.gemm1_alpha,
            gemm1_limit=self.config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            hooks=hooks,
            swiglu_limit=self.config.swiglu_limit,
        )

        return TritonRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if _can_run_musa_moe_gemv_swiglu(
        dispatch_output.hidden_states, quant_info, runner_config
    ):
        output = _run_musa_moe_gemv_swiglu(
            dispatch_output.hidden_states,
            dispatch_output.topk_output.topk_weights,
            dispatch_output.topk_output.topk_ids,
            quant_info,
            runner_config,
        )
        return StandardCombineInput(hidden_states=output)

    output = fused_experts(
        hidden_states=dispatch_output.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=runner_config,
        b1=quant_info.b13,
        b2=quant_info.b2,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        w1_zp=quant_info.w13_zp,
        w2_zp=quant_info.w2_zp,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        block_shape=quant_info.block_shape,
    )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # NOTE: this is dead code as a fused func for standard format is registered.
    # This is left here for testing and examples.

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # NOTE: this is dead code as a fused func for standard format is registered.
    # This is left here for testing and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )
