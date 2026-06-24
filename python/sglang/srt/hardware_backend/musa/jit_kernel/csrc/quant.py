from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op

_SILU_ACTIVATION_TYPE = 0


@cache_once
def _quant_musa_cflags() -> tuple[str, ...]:
    return (
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-mllvm",
        "-mtgpu-opt-level=1",
        "-mllvm",
        "-mtgpu-load-store-opt=1",
        "-mllvm",
        "-mtgpu-fold-global-ldst=1",
    )


@cache_once
def _quant_v2_module():
    return load_musa_jit(
        "sglang_musa_quant_v2",
        ("quant/per_token_group_quant_8bit_v2.mu",),
        extra_musa_cflags=_quant_musa_cflags(),
    )


def per_token_group_quant_8bit_v2(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: torch.Tensor | None = None,
) -> None:
    if masked_m is None:
        masked_m = torch.empty((1,), device=input.device, dtype=torch.int32)
        has_masked_m = False
    else:
        has_masked_m = True
    _quant_v2_module().sgl_per_token_group_quant_8bit_v2(
        input,
        output_q,
        output_s,
        int(group_size),
        float(eps),
        float(min_8bit),
        float(max_8bit),
        bool(scale_ue8m0),
        bool(fuse_silu_and_mul),
        _SILU_ACTIVATION_TYPE,
        masked_m,
        bool(has_masked_m),
    )


@register_custom_op(
    op_name="musa_per_token_group_quant_8bit",
    mutates_args=["output_q", "output_s"],
)
def _per_token_group_quant_8bit_custom(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: torch.Tensor | None = None,
    enable_v2: bool | None = None,
) -> None:
    if enable_v2 is False:
        raise ValueError("MUSA csrc quant only supports the v2 kernel path.")
    per_token_group_quant_8bit_v2(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        min_8bit,
        max_8bit,
        scale_ue8m0,
        fuse_silu_and_mul,
        masked_m,
    )


def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: torch.Tensor | None = None,
    enable_v2: bool | None = None,
) -> None:
    _per_token_group_quant_8bit_custom(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        min_8bit,
        max_8bit,
        scale_ue8m0,
        fuse_silu_and_mul,
        masked_m,
        enable_v2,
    )
