from typing import Optional

import tilelang
import tilelang.language as T
import torch

from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
)

fp8_dtype = torch.float8_e4m3fn
_SUPPORTED_GROUP_SIZES = {16, 32, 64, 128}
_INPUT_DTYPES = {torch.float16, torch.bfloat16}
_OUTPUT_DTYPES = {torch.int8, fp8_dtype}
_LOG2E = 1.4426950408889634
_QUANT_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
if hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _QUANT_PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _QUANT_PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = False

def _flat_storage_view(tensor: torch.Tensor) -> torch.Tensor:
    return torch.as_strided(
        tensor,
        (tensor.untyped_storage().nbytes() // tensor.element_size(),),
        (1,),
        storage_offset=0,
    )


def _scale_storage_view(output_s: torch.Tensor, scale_ue8m0: bool) -> torch.Tensor:
    flat = _flat_storage_view(output_s)
    if scale_ue8m0:
        return flat.view(torch.uint8)
    return flat


def _check_limits(output_dtype: torch.dtype, min_8bit: float, max_8bit: float) -> None:
    info = (
        torch.iinfo(output_dtype)
        if output_dtype == torch.int8
        else torch.finfo(output_dtype)
    )
    assert min_8bit == info.min, f"min_8bit must be {info.min} for {output_dtype}"
    assert max_8bit == info.max, f"max_8bit must be {info.max} for {output_dtype}"


def _flatten_for_kernel(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(-1)


def _scale_storage_metadata(output_s: torch.Tensor, scale_ue8m0: bool):
    scale_storage = _scale_storage_view(output_s, scale_ue8m0)
    scale_element_size = 1 if scale_ue8m0 else output_s.element_size()
    scale_storage_offset = (
        output_s.storage_offset() * output_s.element_size() // scale_element_size
    )
    return (
        scale_storage,
        scale_storage_offset,
        output_s.stride(-2),
        output_s.stride(-1),
    )


def _dummy_mask(device: torch.device) -> torch.Tensor:
    return torch.empty((1,), device=device, dtype=torch.int32)


@tilelang.jit()
def _per_token_group_quant_8bit_kernel(
    group_size,
    input_dtype,
    output_dtype,
    scale_dtype,
    fuse_silu_and_mul,
    masked_layout,
    scale_ue8m0,
):
    input_numel = T.dynamic("input_numel")
    output_numel = T.dynamic("output_numel")
    scale_storage_numel = T.dynamic("scale_storage_numel")
    total_groups = T.dynamic("total_groups")
    num_experts = T.dynamic("num_experts")
    num_threads = max(group_size, 32)

    @T.prim_func
    def sglang_musa_per_token_group_quant_8bit_kernel(
        input: T.Tensor([input_numel], dtype=input_dtype),
        output_q: T.Tensor([output_numel], dtype=output_dtype),
        output_s: T.Tensor([scale_storage_numel], dtype=scale_dtype),
        masked_m: T.Tensor([num_experts], dtype="int32"),
        total_groups: T.int32,
        num_tokens_per_expert: T.int32,
        hidden_dim_num_groups: T.int32,
        scale_storage_offset: T.int32,
        scale_expert_stride: T.int32,
        scale_token_stride: T.int32,
        scale_hidden_stride: T.int32,
        eps: T.float32,
        min_8bit: T.float32,
        max_8bit: T.float32,
    ):
        with T.Kernel(total_groups, threads=num_threads) as (pid,):
            expert_idx = T.alloc_var("int32")
            token_idx = T.alloc_var("int32")
            hidden_group_idx = T.alloc_var("int32")
            valid_group = T.alloc_var("bool")
            group_base = T.alloc_var("int32")
            input_base = T.alloc_var("int32")
            secondary_base = T.alloc_var("int32")
            scale_index = T.alloc_var("int32")
            absmax = T.alloc_var("float32")
            scale = T.alloc_var("float32")
            scale_inv = T.alloc_var("float32")
            exp_scale_inv = T.alloc_var("int32")
            exp_scale_inv_f = T.alloc_var("float32")
            values = T.alloc_fragment([group_size], "float32")

            expert_idx = 0
            if masked_layout:
                hidden_group_idx = pid % hidden_dim_num_groups
                token_idx = (pid // hidden_dim_num_groups) % num_tokens_per_expert
                expert_idx = pid // (hidden_dim_num_groups * num_tokens_per_expert)
                valid_group = token_idx < masked_m[expert_idx]
            else:
                hidden_group_idx = pid % hidden_dim_num_groups
                token_idx = pid // hidden_dim_num_groups
                valid_group = True

            group_base = (
                expert_idx * num_tokens_per_expert * hidden_dim_num_groups * group_size
                + token_idx * hidden_dim_num_groups * group_size
                + hidden_group_idx * group_size
            )
            input_base = group_base
            secondary_base = group_base + hidden_dim_num_groups * group_size
            if fuse_silu_and_mul:
                input_base = (
                    expert_idx
                    * num_tokens_per_expert
                    * hidden_dim_num_groups
                    * group_size
                    * 2
                    + token_idx * hidden_dim_num_groups * group_size * 2
                    + hidden_group_idx * group_size
                )
                secondary_base = input_base + hidden_dim_num_groups * group_size

            absmax = eps
            for i in T.Parallel(group_size):
                values[i] = 0.0
                if valid_group:
                    values[i] = T.Cast("float32", input[input_base + i])
                    if fuse_silu_and_mul:
                        silu = values[i] / (1.0 + T.exp2(-values[i] * _LOG2E))
                        values[i] = silu * T.Cast("float32", input[secondary_base + i])

            for i in T.serial(group_size):
                absmax = T.max(absmax, T.abs(values[i]))

            if scale_ue8m0:
                exp_scale_inv_f = T.log2(absmax / max_8bit)
                exp_scale_inv = T.Cast("int32", exp_scale_inv_f)
                if T.Cast("float32", exp_scale_inv) < exp_scale_inv_f:
                    exp_scale_inv += 1
                scale = T.exp2(-T.Cast("float32", exp_scale_inv))
                scale_inv = T.exp2(T.Cast("float32", exp_scale_inv))
            else:
                scale_inv = absmax / max_8bit
                scale = max_8bit / absmax

            for i in T.Parallel(group_size):
                if valid_group:
                    q_val = T.min(T.max(values[i] * scale, min_8bit), max_8bit)
                    output_q[group_base + i] = T.Cast(output_dtype, q_val)

            if valid_group:
                if scale_ue8m0:
                    scale_index = (
                        scale_storage_offset
                        + expert_idx * scale_expert_stride * 4
                        + token_idx * scale_token_stride * 4
                        + (hidden_group_idx // 4) * scale_hidden_stride * 4
                        + hidden_group_idx % 4
                    )
                    output_s[scale_index] = T.Cast("uint8", exp_scale_inv + 127)
                else:
                    scale_index = (
                        scale_storage_offset
                        + expert_idx * scale_expert_stride
                        + token_idx * scale_token_stride
                        + hidden_group_idx * scale_hidden_stride
                    )
                    output_s[scale_index] = scale_inv

    return sglang_musa_per_token_group_quant_8bit_kernel


@tilelang.jit(pass_configs=_QUANT_PASS_CONFIGS, compile_flags=MUSA_COMPILE_FLAGS)
def _per_token_group_quant_8bit_fast_kernel(
    group_size,
    input_dtype,
    output_dtype,
    scale_dtype,
    fuse_silu_and_mul,
    groups_per_block,
    vec_elems,
):
    input_numel = T.dynamic("input_numel")
    output_numel = T.dynamic("output_numel")
    scale_storage_numel = T.dynamic("scale_storage_numel")
    threads_per_group = group_size // vec_elems
    num_threads = threads_per_group * groups_per_block

    @T.prim_func
    def sglang_musa_per_token_group_quant_8bit_fast_kernel(
        input: T.Tensor([input_numel], dtype=input_dtype),
        output_q: T.Tensor([output_numel], dtype=output_dtype),
        output_s: T.Tensor([scale_storage_numel], dtype=scale_dtype),
        total_blocks: T.int32,
        total_groups: T.int32,
        hidden_dim_num_groups: T.int32,
        scale_storage_offset: T.int32,
        scale_token_stride: T.int32,
        scale_hidden_stride: T.int32,
        eps: T.float32,
        min_8bit: T.float32,
        max_8bit: T.float32,
    ):
        with T.Kernel(total_blocks, threads=num_threads) as (bid,):
            tid = T.get_thread_binding()
            subgroup = tid // threads_per_group
            lane = tid % threads_per_group
            group_id = T.alloc_var("int32")
            token_idx = T.alloc_var("int32")
            hidden_group_idx = T.alloc_var("int32")
            output_base = T.alloc_var("int32")
            input_base = T.alloc_var("int32")
            secondary_base = T.alloc_var("int32")
            scale_index = T.alloc_var("int32")
            local_absmax = T.alloc_local([1], "float32")
            scale = T.alloc_local([1], "float32")
            scale_inv = T.alloc_local([1], "float32")
            values = T.alloc_local([vec_elems], "float32")

            group_id = bid * groups_per_block + subgroup
            token_idx = group_id // hidden_dim_num_groups
            hidden_group_idx = group_id - token_idx * hidden_dim_num_groups
            output_base = group_id * group_size + lane * vec_elems
            input_base = output_base
            secondary_base = output_base + hidden_dim_num_groups * group_size
            if fuse_silu_and_mul:
                input_base = (
                    token_idx * hidden_dim_num_groups * group_size * 2
                    + hidden_group_idx * group_size
                    + lane * vec_elems
                )
                secondary_base = input_base + hidden_dim_num_groups * group_size

            local_absmax[0] = eps
            for i in T.vectorized(vec_elems):
                values[i] = 0.0
                if group_id < total_groups:
                    values[i] = T.Cast("float32", input[input_base + i])
                    if fuse_silu_and_mul:
                        silu = values[i] / (1.0 + T.exp2(-values[i] * _LOG2E))
                        values[i] = silu * T.Cast("float32", input[secondary_base + i])
                    local_absmax[0] = T.max(local_absmax[0], T.abs(values[i]))

            if threads_per_group >= 32:
                local_absmax[0] = T.max(
                    local_absmax[0], T.shfl_xor(local_absmax[0], 16)
                )
            if threads_per_group >= 16:
                local_absmax[0] = T.max(local_absmax[0], T.shfl_xor(local_absmax[0], 8))
            if threads_per_group >= 8:
                local_absmax[0] = T.max(local_absmax[0], T.shfl_xor(local_absmax[0], 4))
            if threads_per_group >= 4:
                local_absmax[0] = T.max(local_absmax[0], T.shfl_xor(local_absmax[0], 2))
            if threads_per_group >= 2:
                local_absmax[0] = T.max(local_absmax[0], T.shfl_xor(local_absmax[0], 1))

            scale_inv[0] = local_absmax[0] / max_8bit
            scale[0] = max_8bit / local_absmax[0]

            for i in T.vectorized(vec_elems):
                if group_id < total_groups:
                    q_val = T.min(T.max(values[i] * scale[0], min_8bit), max_8bit)
                    output_q[output_base + i] = T.Cast(output_dtype, q_val)

            if group_id < total_groups and lane == 0:
                scale_index = (
                    scale_storage_offset
                    + token_idx * scale_token_stride
                    + hidden_group_idx * scale_hidden_stride
                )
                output_s[scale_index] = scale_inv[0]

    return sglang_musa_per_token_group_quant_8bit_fast_kernel


@tilelang.jit(pass_configs=_QUANT_PASS_CONFIGS, compile_flags=MUSA_COMPILE_FLAGS)
def _per_token_group_quant_8bit_row_kernel(
    group_size,
    input_dtype,
    output_dtype,
    scale_dtype,
    fuse_silu_and_mul,
    groups_per_block,
    vec_elems,
):
    input_numel = T.dynamic("input_numel")
    output_numel = T.dynamic("output_numel")
    scale_storage_numel = T.dynamic("scale_storage_numel")
    row_group_blocks = T.dynamic("row_group_blocks")
    num_tokens = T.dynamic("num_tokens")
    threads_per_group = group_size // vec_elems
    num_threads = threads_per_group * groups_per_block

    @T.prim_func
    def sglang_musa_per_token_group_quant_8bit_row_kernel(
        input: T.Tensor([input_numel], dtype=input_dtype),
        output_q: T.Tensor([output_numel], dtype=output_dtype),
        output_s: T.Tensor([scale_storage_numel], dtype=scale_dtype),
        row_group_blocks: T.int32,
        num_tokens: T.int32,
        hidden_dim_num_groups: T.int32,
        scale_storage_offset: T.int32,
        scale_token_stride: T.int32,
        scale_hidden_stride: T.int32,
        eps: T.float32,
        min_8bit: T.float32,
        max_8bit: T.float32,
    ):
        with T.Kernel(row_group_blocks, num_tokens, threads=num_threads) as (bh, bt):
            tid = T.get_thread_binding()
            subgroup = tid // threads_per_group
            lane = tid % threads_per_group
            hidden_group_idx = T.alloc_var("int32")
            output_base = T.alloc_var("int32")
            input_base = T.alloc_var("int32")
            secondary_base = T.alloc_var("int32")
            scale_index = T.alloc_var("int32")
            local_absmax = T.alloc_local([1], "float32")
            scale = T.alloc_local([1], "float32")
            scale_inv = T.alloc_local([1], "float32")
            values = T.alloc_local([vec_elems], "float32")

            hidden_group_idx = bh * groups_per_block + subgroup
            output_base = (
                bt * hidden_dim_num_groups * group_size
                + hidden_group_idx * group_size
                + lane * vec_elems
            )
            input_base = output_base
            secondary_base = output_base + hidden_dim_num_groups * group_size
            if fuse_silu_and_mul:
                input_base = (
                    bt * hidden_dim_num_groups * group_size * 2
                    + hidden_group_idx * group_size
                    + lane * vec_elems
                )
                secondary_base = input_base + hidden_dim_num_groups * group_size

            local_absmax[0] = eps
            for i in T.vectorized(vec_elems):
                values[i] = 0.0
                if hidden_group_idx < hidden_dim_num_groups:
                    values[i] = T.Cast("float32", input[input_base + i])
                    if fuse_silu_and_mul:
                        silu = values[i] / (1.0 + T.exp2(-values[i] * _LOG2E))
                        values[i] = silu * T.Cast("float32", input[secondary_base + i])
                    local_absmax[0] = T.max(local_absmax[0], T.abs(values[i]))

            if threads_per_group >= 32:
                local_absmax[0] = T.max(
                    local_absmax[0], T.shfl_xor(local_absmax[0], 16)
                )
            if threads_per_group >= 16:
                local_absmax[0] = T.max(local_absmax[0], T.shfl_xor(local_absmax[0], 8))
            if threads_per_group >= 8:
                local_absmax[0] = T.max(local_absmax[0], T.shfl_xor(local_absmax[0], 4))
            if threads_per_group >= 4:
                local_absmax[0] = T.max(local_absmax[0], T.shfl_xor(local_absmax[0], 2))
            if threads_per_group >= 2:
                local_absmax[0] = T.max(local_absmax[0], T.shfl_xor(local_absmax[0], 1))

            scale_inv[0] = local_absmax[0] / max_8bit
            scale[0] = max_8bit / local_absmax[0]

            for i in T.vectorized(vec_elems):
                if hidden_group_idx < hidden_dim_num_groups:
                    q_val = T.min(T.max(values[i] * scale[0], min_8bit), max_8bit)
                    output_q[output_base + i] = T.Cast(output_dtype, q_val)

            if hidden_group_idx < hidden_dim_num_groups and lane == 0:
                scale_index = (
                    scale_storage_offset
                    + bt * scale_token_stride
                    + hidden_group_idx * scale_hidden_stride
                )
                output_s[scale_index] = scale_inv[0]

    return sglang_musa_per_token_group_quant_8bit_row_kernel


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
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> None:
    assert input.numel() > 0, "input must be non-empty"
    assert input.is_contiguous(), "input must be contiguous"
    assert output_q.is_contiguous(), "output_q must be contiguous"
    assert group_size in _SUPPORTED_GROUP_SIZES, "Unsupported group_size"
    assert input.dtype in _INPUT_DTYPES, "input dtype must be float16 or bfloat16"
    assert (
        output_q.dtype in _OUTPUT_DTYPES
    ), "output_q dtype must be int8 or float8_e4m3fn"
    assert (
        input.numel() % group_size == 0
    ), "input.numel() must be divisible by group_size"
    _check_limits(output_q.dtype, min_8bit, max_8bit)

    masked_layout = masked_m is not None
    assert output_s.dim() == (3 if masked_layout else 2)
    assert output_q.size(-1) % group_size == 0
    if fuse_silu_and_mul:
        assert input.size(-1) == output_q.size(-1) * 2
    else:
        assert input.numel() == output_q.numel()
    if scale_ue8m0:
        assert output_s.dtype in (torch.int32, torch.uint8)
        assert output_q.dtype == fp8_dtype
    else:
        assert output_s.dtype == torch.float32

    if masked_layout:
        assert masked_m is not None and masked_m.dtype == torch.int32
        assert masked_m.is_contiguous()
        num_experts = output_q.size(0)
        num_tokens_per_expert = output_q.size(-2)
        total_groups = (
            num_experts * num_tokens_per_expert * (output_q.size(-1) // group_size)
        )
        masked_m_arg = masked_m
    else:
        num_experts = 1
        num_tokens_per_expert = output_q.size(-2)
        total_groups = output_q.numel() // group_size
        masked_m_arg = _dummy_mask(input.device)

    hidden_dim_num_groups = output_q.size(-1) // group_size
    (
        scale_storage,
        scale_storage_offset,
        scale_token_stride,
        scale_hidden_stride,
    ) = _scale_storage_metadata(output_s, scale_ue8m0)
    scale_expert_stride = output_s.stride(0) if masked_layout else 0

    if not masked_layout and not scale_ue8m0:
        groups_per_block = 8 if group_size == 128 else 16
        vec_elems = 8 if group_size == 128 else 16
        if group_size == 128:
            row_group_blocks = tilelang.cdiv(hidden_dim_num_groups, groups_per_block)
            kernel = _per_token_group_quant_8bit_row_kernel(
                group_size=group_size,
                input_dtype=input.dtype,
                output_dtype=output_q.dtype,
                scale_dtype=scale_storage.dtype,
                fuse_silu_and_mul=fuse_silu_and_mul,
                groups_per_block=groups_per_block,
                vec_elems=vec_elems,
            )
            kernel(
                _flatten_for_kernel(input),
                _flatten_for_kernel(output_q),
                scale_storage,
                row_group_blocks,
                num_tokens_per_expert,
                hidden_dim_num_groups,
                scale_storage_offset,
                scale_token_stride,
                scale_hidden_stride,
                float(eps),
                float(min_8bit),
                float(max_8bit),
            )
            return

        total_blocks = tilelang.cdiv(total_groups, groups_per_block)
        kernel = _per_token_group_quant_8bit_fast_kernel(
            group_size=group_size,
            input_dtype=input.dtype,
            output_dtype=output_q.dtype,
            scale_dtype=scale_storage.dtype,
            fuse_silu_and_mul=fuse_silu_and_mul,
            groups_per_block=groups_per_block,
            vec_elems=vec_elems,
        )
        kernel(
            _flatten_for_kernel(input),
            _flatten_for_kernel(output_q),
            scale_storage,
            total_blocks,
            total_groups,
            hidden_dim_num_groups,
            scale_storage_offset,
            scale_token_stride,
            scale_hidden_stride,
            float(eps),
            float(min_8bit),
            float(max_8bit),
        )
        return

    kernel = _per_token_group_quant_8bit_kernel(
        group_size=group_size,
        input_dtype=input.dtype,
        output_dtype=output_q.dtype,
        scale_dtype=scale_storage.dtype,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_layout=masked_layout,
        scale_ue8m0=scale_ue8m0,
    )
    kernel(
        _flatten_for_kernel(input),
        _flatten_for_kernel(output_q),
        scale_storage,
        masked_m_arg,
        total_groups,
        num_tokens_per_expert,
        hidden_dim_num_groups,
        scale_storage_offset,
        scale_expert_stride,
        scale_token_stride,
        scale_hidden_stride,
        float(eps),
        float(min_8bit),
        float(max_8bit),
    )
