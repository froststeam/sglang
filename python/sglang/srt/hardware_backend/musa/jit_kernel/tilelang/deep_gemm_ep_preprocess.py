"""TileLang MUSA DeepGEMM EP masked-GEMM preprocess fast paths."""

import functools

import tilelang
import tilelang.language as T
import torch

from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.deep_gemm_contig_preprocess import (
    _ATOMIC_HELPER_H,
    _COMPILE_FLAGS,
    _PASS_CONFIGS,
    _bf16_config,
    _clear_i32_kernel,
    _fp8_config,
)
from sglang.srt.utils.custom_op import register_custom_op

_MAX_TOKENS_PER_ASSIGN_LAUNCH = 32768


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=_COMPILE_FLAGS + ["-include", _ATOMIC_HELPER_H],
)
def _bf16_assign_masked_kernel(
    input_dtype,
    output_dtype,
    hidden_groups: int,
    groups_per_block: int,
    vec_elems: int,
    topk: int,
):
    input_numel = T.dynamic("input_numel")
    output_numel = T.dynamic("output_numel")
    src2dst_numel = T.dynamic("src2dst_numel")
    expert_numel = T.dynamic("expert_numel")
    group_size = 128
    hidden_size = hidden_groups * group_size
    threads_per_group = group_size // vec_elems
    num_threads = threads_per_group * groups_per_block

    @T.prim_func
    def deep_gemm_ep_preprocess_bf16_assign_masked(
        hidden: T.Tensor((input_numel,), input_dtype),
        topk_ids: T.Tensor((src2dst_numel,), "int32"),
        cursor: T.Tensor((expert_numel,), "int32"),
        src2dst: T.Tensor((src2dst_numel,), "int32"),
        output_bf16: T.Tensor((output_numel,), output_dtype),
        num_tokens: T.int32,
        num_local_experts: T.int32,
        m_max: T.int32,
    ):
        with T.Kernel(num_tokens, threads=num_threads) as (bt,):
            tid = T.get_thread_binding()
            subgroup = tid // threads_per_group
            lane = tid % threads_per_group
            elem_base = T.alloc_var("int32")
            hidden_group = T.alloc_var("int32")
            input_base = T.alloc_var("int64")
            local_rank = T.alloc_var("int32")
            dst = T.alloc_var("int32")
            out_base = T.alloc_var("int64")
            dst_shared = T.alloc_shared((topk,), "int32")

            if tid < topk:
                slot = bt * topk + tid
                expert = topk_ids[slot]
                if expert >= 0 and expert < num_local_experts:
                    local_rank = T.call_extern(
                        "int32",
                        "sgl_tl_atomic_add_offset",
                        T.address_of(cursor[0]),
                        expert,
                        T.int32(1),
                    )
                    if local_rank < m_max:
                        dst = expert * m_max + local_rank
                        src2dst[slot] = dst
                        dst_shared[tid] = dst
                    else:
                        src2dst[slot] = T.int32(-1)
                        dst_shared[tid] = T.int32(-1)
                else:
                    src2dst[slot] = T.int32(-1)
                    dst_shared[tid] = T.int32(-1)
            T.sync_threads()

            elem_base = lane * vec_elems
            for tile in T.serial(T.ceildiv(hidden_groups, groups_per_block)):
                hidden_group = tile * groups_per_block + subgroup
                if hidden_group < hidden_groups:
                    input_base = (
                        T.Cast("int64", bt) * hidden_size
                        + hidden_group * group_size
                        + elem_base
                    )
                    for topk_idx in T.serial(topk):
                        dst = dst_shared[topk_idx]
                        if dst >= 0:
                            out_base = (
                                T.Cast("int64", dst) * hidden_size
                                + hidden_group * group_size
                                + elem_base
                            )
                            if vec_elems == 8:
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_copy_bf16x8",
                                    T.address_of(output_bf16[0]),
                                    T.address_of(hidden[0]),
                                    out_base,
                                    input_base,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_copy_bf16x8",
                                    T.address_of(output_bf16[0]),
                                    T.address_of(hidden[0]),
                                    out_base,
                                    input_base,
                                )
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_copy_bf16x8",
                                    T.address_of(output_bf16[0]),
                                    T.address_of(hidden[0]),
                                    out_base + 8,
                                    input_base + 8,
                                )

    return deep_gemm_ep_preprocess_bf16_assign_masked


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=_COMPILE_FLAGS + ["-include", _ATOMIC_HELPER_H],
)
def _fp8_assign_masked_kernel(
    input_dtype,
    output_dtype,
    hidden_groups: int,
    groups_per_block: int,
    vec_elems: int,
    topk: int,
):
    input_numel = T.dynamic("input_numel")
    output_numel = T.dynamic("output_numel")
    scale_numel = T.dynamic("scale_numel")
    topk_ids_numel = T.dynamic("topk_ids_numel")
    expert_numel = T.dynamic("expert_numel")
    group_size = 128
    hidden_size = hidden_groups * group_size
    threads_per_group = group_size // vec_elems
    num_threads = threads_per_group * groups_per_block

    @T.prim_func
    def deep_gemm_ep_preprocess_fp8_assign_masked(
        hidden: T.Tensor((input_numel,), input_dtype),
        topk_ids: T.Tensor((topk_ids_numel,), "int32"),
        cursor: T.Tensor((expert_numel,), "int32"),
        src2dst: T.Tensor((topk_ids_numel,), "int32"),
        output_q: T.Tensor((output_numel,), output_dtype),
        output_s: T.Tensor((scale_numel,), "float32"),
        num_tokens: T.int32,
        num_local_experts: T.int32,
        m_max: T.int32,
        eps: T.float32,
        max_8bit: T.float32,
    ):
        with T.Kernel(num_tokens, threads=num_threads) as (bt,):
            tid = T.get_thread_binding()
            subgroup = tid // threads_per_group
            lane = tid % threads_per_group
            elem_base = T.alloc_var("int32")
            hidden_group = T.alloc_var("int32")
            input_base = T.alloc_var("int64")
            local_absmax = T.alloc_local((1,), "float32")
            scale = T.alloc_local((1,), "float32")
            scale_inv = T.alloc_local((1,), "float32")
            values = T.alloc_local((vec_elems,), "float32")
            local_rank = T.alloc_var("int32")
            dst = T.alloc_var("int32")
            out_base = T.alloc_var("int64")
            dst_shared = T.alloc_shared((topk,), "int32")

            if tid < topk:
                slot = bt * topk + tid
                expert = topk_ids[slot]
                if expert >= 0 and expert < num_local_experts:
                    local_rank = T.call_extern(
                        "int32",
                        "sgl_tl_atomic_add_offset",
                        T.address_of(cursor[0]),
                        expert,
                        T.int32(1),
                    )
                    if local_rank < m_max:
                        dst = expert * m_max + local_rank
                        src2dst[slot] = dst
                        dst_shared[tid] = dst
                    else:
                        src2dst[slot] = T.int32(-1)
                        dst_shared[tid] = T.int32(-1)
                else:
                    src2dst[slot] = T.int32(-1)
                    dst_shared[tid] = T.int32(-1)
            T.sync_threads()

            elem_base = lane * vec_elems
            for tile in T.serial(T.ceildiv(hidden_groups, groups_per_block)):
                hidden_group = tile * groups_per_block + subgroup
                if hidden_group < hidden_groups:
                    input_base = (
                        T.Cast("int64", bt) * hidden_size
                        + hidden_group * group_size
                        + elem_base
                    )
                    local_absmax[0] = eps
                    for i in T.vectorized(vec_elems):
                        values[i] = T.Cast("float32", hidden[input_base + i])
                        local_absmax[0] = T.max(local_absmax[0], T.abs(values[i]))

                    if threads_per_group >= 32:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 16)
                        )
                    if threads_per_group >= 16:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 8)
                        )
                    if threads_per_group >= 8:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 4)
                        )
                    if threads_per_group >= 4:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 2)
                        )
                    if threads_per_group >= 2:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 1)
                        )

                    scale_inv[0] = local_absmax[0] / max_8bit
                    scale[0] = max_8bit / local_absmax[0]

                    for topk_idx in T.serial(topk):
                        dst = dst_shared[topk_idx]
                        if dst >= 0:
                            out_base = (
                                T.Cast("int64", dst) * hidden_size
                                + hidden_group * group_size
                                + elem_base
                            )
                            if vec_elems == 4:
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base,
                                    T.min(
                                        T.max(values[0] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[1] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[2] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[3] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                )
                            elif vec_elems == 8:
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base,
                                    T.min(
                                        T.max(values[0] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[1] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[2] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[3] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                )
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base + 4,
                                    T.min(
                                        T.max(values[4] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[5] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[6] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[7] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                )
                            else:
                                for i in T.serial(4):
                                    T.call_extern(
                                        "handle",
                                        "sgl_tl_store_fp8e4m3x4",
                                        T.address_of(output_q[0]),
                                        out_base + i * 4,
                                        T.min(
                                            T.max(values[i * 4] * scale[0], -max_8bit),
                                            max_8bit,
                                        ),
                                        T.min(
                                            T.max(
                                                values[i * 4 + 1] * scale[0], -max_8bit
                                            ),
                                            max_8bit,
                                        ),
                                        T.min(
                                            T.max(
                                                values[i * 4 + 2] * scale[0], -max_8bit
                                            ),
                                            max_8bit,
                                        ),
                                        T.min(
                                            T.max(
                                                values[i * 4 + 3] * scale[0], -max_8bit
                                            ),
                                            max_8bit,
                                        ),
                                    )
                            if lane == 0:
                                output_s[dst * hidden_groups + hidden_group] = (
                                    scale_inv[0]
                                )

    return deep_gemm_ep_preprocess_fp8_assign_masked


def can_use_ep_bf16_tilelang(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
) -> bool:
    return (
        hidden_states.is_musa
        and topk_ids.is_musa
        and hidden_states.dtype == torch.bfloat16
        and topk_ids.dtype == torch.int32
        and hidden_states.dim() == 2
        and topk_ids.dim() == 2
        and topk_ids.shape[-1] <= 16
        and int(num_local_experts) > 0
        and _bf16_config(hidden_states.shape[1]) is not None
    )


def can_use_ep_fp8_tilelang(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
) -> bool:
    return (
        hidden_states.is_musa
        and topk_ids.is_musa
        and hidden_states.dtype == torch.bfloat16
        and topk_ids.dtype == torch.int32
        and hidden_states.dim() == 2
        and topk_ids.dim() == 2
        and topk_ids.shape[-1] <= 16
        and int(num_local_experts) > 0
        and _fp8_config(hidden_states.shape[1]) is not None
    )


def _impl_bf16(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    src2dst: torch.Tensor,
    output: torch.Tensor,
    num_local_experts: int,
) -> None:
    num_tokens = hidden_states.shape[0]
    topk = topk_ids.shape[-1]
    hidden_groups, groups_per_block, vec_elems = _bf16_config(hidden_states.shape[1])
    clear = _clear_i32_kernel()
    compact = _bf16_assign_masked_kernel(
        hidden_states.dtype,
        output.dtype,
        hidden_groups,
        groups_per_block,
        vec_elems,
        topk,
    )

    clear(masked_m, tilelang.cdiv(int(num_local_experts), 256), int(num_local_experts))
    for start in range(0, num_tokens, _MAX_TOKENS_PER_ASSIGN_LAUNCH):
        end = min(start + _MAX_TOKENS_PER_ASSIGN_LAUNCH, num_tokens)
        compact(
            hidden_states[start:end].reshape(-1),
            topk_ids[start:end].reshape(-1),
            masked_m,
            src2dst[start * topk : end * topk],
            output.reshape(-1),
            end - start,
            int(num_local_experts),
            output.shape[1],
        )


def _impl_fp8(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    src2dst: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    num_local_experts: int,
) -> None:
    num_tokens = hidden_states.shape[0]
    topk = topk_ids.shape[-1]
    hidden_groups, groups_per_block, vec_elems = _fp8_config(hidden_states.shape[1])
    clear = _clear_i32_kernel()
    compact = _fp8_assign_masked_kernel(
        hidden_states.dtype,
        output.dtype,
        hidden_groups,
        groups_per_block,
        vec_elems,
        topk,
    )

    clear(masked_m, tilelang.cdiv(int(num_local_experts), 256), int(num_local_experts))
    for start in range(0, num_tokens, _MAX_TOKENS_PER_ASSIGN_LAUNCH):
        end = min(start + _MAX_TOKENS_PER_ASSIGN_LAUNCH, num_tokens)
        compact(
            hidden_states[start:end].reshape(-1),
            topk_ids[start:end].reshape(-1),
            masked_m,
            src2dst[start * topk : end * topk],
            output.reshape(-1),
            output_scale.reshape(-1),
            end - start,
            int(num_local_experts),
            output.shape[1],
            1.0e-10,
            448.0,
        )


@register_custom_op(
    op_name="musa_deep_gemm_ep_preprocess_bf16_tilelang",
    mutates_args=["masked_m", "src2dst", "output"],
)
def _custom_bf16(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    src2dst: torch.Tensor,
    output: torch.Tensor,
    num_local_experts: int,
) -> None:
    _impl_bf16(
        topk_ids,
        hidden_states,
        masked_m,
        src2dst,
        output,
        int(num_local_experts),
    )


def deep_gemm_ep_preprocess_bf16_tilelang(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    src2dst: torch.Tensor,
    output: torch.Tensor,
    num_local_experts: int,
) -> None:
    _custom_bf16(
        topk_ids,
        hidden_states,
        masked_m,
        src2dst,
        output,
        int(num_local_experts),
    )


@register_custom_op(
    op_name="musa_deep_gemm_ep_preprocess_fp8_tilelang",
    mutates_args=["masked_m", "src2dst", "output", "output_scale"],
)
def _custom_fp8(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    src2dst: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    num_local_experts: int,
) -> None:
    _impl_fp8(
        topk_ids,
        hidden_states,
        masked_m,
        src2dst,
        output,
        output_scale,
        int(num_local_experts),
    )


def deep_gemm_ep_preprocess_fp8_tilelang(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    src2dst: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    num_local_experts: int,
) -> None:
    _custom_fp8(
        topk_ids,
        hidden_states,
        masked_m,
        src2dst,
        output,
        output_scale,
        int(num_local_experts),
    )
