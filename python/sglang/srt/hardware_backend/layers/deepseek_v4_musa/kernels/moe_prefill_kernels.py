from functools import lru_cache

from .kernel_common import _tilelang_jit, _tilelang_musa_aggressive_pass_configs


@lru_cache(maxsize=None)
def _tilelang_moe_deepgemm_compact_quant_scatter_kernel(
    hidden_size: int,
    topk: int,
    input_dtype: str,
    group_size: int = 128,
    # S5000 h4096 TP8 prefill prefers 4 warps/CTA: same instruction body as
    # 8 warps/CTA, but more CTAs improve memory-latency hiding.
    groups_per_cta: int = 4,
):
    import tilelang
    import tilelang.language as T

    if group_size != 128:
        raise ValueError(f"compact quant scatter only supports group_size=128, got {group_size}")
    if hidden_size % group_size != 0:
        raise ValueError(f"hidden_size must be divisible by {group_size}, got {hidden_size}")
    if groups_per_cta <= 0:
        raise ValueError(f"groups_per_cta must be positive, got {groups_per_cta}")

    num_tokens = T.dynamic("num_tokens")
    num_rows = T.dynamic("num_rows")
    num_experts = T.dynamic("num_experts")
    scale_size = hidden_size // group_size
    values_per_lane = 4
    threads = groups_per_cta * 32
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_moe_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    @_tilelang_jit(
        tilelang,
        f"dsv4_moe_compact_quant_scatter_h{hidden_size}_topk{topk}_{input_dtype}_g{groups_per_cta}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang, disable_index_promotion=True, compile_profile="ls"
        ),
    )
    def compact_quant_scatter_kernel(
        hidden_states: T.Tensor[(num_tokens, hidden_size), input_dtype],
        compact_input_u8: T.Tensor[(num_rows, hidden_size), T.uint8],
        compact_scale: T.Tensor[(num_rows, scale_size), T.float32],
        topk_ids: T.Tensor[(num_tokens, topk), T.int32],
        offsets: T.Tensor[(num_experts,), T.int32],
        route_ranks: T.Tensor[(num_tokens, topk), T.int32],
        src2dst: T.Tensor[(num_tokens, topk), T.int32],
        num_local_experts: T.int32,
    ):
        total_groups = num_tokens * scale_size
        with T.Kernel(T.ceildiv(total_groups, groups_per_cta), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx - warp * 32
            group_linear = block_id * groups_per_cta + warp
            token_id = group_linear // scale_size
            scale_group = group_linear - token_id * scale_size
            elem_base = scale_group * group_size + lane * values_per_lane
            valid_group = group_linear < total_groups

            vals = T.alloc_local((values_per_lane,), dtype=T.float32)
            local_amax = T.alloc_local((1,), dtype=T.float32)
            local_amax[0] = 0.0

            if valid_group:
                for vec in T.vectorized(values_per_lane):
                    vals[vec] = T.Cast(
                        "float32",
                        hidden_states[
                            token_id,
                            T.Cast("int64", elem_base + vec),
                        ],
                    )
                    local_amax[0] = T.max(local_amax[0], abs_f32(vals[vec]))

                local_amax[0] = T.warp_reduce_max(local_amax[0])
                scale = T.max(local_amax[0], 1.0e-10) / 448.0
                inv_scale = 1.0 / scale
                packed = T.call_extern(
                    "uint32",
                    "tl_dsv4_moe_pack_fp8x4_e4m3_u32",
                    T.clamp(vals[0] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[1] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[2] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[3] * inv_scale, -448.0, 448.0),
                )

                for route in range(topk):
                    expert_id = topk_ids[token_id, route]
                    if expert_id >= 0 and expert_id < num_local_experts:
                        dst_idx = offsets[T.Cast("int64", expert_id)] + route_ranks[token_id, route]
                        dst_idx_i64 = T.Cast("int64", dst_idx)
                        if scale_group == 0 and lane == 0:
                            src2dst[token_id, route] = dst_idx
                        T.stg32(
                            compact_input_u8[
                                dst_idx_i64,
                                T.Cast("int64", elem_base),
                            ],
                            packed,
                        )
                        if lane == 0:
                            compact_scale[dst_idx_i64, scale_group] = scale
                    else:
                        if scale_group == 0 and lane == 0:
                            src2dst[token_id, route] = -1

    return compact_quant_scatter_kernel


@lru_cache(maxsize=None)
def _tilelang_moe_deepgemm_static_cap_quant_scatter_kernel(
    hidden_size: int,
    topk: int,
    input_dtype: str,
    group_size: int = 128,
    groups_per_cta: int = 8,
):
    import tilelang
    import tilelang.language as T

    if group_size != 128:
        raise ValueError(f"static-cap quant scatter only supports group_size=128, got {group_size}")
    if hidden_size % group_size != 0:
        raise ValueError(f"hidden_size must be divisible by {group_size}, got {hidden_size}")
    if groups_per_cta <= 0:
        raise ValueError(f"groups_per_cta must be positive, got {groups_per_cta}")

    num_tokens = T.dynamic("num_tokens")
    num_rows = T.dynamic("num_rows")
    num_experts = T.dynamic("num_experts")
    scale_size = hidden_size // group_size
    values_per_lane = 4
    threads = groups_per_cta * 32
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_moe_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    @_tilelang_jit(
        tilelang,
        f"dsv4_moe_static_cap_quant_scatter_h{hidden_size}_topk{topk}_{input_dtype}_g{groups_per_cta}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang, disable_index_promotion=True, compile_profile="ls"
        ),
    )
    def static_cap_quant_scatter_kernel(
        hidden_states: T.Tensor[(num_tokens, hidden_size), input_dtype],
        compact_input_u8: T.Tensor[(num_rows, hidden_size), T.uint8],
        compact_scale: T.Tensor[(num_rows, scale_size), T.float32],
        topk_ids: T.Tensor[(num_tokens, topk), T.int32],
        route_ranks: T.Tensor[(num_tokens, topk), T.int32],
        src2dst: T.Tensor[(num_tokens, topk), T.int32],
        overflow_flag: T.Tensor[(1,), T.int32],
        num_local_experts: T.int32,
        cap_per_expert: T.int32,
    ):
        total_groups = num_tokens * scale_size
        with T.Kernel(T.ceildiv(total_groups, groups_per_cta), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx - warp * 32
            group_linear = block_id * groups_per_cta + warp
            token_id = group_linear // scale_size
            scale_group = group_linear - token_id * scale_size
            elem_base = scale_group * group_size + lane * values_per_lane
            valid_group = group_linear < total_groups

            vals = T.alloc_local((values_per_lane,), dtype=T.float32)
            local_amax = T.alloc_local((1,), dtype=T.float32)
            local_amax[0] = 0.0

            if valid_group:
                for vec in T.vectorized(values_per_lane):
                    vals[vec] = T.Cast(
                        "float32",
                        hidden_states[
                            token_id,
                            T.Cast("int64", elem_base + vec),
                        ],
                    )
                    local_amax[0] = T.max(local_amax[0], abs_f32(vals[vec]))

                local_amax[0] = T.warp_reduce_max(local_amax[0])
                scale = T.max(local_amax[0], 1.0e-10) / 448.0
                inv_scale = 1.0 / scale
                packed = T.call_extern(
                    "uint32",
                    "tl_dsv4_moe_pack_fp8x4_e4m3_u32",
                    T.clamp(vals[0] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[1] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[2] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[3] * inv_scale, -448.0, 448.0),
                )

                for route in range(topk):
                    expert_id = topk_ids[token_id, route]
                    if expert_id >= 0 and expert_id < num_local_experts:
                        rank = route_ranks[token_id, route]
                        if rank >= 0 and rank < cap_per_expert:
                            dst_idx = expert_id * cap_per_expert + rank
                            dst_idx_i64 = T.Cast("int64", dst_idx)
                            if scale_group == 0 and lane == 0:
                                src2dst[token_id, route] = dst_idx
                            T.stg32(
                                compact_input_u8[
                                    dst_idx_i64,
                                    T.Cast("int64", elem_base),
                                ],
                                packed,
                            )
                            if lane == 0:
                                compact_scale[dst_idx_i64, scale_group] = scale
                        else:
                            if scale_group == 0 and lane == 0:
                                src2dst[token_id, route] = -1
                                T.atomic_add(overflow_flag[0], 1, memory_order="relaxed")
                    else:
                        if scale_group == 0 and lane == 0:
                            src2dst[token_id, route] = -1

    return static_cap_quant_scatter_kernel


@lru_cache(maxsize=None)
def _tilelang_moe_deepgemm_static_cap_src2dst_quant_scatter_kernel(
    hidden_size: int,
    topk: int,
    input_dtype: str,
    group_size: int = 128,
    groups_per_cta: int = 8,
):
    import tilelang
    import tilelang.language as T

    if group_size != 128:
        raise ValueError(
            f"static-cap src2dst quant scatter only supports group_size=128, got {group_size}"
        )
    if hidden_size % group_size != 0:
        raise ValueError(f"hidden_size must be divisible by {group_size}, got {hidden_size}")
    if groups_per_cta <= 0:
        raise ValueError(f"groups_per_cta must be positive, got {groups_per_cta}")

    num_tokens = T.dynamic("num_tokens")
    num_rows = T.dynamic("num_rows")
    scale_size = hidden_size // group_size
    values_per_lane = 4
    threads = groups_per_cta * 32
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_moe_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    @_tilelang_jit(
        tilelang,
        f"dsv4_moe_static_cap_src2dst_quant_scatter_h{hidden_size}_topk{topk}_{input_dtype}_g{groups_per_cta}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang, disable_index_promotion=True, compile_profile="ls"
        ),
    )
    def src2dst_quant_scatter_kernel(
        hidden_states: T.Tensor[(num_tokens, hidden_size), input_dtype],
        compact_input_u8: T.Tensor[(num_rows, hidden_size), T.uint8],
        compact_scale: T.Tensor[(num_rows, scale_size), T.float32],
        src2dst: T.Tensor[(num_tokens, topk), T.int32],
    ):
        total_groups = num_tokens * scale_size
        with T.Kernel(T.ceildiv(total_groups, groups_per_cta), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx - warp * 32
            group_linear = block_id * groups_per_cta + warp
            token_id = group_linear // scale_size
            scale_group = group_linear - token_id * scale_size
            elem_base = scale_group * group_size + lane * values_per_lane
            valid_group = group_linear < total_groups

            vals = T.alloc_local((values_per_lane,), dtype=T.float32)
            local_amax = T.alloc_local((1,), dtype=T.float32)
            local_amax[0] = 0.0

            if valid_group:
                for vec in T.vectorized(values_per_lane):
                    vals[vec] = T.Cast(
                        "float32",
                        hidden_states[
                            token_id,
                            T.Cast("int64", elem_base + vec),
                        ],
                    )
                    local_amax[0] = T.max(local_amax[0], abs_f32(vals[vec]))

                local_amax[0] = T.warp_reduce_max(local_amax[0])
                scale = T.max(local_amax[0], 1.0e-10) / 448.0
                inv_scale = 1.0 / scale
                packed = T.call_extern(
                    "uint32",
                    "tl_dsv4_moe_pack_fp8x4_e4m3_u32",
                    T.clamp(vals[0] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[1] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[2] * inv_scale, -448.0, 448.0),
                    T.clamp(vals[3] * inv_scale, -448.0, 448.0),
                )

                for route in range(topk):
                    dst_idx = src2dst[token_id, route]
                    if dst_idx >= 0:
                        dst_idx_i64 = T.Cast("int64", dst_idx)
                        T.stg32(
                            compact_input_u8[
                                dst_idx_i64,
                                T.Cast("int64", elem_base),
                            ],
                            packed,
                        )
                        if lane == 0:
                            compact_scale[dst_idx_i64, scale_group] = scale

    return src2dst_quant_scatter_kernel


@lru_cache(maxsize=None)
def _tilelang_moe_post_combine_kernel(
    hidden_size: int,
    topk: int,
    input_dtype: str,
    output_dtype: str,
    block_h: int = 512,
    threads: int = 256,
):
    import tilelang
    import tilelang.language as T

    if block_h % threads != 0:
        raise ValueError(f"block_h={block_h} must be divisible by threads={threads}")

    num_tokens = T.dynamic("num_tokens")
    num_rows = T.dynamic("num_rows")
    values_per_thread = block_h // threads

    @_tilelang_jit(
        tilelang,
        f"dsv4_moe_post_combine_h{hidden_size}_topk{topk}_{input_dtype}_{output_dtype}_b{block_h}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang, disable_index_promotion=True, compile_profile="ls"
        ),
    )
    def post_combine_kernel(
        down_output: T.Tensor[(num_rows, hidden_size), input_dtype],
        output: T.Tensor[(num_tokens, hidden_size), output_dtype],
        src2dst: T.Tensor[(num_tokens, topk), T.int32],
        topk_ids: T.Tensor[(num_tokens, topk), T.int32],
        topk_weights: T.Tensor[(num_tokens, topk), T.float32],
    ):
        with T.Kernel(num_tokens, T.ceildiv(hidden_size, block_h), threads=threads) as (token_id, block_id):
            tx = T.get_thread_binding()
            elem_base = block_id * block_h + tx * values_per_thread
            acc = T.alloc_local((values_per_thread,), dtype=T.float32)

            for vec in T.vectorized(values_per_thread):
                acc[vec] = 0.0

            for route in range(topk):
                expert_id = topk_ids[token_id, route]
                if expert_id >= 0:
                    dst_idx = src2dst[token_id, route]
                    weight = topk_weights[token_id, route]
                    dst_idx_i64 = T.Cast("int64", dst_idx)
                    for vec in T.vectorized(values_per_thread):
                        col = elem_base + vec
                        if col < hidden_size:
                            acc[vec] += T.Cast(
                                "float32",
                                down_output[dst_idx_i64, T.Cast("int64", col)],
                            ) * weight

            for vec in T.vectorized(values_per_thread):
                col = elem_base + vec
                if col < hidden_size:
                    output[token_id, T.Cast("int64", col)] = T.cast(acc[vec], output_dtype)

    return post_combine_kernel


@lru_cache(maxsize=None)
def _tilelang_moe_post_combine_src2dst_kernel(
    hidden_size: int,
    topk: int,
    input_dtype: str,
    output_dtype: str,
    block_h: int = 1024,
    threads: int = 256,
):
    import tilelang
    import tilelang.language as T

    if block_h % threads != 0:
        raise ValueError(f"block_h={block_h} must be divisible by threads={threads}")

    num_tokens = T.dynamic("num_tokens")
    num_rows = T.dynamic("num_rows")
    values_per_thread = block_h // threads

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_moe_post_combine_src2dst_h{hidden_size}_topk{topk}_"
            f"{input_dtype}_{output_dtype}_b{block_h}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang, disable_index_promotion=True, compile_profile="ls"
        ),
    )
    def post_combine_src2dst_kernel(
        down_output: T.Tensor[(num_rows, hidden_size), input_dtype],
        output: T.Tensor[(num_tokens, hidden_size), output_dtype],
        src2dst: T.Tensor[(num_tokens, topk), T.int32],
        topk_weights: T.Tensor[(num_tokens, topk), T.float32],
    ):
        with T.Kernel(num_tokens, T.ceildiv(hidden_size, block_h), threads=threads) as (
            token_id,
            block_id,
        ):
            tx = T.get_thread_binding()
            elem_base = block_id * block_h + tx * values_per_thread
            acc = T.alloc_local((values_per_thread,), dtype=T.float32)

            for vec in T.vectorized(values_per_thread):
                acc[vec] = 0.0

            for route in range(topk):
                dst_idx = src2dst[token_id, route]
                if dst_idx >= 0:
                    weight = topk_weights[token_id, route]
                    dst_idx_i64 = T.Cast("int64", dst_idx)
                    for vec in T.vectorized(values_per_thread):
                        col = elem_base + vec
                        if col < hidden_size:
                            acc[vec] += T.Cast(
                                "float32",
                                down_output[dst_idx_i64, T.Cast("int64", col)],
                            ) * weight

            for vec in T.vectorized(values_per_thread):
                col = elem_base + vec
                if col < hidden_size:
                    output[token_id, T.Cast("int64", col)] = T.cast(acc[vec], output_dtype)

    return post_combine_src2dst_kernel


@lru_cache(maxsize=None)
def _tilelang_moe_post_combine_src2dst_cached_kernel(
    hidden_size: int,
    topk: int,
    input_dtype: str,
    output_dtype: str,
    block_h: int = 1024,
    threads: int = 256,
):
    import tilelang
    import tilelang.language as T

    if block_h % threads != 0:
        raise ValueError(f"block_h={block_h} must be divisible by threads={threads}")

    num_tokens = T.dynamic("num_tokens")
    num_rows = T.dynamic("num_rows")
    values_per_thread = block_h // threads

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_moe_post_combine_src2dst_cached_h{hidden_size}_topk{topk}_"
            f"{input_dtype}_{output_dtype}_b{block_h}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang, disable_index_promotion=True, compile_profile="ls"
        ),
    )
    def post_combine_src2dst_cached_kernel(
        down_output: T.Tensor[(num_rows, hidden_size), input_dtype],
        output: T.Tensor[(num_tokens, hidden_size), output_dtype],
        src2dst: T.Tensor[(num_tokens, topk), T.int32],
        topk_weights: T.Tensor[(num_tokens, topk), T.float32],
    ):
        with T.Kernel(num_tokens, T.ceildiv(hidden_size, block_h), threads=threads) as (
            token_id,
            block_id,
        ):
            tx = T.get_thread_binding()
            route_dst = T.alloc_shared((topk,), dtype=T.int32)
            route_weight = T.alloc_shared((topk,), dtype=T.float32)

            if tx < topk:
                route_dst[tx] = src2dst[token_id, tx]
                route_weight[tx] = topk_weights[token_id, tx]
            T.sync_threads()

            elem_base = block_id * block_h + tx * values_per_thread
            acc = T.alloc_local((values_per_thread,), dtype=T.float32)

            for vec in T.vectorized(values_per_thread):
                acc[vec] = 0.0

            for route in range(topk):
                dst_idx = route_dst[route]
                if dst_idx >= 0:
                    weight = route_weight[route]
                    dst_idx_i64 = T.Cast("int64", dst_idx)
                    for vec in T.vectorized(values_per_thread):
                        col = elem_base + vec
                        if col < hidden_size:
                            acc[vec] += T.Cast(
                                "float32",
                                down_output[dst_idx_i64, T.Cast("int64", col)],
                            ) * weight

            for vec in T.vectorized(values_per_thread):
                col = elem_base + vec
                if col < hidden_size:
                    output[token_id, T.Cast("int64", col)] = T.cast(acc[vec], output_dtype)

    return post_combine_src2dst_cached_kernel


__all__ = [
    "_tilelang_moe_deepgemm_compact_quant_scatter_kernel",
    "_tilelang_moe_deepgemm_static_cap_quant_scatter_kernel",
    "_tilelang_moe_deepgemm_static_cap_src2dst_quant_scatter_kernel",
    "_tilelang_moe_post_combine_kernel",
    "_tilelang_moe_post_combine_src2dst_cached_kernel",
    "_tilelang_moe_post_combine_src2dst_kernel",
]
