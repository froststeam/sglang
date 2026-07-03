from functools import lru_cache

import torch

from .kernel_common import (
    _tilelang_jit,
    _tilelang_musa_aggressive_pass_configs,
    _tilelang_musa_dsa_pass_configs,
)

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_nv_block_kernel(
    page_bytes: int,
    page_size: int,
    input_base_offset: int,
    input_row_stride: int,
    compile_profile: str = "dsa_full",
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    tile_dim = 64
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        "dsv4_flashmla_cache_pack_store_nv_block_bf16",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            disable_index_promotion=True,
            compile_profile=compile_profile,
        ),
    )
    def pack_store_flashmla_cache_nv_block_kernel(
        input_storage: T.Tensor[(num_input_elements,), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            loc = indices[token_id]
            page_idx, token_offset = page_index_and_offset(loc)
            page_idx_i64 = T.Cast("int64", page_idx)

            if warp < scale_dim:
                vals = T.alloc_local((2,), dtype=T.bfloat16)
                fvals = T.alloc_local((2,), dtype=T.float32)
                local_amax = T.alloc_local((1,), dtype=T.float32)
                tile_amax = T.alloc_local((1,), dtype=T.float32)
                elem_base = warp * tile_dim + lane * 2
                input_base_i64 = (
                    T.Cast("int64", input_base_offset)
                    + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                    + T.Cast("int64", elem_base)
                )
                local_amax[0] = 0.0
                for vec in T.vectorized(2):
                    vals[vec] = input_storage[input_base_i64 + T.Cast("int64", vec)]
                    fvals[vec] = T.Cast("float32", vals[vec])
                    local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))
                tile_amax[0] = T.warp_reduce_max(local_amax[0])
                scale_byte, inv_scale = pow2_scale_byte_and_inv(tile_amax[0] / 448.0)
                if lane == 0:
                    cache_u8[
                        page_idx_i64,
                        T.Cast("int64", page_size * token_stride_bytes)
                        + T.Cast("int64", token_offset * (scale_dim + 1))
                        + T.Cast("int64", warp),
                    ] = scale_byte
                tile_offset_i64 = (
                    T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                    + T.Cast("int64", elem_base)
                )
                for vec in T.vectorized(2):
                    cache_fp8[page_idx_i64, tile_offset_i64 + T.Cast("int64", vec)] = T.clamp(
                        fvals[vec] * inv_scale,
                        -448.0,
                        448.0,
                    )
            else:
                elem = lane * rope_pack_elems
                rope_input_i64 = (
                    T.Cast("int64", input_base_offset)
                    + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                    + T.Cast("int64", nope_dim + elem)
                )
                lo = T.reinterpret("uint16", input_storage[rope_input_i64])
                hi = T.reinterpret("uint16", input_storage[rope_input_i64 + T.Cast("int64", 1)])
                rope_offset_u32_i64 = (
                    T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                    + T.Cast("int64", nope_dim)
                ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", lane)
                cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)

    return pack_store_flashmla_cache_nv_block_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_decode_x4_kernel(
    page_bytes: int,
    page_size: int,
    input_base_offset: int,
    input_row_stride: int,
    use_i32_addresses: bool = False,
    dsa_compile_flags: str = "0",
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    if dsa_compile_flags not in {"0", "dsa"}:
        raise ValueError(f"Unsupported decode_x4 dsa_compile_flags={dsa_compile_flags!r}")
    variant = "i32addr" if use_i32_addresses else "i64addr"
    if dsa_compile_flags != "0":
        variant = f"{variant}_{dsa_compile_flags}"
    pass_configs = (
        _tilelang_musa_dsa_pass_configs(
            tilelang,
            full=False,
            disable_index_promotion=True,
        )
        if dsa_compile_flags != "0"
        else _tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True)
    )
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_exp_scale(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        return T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        f"dsv4_flashmla_cache_pack_store_decode_x4_bf16_{variant}",
        pass_configs=pass_configs,
    )
    def pack_store_flashmla_cache_decode_x4_kernel(
        input_storage: T.Tensor[(num_input_elements,), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(num_tokens, threads=threads, prelude=prelude) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            loc = indices[token_id]
            loc_valid = loc >= 0 and T.Cast("int64", loc) < T.Cast("int64", num_pages * page_size)
            if loc_valid:
                page_idx, token_offset = page_index_and_offset(loc)

                if warp < scale_dim:
                    vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                    fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                    local_amax = T.alloc_local((1,), dtype=T.float32)
                    tile_amax = T.alloc_local((1,), dtype=T.float32)
                    elem_base = warp * tile_dim + lane * vector_elems
                    input_base = input_base_offset + token_id * input_row_stride + elem_base
                    if not use_i32_addresses:
                        input_base_i64 = (
                            T.Cast("int64", input_base_offset)
                            + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                            + T.Cast("int64", elem_base)
                        )
                    local_amax[0] = 0.0
                    if lane < 16:
                        for vec in T.vectorized(vector_elems):
                            if use_i32_addresses:
                                vals[vec] = input_storage[input_base + vec]
                            else:
                                vals[vec] = input_storage[input_base_i64 + T.Cast("int64", vec)]
                            fvals[vec] = T.Cast("float32", vals[vec])
                            local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))
                    tile_amax[0] = T.warp_reduce_max(local_amax[0])
                    exp_scale = pow2_exp_scale(tile_amax[0] / 448.0)
                    scale_byte = T.Cast("uint8", exp_scale + 127)
                    inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
                    if lane == 0:
                        scale_offset = page_size * token_stride_bytes + token_offset * (scale_dim + 1) + warp
                        if use_i32_addresses:
                            cache_u8[page_idx, scale_offset] = scale_byte
                        else:
                            cache_u8[T.Cast("int64", page_idx), T.Cast("int64", scale_offset)] = scale_byte
                    if lane < 16:
                        tile_offset = token_offset * token_stride_bytes + elem_base
                        packed = T.call_extern(
                            "uint32",
                            "tl_dsv4_pack_fp8x4_e4m3_u32",
                            T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                        )
                        if use_i32_addresses:
                            T.stg32(cache_u8[page_idx, tile_offset], packed)
                        else:
                            T.stg32(cache_u8[T.Cast("int64", page_idx), T.Cast("int64", tile_offset)], packed)
                else:
                    elem = lane * rope_pack_elems
                    rope_input = input_base_offset + token_id * input_row_stride + nope_dim + elem
                    if use_i32_addresses:
                        lo = T.reinterpret("uint16", input_storage[rope_input])
                        hi = T.reinterpret("uint16", input_storage[rope_input + 1])
                    else:
                        rope_input_i64 = (
                            T.Cast("int64", input_base_offset)
                            + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                            + T.Cast("int64", nope_dim + elem)
                        )
                        lo = T.reinterpret("uint16", input_storage[rope_input_i64])
                        hi = T.reinterpret("uint16", input_storage[rope_input_i64 + T.Cast("int64", 1)])
                    rope_offset_u32 = (token_offset * token_stride_bytes + nope_dim) // (2 * rope_pack_elems) + lane
                    if use_i32_addresses:
                        cache_u32[page_idx, rope_offset_u32] = T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)
                    else:
                        cache_u32[T.Cast("int64", page_idx), T.Cast("int64", rope_offset_u32)] = (
                            T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)
                        )

    return pack_store_flashmla_cache_decode_x4_kernel


@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_decode_x4_fp32_kernel(
    page_bytes: int,
    page_size: int,
    input_base_offset: int,
    input_row_stride: int,
    compile_profile: str = "dsa_full",
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_exp_scale(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        return T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        "dsv4_flashmla_cache_pack_store_decode_x4_fp32",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            disable_index_promotion=True,
            compile_profile=compile_profile,
        ),
    )
    def pack_store_flashmla_cache_decode_x4_fp32_kernel(
        input_storage: T.Tensor[(num_input_elements,), T.float32],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(num_tokens, threads=threads, prelude=prelude) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            loc = indices[token_id]
            loc_valid = loc >= 0 and T.Cast("int64", loc) < T.Cast("int64", num_pages * page_size)
            if loc_valid:
                page_idx, token_offset = page_index_and_offset(loc)

                if warp < scale_dim:
                    fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                    local_amax = T.alloc_local((1,), dtype=T.float32)
                    tile_amax = T.alloc_local((1,), dtype=T.float32)
                    elem_base = warp * tile_dim + lane * vector_elems
                    input_base_i64 = (
                        T.Cast("int64", input_base_offset)
                        + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                        + T.Cast("int64", elem_base)
                    )
                    local_amax[0] = 0.0
                    if lane < 16:
                        for vec in T.vectorized(vector_elems):
                            fvals[vec] = input_storage[input_base_i64 + T.Cast("int64", vec)]
                            local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))
                    tile_amax[0] = T.warp_reduce_max(local_amax[0])
                    exp_scale = pow2_exp_scale(tile_amax[0] / 448.0)
                    scale_byte = T.Cast("uint8", exp_scale + 127)
                    inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
                    if lane == 0:
                        scale_offset = page_size * token_stride_bytes + token_offset * (scale_dim + 1) + warp
                        cache_u8[T.Cast("int64", page_idx), T.Cast("int64", scale_offset)] = scale_byte
                    if lane < 16:
                        tile_offset = token_offset * token_stride_bytes + elem_base
                        packed = T.call_extern(
                            "uint32",
                            "tl_dsv4_pack_fp8x4_e4m3_u32",
                            T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                        )
                        T.stg32(cache_u8[T.Cast("int64", page_idx), T.Cast("int64", tile_offset)], packed)
                else:
                    elem = lane * rope_pack_elems
                    rope_input_i64 = (
                        T.Cast("int64", input_base_offset)
                        + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                        + T.Cast("int64", nope_dim + elem)
                    )
                    lo = T.reinterpret("uint16", T.Cast("bfloat16", input_storage[rope_input_i64]))
                    hi = T.reinterpret(
                        "uint16",
                        T.Cast("bfloat16", input_storage[rope_input_i64 + T.Cast("int64", 1)]),
                    )
                    rope_offset_u32 = (token_offset * token_stride_bytes + nope_dim) // (2 * rope_pack_elems) + lane
                    cache_u32[T.Cast("int64", page_idx), T.Cast("int64", rope_offset_u32)] = (
                        T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)
                    )

    return pack_store_flashmla_cache_decode_x4_fp32_kernel


@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_decode_vec2_kernel(
    page_bytes: int,
    page_size: int,
    input_base_offset: int,
    input_row_stride: int,
    use_i32_addresses: bool = False,
    dsa_compile_flags: str = "0",
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 2
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    if dsa_compile_flags not in {"0", "dsa"}:
        raise ValueError(f"Unsupported decode_vec2 dsa_compile_flags={dsa_compile_flags!r}")
    variant = "i32addr" if use_i32_addresses else "i64addr"
    if dsa_compile_flags != "0":
        variant = f"{variant}_{dsa_compile_flags}"
    pass_configs = (
        _tilelang_musa_dsa_pass_configs(
            tilelang,
            full=False,
            disable_index_promotion=True,
        )
        if dsa_compile_flags != "0"
        else _tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True)
    )

    def pow2_exp_scale(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        return T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        f"dsv4_flashmla_cache_pack_store_decode_vec2_bf16_{variant}",
        pass_configs=pass_configs,
    )
    def pack_store_flashmla_cache_decode_vec2_kernel(
        input_storage: T.Tensor[(num_input_elements,), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            loc = indices[token_id]
            page_idx, token_offset = page_index_and_offset(loc)

            if warp < scale_dim:
                vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                local_amax = T.alloc_local((1,), dtype=T.float32)
                tile_amax = T.alloc_local((1,), dtype=T.float32)
                elem_base = warp * tile_dim + lane * vector_elems
                input_base = input_base_offset + token_id * input_row_stride + elem_base
                if not use_i32_addresses:
                    input_base_i64 = (
                        T.Cast("int64", input_base_offset)
                        + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                        + T.Cast("int64", elem_base)
                    )
                local_amax[0] = 0.0
                for vec in T.vectorized(vector_elems):
                    if use_i32_addresses:
                        vals[vec] = input_storage[input_base + vec]
                    else:
                        vals[vec] = input_storage[input_base_i64 + T.Cast("int64", vec)]
                    fvals[vec] = T.Cast("float32", vals[vec])
                    local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))
                tile_amax[0] = T.warp_reduce_max(local_amax[0])
                exp_scale = pow2_exp_scale(tile_amax[0] / 448.0)
                scale_byte = T.Cast("uint8", exp_scale + 127)
                inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
                if lane == 0:
                    scale_offset = page_size * token_stride_bytes + token_offset * (scale_dim + 1) + warp
                    if use_i32_addresses:
                        cache_u8[page_idx, scale_offset] = scale_byte
                    else:
                        cache_u8[T.Cast("int64", page_idx), T.Cast("int64", scale_offset)] = scale_byte
                tile_offset = token_offset * token_stride_bytes + elem_base
                for vec in T.vectorized(vector_elems):
                    if use_i32_addresses:
                        cache_fp8[page_idx, tile_offset + vec] = T.clamp(
                            fvals[vec] * inv_scale,
                            -448.0,
                            448.0,
                        )
                    else:
                        cache_fp8[T.Cast("int64", page_idx), T.Cast("int64", tile_offset + vec)] = T.clamp(
                            fvals[vec] * inv_scale,
                            -448.0,
                            448.0,
                        )
            else:
                elem = lane * rope_pack_elems
                rope_input = input_base_offset + token_id * input_row_stride + nope_dim + elem
                if use_i32_addresses:
                    lo = T.reinterpret("uint16", input_storage[rope_input])
                    hi = T.reinterpret("uint16", input_storage[rope_input + 1])
                else:
                    rope_input_i64 = (
                        T.Cast("int64", input_base_offset)
                        + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                        + T.Cast("int64", nope_dim + elem)
                    )
                    lo = T.reinterpret("uint16", input_storage[rope_input_i64])
                    hi = T.reinterpret("uint16", input_storage[rope_input_i64 + T.Cast("int64", 1)])
                rope_offset_u32 = (token_offset * token_stride_bytes + nope_dim) // (2 * rope_pack_elems) + lane
                if use_i32_addresses:
                    cache_u32[page_idx, rope_offset_u32] = T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)
                else:
                    cache_u32[T.Cast("int64", page_idx), T.Cast("int64", rope_offset_u32)] = (
                        T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)
                    )

    return pack_store_flashmla_cache_decode_vec2_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_decode_x4_flat_i32_kernel(
    page_bytes: int,
    page_size: int,
    input_base_offset: int,
    input_row_stride: int,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    page_u32_stride = page_bytes // 4
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_exp_scale(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        return T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        "dsv4_flashmla_cache_pack_store_decode_x4_bf16_flat_i32addr",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_decode_x4_flat_i32_kernel(
        input_storage: T.Tensor[(num_input_elements,), T.bfloat16],
        cache_u8_flat: T.Tensor[(num_pages * page_bytes,), T.uint8],
        cache_u32_flat: T.Tensor[(num_pages * page_u32_stride,), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(num_tokens, threads=threads, prelude=prelude) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            loc = indices[token_id]
            page_idx, token_offset = page_index_and_offset(loc)
            page_base = page_idx * page_bytes
            page_u32_base = page_idx * page_u32_stride

            if warp < scale_dim:
                vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                local_amax = T.alloc_local((1,), dtype=T.float32)
                tile_amax = T.alloc_local((1,), dtype=T.float32)
                elem_base = warp * tile_dim + lane * vector_elems
                input_base = input_base_offset + token_id * input_row_stride + elem_base
                local_amax[0] = 0.0
                if lane < 16:
                    for vec in T.vectorized(vector_elems):
                        vals[vec] = input_storage[input_base + vec]
                        fvals[vec] = T.Cast("float32", vals[vec])
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))
                tile_amax[0] = T.warp_reduce_max(local_amax[0])
                exp_scale = pow2_exp_scale(tile_amax[0] / 448.0)
                scale_byte = T.Cast("uint8", exp_scale + 127)
                inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
                if lane == 0:
                    scale_offset = page_base + page_size * token_stride_bytes + token_offset * (scale_dim + 1) + warp
                    cache_u8_flat[scale_offset] = scale_byte
                if lane < 16:
                    tile_offset_u32 = (page_base + token_offset * token_stride_bytes + elem_base) // 4
                    packed = T.call_extern(
                        "uint32",
                        "tl_dsv4_pack_fp8x4_e4m3_u32",
                        T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                    )
                    cache_u32_flat[tile_offset_u32] = packed
            else:
                elem = lane * rope_pack_elems
                rope_input = input_base_offset + token_id * input_row_stride + nope_dim + elem
                lo = T.reinterpret("uint16", input_storage[rope_input])
                hi = T.reinterpret("uint16", input_storage[rope_input + 1])
                rope_offset_u32 = (
                    page_u32_base
                    + (token_offset * token_stride_bytes + nope_dim) // (2 * rope_pack_elems)
                    + lane
                )
                cache_u32_flat[rope_offset_u32] = T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)

    return pack_store_flashmla_cache_decode_x4_flat_i32_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_warp_token_kernel(
    page_bytes: int,
    page_size: int,
    tokens_per_block: int = 8,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    input_row_stride = T.dynamic("input_row_stride")
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        f"dsv4_flashmla_cache_pack_store_warp_token_bf16_tpb{tokens_per_block}_{'full' if full_tiles else 'tail'}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_warp_token_kernel(
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_block), threads=threads) as block_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            token_id = block_id * tokens_per_block + warp
            valid = full_tiles or token_id < num_tokens
            elem_base = lane * 2

            if valid:
                loc = indices[token_id]
                page_idx, token_offset = page_index_and_offset(loc)
                page_idx_i64 = T.Cast("int64", page_idx)
                token_value_offset_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                scale_offset_base_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                    "int64", token_offset * (scale_dim + 1)
                )

                for tile_id in T.serial(0, scale_dim):
                    vals = T.alloc_local((2,), dtype=T.bfloat16)
                    fvals = T.alloc_local((2,), dtype=T.float32)
                    local_amax = T.alloc_local((1,), dtype=T.float32)
                    tile_amax = T.alloc_local((1,), dtype=T.float32)
                    tile_elem_base = tile_id * tile_dim + elem_base

                    local_amax[0] = 0.0
                    for vec in T.vectorized(2):
                        vals[vec] = k_nope[token_id, tile_elem_base + vec]
                        fvals[vec] = T.Cast("float32", vals[vec])
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))
                    tile_amax[0] = T.warp_reduce_max(local_amax[0])
                    scale_byte, inv_scale = pow2_scale_byte_and_inv(tile_amax[0] / 448.0)
                    if lane == 0:
                        cache_u8[page_idx_i64, scale_offset_base_i64 + T.Cast("int64", tile_id)] = scale_byte
                    tile_offset_i64 = token_value_offset_i64 + T.Cast("int64", tile_elem_base)
                    for vec in T.vectorized(2):
                        cache_fp8[page_idx_i64, tile_offset_i64 + T.Cast("int64", vec)] = T.clamp(
                            fvals[vec] * inv_scale,
                            -448.0,
                            448.0,
                        )

                elem = lane * rope_pack_elems
                lo = T.reinterpret("uint16", k_rope[token_id, elem])
                hi = T.reinterpret("uint16", k_rope[token_id, elem + 1])
                rope_offset_u32_i64 = (
                    token_value_offset_i64 + T.Cast("int64", nope_dim)
                ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", lane)
                cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)

    return pack_store_flashmla_cache_warp_token_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_warp_col_fused_kernel(
    page_bytes: int,
    page_size: int,
    tokens_per_warp: int = 8,
    rope_store_128: bool = False,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    warps = 8
    blk_m = warps * tokens_per_warp
    input_row_stride = T.dynamic("input_row_stride")
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_flashmla_cache_pack_store_warp_col_fused_bf16_tpw{tokens_per_warp}_"
            f"{'rope128' if rope_store_128 else 'rope32'}_{'full' if full_tiles else 'tail'}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_warp_col_fused_kernel(
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), threads=threads) as block_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32

            for token_iter in T.serial(0, tokens_per_warp):
                token_id = block_id * blk_m + warp * tokens_per_warp + token_iter
                valid = full_tiles or token_id < num_tokens
                if valid:
                    loc = indices[token_id]
                    page_idx, token_offset = page_index_and_offset(loc)
                    page_idx_i64 = T.Cast("int64", page_idx)
                    token_value_offset_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                    scale_offset_base_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                        "int64", token_offset * (scale_dim + 1)
                    )

                    for tile_id in T.serial(0, scale_dim):
                        v0 = T.alloc_local((1,), dtype=T.bfloat16)
                        v1 = T.alloc_local((1,), dtype=T.bfloat16)
                        f0 = T.alloc_local((1,), dtype=T.float32)
                        f1 = T.alloc_local((1,), dtype=T.float32)
                        tile_amax = T.alloc_local((1,), dtype=T.float32)
                        elem = tile_id * tile_dim + lane
                        v0[0] = k_nope[token_id, elem]
                        v1[0] = k_nope[token_id, elem + 32]
                        f0[0] = T.Cast("float32", v0[0])
                        f1[0] = T.Cast("float32", v1[0])
                        tile_amax[0] = T.warp_reduce_max(T.max(abs_f32(f0[0]), abs_f32(f1[0])))
                        scale_byte, inv_scale = pow2_scale_byte_and_inv(tile_amax[0] / 448.0)
                        if lane == 0:
                            cache_u8[page_idx_i64, scale_offset_base_i64 + T.Cast("int64", tile_id)] = scale_byte
                        tile_offset_i64 = token_value_offset_i64 + T.Cast("int64", tile_id * tile_dim)
                        cache_fp8[page_idx_i64, tile_offset_i64 + T.Cast("int64", lane)] = T.clamp(
                            f0[0] * inv_scale,
                            -448.0,
                            448.0,
                        )
                        cache_fp8[page_idx_i64, tile_offset_i64 + T.Cast("int64", lane + 32)] = T.clamp(
                            f1[0] * inv_scale,
                            -448.0,
                            448.0,
                        )

                    if rope_store_128:
                        if lane < 8:
                            rope_elem = lane * 8
                            packed_rope = T.ldg128(k_rope[token_id, rope_elem: rope_elem + 8])
                            rope_offset_u32_i64 = (
                                token_value_offset_i64 + T.Cast("int64", nope_dim + rope_elem * 2)
                            ) // T.Cast("int64", 4)
                            T.stg128(
                                cache_u32[
                                    page_idx_i64,
                                    rope_offset_u32_i64: rope_offset_u32_i64 + T.Cast("int64", 4),
                                ],
                                packed_rope,
                            )
                    else:
                        if lane < rope_dim // rope_pack_elems:
                            rope_elem = lane * rope_pack_elems
                            lo = T.reinterpret("uint16", k_rope[token_id, rope_elem])
                            hi = T.reinterpret("uint16", k_rope[token_id, rope_elem + 1])
                            rope_offset_u32_i64 = (
                                token_value_offset_i64 + T.Cast("int64", nope_dim)
                            ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", lane)
                            cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (
                                T.Cast("uint32", hi) << 16
                            )

    return pack_store_flashmla_cache_warp_col_fused_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_kernel(
    input_dtype: str,
    page_bytes: int,
    page_size: int,
    input_base_offset: int,
    input_row_stride: int,
    threads: int = 128,
    blk_m: int = 64,
    rope_store_128: bool = False,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    rope_pack_elems = 2
    rope_vector_elems = 4

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    @_tilelang_jit(tilelang, "dsv4_flashmla_cache_pack_store")
    def pack_store_flashmla_cache_kernel(
        input_storage: T.Tensor[(num_input_elements,), input_dtype],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_bf16: T.Tensor[(num_pages, page_bytes // 2), T.bfloat16],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), scale_dim, threads=threads) as (block_id, tile_id):
            nope = T.alloc_fragment((blk_m, tile_dim), dtype=input_dtype)
            amax = T.alloc_fragment((blk_m,), dtype=T.float32)
            inv_scale = T.alloc_fragment((blk_m,), dtype=T.float32)
            scale_byte = T.alloc_fragment((blk_m,), dtype=T.uint8)
            loc = T.alloc_fragment((blk_m,), dtype=T.int32)
            page_idx = T.alloc_fragment((blk_m,), dtype=T.int32)
            token_offset = T.alloc_fragment((blk_m,), dtype=T.int32)
            token_stride_bytes = nope_dim + rope_dim * 2

            for row in T.Parallel(blk_m):
                token_id = block_id * blk_m + row
                for elem_base in T.serial(0, tile_dim, vector_elems):
                    for vec in T.vectorized(vector_elems):
                        elem = elem_base + vec
                        if full_tiles or token_id < num_tokens:
                            nope[row, elem] = input_storage[
                                input_base_offset + token_id * input_row_stride + tile_id * tile_dim + elem
                            ]
                        else:
                            nope[row, elem] = 0.0
            T.reduce_absmax(nope, amax, dim=1)
            for row in T.Parallel(blk_m):
                token_id = block_id * blk_m + row
                if full_tiles or token_id < num_tokens:
                    loc[row] = indices[token_id]
                    page_idx[row] = loc[row] // page_size
                    token_offset[row] = loc[row] % page_size
                    scale_byte[row], inv_scale[row] = pow2_scale_byte_and_inv(amax[row] / 448.0)
                    cache_u8[
                        page_idx[row],
                        page_size * token_stride_bytes + token_offset[row] * (scale_dim + 1) + tile_id,
                    ] = scale_byte[row]
            if tile_id == 0:
                for row in T.Parallel(blk_m):
                    token_id = block_id * blk_m + row
                    for elem_base in T.serial(0, tile_dim, vector_elems):
                        for vec in T.vectorized(vector_elems):
                            elem = elem_base + vec
                            if full_tiles or token_id < num_tokens:
                                value = T.Cast("float32", nope[row, elem])
                                cache_fp8[
                                    page_idx[row],
                                    token_offset[row] * token_stride_bytes + elem,
                                ] = T.clamp(value * inv_scale[row], -448.0, 448.0)
                if rope_store_128:
                    for row in T.Parallel(blk_m):
                        token_id = block_id * blk_m + row
                        for elem in T.serial(0, rope_dim, 8):
                            if full_tiles or token_id < num_tokens:
                                packed_rope = T.ldg128(
                                    input_storage[
                                        input_base_offset + token_id * input_row_stride + nope_dim + elem:
                                        input_base_offset + token_id * input_row_stride + nope_dim + elem + 8
                                    ]
                                )
                                rope_offset_u32 = (token_offset[row] * token_stride_bytes + nope_dim + elem * 2) // 4
                                T.stg128(
                                    cache_u32[page_idx[row], rope_offset_u32: rope_offset_u32 + 4],
                                    packed_rope,
                                )
                else:
                    for row in T.Parallel(blk_m):
                        token_id = block_id * blk_m + row
                        for elem_base in T.serial(0, rope_dim // rope_pack_elems, rope_vector_elems):
                            for vec in T.vectorized(rope_vector_elems):
                                elem_pair = elem_base + vec
                                elem = elem_pair * rope_pack_elems
                                if full_tiles or token_id < num_tokens:
                                    if input_dtype == "bfloat16":
                                        lo = T.reinterpret(
                                            "uint16",
                                            input_storage[input_base_offset + token_id * input_row_stride + nope_dim + elem],
                                        )
                                        hi = T.reinterpret(
                                            "uint16",
                                            input_storage[input_base_offset + token_id * input_row_stride + nope_dim + elem + 1],
                                        )
                                    else:
                                        lo = T.reinterpret(
                                            "uint16",
                                            T.Cast(
                                                "bfloat16",
                                                input_storage[input_base_offset + token_id * input_row_stride + nope_dim + elem],
                                            ),
                                        )
                                        hi = T.reinterpret(
                                            "uint16",
                                            T.Cast(
                                                "bfloat16",
                                                input_storage[input_base_offset + token_id * input_row_stride + nope_dim + elem + 1],
                                            ),
                                        )
                                    cache_u32[
                                        page_idx[row],
                                        (token_offset[row] * token_stride_bytes + nope_dim) // (2 * rope_pack_elems) + elem_pair,
                                    ] = T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)
            else:
                for row in T.Parallel(blk_m):
                    token_id = block_id * blk_m + row
                    for elem_base in T.serial(0, tile_dim, vector_elems):
                        for vec in T.vectorized(vector_elems):
                            elem = elem_base + vec
                            if full_tiles or token_id < num_tokens:
                                value = T.Cast("float32", nope[row, elem])
                                cache_fp8[
                                    page_idx[row],
                                    token_offset[row] * token_stride_bytes + tile_id * tile_dim + elem,
                                ] = T.clamp(value * inv_scale[row], -448.0, 448.0)

    return pack_store_flashmla_cache_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_warp_col_kernel(
    input_dtype: str,
    page_bytes: int,
    page_size: int,
    input_base_offset: int,
    input_row_stride: int,
    rope_store_128: bool = False,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    warps = 8
    tokens_per_warp = 8
    blk_m = warps * tokens_per_warp

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_flashmla_cache_pack_store_warp_col_{input_dtype}_t{threads}_"
            f"tpw{tokens_per_warp}_{'full' if full_tiles else 'tail'}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_warp_col_kernel(
        input_storage: T.Tensor[(num_input_elements,), input_dtype],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), scale_dim, threads=threads) as (block_id, tile_id):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32

            for token_iter in T.serial(0, tokens_per_warp):
                token_id = block_id * blk_m + warp * tokens_per_warp + token_iter
                valid = full_tiles or token_id < num_tokens
                input_base = input_base_offset + token_id * input_row_stride + tile_id * tile_dim
                input_base_i64 = (
                    T.Cast("int64", input_base_offset)
                    + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                    + T.Cast("int64", tile_id * tile_dim)
                )
                v0 = T.alloc_local((1,), dtype=input_dtype)
                v1 = T.alloc_local((1,), dtype=input_dtype)
                f0 = T.alloc_local((1,), dtype=T.float32)
                f1 = T.alloc_local((1,), dtype=T.float32)
                tile_amax = T.alloc_local((1,), dtype=T.float32)

                if valid and input_base + lane < num_input_elements:
                    v0[0] = input_storage[input_base_i64 + T.Cast("int64", lane)]
                else:
                    v0[0] = 0.0
                if valid and input_base + lane + 32 < num_input_elements:
                    v1[0] = input_storage[input_base_i64 + T.Cast("int64", lane + 32)]
                else:
                    v1[0] = 0.0
                f0[0] = T.Cast("float32", v0[0])
                f1[0] = T.Cast("float32", v1[0])
                tile_amax[0] = T.warp_reduce_max(T.max(abs_f32(f0[0]), abs_f32(f1[0])))

                if valid:
                    loc = indices[token_id]
                    page_idx = loc // page_size
                    token_offset = loc % page_size
                    page_idx_i64 = T.Cast("int64", page_idx)
                    tile_offset_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes) + T.Cast(
                        "int64", tile_id * tile_dim
                    )
                    scale_offset_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                        "int64", token_offset * (scale_dim + 1) + tile_id
                    )
                    scale_byte, inv_scale = pow2_scale_byte_and_inv(tile_amax[0] / 448.0)
                    if lane == 0:
                        cache_u8[
                            page_idx_i64,
                            scale_offset_i64,
                        ] = scale_byte
                    cache_fp8[
                        page_idx_i64,
                        tile_offset_i64 + T.Cast("int64", lane),
                    ] = T.clamp(f0[0] * inv_scale, -448.0, 448.0)
                    cache_fp8[
                        page_idx_i64,
                        tile_offset_i64 + T.Cast("int64", lane + 32),
                    ] = T.clamp(f1[0] * inv_scale, -448.0, 448.0)

                    if tile_id == 0:
                        if rope_store_128:
                            if lane < 8:
                                elem = lane * 8
                                rope_input_i64 = (
                                    T.Cast("int64", input_base_offset)
                                    + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                                    + T.Cast("int64", nope_dim + elem)
                                )
                                packed_rope = T.ldg128(
                                    input_storage[
                                        rope_input_i64:
                                        rope_input_i64 + T.Cast("int64", 8)
                                    ]
                                )
                                rope_offset_u32_i64 = (
                                    T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                                    + T.Cast("int64", nope_dim + elem * 2)
                                ) // T.Cast("int64", 4)
                                T.stg128(
                                    cache_u32[page_idx_i64, rope_offset_u32_i64: rope_offset_u32_i64 + T.Cast("int64", 4)],
                                    packed_rope,
                                )
                        else:
                            if lane < rope_dim // rope_pack_elems:
                                elem = lane * rope_pack_elems
                                rope_input_i64 = (
                                    T.Cast("int64", input_base_offset)
                                    + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                                    + T.Cast("int64", nope_dim + elem)
                                )
                                if input_dtype == "bfloat16":
                                    lo = T.reinterpret(
                                        "uint16",
                                        input_storage[rope_input_i64],
                                    )
                                    hi = T.reinterpret(
                                        "uint16",
                                        input_storage[rope_input_i64 + T.Cast("int64", 1)],
                                    )
                                else:
                                    lo = T.reinterpret(
                                        "uint16",
                                        T.Cast("bfloat16", input_storage[rope_input_i64]),
                                    )
                                    hi = T.reinterpret(
                                        "uint16",
                                        T.Cast("bfloat16", input_storage[rope_input_i64 + T.Cast("int64", 1)]),
                                    )
                                rope_offset_u32_i64 = (
                                    T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                                    + T.Cast("int64", nope_dim)
                                ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", lane)
                                cache_u32[
                                    page_idx_i64,
                                    rope_offset_u32_i64,
                                ] = T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)

    return pack_store_flashmla_cache_warp_col_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_warp_col_split_kernel(
    page_bytes: int,
    page_size: int,
    rope_store_128: bool = False,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = 256
    warps = 8
    tokens_per_warp = 8
    blk_m = warps * tokens_per_warp
    input_row_stride = T.dynamic("input_row_stride")

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    @_tilelang_jit(
        tilelang,
        f"dsv4_flashmla_cache_pack_store_warp_col_split_bf16_t{threads}_tpw{tokens_per_warp}_{'full' if full_tiles else 'tail'}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_warp_col_split_kernel(
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), scale_dim, threads=threads) as (block_id, tile_id):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32

            for token_iter in T.serial(0, tokens_per_warp):
                token_id = block_id * blk_m + warp * tokens_per_warp + token_iter
                valid = full_tiles or token_id < num_tokens
                v0 = T.alloc_local((1,), dtype=T.bfloat16)
                v1 = T.alloc_local((1,), dtype=T.bfloat16)
                f0 = T.alloc_local((1,), dtype=T.float32)
                f1 = T.alloc_local((1,), dtype=T.float32)
                tile_amax = T.alloc_local((1,), dtype=T.float32)

                if valid:
                    v0[0] = k_nope[token_id, tile_id * tile_dim + lane]
                    v1[0] = k_nope[token_id, tile_id * tile_dim + lane + 32]
                else:
                    v0[0] = 0.0
                    v1[0] = 0.0
                f0[0] = T.Cast("float32", v0[0])
                f1[0] = T.Cast("float32", v1[0])
                tile_amax[0] = T.warp_reduce_max(T.max(abs_f32(f0[0]), abs_f32(f1[0])))

                if valid:
                    loc = indices[token_id]
                    page_idx = loc // page_size
                    token_offset = loc % page_size
                    page_idx_i64 = T.Cast("int64", page_idx)
                    tile_offset_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes) + T.Cast(
                        "int64", tile_id * tile_dim
                    )
                    scale_offset_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                        "int64", token_offset * (scale_dim + 1) + tile_id
                    )
                    scale_byte, inv_scale = pow2_scale_byte_and_inv(tile_amax[0] / 448.0)
                    if lane == 0:
                        cache_u8[page_idx_i64, scale_offset_i64] = scale_byte
                    cache_fp8[page_idx_i64, tile_offset_i64 + T.Cast("int64", lane)] = T.clamp(
                        f0[0] * inv_scale, -448.0, 448.0
                    )
                    cache_fp8[page_idx_i64, tile_offset_i64 + T.Cast("int64", lane + 32)] = T.clamp(
                        f1[0] * inv_scale, -448.0, 448.0
                    )

                    if tile_id == 0:
                        if rope_store_128:
                            if lane < 8:
                                elem = lane * 8
                                packed_rope = T.ldg128(k_rope[token_id, elem: elem + 8])
                                rope_offset_u32_i64 = (
                                    T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                                    + T.Cast("int64", nope_dim + elem * 2)
                                ) // T.Cast("int64", 4)
                                T.stg128(
                                    cache_u32[page_idx_i64, rope_offset_u32_i64: rope_offset_u32_i64 + T.Cast("int64", 4)],
                                    packed_rope,
                                )
                        else:
                            if lane < rope_dim // rope_pack_elems:
                                elem = lane * rope_pack_elems
                                lo = T.reinterpret("uint16", k_rope[token_id, elem])
                                hi = T.reinterpret("uint16", k_rope[token_id, elem + 1])
                                rope_offset_u32_i64 = (
                                    T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                                    + T.Cast("int64", nope_dim)
                                ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", lane)
                                cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (
                                    T.Cast("uint32", hi) << 16
                                )

    return pack_store_flashmla_cache_warp_col_split_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_vector_kernel(
    page_bytes: int,
    page_size: int,
    vector_elems: int = 2,
    tokens_per_warp: int = 8,
    warps: int = 8,
    use_stg32: bool = False,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = warps * 32
    blk_m = warps * tokens_per_warp
    input_row_stride = T.dynamic("input_row_stride")
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = None
    if use_stg32:
        prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_flashmla_cache_pack_store_vector_bf16_vec{vector_elems}_w{warps}_"
            f"tpw{tokens_per_warp}_{'stg32' if use_stg32 else 'typed'}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_vector_kernel(
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), scale_dim, threads=threads, prelude=prelude) as (block_id, tile_id):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32

            for token_iter in T.serial(0, tokens_per_warp):
                token_id = block_id * blk_m + warp * tokens_per_warp + token_iter
                valid = token_id < num_tokens
                vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                local_amax = T.alloc_local((1,), dtype=T.float32)
                tile_amax = T.alloc_local((1,), dtype=T.float32)
                elem_base = lane * vector_elems
                active_nope_lane = lane < (tile_dim // vector_elems)

                local_amax[0] = 0.0
                if valid and active_nope_lane:
                    for vec in T.vectorized(vector_elems):
                        elem = elem_base + vec
                        vals[vec] = k_nope[token_id, tile_id * tile_dim + elem]
                        fvals[vec] = T.Cast("float32", vals[vec])
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))
                else:
                    for vec in T.vectorized(vector_elems):
                        vals[vec] = 0.0
                        fvals[vec] = 0.0
                tile_amax[0] = T.warp_reduce_max(local_amax[0])

                if valid:
                    loc = indices[token_id]
                    page_idx, token_offset = page_index_and_offset(loc)
                    page_idx_i64 = T.Cast("int64", page_idx)
                    scale_byte, inv_scale = pow2_scale_byte_and_inv(tile_amax[0] / 448.0)
                    if lane == 0:
                        scale_offset_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                            "int64", token_offset * (scale_dim + 1) + tile_id
                        )
                        cache_u8[page_idx_i64, scale_offset_i64] = scale_byte
                    if active_nope_lane:
                        tile_offset_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes) + T.Cast(
                            "int64", tile_id * tile_dim + elem_base
                        )
                        if use_stg32:
                            packed = T.call_extern(
                                "uint32",
                                "tl_dsv4_pack_fp8x4_e4m3_u32",
                                T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                                T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                                T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                                T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                            )
                            T.stg32(cache_u8[page_idx_i64, tile_offset_i64], packed)
                        else:
                            for vec in T.vectorized(vector_elems):
                                cache_fp8[page_idx_i64, tile_offset_i64 + T.Cast("int64", vec)] = T.clamp(
                                    fvals[vec] * inv_scale,
                                    -448.0,
                                    448.0,
                                )

                    if tile_id == 0:
                        rope_elem_base = lane * rope_pack_elems
                        rope_offset_u32_i64 = (
                            T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                            + T.Cast("int64", nope_dim)
                        ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", lane)
                        lo = T.reinterpret("uint16", k_rope[token_id, rope_elem_base])
                        hi = T.reinterpret("uint16", k_rope[token_id, rope_elem_base + 1])
                        cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)

    return pack_store_flashmla_cache_vector_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_vector_x4_remap_kernel(
    page_bytes: int,
    page_size: int,
    tokens_per_half_warp: int = 8,
    warps: int = 4,
    use_shuffle_reduce: bool = False,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = warps * 32
    blk_m = warps * tokens_per_half_warp * 2
    input_row_stride = T.dynamic("input_row_stride")
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_flashmla_cache_pack_store_vector_x4_remap_bf16_w{warps}_"
            f"tphw{tokens_per_half_warp}_{'shuffle' if use_shuffle_reduce else 'shared'}_"
            f"{'full' if full_tiles else 'tail'}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_vector_x4_remap_kernel(
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), scale_dim, threads=threads, prelude=prelude) as (block_id, tile_id):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            half = lane // 16
            half_lane = lane % 16
            elem_base = half_lane * vector_elems
            if not use_shuffle_reduce:
                reductions = T.alloc_shared((threads,), dtype=T.float32)

            for token_iter in T.serial(0, tokens_per_half_warp):
                token_id = block_id * blk_m + warp * (tokens_per_half_warp * 2) + token_iter * 2 + half
                valid = full_tiles or token_id < num_tokens
                vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                local_amax = T.alloc_local((1,), dtype=T.float32)

                local_amax[0] = 0.0
                if valid:
                    for vec in T.vectorized(vector_elems):
                        elem = elem_base + vec
                        vals[vec] = k_nope[token_id, tile_id * tile_dim + elem]
                        fvals[vec] = T.Cast("float32", vals[vec])
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))
                else:
                    for vec in T.vectorized(vector_elems):
                        vals[vec] = 0.0
                        fvals[vec] = 0.0

                if use_shuffle_reduce:
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 8))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 4))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 2))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 1))
                else:
                    reductions[tx] = local_amax[0]
                    T.sync_threads()
                    if half_lane < 8:
                        reductions[tx] = T.max(reductions[tx], reductions[tx + 8])
                    T.sync_threads()
                    if half_lane < 4:
                        reductions[tx] = T.max(reductions[tx], reductions[tx + 4])
                    T.sync_threads()
                    if half_lane < 2:
                        reductions[tx] = T.max(reductions[tx], reductions[tx + 2])
                    T.sync_threads()
                    if half_lane < 1:
                        reductions[tx] = T.max(reductions[tx], reductions[tx + 1])
                    T.sync_threads()

                if valid:
                    loc = indices[token_id]
                    page_idx, token_offset = page_index_and_offset(loc)
                    page_idx_i64 = T.Cast("int64", page_idx)
                    if use_shuffle_reduce:
                        scale_byte, inv_scale = pow2_scale_byte_and_inv(local_amax[0] / 448.0)
                    else:
                        half_base = warp * 32 + half * 16
                        scale_byte, inv_scale = pow2_scale_byte_and_inv(reductions[half_base] / 448.0)
                    if half_lane == 0:
                        scale_offset_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                            "int64", token_offset * (scale_dim + 1) + tile_id
                        )
                        cache_u8[page_idx_i64, scale_offset_i64] = scale_byte

                    tile_offset_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes) + T.Cast(
                        "int64", tile_id * tile_dim + elem_base
                    )
                    packed = T.call_extern(
                        "uint32",
                        "tl_dsv4_pack_fp8x4_e4m3_u32",
                        T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                    )
                    T.stg32(cache_u8[page_idx_i64, tile_offset_i64], packed)

                    if tile_id == 0:
                        for rope_vec in T.vectorized(2):
                            rope_elem_pair = half_lane * 2 + rope_vec
                            rope_elem = rope_elem_pair * rope_pack_elems
                            lo = T.reinterpret("uint16", k_rope[token_id, rope_elem])
                            hi = T.reinterpret("uint16", k_rope[token_id, rope_elem + 1])
                            rope_offset_u32_i64 = (
                                T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                                + T.Cast("int64", nope_dim)
                            ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", rope_elem_pair)
                            cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (
                                T.Cast("uint32", hi) << 16
                            )
                if not use_shuffle_reduce:
                    T.sync_threads()

    return pack_store_flashmla_cache_vector_x4_remap_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_vector_x4_tile_fused_kernel(
    page_bytes: int,
    page_size: int,
    input_base_offset: int,
    tokens_per_half_warp: int = 1,
    warps: int = 4,
    rope_store_128: bool = False,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads = warps * 32
    blk_m = warps * tokens_per_half_warp * 2
    input_row_stride = T.dynamic("input_row_stride")
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_flashmla_cache_pack_store_vector_x4_tile_fused_bf16_w{warps}_"
            f"tphw{tokens_per_half_warp}_{'rope128' if rope_store_128 else 'rope32'}_"
            f"{'full' if full_tiles else 'tail'}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_vector_x4_tile_fused_kernel(
        input_storage: T.Tensor[(num_input_elements,), T.bfloat16],
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            half = lane // 16
            half_lane = lane % 16
            elem_base = half_lane * vector_elems

            for token_iter in T.serial(0, tokens_per_half_warp):
                token_id = block_id * blk_m + warp * (tokens_per_half_warp * 2) + token_iter * 2 + half
                valid = full_tiles or token_id < num_tokens
                scale_packed_lo = T.alloc_local((1,), dtype=T.uint32)
                scale_packed_hi = T.alloc_local((1,), dtype=T.uint32)
                scale_packed_lo[0] = T.Cast("uint32", 0)
                scale_packed_hi[0] = T.Cast("uint32", 0)

                if valid:
                    loc = indices[token_id]
                    page_idx, token_offset = page_index_and_offset(loc)
                    page_idx_i64 = T.Cast("int64", page_idx)
                    token_base_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                    scale_base_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                        "int64", token_offset * (scale_dim + 1)
                    )

                    for tile_id in T.serial(0, scale_dim):
                        vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                        fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                        local_amax = T.alloc_local((1,), dtype=T.float32)

                        local_amax[0] = 0.0
                        for vec in T.vectorized(vector_elems):
                            elem = elem_base + vec
                            vals[vec] = k_nope[token_id, tile_id * tile_dim + elem]
                            fvals[vec] = T.Cast("float32", vals[vec])
                            local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))

                        local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 8))
                        local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 4))
                        local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 2))
                        local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 1))

                        scale_byte, inv_scale = pow2_scale_byte_and_inv(local_amax[0] / 448.0)
                        if half_lane == 0:
                            scale_u32 = T.Cast("uint32", scale_byte)
                            if tile_id < 4:
                                scale_packed_lo[0] = scale_packed_lo[0] | (scale_u32 << T.Cast("uint32", tile_id * 8))
                            else:
                                scale_packed_hi[0] = scale_packed_hi[0] | (scale_u32 << T.Cast("uint32", (tile_id - 4) * 8))

                        tile_offset_i64 = token_base_i64 + T.Cast("int64", tile_id * tile_dim + elem_base)
                        packed = T.call_extern(
                            "uint32",
                            "tl_dsv4_pack_fp8x4_e4m3_u32",
                            T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                            T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                        )
                        T.stg32(cache_u8[page_idx_i64, tile_offset_i64], packed)

                    if half_lane == 0:
                        T.stg32(cache_u8[page_idx_i64, scale_base_i64], scale_packed_lo[0])
                        T.stg32(cache_u8[page_idx_i64, scale_base_i64 + T.Cast("int64", 4)], scale_packed_hi[0])

                    if rope_store_128:
                        if half_lane < 8:
                            rope_elem = half_lane * 8
                            rope_input_i64 = (
                                T.Cast("int64", input_base_offset)
                                + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                                + T.Cast("int64", nope_dim + rope_elem)
                            )
                            packed_rope = T.ldg128(
                                input_storage[
                                    rope_input_i64:
                                    rope_input_i64 + T.Cast("int64", 8)
                                ]
                            )
                            rope_offset_u32_i64 = (
                                token_base_i64 + T.Cast("int64", nope_dim + rope_elem * 2)
                            ) // T.Cast("int64", 4)
                            T.stg128(
                                cache_u32[
                                    page_idx_i64,
                                    rope_offset_u32_i64: rope_offset_u32_i64 + T.Cast("int64", 4),
                                ],
                                packed_rope,
                            )
                    else:
                        for rope_vec in T.vectorized(2):
                            rope_elem_pair = half_lane * 2 + rope_vec
                            rope_elem = rope_elem_pair * rope_pack_elems
                            lo = T.reinterpret("uint16", k_rope[token_id, rope_elem])
                            hi = T.reinterpret("uint16", k_rope[token_id, rope_elem + 1])
                            rope_offset_u32_i64 = (
                                token_base_i64 + T.Cast("int64", nope_dim)
                            ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", rope_elem_pair)
                            cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (
                                T.Cast("uint32", hi) << 16
                            )

    return pack_store_flashmla_cache_vector_x4_tile_fused_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_prefill_tile_parallel_kernel(
    input_dtype: str,
    page_bytes: int,
    page_size: int,
    tokens_per_cta: int = 4,
    compile_profile: str = "dsa_full",
    full_tiles: bool = False,
    rope_store_128: bool = False,
    explicit_unroll_scale_pack: bool = False,
):
    import tilelang
    import tilelang.language as T

    if input_dtype not in {"bfloat16", "float32"}:
        raise ValueError(f"Unsupported FlashMLA tile-parallel input dtype={input_dtype!r}")
    if tokens_per_cta not in {4, 8}:
        raise ValueError("FlashMLA tile-parallel path currently expects tokens_per_cta=4/8")

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    halfwarps_per_token = 16 // tokens_per_cta
    tile_rounds = (scale_dim + halfwarps_per_token - 1) // halfwarps_per_token
    threads_per_token = halfwarps_per_token * 16
    threads = tokens_per_cta * threads_per_token
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_flashmla_cache_pack_store_prefill_tile_parallel_{input_dtype}_"
            f"x{tokens_per_cta}_{compile_profile}_{'full' if full_tiles else 'tail'}_"
            f"rope{'128' if rope_store_128 else '32'}_"
            f"{'unroll_scale_pack' if explicit_unroll_scale_pack else 'serial_scale'}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            disable_index_promotion=True,
            compile_profile=compile_profile,
        ),
    )
    def pack_store_flashmla_cache_prefill_tile_parallel_kernel(
        input_storage: T.Tensor[(num_input_elements,), input_dtype],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
        input_base_offset: T.int64,
        input_row_stride: T.int64,
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_cta), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            token_group = tx // threads_per_token
            token_lane = tx - token_group * threads_per_token
            tile_half = token_lane // 16
            lane = token_lane - tile_half * 16
            elem_base = lane * vector_elems
            token_id = block_id * tokens_per_cta + token_group
            valid = full_tiles or token_id < num_tokens

            if valid:
                loc = indices[token_id]
                loc_valid = loc >= 0 and T.Cast("int64", loc) < T.Cast("int64", num_pages * page_size)
                if loc_valid:
                    page_idx, token_offset = page_index_and_offset(loc)
                    page_idx_i64 = T.Cast("int64", page_idx)
                    token_value_offset_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                    scale_base_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                        "int64", token_offset * (scale_dim + 1)
                    )
                    input_token_base_i64 = (
                        T.Cast("int64", input_base_offset)
                        + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                    )

                    if explicit_unroll_scale_pack and tokens_per_cta == 8:
                        # Split the 7 NoPE tiles into contiguous 4+3 groups so
                        # lane0 can pack scale bytes into two aligned ST.B32s.
                        if tile_half == 0:
                            vals_0 = T.alloc_local((vector_elems,), dtype=T.float32)
                            local_amax_0 = T.alloc_local((1,), dtype=T.float32)
                            input_tile_base_i64_0 = input_token_base_i64 + T.Cast(
                                "int64", 0 * tile_dim + elem_base
                            )
                            local_amax_0[0] = 0.0
                            for vec in T.vectorized(vector_elems):
                                vals_0[vec] = T.Cast(
                                    "float32",
                                    input_storage[input_tile_base_i64_0 + T.Cast("int64", vec)],
                                )
                                local_amax_0[0] = T.max(local_amax_0[0], abs_f32(vals_0[vec]))

                            local_amax_0[0] = T.max(local_amax_0[0], T.shfl_xor(local_amax_0[0], 8))
                            local_amax_0[0] = T.max(local_amax_0[0], T.shfl_xor(local_amax_0[0], 4))
                            local_amax_0[0] = T.max(local_amax_0[0], T.shfl_xor(local_amax_0[0], 2))
                            local_amax_0[0] = T.max(local_amax_0[0], T.shfl_xor(local_amax_0[0], 1))

                            scale_byte_0, inv_scale_0 = pow2_scale_byte_and_inv(local_amax_0[0] / 448.0)
                            tile_offset_i64_0 = token_value_offset_i64 + T.Cast(
                                "int64", 0 * tile_dim + elem_base
                            )
                            packed_0 = T.call_extern(
                                "uint32",
                                "tl_dsv4_pack_fp8x4_e4m3_u32",
                                T.clamp(vals_0[0] * inv_scale_0, -448.0, 448.0),
                                T.clamp(vals_0[1] * inv_scale_0, -448.0, 448.0),
                                T.clamp(vals_0[2] * inv_scale_0, -448.0, 448.0),
                                T.clamp(vals_0[3] * inv_scale_0, -448.0, 448.0),
                            )
                            T.stg32(cache_u8[page_idx_i64, tile_offset_i64_0], packed_0)
                            vals_1 = T.alloc_local((vector_elems,), dtype=T.float32)
                            local_amax_1 = T.alloc_local((1,), dtype=T.float32)
                            input_tile_base_i64_1 = input_token_base_i64 + T.Cast(
                                "int64", 1 * tile_dim + elem_base
                            )
                            local_amax_1[0] = 0.0
                            for vec in T.vectorized(vector_elems):
                                vals_1[vec] = T.Cast(
                                    "float32",
                                    input_storage[input_tile_base_i64_1 + T.Cast("int64", vec)],
                                )
                                local_amax_1[0] = T.max(local_amax_1[0], abs_f32(vals_1[vec]))

                            local_amax_1[0] = T.max(local_amax_1[0], T.shfl_xor(local_amax_1[0], 8))
                            local_amax_1[0] = T.max(local_amax_1[0], T.shfl_xor(local_amax_1[0], 4))
                            local_amax_1[0] = T.max(local_amax_1[0], T.shfl_xor(local_amax_1[0], 2))
                            local_amax_1[0] = T.max(local_amax_1[0], T.shfl_xor(local_amax_1[0], 1))

                            scale_byte_1, inv_scale_1 = pow2_scale_byte_and_inv(local_amax_1[0] / 448.0)
                            tile_offset_i64_1 = token_value_offset_i64 + T.Cast(
                                "int64", 1 * tile_dim + elem_base
                            )
                            packed_1 = T.call_extern(
                                "uint32",
                                "tl_dsv4_pack_fp8x4_e4m3_u32",
                                T.clamp(vals_1[0] * inv_scale_1, -448.0, 448.0),
                                T.clamp(vals_1[1] * inv_scale_1, -448.0, 448.0),
                                T.clamp(vals_1[2] * inv_scale_1, -448.0, 448.0),
                                T.clamp(vals_1[3] * inv_scale_1, -448.0, 448.0),
                            )
                            T.stg32(cache_u8[page_idx_i64, tile_offset_i64_1], packed_1)
                            vals_2 = T.alloc_local((vector_elems,), dtype=T.float32)
                            local_amax_2 = T.alloc_local((1,), dtype=T.float32)
                            input_tile_base_i64_2 = input_token_base_i64 + T.Cast(
                                "int64", 2 * tile_dim + elem_base
                            )
                            local_amax_2[0] = 0.0
                            for vec in T.vectorized(vector_elems):
                                vals_2[vec] = T.Cast(
                                    "float32",
                                    input_storage[input_tile_base_i64_2 + T.Cast("int64", vec)],
                                )
                                local_amax_2[0] = T.max(local_amax_2[0], abs_f32(vals_2[vec]))

                            local_amax_2[0] = T.max(local_amax_2[0], T.shfl_xor(local_amax_2[0], 8))
                            local_amax_2[0] = T.max(local_amax_2[0], T.shfl_xor(local_amax_2[0], 4))
                            local_amax_2[0] = T.max(local_amax_2[0], T.shfl_xor(local_amax_2[0], 2))
                            local_amax_2[0] = T.max(local_amax_2[0], T.shfl_xor(local_amax_2[0], 1))

                            scale_byte_2, inv_scale_2 = pow2_scale_byte_and_inv(local_amax_2[0] / 448.0)
                            tile_offset_i64_2 = token_value_offset_i64 + T.Cast(
                                "int64", 2 * tile_dim + elem_base
                            )
                            packed_2 = T.call_extern(
                                "uint32",
                                "tl_dsv4_pack_fp8x4_e4m3_u32",
                                T.clamp(vals_2[0] * inv_scale_2, -448.0, 448.0),
                                T.clamp(vals_2[1] * inv_scale_2, -448.0, 448.0),
                                T.clamp(vals_2[2] * inv_scale_2, -448.0, 448.0),
                                T.clamp(vals_2[3] * inv_scale_2, -448.0, 448.0),
                            )
                            T.stg32(cache_u8[page_idx_i64, tile_offset_i64_2], packed_2)
                            vals_3 = T.alloc_local((vector_elems,), dtype=T.float32)
                            local_amax_3 = T.alloc_local((1,), dtype=T.float32)
                            input_tile_base_i64_3 = input_token_base_i64 + T.Cast(
                                "int64", 3 * tile_dim + elem_base
                            )
                            local_amax_3[0] = 0.0
                            for vec in T.vectorized(vector_elems):
                                vals_3[vec] = T.Cast(
                                    "float32",
                                    input_storage[input_tile_base_i64_3 + T.Cast("int64", vec)],
                                )
                                local_amax_3[0] = T.max(local_amax_3[0], abs_f32(vals_3[vec]))

                            local_amax_3[0] = T.max(local_amax_3[0], T.shfl_xor(local_amax_3[0], 8))
                            local_amax_3[0] = T.max(local_amax_3[0], T.shfl_xor(local_amax_3[0], 4))
                            local_amax_3[0] = T.max(local_amax_3[0], T.shfl_xor(local_amax_3[0], 2))
                            local_amax_3[0] = T.max(local_amax_3[0], T.shfl_xor(local_amax_3[0], 1))

                            scale_byte_3, inv_scale_3 = pow2_scale_byte_and_inv(local_amax_3[0] / 448.0)
                            tile_offset_i64_3 = token_value_offset_i64 + T.Cast(
                                "int64", 3 * tile_dim + elem_base
                            )
                            packed_3 = T.call_extern(
                                "uint32",
                                "tl_dsv4_pack_fp8x4_e4m3_u32",
                                T.clamp(vals_3[0] * inv_scale_3, -448.0, 448.0),
                                T.clamp(vals_3[1] * inv_scale_3, -448.0, 448.0),
                                T.clamp(vals_3[2] * inv_scale_3, -448.0, 448.0),
                                T.clamp(vals_3[3] * inv_scale_3, -448.0, 448.0),
                            )
                            T.stg32(cache_u8[page_idx_i64, tile_offset_i64_3], packed_3)
                            if lane == 0:
                                scale_pack_0 = (
                                    T.Cast("uint32", scale_byte_0)
                                    | (T.Cast("uint32", scale_byte_1) << 8)
                                    | (T.Cast("uint32", scale_byte_2) << 16)
                                    | (T.Cast("uint32", scale_byte_3) << 24)
                                )
                                cache_u32[page_idx_i64, scale_base_i64 // T.Cast("int64", 4)] = scale_pack_0
                        else:
                            vals_4 = T.alloc_local((vector_elems,), dtype=T.float32)
                            local_amax_4 = T.alloc_local((1,), dtype=T.float32)
                            input_tile_base_i64_4 = input_token_base_i64 + T.Cast(
                                "int64", 4 * tile_dim + elem_base
                            )
                            local_amax_4[0] = 0.0
                            for vec in T.vectorized(vector_elems):
                                vals_4[vec] = T.Cast(
                                    "float32",
                                    input_storage[input_tile_base_i64_4 + T.Cast("int64", vec)],
                                )
                                local_amax_4[0] = T.max(local_amax_4[0], abs_f32(vals_4[vec]))

                            local_amax_4[0] = T.max(local_amax_4[0], T.shfl_xor(local_amax_4[0], 8))
                            local_amax_4[0] = T.max(local_amax_4[0], T.shfl_xor(local_amax_4[0], 4))
                            local_amax_4[0] = T.max(local_amax_4[0], T.shfl_xor(local_amax_4[0], 2))
                            local_amax_4[0] = T.max(local_amax_4[0], T.shfl_xor(local_amax_4[0], 1))

                            scale_byte_4, inv_scale_4 = pow2_scale_byte_and_inv(local_amax_4[0] / 448.0)
                            tile_offset_i64_4 = token_value_offset_i64 + T.Cast(
                                "int64", 4 * tile_dim + elem_base
                            )
                            packed_4 = T.call_extern(
                                "uint32",
                                "tl_dsv4_pack_fp8x4_e4m3_u32",
                                T.clamp(vals_4[0] * inv_scale_4, -448.0, 448.0),
                                T.clamp(vals_4[1] * inv_scale_4, -448.0, 448.0),
                                T.clamp(vals_4[2] * inv_scale_4, -448.0, 448.0),
                                T.clamp(vals_4[3] * inv_scale_4, -448.0, 448.0),
                            )
                            T.stg32(cache_u8[page_idx_i64, tile_offset_i64_4], packed_4)
                            vals_5 = T.alloc_local((vector_elems,), dtype=T.float32)
                            local_amax_5 = T.alloc_local((1,), dtype=T.float32)
                            input_tile_base_i64_5 = input_token_base_i64 + T.Cast(
                                "int64", 5 * tile_dim + elem_base
                            )
                            local_amax_5[0] = 0.0
                            for vec in T.vectorized(vector_elems):
                                vals_5[vec] = T.Cast(
                                    "float32",
                                    input_storage[input_tile_base_i64_5 + T.Cast("int64", vec)],
                                )
                                local_amax_5[0] = T.max(local_amax_5[0], abs_f32(vals_5[vec]))

                            local_amax_5[0] = T.max(local_amax_5[0], T.shfl_xor(local_amax_5[0], 8))
                            local_amax_5[0] = T.max(local_amax_5[0], T.shfl_xor(local_amax_5[0], 4))
                            local_amax_5[0] = T.max(local_amax_5[0], T.shfl_xor(local_amax_5[0], 2))
                            local_amax_5[0] = T.max(local_amax_5[0], T.shfl_xor(local_amax_5[0], 1))

                            scale_byte_5, inv_scale_5 = pow2_scale_byte_and_inv(local_amax_5[0] / 448.0)
                            tile_offset_i64_5 = token_value_offset_i64 + T.Cast(
                                "int64", 5 * tile_dim + elem_base
                            )
                            packed_5 = T.call_extern(
                                "uint32",
                                "tl_dsv4_pack_fp8x4_e4m3_u32",
                                T.clamp(vals_5[0] * inv_scale_5, -448.0, 448.0),
                                T.clamp(vals_5[1] * inv_scale_5, -448.0, 448.0),
                                T.clamp(vals_5[2] * inv_scale_5, -448.0, 448.0),
                                T.clamp(vals_5[3] * inv_scale_5, -448.0, 448.0),
                            )
                            T.stg32(cache_u8[page_idx_i64, tile_offset_i64_5], packed_5)
                            vals_6 = T.alloc_local((vector_elems,), dtype=T.float32)
                            local_amax_6 = T.alloc_local((1,), dtype=T.float32)
                            input_tile_base_i64_6 = input_token_base_i64 + T.Cast(
                                "int64", 6 * tile_dim + elem_base
                            )
                            local_amax_6[0] = 0.0
                            for vec in T.vectorized(vector_elems):
                                vals_6[vec] = T.Cast(
                                    "float32",
                                    input_storage[input_tile_base_i64_6 + T.Cast("int64", vec)],
                                )
                                local_amax_6[0] = T.max(local_amax_6[0], abs_f32(vals_6[vec]))

                            local_amax_6[0] = T.max(local_amax_6[0], T.shfl_xor(local_amax_6[0], 8))
                            local_amax_6[0] = T.max(local_amax_6[0], T.shfl_xor(local_amax_6[0], 4))
                            local_amax_6[0] = T.max(local_amax_6[0], T.shfl_xor(local_amax_6[0], 2))
                            local_amax_6[0] = T.max(local_amax_6[0], T.shfl_xor(local_amax_6[0], 1))

                            scale_byte_6, inv_scale_6 = pow2_scale_byte_and_inv(local_amax_6[0] / 448.0)
                            tile_offset_i64_6 = token_value_offset_i64 + T.Cast(
                                "int64", 6 * tile_dim + elem_base
                            )
                            packed_6 = T.call_extern(
                                "uint32",
                                "tl_dsv4_pack_fp8x4_e4m3_u32",
                                T.clamp(vals_6[0] * inv_scale_6, -448.0, 448.0),
                                T.clamp(vals_6[1] * inv_scale_6, -448.0, 448.0),
                                T.clamp(vals_6[2] * inv_scale_6, -448.0, 448.0),
                                T.clamp(vals_6[3] * inv_scale_6, -448.0, 448.0),
                            )
                            T.stg32(cache_u8[page_idx_i64, tile_offset_i64_6], packed_6)
                            if lane == 0:
                                scale_pack_1 = (
                                    T.Cast("uint32", scale_byte_4)
                                    | (T.Cast("uint32", scale_byte_5) << 8)
                                    | (T.Cast("uint32", scale_byte_6) << 16)
                                )
                                cache_u32[
                                    page_idx_i64,
                                    (scale_base_i64 + T.Cast("int64", 4)) // T.Cast("int64", 4),
                                ] = scale_pack_1
                    else:
                        for tile_round in T.serial(0, tile_rounds):
                            tile_id = tile_round * halfwarps_per_token + tile_half
                            if tile_id < scale_dim:
                                vals = T.alloc_local((vector_elems,), dtype=T.float32)
                                local_amax = T.alloc_local((1,), dtype=T.float32)
                                input_tile_base_i64 = input_token_base_i64 + T.Cast(
                                    "int64", tile_id * tile_dim + elem_base
                                )
                                local_amax[0] = 0.0
                                for vec in T.vectorized(vector_elems):
                                    vals[vec] = T.Cast(
                                        "float32",
                                        input_storage[input_tile_base_i64 + T.Cast("int64", vec)],
                                    )
                                    local_amax[0] = T.max(local_amax[0], abs_f32(vals[vec]))

                                local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 8))
                                local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 4))
                                local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 2))
                                local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 1))

                                scale_byte, inv_scale = pow2_scale_byte_and_inv(local_amax[0] / 448.0)
                                if lane == 0:
                                    cache_u8[
                                        page_idx_i64,
                                        scale_base_i64 + T.Cast("int64", tile_id),
                                    ] = scale_byte

                                tile_offset_i64 = token_value_offset_i64 + T.Cast(
                                    "int64", tile_id * tile_dim + elem_base
                                )
                                packed = T.call_extern(
                                    "uint32",
                                    "tl_dsv4_pack_fp8x4_e4m3_u32",
                                    T.clamp(vals[0] * inv_scale, -448.0, 448.0),
                                    T.clamp(vals[1] * inv_scale, -448.0, 448.0),
                                    T.clamp(vals[2] * inv_scale, -448.0, 448.0),
                                    T.clamp(vals[3] * inv_scale, -448.0, 448.0),
                                )
                                T.stg32(cache_u8[page_idx_i64, tile_offset_i64], packed)
                    if rope_store_128 and input_dtype == "bfloat16":
                        if tile_half < 2 and lane < 4:
                            rope_vec128 = tile_half * 4 + lane
                            rope_elem = rope_vec128 * 8
                            rope_input_i64 = input_token_base_i64 + T.Cast("int64", nope_dim + rope_elem)
                            packed_rope = T.ldg128(
                                input_storage[
                                    rope_input_i64:
                                    rope_input_i64 + T.Cast("int64", 8)
                                ]
                            )
                            rope_offset_u32_i64 = (
                                token_value_offset_i64 + T.Cast("int64", nope_dim + rope_elem * 2)
                            ) // T.Cast("int64", 4)
                            T.stg128(
                                cache_u32[
                                    page_idx_i64,
                                    rope_offset_u32_i64: rope_offset_u32_i64 + T.Cast("int64", 4),
                                ],
                                packed_rope,
                            )
                    elif tile_half < 2:
                        rope_elem_pair = tile_half * 16 + lane
                        rope_elem = rope_elem_pair * rope_pack_elems
                        rope_input_i64 = input_token_base_i64 + T.Cast("int64", nope_dim + rope_elem)
                        if input_dtype == "bfloat16":
                            lo = T.reinterpret("uint16", input_storage[rope_input_i64])
                            hi = T.reinterpret("uint16", input_storage[rope_input_i64 + T.Cast("int64", 1)])
                        else:
                            lo = T.reinterpret("uint16", T.Cast("bfloat16", input_storage[rope_input_i64]))
                            hi = T.reinterpret(
                                "uint16",
                                T.Cast("bfloat16", input_storage[rope_input_i64 + T.Cast("int64", 1)]),
                            )
                        rope_offset_u32_i64 = (
                            token_value_offset_i64 + T.Cast("int64", nope_dim)
                        ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", rope_elem_pair)
                        cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (
                            T.Cast("uint32", hi) << 16
                        )

    return pack_store_flashmla_cache_prefill_tile_parallel_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_c128_kernel(
    page_bytes: int,
    contiguous_indices: bool = False,
    pair_pages: bool = False,
    assume_even_base: bool = False,
    full_tiles: bool = False,
    dsa_compile_flags: str = "0",
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    page_size = 2
    token_stride_bytes = nope_dim + rope_dim * 2
    scale_page_base = page_size * token_stride_bytes
    rope_pack_elems = 2
    warps = 4
    tokens_per_half_warp = 1
    threads = warps * 32
    blk_m = warps * tokens_per_half_warp * 2
    input_row_stride = T.dynamic("input_row_stride")
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    variant = (
        "pair_even"
        if pair_pages and assume_even_base
        else ("pair" if pair_pages else ("contig" if contiguous_indices else "indexed"))
    )
    if dsa_compile_flags not in {"0", "dsa", "dsa_full"}:
        raise ValueError(f"Unsupported c128 dsa_compile_flags={dsa_compile_flags!r}")
    if dsa_compile_flags != "0":
        variant = f"{variant}_{dsa_compile_flags}"
    pass_configs = (
        _tilelang_musa_dsa_pass_configs(
            tilelang,
            full=(dsa_compile_flags == "dsa_full"),
            disable_index_promotion=True,
        )
        if dsa_compile_flags != "0"
        else _tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True)
    )

    @_tilelang_jit(
        tilelang,
        f"dsv4_flashmla_cache_pack_store_c128_bf16_{variant}_{'full' if full_tiles else 'tail'}",
        pass_configs=pass_configs,
    )
    def pack_store_flashmla_cache_c128_kernel(
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            half = lane // 16
            half_lane = lane % 16
            elem_base = half_lane * vector_elems

            token_id = block_id * blk_m + warp * 2 + half
            valid = full_tiles or token_id < num_tokens
            scale_packed_lo = T.alloc_local((1,), dtype=T.uint32)
            scale_packed_hi = T.alloc_local((1,), dtype=T.uint32)
            scale_packed_lo[0] = T.Cast("uint32", 0)
            scale_packed_hi[0] = T.Cast("uint32", 0)
            if valid:
                if pair_pages:
                    loc = indices[0] + token_id
                    page_idx_i64 = T.Cast("int64", loc >> 1)
                    if assume_even_base:
                        token_offset = half
                    else:
                        token_offset = loc & 1
                elif contiguous_indices:
                    loc = indices[0] + token_id
                    page_idx_i64 = T.Cast("int64", loc >> 1)
                    token_offset = loc & 1
                else:
                    loc = indices[token_id]
                    page_idx_i64 = T.Cast("int64", loc >> 1)
                    token_offset = loc & 1
                token_base_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                scale_base_i64 = T.Cast("int64", scale_page_base) + T.Cast(
                    "int64", token_offset * (scale_dim + 1)
                )

                for tile_id in T.serial(0, scale_dim):
                    vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                    fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                    local_amax = T.alloc_local((1,), dtype=T.float32)

                    local_amax[0] = 0.0
                    for vec in T.vectorized(vector_elems):
                        elem = elem_base + vec
                        vals[vec] = k_nope[token_id, tile_id * tile_dim + elem]
                        fvals[vec] = T.Cast("float32", vals[vec])
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))

                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 8))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 4))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 2))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 1))

                    scale_byte, inv_scale = pow2_scale_byte_and_inv(local_amax[0] / 448.0)
                    if half_lane == 0:
                        scale_u32 = T.Cast("uint32", scale_byte)
                        if tile_id < 4:
                            scale_packed_lo[0] = scale_packed_lo[0] | (scale_u32 << T.Cast("uint32", tile_id * 8))
                        else:
                            scale_packed_hi[0] = scale_packed_hi[0] | (scale_u32 << T.Cast("uint32", (tile_id - 4) * 8))

                    tile_offset_i64 = token_base_i64 + T.Cast("int64", tile_id * tile_dim + elem_base)
                    packed = T.call_extern(
                        "uint32",
                        "tl_dsv4_pack_fp8x4_e4m3_u32",
                        T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                    )
                    T.stg32(cache_u8[page_idx_i64, tile_offset_i64], packed)

                if half_lane == 0:
                    T.stg32(cache_u8[page_idx_i64, scale_base_i64], scale_packed_lo[0])
                    T.stg32(cache_u8[page_idx_i64, scale_base_i64 + T.Cast("int64", 4)], scale_packed_hi[0])

                for rope_vec in T.vectorized(2):
                    rope_elem_pair = half_lane * 2 + rope_vec
                    rope_elem = rope_elem_pair * rope_pack_elems
                    lo = T.reinterpret("uint16", k_rope[token_id, rope_elem])
                    hi = T.reinterpret("uint16", k_rope[token_id, rope_elem + 1])
                    rope_offset_u32_i64 = (
                        token_base_i64 + T.Cast("int64", nope_dim)
                    ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", rope_elem_pair)
                    cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (
                        T.Cast("uint32", hi) << 16
                    )

    return pack_store_flashmla_cache_c128_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_c128_flat_kernel(
    page_bytes: int,
    input_base_offset: int,
    contiguous_indices: bool = False,
    pair_pages: bool = False,
    assume_even_base: bool = False,
    rope_store_128: bool = False,
    packstore_variant: bool = False,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    page_size = 2
    token_stride_bytes = nope_dim + rope_dim * 2
    scale_page_base = page_size * token_stride_bytes
    rope_pack_elems = 2
    warps = 4
    tokens_per_half_warp = 1
    threads = warps * 32
    blk_m = warps * tokens_per_half_warp * 2
    input_row_stride = T.dynamic("input_row_stride")
    page_bytes_i64 = T.Cast("int64", page_bytes)
    page_u32_stride_i64 = T.Cast("int64", page_bytes // 4)
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    variant = (
        "flat_packstore"
        if packstore_variant
        else ("flat_rope128" if rope_store_128 else "flat")
    )
    if pair_pages:
        variant += "_pair_even" if assume_even_base else "_pair"
    elif contiguous_indices:
        variant += "_contig"
    else:
        variant += "_indexed"

    @_tilelang_jit(
        tilelang,
        f"dsv4_flashmla_cache_pack_store_c128_bf16_{variant}_{'full' if full_tiles else 'tail'}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_c128_flat_kernel(
        input_storage: T.Tensor[(num_input_elements,), T.bfloat16],
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8_flat: T.Tensor[(num_pages * page_bytes,), T.uint8],
        cache_u32_flat: T.Tensor[(num_pages * (page_bytes // 4),), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            half = lane // 16
            half_lane = lane % 16
            elem_base = half_lane * vector_elems

            token_id = block_id * blk_m + warp * 2 + half
            valid = full_tiles or token_id < num_tokens
            scale_packed_lo = T.alloc_local((1,), dtype=T.uint32)
            scale_packed_hi = T.alloc_local((1,), dtype=T.uint32)
            scale_packed_lo[0] = T.Cast("uint32", 0)
            scale_packed_hi[0] = T.Cast("uint32", 0)

            if valid:
                if pair_pages:
                    loc = indices[0] + token_id
                    page_idx = loc >> 1
                    if assume_even_base:
                        token_offset = half
                    else:
                        token_offset = loc & 1
                elif contiguous_indices:
                    loc = indices[0] + token_id
                    page_idx = loc >> 1
                    token_offset = loc & 1
                else:
                    loc = indices[token_id]
                    page_idx = loc >> 1
                    token_offset = loc & 1

                page_u8_base = T.Cast("int64", page_idx) * page_bytes_i64
                page_u32_base = T.Cast("int64", page_idx) * page_u32_stride_i64
                token_base_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                token_u8_base = page_u8_base + token_base_i64
                scale_u8_base = page_u8_base + T.Cast("int64", scale_page_base) + T.Cast(
                    "int64", token_offset * (scale_dim + 1)
                )
                rope_u32_base = page_u32_base + (
                    token_base_i64 + T.Cast("int64", nope_dim)
                ) // T.Cast("int64", 2 * rope_pack_elems)

                for tile_id in T.serial(0, scale_dim):
                    vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                    fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                    local_amax = T.alloc_local((1,), dtype=T.float32)

                    local_amax[0] = 0.0
                    for vec in T.vectorized(vector_elems):
                        elem = elem_base + vec
                        vals[vec] = k_nope[token_id, tile_id * tile_dim + elem]
                        fvals[vec] = T.Cast("float32", vals[vec])
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))

                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 8))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 4))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 2))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 1))

                    scale_byte, inv_scale = pow2_scale_byte_and_inv(local_amax[0] / 448.0)
                    if half_lane == 0:
                        scale_u32 = T.Cast("uint32", scale_byte)
                        if tile_id < 4:
                            scale_packed_lo[0] = scale_packed_lo[0] | (scale_u32 << T.Cast("uint32", tile_id * 8))
                        else:
                            scale_packed_hi[0] = scale_packed_hi[0] | (scale_u32 << T.Cast("uint32", (tile_id - 4) * 8))

                    tile_offset_i64 = T.Cast("int64", tile_id * tile_dim + elem_base)
                    packed = T.call_extern(
                        "uint32",
                        "tl_dsv4_pack_fp8x4_e4m3_u32",
                        T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                    )
                    T.stg32(cache_u8_flat[token_u8_base + tile_offset_i64], packed)

                if half_lane == 0:
                    T.stg32(cache_u8_flat[scale_u8_base], scale_packed_lo[0])
                    T.stg32(cache_u8_flat[scale_u8_base + T.Cast("int64", 4)], scale_packed_hi[0])

                if rope_store_128:
                    if half_lane < 8:
                        rope_elem = half_lane * 8
                        rope_input_i64 = (
                            T.Cast("int64", input_base_offset)
                            + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                            + T.Cast("int64", nope_dim + rope_elem)
                        )
                        packed_rope = T.ldg128(
                            input_storage[
                                rope_input_i64:
                                rope_input_i64 + T.Cast("int64", 8)
                            ]
                        )
                        rope_offset_u32_i64 = T.Cast("int64", rope_elem // 2)
                        T.stg128(
                            cache_u32_flat[
                                rope_u32_base + rope_offset_u32_i64:
                                rope_u32_base + rope_offset_u32_i64 + T.Cast("int64", 4),
                            ],
                            packed_rope,
                        )
                else:
                    for rope_vec in T.vectorized(2):
                        rope_elem_pair = half_lane * 2 + rope_vec
                        rope_elem = rope_elem_pair * rope_pack_elems
                        lo = T.reinterpret("uint16", k_rope[token_id, rope_elem])
                        hi = T.reinterpret("uint16", k_rope[token_id, rope_elem + 1])
                        cache_u32_flat[rope_u32_base + T.Cast("int64", rope_elem_pair)] = T.Cast("uint32", lo) | (
                            T.Cast("uint32", hi) << 16
                        )

    return pack_store_flashmla_cache_c128_flat_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_c128_page_tile_kernel(
    page_bytes: int,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    page_size = 2
    token_stride_bytes = nope_dim + rope_dim * 2
    scale_page_base = page_size * token_stride_bytes
    rope_pack_elems = 2
    warps = 4
    threads = warps * 32
    tokens_per_block = warps * page_size
    input_row_stride = T.dynamic("input_row_stride")
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    @_tilelang_jit(
        tilelang,
        f"dsv4_flashmla_cache_pack_store_c128_bf16_page_tile_{'full' if full_tiles else 'tail'}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_c128_page_tile_kernel(
        k_nope: T.StridedTensor[(num_tokens, nope_dim), (input_row_stride, 1), T.bfloat16],
        k_rope: T.StridedTensor[(num_tokens, rope_dim), (input_row_stride, 1), T.bfloat16],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_block), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            half = lane // 16
            half_lane = lane % 16
            elem_base = half_lane * vector_elems

            page_pair_id = block_id * warps + warp
            token_id = page_pair_id * page_size + half
            valid = full_tiles or token_id < num_tokens
            loc0 = indices[0]
            page_idx_i64 = T.Cast("int64", (loc0 >> 1) + page_pair_id)
            token_base_i64 = T.Cast("int64", half) * T.Cast("int64", token_stride_bytes)
            scale_base_i64 = T.Cast("int64", scale_page_base) + T.Cast("int64", half * (scale_dim + 1))
            rope_u32_base_i64 = (
                token_base_i64 + T.Cast("int64", nope_dim)
            ) // T.Cast("int64", 2 * rope_pack_elems)
            scale_packed_lo = T.alloc_local((1,), dtype=T.uint32)
            scale_packed_hi = T.alloc_local((1,), dtype=T.uint32)
            scale_packed_lo[0] = T.Cast("uint32", 0)
            scale_packed_hi[0] = T.Cast("uint32", 0)

            if valid:
                for tile_id in T.serial(0, scale_dim):
                    vals = T.alloc_local((vector_elems,), dtype=T.bfloat16)
                    fvals = T.alloc_local((vector_elems,), dtype=T.float32)
                    local_amax = T.alloc_local((1,), dtype=T.float32)

                    local_amax[0] = 0.0
                    for vec in T.vectorized(vector_elems):
                        elem = elem_base + vec
                        vals[vec] = k_nope[token_id, tile_id * tile_dim + elem]
                        fvals[vec] = T.Cast("float32", vals[vec])
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))

                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 8))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 4))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 2))
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 1))

                    scale_byte, inv_scale = pow2_scale_byte_and_inv(local_amax[0] / 448.0)
                    if half_lane == 0:
                        scale_u32 = T.Cast("uint32", scale_byte)
                        if tile_id < 4:
                            scale_packed_lo[0] = scale_packed_lo[0] | (scale_u32 << T.Cast("uint32", tile_id * 8))
                        else:
                            scale_packed_hi[0] = scale_packed_hi[0] | (scale_u32 << T.Cast("uint32", (tile_id - 4) * 8))

                    tile_offset_i64 = token_base_i64 + T.Cast("int64", tile_id * tile_dim + elem_base)
                    packed = T.call_extern(
                        "uint32",
                        "tl_dsv4_pack_fp8x4_e4m3_u32",
                        T.clamp(fvals[0] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[1] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[2] * inv_scale, -448.0, 448.0),
                        T.clamp(fvals[3] * inv_scale, -448.0, 448.0),
                    )
                    T.stg32(cache_u8[page_idx_i64, tile_offset_i64], packed)

                if half_lane == 0:
                    T.stg32(cache_u8[page_idx_i64, scale_base_i64], scale_packed_lo[0])
                    T.stg32(cache_u8[page_idx_i64, scale_base_i64 + T.Cast("int64", 4)], scale_packed_hi[0])

                for rope_vec in T.vectorized(2):
                    rope_elem_pair = half_lane * 2 + rope_vec
                    rope_elem = rope_elem_pair * rope_pack_elems
                    lo = T.reinterpret("uint16", k_rope[token_id, rope_elem])
                    hi = T.reinterpret("uint16", k_rope[token_id, rope_elem + 1])
                    cache_u32[page_idx_i64, rope_u32_base_i64 + T.Cast("int64", rope_elem_pair)] = T.Cast("uint32", lo) | (
                        T.Cast("uint32", hi) << 16
                    )

    return pack_store_flashmla_cache_c128_page_tile_kernel

@lru_cache(maxsize=None)
def _tilelang_store_flashmla_cache_kernel(
    page_bytes: int,
    page_size: int,
    threads: int = 128,
    grid_y: int = 512,
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    token_stride_bytes = nope_dim + rope_dim * 2
    warp_threads = 32
    warps = 8
    tokens_per_warp = 8
    blk_m = warps * tokens_per_warp

    @_tilelang_jit(tilelang, f"dsv4_flashmla_cache_store_{'full' if full_tiles else 'tail'}")
    def store_flashmla_cache_kernel(
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_bf16: T.Tensor[(num_pages, page_bytes // 2), T.bfloat16],
        indices: T.Tensor[(num_tokens,), T.int32],
        k_nope_fp8_u8: T.Tensor[(num_tokens, nope_dim), T.uint8],
        k_rope_bf16: T.Tensor[(num_tokens, rope_dim), T.bfloat16],
        scale_k_nope_ue8m0: T.Tensor[(num_tokens, scale_dim), T.uint8],
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), threads=warps * warp_threads) as block_id:
            tx = T.get_thread_binding()
            lane = tx % warp_threads
            warp = tx // warp_threads

            for token_iter in T.serial(0, tokens_per_warp):
                token_id = block_id * blk_m + warp * tokens_per_warp + token_iter
                if full_tiles or token_id < num_tokens:
                    loc = indices[token_id]
                    page_idx = loc // page_size
                    token_offset = loc % page_size
                    nope_offset = token_offset * token_stride_bytes
                    rope_offset_bf16 = (token_offset * token_stride_bytes + nope_dim) // 2
                    scale_offset = page_size * token_stride_bytes + token_offset * (scale_dim + 1)

                    for chunk in T.serial(0, nope_dim, warp_threads * 2):
                        col0 = chunk + lane
                        col1 = col0 + warp_threads
                        cache_u8[page_idx, nope_offset + col0] = k_nope_fp8_u8[token_id, col0]
                        cache_u8[page_idx, nope_offset + col1] = k_nope_fp8_u8[token_id, col1]

                    cache_bf16[page_idx, rope_offset_bf16 + lane] = k_rope_bf16[token_id, lane]
                    cache_bf16[page_idx, rope_offset_bf16 + lane + warp_threads] = k_rope_bf16[
                        token_id, lane + warp_threads
                    ]
                    if lane < scale_dim:
                        cache_u8[page_idx, scale_offset + lane] = scale_k_nope_ue8m0[token_id, lane]

    return store_flashmla_cache_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_indexer_cache_kernel(
    input_dtype: str,
    page_bytes: int,
    page_size: int,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    head_dim = 128
    scale_bytes = 4
    row_bytes = head_dim + scale_bytes
    blk_m = 16

    @_tilelang_jit(tilelang, "dsv4_indexer_cache_pack_store")
    def pack_store_indexer_cache_kernel(
        input_storage: T.Tensor[(num_input_elements,), input_dtype],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_fp8: T.Tensor[(num_pages, page_bytes), T.float8_e4m3fn],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
        input_base_offset: T.int64,
        input_row_stride: T.int64,
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), threads=threads) as block_id:
            values = T.alloc_fragment((blk_m, head_dim), dtype=input_dtype)
            amax = T.alloc_fragment((blk_m,), dtype=T.float32)
            loc = T.alloc_fragment((blk_m,), dtype=T.int32)
            page_idx = T.alloc_fragment((blk_m,), dtype=T.int32)
            token_offset = T.alloc_fragment((blk_m,), dtype=T.int32)
            scale = T.alloc_fragment((blk_m,), dtype=T.float32)
            inv_scale = T.alloc_fragment((blk_m,), dtype=T.float32)

            for row, elem in T.Parallel(blk_m, head_dim):
                token_id = block_id * blk_m + row
                if token_id < num_tokens:
                    input_offset_i64 = (
                        input_base_offset
                        + T.Cast("int64", token_id) * input_row_stride
                        + T.Cast("int64", elem)
                    )
                    values[row, elem] = input_storage[input_offset_i64]
                else:
                    values[row, elem] = 0.0
            T.reduce_absmax(values, amax, dim=1)

            for row in T.Parallel(blk_m):
                token_id = block_id * blk_m + row
                if token_id < num_tokens:
                    loc[row] = indices[token_id]
                    page_idx[row] = loc[row] // page_size
                    token_offset[row] = loc[row] % page_size
                    scale[row] = T.max(amax[row], 1.0e-4) / 448.0
                    inv_scale[row] = 1.0 / scale[row]

            for row, elem in T.Parallel(blk_m, head_dim):
                token_id = block_id * blk_m + row
                if token_id < num_tokens:
                    cache_fp8[
                        T.Cast("int64", page_idx[row]),
                        T.Cast("int64", token_offset[row]) * T.Cast("int64", row_bytes) + T.Cast("int64", elem),
                    ] = T.clamp(T.Cast("float32", values[row, elem]) * inv_scale[row], -448.0, 448.0)

            for row in T.Parallel(blk_m):
                token_id = block_id * blk_m + row
                if token_id < num_tokens:
                    cache_u32[
                        T.Cast("int64", page_idx[row]),
                        (
                            T.Cast("int64", token_offset[row]) * T.Cast("int64", row_bytes)
                            + T.Cast("int64", head_dim)
                        ) // T.Cast("int64", 4),
                    ] = T.reinterpret("uint32", scale[row])

    return pack_store_indexer_cache_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_indexer_cache_decode_x4_kernel(
    input_dtype: str,
    page_bytes: int,
    page_size: int,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    head_dim = 128
    scale_bytes = 4
    row_bytes = head_dim + scale_bytes
    threads = 128
    tokens_per_cta = 4
    values_per_lane = 4
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        "dsv4_indexer_cache_pack_store_decode_x4",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_indexer_cache_decode_x4_kernel(
        input_storage: T.Tensor[(num_input_elements,), input_dtype],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
        input_base_offset: T.int64,
        input_row_stride: T.int64,
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_cta), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            token_id = block_id * tokens_per_cta + warp
            elem_base = lane * values_per_lane
            valid = token_id < num_tokens

            vals = T.alloc_local((values_per_lane,), dtype=T.float32)
            local_amax = T.alloc_local((1,), dtype=T.float32)
            local_amax[0] = 0.0

            if valid:
                input_base_i64 = (
                    input_base_offset
                    + T.Cast("int64", token_id) * input_row_stride
                    + T.Cast("int64", elem_base)
                )
                for vec in T.vectorized(values_per_lane):
                    vals[vec] = T.Cast("float32", input_storage[input_base_i64 + T.Cast("int64", vec)])
                    local_amax[0] = T.max(local_amax[0], abs_f32(vals[vec]))

                local_amax[0] = T.warp_reduce_max(local_amax[0])

                loc = indices[token_id]
                loc_valid = loc >= 0 and T.Cast("int64", loc) < T.Cast("int64", num_pages * page_size)
                if loc_valid:
                    page_idx, token_offset = page_index_and_offset(loc)
                    page_idx_i64 = T.Cast("int64", page_idx)
                    token_byte_offset_i64 = T.Cast("int64", token_offset * row_bytes + elem_base)
                    token_scale_offset_u32_i64 = T.Cast("int64", token_offset * row_bytes + head_dim) // T.Cast(
                        "int64", 4
                    )

                    scale = T.max(local_amax[0], 1.0e-4) / 448.0
                    inv_scale = 1.0 / scale
                    packed = T.call_extern(
                        "uint32",
                        "tl_dsv4_pack_fp8x4_e4m3_u32",
                        T.clamp(vals[0] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[1] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[2] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[3] * inv_scale, -448.0, 448.0),
                    )
                    T.stg32(cache_u8[page_idx_i64, token_byte_offset_i64], packed)
                    if lane == 0:
                        cache_u32[
                            page_idx_i64,
                            token_scale_offset_u32_i64,
                        ] = T.reinterpret("uint32", scale)

    return pack_store_indexer_cache_decode_x4_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_flashmla_cache_prefill_subwarp16_kernel(
    input_dtype: str,
    page_bytes: int,
    page_size: int,
    tokens_per_cta: int = 8,
    compile_profile: str = "default",
    full_tiles: bool = False,
):
    import tilelang
    import tilelang.language as T

    if input_dtype not in {"bfloat16", "float32"}:
        raise ValueError(f"Unsupported FlashMLA prefill input dtype={input_dtype!r}")
    if tokens_per_cta not in {8, 16}:
        raise ValueError(f"Unsupported FlashMLA prefill tokens_per_cta={tokens_per_cta}")

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    nope_dim = 448
    rope_dim = 64
    scale_dim = 7
    tile_dim = 64
    vector_elems = 4
    token_stride_bytes = nope_dim + rope_dim * 2
    rope_pack_elems = 2
    threads_per_token = 16
    threads = tokens_per_cta * threads_per_token
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_flashmla_cache_pack_store_prefill_subwarp16_{input_dtype}_"
            f"x{tokens_per_cta}_{compile_profile}_{'full' if full_tiles else 'tail'}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_flashmla_cache_prefill_subwarp16_kernel(
        input_storage: T.Tensor[(num_input_elements,), input_dtype],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
        input_base_offset: T.int64,
        input_row_stride: T.int64,
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_cta), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            group = tx // threads_per_token
            lane = tx - group * threads_per_token
            token_id = block_id * tokens_per_cta + group
            valid = full_tiles or token_id < num_tokens

            if valid:
                loc = indices[token_id]
                loc_valid = loc >= 0 and T.Cast("int64", loc) < T.Cast("int64", num_pages * page_size)
                if loc_valid:
                    page_idx, token_offset = page_index_and_offset(loc)
                    page_idx_i64 = T.Cast("int64", page_idx)
                    token_value_offset_i64 = T.Cast("int64", token_offset) * T.Cast("int64", token_stride_bytes)
                    input_token_base_i64 = (
                        T.Cast("int64", input_base_offset)
                        + T.Cast("int64", token_id) * T.Cast("int64", input_row_stride)
                    )

                    for tile_id in T.serial(0, scale_dim):
                        vals = T.alloc_local((vector_elems,), dtype=T.float32)
                        local_amax = T.alloc_local((1,), dtype=T.float32)
                        elem_base = lane * vector_elems
                        input_tile_base_i64 = input_token_base_i64 + T.Cast("int64", tile_id * tile_dim + elem_base)
                        local_amax[0] = 0.0
                        for vec in T.vectorized(vector_elems):
                            vals[vec] = T.Cast("float32", input_storage[input_tile_base_i64 + T.Cast("int64", vec)])
                            local_amax[0] = T.max(local_amax[0], abs_f32(vals[vec]))

                        local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 8))
                        local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 4))
                        local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 2))
                        local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], 1))

                        scale_byte, inv_scale = pow2_scale_byte_and_inv(local_amax[0] / 448.0)
                        if lane == 0:
                            scale_offset_i64 = T.Cast("int64", page_size * token_stride_bytes) + T.Cast(
                                "int64", token_offset * (scale_dim + 1) + tile_id
                            )
                            cache_u8[page_idx_i64, scale_offset_i64] = scale_byte

                        tile_offset_i64 = token_value_offset_i64 + T.Cast("int64", tile_id * tile_dim + elem_base)
                        packed = T.call_extern(
                            "uint32",
                            "tl_dsv4_pack_fp8x4_e4m3_u32",
                            T.clamp(vals[0] * inv_scale, -448.0, 448.0),
                            T.clamp(vals[1] * inv_scale, -448.0, 448.0),
                            T.clamp(vals[2] * inv_scale, -448.0, 448.0),
                            T.clamp(vals[3] * inv_scale, -448.0, 448.0),
                        )
                        T.stg32(cache_u8[page_idx_i64, tile_offset_i64], packed)

                    for rope_vec in T.vectorized(2):
                        rope_elem_pair = lane * 2 + rope_vec
                        rope_elem = rope_elem_pair * rope_pack_elems
                        rope_input_i64 = input_token_base_i64 + T.Cast("int64", nope_dim + rope_elem)
                        if input_dtype == "bfloat16":
                            lo = T.reinterpret("uint16", input_storage[rope_input_i64])
                            hi = T.reinterpret("uint16", input_storage[rope_input_i64 + T.Cast("int64", 1)])
                        else:
                            lo = T.reinterpret("uint16", T.Cast("bfloat16", input_storage[rope_input_i64]))
                            hi = T.reinterpret(
                                "uint16",
                                T.Cast("bfloat16", input_storage[rope_input_i64 + T.Cast("int64", 1)]),
                            )
                        rope_offset_u32_i64 = (
                            token_value_offset_i64 + T.Cast("int64", nope_dim)
                        ) // T.Cast("int64", 2 * rope_pack_elems) + T.Cast("int64", rope_elem_pair)
                        cache_u32[page_idx_i64, rope_offset_u32_i64] = T.Cast("uint32", lo) | (
                            T.Cast("uint32", hi) << 16
                        )

    return pack_store_flashmla_cache_prefill_subwarp16_kernel

@lru_cache(maxsize=None)
def _tilelang_pack_store_indexer_cache_prefill_x8_kernel(
    input_dtype: str,
    page_bytes: int,
    page_size: int,
    tokens_per_cta: int = 8,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_input_elements = T.dynamic("num_input_elements")
    head_dim = 128
    scale_bytes = 4
    row_bytes = head_dim + scale_bytes
    threads = tokens_per_cta * 16
    values_per_lane = 8
    page_size_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
    page_size_shift = page_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = page_size - 1
    prelude = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/musa_fp8.h>
__device__ __forceinline__ uint32_t tl_dsv4_pack_fp8x4_e4m3_u32(float x0, float x1, float x2, float x3) {
  fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(make_float4(x0, x1, x2, x3));
  return *reinterpret_cast<uint32_t*>(&packed);
}
"""

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // page_size, loc % page_size

    @_tilelang_jit(
        tilelang,
        f"dsv4_indexer_cache_pack_store_prefill_x{tokens_per_cta}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=True),
    )
    def pack_store_indexer_cache_prefill_x8_kernel(
        input_storage: T.Tensor[(num_input_elements,), input_dtype],
        cache_u8: T.Tensor[(num_pages, page_bytes), T.uint8],
        cache_u32: T.Tensor[(num_pages, page_bytes // 4), T.uint32],
        indices: T.Tensor[(num_tokens,), T.int32],
        input_base_offset: T.int64,
        input_row_stride: T.int64,
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_cta), threads=threads, prelude=prelude) as block_id:
            tx = T.get_thread_binding()
            warp = tx // 32
            half_warp = (tx % 32) // 16
            lane = tx % 16
            token_id = block_id * tokens_per_cta + warp * 2 + half_warp
            elem_base = lane * values_per_lane
            valid = token_id < num_tokens

            vals = T.alloc_local((values_per_lane,), dtype=T.float32)
            local_amax = T.alloc_local((1,), dtype=T.float32)
            local_amax[0] = 0.0

            if valid:
                input_base_i64 = (
                    input_base_offset
                    + T.Cast("int64", token_id) * input_row_stride
                    + T.Cast("int64", elem_base)
                )
                for vec in T.vectorized(values_per_lane):
                    vals[vec] = T.Cast("float32", input_storage[input_base_i64 + T.Cast("int64", vec)])
                    local_amax[0] = T.max(local_amax[0], abs_f32(vals[vec]))

                for stage in T.unroll(4):
                    local_amax[0] = T.max(local_amax[0], T.shfl_xor(local_amax[0], T.int32(1) << stage))

                loc = indices[token_id]
                loc_valid = loc >= 0 and T.Cast("int64", loc) < T.Cast("int64", num_pages * page_size)
                if loc_valid:
                    page_idx, token_offset = page_index_and_offset(loc)
                    page_idx_i64 = T.Cast("int64", page_idx)
                    token_byte_offset_i64 = T.Cast("int64", token_offset * row_bytes + elem_base)
                    token_scale_offset_u32_i64 = T.Cast("int64", token_offset * row_bytes + head_dim) // T.Cast(
                        "int64", 4
                    )

                    scale = T.max(local_amax[0], 1.0e-4) / 448.0
                    inv_scale = 1.0 / scale
                    packed0 = T.call_extern(
                        "uint32",
                        "tl_dsv4_pack_fp8x4_e4m3_u32",
                        T.clamp(vals[0] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[1] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[2] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[3] * inv_scale, -448.0, 448.0),
                    )
                    packed1 = T.call_extern(
                        "uint32",
                        "tl_dsv4_pack_fp8x4_e4m3_u32",
                        T.clamp(vals[4] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[5] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[6] * inv_scale, -448.0, 448.0),
                        T.clamp(vals[7] * inv_scale, -448.0, 448.0),
                    )
                    T.stg32(cache_u8[page_idx_i64, token_byte_offset_i64], packed0)
                    T.stg32(cache_u8[page_idx_i64, token_byte_offset_i64 + 4], packed1)
                    if lane == 0:
                        cache_u32[
                            page_idx_i64,
                            token_scale_offset_u32_i64,
                        ] = T.reinterpret("uint32", scale)

    return pack_store_indexer_cache_prefill_x8_kernel

__all__ = [
    '_tilelang_pack_store_flashmla_cache_nv_block_kernel',
    '_tilelang_pack_store_flashmla_cache_decode_x4_kernel',
    '_tilelang_pack_store_flashmla_cache_decode_x4_fp32_kernel',
    '_tilelang_pack_store_flashmla_cache_decode_vec2_kernel',
    '_tilelang_pack_store_flashmla_cache_decode_x4_flat_i32_kernel',
    '_tilelang_pack_store_flashmla_cache_warp_token_kernel',
    '_tilelang_pack_store_flashmla_cache_warp_col_fused_kernel',
    '_tilelang_pack_store_flashmla_cache_kernel',
    '_tilelang_pack_store_indexer_cache_kernel',
    '_tilelang_pack_store_indexer_cache_decode_x4_kernel',
    '_tilelang_pack_store_indexer_cache_prefill_x8_kernel',
    '_tilelang_pack_store_flashmla_cache_warp_col_kernel',
    '_tilelang_pack_store_flashmla_cache_warp_col_split_kernel',
    '_tilelang_pack_store_flashmla_cache_c128_kernel',
    '_tilelang_pack_store_flashmla_cache_c128_flat_kernel',
    '_tilelang_pack_store_flashmla_cache_c128_page_tile_kernel',
    '_tilelang_pack_store_flashmla_cache_vector_x4_remap_kernel',
    '_tilelang_pack_store_flashmla_cache_vector_x4_tile_fused_kernel',
    '_tilelang_pack_store_flashmla_cache_prefill_tile_parallel_kernel',
    '_tilelang_pack_store_flashmla_cache_prefill_subwarp16_kernel',
    '_tilelang_pack_store_flashmla_cache_vector_kernel',
    '_tilelang_store_flashmla_cache_kernel',
]
