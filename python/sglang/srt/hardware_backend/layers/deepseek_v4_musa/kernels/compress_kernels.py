from functools import lru_cache

import torch

from .kernel_common import SELECT_TOPK_BITSET_WORDS, _tilelang_jit, _tilelang_musa_aggressive_pass_configs, _tilelang_musa_burst_reduce_pass_configs

_COMPRESS_DECODE_COMPILE_PROFILE = "ls"
_COMPRESS_C128_PREFILL_COMPILE_PROFILE = "ls"

def _compress_c4_cached_reduce_pass_profile(optimization_profile: str) -> str:
    profile = optimization_profile.strip().lower()
    if profile in {"longseq", "long_seq", "c4_longseq", "c4_long_seq"}:
        return "dsa"
    return profile

@lru_cache(maxsize=None)
def _tilelang_compress_forward_ratio4_decode_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.dynamic("num_blocks")
    col_blocks = (head_dim + threads - 1) // threads
    kv_score_input_stride = T.dynamic("kv_score_input_stride")
    ape_stride = T.dynamic("ape_stride")
    indices_stride = T.dynamic("indices_stride")
    seq_lens_stride = T.dynamic("seq_lens_stride")
    out_stride = T.dynamic("out_stride")

    @_tilelang_jit(
        tilelang,
        f"dsv4_c4_decode_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_DECODE_COMPILE_PROFILE,
        ),
    )
    def compress_forward_ratio4_decode_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 8, head_dim * 4), T.float32],
        kv_score_input: T.StridedTensor[(num_tokens, head_dim * 4), (kv_score_input_stride, 1), T.float32],
        ape: T.StridedTensor[(8, head_dim), (ape_stride, 1), T.float32],
        indices: T.StridedTensor[(num_tokens,), (indices_stride,), T.int32],
        seq_lens: T.StridedTensor[(num_tokens,), (seq_lens_stride,), T.int32],
        out: T.StridedTensor[(num_tokens, head_dim), (out_stride, 1), T.float32],
    ):
        with T.Kernel(num_tokens, col_blocks, threads=threads) as (token_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                block_id = indices[token_id]
                block_id_i64 = T.Cast("int64", block_id)
                seq_len = seq_lens[token_id]
                write_pos = (seq_len + 7) % 8
                kv_score_buffer[block_id_i64, write_pos, col] = kv_score_input[token_id, col]
                kv_score_buffer[block_id_i64, write_pos, head_dim + col] = kv_score_input[token_id, head_dim + col]
                kv_score_buffer[block_id_i64, write_pos, head_dim * 2 + col] = kv_score_input[token_id, head_dim * 2 + col]
                kv_score_buffer[block_id_i64, write_pos, head_dim * 3 + col] = kv_score_input[token_id, head_dim * 3 + col]
                out[token_id, col] = 0.0
                if seq_len % 4 == 0:
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    max_logit[0] = -3.4028234663852886e38
                    for slot in T.serial(0, 4):
                        source_slot = (seq_len + slot) % 8
                        # seq_len == 4: first compression window (cold-start).
                        # History buffer has no valid scores yet, so we use a sentinel.
                        if seq_len == 4:
                            max_logit[0] = T.max(max_logit[0], -1.0e9 + ape[slot, col])
                        else:
                            max_logit[0] = T.max(
                                max_logit[0],
                                kv_score_buffer[block_id_i64, source_slot, head_dim * 2 + col] + ape[slot, col],
                            )
                    for slot in T.serial(4, 8):
                        source_slot = (seq_len + slot) % 8
                        max_logit[0] = T.max(
                            max_logit[0],
                            kv_score_buffer[block_id_i64, source_slot, head_dim * 3 + col] + ape[slot, col],
                        )
                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 4):
                        source_slot = (seq_len + slot) % 8
                        # seq_len == 4: first compression window (cold-start).
                        if seq_len == 4:
                            weight[0] = T.exp(-1.0e9 + ape[slot, col] - max_logit[0])
                            denom[0] += weight[0]
                        else:
                            weight[0] = T.exp(
                                kv_score_buffer[block_id_i64, source_slot, head_dim * 2 + col]
                                + ape[slot, col]
                                - max_logit[0]
                            )
                            denom[0] += weight[0]
                            acc[0] += kv_score_buffer[block_id_i64, source_slot, col] * weight[0]
                    for slot in T.serial(4, 8):
                        source_slot = (seq_len + slot) % 8
                        weight[0] = T.exp(
                            kv_score_buffer[block_id_i64, source_slot, head_dim * 3 + col]
                            + ape[slot, col]
                            - max_logit[0]
                        )
                        denom[0] += weight[0]
                        acc[0] += kv_score_buffer[block_id_i64, source_slot, head_dim + col] * weight[0]
                    out[token_id, col] = acc[0] / denom[0]

    return compress_forward_ratio4_decode_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_forward_ratio4_decode_page_kernel(head_dim: int, extra_data_cols: int = 4, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.dynamic("num_blocks")
    num_indices = T.dynamic("num_indices")
    col_blocks = (head_dim + threads - 1) // threads
    kv_score_input_stride = T.dynamic("kv_score_input_stride")
    ape_stride = T.dynamic("ape_stride")
    indices_stride = T.dynamic("indices_stride")
    seq_lens_stride = T.dynamic("seq_lens_stride")
    extra_data_row_stride = T.dynamic("extra_data_row_stride")
    extra_data_col_stride = T.dynamic("extra_data_col_stride")
    out_stride = T.dynamic("out_stride")

    @_tilelang_jit(
        tilelang,
        f"dsv4_c4_decode_page4_c{extra_data_cols}_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_DECODE_COMPILE_PROFILE,
        ),
    )
    def compress_forward_ratio4_decode_page_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 4, head_dim * 4), T.float32],
        kv_score_input: T.StridedTensor[(num_tokens, head_dim * 4), (kv_score_input_stride, 1), T.float32],
        ape: T.StridedTensor[(8, head_dim), (ape_stride, 1), T.float32],
        indices: T.StridedTensor[(num_tokens,), (indices_stride,), T.int32],
        seq_lens: T.StridedTensor[(num_tokens,), (seq_lens_stride,), T.int32],
        extra_data: T.StridedTensor[(num_indices, extra_data_cols), (extra_data_row_stride, extra_data_col_stride), T.int32],
        out: T.StridedTensor[(num_tokens, head_dim), (out_stride, 1), T.float32],
    ):
        with T.Kernel(num_tokens, col_blocks, threads=threads) as (token_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                block_id = indices[token_id]
                block_id_i64 = T.Cast("int64", block_id)
                seq_len = seq_lens[token_id]
                write_pos = (seq_len + 3) % 4
                kv_score_buffer[block_id_i64, write_pos, col] = kv_score_input[token_id, col]
                kv_score_buffer[block_id_i64, write_pos, head_dim + col] = kv_score_input[token_id, head_dim + col]
                kv_score_buffer[block_id_i64, write_pos, head_dim * 2 + col] = kv_score_input[token_id, head_dim * 2 + col]
                kv_score_buffer[block_id_i64, write_pos, head_dim * 3 + col] = kv_score_input[token_id, head_dim * 3 + col]
                out[token_id, col] = 0.0
                if seq_len % 4 == 0:
                    overlap_block = extra_data[token_id, 0]
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    overlap_block_i64 = T.Cast("int64", overlap_block)
                    max_logit[0] = -3.4028234663852886e38
                    for slot in T.serial(0, 4):
                        if seq_len == 4:
                            max_logit[0] = T.max(max_logit[0], -1.0e9 + ape[slot, col])
                        else:
                            max_logit[0] = T.max(
                                max_logit[0],
                                kv_score_buffer[overlap_block_i64, slot, head_dim * 2 + col] + ape[slot, col],
                            )
                    for slot in T.serial(4, 8):
                        source_slot = slot - 4
                        max_logit[0] = T.max(
                            max_logit[0],
                            kv_score_buffer[block_id_i64, source_slot, head_dim * 3 + col] + ape[slot, col],
                        )
                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 4):
                        if seq_len == 4:
                            weight[0] = T.exp(-1.0e9 + ape[slot, col] - max_logit[0])
                            denom[0] += weight[0]
                        else:
                            weight[0] = T.exp(
                                kv_score_buffer[overlap_block_i64, slot, head_dim * 2 + col]
                                + ape[slot, col]
                                - max_logit[0]
                            )
                            denom[0] += weight[0]
                            acc[0] += kv_score_buffer[overlap_block_i64, slot, col] * weight[0]
                    for slot in T.serial(4, 8):
                        source_slot = slot - 4
                        weight[0] = T.exp(
                            kv_score_buffer[block_id_i64, source_slot, head_dim * 3 + col]
                            + ape[slot, col]
                            - max_logit[0]
                        )
                        denom[0] += weight[0]
                        acc[0] += kv_score_buffer[block_id_i64, source_slot, head_dim + col] * weight[0]
                    out[token_id, col] = acc[0] / denom[0]

    return compress_forward_ratio4_decode_page_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_forward_ratio4_decode_flat_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_slots = T.dynamic("num_slots")
    col_blocks = (head_dim + threads - 1) // threads
    kv_score_input_stride = T.dynamic("kv_score_input_stride")
    ape_stride = T.dynamic("ape_stride")
    indices_stride = T.dynamic("indices_stride")
    seq_lens_stride = T.dynamic("seq_lens_stride")
    out_stride = T.dynamic("out_stride")

    @_tilelang_jit(
        tilelang,
        f"dsv4_c4_decode_flat_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_DECODE_COMPILE_PROFILE,
        ),
    )
    def compress_forward_ratio4_decode_flat_kernel(
        kv_score_buffer: T.Tensor[(num_slots, head_dim * 4), T.float32],
        kv_score_input: T.StridedTensor[(num_tokens, head_dim * 4), (kv_score_input_stride, 1), T.float32],
        ape: T.StridedTensor[(8, head_dim), (ape_stride, 1), T.float32],
        indices: T.StridedTensor[(num_tokens,), (indices_stride,), T.int32],
        seq_lens: T.StridedTensor[(num_tokens,), (seq_lens_stride,), T.int32],
        out: T.StridedTensor[(num_tokens, head_dim), (out_stride, 1), T.float32],
    ):
        with T.Kernel(num_tokens, col_blocks, threads=threads) as (token_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                block_id = indices[token_id]
                block_id_i64 = T.Cast("int64", block_id)
                seq_len = seq_lens[token_id]
                write_pos = (seq_len + 7) % 8
                write_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", write_pos)
                kv_score_buffer[write_row, col] = kv_score_input[token_id, col]
                kv_score_buffer[write_row, head_dim + col] = kv_score_input[token_id, head_dim + col]
                kv_score_buffer[write_row, head_dim * 2 + col] = kv_score_input[token_id, head_dim * 2 + col]
                kv_score_buffer[write_row, head_dim * 3 + col] = kv_score_input[token_id, head_dim * 3 + col]
                out[token_id, col] = 0.0
                if seq_len % 4 == 0:
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    max_logit[0] = -3.4028234663852886e38
                    for slot in T.serial(0, 4):
                        source_slot = (seq_len + slot) % 8
                        source_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", source_slot)
                        # seq_len == 4: first compression window (cold-start).
                        if seq_len == 4:
                            max_logit[0] = T.max(max_logit[0], -1.0e9 + ape[slot, col])
                        else:
                            max_logit[0] = T.max(
                                max_logit[0],
                                kv_score_buffer[source_row, head_dim * 2 + col] + ape[slot, col],
                            )
                    for slot in T.serial(4, 8):
                        source_slot = (seq_len + slot) % 8
                        source_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", source_slot)
                        max_logit[0] = T.max(
                            max_logit[0],
                            kv_score_buffer[source_row, head_dim * 3 + col] + ape[slot, col],
                        )
                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 4):
                        source_slot = (seq_len + slot) % 8
                        source_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", source_slot)
                        # seq_len == 4: first compression window (cold-start).
                        if seq_len == 4:
                            weight[0] = T.exp(-1.0e9 + ape[slot, col] - max_logit[0])
                            denom[0] += weight[0]
                        else:
                            weight[0] = T.exp(
                                kv_score_buffer[source_row, head_dim * 2 + col]
                                + ape[slot, col]
                                - max_logit[0]
                            )
                            denom[0] += weight[0]
                            acc[0] += kv_score_buffer[source_row, col] * weight[0]
                    for slot in T.serial(4, 8):
                        source_slot = (seq_len + slot) % 8
                        source_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", source_slot)
                        weight[0] = T.exp(
                            kv_score_buffer[source_row, head_dim * 3 + col]
                            + ape[slot, col]
                            - max_logit[0]
                        )
                        denom[0] += weight[0]
                        acc[0] += kv_score_buffer[source_row, head_dim + col] * weight[0]
                    out[token_id, col] = acc[0] / denom[0]

    return compress_forward_ratio4_decode_flat_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_forward_ratio128_decode_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.dynamic("num_blocks")
    col_blocks = (head_dim + threads - 1) // threads
    kv_score_input_stride = T.dynamic("kv_score_input_stride")
    ape_stride = T.dynamic("ape_stride")
    indices_stride = T.dynamic("indices_stride")
    seq_lens_stride = T.dynamic("seq_lens_stride")
    out_stride = T.dynamic("out_stride")

    @_tilelang_jit(
        tilelang,
        f"dsv4_c128_decode_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_DECODE_COMPILE_PROFILE,
        ),
    )
    def compress_forward_ratio128_decode_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 128, head_dim * 2), T.float32],
        kv_score_input: T.StridedTensor[(num_tokens, head_dim * 2), (kv_score_input_stride, 1), T.float32],
        ape: T.StridedTensor[(128, head_dim), (ape_stride, 1), T.float32],
        indices: T.StridedTensor[(num_tokens,), (indices_stride,), T.int32],
        seq_lens: T.StridedTensor[(num_tokens,), (seq_lens_stride,), T.int32],
        out: T.StridedTensor[(num_tokens, head_dim), (out_stride, 1), T.float32],
    ):
        with T.Kernel(num_tokens, col_blocks, threads=threads) as (token_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                block_id = indices[token_id]
                block_id_i64 = T.Cast("int64", block_id)
                seq_len = seq_lens[token_id]
                write_pos = (seq_len + 127) % 128
                kv_score_buffer[block_id_i64, write_pos, col] = kv_score_input[token_id, col]
                kv_score_buffer[block_id_i64, write_pos, head_dim + col] = kv_score_input[token_id, head_dim + col]
                out[token_id, col] = 0.0
                if seq_len % 128 == 0:
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    max_logit[0] = -3.4028234663852886e38
                    for slot in T.serial(0, 128):
                        logit = kv_score_buffer[block_id_i64, slot, head_dim + col] + ape[slot, col]
                        max_logit[0] = T.max(max_logit[0], logit)
                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 128):
                        logit = kv_score_buffer[block_id_i64, slot, head_dim + col] + ape[slot, col]
                        weight[0] = T.exp(logit - max_logit[0])
                        denom[0] += weight[0]
                        acc[0] += kv_score_buffer[block_id_i64, slot, col] * weight[0]
                    out[token_id, col] = acc[0] / denom[0]

    return compress_forward_ratio128_decode_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_forward_ratio128_decode_flat_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_slots = T.dynamic("num_slots")
    col_blocks = (head_dim + threads - 1) // threads
    kv_score_input_stride = T.dynamic("kv_score_input_stride")
    ape_stride = T.dynamic("ape_stride")
    indices_stride = T.dynamic("indices_stride")
    seq_lens_stride = T.dynamic("seq_lens_stride")
    out_stride = T.dynamic("out_stride")

    @_tilelang_jit(
        tilelang,
        f"dsv4_c128_decode_flat_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_DECODE_COMPILE_PROFILE,
        ),
    )
    def compress_forward_ratio128_decode_flat_kernel(
        kv_score_buffer: T.Tensor[(num_slots, head_dim * 2), T.float32],
        kv_score_input: T.StridedTensor[(num_tokens, head_dim * 2), (kv_score_input_stride, 1), T.float32],
        ape: T.StridedTensor[(128, head_dim), (ape_stride, 1), T.float32],
        indices: T.StridedTensor[(num_tokens,), (indices_stride,), T.int32],
        seq_lens: T.StridedTensor[(num_tokens,), (seq_lens_stride,), T.int32],
        out: T.StridedTensor[(num_tokens, head_dim), (out_stride, 1), T.float32],
    ):
        with T.Kernel(num_tokens, col_blocks, threads=threads) as (token_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                block_id = indices[token_id]
                block_id_i64 = T.Cast("int64", block_id)
                seq_len = seq_lens[token_id]
                write_pos = (seq_len + 127) % 128
                write_row = block_id_i64 * T.Cast("int64", 128) + T.Cast("int64", write_pos)
                kv_score_buffer[write_row, col] = kv_score_input[token_id, col]
                kv_score_buffer[write_row, head_dim + col] = kv_score_input[token_id, head_dim + col]
                out[token_id, col] = 0.0
                if seq_len % 128 == 0:
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    max_logit[0] = -3.4028234663852886e38
                    for slot in T.serial(0, 128):
                        source_slot = (seq_len + slot) % 128
                        source_row = block_id_i64 * T.Cast("int64", 128) + T.Cast("int64", source_slot)
                        logit = kv_score_buffer[source_row, head_dim + col] + ape[slot, col]
                        max_logit[0] = T.max(max_logit[0], logit)
                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 128):
                        source_slot = (seq_len + slot) % 128
                        source_row = block_id_i64 * T.Cast("int64", 128) + T.Cast("int64", source_slot)
                        logit = kv_score_buffer[source_row, head_dim + col] + ape[slot, col]
                        weight[0] = T.exp(logit - max_logit[0])
                        denom[0] += weight[0]
                        acc[0] += kv_score_buffer[source_row, col] * weight[0]
                    out[token_id, col] = acc[0] / denom[0]

    return compress_forward_ratio128_decode_flat_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_forward_ratio128_decode_flat_parallel_kernel(head_dim: int, final_merge: str = "shared"):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_slots = T.dynamic("num_slots")
    tile_cols = 64
    slots_per_warp = 8
    num_warps = 16
    threads = 512
    col_blocks = (head_dim + tile_cols - 1) // tile_cols
    kv_score_input_stride = T.dynamic("kv_score_input_stride")
    ape_stride = T.dynamic("ape_stride")
    indices_stride = T.dynamic("indices_stride")
    seq_lens_stride = T.dynamic("seq_lens_stride")
    out_stride = T.dynamic("out_stride")

    if final_merge not in ("shared", "warp"):
        raise ValueError(f"unsupported c128 decode final_merge={final_merge!r}")

    @_tilelang_jit(
        tilelang,
        f"dsv4_c128_decode_flat_parallel_{final_merge}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_DECODE_COMPILE_PROFILE,
        ),
    )
    def compress_forward_ratio128_decode_flat_parallel_kernel(
        kv_score_buffer: T.Tensor[(num_slots, head_dim * 2), T.float32],
        kv_score_input: T.StridedTensor[(num_tokens, head_dim * 2), (kv_score_input_stride, 1), T.float32],
        ape: T.StridedTensor[(128, head_dim), (ape_stride, 1), T.float32],
        indices: T.StridedTensor[(num_tokens,), (indices_stride,), T.int32],
        seq_lens: T.StridedTensor[(num_tokens,), (seq_lens_stride,), T.int32],
        out: T.StridedTensor[(num_tokens, head_dim), (out_stride, 1), T.float32],
    ):
        with T.Kernel(num_tokens, col_blocks, threads=threads) as (token_id, col_block):
            tx = T.get_thread_binding()
            warp_id = tx // 32
            lane_id = tx - warp_id * 32
            col_base = col_block * tile_cols + lane_id * 2
            block_id = indices[token_id]
            block_id_i64 = T.Cast("int64", block_id)
            seq_len = seq_lens[token_id]

            if warp_id == num_warps - 1:
                write_pos = (seq_len + 127) % 128
                write_row = block_id_i64 * T.Cast("int64", 128) + T.Cast("int64", write_pos)
                for elem in T.serial(0, 2):
                    col = col_base + elem
                    if col < head_dim:
                        kv_score_buffer[write_row, col] = kv_score_input[token_id, col]
                        kv_score_buffer[write_row, head_dim + col] = kv_score_input[token_id, head_dim + col]

            local_max = T.alloc_local((2,), dtype=T.float32)
            local_sum = T.alloc_local((2,), dtype=T.float32)
            local_prod = T.alloc_local((2,), dtype=T.float32)
            tmp_weight = T.alloc_local((1,), dtype=T.float32)
            smem_max = T.alloc_shared((16, 32, 2), dtype=T.float32)
            smem_sum = T.alloc_shared((16, 32, 2), dtype=T.float32)
            smem_prod = T.alloc_shared((16, 32, 2), dtype=T.float32)

            for elem in T.serial(0, 2):
                local_max[elem] = -3.4028234663852886e38
                local_sum[elem] = 0.0
                local_prod[elem] = 0.0

            if seq_len % 128 == 0:
                for slot_i in T.serial(0, slots_per_warp):
                    slot = warp_id * slots_per_warp + slot_i
                    source_slot = (seq_len + slot) % 128
                    source_row = block_id_i64 * T.Cast("int64", 128) + T.Cast("int64", source_slot)
                    for elem in T.serial(0, 2):
                        col = col_base + elem
                        if col < head_dim:
                            local_max[elem] = T.max(
                                local_max[elem],
                                kv_score_buffer[source_row, head_dim + col] + ape[slot, col],
                            )

                for slot_i in T.serial(0, slots_per_warp):
                    slot = warp_id * slots_per_warp + slot_i
                    source_slot = (seq_len + slot) % 128
                    source_row = block_id_i64 * T.Cast("int64", 128) + T.Cast("int64", source_slot)
                    for elem in T.serial(0, 2):
                        col = col_base + elem
                        if col < head_dim:
                            tmp_weight[0] = T.exp(kv_score_buffer[source_row, head_dim + col] + ape[slot, col] - local_max[elem])
                            local_sum[elem] += tmp_weight[0]
                            local_prod[elem] += kv_score_buffer[source_row, col] * tmp_weight[0]

                for elem in T.serial(0, 2):
                    smem_max[warp_id, lane_id, elem] = local_max[elem]
                    smem_sum[warp_id, lane_id, elem] = local_sum[elem]
                    smem_prod[warp_id, lane_id, elem] = local_prod[elem]
                T.sync_threads()

                if final_merge == "warp":
                    for reduce_group in T.serial(0, 4):
                        reduce_col_in_tile = reduce_group * num_warps + warp_id
                        out_lane = reduce_col_in_tile // 2
                        out_elem = reduce_col_in_tile - out_lane * 2
                        col = col_block * tile_cols + reduce_col_in_tile
                        partial_max = T.alloc_local((1,), dtype=T.float32)
                        partial_sum = T.alloc_local((1,), dtype=T.float32)
                        partial_prod = T.alloc_local((1,), dtype=T.float32)
                        global_max = T.alloc_local((1,), dtype=T.float32)
                        global_sum = T.alloc_local((1,), dtype=T.float32)
                        global_prod = T.alloc_local((1,), dtype=T.float32)
                        rescale = T.alloc_local((1,), dtype=T.float32)
                        partial_max[0] = -3.4028234663852886e38
                        partial_sum[0] = 0.0
                        partial_prod[0] = 0.0
                        if lane_id < num_warps:
                            partial_max[0] = smem_max[lane_id, out_lane, out_elem]
                            partial_sum[0] = smem_sum[lane_id, out_lane, out_elem]
                            partial_prod[0] = smem_prod[lane_id, out_lane, out_elem]
                        global_max[0] = T.warp_reduce_max(partial_max[0])
                        rescale[0] = 0.0
                        if lane_id < num_warps:
                            rescale[0] = T.exp(partial_max[0] - global_max[0])
                        global_sum[0] = T.warp_reduce_sum(partial_sum[0] * rescale[0])
                        global_prod[0] = T.warp_reduce_sum(partial_prod[0] * rescale[0])
                        if lane_id == 0 and col < head_dim:
                            out[token_id, col] = global_prod[0] / global_sum[0]
                else:
                    if tx < tile_cols:
                        col = col_block * tile_cols + tx
                        out_lane = tx // 2
                        out_elem = tx - out_lane * 2
                        global_max = T.alloc_local((1,), dtype=T.float32)
                        global_sum = T.alloc_local((1,), dtype=T.float32)
                        global_prod = T.alloc_local((1,), dtype=T.float32)
                        rescale = T.alloc_local((1,), dtype=T.float32)
                        global_max[0] = -3.4028234663852886e38
                        for wid in T.serial(0, num_warps):
                            global_max[0] = T.max(global_max[0], smem_max[wid, out_lane, out_elem])
                        global_sum[0] = 0.0
                        global_prod[0] = 0.0
                        for wid in T.serial(0, num_warps):
                            rescale[0] = T.exp(smem_max[wid, out_lane, out_elem] - global_max[0])
                            global_sum[0] += smem_sum[wid, out_lane, out_elem] * rescale[0]
                            global_prod[0] += smem_prod[wid, out_lane, out_elem] * rescale[0]
                        if col < head_dim:
                            out[token_id, col] = global_prod[0] / global_sum[0]
            else:
                if tx < tile_cols:
                    col = col_block * tile_cols + tx
                    if col < head_dim:
                        out[token_id, col] = 0.0

    return compress_forward_ratio128_decode_flat_parallel_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_forward_ratio128_decode_parallel_kernel(head_dim: int, final_merge: str = "shared"):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.dynamic("num_blocks")
    tile_cols = 64
    slots_per_warp = 8
    num_warps = 16
    threads = 512
    col_blocks = (head_dim + tile_cols - 1) // tile_cols
    kv_score_input_stride = T.dynamic("kv_score_input_stride")
    ape_stride = T.dynamic("ape_stride")
    indices_stride = T.dynamic("indices_stride")
    seq_lens_stride = T.dynamic("seq_lens_stride")
    out_stride = T.dynamic("out_stride")

    if final_merge not in ("shared", "warp"):
        raise ValueError(f"unsupported c128 decode final_merge={final_merge!r}")

    @_tilelang_jit(
        tilelang,
        f"dsv4_c128_decode_parallel_{final_merge}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_DECODE_COMPILE_PROFILE,
        ),
    )
    def compress_forward_ratio128_decode_parallel_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 128, head_dim * 2), T.float32],
        kv_score_input: T.StridedTensor[(num_tokens, head_dim * 2), (kv_score_input_stride, 1), T.float32],
        ape: T.StridedTensor[(128, head_dim), (ape_stride, 1), T.float32],
        indices: T.StridedTensor[(num_tokens,), (indices_stride,), T.int32],
        seq_lens: T.StridedTensor[(num_tokens,), (seq_lens_stride,), T.int32],
        out: T.StridedTensor[(num_tokens, head_dim), (out_stride, 1), T.float32],
    ):
        with T.Kernel(num_tokens, col_blocks, threads=threads) as (token_id, col_block):
            tx = T.get_thread_binding()
            warp_id = tx // 32
            lane_id = tx - warp_id * 32
            col_base = col_block * tile_cols + lane_id * 2
            block_id = indices[token_id]
            block_id_i64 = T.Cast("int64", block_id)
            seq_len = seq_lens[token_id]

            if warp_id == num_warps - 1:
                write_pos = (seq_len + 127) % 128
                for elem in T.serial(0, 2):
                    col = col_base + elem
                    if col < head_dim:
                        kv_score_buffer[block_id_i64, write_pos, col] = kv_score_input[token_id, col]
                        kv_score_buffer[block_id_i64, write_pos, head_dim + col] = kv_score_input[token_id, head_dim + col]

            local_max = T.alloc_local((2,), dtype=T.float32)
            local_sum = T.alloc_local((2,), dtype=T.float32)
            local_prod = T.alloc_local((2,), dtype=T.float32)
            tmp_weight = T.alloc_local((1,), dtype=T.float32)
            smem_max = T.alloc_shared((16, 32, 2), dtype=T.float32)
            smem_sum = T.alloc_shared((16, 32, 2), dtype=T.float32)
            smem_prod = T.alloc_shared((16, 32, 2), dtype=T.float32)

            for elem in T.serial(0, 2):
                local_max[elem] = -3.4028234663852886e38
                local_sum[elem] = 0.0
                local_prod[elem] = 0.0

            if seq_len % 128 == 0:
                for slot_i in T.serial(0, slots_per_warp):
                    slot = warp_id * slots_per_warp + slot_i
                    for elem in T.serial(0, 2):
                        col = col_base + elem
                        if col < head_dim:
                            local_max[elem] = T.max(
                                local_max[elem],
                                kv_score_buffer[block_id_i64, slot, head_dim + col] + ape[slot, col],
                            )

                for slot_i in T.serial(0, slots_per_warp):
                    slot = warp_id * slots_per_warp + slot_i
                    for elem in T.serial(0, 2):
                        col = col_base + elem
                        if col < head_dim:
                            tmp_weight[0] = T.exp(
                                kv_score_buffer[block_id_i64, slot, head_dim + col]
                                + ape[slot, col]
                                - local_max[elem]
                            )
                            local_sum[elem] += tmp_weight[0]
                            local_prod[elem] += kv_score_buffer[block_id_i64, slot, col] * tmp_weight[0]

                for elem in T.serial(0, 2):
                    smem_max[warp_id, lane_id, elem] = local_max[elem]
                    smem_sum[warp_id, lane_id, elem] = local_sum[elem]
                    smem_prod[warp_id, lane_id, elem] = local_prod[elem]
                T.sync_threads()

                if final_merge == "warp":
                    for reduce_group in T.serial(0, 4):
                        reduce_col_in_tile = reduce_group * num_warps + warp_id
                        out_lane = reduce_col_in_tile // 2
                        out_elem = reduce_col_in_tile - out_lane * 2
                        col = col_block * tile_cols + reduce_col_in_tile
                        partial_max = T.alloc_local((1,), dtype=T.float32)
                        partial_sum = T.alloc_local((1,), dtype=T.float32)
                        partial_prod = T.alloc_local((1,), dtype=T.float32)
                        global_max = T.alloc_local((1,), dtype=T.float32)
                        global_sum = T.alloc_local((1,), dtype=T.float32)
                        global_prod = T.alloc_local((1,), dtype=T.float32)
                        rescale = T.alloc_local((1,), dtype=T.float32)
                        partial_max[0] = -3.4028234663852886e38
                        partial_sum[0] = 0.0
                        partial_prod[0] = 0.0
                        if lane_id < num_warps:
                            partial_max[0] = smem_max[lane_id, out_lane, out_elem]
                            partial_sum[0] = smem_sum[lane_id, out_lane, out_elem]
                            partial_prod[0] = smem_prod[lane_id, out_lane, out_elem]
                        global_max[0] = T.warp_reduce_max(partial_max[0])
                        rescale[0] = 0.0
                        if lane_id < num_warps:
                            rescale[0] = T.exp(partial_max[0] - global_max[0])
                        global_sum[0] = T.warp_reduce_sum(partial_sum[0] * rescale[0])
                        global_prod[0] = T.warp_reduce_sum(partial_prod[0] * rescale[0])
                        if lane_id == 0 and col < head_dim:
                            out[token_id, col] = global_prod[0] / global_sum[0]
                else:
                    if tx < tile_cols:
                        col = col_block * tile_cols + tx
                        out_lane = tx // 2
                        out_elem = tx - out_lane * 2
                        global_max = T.alloc_local((1,), dtype=T.float32)
                        global_sum = T.alloc_local((1,), dtype=T.float32)
                        global_prod = T.alloc_local((1,), dtype=T.float32)
                        rescale = T.alloc_local((1,), dtype=T.float32)
                        global_max[0] = -3.4028234663852886e38
                        for wid in T.serial(0, num_warps):
                            global_max[0] = T.max(global_max[0], smem_max[wid, out_lane, out_elem])
                        global_sum[0] = 0.0
                        global_prod[0] = 0.0
                        for wid in T.serial(0, num_warps):
                            rescale[0] = T.exp(smem_max[wid, out_lane, out_elem] - global_max[0])
                            global_sum[0] += smem_sum[wid, out_lane, out_elem] * rescale[0]
                            global_prod[0] += smem_prod[wid, out_lane, out_elem] * rescale[0]
                        if col < head_dim:
                            out[token_id, col] = global_prod[0] / global_sum[0]
            else:
                if tx < tile_cols:
                    col = col_block * tile_cols + tx
                    if col < head_dim:
                        out[token_id, col] = 0.0

    return compress_forward_ratio128_decode_parallel_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_prefill_zero_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    col_blocks = (head_dim + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_compress_prefill_zero_t{threads}")
    def compress_prefill_zero_kernel(out: T.Tensor[(num_tokens, head_dim), T.float32]):
        with T.Kernel(num_tokens, col_blocks, threads=threads) as (token_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                out[token_id, col] = 0.0

    return compress_prefill_zero_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_write_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_write_rows = T.dynamic("num_write_rows")
    write_rows_stride = T.dynamic("write_rows_stride")
    cols = head_dim * 4
    col_blocks = (cols + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_write_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_write_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 8, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        write_rows: T.StridedTensor[(num_write_rows, 4), (write_rows_stride, 1), T.int32],
    ):
        with T.Kernel(num_write_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < cols:
                ragged_id = write_rows[row_id, 0]
                batch_id = write_rows[row_id, 1]
                position = write_rows[row_id, 2]
                window_len = write_rows[row_id, 3]
                if window_len >= 0:
                    block_id = indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    kv_score_buffer[block_id_i64, position % 8, col] = kv_score_input[ragged_id, col]

    return compress_ratio4_prefill_write_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_write_vec4_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_write_rows = T.dynamic("num_write_rows")
    write_rows_stride = T.dynamic("write_rows_stride")
    vector_elems = 4
    cols = head_dim * 4
    col_blocks = ((cols // vector_elems) + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_write_vec4_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_write_vec4_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 8, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        write_rows: T.StridedTensor[(num_write_rows, 4), (write_rows_stride, 1), T.int32],
    ):
        with T.Kernel(num_write_rows, col_blocks, threads=threads) as (row_id, col_block):
            col_base = (col_block * threads + T.get_thread_binding()) * vector_elems
            if col_base < cols:
                ragged_id = write_rows[row_id, 0]
                batch_id = write_rows[row_id, 1]
                position = write_rows[row_id, 2]
                window_len = write_rows[row_id, 3]
                if window_len >= 0:
                    block_id = indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    for vec in T.vectorized(vector_elems):
                        col = col_base + vec
                        kv_score_buffer[block_id_i64, position % 8, col] = kv_score_input[ragged_id, col]

    return compress_ratio4_prefill_write_vec4_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_reduce_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_compress_rows = T.dynamic("num_compress_rows")
    compress_rows_stride = T.dynamic("compress_rows_stride")
    col_blocks = (head_dim + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_reduce_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_reduce_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 8, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        ape: T.Tensor[(8, head_dim), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        compress_rows: T.StridedTensor[(num_compress_rows, 4), (compress_rows_stride, 1), T.int32],
        out: T.Tensor[(num_tokens, head_dim), T.float32],
    ):
        with T.Kernel(num_compress_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                ragged_id = compress_rows[row_id, 0]
                batch_id = compress_rows[row_id, 1]
                position = compress_rows[row_id, 2]
                window_len = compress_rows[row_id, 3]
                if window_len >= 0:
                    block_id = indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    seq_len = position + 1
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    logit = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    max_logit[0] = -3.4028234663852886e38

                    for slot in T.serial(0, 8):
                        if seq_len == 4 and slot < 4:
                            logit[0] = -1.0e9 + ape[slot, col]
                        else:
                            if slot < window_len:
                                source_slot = (seq_len + slot) % 8
                                if slot < 4:
                                    logit[0] = kv_score_buffer[block_id_i64, source_slot, head_dim * 2 + col] + ape[slot, col]
                                else:
                                    logit[0] = kv_score_buffer[block_id_i64, source_slot, head_dim * 3 + col] + ape[slot, col]
                            else:
                                input_row = ragged_id + slot - 7
                                if slot < 4:
                                    logit[0] = kv_score_input[input_row, head_dim * 2 + col] + ape[slot, col]
                                else:
                                    logit[0] = kv_score_input[input_row, head_dim * 3 + col] + ape[slot, col]
                        max_logit[0] = T.max(max_logit[0], logit[0])

                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 8):
                        if seq_len == 4 and slot < 4:
                            weight[0] = T.exp(-1.0e9 + ape[slot, col] - max_logit[0])
                            denom[0] += weight[0]
                        else:
                            if slot < window_len:
                                source_slot = (seq_len + slot) % 8
                                if slot < 4:
                                    weight[0] = T.exp(kv_score_buffer[block_id_i64, source_slot, head_dim * 2 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_buffer[block_id_i64, source_slot, col] * weight[0]
                                else:
                                    weight[0] = T.exp(kv_score_buffer[block_id_i64, source_slot, head_dim * 3 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_buffer[block_id_i64, source_slot, head_dim + col] * weight[0]
                            else:
                                input_row = ragged_id + slot - 7
                                if slot < 4:
                                    weight[0] = T.exp(kv_score_input[input_row, head_dim * 2 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_input[input_row, col] * weight[0]
                                else:
                                    weight[0] = T.exp(kv_score_input[input_row, head_dim * 3 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_input[input_row, head_dim + col] * weight[0]
                            denom[0] += weight[0]
                    out[ragged_id, col] = acc[0] / denom[0]

    return compress_ratio4_prefill_reduce_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_page_write_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_write_rows = T.dynamic("num_write_rows")
    write_rows_stride = T.dynamic("write_rows_stride")
    cols = head_dim * 4
    col_blocks = (cols + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_page_write_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_page_write_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 4, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        extra_data: T.Tensor[(num_indices, 4), T.int32],
        write_rows: T.StridedTensor[(num_write_rows, 4), (write_rows_stride, 1), T.int32],
    ):
        with T.Kernel(num_write_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < cols:
                ragged_id = write_rows[row_id, 0]
                batch_id = write_rows[row_id, 1]
                position = write_rows[row_id, 2]
                window_len = write_rows[row_id, 3]
                if window_len >= 0:
                    block_id = T.alloc_local((1,), dtype=T.int32)
                    block_id[0] = indices[batch_id]
                    if position < extra_data[batch_id, 3]:
                        block_id[0] = extra_data[batch_id, 2]
                    block_id_i64 = T.Cast("int64", block_id[0])
                    kv_score_buffer[block_id_i64, position % 4, col] = kv_score_input[ragged_id, col]

    return compress_ratio4_prefill_page_write_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_page_write_vec4_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_write_rows = T.dynamic("num_write_rows")
    write_rows_stride = T.dynamic("write_rows_stride")
    vector_elems = 4
    cols = head_dim * 4
    col_blocks = ((cols // vector_elems) + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_page_write_vec4_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_page_write_vec4_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 4, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        extra_data: T.Tensor[(num_indices, 4), T.int32],
        write_rows: T.StridedTensor[(num_write_rows, 4), (write_rows_stride, 1), T.int32],
    ):
        with T.Kernel(num_write_rows, col_blocks, threads=threads) as (row_id, col_block):
            col_base = (col_block * threads + T.get_thread_binding()) * vector_elems
            if col_base < cols:
                ragged_id = write_rows[row_id, 0]
                batch_id = write_rows[row_id, 1]
                position = write_rows[row_id, 2]
                window_len = write_rows[row_id, 3]
                if window_len >= 0:
                    block_id = T.alloc_local((1,), dtype=T.int32)
                    block_id[0] = indices[batch_id]
                    if position < extra_data[batch_id, 3]:
                        block_id[0] = extra_data[batch_id, 2]
                    block_id_i64 = T.Cast("int64", block_id[0])
                    for vec in T.vectorized(vector_elems):
                        col = col_base + vec
                        kv_score_buffer[block_id_i64, position % 4, col] = kv_score_input[ragged_id, col]

    return compress_ratio4_prefill_page_write_vec4_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_page_reduce_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_compress_rows = T.dynamic("num_compress_rows")
    compress_rows_stride = T.dynamic("compress_rows_stride")
    col_blocks = (head_dim + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_page_reduce_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_page_reduce_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 4, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        ape: T.Tensor[(8, head_dim), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        extra_data: T.Tensor[(num_indices, 4), T.int32],
        compress_rows: T.StridedTensor[(num_compress_rows, 4), (compress_rows_stride, 1), T.int32],
        out: T.Tensor[(num_tokens, head_dim), T.float32],
    ):
        with T.Kernel(num_compress_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                ragged_id = compress_rows[row_id, 0]
                batch_id = compress_rows[row_id, 1]
                position = compress_rows[row_id, 2]
                window_len = compress_rows[row_id, 3]
                if window_len >= 0:
                    seq_len = position + 1
                    load_first_page = extra_data[batch_id, 0]
                    load_second_page = extra_data[batch_id, 1]
                    load_first_page_i64 = T.Cast("int64", load_first_page)
                    load_second_page_i64 = T.Cast("int64", load_second_page)
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    logit = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    source_block = T.alloc_local((1,), dtype=T.int64)
                    max_logit[0] = -3.4028234663852886e38

                    for slot in T.serial(0, 8):
                        if seq_len == 4 and slot < 4:
                            logit[0] = -1.0e9 + ape[slot, col]
                        else:
                            if slot < window_len:
                                source_block[0] = load_second_page_i64
                                if window_len > 4 and slot < 4:
                                    source_block[0] = load_first_page_i64
                                if slot < 4:
                                    logit[0] = kv_score_buffer[source_block[0], slot % 4, head_dim * 2 + col] + ape[slot, col]
                                else:
                                    logit[0] = kv_score_buffer[source_block[0], slot % 4, head_dim * 3 + col] + ape[slot, col]
                            else:
                                input_row = ragged_id + slot - 7
                                if slot < 4:
                                    logit[0] = kv_score_input[input_row, head_dim * 2 + col] + ape[slot, col]
                                else:
                                    logit[0] = kv_score_input[input_row, head_dim * 3 + col] + ape[slot, col]
                        max_logit[0] = T.max(max_logit[0], logit[0])

                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 8):
                        if seq_len == 4 and slot < 4:
                            weight[0] = T.exp(-1.0e9 + ape[slot, col] - max_logit[0])
                            denom[0] += weight[0]
                        else:
                            if slot < window_len:
                                source_block[0] = load_second_page_i64
                                if window_len > 4 and slot < 4:
                                    source_block[0] = load_first_page_i64
                                if slot < 4:
                                    weight[0] = T.exp(kv_score_buffer[source_block[0], slot % 4, head_dim * 2 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_buffer[source_block[0], slot % 4, col] * weight[0]
                                else:
                                    weight[0] = T.exp(kv_score_buffer[source_block[0], slot % 4, head_dim * 3 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_buffer[source_block[0], slot % 4, head_dim + col] * weight[0]
                            else:
                                input_row = ragged_id + slot - 7
                                if slot < 4:
                                    weight[0] = T.exp(kv_score_input[input_row, head_dim * 2 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_input[input_row, col] * weight[0]
                                else:
                                    weight[0] = T.exp(kv_score_input[input_row, head_dim * 3 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_input[input_row, head_dim + col] * weight[0]
                            denom[0] += weight[0]
                    out[ragged_id, col] = acc[0] / denom[0]

    return compress_ratio4_prefill_page_reduce_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_page_reduce_cached_kernel(
    head_dim: int,
    threads: int = 128,
    compile_profile: str = "longseq",
):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_compress_rows = T.dynamic("num_compress_rows")
    compress_rows_stride = T.dynamic("compress_rows_stride")
    col_blocks = (head_dim + threads - 1) // threads
    pass_profile = _compress_c4_cached_reduce_pass_profile(compile_profile)

    @_tilelang_jit(
        tilelang,
        f"dsv4_c4_prefill_page_reduce_cached_t{threads}_{compile_profile}",
        pass_configs=_tilelang_musa_burst_reduce_pass_configs(
            tilelang,
            compile_profile=pass_profile,
        ),
    )
    def compress_ratio4_prefill_page_reduce_cached_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 4, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        ape: T.Tensor[(8, head_dim), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        extra_data: T.Tensor[(num_indices, 4), T.int32],
        compress_rows: T.StridedTensor[(num_compress_rows, 4), (compress_rows_stride, 1), T.int32],
        out: T.Tensor[(num_tokens, head_dim), T.float32],
    ):
        with T.Kernel(num_compress_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                ragged_id = compress_rows[row_id, 0]
                batch_id = compress_rows[row_id, 1]
                position = compress_rows[row_id, 2]
                window_len = compress_rows[row_id, 3]
                if window_len >= 0:
                    seq_len = position + 1
                    load_first_page = extra_data[batch_id, 0]
                    load_second_page = extra_data[batch_id, 1]
                    load_first_page_i64 = T.Cast("int64", load_first_page)
                    load_second_page_i64 = T.Cast("int64", load_second_page)
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    logit = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    source_block = T.alloc_local((1,), dtype=T.int64)
                    logits = T.alloc_local((8,), dtype=T.float32)
                    values = T.alloc_local((8,), dtype=T.float32)
                    max_logit[0] = -3.4028234663852886e38

                    for slot in T.serial(0, 8):
                        if seq_len == 4 and slot < 4:
                            logits[slot] = -1.0e9 + ape[slot, col]
                            values[slot] = 0.0
                        else:
                            if slot < window_len:
                                source_block[0] = load_second_page_i64
                                if window_len > 4 and slot < 4:
                                    source_block[0] = load_first_page_i64
                                if slot < 4:
                                    logits[slot] = kv_score_buffer[source_block[0], slot % 4, head_dim * 2 + col] + ape[slot, col]
                                    values[slot] = kv_score_buffer[source_block[0], slot % 4, col]
                                else:
                                    logits[slot] = kv_score_buffer[source_block[0], slot % 4, head_dim * 3 + col] + ape[slot, col]
                                    values[slot] = kv_score_buffer[source_block[0], slot % 4, head_dim + col]
                            else:
                                input_row = ragged_id + slot - 7
                                if slot < 4:
                                    logits[slot] = kv_score_input[input_row, head_dim * 2 + col] + ape[slot, col]
                                    values[slot] = kv_score_input[input_row, col]
                                else:
                                    logits[slot] = kv_score_input[input_row, head_dim * 3 + col] + ape[slot, col]
                                    values[slot] = kv_score_input[input_row, head_dim + col]
                        max_logit[0] = T.max(max_logit[0], logits[slot])

                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 8):
                        weight[0] = T.exp(logits[slot] - max_logit[0])
                        denom[0] += weight[0]
                        acc[0] += values[slot] * weight[0]
                    out[ragged_id, col] = acc[0] / denom[0]

    return compress_ratio4_prefill_page_reduce_cached_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_flat_write_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_slots = T.dynamic("num_slots")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_write_rows = T.dynamic("num_write_rows")
    write_rows_stride = T.dynamic("write_rows_stride")
    cols = head_dim * 4
    col_blocks = (cols + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_flat_write_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_flat_write_kernel(
        kv_score_buffer: T.Tensor[(num_slots, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        write_rows: T.StridedTensor[(num_write_rows, 4), (write_rows_stride, 1), T.int32],
    ):
        with T.Kernel(num_write_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < cols:
                ragged_id = write_rows[row_id, 0]
                batch_id = write_rows[row_id, 1]
                position = write_rows[row_id, 2]
                window_len = write_rows[row_id, 3]
                if window_len >= 0:
                    block_id = indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    write_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", position % 8)
                    kv_score_buffer[write_row, col] = kv_score_input[ragged_id, col]

    return compress_ratio4_prefill_flat_write_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_flat_write_vec4_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_slots = T.dynamic("num_slots")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_write_rows = T.dynamic("num_write_rows")
    write_rows_stride = T.dynamic("write_rows_stride")
    vector_elems = 4
    cols = head_dim * 4
    col_blocks = ((cols // vector_elems) + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_flat_write_vec4_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_flat_write_vec4_kernel(
        kv_score_buffer: T.Tensor[(num_slots, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        write_rows: T.StridedTensor[(num_write_rows, 4), (write_rows_stride, 1), T.int32],
    ):
        with T.Kernel(num_write_rows, col_blocks, threads=threads) as (row_id, col_block):
            col_base = (col_block * threads + T.get_thread_binding()) * vector_elems
            if col_base < cols:
                ragged_id = write_rows[row_id, 0]
                batch_id = write_rows[row_id, 1]
                position = write_rows[row_id, 2]
                window_len = write_rows[row_id, 3]
                if window_len >= 0:
                    block_id = indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    write_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", position % 8)
                    for vec in T.vectorized(vector_elems):
                        col = col_base + vec
                        kv_score_buffer[write_row, col] = kv_score_input[ragged_id, col]

    return compress_ratio4_prefill_flat_write_vec4_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio4_prefill_flat_reduce_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_slots = T.dynamic("num_slots")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_compress_rows = T.dynamic("num_compress_rows")
    compress_rows_stride = T.dynamic("compress_rows_stride")
    col_blocks = (head_dim + threads - 1) // threads

    @_tilelang_jit(tilelang, f"dsv4_c4_prefill_flat_reduce_t{threads}", pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang))
    def compress_ratio4_prefill_flat_reduce_kernel(
        kv_score_buffer: T.Tensor[(num_slots, head_dim * 4), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 4), T.float32],
        ape: T.Tensor[(8, head_dim), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        compress_rows: T.StridedTensor[(num_compress_rows, 4), (compress_rows_stride, 1), T.int32],
        out: T.Tensor[(num_tokens, head_dim), T.float32],
    ):
        with T.Kernel(num_compress_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                ragged_id = compress_rows[row_id, 0]
                batch_id = compress_rows[row_id, 1]
                position = compress_rows[row_id, 2]
                window_len = compress_rows[row_id, 3]
                if window_len >= 0:
                    block_id = indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    seq_len = position + 1
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    logit = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    max_logit[0] = -3.4028234663852886e38

                    for slot in T.serial(0, 8):
                        if seq_len == 4 and slot < 4:
                            logit[0] = -1.0e9 + ape[slot, col]
                        else:
                            if slot < window_len:
                                source_slot = (seq_len + slot) % 8
                                source_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", source_slot)
                                if slot < 4:
                                    logit[0] = kv_score_buffer[source_row, head_dim * 2 + col] + ape[slot, col]
                                else:
                                    logit[0] = kv_score_buffer[source_row, head_dim * 3 + col] + ape[slot, col]
                            else:
                                input_row = ragged_id + slot - 7
                                if slot < 4:
                                    logit[0] = kv_score_input[input_row, head_dim * 2 + col] + ape[slot, col]
                                else:
                                    logit[0] = kv_score_input[input_row, head_dim * 3 + col] + ape[slot, col]
                        max_logit[0] = T.max(max_logit[0], logit[0])

                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 8):
                        if seq_len == 4 and slot < 4:
                            weight[0] = T.exp(-1.0e9 + ape[slot, col] - max_logit[0])
                            denom[0] += weight[0]
                        else:
                            if slot < window_len:
                                source_slot = (seq_len + slot) % 8
                                source_row = block_id_i64 * T.Cast("int64", 8) + T.Cast("int64", source_slot)
                                if slot < 4:
                                    weight[0] = T.exp(kv_score_buffer[source_row, head_dim * 2 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_buffer[source_row, col] * weight[0]
                                else:
                                    weight[0] = T.exp(kv_score_buffer[source_row, head_dim * 3 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_buffer[source_row, head_dim + col] * weight[0]
                            else:
                                input_row = ragged_id + slot - 7
                                if slot < 4:
                                    weight[0] = T.exp(kv_score_input[input_row, head_dim * 2 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_input[input_row, col] * weight[0]
                                else:
                                    weight[0] = T.exp(kv_score_input[input_row, head_dim * 3 + col] + ape[slot, col] - max_logit[0])
                                    acc[0] += kv_score_input[input_row, head_dim + col] * weight[0]
                            denom[0] += weight[0]
                    out[ragged_id, col] = acc[0] / denom[0]

    return compress_ratio4_prefill_flat_reduce_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio128_prefill_write_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_write_rows = T.dynamic("num_write_rows")
    write_rows_stride = T.dynamic("write_rows_stride")
    cols = head_dim * 2
    col_blocks = (cols + threads - 1) // threads

    @_tilelang_jit(
        tilelang,
        f"dsv4_c128_prefill_write_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_C128_PREFILL_COMPILE_PROFILE,
        ),
    )
    def compress_ratio128_prefill_write_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 128, head_dim * 2), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 2), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        write_rows: T.StridedTensor[(num_write_rows, 4), (write_rows_stride, 1), T.int32],
    ):
        with T.Kernel(num_write_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < cols:
                ragged_id = write_rows[row_id, 0]
                batch_id = write_rows[row_id, 1]
                position = write_rows[row_id, 2]
                window_len = write_rows[row_id, 3]
                if window_len >= 0:
                    block_id = indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    kv_score_buffer[block_id_i64, position % 128, col] = kv_score_input[ragged_id, col]

    return compress_ratio128_prefill_write_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio128_prefill_write_vec4_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_write_rows = T.dynamic("num_write_rows")
    write_rows_stride = T.dynamic("write_rows_stride")
    vector_elems = 4
    cols = head_dim * 2
    col_blocks = ((cols // vector_elems) + threads - 1) // threads

    @_tilelang_jit(
        tilelang,
        f"dsv4_c128_prefill_write_vec4_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_C128_PREFILL_COMPILE_PROFILE,
        ),
    )
    def compress_ratio128_prefill_write_vec4_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 128, head_dim * 2), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 2), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        write_rows: T.StridedTensor[(num_write_rows, 4), (write_rows_stride, 1), T.int32],
    ):
        with T.Kernel(num_write_rows, col_blocks, threads=threads) as (row_id, col_block):
            col_base = (col_block * threads + T.get_thread_binding()) * vector_elems
            if col_base < cols:
                ragged_id = write_rows[row_id, 0]
                batch_id = write_rows[row_id, 1]
                position = write_rows[row_id, 2]
                window_len = write_rows[row_id, 3]
                if window_len >= 0:
                    block_id = indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    for vec in T.vectorized(vector_elems):
                        col = col_base + vec
                        kv_score_buffer[block_id_i64, position % 128, col] = kv_score_input[ragged_id, col]

    return compress_ratio128_prefill_write_vec4_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio128_prefill_reduce_kernel(head_dim: int, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_compress_rows = T.dynamic("num_compress_rows")
    compress_rows_stride = T.dynamic("compress_rows_stride")
    col_blocks = (head_dim + threads - 1) // threads

    @_tilelang_jit(
        tilelang,
        f"dsv4_c128_prefill_reduce_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_C128_PREFILL_COMPILE_PROFILE,
        ),
    )
    def compress_ratio128_prefill_reduce_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 128, head_dim * 2), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 2), T.float32],
        ape: T.Tensor[(128, head_dim), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        load_indices: T.Tensor[(num_indices,), T.int32],
        compress_rows: T.StridedTensor[(num_compress_rows, 4), (compress_rows_stride, 1), T.int32],
        out: T.Tensor[(num_tokens, head_dim), T.float32],
    ):
        with T.Kernel(num_compress_rows, col_blocks, threads=threads) as (row_id, col_block):
            col = col_block * threads + T.get_thread_binding()
            if col < head_dim:
                ragged_id = compress_rows[row_id, 0]
                batch_id = compress_rows[row_id, 1]
                window_len = compress_rows[row_id, 3]
                if window_len >= 0:
                    block_id = load_indices[batch_id]
                    block_id_i64 = T.Cast("int64", block_id)
                    max_logit = T.alloc_local((1,), dtype=T.float32)
                    denom = T.alloc_local((1,), dtype=T.float32)
                    acc = T.alloc_local((1,), dtype=T.float32)
                    weight = T.alloc_local((1,), dtype=T.float32)
                    max_logit[0] = -3.4028234663852886e38

                    for slot in T.serial(0, 128):
                        if slot < window_len:
                            max_logit[0] = T.max(max_logit[0], kv_score_buffer[block_id_i64, slot, head_dim + col] + ape[slot, col])
                        else:
                            input_row = ragged_id + slot - 127
                            max_logit[0] = T.max(max_logit[0], kv_score_input[input_row, head_dim + col] + ape[slot, col])

                    denom[0] = 0.0
                    acc[0] = 0.0
                    for slot in T.serial(0, 128):
                        if slot < window_len:
                            weight[0] = T.exp(kv_score_buffer[block_id_i64, slot, head_dim + col] + ape[slot, col] - max_logit[0])
                            acc[0] += kv_score_buffer[block_id_i64, slot, col] * weight[0]
                            denom[0] += weight[0]
                        else:
                            input_row = ragged_id + slot - 127
                            weight[0] = T.exp(kv_score_input[input_row, head_dim + col] + ape[slot, col] - max_logit[0])
                            acc[0] += kv_score_input[input_row, col] * weight[0]
                            denom[0] += weight[0]
                    out[ragged_id, col] = acc[0] / denom[0]

    return compress_ratio128_prefill_reduce_kernel

@lru_cache(maxsize=None)
def _tilelang_compress_ratio128_prefill_reduce_parallel_kernel(head_dim: int, final_merge: str = "shared"):
    import tilelang
    import tilelang.language as T

    num_blocks = T.dynamic("num_blocks")
    num_tokens = T.dynamic("num_tokens")
    num_indices = T.dynamic("num_indices")
    num_compress_rows = T.dynamic("num_compress_rows")
    compress_rows_stride = T.dynamic("compress_rows_stride")
    tile_cols = 64
    slots_per_warp = 8
    num_warps = 16
    threads = 512
    col_blocks = (head_dim + tile_cols - 1) // tile_cols

    if final_merge not in ("shared", "warp"):
        raise ValueError(f"unsupported c128 prefill final_merge={final_merge!r}")

    @_tilelang_jit(
        tilelang,
        f"dsv4_c128_prefill_reduce_parallel_{final_merge}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            compile_profile=_COMPRESS_C128_PREFILL_COMPILE_PROFILE,
        ),
    )
    def compress_ratio128_prefill_reduce_parallel_kernel(
        kv_score_buffer: T.Tensor[(num_blocks, 128, head_dim * 2), T.float32],
        kv_score_input: T.Tensor[(num_tokens, head_dim * 2), T.float32],
        ape: T.Tensor[(128, head_dim), T.float32],
        indices: T.Tensor[(num_indices,), T.int32],
        load_indices: T.Tensor[(num_indices,), T.int32],
        compress_rows: T.StridedTensor[(num_compress_rows, 4), (compress_rows_stride, 1), T.int32],
        out: T.Tensor[(num_tokens, head_dim), T.float32],
    ):
        with T.Kernel(num_compress_rows, col_blocks, threads=threads) as (row_id, col_block):
            tx = T.get_thread_binding()
            warp_id = tx // 32
            lane_id = tx - warp_id * 32
            col_base = col_block * tile_cols + lane_id * 2
            ragged_id = compress_rows[row_id, 0]
            batch_id = compress_rows[row_id, 1]
            window_len = compress_rows[row_id, 3]
            if window_len >= 0:
                block_id = load_indices[batch_id]
                block_id_i64 = T.Cast("int64", block_id)

                local_max = T.alloc_local((2,), dtype=T.float32)
                local_sum = T.alloc_local((2,), dtype=T.float32)
                local_prod = T.alloc_local((2,), dtype=T.float32)
                tmp_weight = T.alloc_local((1,), dtype=T.float32)
                smem_max = T.alloc_shared((16, 32, 2), dtype=T.float32)
                smem_sum = T.alloc_shared((16, 32, 2), dtype=T.float32)
                smem_prod = T.alloc_shared((16, 32, 2), dtype=T.float32)

                for elem in T.serial(0, 2):
                    local_max[elem] = -3.4028234663852886e38
                    local_sum[elem] = 0.0
                    local_prod[elem] = 0.0

                for slot_i in T.serial(0, slots_per_warp):
                    slot = warp_id * slots_per_warp + slot_i
                    for elem in T.serial(0, 2):
                        col = col_base + elem
                        if col < head_dim:
                            if slot < window_len:
                                local_max[elem] = T.max(
                                    local_max[elem],
                                    kv_score_buffer[block_id_i64, slot, head_dim + col] + ape[slot, col],
                                )
                            else:
                                input_row = ragged_id + slot - 127
                                local_max[elem] = T.max(
                                    local_max[elem],
                                    kv_score_input[input_row, head_dim + col] + ape[slot, col],
                                )

                for slot_i in T.serial(0, slots_per_warp):
                    slot = warp_id * slots_per_warp + slot_i
                    for elem in T.serial(0, 2):
                        col = col_base + elem
                        if col < head_dim:
                            if slot < window_len:
                                tmp_weight[0] = T.exp(kv_score_buffer[block_id_i64, slot, head_dim + col] + ape[slot, col] - local_max[elem])
                                local_prod[elem] += kv_score_buffer[block_id_i64, slot, col] * tmp_weight[0]
                                local_sum[elem] += tmp_weight[0]
                            else:
                                input_row = ragged_id + slot - 127
                                tmp_weight[0] = T.exp(kv_score_input[input_row, head_dim + col] + ape[slot, col] - local_max[elem])
                                local_prod[elem] += kv_score_input[input_row, col] * tmp_weight[0]
                                local_sum[elem] += tmp_weight[0]

                for elem in T.serial(0, 2):
                    smem_max[warp_id, lane_id, elem] = local_max[elem]
                    smem_sum[warp_id, lane_id, elem] = local_sum[elem]
                    smem_prod[warp_id, lane_id, elem] = local_prod[elem]
                T.sync_threads()

                if final_merge == "warp":
                    for reduce_group in T.serial(0, 4):
                        reduce_col_in_tile = reduce_group * num_warps + warp_id
                        out_lane = reduce_col_in_tile // 2
                        out_elem = reduce_col_in_tile - out_lane * 2
                        col = col_block * tile_cols + reduce_col_in_tile
                        partial_max = T.alloc_local((1,), dtype=T.float32)
                        partial_sum = T.alloc_local((1,), dtype=T.float32)
                        partial_prod = T.alloc_local((1,), dtype=T.float32)
                        global_max = T.alloc_local((1,), dtype=T.float32)
                        global_sum = T.alloc_local((1,), dtype=T.float32)
                        global_prod = T.alloc_local((1,), dtype=T.float32)
                        rescale = T.alloc_local((1,), dtype=T.float32)
                        partial_max[0] = -3.4028234663852886e38
                        partial_sum[0] = 0.0
                        partial_prod[0] = 0.0
                        if lane_id < num_warps:
                            partial_max[0] = smem_max[lane_id, out_lane, out_elem]
                            partial_sum[0] = smem_sum[lane_id, out_lane, out_elem]
                            partial_prod[0] = smem_prod[lane_id, out_lane, out_elem]
                        global_max[0] = T.warp_reduce_max(partial_max[0])
                        rescale[0] = 0.0
                        if lane_id < num_warps:
                            rescale[0] = T.exp(partial_max[0] - global_max[0])
                        global_sum[0] = T.warp_reduce_sum(partial_sum[0] * rescale[0])
                        global_prod[0] = T.warp_reduce_sum(partial_prod[0] * rescale[0])
                        if lane_id == 0 and col < head_dim:
                            out[ragged_id, col] = global_prod[0] / global_sum[0]
                else:
                    if tx < tile_cols:
                        col = col_block * tile_cols + tx
                        out_lane = tx // 2
                        out_elem = tx - out_lane * 2
                        global_max = T.alloc_local((1,), dtype=T.float32)
                        global_sum = T.alloc_local((1,), dtype=T.float32)
                        global_prod = T.alloc_local((1,), dtype=T.float32)
                        rescale = T.alloc_local((1,), dtype=T.float32)
                        global_max[0] = -3.4028234663852886e38
                        for wid in T.serial(0, num_warps):
                            global_max[0] = T.max(global_max[0], smem_max[wid, out_lane, out_elem])
                        global_sum[0] = 0.0
                        global_prod[0] = 0.0
                        for wid in T.serial(0, num_warps):
                            rescale[0] = T.exp(smem_max[wid, out_lane, out_elem] - global_max[0])
                            global_sum[0] += smem_sum[wid, out_lane, out_elem] * rescale[0]
                            global_prod[0] += smem_prod[wid, out_lane, out_elem] * rescale[0]
                        if col < head_dim:
                            out[ragged_id, col] = global_prod[0] / global_sum[0]

    return compress_ratio128_prefill_reduce_parallel_kernel

__all__ = [
    '_tilelang_compress_forward_ratio4_decode_kernel',
    '_tilelang_compress_forward_ratio4_decode_page_kernel',
    '_tilelang_compress_forward_ratio4_decode_flat_kernel',
    '_tilelang_compress_forward_ratio128_decode_kernel',
    '_tilelang_compress_forward_ratio128_decode_parallel_kernel',
    '_tilelang_compress_forward_ratio128_decode_flat_kernel',
    '_tilelang_compress_forward_ratio128_decode_flat_parallel_kernel',
    '_tilelang_compress_prefill_zero_kernel',
    '_tilelang_compress_ratio4_prefill_write_kernel',
    '_tilelang_compress_ratio4_prefill_write_vec4_kernel',
    '_tilelang_compress_ratio4_prefill_reduce_kernel',
    '_tilelang_compress_ratio4_prefill_page_write_kernel',
    '_tilelang_compress_ratio4_prefill_page_write_vec4_kernel',
    '_tilelang_compress_ratio4_prefill_page_reduce_kernel',
    '_tilelang_compress_ratio4_prefill_page_reduce_cached_kernel',
    '_tilelang_compress_ratio4_prefill_flat_write_kernel',
    '_tilelang_compress_ratio4_prefill_flat_write_vec4_kernel',
    '_tilelang_compress_ratio4_prefill_flat_reduce_kernel',
    '_tilelang_compress_ratio128_prefill_write_kernel',
    '_tilelang_compress_ratio128_prefill_write_vec4_kernel',
    '_tilelang_compress_ratio128_prefill_reduce_kernel',
    '_tilelang_compress_ratio128_prefill_reduce_parallel_kernel',
]
