from functools import lru_cache

import torch

from .kernel_common import (
    SELECT_TOPK_BITSET_WORDS,
    _tilelang_jit,
    _tilelang_musa_aggressive_pass_configs,
)

def _hash_topk_input_dtype(input_dtype: str, T):
    input_dtype = input_dtype.lower().strip()
    if input_dtype == "float32":
        return T.float32
    if input_dtype == "bfloat16":
        return T.bfloat16
    raise ValueError(f"unsupported hash_topk input_dtype={input_dtype!r}")


def _hash_topk_tid2eid_dtype(tid2eid_dtype: str, T):
    tid2eid_dtype = tid2eid_dtype.lower().strip()
    if tid2eid_dtype == "int64":
        return T.int64
    if tid2eid_dtype == "int32":
        return T.int32
    raise ValueError(f"unsupported hash_topk tid2eid_dtype={tid2eid_dtype!r}")


def _hash_topk_input_ids_dtype(input_ids_dtype: str, T):
    input_ids_dtype = input_ids_dtype.lower().strip()
    if input_ids_dtype == "int64":
        return T.int64
    if input_ids_dtype == "int32":
        return T.int32
    raise ValueError(f"unsupported hash_topk input_ids_dtype={input_ids_dtype!r}")


@lru_cache(maxsize=None)
def _tilelang_hash_topk_kernel(
    topk: int,
    num_fused_shared_experts: int,
    threads: int = 128,
    input_dtype: str = "float32",
    input_ids_dtype: str = "int64",
    tid2eid_dtype: str = "int64",
):
    import tilelang
    import tilelang.language as T

    tl_input_dtype = _hash_topk_input_dtype(input_dtype, T)
    tl_input_ids_dtype = _hash_topk_input_ids_dtype(input_ids_dtype, T)
    tl_tid2eid_dtype = _hash_topk_tid2eid_dtype(tid2eid_dtype, T)
    num_tokens = T.dynamic("num_tokens")
    num_experts = T.dynamic("num_experts")
    vocab_size = T.dynamic("vocab_size")
    router_logits_stride_m = T.dynamic("router_logits_stride_m")
    router_logits_stride_n = T.dynamic("router_logits_stride_n")
    input_ids_stride_m = T.dynamic("input_ids_stride_m")
    tid2eid_stride_m = T.dynamic("tid2eid_stride_m")
    tid2eid_stride_n = T.dynamic("tid2eid_stride_n")
    output_topk = topk + num_fused_shared_experts
    reduce_width = 1 << (topk - 1).bit_length()

    @_tilelang_jit(tilelang, f"dsv4_hash_topk_{input_dtype}_{input_ids_dtype}_{tid2eid_dtype}")
    def hash_topk_kernel(
        router_logits: T.StridedTensor[
            (num_tokens, num_experts),
            (router_logits_stride_m, router_logits_stride_n),
            tl_input_dtype,
        ],
        input_ids: T.StridedTensor[(num_tokens,), (input_ids_stride_m,), tl_input_ids_dtype],
        tid2eid: T.StridedTensor[
            (vocab_size, topk),
            (tid2eid_stride_m, tid2eid_stride_n),
            tl_tid2eid_dtype,
        ],
        routed_scores: T.Tensor[(num_tokens, output_topk), T.float32],
        routed_ids: T.Tensor[(num_tokens, output_topk), T.int64],
        shared_weight: T.float32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            scores = T.alloc_shared((topk,), dtype=T.float32)
            expert_ids = T.alloc_shared((topk,), dtype=T.int64)
            reductions = T.alloc_shared((reduce_width,), dtype=T.float32)
            input_id = input_ids[token_id]

            if tx < reduce_width:
                reductions[tx] = 0.0
            if tx < topk:
                expert_id = T.cast(tid2eid[input_id, tx], T.int64)
                expert_ids[tx] = expert_id
                logit = T.cast(router_logits[token_id, expert_id], T.float32)
                score = T.sqrt(T.log(1.0 + T.exp(logit)))
                scores[tx] = score
                reductions[tx] = score
            T.sync_threads()

            if reduce_width >= 64:
                if tx < 32:
                    reductions[tx] += reductions[tx + 32]
                T.sync_threads()
            if reduce_width >= 32:
                if tx < 16:
                    reductions[tx] += reductions[tx + 16]
                T.sync_threads()
            if reduce_width >= 16:
                if tx < 8:
                    reductions[tx] += reductions[tx + 8]
                T.sync_threads()
            if reduce_width >= 8:
                if tx < 4:
                    reductions[tx] += reductions[tx + 4]
                T.sync_threads()
            if reduce_width >= 4:
                if tx < 2:
                    reductions[tx] += reductions[tx + 2]
                T.sync_threads()
            if reduce_width >= 2:
                if tx < 1:
                    reductions[tx] += reductions[tx + 1]
                T.sync_threads()

            if tx < output_topk:
                if tx < topk:
                    routed_ids[token_id, tx] = expert_ids[tx]
                    routed_scores[token_id, tx] = scores[tx] / T.max(reductions[0], 1e-20)
                else:
                    routed_ids[token_id, tx] = num_experts + tx - topk
                    routed_scores[token_id, tx] = shared_weight

    return hash_topk_kernel

@lru_cache(maxsize=None)
def _tilelang_hash_topk_warp_kernel(
    topk: int,
    num_fused_shared_experts: int,
    input_dtype: str = "float32",
    input_ids_dtype: str = "int64",
    tid2eid_dtype: str = "int64",
):
    import tilelang
    import tilelang.language as T

    if topk + num_fused_shared_experts > 32:
        raise ValueError("warp hash_topk kernel requires topk + shared <= 32")

    tl_input_dtype = _hash_topk_input_dtype(input_dtype, T)
    tl_input_ids_dtype = _hash_topk_input_ids_dtype(input_ids_dtype, T)
    tl_tid2eid_dtype = _hash_topk_tid2eid_dtype(tid2eid_dtype, T)
    num_tokens = T.dynamic("num_tokens")
    num_experts = T.dynamic("num_experts")
    vocab_size = T.dynamic("vocab_size")
    router_logits_stride_m = T.dynamic("router_logits_stride_m")
    router_logits_stride_n = T.dynamic("router_logits_stride_n")
    input_ids_stride_m = T.dynamic("input_ids_stride_m")
    tid2eid_stride_m = T.dynamic("tid2eid_stride_m")
    tid2eid_stride_n = T.dynamic("tid2eid_stride_n")
    output_topk = topk + num_fused_shared_experts

    @_tilelang_jit(
        tilelang,
        f"dsv4_hash_topk_warp_fast_{input_dtype}_{input_ids_dtype}_{tid2eid_dtype}_k{topk}_s{num_fused_shared_experts}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=False),
    )
    def hash_topk_warp_kernel(
        router_logits: T.StridedTensor[
            (num_tokens, num_experts),
            (router_logits_stride_m, router_logits_stride_n),
            tl_input_dtype,
        ],
        input_ids: T.StridedTensor[(num_tokens,), (input_ids_stride_m,), tl_input_ids_dtype],
        tid2eid: T.StridedTensor[
            (vocab_size, topk),
            (tid2eid_stride_m, tid2eid_stride_n),
            tl_tid2eid_dtype,
        ],
        routed_scores: T.Tensor[(num_tokens, output_topk), T.float32],
        routed_ids: T.Tensor[(num_tokens, output_topk), T.int64],
        shared_weight: T.float32,
    ):
        with T.Kernel(num_tokens, threads=32) as token_id:
            tx = T.get_thread_binding()
            input_id = input_ids[token_id]
            expert_id = T.alloc_local((1,), dtype=T.int64)
            score = T.alloc_local((1,), dtype=T.float32)
            expert_id[0] = 0
            score[0] = 0.0

            if tx < topk:
                expert_id[0] = T.cast(tid2eid[input_id, tx], T.int64)
                logit = T.cast(router_logits[token_id, expert_id[0]], T.float32)
                score[0] = T.sqrt(T.log(1.0 + T.exp(logit)))

            denominator = T.warp_reduce_sum(score[0])
            if tx < output_topk:
                if tx < topk:
                    routed_ids[token_id, tx] = expert_id[0]
                    routed_scores[token_id, tx] = score[0] / T.max(denominator, 1e-20)
                else:
                    routed_ids[token_id, tx] = num_experts + tx - topk
                    routed_scores[token_id, tx] = shared_weight

    return hash_topk_warp_kernel

@lru_cache(maxsize=None)
def _tilelang_hash_topk_warp_block_kernel(
    topk: int,
    num_fused_shared_experts: int,
    threads: int = 128,
    input_dtype: str = "float32",
    input_ids_dtype: str = "int64",
    tid2eid_dtype: str = "int64",
):
    import tilelang
    import tilelang.language as T

    output_topk = topk + num_fused_shared_experts
    if output_topk > threads:
        raise ValueError("warp-block hash_topk kernel requires topk + shared <= threads")

    tl_input_dtype = _hash_topk_input_dtype(input_dtype, T)
    tl_input_ids_dtype = _hash_topk_input_ids_dtype(input_ids_dtype, T)
    tl_tid2eid_dtype = _hash_topk_tid2eid_dtype(tid2eid_dtype, T)
    num_tokens = T.dynamic("num_tokens")
    num_experts = T.dynamic("num_experts")
    vocab_size = T.dynamic("vocab_size")
    router_logits_stride_m = T.dynamic("router_logits_stride_m")
    router_logits_stride_n = T.dynamic("router_logits_stride_n")
    input_ids_stride_m = T.dynamic("input_ids_stride_m")
    tid2eid_stride_m = T.dynamic("tid2eid_stride_m")
    tid2eid_stride_n = T.dynamic("tid2eid_stride_n")
    num_warps = (threads + 31) // 32

    @_tilelang_jit(
        tilelang,
        f"dsv4_hash_topk_warp_block_{input_dtype}_{input_ids_dtype}_{tid2eid_dtype}_k{topk}_s{num_fused_shared_experts}_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=False),
    )
    def hash_topk_warp_block_kernel(
        router_logits: T.StridedTensor[
            (num_tokens, num_experts),
            (router_logits_stride_m, router_logits_stride_n),
            tl_input_dtype,
        ],
        input_ids: T.StridedTensor[(num_tokens,), (input_ids_stride_m,), tl_input_ids_dtype],
        tid2eid: T.StridedTensor[
            (vocab_size, topk),
            (tid2eid_stride_m, tid2eid_stride_n),
            tl_tid2eid_dtype,
        ],
        routed_scores: T.Tensor[(num_tokens, output_topk), T.float32],
        routed_ids: T.Tensor[(num_tokens, output_topk), T.int64],
        shared_weight: T.float32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            warp_id = tx // 32
            lane_id = tx - warp_id * 32
            input_id = input_ids[token_id]
            expert_id = T.alloc_local((1,), dtype=T.int64)
            score = T.alloc_local((1,), dtype=T.float32)
            expert_id[0] = 0
            score[0] = 0.0
            warp_sums = T.alloc_shared((num_warps,), dtype=T.float32)
            denominator = T.alloc_shared((1,), dtype=T.float32)

            if tx < topk:
                expert_id[0] = T.cast(tid2eid[input_id, tx], T.int64)
                logit = T.cast(router_logits[token_id, expert_id[0]], T.float32)
                score[0] = T.sqrt(T.log(1.0 + T.exp(logit)))

            warp_sum = T.warp_reduce_sum(score[0])
            if lane_id == 0:
                warp_sums[warp_id] = warp_sum
            T.sync_threads()

            block_partial = T.if_then_else(tx < num_warps, warp_sums[tx], 0.0)
            block_sum = T.warp_reduce_sum(block_partial)
            if tx == 0:
                denominator[0] = block_sum
            T.sync_threads()

            if tx < output_topk:
                if tx < topk:
                    routed_ids[token_id, tx] = expert_id[0]
                    routed_scores[token_id, tx] = score[0] / T.max(denominator[0], 1e-20)
                else:
                    routed_ids[token_id, tx] = num_experts + tx - topk
                    routed_scores[token_id, tx] = shared_weight

    return hash_topk_warp_block_kernel


@lru_cache(maxsize=None)
def _tilelang_moe_fused_gate_kernel(
    num_experts: int,
    topk: int,
    threads: int = 256,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_threads = 32
    num_aligned_experts = ((num_experts + num_threads - 1) // num_threads) * num_threads

    @tilelang.jit(
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            disable_index_promotion=True,
        )
    )
    def get_moe_fused_gate_kernel():
        @T.prim_func
        def moe_fused_gate_kernel(
            gating_output: T.Tensor[(num_tokens, num_experts), T.float32],
            correction_bias: T.Tensor[(num_experts,), T.float32],
            topk_weights: T.Tensor[(num_tokens, topk), T.float32],
            topk_ids: T.Tensor[(num_tokens, topk), T.int32],
        ):
            with T.Kernel(num_tokens, threads=num_threads) as token_id:
                tx = T.get_thread_binding()
                scores = T.alloc_fragment((num_aligned_experts,), dtype=T.float32)
                amax = T.alloc_fragment((1,), dtype=T.float32)
                idx_reducer = T.alloc_reducer((1,), T.int32, "min", replication="all")
                selected_indices = T.alloc_shared((topk,), dtype=T.int32)
                selected_scores = T.alloc_shared((topk,), dtype=T.float32)
                denom = T.alloc_shared((1,), dtype=T.float32)

                for i in T.Parallel(num_aligned_experts):
                    if i < num_experts:
                        logit = gating_output[token_id, i]
                        score = T.sqrt(T.log(1.0 + T.exp(logit)))
                        scores[i] = score + correction_bias[i]
                    else:
                        scores[i] = -T.infinity(T.float32)

                for kth in T.unroll(topk):
                    T.reduce_max(scores, amax)
                    T.fill(idx_reducer, T.max_value(T.int32))
                    for i in T.Parallel(num_aligned_experts):
                        if scores[i] == amax[0]:
                            idx_reducer[0] = T.min(idx_reducer[0], i)
                    T.finalize_reducer(idx_reducer)
                    selected_indices[kth] = idx_reducer[0]
                    selected_logit = gating_output[token_id, idx_reducer[0]]
                    selected_scores[kth] = T.sqrt(T.log(1.0 + T.exp(selected_logit)))
                    for i in T.Parallel(num_aligned_experts):
                        if i == idx_reducer[0]:
                            scores[i] = -T.infinity(T.float32)

                if tx == 0:
                    denom[0] = 0.0
                    for kth in T.serial(0, topk):
                        denom[0] += selected_scores[kth]
                T.sync_threads()
                if tx < topk:
                    topk_ids[token_id, tx] = selected_indices[tx]
                    topk_weights[token_id, tx] = selected_scores[tx] / T.max(denom[0], 1e-20)

        return moe_fused_gate_kernel

    return get_moe_fused_gate_kernel()

@lru_cache(maxsize=None)
def _tilelang_mask_topk_ids_int64_kernel(topk: int, single_block: bool = False, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_rows = T.dynamic("num_rows")

    if single_block:
        @_tilelang_jit(tilelang, "dsv4_mask_topk_ids_int64_single_block")
        def mask_topk_ids_kernel(
            topk_ids: T.Tensor[(num_rows, topk), T.int64],
            num_token_non_padded: T.Tensor[(1,), T.int32],
        ):
            with T.Kernel(1, threads=threads) as _:
                tx = T.get_thread_binding()
                for elem_base in T.serial(0, num_rows * topk, threads):
                    elem_id = elem_base + tx
                    if elem_id < num_rows * topk:
                        row_id = elem_id // topk
                        if row_id >= num_token_non_padded[0]:
                            topk_ids[row_id, elem_id - row_id * topk] = -1

        return mask_topk_ids_kernel

    @_tilelang_jit(tilelang, "dsv4_mask_topk_ids_int64_parallel")
    def mask_topk_ids_kernel(
        topk_ids: T.Tensor[(num_rows, topk), T.int64],
        num_token_non_padded: T.Tensor[(1,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_rows * topk, threads), threads=threads) as block_id:
            tx = T.get_thread_binding()
            elem_id = block_id * threads + tx
            if elem_id < num_rows * topk:
                row_id = elem_id // topk
                if row_id >= num_token_non_padded[0]:
                    topk_ids[row_id, elem_id - row_id * topk] = -1

    return mask_topk_ids_kernel

@lru_cache(maxsize=None)
def _tilelang_mask_topk_ids_int32_kernel(topk: int, single_block: bool = False, threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_rows = T.dynamic("num_rows")

    if single_block:
        @_tilelang_jit(tilelang, "dsv4_mask_topk_ids_int32_single_block")
        def mask_topk_ids_kernel(
            topk_ids: T.Tensor[(num_rows, topk), T.int32],
            num_token_non_padded: T.Tensor[(1,), T.int32],
        ):
            with T.Kernel(1, threads=threads) as _:
                tx = T.get_thread_binding()
                for elem_base in T.serial(0, num_rows * topk, threads):
                    elem_id = elem_base + tx
                    if elem_id < num_rows * topk:
                        row_id = elem_id // topk
                        if row_id >= num_token_non_padded[0]:
                            topk_ids[row_id, elem_id - row_id * topk] = -1

        return mask_topk_ids_kernel

    @_tilelang_jit(tilelang, "dsv4_mask_topk_ids_int32_parallel")
    def mask_topk_ids_kernel(
        topk_ids: T.Tensor[(num_rows, topk), T.int32],
        num_token_non_padded: T.Tensor[(1,), T.int32],
    ):
        with T.Kernel(T.ceildiv(num_rows * topk, threads), threads=threads) as block_id:
            tx = T.get_thread_binding()
            elem_id = block_id * threads + tx
            if elem_id < num_rows * topk:
                row_id = elem_id // topk
                if row_id >= num_token_non_padded[0]:
                    topk_ids[row_id, elem_id - row_id * topk] = -1

    return mask_topk_ids_kernel

@lru_cache(maxsize=None)
def _tilelang_topk_ids_logical_to_physical_static_kernel(
    topk: int,
    topk_ids_dtype: str,
    map_dtype: str,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    num_rows = T.dynamic("num_rows")
    num_logical_experts = T.dynamic("num_logical_experts")
    topk_ids_dtype = topk_ids_dtype.lower().strip()
    map_dtype = map_dtype.lower().strip()
    tl_topk_ids_dtype = T.int64 if topk_ids_dtype == "int64" else T.int32
    tl_map_dtype = T.int64 if map_dtype == "int64" else T.int32

    @_tilelang_jit(
        tilelang,
        f"dsv4_topk_ids_logical_to_physical_static_{topk_ids_dtype}_{map_dtype}",
    )
    def logical_to_physical_kernel(
        topk_ids: T.Tensor[(num_rows, topk), tl_topk_ids_dtype],
        logical_to_physical_map: T.Tensor[(num_logical_experts,), tl_map_dtype],
    ):
        with T.Kernel(T.ceildiv(num_rows * topk, threads), threads=threads) as block_id:
            tx = T.get_thread_binding()
            elem_id = block_id * threads + tx
            if elem_id < num_rows * topk:
                row_id = elem_id // topk
                col_id = elem_id - row_id * topk
                logical_id = topk_ids[row_id, col_id]
                if logical_id >= 0:
                    topk_ids[row_id, col_id] = logical_to_physical_map[logical_id]

    return logical_to_physical_kernel

def _tilelang_topk_ids_logical_to_physical_static_int32_kernel(topk: int, threads: int = 128):
    return _tilelang_topk_ids_logical_to_physical_static_kernel(
        topk,
        "int32",
        "int32",
        threads,
    )

def _tilelang_topk_ids_logical_to_physical_static_int64_kernel(
    topk: int,
    map_dtype: str = "int64",
    threads: int = 128,
):
    return _tilelang_topk_ids_logical_to_physical_static_kernel(
        topk,
        "int64",
        map_dtype,
        threads,
    )

__all__ = [
    '_tilelang_hash_topk_kernel',
    '_tilelang_hash_topk_warp_kernel',
    '_tilelang_hash_topk_warp_block_kernel',
    '_tilelang_moe_fused_gate_kernel',
    '_tilelang_mask_topk_ids_int64_kernel',
    '_tilelang_mask_topk_ids_int32_kernel',
    '_tilelang_topk_ids_logical_to_physical_static_kernel',
    '_tilelang_topk_ids_logical_to_physical_static_int32_kernel',
    '_tilelang_topk_ids_logical_to_physical_static_int64_kernel',
]
