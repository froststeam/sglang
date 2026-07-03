from functools import lru_cache


WO_A_D = 4096
WO_A_R = 1024


@lru_cache(maxsize=None)
def _tilelang_wo_a_large_gemm_kernel(
    a_stride_t: int,
    b_stride_r: int = WO_A_D,
    c_stride_t: int = WO_A_R,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 3,
):
    import tilelang
    import tilelang.language as T

    from .kernel_common import _tilelang_jit, _tilelang_musa_aggressive_pass_configs

    num_tokens = T.dynamic("num_tokens")

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_wo_a_large_gemm_as{a_stride_t}_bm{block_m}_bn{block_n}"
            f"_bs{b_stride_r}_cs{c_stride_t}_bk{block_k}_th{threads}_st{num_stages}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            disable_index_promotion=False,
            compile_profile="ls",
        ),
    )
    def wo_a_large_gemm_kernel(
        a: T.StridedTensor[(num_tokens, WO_A_D), (a_stride_t, 1), T.bfloat16],
        b: T.StridedTensor[(WO_A_R, WO_A_D), (b_stride_r, 1), T.bfloat16],
        c: T.StridedTensor[(num_tokens, WO_A_R), (c_stride_t, 1), T.bfloat16],
    ):
        with T.Kernel(
            T.ceildiv(WO_A_R, block_n),
            T.ceildiv(num_tokens, block_m),
            threads=threads,
        ) as (pid_n, pid_m):
            a_shared = T.alloc_shared((block_m, block_k), T.bfloat16)
            b_shared = T.alloc_shared((block_n, block_k), T.bfloat16)
            c_local = T.alloc_fragment((block_m, block_n), T.float32)

            T.use_swizzle(panel_size=10)
            T.clear(c_local)
            for pid_k in T.Pipelined(WO_A_D // block_k, num_stages=num_stages):
                T.copy(a[pid_m * block_m, pid_k * block_k], a_shared)
                T.copy(b[pid_n * block_n, pid_k * block_k], b_shared)
                T.gemm(
                    a_shared,
                    b_shared,
                    c_local,
                    transpose_B=True,
                    clear_accum=False,
                )
            T.copy(c_local, c[pid_m * block_m, pid_n * block_n])

    return wo_a_large_gemm_kernel


@lru_cache(maxsize=None)
def _tilelang_wo_a_m1_splitk_kernel(
    b_stride_r: int = WO_A_D,
    block_n: int = 2,
    reduce_threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    tile_k = 8
    block_k = reduce_threads * tile_k

    @tilelang.jit(target="musa")
    def wo_a_m1_splitk_kernel(
        a: T.Tensor[(WO_A_D,), T.bfloat16],
        b: T.StridedTensor[(WO_A_R, WO_A_D), (b_stride_r, 1), T.bfloat16],
    ):
        c = T.empty((WO_A_R,), T.bfloat16)
        with T.Kernel(
            T.ceildiv(WO_A_R, block_n), threads=(block_n, reduce_threads)
        ) as pid_n:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            a_local = T.alloc_local((tile_k,), T.bfloat16)
            b_local = T.alloc_local((tile_k,), T.bfloat16)
            accum = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)

            T.clear(accum)
            for bk in T.serial(T.ceildiv(WO_A_D, block_k)):
                for kk in T.vectorized(tile_k):
                    k_idx = bk * block_k + tk * tile_k + kk
                    r_idx = pid_n * block_n + tn
                    if k_idx < WO_A_D and r_idx < WO_A_R:
                        a_local[kk] = a[k_idx]
                        b_local[kk] = b[r_idx, k_idx]
                    else:
                        a_local[kk] = T.cast(0, T.bfloat16)
                        b_local[kk] = T.cast(0, T.bfloat16)
                for kk in T.serial(tile_k):
                    accum[0] += a_local[kk].astype(T.float32) * b_local[kk].astype(
                        T.float32
                    )

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, T.float32)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum[0],
                        True,
                        reduced[0],
                        tk,
                        dtype="handle",
                    )
                )
            if pid_n * block_n + tn < WO_A_R:
                c[pid_n * block_n + tn] = reduced[0]
        return c

    return wo_a_m1_splitk_kernel


@lru_cache(maxsize=None)
def _tilelang_wo_a_small_gemm_kernel(
    block_m: int,
    a_stride_t: int,
    b_stride_r: int = WO_A_D,
    c_stride_t: int = WO_A_R,
    block_n: int = 64,
    block_k: int = 128,
    threads: int = 256,
    num_stages: int = 2,
):
    import tilelang
    import tilelang.language as T

    from .kernel_common import _tilelang_jit

    num_tokens = T.dynamic("num_tokens")
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_wo_a_small_bm{block_m}_as{a_stride_t}"
            f"_bs{b_stride_r}_cs{c_stride_t}_bn{block_n}_bk{block_k}"
            f"_th{threads}_st{num_stages}"
        ),
        pass_configs=pass_configs,
    )
    def wo_a_small_gemm_kernel(
        a: T.StridedTensor[(num_tokens, WO_A_D), (a_stride_t, 1), T.bfloat16],
        b: T.StridedTensor[(WO_A_R, WO_A_D), (b_stride_r, 1), T.bfloat16],
        c: T.StridedTensor[(num_tokens, WO_A_R), (c_stride_t, 1), T.bfloat16],
    ):
        with T.Kernel(T.ceildiv(WO_A_R, block_n), threads=threads) as pid_n:
            a_shared = T.alloc_shared((num_stages, block_m, block_k), T.bfloat16)
            b_shared = T.alloc_shared((num_stages, block_n, block_k), T.bfloat16)
            c_local = T.alloc_fragment((block_m, block_n), T.float32)
            mbars = T.alloc_barrier([128, 128] * num_stages)

            with T.ws(0):
                T.clear(c_local)

            for pid_k in T.serial(T.ceildiv(WO_A_D, block_k)):
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[pid_k % num_stages + num_stages],
                        parity=((pid_k // num_stages) % 2) ^ 1,
                    )
                    T.copy(
                        a[
                            0:block_m,
                            pid_k * block_k : (pid_k + 1) * block_k,
                        ],
                        a_shared[pid_k % num_stages, :, :],
                        barrier=mbars[pid_k % num_stages],
                    )
                    T.copy(
                        b[
                            pid_n * block_n : (pid_n + 1) * block_n,
                            pid_k * block_k : (pid_k + 1) * block_k,
                        ],
                        b_shared[pid_k % num_stages, :, :],
                        barrier=mbars[pid_k % num_stages],
                    )
                    T.mbarrier_arrive(mbarrier=mbars[pid_k % num_stages])
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[pid_k % num_stages],
                        parity=(pid_k // num_stages) % 2,
                    )
                    T.gemm(
                        a_shared[pid_k % num_stages, :, :],
                        b_shared[pid_k % num_stages, :, :],
                        c_local,
                        transpose_B=True,
                    )
                    T.mbarrier_arrive(
                        mbarrier=mbars[pid_k % num_stages + num_stages]
                    )

            with T.ws(0):
                for mi, ni in T.Parallel(block_m, block_n):
                    r_idx = pid_n * block_n + ni
                    if mi < num_tokens and r_idx < WO_A_R:
                        c[mi, r_idx] = c_local[mi, ni]

    return wo_a_small_gemm_kernel


@lru_cache(maxsize=None)
def _tilelang_wo_a_small_static_gemm_kernel(
    num_tokens_static: int,
    block_m: int,
    a_stride_t: int = WO_A_D,
    b_stride_r: int = WO_A_D,
    c_stride_t: int = WO_A_R,
    block_n: int = 32,
    block_k: int = 128,
    threads: int = 256,
    num_stages: int = 2,
):
    import tilelang
    import tilelang.language as T

    from .kernel_common import _tilelang_jit

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_wo_a_small_static_t{num_tokens_static}_bm{block_m}"
            f"_as{a_stride_t}_bs{b_stride_r}_cs{c_stride_t}_bn{block_n}"
            f"_bk{block_k}_th{threads}_st{num_stages}"
        ),
        pass_configs=pass_configs,
    )
    def wo_a_small_static_gemm_kernel(
        a: T.StridedTensor[
            (num_tokens_static, WO_A_D), (a_stride_t, 1), T.bfloat16
        ],
        b: T.StridedTensor[(WO_A_R, WO_A_D), (b_stride_r, 1), T.bfloat16],
        c: T.StridedTensor[(num_tokens_static, WO_A_R), (c_stride_t, 1), T.bfloat16],
    ):
        with T.Kernel(T.ceildiv(WO_A_R, block_n), threads=threads) as pid_n:
            a_shared = T.alloc_shared((num_stages, block_m, block_k), T.bfloat16)
            b_shared = T.alloc_shared((num_stages, block_n, block_k), T.bfloat16)
            c_local = T.alloc_fragment((block_m, block_n), T.float32)
            mbars = T.alloc_barrier([128, 128] * num_stages)

            with T.ws(0):
                T.clear(c_local)

            for pid_k in T.serial(T.ceildiv(WO_A_D, block_k)):
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[pid_k % num_stages + num_stages],
                        parity=((pid_k // num_stages) % 2) ^ 1,
                    )
                    T.copy(
                        a[
                            0:block_m,
                            pid_k * block_k : (pid_k + 1) * block_k,
                        ],
                        a_shared[pid_k % num_stages, :, :],
                        barrier=mbars[pid_k % num_stages],
                    )
                    T.copy(
                        b[
                            pid_n * block_n : (pid_n + 1) * block_n,
                            pid_k * block_k : (pid_k + 1) * block_k,
                        ],
                        b_shared[pid_k % num_stages, :, :],
                        barrier=mbars[pid_k % num_stages],
                    )
                    T.mbarrier_arrive(mbarrier=mbars[pid_k % num_stages])
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[pid_k % num_stages],
                        parity=(pid_k // num_stages) % 2,
                    )
                    T.gemm(
                        a_shared[pid_k % num_stages, :, :],
                        b_shared[pid_k % num_stages, :, :],
                        c_local,
                        transpose_B=True,
                    )
                    T.mbarrier_arrive(
                        mbarrier=mbars[pid_k % num_stages + num_stages]
                    )

            with T.ws(0):
                for mi, ni in T.Parallel(block_m, block_n):
                    r_idx = pid_n * block_n + ni
                    if mi < num_tokens_static and r_idx < WO_A_R:
                        c[mi, r_idx] = c_local[mi, ni]

    return wo_a_small_static_gemm_kernel


__all__ = [
    "WO_A_D",
    "WO_A_R",
    "_tilelang_wo_a_large_gemm_kernel",
    "_tilelang_wo_a_m1_splitk_kernel",
    "_tilelang_wo_a_small_gemm_kernel",
    "_tilelang_wo_a_small_static_gemm_kernel",
]
