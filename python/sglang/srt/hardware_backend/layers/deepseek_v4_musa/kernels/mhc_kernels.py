import math
import os
from functools import lru_cache

import torch

from .kernel_common import _tilelang_jit


def round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    return (x.view(torch.int32) + 0x1000).view(torch.float32)


def _mhc_pass_configs(tilelang, mode: str = "burst"):
    mode = mode.lower()
    if mode == "none":
        return None

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
    if mode in ("burst", "aggressive", "aggressive_index32"):
        pass_configs.update(
            {
                tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
                tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
                tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
            }
        )
    if mode in ("aggressive", "aggressive_index32"):
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS] = True
        if (
            mode == "aggressive_index32"
            or os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_DISABLE_INDEX_PROMOTION") == "1"
        ):
            pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = True
    if mode not in ("safe", "burst", "aggressive", "aggressive_index32"):
        raise ValueError(
            "MHC TileLang pass config must be one of "
            "'safe', 'burst', 'aggressive', 'aggressive_index32', or 'none', "
            f"got {mode!r}"
        )
    return pass_configs


def _mhc_tme_ws_pass_configs(tilelang):
    return {
        tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }


@lru_cache(maxsize=None)
def _tilelang_mhc_pre_norm_fn_fwd_mul_kernel(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
):
    import tilelang
    import tilelang.language as T

    assert mhc_mult3 <= 32
    assert rms_group_size % hidden_block == 0
    num_tokens = T.dynamic("num_tokens")

    @_tilelang_jit(
        tilelang,
        "dsv4_mhc_pre_norm_fn_fwd_mul",
        pass_configs=_mhc_pass_configs(tilelang, "safe"),
    )
    def mhc_pre_norm_fn_fwd_mul_kernel(
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
        out: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block), n_rms_group) as (
            pid_x,
            pid_y,
        ):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)
            for pz in T.Pipelined(rms_group_size // hidden_block, num_stages=2):
                x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                for i, k in T.Parallel(token_block, hidden_block):
                    t = pid_x * token_block + i
                    x_smem[i, k] = T.if_then_else(
                        t < num_tokens,
                        x[t, pid_y * rms_group_size + pz * hidden_block + k],
                        0,
                    )
                for j, k in T.Parallel(32, hidden_block):
                    fn_smem[j, k] = T.if_then_else(
                        j < mhc_mult3,
                        fn[j, pid_y * rms_group_size + pz * hidden_block + k],
                        0,
                    )

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        v = T.cast(x_smem[i, jj * 4 + j], T.float32)
                        sqrsum_part[i, j] += v * v

                for kk in T.serial(hidden_block):
                    for i, j in T.Parallel(token_block, 32):
                        out_frag[i, j] += (
                            T.cast(x_smem[i, kk], T.float32) * fn_smem[j, kk]
                        )

            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)
            for i in T.Parallel(token_block):
                t = pid_x * token_block + i
                if t < num_tokens:
                    sqrsum[t, pid_y] = sqrsum_l[i]
            for i, j in T.Parallel(token_block, 32):
                t = pid_x * token_block + i
                if t < num_tokens and j < mhc_mult3:
                    out[t, pid_y, j] = out_frag[i, j]

    return mhc_pre_norm_fn_fwd_mul_kernel


@lru_cache(maxsize=None)
def _tilelang_mhc_prenorm_splitk_x_tme_cast_kernel(
    mhc_mult3: int,
    hc_hidden_size: int,
    split_k: int,
    token_block: int = 32,
    hidden_block: int = 128,
    num_stages: int = 2,
    threads: int = 384,
):
    import tilelang
    import tilelang.language as T

    assert mhc_mult3 <= 32
    assert hc_hidden_size % hidden_block == 0
    assert hc_hidden_size % split_k == 0
    split_size = hc_hidden_size // split_k
    assert split_size % hidden_block == 0
    assert hidden_block in (64, 128)
    assert num_stages == 2
    assert token_block == 32
    assert threads >= 384

    num_tokens = T.dynamic("num_tokens")
    mbarrier_list = [128, 128, 128] * num_stages

    @_tilelang_jit(
        tilelang,
        "dsv4_mhc_prenorm_splitk_x_tme_cast_stage0",
        pass_configs=_mhc_tme_ws_pass_configs(tilelang),
    )
    def mhc_prenorm_splitk_x_tme_cast_stage_0(
        x: T.Tensor[(num_tokens, hc_hidden_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, hc_hidden_size), T.float32],
        out_partial: T.Tensor[(split_k, num_tokens, mhc_mult3), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, token_block), split_k, threads=threads) as (
            px,
            bz,
        ):
            x_bf16_shared = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            x_fp32_shared = T.alloc_shared((token_block, hidden_block), T.float32)
            fn_shared = T.alloc_shared((32, hidden_block), T.float32)
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sq_part4 = T.alloc_fragment((token_block, 4), T.float32)
            mbars = T.alloc_barrier(mbarrier_list)

            k_base = bz * split_size

            with T.ws(0):
                T.clear(out_frag)
                T.clear(sq_part4)

            for pz in range(split_size // hidden_block):
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[2],
                        parity=(pz % 2) ^ 1,
                    )
                    T.copy(
                        x[
                            px * token_block : (px + 1) * token_block,
                            k_base
                            + pz * hidden_block : k_base
                            + (pz + 1) * hidden_block,
                        ],
                        x_bf16_shared,
                        barrier=mbars[0],
                        annotations={"musa_tma_k_major": T.int32(1)},
                    )
                    T.copy(
                        fn[
                            0:32,
                            k_base
                            + pz * hidden_block : k_base
                            + (pz + 1) * hidden_block,
                        ],
                        fn_shared,
                        barrier=mbars[1],
                        eviction_policy="evict_first",
                    )
                    T.mbarrier_arrive(mbarrier=mbars[0])
                    T.mbarrier_arrive(mbarrier=mbars[1])

                with T.ws(0):
                    T.mbarrier_wait_parity(mbarrier=mbars[0], parity=pz % 2)
                    T.mbarrier_wait_parity(mbarrier=mbars[1], parity=pz % 2)
                    for i, k in T.Parallel(token_block, hidden_block):
                        x_fp32_shared[i, k] = T.cast(x_bf16_shared[i, k], T.float32)
                    for jj in T.serial(hidden_block // 4):
                        for i, j in T.Parallel(token_block, 4):
                            # GEMM consumes x_fp32_shared through SQMMA layout.
                            # Scalar sqsum must read the TME-loaded bf16 tile
                            # directly; scalar reads from x_fp32_shared are not
                            # layout-equivalent after GEMM lowering.
                            v = T.cast(x_bf16_shared[i, jj * 4 + j], T.float32)
                            sq_part4[i, j] += v * v
                    T.gemm(
                        x_fp32_shared,
                        fn_shared,
                        out_frag,
                        clear_accum=False,
                        transpose_B=True,
                    )
                    T.mbarrier_arrive(mbarrier=mbars[2])

            with T.ws(0):
                sq_l = T.alloc_fragment((token_block,), T.float32)
                T.reduce_sum(sq_part4, sq_l)
                for i in T.Parallel(token_block):
                    t = px * token_block + i
                    if t < num_tokens:
                        sqrsum_partial[bz, t] = sq_l[i]
                for i, j in T.Parallel(token_block, 32):
                    t = px * token_block + i
                    if t < num_tokens and j < mhc_mult3:
                        out_partial[bz, t, j] = out_frag[i, j]

    @_tilelang_jit(
        tilelang,
        "dsv4_mhc_prenorm_splitk_x_tme_cast_stage1",
        pass_configs=_mhc_pass_configs(tilelang, "safe"),
    )
    def mhc_prenorm_splitk_x_tme_cast_stage_1(
        out_partial: T.Tensor[(split_k, num_tokens, mhc_mult3), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens,), T.float32],
    ):
        warps_per_cta = 4
        num_reduce = T.ceildiv(split_k, 32)
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), threads=128) as (px,):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            t = px * warps_per_cta + warp
            s = T.alloc_local((1,), T.float32)
            acc = T.alloc_local((1,), T.float32)
            s[0] = 0
            acc[0] = 0

            if t < num_tokens:
                for r in T.serial(num_reduce):
                    bz = r * 32 + lane
                    s[0] += T.if_then_else(bz < split_k, sqrsum_partial[bz, t], 0.0)
                sqrsum[t] = T.warp_reduce_sum(s[0])
                if lane < mhc_mult3:
                    for bz in T.serial(split_k):
                        acc[0] += out_partial[bz, t, lane]
                    out[t, lane] = acc[0]

    return (
        mhc_prenorm_splitk_x_tme_cast_stage_0,
        mhc_prenorm_splitk_x_tme_cast_stage_1,
    )


@lru_cache(maxsize=None)
def _tilelang_mhc_prenorm_splitk_deepgemm_ws_like_v0_kernel(
    mhc_mult3: int,
    hc_hidden_size: int,
    split_k: int,
    token_block: int = 64,
    hidden_block: int = 32,
    num_stages: int = 2,
    threads: int = 384,
    split_pipelines: bool = False,
    reannotate_layout: bool = False,
    reannotate_x_tma_layout: bool = False,
):
    # The reannotation knobs are intentionally private: direct full/x-only
    # region reannotation is a known-incorrect layout experiment and must not be
    # surfaced in dispatch or benchmark reporting.
    import tilelang
    import tilelang.language as T
    from tilelang.layout import make_sqmma_swizzled_layout

    assert mhc_mult3 <= 32
    assert hc_hidden_size % hidden_block == 0
    assert hc_hidden_size % split_k == 0
    split_size = hc_hidden_size // split_k
    assert split_size % hidden_block == 0
    assert token_block in (32, 64)
    assert hidden_block == 32
    assert num_stages in (2, 3)
    assert threads >= 384

    num_tokens = T.dynamic("num_tokens")
    if split_pipelines:
        # Per stage: A ready, B ready, cast ready, A free, B free, cast free.
        # This is closer to DeepGEMM's independent PipelineA/PipelineB model
        # while still keeping sqrsum in the cast squad for v0.
        mbarrier_list = [128, 128, 128, 128, 128, 128] * num_stages
    else:
        # Per stage: TME ready, cast ready, consumer done.
        mbarrier_list = [128, 128, 128] * num_stages

    @_tilelang_jit(
        tilelang,
        (
            "dsv4_mhc_prenorm_splitk_deepgemm_ws_like_v0_relayout_stage0"
            if reannotate_layout
            else "dsv4_mhc_prenorm_splitk_deepgemm_ws_like_v0_xlayout_stage0"
            if reannotate_x_tma_layout
            else
            "dsv4_mhc_prenorm_splitk_deepgemm_ws_like_v0_ab_stage0"
            if split_pipelines
            else "dsv4_mhc_prenorm_splitk_deepgemm_ws_like_v0_stage0"
        ),
        pass_configs=_mhc_tme_ws_pass_configs(tilelang),
    )
    def mhc_prenorm_splitk_deepgemm_ws_like_v0_stage_0(
        x: T.Tensor[(num_tokens, hc_hidden_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, hc_hidden_size), T.float32],
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, token_block), split_k, threads=threads) as (
            px,
            bz,
        ):
            x_bf16_shared = T.alloc_shared(
                (num_stages, token_block, hidden_block), T.bfloat16
            )
            x_fp32_shared = T.alloc_shared(
                (num_stages, token_block, hidden_block), T.float32
            )
            fn_shared = T.alloc_shared((num_stages, 32, hidden_block), T.float32)
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sq_part4 = T.alloc_fragment((token_block, 4), T.float32)
            mbars = T.alloc_barrier(mbarrier_list)

            k_base = bz * split_size

            with T.ws(2):
                T.clear(out_frag)
            with T.ws(1):
                T.clear(sq_part4)

            for pz in range(split_size // hidden_block):
                if split_pipelines:
                    with T.ws(0):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages + num_stages * 3],
                            parity=((pz // num_stages) % num_stages) ^ 1,
                        )
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages + num_stages * 4],
                            parity=((pz // num_stages) % num_stages) ^ 1,
                        )
                        T.annotate_layout(
                            {
                                x_bf16_shared[
                                    pz % num_stages, :, :
                                ]: make_sqmma_swizzled_layout(
                                    x_bf16_shared[pz % num_stages, :, :],
                                    k_major=True,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.copy(
                            x[
                                px * token_block : (px + 1) * token_block,
                                k_base
                                + pz * hidden_block : k_base
                                + (pz + 1) * hidden_block,
                            ],
                            x_bf16_shared[pz % num_stages, :, :],
                            barrier=mbars[pz % num_stages],
                            annotations={"musa_tma_k_major": T.int32(1)},
                        )
                        T.copy(
                            fn[
                                0:32,
                                k_base
                                + pz * hidden_block : k_base
                                + (pz + 1) * hidden_block,
                            ],
                            fn_shared[pz % num_stages, :, :],
                            barrier=mbars[pz % num_stages + num_stages],
                            eviction_policy="evict_first",
                        )
                        T.mbarrier_arrive(mbarrier=mbars[pz % num_stages])
                        T.mbarrier_arrive(
                            mbarrier=mbars[pz % num_stages + num_stages]
                        )

                    with T.ws(1):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages + num_stages * 5],
                            parity=((pz // num_stages) % num_stages) ^ 1,
                        )
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages],
                            parity=(pz // num_stages) % num_stages,
                        )
                        for i, k in T.Parallel(token_block, hidden_block):
                            x_fp32_shared[pz % num_stages, i, k] = T.cast(
                                x_bf16_shared[pz % num_stages, i, k], T.float32
                            )
                        for jj in T.serial(hidden_block // 4):
                            for i, j in T.Parallel(token_block, 4):
                                v = x_fp32_shared[pz % num_stages, i, jj * 4 + j]
                                sq_part4[i, j] += v * v
                        T.mbarrier_arrive(
                            mbarrier=mbars[pz % num_stages + num_stages * 3]
                        )
                        T.mbarrier_arrive(
                            mbarrier=mbars[pz % num_stages + num_stages * 2]
                        )

                    with T.ws(2):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages + num_stages],
                            parity=(pz // num_stages) % num_stages,
                        )
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages + num_stages * 2],
                            parity=(pz // num_stages) % num_stages,
                        )
                        T.gemm(
                            x_fp32_shared[pz % num_stages, :, :],
                            fn_shared[pz % num_stages, :, :],
                            out_frag,
                            clear_accum=False,
                            transpose_B=True,
                            wg_wait=-1,
                        )
                        T.wait_wgmma(0)
                        T.mbarrier_arrive(
                            mbarrier=mbars[pz % num_stages + num_stages * 4]
                        )
                        T.mbarrier_arrive(
                            mbarrier=mbars[pz % num_stages + num_stages * 5]
                        )
                else:
                    with T.ws(0):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages + num_stages * 2],
                            parity=((pz // num_stages) % num_stages) ^ 1,
                        )
                        T.annotate_layout(
                            {
                                x_bf16_shared[
                                    pz % num_stages, :, :
                                ]: make_sqmma_swizzled_layout(
                                    x_bf16_shared[pz % num_stages, :, :],
                                    k_major=True,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        if reannotate_layout:
                            T.annotate_layout(
                                {
                                    x_bf16_shared[
                                        pz % num_stages, :, :
                                    ]: make_sqmma_swizzled_layout(
                                        x_bf16_shared[pz % num_stages, :, :],
                                        k_major=True,
                                    ),
                                    fn_shared[
                                        pz % num_stages, :, :
                                    ]: make_sqmma_swizzled_layout(
                                        fn_shared[pz % num_stages, :, :],
                                        k_major=True,
                                    ),
                                },
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                        if reannotate_x_tma_layout:
                            T.annotate_layout(
                                {
                                    x_bf16_shared[
                                        pz % num_stages, :, :
                                    ]: make_sqmma_swizzled_layout(
                                        x_bf16_shared[pz % num_stages, :, :],
                                        k_major=True,
                                    ),
                                },
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                        T.copy(
                            x[
                                px * token_block : (px + 1) * token_block,
                                k_base
                                + pz * hidden_block : k_base
                                + (pz + 1) * hidden_block,
                            ],
                            x_bf16_shared[pz % num_stages, :, :],
                            barrier=mbars[pz % num_stages],
                            annotations={"musa_tma_k_major": T.int32(1)},
                        )
                        T.copy(
                            fn[
                                0:32,
                                k_base
                                + pz * hidden_block : k_base
                                + (pz + 1) * hidden_block,
                            ],
                            fn_shared[pz % num_stages, :, :],
                            barrier=mbars[pz % num_stages],
                            eviction_policy="evict_first",
                        )
                        T.mbarrier_arrive(mbarrier=mbars[pz % num_stages])

                    with T.ws(1):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages],
                            parity=(pz // num_stages) % num_stages,
                        )
                        if reannotate_layout:
                            T.annotate_layout(
                                {
                                    x_bf16_shared[
                                        pz % num_stages, :, :
                                    ]: make_sqmma_swizzled_layout(
                                        x_bf16_shared[pz % num_stages, :, :],
                                        k_major=True,
                                    ),
                                    x_fp32_shared[
                                        pz % num_stages, :, :
                                    ]: make_sqmma_swizzled_layout(
                                        x_fp32_shared[pz % num_stages, :, :],
                                        k_major=True,
                                    ),
                                },
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                        for i, k in T.Parallel(token_block, hidden_block):
                            x_fp32_shared[pz % num_stages, i, k] = T.cast(
                                x_bf16_shared[pz % num_stages, i, k], T.float32
                            )
                        for jj in T.serial(hidden_block // 4):
                            for i, j in T.Parallel(token_block, 4):
                                v = x_fp32_shared[pz % num_stages, i, jj * 4 + j]
                                sq_part4[i, j] += v * v
                        T.mbarrier_arrive(mbarrier=mbars[pz % num_stages + num_stages])

                    with T.ws(2):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[pz % num_stages + num_stages],
                            parity=(pz // num_stages) % num_stages,
                        )
                        if reannotate_layout:
                            T.annotate_layout(
                                {
                                    x_fp32_shared[
                                        pz % num_stages, :, :
                                    ]: make_sqmma_swizzled_layout(
                                        x_fp32_shared[pz % num_stages, :, :],
                                        k_major=True,
                                    ),
                                    fn_shared[
                                        pz % num_stages, :, :
                                    ]: make_sqmma_swizzled_layout(
                                        fn_shared[pz % num_stages, :, :],
                                        k_major=True,
                                    ),
                                },
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                        T.gemm(
                            x_fp32_shared[pz % num_stages, :, :],
                            fn_shared[pz % num_stages, :, :],
                            out_frag,
                            clear_accum=False,
                            transpose_B=True,
                            wg_wait=-1,
                        )
                        T.wait_wgmma(0)
                        T.mbarrier_arrive(
                            mbarrier=mbars[pz % num_stages + num_stages * 2]
                        )

            with T.ws(1):
                sq_l = T.alloc_fragment((token_block,), T.float32)
                T.reduce_sum(sq_part4, sq_l)
                for i in T.Parallel(token_block):
                    t = px * token_block + i
                    if t < num_tokens:
                        sqrsum_partial[bz, t] = sq_l[i]

            with T.ws(2):
                for i, j in T.Parallel(token_block, 32):
                    t = px * token_block + i
                    if t < num_tokens and j < mhc_mult3:
                        out_partial[bz, t, j] = out_frag[i, j]

    @_tilelang_jit(
        tilelang,
        (
            "dsv4_mhc_prenorm_splitk_deepgemm_ws_like_v0_relayout_stage1"
            if reannotate_layout
            else "dsv4_mhc_prenorm_splitk_deepgemm_ws_like_v0_xlayout_stage1"
            if reannotate_x_tma_layout
            else
            "dsv4_mhc_prenorm_splitk_deepgemm_ws_like_v0_ab_stage1"
            if split_pipelines
            else "dsv4_mhc_prenorm_splitk_deepgemm_ws_like_v0_stage1"
        ),
        pass_configs=_mhc_pass_configs(tilelang, "safe"),
    )
    def mhc_prenorm_splitk_deepgemm_ws_like_v0_stage_1(
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens,), T.float32],
    ):
        warps_per_cta = 4
        num_reduce = T.ceildiv(split_k, 32)
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), threads=128) as (px,):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            t = px * warps_per_cta + warp
            s = T.alloc_local((1,), T.float32)
            acc = T.alloc_local((1,), T.float32)
            s[0] = 0
            acc[0] = 0

            if t < num_tokens:
                for r in T.serial(num_reduce):
                    bz = r * 32 + lane
                    s[0] += T.if_then_else(bz < split_k, sqrsum_partial[bz, t], 0.0)
                sqrsum[t] = T.warp_reduce_sum(s[0])
                if lane < mhc_mult3:
                    for bz in T.serial(split_k):
                        acc[0] += out_partial[bz, t, lane]
                    out[t, lane] = acc[0]

    return (
        mhc_prenorm_splitk_deepgemm_ws_like_v0_stage_0,
        mhc_prenorm_splitk_deepgemm_ws_like_v0_stage_1,
    )


@lru_cache(maxsize=None)
def _tilelang_mhc_pre_big_fuse_kernel(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    mhc_mult: int = 4,
    threads: int = 256,
    hidden_block: int = 512,
    pass_config: str = "burst",
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(hidden_block, hidden_size)
    assert threads in (128, 256)
    assert hidden_block > 0
    assert hidden_size % hidden_block == 0

    @_tilelang_jit(
        tilelang,
        "dsv4_mhc_pre_big_fuse",
        pass_configs=_mhc_pass_configs(tilelang, pass_config),
    )
    def mhc_pre_big_fuse_kernel(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=threads) as pid:
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                if n_splits == 1:
                    rms[0] = gemm_out_sqrsum[0, pid]
                else:
                    tx = T.get_thread_binding()
                    rms_part = T.alloc_fragment(1, T.float32)
                    rms_part[0] = 0
                    for i_base in T.serial(T.ceildiv(n_splits, 32)):
                        i_split = i_base * 32 + tx
                        rms_part[0] += T.if_then_else(
                            i_split < n_splits,
                            gemm_out_sqrsum[i_split, pid],
                            0.0,
                        )
                    rms[0] = T.warp_reduce_sum(rms_part[0])
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            T.sync_threads()

            if T.get_thread_binding() < 32:
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * mhc_scale[0] + mhc_base[j])
                        + mhc_pre_eps
                    )
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = (
                        T.sigmoid(
                            mixes_shared[j + mhc_mult] * mhc_scale[1]
                            + mhc_base[j + mhc_mult]
                        )
                        * mhc_post_mult_value
                    )
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = (
                        mixes_shared[j * mhc_mult + k + mhc_mult * 2]
                        * mhc_scale[2]
                        + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                    )

                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]

            T.sync_threads()

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=1):
                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_mhc in T.serial(mhc_mult):
                    pre = pre_mix_shared[i_mhc]
                    for i1_h in T.Parallel(hidden_block):
                        h = i0_h * hidden_block + i1_h
                        ol[i1_h] += pre * T.cast(residual[pid, i_mhc, h], T.float32)

                T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return mhc_pre_big_fuse_kernel


@lru_cache(maxsize=None)
def _tilelang_mhc_pre_big_fuse_decode_split_kernel(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    mhc_mult: int = 4,
    threads: int = 256,
    hidden_block: int = 512,
    pass_config: str = "burst",
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(hidden_block, hidden_size)
    num_hidden_tiles = hidden_size // hidden_block
    assert threads in (128, 256)
    assert hidden_block > 0
    assert hidden_size % hidden_block == 0
    assert n_splits > 0
    assert sinkhorn_repeat > 0

    @_tilelang_jit(
        tilelang,
        "dsv4_mhc_pre_big_fuse_decode_split",
        pass_configs=_mhc_pass_configs(tilelang, pass_config),
    )
    def mhc_pre_big_fuse_decode_split_kernel(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, num_hidden_tiles, threads=threads) as (pid, pid_h):
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            rms_for_layer = T.alloc_fragment(1, T.float32)
            pre_mix = T.alloc_fragment(mhc_mult, T.float32)
            rms_for_layer[0] = 0
            for i_split in T.serial(n_splits):
                rms_for_layer[0] += gemm_out_sqrsum[i_split, pid]
            rms_for_layer[0] = T.rsqrt(
                rms_for_layer[0] / (mhc_mult * hidden_size) + rms_eps
            )
            for j in T.serial(mhc_mult):
                pre_mix[j] = 0
                for i_split in T.serial(n_splits):
                    pre_mix[j] += gemm_out_mul[i_split, pid, j]
                pre_mix[j] = (
                    T.sigmoid(
                        pre_mix[j] * rms_for_layer[0] * mhc_scale[0]
                        + mhc_base[j]
                    )
                    + mhc_pre_eps
                )

            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                if n_splits == 1:
                    rms[0] = gemm_out_sqrsum[0, pid]
                else:
                    tx = T.get_thread_binding()
                    rms_part = T.alloc_fragment(1, T.float32)
                    rms_part[0] = 0
                    for i_base in T.serial(T.ceildiv(n_splits, 32)):
                        i_split = i_base * 32 + tx
                        rms_part[0] += T.if_then_else(
                            i_split < n_splits,
                            gemm_out_sqrsum[i_split, pid],
                            0.0,
                        )
                    rms[0] = T.warp_reduce_sum(rms_part[0])
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                if pid_h == 0:
                    for j in T.Parallel(mhc_mult3):
                        mixes[j] = 0
                        for i_split in T.serial(n_splits):
                            mixes[j] += gemm_out_mul[i_split, pid, j]
                        mixes[j] *= rms[0]
                    T.copy(mixes, mixes_shared, disable_tma=True)

            T.sync_threads()

            if T.get_thread_binding() < 32:
                if pid_h == 0:
                    cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                    for j in T.Parallel(mhc_mult):
                        post_mix[pid, j] = (
                            T.sigmoid(
                                mixes_shared[j + mhc_mult] * mhc_scale[1]
                                + mhc_base[j + mhc_mult]
                            )
                            * mhc_post_mult_value
                        )
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = (
                            mixes_shared[j * mhc_mult + k + mhc_mult * 2]
                            * mhc_scale[2]
                            + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                        )

                    row_sum = T.alloc_fragment(mhc_mult, T.float32)
                    col_sum = T.alloc_fragment(mhc_mult, T.float32)
                    row_max = T.alloc_fragment(mhc_mult, T.float32)
                    T.reduce_max(cm, row_max, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = T.exp(cm[j, k] - row_max[j])
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                    for _ in T.serial(sinkhorn_repeat - 1):
                        T.reduce_sum(cm, row_sum, dim=1)
                        for j, k in T.Parallel(mhc_mult, mhc_mult):
                            cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                        T.reduce_sum(cm, col_sum, dim=0)
                        for j, k in T.Parallel(mhc_mult, mhc_mult):
                            cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        comb_mix[pid, j * mhc_mult + k] = cm[j, k]

            ol = T.alloc_fragment(hidden_block, T.float32)
            T.clear(ol)
            for i_mhc in T.serial(mhc_mult):
                pre = pre_mix[i_mhc]
                for i1_h in T.Parallel(hidden_block):
                    h = pid_h * hidden_block + i1_h
                    ol[i1_h] += pre * T.cast(residual[pid, i_mhc, h], T.float32)

            for i1_h in T.Parallel(hidden_block):
                layer_input[pid, pid_h * hidden_block + i1_h] = T.cast(
                    ol[i1_h], T.bfloat16
                )

    return mhc_pre_big_fuse_decode_split_kernel


@lru_cache(maxsize=None)
def _tilelang_mhc_post_kernel(
    hidden_size: int,
    mhc_mult: int = 4,
    threads: int = 128,
    hidden_block: int = 1024,
    pass_config: str = "safe",
    direct_store: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    hidden_block = math.gcd(hidden_block, hidden_size)
    assert mhc_mult == 4
    assert threads in (128, 256)
    assert hidden_block > 0
    assert hidden_size % hidden_block == 0

    @_tilelang_jit(
        tilelang,
        "dsv4_mhc_post",
        pass_configs=_mhc_pass_configs(tilelang, pass_config),
    )
    def mhc_post_kernel(
        comb_res_mix: T.Tensor[(num_tokens, mhc_mult, mhc_mult), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        out: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=threads) as pid:
            out_shared = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            residual_shared = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            layer_input_shared = T.alloc_shared(hidden_block, T.bfloat16)

            out_local = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
            residual_local = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
            layer_input_local = T.alloc_fragment(hidden_block, T.float32)
            comb_local = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
            post_local = T.alloc_fragment(mhc_mult, T.float32)

            T.copy(comb_res_mix[pid, 0, 0], comb_local)
            T.copy(post_layer_mix[pid, 0], post_local)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                T.copy(
                    residual[pid, 0, i0_h * hidden_block],
                    residual_shared,
                )
                T.copy(
                    layer_input[pid, i0_h * hidden_block],
                    layer_input_shared,
                )
                T.copy(residual_shared, residual_local)
                T.copy(layer_input_shared, layer_input_local)

                for i_hco, i1_h in T.Parallel(mhc_mult, hidden_block):
                    out_local[i_hco, i1_h] = (
                        post_local[i_hco] * layer_input_local[i1_h]
                    )
                    for i_hci in T.serial(mhc_mult):
                        out_local[i_hco, i1_h] += (
                            comb_local[i_hci, i_hco] * residual_local[i_hci, i1_h]
                        )

                if direct_store:
                    T.copy(
                        out_local,
                        out[pid, 0, i0_h * hidden_block],
                        disable_tma=True,
                    )
                else:
                    T.copy(out_local, out_shared)
                    T.copy(
                        out_shared,
                        out[pid, 0, i0_h * hidden_block],
                    )

    return mhc_post_kernel


@lru_cache(maxsize=None)
def _tilelang_mhc_post_2d_kernel(
    hidden_size: int,
    mhc_mult: int = 4,
    threads: int = 128,
    hidden_block: int = 1024,
    pass_config: str = "safe",
    direct_store: bool = False,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    hidden_block = math.gcd(hidden_block, hidden_size)
    assert mhc_mult == 4
    assert threads in (128, 256)
    assert hidden_block > 0
    assert hidden_size % hidden_block == 0

    @_tilelang_jit(
        tilelang,
        "dsv4_mhc_post_2d",
        pass_configs=_mhc_pass_configs(tilelang, pass_config),
    )
    def mhc_post_2d_kernel(
        comb_res_mix: T.Tensor[(num_tokens, mhc_mult, mhc_mult), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        out: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(
            num_tokens, hidden_size // hidden_block, threads=threads
        ) as (pid, hid):
            out_shared = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            residual_shared = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
            layer_input_shared = T.alloc_shared(hidden_block, T.bfloat16)

            out_local = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
            residual_local = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
            layer_input_local = T.alloc_fragment(hidden_block, T.float32)
            comb_local = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
            post_local = T.alloc_fragment(mhc_mult, T.float32)
            h_start = hid * hidden_block

            T.copy(comb_res_mix[pid, 0, 0], comb_local)
            T.copy(post_layer_mix[pid, 0], post_local)
            T.copy(residual[pid, 0, h_start], residual_shared)
            T.copy(layer_input[pid, h_start], layer_input_shared)
            T.copy(residual_shared, residual_local)
            T.copy(layer_input_shared, layer_input_local)

            for i_hco, i1_h in T.Parallel(mhc_mult, hidden_block):
                out_local[i_hco, i1_h] = post_local[i_hco] * layer_input_local[i1_h]
                for i_hci in T.serial(mhc_mult):
                    out_local[i_hco, i1_h] += (
                        comb_local[i_hci, i_hco] * residual_local[i_hci, i1_h]
                    )

            if direct_store:
                T.copy(out_local, out[pid, 0, h_start], disable_tma=True)
            else:
                T.copy(out_local, out_shared)
                T.copy(out_shared, out[pid, 0, h_start])

    return mhc_post_2d_kernel
