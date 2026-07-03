from functools import lru_cache

from .kernel_common import _tilelang_jit, _tilelang_musa_burst_reduce_pass_configs


@lru_cache(maxsize=None)
def _tilelang_hc_head_linear_splitk_kernel(
    hidden_size: int,
    hc_mult: int,
    split_k: int,
    token_block: int = 16,
    hidden_block: int = 128,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    assert hc_mult <= 32
    assert hidden_size % split_k == 0
    split_size = hidden_size // split_k
    assert split_size % hidden_block == 0

    num_tokens = T.dynamic("num_tokens")

    @_tilelang_jit(
        tilelang,
        "dsv4_hc_head_linear_splitk_stage0",
        pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang),
    )
    def hc_head_linear_splitk_stage0(
        x: T.Tensor[(num_tokens, hidden_size), T.float32],
        weight: T.Tensor[(hc_mult, hidden_size), T.float32],
        partial: T.Tensor[(split_k, num_tokens, hc_mult), T.float32],
    ):
        with T.Kernel(
            T.ceildiv(num_tokens, token_block), split_k, threads=threads
        ) as (pid_m, pid_k):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            T.clear(out_frag)
            k_base = pid_k * split_size

            for pz in T.Pipelined(split_size // hidden_block, num_stages=2):
                x_smem = T.alloc_shared((token_block, hidden_block), T.float32)
                w_smem = T.alloc_shared((32, hidden_block), T.float32)

                for i, k in T.Parallel(token_block, hidden_block):
                    token = pid_m * token_block + i
                    x_smem[i, k] = T.if_then_else(
                        token < num_tokens,
                        x[token, k_base + pz * hidden_block + k],
                        0.0,
                    )
                for j, k in T.Parallel(32, hidden_block):
                    w_smem[j, k] = T.if_then_else(
                        j < hc_mult,
                        weight[j, k_base + pz * hidden_block + k],
                        0.0,
                    )

                for kk in T.serial(hidden_block):
                    for i, j in T.Parallel(token_block, 32):
                        out_frag[i, j] += x_smem[i, kk] * w_smem[j, kk]

            for i, j in T.Parallel(token_block, 32):
                token = pid_m * token_block + i
                if token < num_tokens and j < hc_mult:
                    partial[pid_k, token, j] = out_frag[i, j]

    @_tilelang_jit(
        tilelang,
        "dsv4_hc_head_linear_splitk_stage1",
        pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang),
    )
    def hc_head_linear_splitk_stage1(
        partial: T.Tensor[(split_k, num_tokens, hc_mult), T.float32],
        out: T.Tensor[(num_tokens, hc_mult), T.float32],
    ):
        warps_per_cta = threads // 32
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), threads=threads) as (pid_m,):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            token = pid_m * warps_per_cta + warp
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            if token < num_tokens and lane < hc_mult:
                for pid_k in T.serial(split_k):
                    acc[0] += partial[pid_k, token, lane]
                out[token, lane] = acc[0]

    return hc_head_linear_splitk_stage0, hc_head_linear_splitk_stage1


@lru_cache(maxsize=None)
def _tilelang_hc_head_linear_splitk_warp_kernel(
    hidden_size: int,
    hc_mult: int,
    split_k: int,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    assert hc_mult == 4
    assert hidden_size % split_k == 0
    split_size = hidden_size // split_k
    assert split_size % 32 == 0
    assert threads % 32 == 0

    num_tokens = T.dynamic("num_tokens")

    @_tilelang_jit(
        tilelang,
        "dsv4_hc_head_linear_splitk_warp_stage0",
        pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang),
    )
    def hc_head_linear_splitk_warp_stage0(
        x: T.Tensor[(num_tokens, hidden_size), T.float32],
        weight: T.Tensor[(hc_mult, hidden_size), T.float32],
        partial: T.Tensor[(split_k, num_tokens, hc_mult), T.float32],
    ):
        warps_per_cta = threads // 32
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), split_k, threads=threads) as (
            pid_m,
            pid_k,
        ):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            token = pid_m * warps_per_cta + warp
            k_base = pid_k * split_size
            acc0 = T.alloc_local((1,), T.float32)
            acc1 = T.alloc_local((1,), T.float32)
            acc2 = T.alloc_local((1,), T.float32)
            acc3 = T.alloc_local((1,), T.float32)
            acc0[0] = 0.0
            acc1[0] = 0.0
            acc2[0] = 0.0
            acc3[0] = 0.0

            if token < num_tokens:
                for kk in T.serial(split_size // 32):
                    k = kk * 32 + lane
                    xv = x[token, k_base + k]
                    acc0[0] += xv * weight[0, k_base + k]
                    acc1[0] += xv * weight[1, k_base + k]
                    acc2[0] += xv * weight[2, k_base + k]
                    acc3[0] += xv * weight[3, k_base + k]

                sum0 = T.warp_reduce_sum(acc0[0])
                sum1 = T.warp_reduce_sum(acc1[0])
                sum2 = T.warp_reduce_sum(acc2[0])
                sum3 = T.warp_reduce_sum(acc3[0])
                if lane == 0:
                    partial[pid_k, token, 0] = sum0
                    partial[pid_k, token, 1] = sum1
                    partial[pid_k, token, 2] = sum2
                    partial[pid_k, token, 3] = sum3

    @_tilelang_jit(
        tilelang,
        "dsv4_hc_head_linear_splitk_warp_stage1",
        pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang),
    )
    def hc_head_linear_splitk_warp_stage1(
        partial: T.Tensor[(split_k, num_tokens, hc_mult), T.float32],
        out: T.Tensor[(num_tokens, hc_mult), T.float32],
    ):
        with T.Kernel(num_tokens, threads=threads) as (token,):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            if warp < hc_mult:
                for kk in T.serial(T.ceildiv(split_k, 32)):
                    pid_k = kk * 32 + lane
                    acc[0] += T.if_then_else(
                        pid_k < split_k,
                        partial[pid_k, token, warp],
                        0.0,
                    )
                reduced = T.warp_reduce_sum(acc[0])
                if lane == 0:
                    out[token, warp] = reduced

    return hc_head_linear_splitk_warp_stage0, hc_head_linear_splitk_warp_stage1


@lru_cache(maxsize=None)
def _tilelang_hc_head_fused_splitk_warp_kernel(
    hidden_size: int,
    hc_mult: int,
    split_k: int,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    hidden_block: int = 1024,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    assert hc_mult == 4
    flat_hidden_size = hidden_size * hc_mult
    assert flat_hidden_size % split_k == 0
    split_size = flat_hidden_size // split_k
    assert split_size % 32 == 0
    assert hidden_size % hidden_block == 0
    assert threads % 32 == 0

    num_tokens = T.dynamic("num_tokens")

    @_tilelang_jit(
        tilelang,
        "dsv4_hc_head_fused_splitk_warp_stage0",
        pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang),
    )
    def hc_head_fused_splitk_warp_stage0(
        residual: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        weight: T.Tensor[(hc_mult, flat_hidden_size), T.float32],
        partial: T.Tensor[(split_k, num_tokens, hc_mult + 1), T.float32],
    ):
        warps_per_cta = threads // 32
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), split_k, threads=threads) as (
            pid_m,
            pid_k,
        ):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            token = pid_m * warps_per_cta + warp
            k_base = pid_k * split_size
            acc0 = T.alloc_local((1,), T.float32)
            acc1 = T.alloc_local((1,), T.float32)
            acc2 = T.alloc_local((1,), T.float32)
            acc3 = T.alloc_local((1,), T.float32)
            sqsum = T.alloc_local((1,), T.float32)
            acc0[0] = 0.0
            acc1[0] = 0.0
            acc2[0] = 0.0
            acc3[0] = 0.0
            sqsum[0] = 0.0

            if token < num_tokens:
                for kk in T.serial(split_size // 32):
                    k = k_base + kk * 32 + lane
                    mhc_id = k // hidden_size
                    hidden_id = k - mhc_id * hidden_size
                    xv = T.cast(residual[token, mhc_id, hidden_id], T.float32)
                    sqsum[0] += xv * xv
                    acc0[0] += xv * weight[0, k]
                    acc1[0] += xv * weight[1, k]
                    acc2[0] += xv * weight[2, k]
                    acc3[0] += xv * weight[3, k]

                sum0 = T.warp_reduce_sum(acc0[0])
                sum1 = T.warp_reduce_sum(acc1[0])
                sum2 = T.warp_reduce_sum(acc2[0])
                sum3 = T.warp_reduce_sum(acc3[0])
                sumsq = T.warp_reduce_sum(sqsum[0])
                if lane == 0:
                    partial[pid_k, token, 0] = sum0
                    partial[pid_k, token, 1] = sum1
                    partial[pid_k, token, 2] = sum2
                    partial[pid_k, token, 3] = sum3
                    partial[pid_k, token, 4] = sumsq

    @_tilelang_jit(
        tilelang,
        "dsv4_hc_head_fused_splitk_warp_stage1",
        pass_configs=_tilelang_musa_burst_reduce_pass_configs(tilelang),
    )
    def hc_head_fused_splitk_warp_stage1(
        residual: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        partial: T.Tensor[(split_k, num_tokens, hc_mult + 1), T.float32],
        hc_scale: T.Tensor[(1,), T.float32],
        hc_base: T.Tensor[(hc_mult,), T.float32],
        out: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ):
        with T.Kernel(num_tokens, threads=threads) as (token,):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            pre_shared = T.alloc_shared((hc_mult,), T.float32)

            if warp < hc_mult + 1:
                acc = T.alloc_local((1,), T.float32)
                acc[0] = 0.0
                for kk in T.serial(T.ceildiv(split_k, 32)):
                    pid_k = kk * 32 + lane
                    acc[0] += T.if_then_else(
                        pid_k < split_k,
                        partial[pid_k, token, warp],
                        0.0,
                    )
                reduced = T.warp_reduce_sum(acc[0])
                if lane == 0 and warp < hc_mult:
                    sqsum = T.alloc_local((1,), T.float32)
                    sqsum[0] = 0.0
                    for pid_k in T.serial(split_k):
                        sqsum[0] += partial[pid_k, token, hc_mult]
                    rrms = T.rsqrt(sqsum[0] / flat_hidden_size + norm_eps)
                    pre_shared[warp] = T.sigmoid(
                        reduced * rrms * hc_scale[0] + hc_base[warp]
                    ) + hc_eps

            T.sync_threads()

            for block in T.Pipelined(hidden_size // hidden_block, num_stages=1):
                vals = T.alloc_fragment((hidden_block,), T.float32)
                T.clear(vals)
                for mhc_id in T.serial(hc_mult):
                    coeff = pre_shared[mhc_id]
                    for h in T.Parallel(hidden_block):
                        hidden_id = block * hidden_block + h
                        vals[h] += coeff * T.cast(
                            residual[token, mhc_id, hidden_id], T.float32
                        )
                T.copy(vals, out[token, block * hidden_block], disable_tma=True)

    return hc_head_fused_splitk_warp_stage0, hc_head_fused_splitk_warp_stage1


__all__ = [
    "_tilelang_hc_head_fused_splitk_warp_kernel",
    "_tilelang_hc_head_linear_splitk_kernel",
    "_tilelang_hc_head_linear_splitk_warp_kernel",
]
