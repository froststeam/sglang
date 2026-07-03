from functools import lru_cache

import torch

from .kernel_common import _tilelang_jit


def _rmsnorm_variant_suffix(rsqrt_mode: str, mul_order: str) -> str:
    return f"{rsqrt_mode}_{mul_order}"


def _validate_rmsnorm_variants(rsqrt_mode: str, mul_order: str) -> tuple[str, str]:
    rsqrt_mode = (rsqrt_mode or "rsqrt").strip().lower()
    mul_order = (mul_order or "x_rrms_weight").strip().lower()
    if rsqrt_mode not in {
        "rsqrt",
        "recip_sqrt",
        "recip_sqrt_newton",
        "ieee_frsqrt",
        "ieee_recip_sqrt",
        "mudnn_fast_rsqrt",
    }:
        raise ValueError(
            f"Unsupported RMSNorm rsqrt_mode={rsqrt_mode!r}; expected one of "
            "rsqrt,recip_sqrt,recip_sqrt_newton,ieee_frsqrt,"
            "ieee_recip_sqrt,mudnn_fast_rsqrt"
        )
    if mul_order not in {"x_rrms_weight", "x_weight_rrms"}:
        raise ValueError(
            f"Unsupported RMSNorm mul_order={mul_order!r}; expected one of "
            "x_rrms_weight,x_weight_rrms"
        )
    return rsqrt_mode, mul_order


def _rmsnorm_rrms_expr(T, value, rsqrt_mode: str):
    if rsqrt_mode == "rsqrt":
        return T.rsqrt(value)
    if rsqrt_mode == "recip_sqrt":
        return T.cast(1.0, "float32") / T.sqrt(value)
    if rsqrt_mode == "recip_sqrt_newton":
        # Test whether refining the division-based reciprocal improves bf16
        # exactness against torch/muDNN in E2E RMSNorm paths.
        half_value = T.cast(0.5, "float32") * value
        y = T.cast(1.0, "float32") / T.sqrt(value)
        return y * (T.cast(1.5, "float32") - half_value * y * y)
    if rsqrt_mode == "ieee_frsqrt":
        return T.ieee_frsqrt(value)
    if rsqrt_mode == "ieee_recip_sqrt":
        return T.cast(1.0, "float32") / T.ieee_fsqrt(value, "rn")
    if rsqrt_mode == "mudnn_fast_rsqrt":
        # Match muDNN fast_rsqrtf: __frsqrt_rn plus one Newton refinement.
        half_value = T.cast(0.5, "float32") * value
        y = T.ieee_frsqrt(value)
        return y * (T.cast(1.5, "float32") - half_value * y * y)
    raise AssertionError(f"unreachable rsqrt_mode={rsqrt_mode}")


def _rmsnorm_weighted_out_expr(T, x_f32, rrms, weight_f32, mul_order: str):
    if mul_order == "x_rrms_weight":
        return (x_f32 * rrms) * weight_f32
    if mul_order == "x_weight_rrms":
        return (x_f32 * weight_f32) * rrms
    raise AssertionError(f"unreachable mul_order={mul_order}")


def _rmsnorm_rcp_expr(T, value, rcp_mode: str):
    if rcp_mode == "ieee_frcp":
        return T.ieee_frcp(value)
    if rcp_mode == "ieee_frcp_newton":
        rcp = T.ieee_frcp(value)
        return rcp * (T.cast(2.0, "float32") - value * rcp)
    raise AssertionError(f"unreachable rcp_mode={rcp_mode}")


def _validate_rmsnorm_rcp_mode(rcp_mode: str) -> str:
    rcp_mode = (rcp_mode or "ieee_frcp_newton").strip().lower()
    if rcp_mode not in {"ieee_frcp", "ieee_frcp_newton"}:
        raise ValueError(
            f"Unsupported RMSNorm rcp_mode={rcp_mode!r}; expected one of "
            "ieee_frcp,ieee_frcp_newton"
        )
    return rcp_mode


@lru_cache(maxsize=None)
def _tilelang_rmsnorm_self_kernel(hidden_size: int, threads: int = 64):
    import tilelang
    import tilelang.language as T

    num_rows = T.dynamic("num_rows")

    @_tilelang_jit(
        tilelang,
        f"dsv4_rmsnorm_self_h{hidden_size}_t{threads}",
    )
    def rmsnorm_self_kernel(
        q: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(num_rows, threads=threads) as row_id:
            q_local = T.alloc_fragment((1, hidden_size), T.bfloat16)
            xsq_f32 = T.alloc_fragment((1, hidden_size), "float32")
            sumsq = T.alloc_fragment((1,), "float32")
            rrms = T.alloc_fragment((1,), "float32")

            for col in T.Parallel(hidden_size):
                q_local[0, col] = q[row_id, col]
                value = T.cast(q_local[0, col], "float32")
                xsq_f32[0, col] = value * value

            T.reduce_sum(xsq_f32, sumsq, dim=1)
            rrms[0] = T.rsqrt(sumsq[0] / float(hidden_size) + eps)

            for col in T.Parallel(hidden_size):
                out[row_id, col] = T.cast(T.cast(q_local[0, col], "float32") * rrms[0], T.bfloat16)

    return rmsnorm_self_kernel


@lru_cache(maxsize=None)
def _tilelang_rmsnorm_self_strided_kernel(hidden_size: int, row_stride: int, threads: int = 64):
    import tilelang
    import tilelang.language as T

    num_rows = T.dynamic("num_rows")

    @_tilelang_jit(
        tilelang,
        f"dsv4_rmsnorm_self_strided_h{hidden_size}_rs{row_stride}_t{threads}",
    )
    def rmsnorm_self_strided_kernel(
        q: T.StridedTensor[(num_rows, hidden_size), (row_stride, 1), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(num_rows, threads=threads) as row_id:
            q_local = T.alloc_fragment((1, hidden_size), T.bfloat16)
            xsq_f32 = T.alloc_fragment((1, hidden_size), "float32")
            sumsq = T.alloc_fragment((1,), "float32")
            rrms = T.alloc_fragment((1,), "float32")

            for col in T.Parallel(hidden_size):
                q_local[0, col] = q[row_id, col]
                value = T.cast(q_local[0, col], "float32")
                xsq_f32[0, col] = value * value

            T.reduce_sum(xsq_f32, sumsq, dim=1)
            rrms[0] = T.rsqrt(sumsq[0] / float(hidden_size) + eps)

            for col in T.Parallel(hidden_size):
                out[row_id, col] = T.cast(T.cast(q_local[0, col], "float32") * rrms[0], T.bfloat16)

    return rmsnorm_self_strided_kernel


@lru_cache(maxsize=None)
def _tilelang_weighted_rmsnorm_kernel(
    hidden_size: int,
    threads: int = 128,
    compile_profile: str | None = None,
    reduce_profile: str = "burst",
    rsqrt_mode: str = "rsqrt",
    mul_order: str = "x_rrms_weight",
):
    import tilelang
    import tilelang.language as T

    from .kernel_common import _tilelang_musa_reduce_profile_pass_configs

    rsqrt_mode, mul_order = _validate_rmsnorm_variants(rsqrt_mode, mul_order)
    num_rows = T.dynamic("num_rows")

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_weighted_rmsnorm_h{hidden_size}_t{threads}"
            f"_{compile_profile or 'default'}_{reduce_profile}"
            f"_{_rmsnorm_variant_suffix(rsqrt_mode, mul_order)}"
        ),
        pass_configs=_tilelang_musa_reduce_profile_pass_configs(
            tilelang,
            compile_profile=compile_profile,
            reduce_profile=reduce_profile,
        ),
    )
    def weighted_rmsnorm_kernel(
        x: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        weight: T.Tensor[(hidden_size,), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(num_rows, threads=threads) as row_id:
            xsq = T.alloc_fragment((hidden_size,), "float32")
            sumsq = T.alloc_fragment((1,), "float32")
            rrms = T.alloc_fragment((1,), "float32")

            for col in T.Parallel(hidden_size):
                value = T.cast(x[row_id, col], "float32")
                xsq[col] = value * value

            T.reduce_sum(xsq, sumsq, dim=0)
            rrms[0] = _rmsnorm_rrms_expr(T, sumsq[0] / float(hidden_size) + eps, rsqrt_mode)

            for col in T.Parallel(hidden_size):
                value = T.cast(x[row_id, col], "float32")
                weight_value = T.cast(weight[col], "float32")
                out[row_id, col] = T.cast(
                    _rmsnorm_weighted_out_expr(T, value, rrms[0], weight_value, mul_order),
                    T.bfloat16,
                )

    return weighted_rmsnorm_kernel


@lru_cache(maxsize=None)
def _tilelang_weighted_rmsnorm_mudnn_like_kernel(
    hidden_size: int,
    threads: int = 128,
    compile_profile: str | None = None,
    rsqrt_mode: str = "mudnn_fast_rsqrt",
    mul_order: str = "x_rrms_weight",
    variance_mode: str = "sum",
    rcp_mode: str = "ieee_frcp_newton",
):
    import tilelang
    import tilelang.language as T

    rsqrt_mode, mul_order = _validate_rmsnorm_variants(rsqrt_mode, mul_order)
    rcp_mode = _validate_rmsnorm_rcp_mode(rcp_mode)
    variance_mode = (variance_mode or "sum").strip().lower()
    if variance_mode not in {"sum", "welford_mean", "chunk_mean"}:
        raise ValueError(f"Unsupported RMSNorm variance_mode={variance_mode!r}")
    if hidden_size % (threads * 8) != 0:
        raise ValueError(
            f"mudnn_like RMSNorm requires hidden_size % (threads * 8) == 0, "
            f"got hidden_size={hidden_size}, threads={threads}"
        )
    num_rows = T.dynamic("num_rows")
    chunks = hidden_size // (threads * 8)

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_weighted_rmsnorm_mudnn_like_h{hidden_size}_t{threads}"
            f"_{variance_mode}_{rcp_mode}"
            f"_{_rmsnorm_variant_suffix(rsqrt_mode, mul_order)}"
        ),
        pass_configs={},
    )
    def weighted_rmsnorm_mudnn_like_kernel(
        x: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        weight: T.Tensor[(hidden_size,), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(num_rows, threads=threads) as row_id:
            tx = T.get_thread_binding()
            sumsq = T.alloc_local((1,), T.float32)
            cnt = T.alloc_local((1,), T.float32)
            mean = T.alloc_local((1,), T.float32)
            delta = T.alloc_local((1,), T.float32)
            rhs_cnt = T.alloc_local((1,), T.float32)
            rhs_mean = T.alloc_local((1,), T.float32)
            n_ab = T.alloc_local((1,), T.float32)
            rcp = T.alloc_local((1,), T.float32)
            rrms = T.alloc_local((1,), T.float32)
            norm_value = T.alloc_local((1,), T.float32)
            y = T.alloc_local((1,), T.float32)
            shared = T.alloc_shared((threads,), T.float32)
            shared_cnt = T.alloc_shared((threads,), T.float32)

            sumsq[0] = 0.0
            cnt[0] = 0.0
            mean[0] = 0.0
            for chunk in T.serial(0, chunks):
                base = chunk * threads * 8 + tx * 8
                for k in T.serial(0, 8):
                    value = T.cast(x[row_id, base + k], T.float32)
                    square = value * value
                    if variance_mode == "welford_mean":
                        cnt[0] += 1.0
                        delta[0] = square - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, cnt[0], rcp_mode)
                        mean[0] += delta[0] * rcp[0]
                    elif variance_mode == "chunk_mean":
                        sumsq[0] += square
                        cnt[0] += 1.0
                    else:
                        sumsq[0] += square

            if variance_mode == "chunk_mean":
                rcp[0] = _rmsnorm_rcp_expr(T, cnt[0], rcp_mode)
                mean[0] = sumsq[0] * rcp[0]
            if variance_mode in {"welford_mean", "chunk_mean"}:
                if threads >= 128:
                    if tx >= 64:
                        shared[tx] = mean[0]
                        shared_cnt[tx] = cnt[0]
                    T.sync_threads()
                    if tx < 64:
                        rhs_mean[0] = shared[tx + 64]
                        rhs_cnt[0] = shared_cnt[tx + 64]
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if threads >= 64:
                    T.sync_threads()
                    if tx >= 32:
                        shared[tx] = mean[0]
                        shared_cnt[tx] = cnt[0]
                    T.sync_threads()
                    if tx < 32:
                        rhs_mean[0] = shared[tx + 32]
                        rhs_cnt[0] = shared_cnt[tx + 32]
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if threads >= 32:
                    rhs_mean[0] = T.shfl_down(mean[0], 16)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 16)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 8)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 8)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 4)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 4)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 2)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 2)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 1)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 1)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if tx == 0:
                    shared[0] = mean[0]
                T.sync_threads()
                norm_value[0] = shared[0] + eps
            else:
                if threads >= 128:
                    if tx >= 64:
                        shared[tx] = sumsq[0]
                    T.sync_threads()
                    if tx < 64:
                        sumsq[0] += shared[tx + 64]
                if threads >= 64:
                    T.sync_threads()
                    if tx >= 32:
                        shared[tx] = sumsq[0]
                    T.sync_threads()
                    if tx < 32:
                        sumsq[0] += shared[tx + 32]
                if threads >= 32:
                    sumsq[0] += T.shfl_down(sumsq[0], 16)
                    sumsq[0] += T.shfl_down(sumsq[0], 8)
                    sumsq[0] += T.shfl_down(sumsq[0], 4)
                    sumsq[0] += T.shfl_down(sumsq[0], 2)
                    sumsq[0] += T.shfl_down(sumsq[0], 1)
                if tx == 0:
                    shared[0] = sumsq[0]
                T.sync_threads()
                sumsq[0] = shared[0]
                norm_value[0] = sumsq[0] / float(hidden_size) + eps
            if rsqrt_mode == "mudnn_fast_rsqrt":
                y[0] = T.ieee_frsqrt(norm_value[0])
                rrms[0] = y[0] * (
                    T.cast(1.5, "float32")
                    - (T.cast(0.5, "float32") * norm_value[0]) * y[0] * y[0]
                )
            else:
                rrms[0] = _rmsnorm_rrms_expr(T, norm_value[0], rsqrt_mode)

            for chunk in T.serial(0, chunks):
                base = chunk * threads * 8 + tx * 8
                for k in T.serial(0, 8):
                    value = T.cast(x[row_id, base + k], T.float32)
                    weight_value = T.cast(weight[base + k], T.float32)
                    out[row_id, base + k] = T.cast(
                        _rmsnorm_weighted_out_expr(T, value, rrms[0], weight_value, mul_order),
                        T.bfloat16,
                    )

    return weighted_rmsnorm_mudnn_like_kernel


@lru_cache(maxsize=None)
def _tilelang_weighted_rmsnorm_strided_mudnn_like_kernel(
    hidden_size: int,
    row_stride: int,
    threads: int = 64,
    compile_profile: str | None = None,
    rsqrt_mode: str = "mudnn_fast_rsqrt",
    mul_order: str = "x_rrms_weight",
    variance_mode: str = "sum",
    rcp_mode: str = "ieee_frcp_newton",
):
    import tilelang
    import tilelang.language as T

    rsqrt_mode, mul_order = _validate_rmsnorm_variants(rsqrt_mode, mul_order)
    rcp_mode = _validate_rmsnorm_rcp_mode(rcp_mode)
    variance_mode = (variance_mode or "sum").strip().lower()
    if variance_mode not in {"sum", "welford_mean", "chunk_mean"}:
        raise ValueError(f"Unsupported RMSNorm variance_mode={variance_mode!r}")
    if hidden_size % (threads * 8) != 0:
        raise ValueError(
            f"mudnn_like RMSNorm requires hidden_size % (threads * 8) == 0, "
            f"got hidden_size={hidden_size}, threads={threads}"
        )
    num_rows = T.dynamic("num_rows")
    chunks = hidden_size // (threads * 8)

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_weighted_rmsnorm_strided_mudnn_like_h{hidden_size}"
            f"_rs{row_stride}_t{threads}_{variance_mode}"
            f"_{_rmsnorm_variant_suffix(rsqrt_mode, mul_order)}"
        ),
        pass_configs={},
    )
    def weighted_rmsnorm_strided_mudnn_like_kernel(
        x: T.StridedTensor[(num_rows, hidden_size), (row_stride, 1), T.bfloat16],
        weight: T.Tensor[(hidden_size,), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(num_rows, threads=threads) as row_id:
            tx = T.get_thread_binding()
            sumsq = T.alloc_local((1,), T.float32)
            cnt = T.alloc_local((1,), T.float32)
            mean = T.alloc_local((1,), T.float32)
            delta = T.alloc_local((1,), T.float32)
            rhs_cnt = T.alloc_local((1,), T.float32)
            rhs_mean = T.alloc_local((1,), T.float32)
            n_ab = T.alloc_local((1,), T.float32)
            rcp = T.alloc_local((1,), T.float32)
            rrms = T.alloc_local((1,), T.float32)
            norm_value = T.alloc_local((1,), T.float32)
            y = T.alloc_local((1,), T.float32)
            shared = T.alloc_shared((threads,), T.float32)
            shared_cnt = T.alloc_shared((threads,), T.float32)

            sumsq[0] = 0.0
            cnt[0] = 0.0
            mean[0] = 0.0
            for chunk in T.serial(0, chunks):
                base = chunk * threads * 8 + tx * 8
                for k in T.serial(0, 8):
                    value = T.cast(x[row_id, base + k], T.float32)
                    square = value * value
                    if variance_mode == "welford_mean":
                        cnt[0] += 1.0
                        delta[0] = square - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, cnt[0], rcp_mode)
                        mean[0] += delta[0] * rcp[0]
                    elif variance_mode == "chunk_mean":
                        sumsq[0] += square
                        cnt[0] += 1.0
                    else:
                        sumsq[0] += square

            if variance_mode == "chunk_mean":
                rcp[0] = _rmsnorm_rcp_expr(T, cnt[0], rcp_mode)
                mean[0] = sumsq[0] * rcp[0]
            if variance_mode in {"welford_mean", "chunk_mean"}:
                if threads >= 128:
                    if tx >= 64:
                        shared[tx] = mean[0]
                        shared_cnt[tx] = cnt[0]
                    T.sync_threads()
                    if tx < 64:
                        rhs_mean[0] = shared[tx + 64]
                        rhs_cnt[0] = shared_cnt[tx + 64]
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if threads >= 64:
                    T.sync_threads()
                    if tx >= 32:
                        shared[tx] = mean[0]
                        shared_cnt[tx] = cnt[0]
                    T.sync_threads()
                    if tx < 32:
                        rhs_mean[0] = shared[tx + 32]
                        rhs_cnt[0] = shared_cnt[tx + 32]
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if threads >= 32:
                    rhs_mean[0] = T.shfl_down(mean[0], 16)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 16)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 8)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 8)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 4)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 4)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 2)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 2)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 1)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 1)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if tx == 0:
                    shared[0] = mean[0]
                T.sync_threads()
                norm_value[0] = shared[0] + eps
            else:
                if threads >= 128:
                    if tx >= 64:
                        shared[tx] = sumsq[0]
                    T.sync_threads()
                    if tx < 64:
                        sumsq[0] += shared[tx + 64]
                if threads >= 64:
                    T.sync_threads()
                    if tx >= 32:
                        shared[tx] = sumsq[0]
                    T.sync_threads()
                    if tx < 32:
                        sumsq[0] += shared[tx + 32]
                if threads >= 32:
                    sumsq[0] += T.shfl_down(sumsq[0], 16)
                    sumsq[0] += T.shfl_down(sumsq[0], 8)
                    sumsq[0] += T.shfl_down(sumsq[0], 4)
                    sumsq[0] += T.shfl_down(sumsq[0], 2)
                    sumsq[0] += T.shfl_down(sumsq[0], 1)
                if tx == 0:
                    shared[0] = sumsq[0]
                T.sync_threads()
                sumsq[0] = shared[0]
                norm_value[0] = sumsq[0] / float(hidden_size) + eps
            if rsqrt_mode == "mudnn_fast_rsqrt":
                y[0] = T.ieee_frsqrt(norm_value[0])
                rrms[0] = y[0] * (
                    T.cast(1.5, "float32")
                    - (T.cast(0.5, "float32") * norm_value[0]) * y[0] * y[0]
                )
            else:
                rrms[0] = _rmsnorm_rrms_expr(T, norm_value[0], rsqrt_mode)

            for chunk in T.serial(0, chunks):
                base = chunk * threads * 8 + tx * 8
                for k in T.serial(0, 8):
                    value = T.cast(x[row_id, base + k], T.float32)
                    weight_value = T.cast(weight[base + k], T.float32)
                    out[row_id, base + k] = T.cast(
                        _rmsnorm_weighted_out_expr(T, value, rrms[0], weight_value, mul_order),
                        T.bfloat16,
                    )

    return weighted_rmsnorm_strided_mudnn_like_kernel


@lru_cache(maxsize=None)
def _tilelang_weighted_rmsnorm_base_offset_mudnn_like_kernel(
    hidden_size: int,
    row_stride: int,
    input_offset: int,
    threads: int = 64,
    rsqrt_mode: str = "mudnn_fast_rsqrt",
    mul_order: str = "x_rrms_weight",
    rcp_mode: str = "ieee_frcp_newton",
    variance_mode: str = "welford_mean",
):
    import tilelang
    import tilelang.language as T

    from .kernel_common import _tilelang_musa_aggressive_pass_configs

    rsqrt_mode, mul_order = _validate_rmsnorm_variants(rsqrt_mode, mul_order)
    rcp_mode = _validate_rmsnorm_rcp_mode(rcp_mode)
    variance_mode = (variance_mode or "welford_mean").strip().lower()
    if variance_mode not in {"welford_mean", "sum"}:
        raise ValueError(f"Unsupported base-offset RMSNorm variance_mode={variance_mode!r}")
    if hidden_size % (threads * 8) != 0:
        raise ValueError(
            f"base-offset RMSNorm requires hidden_size % (threads * 8) == 0, "
            f"got hidden_size={hidden_size}, threads={threads}"
        )
    num_rows = T.dynamic("num_rows")
    chunks = hidden_size // (threads * 8)

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_weighted_rmsnorm_base_offset_h{hidden_size}"
            f"_rs{row_stride}_off{input_offset}_t{threads}_{rcp_mode}"
            f"_{variance_mode}"
            f"_{_rmsnorm_variant_suffix(rsqrt_mode, mul_order)}"
        ),
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            disable_index_promotion=False,
        ),
    )
    def weighted_rmsnorm_base_offset_mudnn_like_kernel(
        x_base: T.Tensor[(num_rows, row_stride), T.bfloat16],
        weight: T.Tensor[(hidden_size,), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(num_rows, threads=threads) as row_id:
            tx = T.get_thread_binding()
            sumsq = T.alloc_local((1,), T.float32)
            cnt = T.alloc_local((1,), T.float32)
            mean = T.alloc_local((1,), T.float32)
            delta = T.alloc_local((1,), T.float32)
            rhs_cnt = T.alloc_local((1,), T.float32)
            rhs_mean = T.alloc_local((1,), T.float32)
            n_ab = T.alloc_local((1,), T.float32)
            rcp = T.alloc_local((1,), T.float32)
            rrms = T.alloc_local((1,), T.float32)
            norm_value = T.alloc_local((1,), T.float32)
            y = T.alloc_local((1,), T.float32)
            shared = T.alloc_shared((threads,), T.float32)
            shared_cnt = T.alloc_shared((threads,), T.float32)

            sumsq[0] = 0.0
            cnt[0] = 0.0
            mean[0] = 0.0
            for chunk in T.serial(0, chunks):
                base = chunk * threads * 8 + tx * 8
                for k in T.serial(0, 8):
                    value = T.cast(x_base[row_id, input_offset + base + k], T.float32)
                    square = value * value
                    if variance_mode == "sum":
                        sumsq[0] += square
                    else:
                        cnt[0] += 1.0
                        delta[0] = square - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, cnt[0], rcp_mode)
                        mean[0] += delta[0] * rcp[0]

            if variance_mode == "sum":
                if threads >= 128:
                    if tx >= 64:
                        shared[tx] = sumsq[0]
                    T.sync_threads()
                    if tx < 64:
                        sumsq[0] += shared[tx + 64]
                if threads >= 64:
                    T.sync_threads()
                    if tx >= 32:
                        shared[tx] = sumsq[0]
                    T.sync_threads()
                    if tx < 32:
                        sumsq[0] += shared[tx + 32]
                if threads >= 32:
                    sumsq[0] += T.shfl_xor(sumsq[0], 16)
                    sumsq[0] += T.shfl_xor(sumsq[0], 8)
                    sumsq[0] += T.shfl_xor(sumsq[0], 4)
                    sumsq[0] += T.shfl_xor(sumsq[0], 2)
                    sumsq[0] += T.shfl_xor(sumsq[0], 1)
                if tx == 0:
                    shared[0] = sumsq[0]
                T.sync_threads()
                norm_value[0] = shared[0] / float(hidden_size) + eps
            else:
                if threads >= 128:
                    if tx >= 64:
                        shared[tx] = mean[0]
                        shared_cnt[tx] = cnt[0]
                    T.sync_threads()
                    if tx < 64:
                        rhs_mean[0] = shared[tx + 64]
                        rhs_cnt[0] = shared_cnt[tx + 64]
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if threads >= 64:
                    T.sync_threads()
                    if tx >= 32:
                        shared[tx] = mean[0]
                        shared_cnt[tx] = cnt[0]
                    T.sync_threads()
                    if tx < 32:
                        rhs_mean[0] = shared[tx + 32]
                        rhs_cnt[0] = shared_cnt[tx + 32]
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if threads >= 32:
                    rhs_mean[0] = T.shfl_down(mean[0], 16)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 16)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 8)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 8)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 4)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 4)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 2)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 2)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                    rhs_mean[0] = T.shfl_down(mean[0], 1)
                    rhs_cnt[0] = T.shfl_down(cnt[0], 1)
                    if rhs_cnt[0] > 0.0:
                        n_ab[0] = rhs_cnt[0] + cnt[0]
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
                if tx == 0:
                    shared[0] = mean[0]
                T.sync_threads()
                norm_value[0] = shared[0] + eps
            if rsqrt_mode == "mudnn_fast_rsqrt":
                y[0] = T.ieee_frsqrt(norm_value[0])
                rrms[0] = y[0] * (
                    T.cast(1.5, "float32")
                    - (T.cast(0.5, "float32") * norm_value[0]) * y[0] * y[0]
                )
            else:
                rrms[0] = _rmsnorm_rrms_expr(T, norm_value[0], rsqrt_mode)

            for chunk in T.serial(0, chunks):
                base = chunk * threads * 8 + tx * 8
                for k in T.serial(0, 8):
                    value = T.cast(x_base[row_id, input_offset + base + k], T.float32)
                    weight_value = T.cast(weight[base + k], T.float32)
                    out[row_id, base + k] = T.cast(
                        _rmsnorm_weighted_out_expr(T, value, rrms[0], weight_value, mul_order),
                        T.bfloat16,
                    )

    return weighted_rmsnorm_base_offset_mudnn_like_kernel


@lru_cache(maxsize=None)
def _tilelang_weighted_rmsnorm_mudnn_like_blocky_kernel(
    hidden_size: int,
    threads: int,
    rows_per_cta: int,
    rsqrt_mode: str = "mudnn_fast_rsqrt",
    mul_order: str = "x_rrms_weight",
    variance_mode: str = "welford_mean",
    rcp_mode: str = "ieee_frcp_newton",
):
    import tilelang
    import tilelang.language as T

    rsqrt_mode, mul_order = _validate_rmsnorm_variants(rsqrt_mode, mul_order)
    rcp_mode = _validate_rmsnorm_rcp_mode(rcp_mode)
    variance_mode = (variance_mode or "welford_mean").strip().lower()
    if variance_mode != "welford_mean":
        raise ValueError("block-y RMSNorm currently supports only welford_mean")
    if hidden_size % (threads * 8) != 0:
        raise ValueError(
            f"block-y RMSNorm requires hidden_size % (threads * 8) == 0, "
            f"got hidden_size={hidden_size}, threads={threads}"
        )
    num_rows = T.dynamic("num_rows")
    chunks = hidden_size // (threads * 8)

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_weighted_rmsnorm_mudnn_like_blocky_h{hidden_size}"
            f"_t{threads}_by{rows_per_cta}_{variance_mode}_{rcp_mode}"
            f"_{_rmsnorm_variant_suffix(rsqrt_mode, mul_order)}"
        ),
        pass_configs={},
    )
    def weighted_rmsnorm_mudnn_like_blocky_kernel(
        x: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        weight: T.Tensor[(hidden_size,), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(T.ceildiv(num_rows, rows_per_cta), threads=(threads, rows_per_cta)) as row_block:
            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            row_id = row_block * rows_per_cta + ty
            sumsq = T.alloc_local((1,), T.float32)
            cnt = T.alloc_local((1,), T.float32)
            mean = T.alloc_local((1,), T.float32)
            delta = T.alloc_local((1,), T.float32)
            rhs_cnt = T.alloc_local((1,), T.float32)
            rhs_mean = T.alloc_local((1,), T.float32)
            n_ab = T.alloc_local((1,), T.float32)
            rcp = T.alloc_local((1,), T.float32)
            rrms = T.alloc_local((1,), T.float32)
            norm_value = T.alloc_local((1,), T.float32)
            y = T.alloc_local((1,), T.float32)
            shared = T.alloc_shared((rows_per_cta, threads), T.float32)
            shared_cnt = T.alloc_shared((rows_per_cta, threads), T.float32)

            sumsq[0] = 0.0
            cnt[0] = 0.0
            mean[0] = 0.0
            if row_id < num_rows:
                for chunk in T.serial(0, chunks):
                    base = chunk * threads * 8 + tx * 8
                    for k in T.serial(0, 8):
                        value = T.cast(x[row_id, base + k], T.float32)
                        square = value * value
                        cnt[0] += 1.0
                        delta[0] = square - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, cnt[0], rcp_mode)
                        mean[0] += delta[0] * rcp[0]

            if threads >= 128:
                if tx >= 64:
                    shared[ty, tx] = mean[0]
                    shared_cnt[ty, tx] = cnt[0]
                T.sync_threads()
                if tx < 64:
                    rhs_mean[0] = shared[ty, tx + 64]
                    rhs_cnt[0] = shared_cnt[ty, tx + 64]
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    if n_ab[0] > 0.0:
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
            if threads >= 64:
                T.sync_threads()
                if tx >= 32:
                    shared[ty, tx] = mean[0]
                    shared_cnt[ty, tx] = cnt[0]
                T.sync_threads()
                if tx < 32:
                    rhs_mean[0] = shared[ty, tx + 32]
                    rhs_cnt[0] = shared_cnt[ty, tx + 32]
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    if n_ab[0] > 0.0:
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
            if threads >= 32:
                rhs_mean[0] = T.shfl_down(mean[0], 16)
                rhs_cnt[0] = T.shfl_down(cnt[0], 16)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
                rhs_mean[0] = T.shfl_down(mean[0], 8)
                rhs_cnt[0] = T.shfl_down(cnt[0], 8)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
                rhs_mean[0] = T.shfl_down(mean[0], 4)
                rhs_cnt[0] = T.shfl_down(cnt[0], 4)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
                rhs_mean[0] = T.shfl_down(mean[0], 2)
                rhs_cnt[0] = T.shfl_down(cnt[0], 2)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
                rhs_mean[0] = T.shfl_down(mean[0], 1)
                rhs_cnt[0] = T.shfl_down(cnt[0], 1)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
            if tx == 0:
                shared[ty, 0] = mean[0]
            T.sync_threads()
            norm_value[0] = shared[ty, 0] + eps
            if rsqrt_mode == "mudnn_fast_rsqrt":
                y[0] = T.ieee_frsqrt(norm_value[0])
                rrms[0] = y[0] * (
                    T.cast(1.5, "float32")
                    - (T.cast(0.5, "float32") * norm_value[0]) * y[0] * y[0]
                )
            else:
                rrms[0] = _rmsnorm_rrms_expr(T, norm_value[0], rsqrt_mode)

            if row_id < num_rows:
                for chunk in T.serial(0, chunks):
                    base = chunk * threads * 8 + tx * 8
                    for k in T.serial(0, 8):
                        value = T.cast(x[row_id, base + k], T.float32)
                        weight_value = T.cast(weight[base + k], T.float32)
                        out[row_id, base + k] = T.cast(
                            _rmsnorm_weighted_out_expr(T, value, rrms[0], weight_value, mul_order),
                            T.bfloat16,
                        )

    return weighted_rmsnorm_mudnn_like_blocky_kernel


@lru_cache(maxsize=None)
def _tilelang_weighted_rmsnorm_strided_mudnn_like_blocky_kernel(
    hidden_size: int,
    row_stride: int,
    threads: int,
    rows_per_cta: int,
    rsqrt_mode: str = "mudnn_fast_rsqrt",
    mul_order: str = "x_rrms_weight",
    variance_mode: str = "welford_mean",
    rcp_mode: str = "ieee_frcp_newton",
):
    import tilelang
    import tilelang.language as T

    rsqrt_mode, mul_order = _validate_rmsnorm_variants(rsqrt_mode, mul_order)
    rcp_mode = _validate_rmsnorm_rcp_mode(rcp_mode)
    variance_mode = (variance_mode or "welford_mean").strip().lower()
    if variance_mode != "welford_mean":
        raise ValueError("block-y RMSNorm currently supports only welford_mean")
    if hidden_size % (threads * 8) != 0:
        raise ValueError(
            f"block-y RMSNorm requires hidden_size % (threads * 8) == 0, "
            f"got hidden_size={hidden_size}, threads={threads}"
        )
    num_rows = T.dynamic("num_rows")
    chunks = hidden_size // (threads * 8)

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_weighted_rmsnorm_strided_mudnn_like_blocky_h{hidden_size}"
            f"_rs{row_stride}_t{threads}_by{rows_per_cta}_{variance_mode}_{rcp_mode}"
            f"_{_rmsnorm_variant_suffix(rsqrt_mode, mul_order)}"
        ),
        pass_configs={},
    )
    def weighted_rmsnorm_strided_mudnn_like_blocky_kernel(
        x: T.StridedTensor[(num_rows, hidden_size), (row_stride, 1), T.bfloat16],
        weight: T.Tensor[(hidden_size,), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(T.ceildiv(num_rows, rows_per_cta), threads=(threads, rows_per_cta)) as row_block:
            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            row_id = row_block * rows_per_cta + ty
            cnt = T.alloc_local((1,), T.float32)
            mean = T.alloc_local((1,), T.float32)
            delta = T.alloc_local((1,), T.float32)
            rhs_cnt = T.alloc_local((1,), T.float32)
            rhs_mean = T.alloc_local((1,), T.float32)
            n_ab = T.alloc_local((1,), T.float32)
            rcp = T.alloc_local((1,), T.float32)
            rrms = T.alloc_local((1,), T.float32)
            norm_value = T.alloc_local((1,), T.float32)
            y = T.alloc_local((1,), T.float32)
            shared = T.alloc_shared((rows_per_cta, threads), T.float32)
            shared_cnt = T.alloc_shared((rows_per_cta, threads), T.float32)

            cnt[0] = 0.0
            mean[0] = 0.0
            if row_id < num_rows:
                for chunk in T.serial(0, chunks):
                    base = chunk * threads * 8 + tx * 8
                    for k in T.serial(0, 8):
                        value = T.cast(x[row_id, base + k], T.float32)
                        square = value * value
                        cnt[0] += 1.0
                        delta[0] = square - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, cnt[0], rcp_mode)
                        mean[0] += delta[0] * rcp[0]

            if threads >= 128:
                if tx >= 64:
                    shared[ty, tx] = mean[0]
                    shared_cnt[ty, tx] = cnt[0]
                T.sync_threads()
                if tx < 64:
                    rhs_mean[0] = shared[ty, tx + 64]
                    rhs_cnt[0] = shared_cnt[ty, tx + 64]
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    if n_ab[0] > 0.0:
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
            if threads >= 64:
                T.sync_threads()
                if tx >= 32:
                    shared[ty, tx] = mean[0]
                    shared_cnt[ty, tx] = cnt[0]
                T.sync_threads()
                if tx < 32:
                    rhs_mean[0] = shared[ty, tx + 32]
                    rhs_cnt[0] = shared_cnt[ty, tx + 32]
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    if n_ab[0] > 0.0:
                        delta[0] = rhs_mean[0] - mean[0]
                        rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                        mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                        cnt[0] = n_ab[0]
            if threads >= 32:
                rhs_mean[0] = T.shfl_down(mean[0], 16)
                rhs_cnt[0] = T.shfl_down(cnt[0], 16)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
                rhs_mean[0] = T.shfl_down(mean[0], 8)
                rhs_cnt[0] = T.shfl_down(cnt[0], 8)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
                rhs_mean[0] = T.shfl_down(mean[0], 4)
                rhs_cnt[0] = T.shfl_down(cnt[0], 4)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
                rhs_mean[0] = T.shfl_down(mean[0], 2)
                rhs_cnt[0] = T.shfl_down(cnt[0], 2)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
                rhs_mean[0] = T.shfl_down(mean[0], 1)
                rhs_cnt[0] = T.shfl_down(cnt[0], 1)
                if rhs_cnt[0] > 0.0:
                    n_ab[0] = rhs_cnt[0] + cnt[0]
                    delta[0] = rhs_mean[0] - mean[0]
                    rcp[0] = _rmsnorm_rcp_expr(T, n_ab[0], rcp_mode)
                    mean[0] += delta[0] * rhs_cnt[0] * rcp[0]
                    cnt[0] = n_ab[0]
            if tx == 0:
                shared[ty, 0] = mean[0]
            T.sync_threads()
            norm_value[0] = shared[ty, 0] + eps
            if rsqrt_mode == "mudnn_fast_rsqrt":
                y[0] = T.ieee_frsqrt(norm_value[0])
                rrms[0] = y[0] * (
                    T.cast(1.5, "float32")
                    - (T.cast(0.5, "float32") * norm_value[0]) * y[0] * y[0]
                )
            else:
                rrms[0] = _rmsnorm_rrms_expr(T, norm_value[0], rsqrt_mode)

            if row_id < num_rows:
                for chunk in T.serial(0, chunks):
                    base = chunk * threads * 8 + tx * 8
                    for k in T.serial(0, 8):
                        value = T.cast(x[row_id, base + k], T.float32)
                        weight_value = T.cast(weight[base + k], T.float32)
                        out[row_id, base + k] = T.cast(
                            _rmsnorm_weighted_out_expr(T, value, rrms[0], weight_value, mul_order),
                            T.bfloat16,
                        )

    return weighted_rmsnorm_strided_mudnn_like_blocky_kernel




@lru_cache(maxsize=None)
def _tilelang_weighted_rmsnorm_strided_kernel(
    hidden_size: int,
    row_stride: int,
    threads: int = 128,
    compile_profile: str | None = None,
    reduce_profile: str = "burst",
    rsqrt_mode: str = "rsqrt",
    mul_order: str = "x_rrms_weight",
):
    import tilelang
    import tilelang.language as T

    from .kernel_common import _tilelang_musa_reduce_profile_pass_configs

    rsqrt_mode, mul_order = _validate_rmsnorm_variants(rsqrt_mode, mul_order)
    num_rows = T.dynamic("num_rows")

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_weighted_rmsnorm_strided_h{hidden_size}_rs{row_stride}"
            f"_t{threads}_{compile_profile or 'default'}_{reduce_profile}"
            f"_{_rmsnorm_variant_suffix(rsqrt_mode, mul_order)}"
        ),
        pass_configs=_tilelang_musa_reduce_profile_pass_configs(
            tilelang,
            compile_profile=compile_profile,
            reduce_profile=reduce_profile,
        ),
    )
    def weighted_rmsnorm_strided_kernel(
        x: T.StridedTensor[(num_rows, hidden_size), (row_stride, 1), T.bfloat16],
        weight: T.Tensor[(hidden_size,), T.bfloat16],
        out: T.Tensor[(num_rows, hidden_size), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(num_rows, threads=threads) as row_id:
            xsq = T.alloc_fragment((hidden_size,), "float32")
            sumsq = T.alloc_fragment((1,), "float32")
            rrms = T.alloc_fragment((1,), "float32")

            for col in T.Parallel(hidden_size):
                value = T.cast(x[row_id, col], "float32")
                xsq[col] = value * value

            T.reduce_sum(xsq, sumsq, dim=0)
            rrms[0] = _rmsnorm_rrms_expr(T, sumsq[0] / float(hidden_size) + eps, rsqrt_mode)

            for col in T.Parallel(hidden_size):
                value = T.cast(x[row_id, col], "float32")
                weight_value = T.cast(weight[col], "float32")
                out[row_id, col] = T.cast(
                    _rmsnorm_weighted_out_expr(T, value, rrms[0], weight_value, mul_order),
                    T.bfloat16,
                )

    return weighted_rmsnorm_strided_kernel


@lru_cache(maxsize=None)
def _tilelang_weighted_rmsnorm_strided_inplace_kernel(
    hidden_size: int,
    row_stride: int,
    threads: int = 128,
    compile_profile: str | None = None,
    reduce_profile: str = "burst",
    rsqrt_mode: str = "rsqrt",
    mul_order: str = "x_rrms_weight",
):
    import tilelang
    import tilelang.language as T

    from .kernel_common import _tilelang_musa_reduce_profile_pass_configs

    rsqrt_mode, mul_order = _validate_rmsnorm_variants(rsqrt_mode, mul_order)
    num_rows = T.dynamic("num_rows")

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_weighted_rmsnorm_strided_inplace_h{hidden_size}_rs{row_stride}"
            f"_t{threads}_{compile_profile or 'default'}_{reduce_profile}"
            f"_{_rmsnorm_variant_suffix(rsqrt_mode, mul_order)}"
        ),
        pass_configs=_tilelang_musa_reduce_profile_pass_configs(
            tilelang,
            compile_profile=compile_profile,
            reduce_profile=reduce_profile,
        ),
    )
    def weighted_rmsnorm_strided_inplace_kernel(
        x: T.StridedTensor[(num_rows, hidden_size), (row_stride, 1), T.bfloat16],
        weight: T.Tensor[(hidden_size,), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(num_rows, threads=threads) as row_id:
            xsq = T.alloc_fragment((hidden_size,), "float32")
            sumsq = T.alloc_fragment((1,), "float32")
            rrms = T.alloc_fragment((1,), "float32")

            for col in T.Parallel(hidden_size):
                value = T.cast(x[row_id, col], "float32")
                xsq[col] = value * value

            T.reduce_sum(xsq, sumsq, dim=0)
            rrms[0] = _rmsnorm_rrms_expr(T, sumsq[0] / float(hidden_size) + eps, rsqrt_mode)

            for col in T.Parallel(hidden_size):
                value = T.cast(x[row_id, col], "float32")
                weight_value = T.cast(weight[col], "float32")
                x[row_id, col] = T.cast(
                    _rmsnorm_weighted_out_expr(T, value, rrms[0], weight_value, mul_order),
                    T.bfloat16,
                )

    return weighted_rmsnorm_strided_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_hadamard128_inplace_kernel(
    input_dtype: str,
    threads: int = 16,
):
    import tilelang
    import tilelang.language as T

    num_rows = T.dynamic("num_rows")
    head_dim = 128
    items_per_thread = 8
    active_threads = head_dim // items_per_thread
    tl_input_dtype = T.float32 if input_dtype.lower().strip() == "float32" else T.bfloat16

    @_tilelang_jit(
        tilelang,
        f"dsv4_hadamard128_inplace_{input_dtype}_t{threads}",
    )
    def hadamard128_inplace_kernel(
        x: T.Tensor[(num_rows, head_dim), tl_input_dtype],
        scale: T.float32,
    ):
        # DeepSeekV4 uses N=128; mirror the upstream fast_hadamard N=128
        # shape with one 16-lane row per block and 8 values per active lane.
        with T.Kernel(num_rows, threads=threads) as row_id:
            tx = T.get_thread_binding()
            v0 = T.alloc_fragment((1,), dtype=T.float32)
            v1 = T.alloc_fragment((1,), dtype=T.float32)
            v2 = T.alloc_fragment((1,), dtype=T.float32)
            v3 = T.alloc_fragment((1,), dtype=T.float32)
            v4 = T.alloc_fragment((1,), dtype=T.float32)
            v5 = T.alloc_fragment((1,), dtype=T.float32)
            v6 = T.alloc_fragment((1,), dtype=T.float32)
            v7 = T.alloc_fragment((1,), dtype=T.float32)
            a = T.alloc_var(T.float32)
            b = T.alloc_var(T.float32)
            other = T.alloc_var(T.float32)

            if tx < active_threads:
                base = tx * items_per_thread
                v0[0] = T.cast(x[row_id, base], T.float32)
                v1[0] = T.cast(x[row_id, base + 1], T.float32)
                v2[0] = T.cast(x[row_id, base + 2], T.float32)
                v3[0] = T.cast(x[row_id, base + 3], T.float32)
                v4[0] = T.cast(x[row_id, base + 4], T.float32)
                v5[0] = T.cast(x[row_id, base + 5], T.float32)
                v6[0] = T.cast(x[row_id, base + 6], T.float32)
                v7[0] = T.cast(x[row_id, base + 7], T.float32)

                a = v0[0]
                b = v1[0]
                v0[0] = a + b
                v1[0] = a - b
                a = v2[0]
                b = v3[0]
                v2[0] = a + b
                v3[0] = a - b
                a = v4[0]
                b = v5[0]
                v4[0] = a + b
                v5[0] = a - b
                a = v6[0]
                b = v7[0]
                v6[0] = a + b
                v7[0] = a - b

                a = v0[0]
                b = v2[0]
                v0[0] = a + b
                v2[0] = a - b
                a = v1[0]
                b = v3[0]
                v1[0] = a + b
                v3[0] = a - b
                a = v4[0]
                b = v6[0]
                v4[0] = a + b
                v6[0] = a - b
                a = v5[0]
                b = v7[0]
                v5[0] = a + b
                v7[0] = a - b

                a = v0[0]
                b = v4[0]
                v0[0] = a + b
                v4[0] = a - b
                a = v1[0]
                b = v5[0]
                v1[0] = a + b
                v5[0] = a - b
                a = v2[0]
                b = v6[0]
                v2[0] = a + b
                v6[0] = a - b
                a = v3[0]
                b = v7[0]
                v3[0] = a + b
                v7[0] = a - b

                for stage in T.unroll(4):
                    lane_mask = T.int32(1) << stage
                    sign = T.if_then_else((tx & lane_mask) == 0, 1.0, -1.0)
                    other = T.shfl_xor(v0[0], lane_mask)
                    v0[0] = sign * v0[0] + other
                    other = T.shfl_xor(v1[0], lane_mask)
                    v1[0] = sign * v1[0] + other
                    other = T.shfl_xor(v2[0], lane_mask)
                    v2[0] = sign * v2[0] + other
                    other = T.shfl_xor(v3[0], lane_mask)
                    v3[0] = sign * v3[0] + other
                    other = T.shfl_xor(v4[0], lane_mask)
                    v4[0] = sign * v4[0] + other
                    other = T.shfl_xor(v5[0], lane_mask)
                    v5[0] = sign * v5[0] + other
                    other = T.shfl_xor(v6[0], lane_mask)
                    v6[0] = sign * v6[0] + other
                    other = T.shfl_xor(v7[0], lane_mask)
                    v7[0] = sign * v7[0] + other

                x[row_id, base] = T.cast(v0[0] * scale, tl_input_dtype)
                x[row_id, base + 1] = T.cast(v1[0] * scale, tl_input_dtype)
                x[row_id, base + 2] = T.cast(v2[0] * scale, tl_input_dtype)
                x[row_id, base + 3] = T.cast(v3[0] * scale, tl_input_dtype)
                x[row_id, base + 4] = T.cast(v4[0] * scale, tl_input_dtype)
                x[row_id, base + 5] = T.cast(v5[0] * scale, tl_input_dtype)
                x[row_id, base + 6] = T.cast(v6[0] * scale, tl_input_dtype)
                x[row_id, base + 7] = T.cast(v7[0] * scale, tl_input_dtype)

    return hadamard128_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_rope_hadamard_inplace_kernel_fast(
    input_dtype: str,
    num_heads: int,
    positions_dtype: str,
    heads_per_block: int = 1,
):
    import tilelang
    import tilelang.language as T

    from .kernel_common import _tilelang_musa_aggressive_pass_configs

    num_tokens = T.dynamic("num_tokens")
    q_numel = T.dynamic("q_numel")
    num_freq_positions = T.dynamic("num_freq_positions")
    head_dim = 128
    rope_dim = 64
    half_dim = rope_dim // 2
    items_per_thread = 8
    active_threads = head_dim // items_per_thread
    head_blocks = (num_heads + heads_per_block - 1) // heads_per_block
    tl_input_dtype = T.float32 if input_dtype.lower().strip() == "float32" else T.bfloat16
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    positions_stride = T.dynamic("positions_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")
    @_tilelang_jit(
        tilelang,
        f"dsv4_rope_hadamard_fast_h128_r64_n{num_heads}_hpb{heads_per_block}_{input_dtype}_{positions_dtype}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang,
            disable_index_promotion=True,
        ),
    )
    def rope_hadamard_fast_inplace_kernel(
        q: T.Tensor[(q_numel,), tl_input_dtype],
        freqs_real_imag: T.StridedTensor[
            (num_freq_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
        scale: T.float32,
    ):
        with T.Kernel(num_tokens, head_blocks, threads=active_threads * heads_per_block) as (token_id, head_block):
            tx = T.get_thread_binding()
            head_lane = tx // active_threads
            lane = tx - head_lane * active_threads
            head_id = head_block * heads_per_block + head_lane
            base = lane * items_per_thread
            q_base = (
                T.Cast("int32", token_id) * (num_heads * head_dim)
                + T.Cast("int32", head_id) * head_dim
                + T.Cast("int32", base)
            )
            v0 = T.alloc_fragment((1,), dtype=T.float32)
            v1 = T.alloc_fragment((1,), dtype=T.float32)
            v2 = T.alloc_fragment((1,), dtype=T.float32)
            v3 = T.alloc_fragment((1,), dtype=T.float32)
            v4 = T.alloc_fragment((1,), dtype=T.float32)
            v5 = T.alloc_fragment((1,), dtype=T.float32)
            v6 = T.alloc_fragment((1,), dtype=T.float32)
            v7 = T.alloc_fragment((1,), dtype=T.float32)
            a = T.alloc_var(T.float32)
            b = T.alloc_var(T.float32)
            other = T.alloc_var(T.float32)

            if base < rope_dim:
                v0[0] = T.cast(q[q_base], T.float32)
                v1[0] = T.cast(q[q_base + 1], T.float32)
                v2[0] = T.cast(q[q_base + 2], T.float32)
                v3[0] = T.cast(q[q_base + 3], T.float32)
                v4[0] = T.cast(q[q_base + 4], T.float32)
                v5[0] = T.cast(q[q_base + 5], T.float32)
                v6[0] = T.cast(q[q_base + 6], T.float32)
                v7[0] = T.cast(q[q_base + 7], T.float32)
            else:
                pos = T.Cast("int32", positions[token_id])
                pair_base = (base - rope_dim) // 2
                even = T.alloc_var(T.float32)
                odd = T.alloc_var(T.float32)
                c = T.alloc_var(T.float32)
                s = T.alloc_var(T.float32)

                even = T.cast(q[q_base], T.float32)
                odd = T.cast(q[q_base + 1], T.float32)
                c = freqs_real_imag[pos, pair_base, 0]
                s = freqs_real_imag[pos, pair_base, 1]
                v0[0] = even * c - odd * s
                v1[0] = even * s + odd * c

                even = T.cast(q[q_base + 2], T.float32)
                odd = T.cast(q[q_base + 3], T.float32)
                c = freqs_real_imag[pos, pair_base + 1, 0]
                s = freqs_real_imag[pos, pair_base + 1, 1]
                v2[0] = even * c - odd * s
                v3[0] = even * s + odd * c

                even = T.cast(q[q_base + 4], T.float32)
                odd = T.cast(q[q_base + 5], T.float32)
                c = freqs_real_imag[pos, pair_base + 2, 0]
                s = freqs_real_imag[pos, pair_base + 2, 1]
                v4[0] = even * c - odd * s
                v5[0] = even * s + odd * c

                even = T.cast(q[q_base + 6], T.float32)
                odd = T.cast(q[q_base + 7], T.float32)
                c = freqs_real_imag[pos, pair_base + 3, 0]
                s = freqs_real_imag[pos, pair_base + 3, 1]
                v6[0] = even * c - odd * s
                v7[0] = even * s + odd * c

            a = v0[0]
            b = v1[0]
            v0[0] = a + b
            v1[0] = a - b
            a = v2[0]
            b = v3[0]
            v2[0] = a + b
            v3[0] = a - b
            a = v4[0]
            b = v5[0]
            v4[0] = a + b
            v5[0] = a - b
            a = v6[0]
            b = v7[0]
            v6[0] = a + b
            v7[0] = a - b

            a = v0[0]
            b = v2[0]
            v0[0] = a + b
            v2[0] = a - b
            a = v1[0]
            b = v3[0]
            v1[0] = a + b
            v3[0] = a - b
            a = v4[0]
            b = v6[0]
            v4[0] = a + b
            v6[0] = a - b
            a = v5[0]
            b = v7[0]
            v5[0] = a + b
            v7[0] = a - b

            a = v0[0]
            b = v4[0]
            v0[0] = a + b
            v4[0] = a - b
            a = v1[0]
            b = v5[0]
            v1[0] = a + b
            v5[0] = a - b
            a = v2[0]
            b = v6[0]
            v2[0] = a + b
            v6[0] = a - b
            a = v3[0]
            b = v7[0]
            v3[0] = a + b
            v7[0] = a - b

            for stage in T.unroll(4):
                lane_mask = T.int32(1) << stage
                sign = T.if_then_else((lane & lane_mask) == 0, 1.0, -1.0)
                other = T.shfl_xor(v0[0], lane_mask)
                v0[0] = sign * v0[0] + other
                other = T.shfl_xor(v1[0], lane_mask)
                v1[0] = sign * v1[0] + other
                other = T.shfl_xor(v2[0], lane_mask)
                v2[0] = sign * v2[0] + other
                other = T.shfl_xor(v3[0], lane_mask)
                v3[0] = sign * v3[0] + other
                other = T.shfl_xor(v4[0], lane_mask)
                v4[0] = sign * v4[0] + other
                other = T.shfl_xor(v5[0], lane_mask)
                v5[0] = sign * v5[0] + other
                other = T.shfl_xor(v6[0], lane_mask)
                v6[0] = sign * v6[0] + other
                other = T.shfl_xor(v7[0], lane_mask)
                v7[0] = sign * v7[0] + other

            if input_dtype.lower().strip() == "bfloat16":
                out_f0 = T.alloc_local((4,), T.float32)
                out_b0 = T.alloc_local((4,), T.bfloat16)
                out_f1 = T.alloc_local((4,), T.float32)
                out_b1 = T.alloc_local((4,), T.bfloat16)
                out_f0[0] = v0[0] * scale
                out_f0[1] = v1[0] * scale
                out_f0[2] = v2[0] * scale
                out_f0[3] = v3[0] * scale
                out_f1[0] = v4[0] * scale
                out_f1[1] = v5[0] * scale
                out_f1[2] = v6[0] * scale
                out_f1[3] = v7[0] * scale
                for idx in T.vectorized(4):
                    out_b0[idx] = T.cast(out_f0[idx], T.bfloat16)
                    out_b1[idx] = T.cast(out_f1[idx], T.bfloat16)
                p01 = T.Cast("uint32", T.reinterpret(out_b0[0], "uint16")) | (
                    T.Cast("uint32", T.reinterpret(out_b0[1], "uint16")) << 16
                )
                p23 = T.Cast("uint32", T.reinterpret(out_b0[2], "uint16")) | (
                    T.Cast("uint32", T.reinterpret(out_b0[3], "uint16")) << 16
                )
                p45 = T.Cast("uint32", T.reinterpret(out_b1[0], "uint16")) | (
                    T.Cast("uint32", T.reinterpret(out_b1[1], "uint16")) << 16
                )
                p67 = T.Cast("uint32", T.reinterpret(out_b1[2], "uint16")) | (
                    T.Cast("uint32", T.reinterpret(out_b1[3], "uint16")) << 16
                )
                T.stg32(q[q_base], p01)
                T.stg32(q[q_base + 2], p23)
                T.stg32(q[q_base + 4], p45)
                T.stg32(q[q_base + 6], p67)
            else:
                q[q_base] = T.cast(v0[0] * scale, tl_input_dtype)
                q[q_base + 1] = T.cast(v1[0] * scale, tl_input_dtype)
                q[q_base + 2] = T.cast(v2[0] * scale, tl_input_dtype)
                q[q_base + 3] = T.cast(v3[0] * scale, tl_input_dtype)
                q[q_base + 4] = T.cast(v4[0] * scale, tl_input_dtype)
                q[q_base + 5] = T.cast(v5[0] * scale, tl_input_dtype)
                q[q_base + 6] = T.cast(v6[0] * scale, tl_input_dtype)
                q[q_base + 7] = T.cast(v7[0] * scale, tl_input_dtype)

    return rope_hadamard_fast_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_neox_rope_hadamard_inplace_kernel_fast(
    input_dtype: str,
    num_heads: int,
    positions_dtype: str,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_freq_positions = T.dynamic("num_freq_positions")
    head_dim = 128
    rope_dim = 64
    half_dim = rope_dim // 2
    items_per_thread = 8
    active_threads = head_dim // items_per_thread
    tl_input_dtype = T.float32 if input_dtype.lower().strip() == "float32" else T.bfloat16
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    positions_stride = T.dynamic("positions_stride")
    cos_sin_pos_stride = T.dynamic("cos_sin_pos_stride")
    cos_sin_col_stride = T.dynamic("cos_sin_col_stride")

    @_tilelang_jit(
        tilelang,
        f"dsv4_neox_rope_hadamard_fast_h128_r64_n{num_heads}_{input_dtype}_{positions_dtype}",
    )
    def neox_rope_hadamard_fast_inplace_kernel(
        q: T.Tensor[(num_tokens, num_heads, head_dim), tl_input_dtype],
        cos_sin_cache: T.StridedTensor[
            (num_freq_positions, rope_dim),
            (cos_sin_pos_stride, cos_sin_col_stride),
            T.float32,
        ],
        positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
        scale: T.float32,
    ):
        with T.Kernel(num_tokens, num_heads, threads=active_threads) as (token_id, head_id):
            tx = T.get_thread_binding()
            base = tx * items_per_thread
            v0 = T.alloc_fragment((1,), dtype=T.float32)
            v1 = T.alloc_fragment((1,), dtype=T.float32)
            v2 = T.alloc_fragment((1,), dtype=T.float32)
            v3 = T.alloc_fragment((1,), dtype=T.float32)
            v4 = T.alloc_fragment((1,), dtype=T.float32)
            v5 = T.alloc_fragment((1,), dtype=T.float32)
            v6 = T.alloc_fragment((1,), dtype=T.float32)
            v7 = T.alloc_fragment((1,), dtype=T.float32)
            a = T.alloc_var(T.float32)
            b = T.alloc_var(T.float32)
            other = T.alloc_var(T.float32)
            pos = positions[token_id]

            if base < half_dim:
                for i in T.unroll(8):
                    x0 = T.cast(q[token_id, head_id, base + i], T.float32)
                    x1 = T.cast(q[token_id, head_id, base + half_dim + i], T.float32)
                    c = T.cast(T.cast(cos_sin_cache[pos, base + i], tl_input_dtype), T.float32)
                    s = T.cast(T.cast(cos_sin_cache[pos, base + half_dim + i], tl_input_dtype), T.float32)
                    if i == 0:
                        v0[0] = T.cast(T.cast(x0 * c - x1 * s, tl_input_dtype), T.float32)
                    elif i == 1:
                        v1[0] = T.cast(T.cast(x0 * c - x1 * s, tl_input_dtype), T.float32)
                    elif i == 2:
                        v2[0] = T.cast(T.cast(x0 * c - x1 * s, tl_input_dtype), T.float32)
                    elif i == 3:
                        v3[0] = T.cast(T.cast(x0 * c - x1 * s, tl_input_dtype), T.float32)
                    elif i == 4:
                        v4[0] = T.cast(T.cast(x0 * c - x1 * s, tl_input_dtype), T.float32)
                    elif i == 5:
                        v5[0] = T.cast(T.cast(x0 * c - x1 * s, tl_input_dtype), T.float32)
                    elif i == 6:
                        v6[0] = T.cast(T.cast(x0 * c - x1 * s, tl_input_dtype), T.float32)
                    else:
                        v7[0] = T.cast(T.cast(x0 * c - x1 * s, tl_input_dtype), T.float32)
            elif base < rope_dim:
                idx = base - half_dim
                for i in T.unroll(8):
                    x0 = T.cast(q[token_id, head_id, idx + i], T.float32)
                    x1 = T.cast(q[token_id, head_id, base + i], T.float32)
                    c = T.cast(T.cast(cos_sin_cache[pos, idx + i], tl_input_dtype), T.float32)
                    s = T.cast(T.cast(cos_sin_cache[pos, idx + half_dim + i], tl_input_dtype), T.float32)
                    if i == 0:
                        v0[0] = T.cast(T.cast(x1 * c + x0 * s, tl_input_dtype), T.float32)
                    elif i == 1:
                        v1[0] = T.cast(T.cast(x1 * c + x0 * s, tl_input_dtype), T.float32)
                    elif i == 2:
                        v2[0] = T.cast(T.cast(x1 * c + x0 * s, tl_input_dtype), T.float32)
                    elif i == 3:
                        v3[0] = T.cast(T.cast(x1 * c + x0 * s, tl_input_dtype), T.float32)
                    elif i == 4:
                        v4[0] = T.cast(T.cast(x1 * c + x0 * s, tl_input_dtype), T.float32)
                    elif i == 5:
                        v5[0] = T.cast(T.cast(x1 * c + x0 * s, tl_input_dtype), T.float32)
                    elif i == 6:
                        v6[0] = T.cast(T.cast(x1 * c + x0 * s, tl_input_dtype), T.float32)
                    else:
                        v7[0] = T.cast(T.cast(x1 * c + x0 * s, tl_input_dtype), T.float32)
            else:
                v0[0] = T.cast(q[token_id, head_id, base], T.float32)
                v1[0] = T.cast(q[token_id, head_id, base + 1], T.float32)
                v2[0] = T.cast(q[token_id, head_id, base + 2], T.float32)
                v3[0] = T.cast(q[token_id, head_id, base + 3], T.float32)
                v4[0] = T.cast(q[token_id, head_id, base + 4], T.float32)
                v5[0] = T.cast(q[token_id, head_id, base + 5], T.float32)
                v6[0] = T.cast(q[token_id, head_id, base + 6], T.float32)
                v7[0] = T.cast(q[token_id, head_id, base + 7], T.float32)

            a = v0[0]; b = v1[0]; v0[0] = a + b; v1[0] = a - b
            a = v2[0]; b = v3[0]; v2[0] = a + b; v3[0] = a - b
            a = v4[0]; b = v5[0]; v4[0] = a + b; v5[0] = a - b
            a = v6[0]; b = v7[0]; v6[0] = a + b; v7[0] = a - b

            a = v0[0]; b = v2[0]; v0[0] = a + b; v2[0] = a - b
            a = v1[0]; b = v3[0]; v1[0] = a + b; v3[0] = a - b
            a = v4[0]; b = v6[0]; v4[0] = a + b; v6[0] = a - b
            a = v5[0]; b = v7[0]; v5[0] = a + b; v7[0] = a - b

            a = v0[0]; b = v4[0]; v0[0] = a + b; v4[0] = a - b
            a = v1[0]; b = v5[0]; v1[0] = a + b; v5[0] = a - b
            a = v2[0]; b = v6[0]; v2[0] = a + b; v6[0] = a - b
            a = v3[0]; b = v7[0]; v3[0] = a + b; v7[0] = a - b

            for stage in T.unroll(4):
                lane_mask = T.int32(1) << stage
                sign = T.if_then_else((tx & lane_mask) == 0, 1.0, -1.0)
                other = T.shfl_xor(v0[0], lane_mask); v0[0] = sign * v0[0] + other
                other = T.shfl_xor(v1[0], lane_mask); v1[0] = sign * v1[0] + other
                other = T.shfl_xor(v2[0], lane_mask); v2[0] = sign * v2[0] + other
                other = T.shfl_xor(v3[0], lane_mask); v3[0] = sign * v3[0] + other
                other = T.shfl_xor(v4[0], lane_mask); v4[0] = sign * v4[0] + other
                other = T.shfl_xor(v5[0], lane_mask); v5[0] = sign * v5[0] + other
                other = T.shfl_xor(v6[0], lane_mask); v6[0] = sign * v6[0] + other
                other = T.shfl_xor(v7[0], lane_mask); v7[0] = sign * v7[0] + other

            q[token_id, head_id, base] = T.cast(v0[0] * scale, tl_input_dtype)
            q[token_id, head_id, base + 1] = T.cast(v1[0] * scale, tl_input_dtype)
            q[token_id, head_id, base + 2] = T.cast(v2[0] * scale, tl_input_dtype)
            q[token_id, head_id, base + 3] = T.cast(v3[0] * scale, tl_input_dtype)
            q[token_id, head_id, base + 4] = T.cast(v4[0] * scale, tl_input_dtype)
            q[token_id, head_id, base + 5] = T.cast(v5[0] * scale, tl_input_dtype)
            q[token_id, head_id, base + 6] = T.cast(v6[0] * scale, tl_input_dtype)
            q[token_id, head_id, base + 7] = T.cast(v7[0] * scale, tl_input_dtype)

    return neox_rope_hadamard_fast_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_rope_hadamard_inplace_kernel(
    input_dtype: str,
    num_heads: int,
    positions_dtype: str,
    heads_per_block: int = 1,
    pingpong: bool = False,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_freq_positions = T.dynamic("num_freq_positions")
    head_dim = 128
    rope_dim = 64
    half_dim = rope_dim // 2
    num_head_groups = (num_heads + heads_per_block - 1) // heads_per_block
    tl_input_dtype = T.float32 if input_dtype.lower().strip() == "float32" else T.bfloat16
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    positions_stride = T.dynamic("positions_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")

    @_tilelang_jit(
        tilelang,
        (
            f"dsv4_rope_hadamard_h128_r64_n{num_heads}_{input_dtype}_{positions_dtype}"
            f"_hpb{heads_per_block}_pp{int(pingpong)}_t{threads}"
        ),
    )
    def rope_hadamard_inplace_kernel(
        q: T.Tensor[(num_tokens, num_heads, head_dim), tl_input_dtype],
        freqs_real_imag: T.StridedTensor[
            (num_freq_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
        scale: T.float32,
    ):
        with T.Kernel(num_tokens, num_head_groups, threads=threads) as (token_id, head_group_id):
            tx = T.get_thread_binding()
            local_head = tx // head_dim
            col = tx - local_head * head_dim
            head_id = head_group_id * heads_per_block + local_head

            if pingpong:
                values = T.alloc_shared((2, heads_per_block, head_dim), dtype=T.float32)

                if local_head < heads_per_block and head_id < num_heads:
                    if col < rope_dim:
                        values[0, local_head, col] = T.cast(q[token_id, head_id, col], T.float32)
                    if col < half_dim:
                        pos = positions[token_id]
                        even_col = rope_dim + col * 2
                        odd_col = even_col + 1
                        c = freqs_real_imag[pos, col, 0]
                        s = freqs_real_imag[pos, col, 1]
                        even = T.cast(q[token_id, head_id, even_col], T.float32)
                        odd = T.cast(q[token_id, head_id, odd_col], T.float32)
                        values[0, local_head, even_col] = even * c - odd * s
                        values[0, local_head, odd_col] = even * s + odd * c
                T.sync_threads()

                if local_head < heads_per_block and head_id < num_heads and col < head_dim:
                    for stage in T.serial(0, 7):
                        step = T.int32(1) << stage
                        src = stage & 1
                        dst = 1 - src
                        peer = col ^ step
                        self_value = values[src, local_head, col]
                        peer_value = values[src, local_head, peer]
                        if (col & step) == 0:
                            values[dst, local_head, col] = self_value + peer_value
                        else:
                            values[dst, local_head, col] = peer_value - self_value
                        T.sync_threads()
                    q[token_id, head_id, col] = T.cast(values[1, local_head, col] * scale, tl_input_dtype)
            else:
                values = T.alloc_shared((heads_per_block, head_dim), dtype=T.float32)

                if local_head < heads_per_block and head_id < num_heads:
                    if col < rope_dim:
                        values[local_head, col] = T.cast(q[token_id, head_id, col], T.float32)
                    if col < half_dim:
                        pos = positions[token_id]
                        even_col = rope_dim + col * 2
                        odd_col = even_col + 1
                        c = freqs_real_imag[pos, col, 0]
                        s = freqs_real_imag[pos, col, 1]
                        even = T.cast(q[token_id, head_id, even_col], T.float32)
                        odd = T.cast(q[token_id, head_id, odd_col], T.float32)
                        values[local_head, even_col] = even * c - odd * s
                        values[local_head, odd_col] = even * s + odd * c
                T.sync_threads()

                if local_head < heads_per_block and head_id < num_heads and col < head_dim:
                    for stage in T.serial(0, 7):
                        step = T.int32(1) << stage
                        peer = col ^ step
                        self_value = values[local_head, col]
                        peer_value = values[local_head, peer]
                        T.sync_threads()
                        if (col & step) == 0:
                            values[local_head, col] = self_value + peer_value
                        else:
                            values[local_head, col] = peer_value - self_value
                        T.sync_threads()
                    q[token_id, head_id, col] = T.cast(values[local_head, col] * scale, tl_input_dtype)

    return rope_hadamard_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_rope_hadamard_inplace_kernel_v1(
    input_dtype: str,
    num_heads: int,
    positions_dtype: str,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_freq_positions = T.dynamic("num_freq_positions")
    head_dim = 128
    rope_dim = 64
    half_dim = rope_dim // 2
    tl_input_dtype = T.float32 if input_dtype.lower().strip() == "float32" else T.bfloat16
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    positions_stride = T.dynamic("positions_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")

    @_tilelang_jit(
        tilelang,
        f"dsv4_rope_hadamard_v1_h128_r64_n{num_heads}_{input_dtype}_{positions_dtype}_t{threads}",
    )
    def rope_hadamard_inplace_kernel(
        q: T.Tensor[(num_tokens, num_heads, head_dim), tl_input_dtype],
        freqs_real_imag: T.StridedTensor[
            (num_freq_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
        scale: T.float32,
    ):
        with T.Kernel(num_tokens, num_heads, threads=threads) as (token_id, head_id):
            tx = T.get_thread_binding()
            values = T.alloc_shared((head_dim,), dtype=T.float32)

            if tx < rope_dim:
                values[tx] = T.cast(q[token_id, head_id, tx], T.float32)
            if tx < half_dim:
                pos = positions[token_id]
                even_col = rope_dim + tx * 2
                odd_col = even_col + 1
                c = freqs_real_imag[pos, tx, 0]
                s = freqs_real_imag[pos, tx, 1]
                even = T.cast(q[token_id, head_id, even_col], T.float32)
                odd = T.cast(q[token_id, head_id, odd_col], T.float32)
                values[even_col] = even * c - odd * s
                values[odd_col] = even * s + odd * c
            T.sync_threads()

            if tx < head_dim:
                for stage in T.serial(0, 7):
                    step = T.int32(1) << stage
                    peer = tx ^ step
                    self_value = values[tx]
                    peer_value = values[peer]
                    T.sync_threads()
                    if (tx & step) == 0:
                        values[tx] = self_value + peer_value
                    else:
                        values[tx] = peer_value - self_value
                    T.sync_threads()
                q[token_id, head_id, tx] = T.cast(values[tx] * scale, tl_input_dtype)

    return rope_hadamard_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_rope_inplace_kernel(
    input_dtype: str,
    head_dim: int,
    num_heads: int,
    inverse: bool,
    positions_dtype: str,
    threads: int = 32,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_elements = T.dynamic("num_elements")
    num_positions = T.dynamic("num_positions")
    half_dim = head_dim // 2
    sign = -1.0 if inverse else 1.0
    tl_input_dtype = T.float32 if input_dtype.lower().strip() == "float32" else T.bfloat16
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    positions_stride = T.dynamic("positions_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")

    jit_name = (
        f"dsv4_rope_inplace_{input_dtype}_h{head_dim}_n{num_heads}_"
        f"{positions_dtype}_dynstride_{'inv' if inverse else 'fwd'}_t{threads}"
    )

    @_tilelang_jit(
        tilelang,
        jit_name,
    )
    def rope_inplace_kernel(
        x: T.Tensor[(num_elements,), tl_input_dtype],
        freqs_real_imag: T.StridedTensor[
            (num_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
        base_offset: T.int32,
        stride_token: T.int32,
        stride_head: T.int32,
        stride_dim: T.int32,
    ):
        with T.Kernel(num_tokens, num_heads, threads=threads) as (token_id, head_id):
            tx = T.get_thread_binding()
            if tx < half_dim:
                pair_idx = tx
                pos = positions[token_id]
                c = freqs_real_imag[pos, pair_idx, 0]
                s = freqs_real_imag[pos, pair_idx, 1] * sign
                even_offset = base_offset + token_id * stride_token + head_id * stride_head + pair_idx * 2 * stride_dim
                odd_offset = even_offset + stride_dim
                even = T.cast(x[even_offset], T.float32)
                odd = T.cast(x[odd_offset], T.float32)
                x[even_offset] = T.cast(even * c - odd * s, tl_input_dtype)
                x[odd_offset] = T.cast(even * s + odd * c, tl_input_dtype)

    return rope_inplace_kernel

@lru_cache(maxsize=None)
def _tilelang_rope_inplace_flat_kernel(
    input_dtype: str,
    head_dim: int,
    num_heads: int,
    inverse: bool,
    positions_dtype: str,
    threads: int = 256,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_elements = T.dynamic("num_elements")
    num_positions = T.dynamic("num_positions")
    half_dim = head_dim // 2
    sign = -1.0 if inverse else 1.0
    tl_input_dtype = T.float32 if input_dtype.lower().strip() == "float32" else T.bfloat16
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    positions_stride = T.dynamic("positions_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")

    jit_name = (
        f"dsv4_rope_inplace_flat_{input_dtype}_h{head_dim}_n{num_heads}_"
        f"{positions_dtype}_dynstride_{'inv' if inverse else 'fwd'}_t{threads}"
    )

    @_tilelang_jit(
        tilelang,
        jit_name,
    )
    def rope_inplace_flat_kernel(
        x: T.Tensor[(num_elements,), tl_input_dtype],
        freqs_real_imag: T.StridedTensor[
            (num_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
        base_offset: T.int32,
        stride_token: T.int32,
        stride_head: T.int32,
        stride_dim: T.int32,
    ):
        with T.Kernel(T.ceildiv(num_tokens * num_heads * half_dim, threads), threads=threads) as block_id:
            tx = T.get_thread_binding()
            flat_pair = block_id * threads + tx
            if flat_pair < num_tokens * num_heads * half_dim:
                pair_idx = flat_pair % half_dim
                token_head = flat_pair // half_dim
                head_id = token_head % num_heads
                token_id = token_head // num_heads
                pos = positions[token_id]
                c = freqs_real_imag[pos, pair_idx, 0]
                s = freqs_real_imag[pos, pair_idx, 1] * sign
                base_i64 = T.Cast("int64", base_offset) + T.Cast("int64", token_id) * T.Cast(
                    "int64", stride_token
                ) + T.Cast("int64", head_id) * T.Cast("int64", stride_head)
                even_offset = base_i64 + T.Cast("int64", pair_idx * 2) * T.Cast("int64", stride_dim)
                odd_offset = even_offset + T.Cast("int64", stride_dim)
                even = T.cast(x[even_offset], T.float32)
                odd = T.cast(x[odd_offset], T.float32)
                x[even_offset] = T.cast(even * c - odd * s, tl_input_dtype)
                x[odd_offset] = T.cast(even * s + odd * c, tl_input_dtype)

    return rope_inplace_flat_kernel

@lru_cache(maxsize=None)
def _tilelang_fused_norm_rope_inplace_kernel(
    hidden_size: int,
    rope_dim: int,
    kv_dtype: torch.dtype,
    positions_dtype: str,
    threads: int = 256,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_freq_positions = T.dynamic("num_freq_positions")
    half_dim = rope_dim // 2
    nope_dim = hidden_size - rope_dim
    tl_dtype = T.float32 if kv_dtype == torch.float32 else T.bfloat16
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    dtype_name = str(kv_dtype).split(".")[-1]
    positions_stride = T.dynamic("positions_stride")
    weight_stride = T.dynamic("weight_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")

    # Threshold 128: T.reduce_sum on alloc_fragment works reliably up to this size
    # in the current TileLang version. The production h=512/r=64 path below uses
    # the same warp-per-token mapping as the NV kernel: one warp owns one token,
    # and one CTA processes several tokens.
    if hidden_size <= 128:
        @_tilelang_jit(
            tilelang,
            f"dsv4_fused_norm_rope_direct_h{hidden_size}_r{rope_dim}_{dtype_name}_{positions_dtype}_dynstride_t{threads}",
        )
        def fused_norm_rope_inplace_kernel(
            kv: T.Tensor[(num_tokens, hidden_size), tl_dtype],
            weight: T.StridedTensor[(hidden_size,), (weight_stride,), tl_dtype],
            freqs_real_imag: T.StridedTensor[
                (num_freq_positions, half_dim, 2),
                (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
                T.float32,
            ],
            positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
            eps: T.float32,
        ):
            with T.Kernel(num_tokens, threads=threads) as token_id:
                xsq_f32 = T.alloc_fragment((1, hidden_size), T.float32)
                sumsq = T.alloc_fragment((1,), T.float32)
                rrms = T.alloc_fragment((1,), T.float32)

                for col in T.Parallel(hidden_size):
                    value = T.cast(kv[token_id, col], T.float32)
                    xsq_f32[0, col] = value * value

                T.reduce_sum(xsq_f32, sumsq, dim=1)
                rrms[0] = T.rsqrt(sumsq[0] / float(hidden_size) + eps)

                if nope_dim > 0:
                    for col in T.Parallel(nope_dim):
                        value = T.cast(kv[token_id, col], T.float32) * rrms[0] * T.cast(weight[col], T.float32)
                        kv[token_id, col] = T.cast(value, tl_dtype)

                pos = positions[token_id]
                for pair_idx in T.Parallel(half_dim):
                    even_col = nope_dim + pair_idx * 2
                    odd_col = even_col + 1
                    even = T.cast(kv[token_id, even_col], T.float32) * rrms[0] * T.cast(weight[even_col], T.float32)
                    odd = T.cast(kv[token_id, odd_col], T.float32) * rrms[0] * T.cast(weight[odd_col], T.float32)
                    c = freqs_real_imag[pos, pair_idx, 0]
                    s = freqs_real_imag[pos, pair_idx, 1]
                    kv[token_id, even_col] = T.cast(even * c - odd * s, tl_dtype)
                    kv[token_id, odd_col] = T.cast(even * s + odd * c, tl_dtype)

        return fused_norm_rope_inplace_kernel

    if hidden_size % 32 == 0 and half_dim <= 32:
        warps_per_cta = threads // 32

        @_tilelang_jit(
            tilelang,
            f"dsv4_fused_norm_rope_warp_h{hidden_size}_r{rope_dim}_{dtype_name}_{positions_dtype}_dynstride_t{threads}",
        )
        def fused_norm_rope_inplace_kernel(
            kv: T.Tensor[(num_tokens, hidden_size), tl_dtype],
            weight: T.StridedTensor[(hidden_size,), (weight_stride,), tl_dtype],
            freqs_real_imag: T.StridedTensor[
                (num_freq_positions, half_dim, 2),
                (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
                T.float32,
            ],
            positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
            eps: T.float32,
        ):
            with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), threads=threads) as block_id:
                tx = T.get_thread_binding()
                lane = tx % 32
                warp = tx // 32
                token_id = block_id * warps_per_cta + warp
                partial_sumsq = T.alloc_local((1,), T.float32)

                if token_id < num_tokens:
                    partial_sumsq[0] = 0.0
                    for col_base in T.serial(0, hidden_size, 32):
                        col = col_base + lane
                        value = T.cast(kv[token_id, col], T.float32)
                        partial_sumsq[0] += value * value

                    partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
                    partial_sumsq[0] = T.rsqrt(partial_sumsq[0] / float(hidden_size) + eps)

                    if nope_dim > 0:
                        for col_base in T.serial(0, nope_dim, 32):
                            col = col_base + lane
                            if col < nope_dim:
                                value = T.cast(kv[token_id, col], T.float32) * partial_sumsq[0] * T.cast(weight[col], T.float32)
                                kv[token_id, col] = T.cast(value, tl_dtype)

                    if lane < half_dim:
                        pos = positions[token_id]
                        even_col = nope_dim + lane * 2
                        odd_col = even_col + 1
                        even = T.cast(kv[token_id, even_col], T.float32) * partial_sumsq[0] * T.cast(weight[even_col], T.float32)
                        odd = T.cast(kv[token_id, odd_col], T.float32) * partial_sumsq[0] * T.cast(weight[odd_col], T.float32)
                        c = freqs_real_imag[pos, lane, 0]
                        s = freqs_real_imag[pos, lane, 1]
                        kv[token_id, even_col] = T.cast(even * c - odd * s, tl_dtype)
                        kv[token_id, odd_col] = T.cast(even * s + odd * c, tl_dtype)

        return fused_norm_rope_inplace_kernel

    @_tilelang_jit(
        tilelang,
        f"dsv4_fused_norm_rope_local_h{hidden_size}_r{rope_dim}_{dtype_name}_{positions_dtype}_dynstride_t{threads}",
    )
    def fused_norm_rope_inplace_kernel(
        kv: T.Tensor[(num_tokens, hidden_size), tl_dtype],
        weight: T.StridedTensor[(hidden_size,), (weight_stride,), tl_dtype],
        freqs_real_imag: T.StridedTensor[
            (num_freq_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
        eps: T.float32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            warps_per_cta = threads // 32
            partial_sumsq = T.alloc_local((1,), T.float32)
            warp_sumsq = T.alloc_shared((warps_per_cta,), T.float32)

            partial_sumsq[0] = 0.0
            for col_base in T.serial(0, hidden_size, threads):
                col = col_base + tx
                if col < hidden_size:
                    value = T.cast(kv[token_id, col], T.float32)
                    partial_sumsq[0] += value * value

            partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
            if lane == 0:
                warp_sumsq[warp] = partial_sumsq[0]
            T.sync_threads()

            partial_sumsq[0] = T.if_then_else(tx < warps_per_cta, warp_sumsq[tx], 0.0)
            if warp == 0:
                partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
                if lane == 0:
                    warp_sumsq[0] = T.rsqrt(partial_sumsq[0] / float(hidden_size) + eps)
            T.sync_threads()

            if nope_dim > 0:
                for col_base in T.serial(0, nope_dim, threads):
                    col = col_base + tx
                    if col < nope_dim:
                        value = T.cast(kv[token_id, col], T.float32) * warp_sumsq[0] * T.cast(weight[col], T.float32)
                        kv[token_id, col] = T.cast(value, tl_dtype)

            pos = positions[token_id]
            for pair_base in T.serial(0, half_dim, threads):
                pair_idx = pair_base + tx
                if pair_idx < half_dim:
                    even_col = nope_dim + pair_idx * 2
                    odd_col = even_col + 1
                    even = T.cast(kv[token_id, even_col], T.float32) * warp_sumsq[0] * T.cast(weight[even_col], T.float32)
                    odd = T.cast(kv[token_id, odd_col], T.float32) * warp_sumsq[0] * T.cast(weight[odd_col], T.float32)
                    c = freqs_real_imag[pos, pair_idx, 0]
                    s = freqs_real_imag[pos, pair_idx, 1]
                    kv[token_id, even_col] = T.cast(even * c - odd * s, tl_dtype)
                    kv[token_id, odd_col] = T.cast(even * s + odd * c, tl_dtype)

    return fused_norm_rope_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_fused_q_rmsnorm_rope_inplace_kernel(
    num_heads: int,
    positions_dtype: str,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_freq_positions = T.dynamic("num_freq_positions")
    hidden_size = 512
    rope_dim = 64
    half_dim = rope_dim // 2
    nope_dim = hidden_size - rope_dim
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    positions_stride = T.dynamic("positions_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")
    warps_per_cta = threads // 32

    @_tilelang_jit(
        tilelang,
        f"dsv4_fused_q_rmsnorm_rope_h512_r64_n{num_heads}_{positions_dtype}_dynstride_t{threads}",
    )
    def fused_q_rmsnorm_rope_inplace_kernel(
        q: T.Tensor[(num_tokens, num_heads, hidden_size), T.bfloat16],
        freqs_real_imag: T.StridedTensor[
            (num_freq_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        positions: T.StridedTensor[(num_tokens,), (positions_stride,), tl_positions_dtype],
        eps: T.float32,
    ):
        with T.Kernel(T.ceildiv(num_tokens * num_heads, warps_per_cta), threads=threads) as block_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            row_id = block_id * warps_per_cta + warp
            token_id = row_id // num_heads
            head_id = row_id % num_heads
            partial_sumsq = T.alloc_local((1,), T.float32)
            rrms = T.alloc_local((1,), T.float32)

            if row_id < num_tokens * num_heads:
                partial_sumsq[0] = 0.0
                for tile in T.unroll(4):
                    for vec in T.vectorized(4):
                        col = (tile * 4 + vec) * 32 + lane
                        value = T.cast(q[token_id, head_id, col], T.float32)
                        partial_sumsq[0] += value * value

                partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
                rrms[0] = T.rsqrt(partial_sumsq[0] / float(hidden_size) + eps)

                for tile in T.unroll(3):
                    for vec in T.vectorized(4):
                        col = (tile * 4 + vec) * 32 + lane
                        value = T.cast(q[token_id, head_id, col], T.float32) * rrms[0]
                        q[token_id, head_id, col] = T.cast(value, T.bfloat16)
                for i in T.unroll(2):
                    col = (12 + i) * 32 + lane
                    value = T.cast(q[token_id, head_id, col], T.float32) * rrms[0]
                    q[token_id, head_id, col] = T.cast(value, T.bfloat16)

                pos = positions[token_id]
                if lane < half_dim:
                    pair_idx = lane
                    even_col = nope_dim + pair_idx * 2
                    odd_col = even_col + 1
                    even_norm = T.cast(
                        T.cast(T.cast(q[token_id, head_id, even_col], T.float32) * rrms[0], T.bfloat16),
                        T.float32,
                    )
                    odd_norm = T.cast(
                        T.cast(T.cast(q[token_id, head_id, odd_col], T.float32) * rrms[0], T.bfloat16),
                        T.float32,
                    )
                    c = freqs_real_imag[pos, pair_idx, 0]
                    s = freqs_real_imag[pos, pair_idx, 1]
                    q[token_id, head_id, even_col] = T.cast(even_norm * c - odd_norm * s, T.bfloat16)
                    q[token_id, head_id, odd_col] = T.cast(even_norm * s + odd_norm * c, T.bfloat16)

    return fused_q_rmsnorm_rope_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_compress_fused_norm_rope_inplace_kernel(
    hidden_size: int,
    rope_dim: int,
    compress_ratio: int,
    kv_dtype: torch.dtype,
    seq_lens_dtype: str,
    threads: int = 256,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_positions = T.dynamic("num_positions")
    half_dim = rope_dim // 2
    nope_dim = hidden_size - rope_dim
    tl_dtype = T.float32 if kv_dtype == torch.float32 else T.bfloat16
    tl_seq_lens_dtype = T.int32 if seq_lens_dtype.lower().strip() == "int32" else T.int64
    weight_stride = T.dynamic("weight_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")
    seq_lens_stride = T.dynamic("seq_lens_stride")

    warps_per_cta = threads // 32

    @_tilelang_jit(
        tilelang,
        f"dsv4_compress_norm_rope_h{hidden_size}_r{rope_dim}_c{compress_ratio}_{str(kv_dtype).split('.')[-1]}_{seq_lens_dtype}_dynstride_t{threads}",
    )
    def compress_fused_norm_rope_inplace_kernel(
        kv: T.Tensor[(num_tokens, hidden_size), tl_dtype],
        weight: T.StridedTensor[(hidden_size,), (weight_stride,), tl_dtype],
        freqs_real_imag: T.StridedTensor[
            (num_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        seq_lens: T.StridedTensor[(num_tokens,), (seq_lens_stride,), tl_seq_lens_dtype],
        eps: T.float32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            should_process = seq_lens[token_id] % compress_ratio == 0
            partial_sumsq = T.alloc_local((1,), T.float32)
            warp_sumsq = T.alloc_shared((warps_per_cta,), T.float32)

            if should_process:
                partial_sumsq[0] = 0.0
                for col_base in T.serial(0, hidden_size, threads):
                    col = col_base + tx
                    if col < hidden_size:
                        value = T.cast(kv[token_id, col], T.float32)
                        partial_sumsq[0] += value * value
                partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
                if lane == 0:
                    warp_sumsq[warp] = partial_sumsq[0]
                T.sync_threads()

                partial_sumsq[0] = T.if_then_else(tx < warps_per_cta, warp_sumsq[tx], 0.0)
                if warp == 0:
                    partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
                    if lane == 0:
                        warp_sumsq[0] = T.rsqrt(partial_sumsq[0] / float(hidden_size) + eps)
                T.sync_threads()

                if nope_dim > 0:
                    for col_base in T.serial(0, nope_dim, threads):
                        col = col_base + tx
                        if col < nope_dim:
                            value = T.cast(kv[token_id, col], T.float32) * warp_sumsq[0] * T.cast(weight[col], T.float32)
                            kv[token_id, col] = T.cast(value, tl_dtype)

                pos = seq_lens[token_id] - compress_ratio
                if tx < half_dim:
                    pair_idx = tx
                    even_col = nope_dim + pair_idx * 2
                    odd_col = even_col + 1
                    even = T.cast(kv[token_id, even_col], T.float32) * warp_sumsq[0] * T.cast(weight[even_col], T.float32)
                    odd = T.cast(kv[token_id, odd_col], T.float32) * warp_sumsq[0] * T.cast(weight[odd_col], T.float32)
                    c = freqs_real_imag[pos, pair_idx, 0]
                    s = freqs_real_imag[pos, pair_idx, 1]
                    kv[token_id, even_col] = T.cast(even * c - odd * s, tl_dtype)
                    kv[token_id, odd_col] = T.cast(even * s + odd * c, tl_dtype)

    return compress_fused_norm_rope_inplace_kernel


@lru_cache(maxsize=None)
def _tilelang_compress_fused_norm_rope_prefill_inplace_kernel(
    hidden_size: int,
    rope_dim: int,
    kv_dtype: torch.dtype,
    positions_dtype: str = "int32",
    threads: int = 256,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_positions = T.dynamic("num_positions")
    num_compress_rows = T.dynamic("num_compress_rows")
    compress_rows_stride = T.dynamic("compress_rows_stride")
    half_dim = rope_dim // 2
    nope_dim = hidden_size - rope_dim
    tl_dtype = T.float32 if kv_dtype == torch.float32 else T.bfloat16
    tl_positions_dtype = T.int32 if positions_dtype.lower().strip() == "int32" else T.int64
    weight_stride = T.dynamic("weight_stride")
    freqs_pos_stride = T.dynamic("freqs_pos_stride")
    freqs_pair_stride = T.dynamic("freqs_pair_stride")
    freqs_component_stride = T.dynamic("freqs_component_stride")
    warps_per_cta = threads // 32

    if hidden_size % 32 == 0 and half_dim <= 32:
        @_tilelang_jit(
            tilelang,
            f"dsv4_compress_norm_rope_prefill_warp_h{hidden_size}_r{rope_dim}_{str(kv_dtype).split('.')[-1]}_{positions_dtype}_dynstride_t{threads}",
        )
        def compress_fused_norm_rope_prefill_inplace_kernel(
            kv: T.Tensor[(num_tokens, hidden_size), tl_dtype],
            weight: T.StridedTensor[(hidden_size,), (weight_stride,), tl_dtype],
            freqs_real_imag: T.StridedTensor[
                (num_positions, half_dim, 2),
                (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
                T.float32,
            ],
            compress_rows: T.StridedTensor[(num_compress_rows, 4), (compress_rows_stride, 1), tl_positions_dtype],
            eps: T.float32,
        ):
            with T.Kernel(T.ceildiv(num_compress_rows, warps_per_cta), threads=threads) as block_id:
                tx = T.get_thread_binding()
                lane = tx % 32
                warp = tx // 32
                row_id = block_id * warps_per_cta + warp
                partial_sumsq = T.alloc_local((1,), T.float32)

                if row_id < num_compress_rows:
                    ragged_id = compress_rows[row_id, 0]
                    pos = compress_rows[row_id, 2]
                    partial_sumsq[0] = 0.0
                    for col_base in T.serial(0, hidden_size, 32):
                        col = col_base + lane
                        value = T.cast(kv[ragged_id, col], T.float32)
                        partial_sumsq[0] += value * value

                    partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
                    partial_sumsq[0] = T.rsqrt(partial_sumsq[0] / float(hidden_size) + eps)

                    if nope_dim > 0:
                        for col_base in T.serial(0, nope_dim, 32):
                            col = col_base + lane
                            if col < nope_dim:
                                value = T.cast(kv[ragged_id, col], T.float32) * partial_sumsq[0] * T.cast(weight[col], T.float32)
                                kv[ragged_id, col] = T.cast(value, tl_dtype)

                    if lane < half_dim:
                        even_col = nope_dim + lane * 2
                        odd_col = even_col + 1
                        even = T.cast(kv[ragged_id, even_col], T.float32) * partial_sumsq[0] * T.cast(weight[even_col], T.float32)
                        odd = T.cast(kv[ragged_id, odd_col], T.float32) * partial_sumsq[0] * T.cast(weight[odd_col], T.float32)
                        c = freqs_real_imag[pos, lane, 0]
                        s = freqs_real_imag[pos, lane, 1]
                        kv[ragged_id, even_col] = T.cast(even * c - odd * s, tl_dtype)
                        kv[ragged_id, odd_col] = T.cast(even * s + odd * c, tl_dtype)

        return compress_fused_norm_rope_prefill_inplace_kernel

    @_tilelang_jit(
        tilelang,
        f"dsv4_compress_norm_rope_prefill_local_h{hidden_size}_r{rope_dim}_{str(kv_dtype).split('.')[-1]}_{positions_dtype}_dynstride_t{threads}",
    )
    def compress_fused_norm_rope_prefill_inplace_kernel(
        kv: T.Tensor[(num_tokens, hidden_size), tl_dtype],
        weight: T.StridedTensor[(hidden_size,), (weight_stride,), tl_dtype],
        freqs_real_imag: T.StridedTensor[
            (num_positions, half_dim, 2),
            (freqs_pos_stride, freqs_pair_stride, freqs_component_stride),
            T.float32,
        ],
        compress_rows: T.StridedTensor[(num_compress_rows, 4), (compress_rows_stride, 1), tl_positions_dtype],
        eps: T.float32,
    ):
        with T.Kernel(num_compress_rows, threads=threads) as row_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            ragged_id = compress_rows[row_id, 0]
            pos = compress_rows[row_id, 2]
            partial_sumsq = T.alloc_local((1,), T.float32)
            warp_sumsq = T.alloc_shared((warps_per_cta,), T.float32)

            partial_sumsq[0] = 0.0
            for col_base in T.serial(0, hidden_size, threads):
                col = col_base + tx
                if col < hidden_size:
                    value = T.cast(kv[ragged_id, col], T.float32)
                    partial_sumsq[0] += value * value

            partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
            if lane == 0:
                warp_sumsq[warp] = partial_sumsq[0]
            T.sync_threads()

            partial_sumsq[0] = T.if_then_else(tx < warps_per_cta, warp_sumsq[tx], 0.0)
            if warp == 0:
                partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
                if lane == 0:
                    warp_sumsq[0] = T.rsqrt(partial_sumsq[0] / float(hidden_size) + eps)
            T.sync_threads()

            if nope_dim > 0:
                for col_base in T.serial(0, nope_dim, threads):
                    col = col_base + tx
                    if col < nope_dim:
                        value = T.cast(kv[ragged_id, col], T.float32) * warp_sumsq[0] * T.cast(weight[col], T.float32)
                        kv[ragged_id, col] = T.cast(value, tl_dtype)

            for pair_base in T.serial(0, half_dim, threads):
                pair_idx = pair_base + tx
                if pair_idx < half_dim:
                    even_col = nope_dim + pair_idx * 2
                    odd_col = even_col + 1
                    even = T.cast(kv[ragged_id, even_col], T.float32) * warp_sumsq[0] * T.cast(weight[even_col], T.float32)
                    odd = T.cast(kv[ragged_id, odd_col], T.float32) * warp_sumsq[0] * T.cast(weight[odd_col], T.float32)
                    c = freqs_real_imag[pos, pair_idx, 0]
                    s = freqs_real_imag[pos, pair_idx, 1]
                    kv[ragged_id, even_col] = T.cast(even * c - odd * s, tl_dtype)
                    kv[ragged_id, odd_col] = T.cast(even * s + odd * c, tl_dtype)

    return compress_fused_norm_rope_prefill_inplace_kernel

__all__ = [
    '_tilelang_rmsnorm_self_kernel',
    '_tilelang_rmsnorm_self_strided_kernel',
    '_tilelang_weighted_rmsnorm_kernel',
    '_tilelang_weighted_rmsnorm_strided_kernel',
    '_tilelang_weighted_rmsnorm_strided_inplace_kernel',
    '_tilelang_hadamard128_inplace_kernel',
    '_tilelang_rope_hadamard_inplace_kernel_fast',
    '_tilelang_neox_rope_hadamard_inplace_kernel_fast',
    '_tilelang_rope_hadamard_inplace_kernel',
    '_tilelang_rope_inplace_kernel',
    '_tilelang_rope_inplace_flat_kernel',
    '_tilelang_fused_norm_rope_inplace_kernel',
    '_tilelang_fused_q_rmsnorm_rope_inplace_kernel',
    '_tilelang_compress_fused_norm_rope_inplace_kernel',
    '_tilelang_compress_fused_norm_rope_prefill_inplace_kernel',
]
