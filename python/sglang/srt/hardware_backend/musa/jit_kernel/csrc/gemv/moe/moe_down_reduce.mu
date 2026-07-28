#include "../common.muh"

// Fused MoE down projection and route reduction.
//
// One subwarp owns one output channel for one token.  It walks all routes,
// accumulates route_weight * dot(intermediate[route], expert_weight) in FP32,
// and writes the final BF16 value once.  Compared with the regular path this
// removes the [token, topk, hidden] down-projection output and the following
// moe_sum_reduce launch.  Route parallelism is deliberately traded for fewer
// launches and less intermediate traffic, so this remains an opt-in prototype
// until exact-shape pipeline measurements justify dispatching it.

// Route-parallel BF16 variant for the launch-bound M=1..4 regime.  The
// serial kernel below gives each output subwarp all routes; this variant gives
// each (output, route) pair its own subwarp, then performs the route reduction
// in shared memory.  It restores route parallelism without materializing the
// [token, top-k, hidden] tensor.  Configs 3..6 are intentionally BF16-only
// and are selected only when exact-shape measurements clear the same guard.
template <int kBlockN, int kBlockK, int kTopK>
__global__ void musa_moe_down_reduce_bf16_route_parallel_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int nr_experts,
    int n,
    int k,
    float routed_scale) {
    constexpr int kVec = 8;
    constexpr int kKStep = kBlockK * kVec;
    static_assert(kBlockK <= 32 && (kBlockK & (kBlockK - 1)) == 0);
    __shared__ float route_partial[kBlockN * kTopK];

    const int token_idx = blockIdx.y;
    const int subwarp = threadIdx.x / kBlockK;
    const int t_k_idx = threadIdx.x % kBlockK;
    const int route = subwarp % kTopK;
    const int t_n_idx = subwarp / kTopK;
    const int n_idx = blockIdx.x * kBlockN + t_n_idx;
    const bool valid_n = n_idx < n;
    const int expert_idx = topk_ids[token_idx * kTopK + route];
    float sum = 0.f;
    if (valid_n && expert_idx >= 0 && expert_idx < nr_experts) {
        const bfloat16_t* route_a =
            a_ptr + (static_cast<size_t>(token_idx) * kTopK + route) * k;
        const bfloat16_t* row_b =
            b_ptr + (static_cast<size_t>(expert_idx) * n + n_idx) * k;
        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += kKStep) {
            const int k_offset = k_base + t_k_idx * kVec;
            bfloat16_t a_reg[kVec];
            bfloat16_t b_reg[kVec];
            using Load = typename VecType<bfloat16_t, 128>::Ttype;
            if (k_offset + kVec <= k) {
                *reinterpret_cast<Load*>(a_reg) =
                    *reinterpret_cast<const Load*>(route_a + k_offset);
                *reinterpret_cast<int4*>(b_reg) = __lsu_ld_cache_hint(
                    reinterpret_cast<const int4*>(row_b + k_offset),
                    0, 2, 1, 1);
            } else {
                #pragma unroll
                for (int i = 0; i < kVec; ++i) {
                    const int idx = k_offset + i;
                    a_reg[i] = idx < k ? route_a[idx] : bfloat16_t(0);
                    b_reg[i] = idx < k ? row_b[idx] : bfloat16_t(0);
                }
            }
            #pragma unroll
            for (int i = 0; i < kVec; ++i) {
                sum += float(a_reg[i]) * float(b_reg[i]);
            }
        }
        sum *= topk_weights[token_idx * kTopK + route] * routed_scale;
    }
    #pragma unroll
    for (int offset = kBlockK / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, kBlockK);
    }
    if (t_k_idx == 0) {
        route_partial[t_n_idx * kTopK + route] = sum;
    }
    __SYNCTHREADS_LM;
    if (threadIdx.x < kBlockN) {
        const int out_n_idx = blockIdx.x * kBlockN + threadIdx.x;
        float total = 0.f;
        #pragma unroll
        for (int route_idx = 0; route_idx < kTopK; ++route_idx) {
            total += route_partial[threadIdx.x * kTopK + route_idx];
        }
        if (out_n_idx < n) {
            c_ptr[token_idx * n + out_n_idx] = bfloat16_t(total);
        }
    }
}

template <int kBlockN, int kBlockK, int kTopK>
void launch_moe_down_reduce_bf16_route_parallel(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView topk_weights,
    ffi::TensorView topk_ids,
    float routed_scale,
    musaStream_t stream) {
    const int tokens = A.size(0);
    const int k = A.size(2);
    const int experts = B.size(0);
    const int n = B.size(1);
    dim3 block(kBlockN * kBlockK * kTopK, 1, 1);
    dim3 grid(ceil_div(n, kBlockN), tokens, 1);
    musa_moe_down_reduce_bf16_route_parallel_kernel<kBlockN, kBlockK, kTopK>
        <<<grid, block, 0, stream>>>(
            static_cast<bfloat16_t*>(C.data_ptr()),
            static_cast<const bfloat16_t*>(A.data_ptr()),
            static_cast<const bfloat16_t*>(B.data_ptr()),
            static_cast<const int*>(topk_ids.data_ptr()),
            static_cast<const float*>(topk_weights.data_ptr()),
            experts, n, k, routed_scale);
}

template <int kBlockN, int kBlockK, int kTopK>
__global__ void musa_moe_down_reduce_fp8_route_parallel_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const __mt_fp8_e4m3* __restrict__ b_ptr,
    const float* __restrict__ b_scale,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int nr_experts,
    int n,
    int k,
    int scale_n_len,
    int scale_k_len,
    float routed_scale) {
    constexpr int kVec = 16;
    constexpr int kKStep = kBlockK * kVec;
    static_assert(kBlockK <= 32 && (kBlockK & (kBlockK - 1)) == 0);
    using ALoad = typename VecType<bfloat16_t, 128>::Ttype;
    using Fp8x4 = unsigned char __attribute__((vector_size(4)));
    using Half4 = _Float16 __attribute__((ext_vector_type(4)));
    __shared__ float route_partial[kBlockN * kTopK];

    const int token_idx = blockIdx.y;
    const int subwarp = threadIdx.x / kBlockK;
    const int t_k_idx = threadIdx.x % kBlockK;
    const int route = subwarp % kTopK;
    const int t_n_idx = subwarp / kTopK;
    const int n_idx = blockIdx.x * kBlockN + t_n_idx;
    const bool valid_n = n_idx < n;
    const int expert_idx = topk_ids[token_idx * kTopK + route];
    float sum = 0.f;
    if (valid_n && expert_idx >= 0 && expert_idx < nr_experts) {
        const float route_weight =
            topk_weights[token_idx * kTopK + route] * routed_scale;
        const bfloat16_t* route_a =
            a_ptr + (static_cast<size_t>(token_idx) * kTopK + route) * k;
        const __mt_fp8_e4m3* row_b =
            b_ptr + (static_cast<size_t>(expert_idx) * n + n_idx) * k;
        const float* row_scale =
            b_scale + (static_cast<size_t>(expert_idx) * scale_n_len +
                       n_idx / 128) * scale_k_len;
        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += kKStep) {
            const int k_offset = k_base + t_k_idx * kVec;
            if (k_offset < k) {
                bfloat16_t a_reg[kVec];
                __mt_fp8_e4m3 b_reg[kVec];
                *reinterpret_cast<ALoad*>(a_reg) =
                    *reinterpret_cast<const ALoad*>(route_a + k_offset);
                *reinterpret_cast<ALoad*>(a_reg + 8) =
                    *reinterpret_cast<const ALoad*>(route_a + k_offset + 8);
                *reinterpret_cast<int4*>(b_reg) = __lsu_ld_cache_hint(
                    reinterpret_cast<const int4*>(row_b + k_offset),
                    0, 2, 1, 1);
                const float scale = row_scale[k_offset / 128];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const Half4 b4 = __musa_e4m32f16_rn_bst4(
                        reinterpret_cast<const Fp8x4*>(b_reg)[i]);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        sum += route_weight * scale * float(a_reg[i * 4 + j]) *
                               b4[j];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int offset = kBlockK / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, kBlockK);
    }
    if (t_k_idx == 0) {
        route_partial[t_n_idx * kTopK + route] = sum;
    }
    __SYNCTHREADS_LM;
    if (threadIdx.x < kBlockN) {
        const int out_n_idx = blockIdx.x * kBlockN + threadIdx.x;
        float total = 0.f;
        #pragma unroll
        for (int route_idx = 0; route_idx < kTopK; ++route_idx) {
            total += route_partial[threadIdx.x * kTopK + route_idx];
        }
        if (out_n_idx < n) {
            c_ptr[token_idx * n + out_n_idx] = bfloat16_t(total);
        }
    }
}

template <int kBlockN, int kBlockK, int kTopK>
void launch_moe_down_reduce_fp8_route_parallel(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView B_scale,
    ffi::TensorView topk_weights,
    ffi::TensorView topk_ids,
    float routed_scale,
    musaStream_t stream) {
    const int tokens = A.size(0);
    const int k = A.size(2);
    const int experts = B.size(0);
    const int n = B.size(1);
    const int scale_n_len = ceil_div(n, 128);
    const int scale_k_len = ceil_div(k, 128);
    dim3 block(kBlockN * kBlockK * kTopK, 1, 1);
    dim3 grid(ceil_div(n, kBlockN), tokens, 1);
    musa_moe_down_reduce_fp8_route_parallel_kernel<kBlockN, kBlockK, kTopK>
        <<<grid, block, 0, stream>>>(
            static_cast<bfloat16_t*>(C.data_ptr()),
            static_cast<const bfloat16_t*>(A.data_ptr()),
            static_cast<const __mt_fp8_e4m3*>(B.data_ptr()),
            static_cast<const float*>(B_scale.data_ptr()),
            static_cast<const int*>(topk_ids.data_ptr()),
            static_cast<const float*>(topk_weights.data_ptr()),
            experts, n, k, scale_n_len, scale_k_len, routed_scale);
}

template <int kBlockN, int kBlockK, int kTopK>
__global__ void musa_moe_down_reduce_bf16_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int nr_experts,
    int n,
    int k,
    float routed_scale) {
    constexpr int kVec = 8;
    constexpr int kKStep = kBlockK * kVec;
    static_assert(kBlockK <= 32 && (kBlockK & (kBlockK - 1)) == 0);

    const int token_idx = blockIdx.y;
    const int t_n_idx = threadIdx.x / kBlockK;
    const int t_k_idx = threadIdx.x % kBlockK;
    const int n_idx = blockIdx.x * kBlockN + t_n_idx;
    if (n_idx >= n) {
        return;
    }

    float partial[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int route = 0; route < kTopK; ++route) {
        const int expert_idx = topk_ids[token_idx * kTopK + route];
        if (expert_idx < 0 || expert_idx >= nr_experts) {
            continue;
        }
        const float route_weight =
            topk_weights[token_idx * kTopK + route] * routed_scale;
        const bfloat16_t* route_a =
            a_ptr + (static_cast<size_t>(token_idx) * kTopK + route) * k;
        const bfloat16_t* row_b =
            b_ptr + (static_cast<size_t>(expert_idx) * n + n_idx) * k;

        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += kKStep) {
            const int k_offset = k_base + t_k_idx * kVec;
            bfloat16_t a_reg[kVec];
            bfloat16_t b_reg[kVec];
            using Load = typename VecType<bfloat16_t, 128>::Ttype;
            if (k_offset + kVec <= k) {
                *reinterpret_cast<Load*>(a_reg) =
                    *reinterpret_cast<const Load*>(route_a + k_offset);
                *reinterpret_cast<int4*>(b_reg) = __lsu_ld_cache_hint(
                    reinterpret_cast<const int4*>(row_b + k_offset),
                    0, 2, 1, 1);
            } else {
                #pragma unroll
                for (int i = 0; i < kVec; ++i) {
                    const int idx = k_offset + i;
                    a_reg[i] = idx < k ? route_a[idx] : bfloat16_t(0);
                    b_reg[i] = idx < k ? row_b[idx] : bfloat16_t(0);
                }
            }
            #pragma unroll
            for (int i = 0; i < kVec; ++i) {
                partial[i & 3] +=
                    route_weight * float(a_reg[i]) * float(b_reg[i]);
            }
        }
    }

    float sum = partial[0] + partial[1] + partial[2] + partial[3];
    #pragma unroll
    for (int offset = kBlockK / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, kBlockK);
    }
    if (t_k_idx == 0) {
        c_ptr[token_idx * n + n_idx] = bfloat16_t(sum);
    }
}

template <int kBlockN, int kBlockK, int kTopK>
__global__ void musa_moe_down_reduce_fp8_block128_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const __mt_fp8_e4m3* __restrict__ b_ptr,
    const float* __restrict__ b_scale,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int nr_experts,
    int n,
    int k,
    int scale_n_len,
    int scale_k_len,
    float routed_scale) {
    constexpr int kVec = 16;
    constexpr int kKStep = kBlockK * kVec;
    static_assert(kBlockK <= 32 && (kBlockK & (kBlockK - 1)) == 0);

    const int token_idx = blockIdx.y;
    const int t_n_idx = threadIdx.x / kBlockK;
    const int t_k_idx = threadIdx.x % kBlockK;
    const int n_idx = blockIdx.x * kBlockN + t_n_idx;
    if (n_idx >= n) {
        return;
    }

    using ALoad = typename VecType<bfloat16_t, 128>::Ttype;
    using Half4 = _Float16 __attribute__((ext_vector_type(4)));
    using Fp8x4 = unsigned char __attribute__((vector_size(4)));
    float partial[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int route = 0; route < kTopK; ++route) {
        const int expert_idx = topk_ids[token_idx * kTopK + route];
        if (expert_idx < 0 || expert_idx >= nr_experts) {
            continue;
        }
        const float route_weight =
            topk_weights[token_idx * kTopK + route] * routed_scale;
        const bfloat16_t* route_a =
            a_ptr + (static_cast<size_t>(token_idx) * kTopK + route) * k;
        const __mt_fp8_e4m3* row_b =
            b_ptr + (static_cast<size_t>(expert_idx) * n + n_idx) * k;
        const float* row_scale =
            b_scale + (static_cast<size_t>(expert_idx) * scale_n_len +
                       n_idx / 128) * scale_k_len;

        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += kKStep) {
            const int k_offset = k_base + t_k_idx * kVec;
            // FP8 CSV shapes are block-128 aligned, but kBlockK*kVec can be
            // 256.  Guard the inactive half-subwarps for K=384, 640, ...;
            // each active lane still owns a complete aligned 16-element load.
            if (k_offset < k) {
                bfloat16_t a_reg[kVec];
                __mt_fp8_e4m3 b_reg[kVec];
                *reinterpret_cast<ALoad*>(a_reg) =
                    *reinterpret_cast<const ALoad*>(route_a + k_offset);
                *reinterpret_cast<ALoad*>(a_reg + 8) =
                    *reinterpret_cast<const ALoad*>(route_a + k_offset + 8);
                *reinterpret_cast<int4*>(b_reg) = __lsu_ld_cache_hint(
                    reinterpret_cast<const int4*>(row_b + k_offset),
                    0, 2, 1, 1);
                const float scale = row_scale[k_offset / 128];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const Half4 b4 = __musa_e4m32f16_rn_bst4(
                        reinterpret_cast<const Fp8x4*>(b_reg)[i]);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        partial[i] += route_weight * scale *
                            float(a_reg[i * 4 + j]) * b4[j];
                    }
                }
            }
        }
    }

    float sum = partial[0] + partial[1] + partial[2] + partial[3];
    #pragma unroll
    for (int offset = kBlockK / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, kBlockK);
    }
    if (t_k_idx == 0) {
        c_ptr[token_idx * n + n_idx] = bfloat16_t(sum);
    }
}

template <int kBlockN, int kBlockK, int kTopK>
void launch_moe_down_reduce_typed(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView B_scale,
    ffi::TensorView topk_weights,
    ffi::TensorView topk_ids,
    float routed_scale,
    musaStream_t stream) {
    const int tokens = A.size(0);
    const int k = A.size(2);
    const int experts = B.size(0);
    const int n = B.size(1);
    dim3 block(kBlockN * kBlockK, 1, 1);
    dim3 grid(ceil_div(n, kBlockN), tokens, 1);
    if (dtype_equal(B.dtype(), dl_bfloat16)) {
        musa_moe_down_reduce_bf16_kernel<kBlockN, kBlockK, kTopK>
            <<<grid, block, 0, stream>>>(
                static_cast<bfloat16_t*>(C.data_ptr()),
                static_cast<const bfloat16_t*>(A.data_ptr()),
                static_cast<const bfloat16_t*>(B.data_ptr()),
                static_cast<const int*>(topk_ids.data_ptr()),
                static_cast<const float*>(topk_weights.data_ptr()),
                experts, n, k, routed_scale);
    } else {
        const int scale_n_len = ceil_div(n, 128);
        const int scale_k_len = ceil_div(k, 128);
        musa_moe_down_reduce_fp8_block128_kernel<kBlockN, kBlockK, kTopK>
            <<<grid, block, 0, stream>>>(
                static_cast<bfloat16_t*>(C.data_ptr()),
                static_cast<const bfloat16_t*>(A.data_ptr()),
                static_cast<const __mt_fp8_e4m3*>(B.data_ptr()),
                static_cast<const float*>(B_scale.data_ptr()),
                static_cast<const int*>(topk_ids.data_ptr()),
                static_cast<const float*>(topk_weights.data_ptr()),
                experts, n, k, scale_n_len, scale_k_len, routed_scale);
    }
}

void sgl_musa_moe_down_reduce(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView B_scale,
    ffi::TensorView topk_weights,
    ffi::TensorView topk_ids,
    int64_t topk,
    double routed_scale,
    int64_t config_id) {
    TORCH_CHECK(A.ndim() == 3, "A must be [tokens, topk, intermediate]");
    TORCH_CHECK(B.ndim() == 3, "B must be [experts, hidden, intermediate]");
    TORCH_CHECK(C.ndim() == 2, "C must be [tokens, hidden]");
    TORCH_CHECK(dtype_equal(A.dtype(), dl_bfloat16), "A must be BF16");
    TORCH_CHECK(dtype_equal(C.dtype(), dl_bfloat16), "C must be BF16");
    TORCH_CHECK(dtype_equal(topk_ids.dtype(), dl_int32), "topk_ids must be int32");
    TORCH_CHECK(dtype_equal(topk_weights.dtype(), dl_float32),
                "topk_weights must be float32");
    TORCH_CHECK(dtype_equal(B.dtype(), dl_bfloat16) ||
                dtype_equal(B.dtype(), dl_float8_e4m3fn),
                "B must be BF16 or FP8 E4M3");
    TORCH_CHECK(A.size(1) == topk, "A topk mismatch");
    TORCH_CHECK(topk_ids.size(0) == A.size(0) && topk_ids.size(1) == topk,
                "topk_ids shape mismatch");
    TORCH_CHECK(topk_weights.size(0) == A.size(0) &&
                topk_weights.size(1) == topk, "topk_weights shape mismatch");
    TORCH_CHECK(B.size(2) == A.size(2), "intermediate mismatch");
    TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1),
                "output shape mismatch");
    TORCH_CHECK(A.stride(2) == 1 && A.stride(1) == A.size(2) &&
                A.stride(0) == A.size(1) * A.size(2), "A must be contiguous");
    TORCH_CHECK(B.stride(2) == 1 && B.stride(1) == B.size(2) &&
                B.stride(0) == B.size(1) * B.size(2), "B must be contiguous");
    TORCH_CHECK(C.stride(1) == 1 && C.stride(0) == C.size(1),
                "C must be contiguous");
    TORCH_CHECK(topk_ids.stride(1) == 1 &&
                topk_ids.stride(0) == topk_ids.size(1),
                "topk_ids must be contiguous");
    TORCH_CHECK(topk_weights.stride(1) == 1 &&
                topk_weights.stride(0) == topk_weights.size(1),
                "topk_weights must be contiguous");
    if (dtype_equal(B.dtype(), dl_float8_e4m3fn)) {
        TORCH_CHECK(A.size(2) % 128 == 0,
                    "FP8 block-128 prototype requires K divisible by 128");
        TORCH_CHECK(dtype_equal(B_scale.dtype(), dl_float32),
                    "FP8 B_scale must be float32");
        TORCH_CHECK(B_scale.ndim() == 3 && B_scale.stride(2) == 1 &&
                    B_scale.stride(1) == B_scale.size(2) &&
                    B_scale.stride(0) == B_scale.size(1) * B_scale.size(2),
                    "FP8 B_scale must be contiguous");
        TORCH_CHECK(B_scale.size(0) == B.size(0) &&
                    B_scale.size(1) == ceil_div(B.size(1), 128) &&
                    B_scale.size(2) == ceil_div(B.size(2), 128),
                    "FP8 B_scale must use block shape 128x128");
    }
    TORCH_CHECK(config_id >= -1 && config_id <= 8,
                "down-reduce config_id must be -1 through 8");

    ffi::MUSADeviceGuard device_guard(A.device().device_id);
    musaStream_t stream = get_stream(A.device());
    const int selected = config_id < 0 ? 0 : config_id;

    if (selected >= 7) {
        TORCH_CHECK(dtype_equal(B.dtype(), dl_float8_e4m3fn),
                    "FP8 route-parallel configs require FP8 weights");
#define LAUNCH_FP8_ROUTE_PARALLEL(BN, BK) \
        do { \
            switch (topk) { \
                case 6: launch_moe_down_reduce_fp8_route_parallel<BN, BK, 6>( \
                    A, B, C, B_scale, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                case 8: launch_moe_down_reduce_fp8_route_parallel<BN, BK, 8>( \
                    A, B, C, B_scale, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                case 9: launch_moe_down_reduce_fp8_route_parallel<BN, BK, 9>( \
                    A, B, C, B_scale, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                case 10: launch_moe_down_reduce_fp8_route_parallel<BN, BK, 10>( \
                    A, B, C, B_scale, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                case 11: launch_moe_down_reduce_fp8_route_parallel<BN, BK, 11>( \
                    A, B, C, B_scale, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                default: TORCH_CHECK(false, \
                    "prototype supports topk 6, 8, 9, 10, or 11"); \
            } \
        } while (0)
        if (selected == 7) {
            LAUNCH_FP8_ROUTE_PARALLEL(4, 4);
        } else {
            LAUNCH_FP8_ROUTE_PARALLEL(2, 4);
        }
#undef LAUNCH_FP8_ROUTE_PARALLEL
        const musaError_t fp8_route_parallel_err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(fp8_route_parallel_err, musaSuccess)
            << "MUSA MoE FP8 route-parallel down-reduce failed: "
            << musaGetErrorString(fp8_route_parallel_err);
        return;
    }

    if (selected >= 3) {
        TORCH_CHECK(dtype_equal(B.dtype(), dl_bfloat16),
                    "route-parallel down-reduce configs are BF16-only");
#define LAUNCH_ROUTE_PARALLEL(BN, BK) \
        do { \
            switch (topk) { \
                case 6: launch_moe_down_reduce_bf16_route_parallel<BN, BK, 6>( \
                    A, B, C, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                case 8: launch_moe_down_reduce_bf16_route_parallel<BN, BK, 8>( \
                    A, B, C, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                case 9: launch_moe_down_reduce_bf16_route_parallel<BN, BK, 9>( \
                    A, B, C, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                case 10: launch_moe_down_reduce_bf16_route_parallel<BN, BK, 10>( \
                    A, B, C, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                case 11: launch_moe_down_reduce_bf16_route_parallel<BN, BK, 11>( \
                    A, B, C, topk_weights, topk_ids, \
                    static_cast<float>(routed_scale), stream); break; \
                default: TORCH_CHECK(false, \
                    "prototype supports topk 6, 8, 9, 10, or 11"); \
            } \
        } while (0)
        switch (selected) {
            case 3: LAUNCH_ROUTE_PARALLEL(4, 8); break;
            case 4: LAUNCH_ROUTE_PARALLEL(2, 8); break;
            case 5: LAUNCH_ROUTE_PARALLEL(8, 8); break;
            default: LAUNCH_ROUTE_PARALLEL(4, 4); break;
        }
#undef LAUNCH_ROUTE_PARALLEL
        const musaError_t route_parallel_err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(route_parallel_err, musaSuccess)
            << "MUSA MoE BF16 route-parallel down-reduce failed: "
            << musaGetErrorString(route_parallel_err);
        return;
    }

#define LAUNCH_CONFIG(BN, BK, TOPK) \
    launch_moe_down_reduce_typed<BN, BK, TOPK>( \
        A, B, C, B_scale, topk_weights, topk_ids, \
        static_cast<float>(routed_scale), stream)
#define LAUNCH_TOPK(BN, BK) \
    do { \
        switch (topk) { \
            case 6: LAUNCH_CONFIG(BN, BK, 6); break; \
            case 8: LAUNCH_CONFIG(BN, BK, 8); break; \
            case 9: LAUNCH_CONFIG(BN, BK, 9); break; \
            case 10: LAUNCH_CONFIG(BN, BK, 10); break; \
            case 11: LAUNCH_CONFIG(BN, BK, 11); break; \
            default: TORCH_CHECK(false, "prototype supports topk 6, 8, 9, 10, or 11"); \
        } \
    } while (0)
    switch (selected) {
        case 0: LAUNCH_TOPK(32, 4); break;
        case 1: LAUNCH_TOPK(16, 8); break;
        default: LAUNCH_TOPK(8, 16); break;
    }
#undef LAUNCH_TOPK
#undef LAUNCH_CONFIG

    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA MoE down-reduce failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_moe_down_reduce, sgl_musa_moe_down_reduce);
