#include "moe_gemv_kernel.muh"

// One subwarp owns a gate/up output pair for one token and routed expert.
// Keeping both accumulators together removes the shared-memory gate/up
// rendezvous from the BF16 MoE path while preserving one independent route
// per CTA.  Unlike dense BGEMV, routes are intentionally not batched because
// adjacent tokens generally select different experts.
template <int kBlockN, int kBlockK, int kStaticK, int kTopK, bool kMulRoutedWeight,
          bool kCheckBounds>
__global__ void musa_moe_gemv_bf16_swiglu_dual_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int nr_experts,
    int full_n,
    int k) {
    constexpr int VEC = 8;
    constexpr int K_STEP = kBlockK * VEC;
    const int matrix_k = kStaticK > 0 ? kStaticK : k;
    static_assert(kBlockK <= 32 && (kBlockK & (kBlockK - 1)) == 0);

    const int route_idx = blockIdx.x % kTopK;
    const int n_block_idx = blockIdx.x / kTopK;
    const int token_idx = blockIdx.y;
    const int t_n_idx = threadIdx.x / kBlockK;
    const int t_k_idx = threadIdx.x % kBlockK;
    const int out_n_idx = n_block_idx * kBlockN + t_n_idx;
    const int up_n_idx = out_n_idx + full_n / 2;
    const int expert_idx = topk_ids[token_idx * kTopK + route_idx];
    const bool valid_n = !kCheckBounds || out_n_idx < full_n / 2;

    if (expert_idx < 0 || expert_idx >= nr_experts) {
        if (t_k_idx == 0 && valid_n) {
            c_ptr[(token_idx * kTopK + route_idx) * (full_n / 2) + out_n_idx] =
                bfloat16_t(0);
        }
        return;
    }

    float gate = 0.f;
    float up = 0.f;
    if (valid_n) {
        #pragma unroll 1
        for (int k_base = 0; k_base < matrix_k; k_base += K_STEP) {
            const int k_offset = k_base + t_k_idx * VEC;
            bfloat16_t a_reg[VEC];
            bfloat16_t gate_reg[VEC];
            bfloat16_t up_reg[VEC];
            using Load = typename VecType<bfloat16_t, 128>::Ttype;
            if (k_offset + VEC <= matrix_k) {
                *reinterpret_cast<Load*>(a_reg) = *reinterpret_cast<const Load*>(
                    a_ptr + token_idx * matrix_k + k_offset);
                *reinterpret_cast<int4*>(gate_reg) = __lsu_ld_cache_hint(
                    reinterpret_cast<const int4*>(
                        b_ptr + static_cast<size_t>(expert_idx) * full_n * matrix_k +
                        out_n_idx * matrix_k + k_offset),
                    0, 2, 1, 1);
                *reinterpret_cast<int4*>(up_reg) = __lsu_ld_cache_hint(
                    reinterpret_cast<const int4*>(
                        b_ptr + static_cast<size_t>(expert_idx) * full_n * matrix_k +
                        up_n_idx * matrix_k + k_offset),
                    0, 2, 1, 1);
            } else {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int idx = k_offset + i;
                    a_reg[i] = idx < matrix_k
                        ? a_ptr[token_idx * matrix_k + idx] : bfloat16_t(0);
                    gate_reg[i] = idx < matrix_k
                        ? b_ptr[static_cast<size_t>(expert_idx) * full_n * matrix_k +
                                out_n_idx * matrix_k + idx]
                        : bfloat16_t(0);
                    up_reg[i] = idx < matrix_k
                        ? b_ptr[static_cast<size_t>(expert_idx) * full_n * matrix_k +
                                up_n_idx * matrix_k + idx]
                        : bfloat16_t(0);
                }
            }
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                const float a = float(a_reg[i]);
                gate += a * float(gate_reg[i]);
                up += a * float(up_reg[i]);
            }
        }
    }

    #pragma unroll
    for (int offset = kBlockK / 2; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffff, gate, offset, kBlockK);
        up += __shfl_down_sync(0xffffffff, up, offset, kBlockK);
    }
    if (t_k_idx == 0 && valid_n) {
        if constexpr (kMulRoutedWeight) {
            const float routed_weight = topk_weights[token_idx * kTopK + route_idx];
            gate *= routed_weight;
            up *= routed_weight;
        }
        c_ptr[(token_idx * kTopK + route_idx) * (full_n / 2) + out_n_idx] =
            gate * sigmoid(gate) * up;
    }
}

template <int kBlockN, int kBlockK, int kK, int kTopK, bool kMulRoutedWeight>
__global__ void musa_moe_gemv_bf16_swiglu_static_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int nr_experts,
    int full_n) {
    constexpr int kHalfBlockN = kBlockN / 2;
    constexpr int kVec = 8;
    constexpr int kKStep = kBlockK * kVec;

    const int route_idx = blockIdx.x % kTopK;
    const int n_block_idx = blockIdx.x / kTopK;
    const int token_idx = blockIdx.y;
    const int t_n_idx = threadIdx.x / kBlockK;
    const int t_k_idx = threadIdx.x % kBlockK;
    const int out_n_idx = n_block_idx * kHalfBlockN +
        (t_n_idx & (kHalfBlockN - 1));
    const int n_idx = out_n_idx +
        (t_n_idx >= kHalfBlockN ? full_n / 2 : 0);
    const int expert_idx = topk_ids[token_idx * kTopK + route_idx];

    if (expert_idx < 0 || expert_idx >= nr_experts) {
        if (threadIdx.x < kHalfBlockN) {
            const int dst_n_idx = n_block_idx * kHalfBlockN + threadIdx.x;
            if (dst_n_idx < full_n / 2) {
                c_ptr[(token_idx * kTopK + route_idx) * (full_n / 2) + dst_n_idx] = 0;
            }
        }
        return;
    }

    float sum = 0.f;
    #pragma unroll 1
    for (int k_base = 0; k_base < kK; k_base += kKStep) {
        bfloat16_t a_reg[kVec];
        bfloat16_t b_reg[kVec];
        *reinterpret_cast<int4*>(a_reg) = *reinterpret_cast<const int4*>(
            a_ptr + token_idx * kK + k_base + t_k_idx * kVec);
        if (n_idx < full_n) {
            *reinterpret_cast<int4*>(b_reg) = __lsu_ld_cache_hint(
                reinterpret_cast<const int4*>(
                    b_ptr + static_cast<size_t>(expert_idx) * full_n * kK +
                    n_idx * kK + k_base + t_k_idx * kVec),
                4);
        }
        #pragma unroll
        for (int i = 0; i < kVec; ++i) {
            sum += float(a_reg[i]) * float(b_reg[i]);
        }
    }

    __shared__ float reduction[kBlockN * kBlockK];
    reduction[threadIdx.x] = sum;
    __SYNCTHREADS_LM;
    if (threadIdx.x < kBlockN) {
        sum = 0.f;
        #pragma unroll
        for (int i = 0; i < kBlockK; ++i) {
            sum += reduction[threadIdx.x * kBlockK + i];
        }
        if constexpr (kMulRoutedWeight) {
            sum *= topk_weights[token_idx * kTopK + route_idx];
        }
        reduction[threadIdx.x] = sum;
    }
    __SYNCTHREADS_LM;
    if (threadIdx.x < kHalfBlockN) {
        const int dst_n_idx = n_block_idx * kHalfBlockN + threadIdx.x;
        if (dst_n_idx < full_n / 2) {
            const float gate = reduction[threadIdx.x];
            const float up = reduction[threadIdx.x + kHalfBlockN];
            c_ptr[(token_idx * kTopK + route_idx) * (full_n / 2) + dst_n_idx] =
                gate * sigmoid(gate) * up;
        }
    }
}

#define CAL_MOE_GEMV(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU) \
    musa_moe_gemv_generic_kernel<_ADTYPE, _BDTYPE, _ADTYPE, _TOPK_WEIGHT_DTYPE, float, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, false, false, false, 1, false, check_bounds> \
        <<<grid_size, block_size, shmem_size, stream>>>( \
            static_cast<_ADTYPE*>(C.data_ptr()), \
            static_cast<_ADTYPE*>(A.data_ptr()), \
            static_cast<_BDTYPE*>(B.data_ptr()), \
            static_cast<int*>(topk_ids_ptr), \
            static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
            nullptr, \
            nullptr, \
            topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
            nullptr, nullptr, nullptr, eps); \
    return;

#define RUN_BF16_ROUTE(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE) \
    if (mul_routed_weight) { \
        if (use_swigelu) { \
            CAL_MOE_GEMV(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, true, true) \
        } else { \
            CAL_MOE_GEMV(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, true, false) \
        } \
    } else { \
        if (use_swigelu) { \
            CAL_MOE_GEMV(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, false, true) \
        } else { \
            CAL_MOE_GEMV(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, false, false) \
        } \
    }

#define GEN_LAUNCH_BF16(_BLK_N, _BLK_K, _CHECK_BOUNDS) \
    { \
        launch_kernel = [&]() { \
            constexpr int block_n = _BLK_N; \
            constexpr int block_k = _BLK_K; \
            constexpr bool check_bounds = _CHECK_BOUNDS; \
            TORCH_CHECK(hidden_size % block_k == 0, "gemv k need align"); \
            dim3 block_size{block_n * block_k, 1, 1}; \
            dim3 grid_size{(uint32_t)ceil_div(reduce_size, block_n), (uint32_t)topk, (uint32_t)bseqlen}; \
            int shmem_size = block_n * sizeof(float) * block_k; \
            if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                if (dtype_equal(topk_weights.dtype(), dl_float32)) { \
                    RUN_BF16_ROUTE(bfloat16_t, bfloat16_t, float) \
                } else if (dtype_equal(topk_weights.dtype(), dl_bfloat16)) { \
                    RUN_BF16_ROUTE(bfloat16_t, bfloat16_t, bfloat16_t) \
                } \
            } else if (dtype_equal(A.dtype(), dl_float16)) { \
                if (dtype_equal(topk_weights.dtype(), dl_float32)) { \
                    RUN_BF16_ROUTE(float16_t, float16_t, float) \
                } else if (dtype_equal(topk_weights.dtype(), dl_float16)) { \
                    RUN_BF16_ROUTE(float16_t, float16_t, float16_t) \
                } \
            } \
            TORCH_CHECK(false, "no support on bf16 moe gemv"); \
        }; \
    }

#define SELECT_LAUNCH_BF16(_BLK_N, _BLK_K) \
    if (nr_n % _BLK_N == 0) { \
        GEN_LAUNCH_BF16(_BLK_N, _BLK_K, false) \
    } else { \
        GEN_LAUNCH_BF16(_BLK_N, _BLK_K, true) \
    }

void launch_moe_gemv(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView topk_weights,
    ffi::TensorView topk_ids,
    bool mul_routed_weight,
    int64_t topk,
    bool use_swigelu,
    int64_t config_id) {
    TORCH_CHECK(A.ndim() == 2, "A must be dim 2.")
    TORCH_CHECK(B.ndim() == 3, "B must be dim 3.")
    TORCH_CHECK(C.ndim() == 3, "C must be dim 3.")
    TVM_FFI_ICHECK_EQ(A.device().device_id, B.device().device_id);
    TVM_FFI_ICHECK_EQ(A.device().device_id, C.device().device_id);
    TVM_FFI_ICHECK_EQ(A.device().device_id, topk_weights.device().device_id);
    TVM_FFI_ICHECK_EQ(A.device().device_id, topk_ids.device().device_id);
    TVM_FFI_ICHECK(dtype_equal(topk_ids.dtype(), dl_int32));

    const int32_t bseqlen = A.size(0);
    const int32_t hidden_size = A.size(1);
    const int32_t num_experts = B.size(0);
    const int32_t reduce_size = B.size(1);
    const int32_t expert_offset_stride = reduce_size * hidden_size;
    const int32_t half_n_idx = reduce_size / 2;
    const int scale_k_len = 1;
    const float eps = 1e-6;
    const int nr_n = use_swigelu ? reduce_size / 2 : reduce_size;

    ffi::MUSADeviceGuard device_guard(A.device().device_id);
    musaStream_t stream = get_stream(A.device());

    void *topk_ids_ptr = topk_ids.data_ptr();
    void *topk_weights_ptr = topk_weights.data_ptr();
    std::function<void()> launch_kernel;

    // The static kernel is the latency winner for the single-token decode
    // shape. Keep wider-M use opt-in until a shape-specific matrix justifies
    // it; SGLANG_MUSA_MOE_BF16_STATIC_K=0 disables the default M=1 path and
    // =1 enables the wider experimental variants as well.
    const char* static_bf16_env = std::getenv("SGLANG_MUSA_MOE_BF16_STATIC_K");
    constexpr int64_t kAutoConfig = -1;
    constexpr int64_t kStaticConfig = 100;
    constexpr int64_t kDualConfig = 101;
    bool use_static_bf16 = config_id == kStaticConfig ||
        (config_id == kAutoConfig && bseqlen == 1);
    if (config_id == kAutoConfig && static_bf16_env != nullptr) {
        use_static_bf16 = std::atoi(static_bf16_env) != 0;
    }
    if (use_static_bf16 && dtype_equal(A.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16) &&
        dtype_equal(topk_weights.dtype(), dl_float32) && use_swigelu &&
        (topk == 8 || topk == 9 || topk == 10 || topk == 11) &&
        (hidden_size == 2048 || hidden_size == 3072 || hidden_size == 4096) &&
        reduce_size % 2 == 0) {
#define LAUNCH_BF16_STATIC(_BN, _BK, _K, _TOPK) \
        do { \
            dim3 block_size{_BN * _BK, 1, 1}; \
            dim3 grid_size{ \
                static_cast<uint32_t>(ceil_div(nr_n, _BN / 2) * _TOPK), \
                static_cast<uint32_t>(bseqlen), 1}; \
            if (mul_routed_weight) { \
                musa_moe_gemv_bf16_swiglu_static_kernel<_BN, _BK, _K, _TOPK, true> \
                    <<<grid_size, block_size, 0, stream>>>( \
                        static_cast<bfloat16_t*>(C.data_ptr()), \
                        static_cast<const bfloat16_t*>(A.data_ptr()), \
                        static_cast<const bfloat16_t*>(B.data_ptr()), \
                        static_cast<const int*>(topk_ids_ptr), \
                        static_cast<const float*>(topk_weights_ptr), num_experts, reduce_size); \
            } else { \
                musa_moe_gemv_bf16_swiglu_static_kernel<_BN, _BK, _K, _TOPK, false> \
                    <<<grid_size, block_size, 0, stream>>>( \
                        static_cast<bfloat16_t*>(C.data_ptr()), \
                        static_cast<const bfloat16_t*>(A.data_ptr()), \
                        static_cast<const bfloat16_t*>(B.data_ptr()), \
                        static_cast<const int*>(topk_ids_ptr), \
                        static_cast<const float*>(topk_weights_ptr), num_experts, reduce_size); \
            } \
        } while (0)
#define DISPATCH_BF16_STATIC_K(_BN, _BK, _TOPK) \
        do { \
            if (hidden_size == 2048) LAUNCH_BF16_STATIC(_BN, _BK, 2048, _TOPK); \
            else if (hidden_size == 3072) LAUNCH_BF16_STATIC(_BN, _BK, 3072, _TOPK); \
            else LAUNCH_BF16_STATIC(_BN, _BK, 4096, _TOPK); \
        } while (0)
        if (bseqlen <= 4) {
            if (topk == 8) DISPATCH_BF16_STATIC_K(16, 8, 8);
            else if (topk == 9) DISPATCH_BF16_STATIC_K(16, 8, 9);
            else if (topk == 10) DISPATCH_BF16_STATIC_K(16, 8, 10);
            else DISPATCH_BF16_STATIC_K(16, 8, 11);
        } else {
            if (topk == 8) DISPATCH_BF16_STATIC_K(32, 4, 8);
            else if (topk == 9) DISPATCH_BF16_STATIC_K(32, 4, 9);
            else if (topk == 10) DISPATCH_BF16_STATIC_K(32, 4, 10);
            else DISPATCH_BF16_STATIC_K(32, 4, 11);
        }
#undef DISPATCH_BF16_STATIC_K
#undef LAUNCH_BF16_STATIC
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA static BF16 MoE GEMV kernel failed: " << musaGetErrorString(err);
        return;
    }

#define LAUNCH_BF16_DUAL(_BN, _BK, _STATIC_K, _TOPK, _CHECK_N, _MUL) \
    musa_moe_gemv_bf16_swiglu_dual_kernel< \
        _BN, _BK, _STATIC_K, _TOPK, _MUL, _CHECK_N> \
        <<<dim3{static_cast<uint32_t>(ceil_div(nr_n, _BN) * _TOPK), \
                 static_cast<uint32_t>(bseqlen), 1}, \
           dim3{_BN * _BK, 1, 1}, 0, stream>>>( \
            static_cast<bfloat16_t*>(C.data_ptr()), \
            static_cast<const bfloat16_t*>(A.data_ptr()), \
            static_cast<const bfloat16_t*>(B.data_ptr()), \
            static_cast<const int*>(topk_ids_ptr), \
            static_cast<const float*>(topk_weights_ptr), num_experts, \
            reduce_size, hidden_size)

#define DISPATCH_BF16_DUAL(_BN, _BK, _STATIC_K, _TOPK, _CHECK_N) \
    do { \
        if (mul_routed_weight) { \
            LAUNCH_BF16_DUAL(_BN, _BK, _STATIC_K, _TOPK, _CHECK_N, true); \
        } else { \
            LAUNCH_BF16_DUAL(_BN, _BK, _STATIC_K, _TOPK, _CHECK_N, false); \
        } \
    } while (0)

#define SELECT_BF16_DUAL(_BN, _BK, _TOPK) \
    do { \
        if (nr_n % _BN == 0) { \
            if (hidden_size == 2048) { \
                DISPATCH_BF16_DUAL(_BN, _BK, 2048, _TOPK, false); \
            } else { \
                DISPATCH_BF16_DUAL(_BN, _BK, 0, _TOPK, false); \
            } \
        } else { \
            if (hidden_size == 2048) { \
                DISPATCH_BF16_DUAL(_BN, _BK, 2048, _TOPK, true); \
            } else { \
                DISPATCH_BF16_DUAL(_BN, _BK, 0, _TOPK, true); \
            } \
        } \
    } while (0)

    // The generic BF16 MoE kernel has two CTA barriers for fused SwiGLU.
    // Use the independent route-specialized dual-reduction path for the
    // multi-token decode range; M=1 retains the existing static latency path.
    // Auto dispatch keeps the conservative small-N boundary. The explicit
    // structural config must remain measurable for every legal exact shape;
    // the offline tuner and full-pipeline guard decide whether it is promoted.
    const bool request_bf16_dual = config_id == kDualConfig ||
        (config_id == kAutoConfig && nr_n <= 256);
    const bool use_bf16_dual =
        request_bf16_dual && bseqlen >= 2 && bseqlen <= 32 &&
        dtype_equal(A.dtype(), dl_bfloat16) &&
        dtype_equal(B.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16) &&
        dtype_equal(topk_weights.dtype(), dl_float32) && use_swigelu &&
        (topk == 8 || topk == 9 || topk == 10 || topk == 11) &&
        reduce_size % 2 == 0;
    if (use_bf16_dual) {
        if (bseqlen <= 4) {
            if (topk == 8) SELECT_BF16_DUAL(16, 8, 8);
            else if (topk == 9) SELECT_BF16_DUAL(16, 8, 9);
            else if (topk == 10) SELECT_BF16_DUAL(16, 8, 10);
            else SELECT_BF16_DUAL(16, 8, 11);
        } else {
            if (topk == 8) SELECT_BF16_DUAL(8, 16, 8);
            else if (topk == 9) SELECT_BF16_DUAL(8, 16, 9);
            else if (topk == 10) SELECT_BF16_DUAL(8, 16, 10);
            else SELECT_BF16_DUAL(8, 16, 11);
        }
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA dual BF16 MoE GEMV kernel failed: "
            << musaGetErrorString(err);
        return;
    }
#undef SELECT_BF16_DUAL
#undef DISPATCH_BF16_DUAL
#undef LAUNCH_BF16_DUAL

MoeGemvBlockConfig configs[] = {
        {8, 16, 0.f, false},
        {16, 8, 0.f, false},
        {32, 4, 0.f, false},
        {4, 32, 0.f, false},
        {16, 4, 0.f, false},
    };

    constexpr int iobit = 128;
    const int bits_of_byte = 8;
    const int vlen = iobit / (tensor_element_size(B.dtype()) * bits_of_byte);
    float target_ratio = static_cast<float>(reduce_size) / hidden_size;
    // Multi-token decode benefits from wider N tiles on tiny TP-local
    // projections. Keep the latency-oriented choice for M<=4, then increase
    // N parallelism as the token dimension supplies enough independent CTAs.
    if (bseqlen >= 8 && reduce_size <= 512) {
        target_ratio = fmaxf(target_ratio, 8.0f);
    }

    for (auto& config : configs) {
        int load_size = config.block_k * vlen;
        config.valid = (hidden_size % load_size == 0);
        if (config.valid) {
            float block_ratio = static_cast<float>(config.block_n) / config.block_k;
            config.score = 1.0f / (1.0f + fabsf(block_ratio - target_ratio));
        }
    }

    // CSV-derived TP/EP shapes include narrow intermediates such as 88 and
    // 176.  They are 128-bit-load aligned but are not divisible by any of the
    // wider BLOCK_K candidates above.  Keep BLOCK_K=1 as an automatic-only
    // safety fallback; explicit config ids remain stable and the tuner can
    // still select a faster wider layout whenever one is legal.
    MoeGemvBlockConfig best_config_storage = {
        32, 1, -1.0f, hidden_size % vlen == 0};
    MoeGemvBlockConfig* best_config = &best_config_storage;
    constexpr int kNumGenericConfigs = sizeof(configs) / sizeof(configs[0]);
    if (config_id >= 0 && config_id < kNumGenericConfigs) {
        TORCH_CHECK(configs[config_id].valid,
                    "Requested BF16 MoE GEMV config is invalid for this shape");
        best_config = &configs[config_id];
    } else {
        TORCH_CHECK(config_id == kAutoConfig,
                    "Unsupported BF16 MoE GEMV config_id");
        for (auto& config : configs) {
            if (config.valid && (nr_n % config.block_n == 0) &&
                config.score > best_config->score) {
                best_config = &config;
            }
        }
        for (auto& config : configs) {
            if (!best_config->valid && config.valid &&
                config.score > best_config->score) {
                best_config = &config;
            }
        }
    }

    TORCH_CHECK(best_config->valid, "Unsupported MoE BF16 GEMV block configuration");

    switch (best_config->block_n) {
        case 4:
            switch (best_config->block_k) {
                case 32: SELECT_LAUNCH_BF16(4, 32); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=4");
            }
            break;
        case 8:
            switch (best_config->block_k) {
                case 16: SELECT_LAUNCH_BF16(8, 16); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=8");
            }
            break;
        case 16:
            switch (best_config->block_k) {
                case 8: SELECT_LAUNCH_BF16(16, 8); break;
                case 4: SELECT_LAUNCH_BF16(16, 4); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=16");
            }
            break;
        case 32:
            switch (best_config->block_k) {
                case 4: SELECT_LAUNCH_BF16(32, 4); break;
                case 1: SELECT_LAUNCH_BF16(32, 1); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=32");
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported block configuration");
    }

    launch_kernel();

    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA MoE BF16 GEMV kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_moe_gemv(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView A_scale,
    ffi::TensorView B_scale,
    ffi::TensorView topk_weights,
    ffi::TensorView topk_ids,
    bool has_a_scale,
    bool has_b_scale,
    bool mul_routed_weight,
    int64_t topk,
    bool use_w4a16,
    bool use_swigelu,
    int64_t config_id) {
    TORCH_CHECK(!has_a_scale, "bf16 moe gemv does not use A_scale");
    TORCH_CHECK(!has_b_scale, "bf16 moe gemv does not use B_scale");
    TORCH_CHECK(!use_w4a16, "default MoE GEMV does not use the W4A16 path");
    launch_moe_gemv(A, B, C, topk_weights, topk_ids, mul_routed_weight,
                         topk, use_swigelu, config_id);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_moe_gemv, sgl_musa_moe_gemv);
