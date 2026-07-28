#include "moe_gemv_kernel.muh"

// One subwarp owns both gate and up rows for an output channel.  The previous
// layout assigned the two rows to different subwarps, which loaded the same A
// vector twice and then paired the reductions through shared memory.  Keeping
// both accumulators together reuses A and leaves both reduced values in the
// same lane, so the common block-128 SwiGLU path needs no CTA barrier.
template <int kBlockN, int kBlockK, int kK, int kTopK, bool kMulRoutedWeight>
__global__ void musa_moe_gemv_fp8_block128_swiglu_static_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const __mt_fp8_e4m3* __restrict__ b_ptr,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const float* __restrict__ b_scale,
    int nr_experts,
    int full_n,
    int scale_n_len,
    int scale_k_len) {
    constexpr int kVec = 16;
    constexpr int kKStep = kBlockK * kVec;

    const int route_idx = blockIdx.x % kTopK;
    const int n_block_idx = blockIdx.x / kTopK;
    const int token_idx = blockIdx.y;
    const int t_n_idx = threadIdx.x / kBlockK;
    const int t_k_idx = threadIdx.x % kBlockK;
    const int out_n_idx = n_block_idx * kBlockN + t_n_idx;
    const int up_n_idx = out_n_idx + full_n / 2;
    const int expert_idx = topk_ids[token_idx * kTopK + route_idx];
    const size_t expert_base =
        static_cast<size_t>(expert_idx) * full_n * kK;
    const int token_base = token_idx * kK;

    if (expert_idx < 0 || expert_idx >= nr_experts) {
        if (t_k_idx == 0) {
            const int dst_n_idx = n_block_idx * kBlockN + t_n_idx;
            if (dst_n_idx < full_n / 2) {
                c_ptr[(token_idx * kTopK + route_idx) * (full_n / 2) + dst_n_idx] = 0;
            }
        }
        return;
    }

    float gate_partial[4] = {0.f, 0.f, 0.f, 0.f};
    float up_partial[4] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll 1
    for (int k_base = 0; k_base < kK; k_base += kKStep) {
        bfloat16_t a_reg[kVec];
        __mt_fp8_e4m3 gate_reg[kVec];
        __mt_fp8_e4m3 up_reg[kVec];
        using ALoad = typename VecType<bfloat16_t, 128>::Ttype;
        *reinterpret_cast<ALoad*>(a_reg) = *reinterpret_cast<const ALoad*>(
            a_ptr + token_base + k_base + t_k_idx * kVec);
        *reinterpret_cast<ALoad*>(a_reg + 8) = *reinterpret_cast<const ALoad*>(
            a_ptr + token_base + k_base + t_k_idx * kVec + 8);
        if (out_n_idx < full_n / 2) {
            *reinterpret_cast<int4*>(gate_reg) = __lsu_ld_cache_hint(
                reinterpret_cast<const int4*>(
                    b_ptr + expert_base + out_n_idx * kK + k_base + t_k_idx * kVec),
                0, 2, 1, 1);
            *reinterpret_cast<int4*>(up_reg) = __lsu_ld_cache_hint(
                reinterpret_cast<const int4*>(
                    b_ptr + expert_base + up_n_idx * kK + k_base + t_k_idx * kVec),
                0, 2, 1, 1);
        }

        const int scale_k_idx = k_base / 128 + t_k_idx / 8;
        const float gate_scale = out_n_idx < full_n / 2
            ? b_scale[(expert_idx * scale_n_len + out_n_idx / 128) *
                      scale_k_len + scale_k_idx]
            : 0.f;
        const float up_scale = out_n_idx < full_n / 2
            ? b_scale[(expert_idx * scale_n_len + up_n_idx / 128) *
                      scale_k_len + scale_k_idx]
            : 0.f;

        using Half4 = _Float16 __attribute__((ext_vector_type(4)));
        using Fp8x4 = unsigned char __attribute__((vector_size(4)));
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const Half4 gate4 = __musa_e4m32f16_rn_bst4(
                reinterpret_cast<const Fp8x4*>(gate_reg)[i]);
            const Half4 up4 = __musa_e4m32f16_rn_bst4(
                reinterpret_cast<const Fp8x4*>(up_reg)[i]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float a = float(a_reg[i * 4 + j]);
                gate_partial[i] += gate_scale * a * gate4[j];
                up_partial[i] += up_scale * a * up4[j];
            }
        }
    }

    float gate = gate_partial[0] + gate_partial[1] +
        gate_partial[2] + gate_partial[3];
    float up = up_partial[0] + up_partial[1] +
        up_partial[2] + up_partial[3];
    static_assert(kBlockK <= 32 && (kBlockK & (kBlockK - 1)) == 0);
    #pragma unroll
    for (int offset = kBlockK / 2; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffff, gate, offset, kBlockK);
        up += __shfl_down_sync(0xffffffff, up, offset, kBlockK);
    }

    if (t_k_idx == 0) {
        if constexpr (kMulRoutedWeight) {
            const float routed_weight =
                topk_weights[token_idx * kTopK + route_idx];
            gate *= routed_weight;
            up *= routed_weight;
        }
        if (out_n_idx < full_n / 2) {
            c_ptr[(token_idx * kTopK + route_idx) * (full_n / 2) + out_n_idx] =
                gate * sigmoid(gate) * up;
        }
    }
}

#define CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, _ROUTE_INTERLEAVED) \
    if (scale_k_group_tile == 128) { \
        musa_moe_gemv_generic_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, float, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, false, true, true, 128, false, check_bounds, true, _ROUTE_INTERLEAVED> \
            <<<grid_size, block_size, shmem_size, stream>>>( \
                static_cast<_CDTYPE*>(C.data_ptr()), \
                static_cast<_ADTYPE*>(A.data_ptr()), \
                static_cast<_BDTYPE*>(B.data_ptr()), \
                static_cast<int*>(topk_ids_ptr), \
                static_cast<float*>(topk_weights_ptr), \
                static_cast<float*>(a_scale_ptr), \
                static_cast<float*>(b_scale_ptr), \
                topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                nullptr, nullptr, nullptr, eps); \
    } else if (scale_k_group_tile == 64) { \
        musa_moe_gemv_generic_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, float, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, false, true, true, 64, false, check_bounds, true, _ROUTE_INTERLEAVED> \
            <<<grid_size, block_size, shmem_size, stream>>>( \
                static_cast<_CDTYPE*>(C.data_ptr()), \
                static_cast<_ADTYPE*>(A.data_ptr()), \
                static_cast<_BDTYPE*>(B.data_ptr()), \
                static_cast<int*>(topk_ids_ptr), \
                static_cast<float*>(topk_weights_ptr), \
                static_cast<float*>(a_scale_ptr), \
                static_cast<float*>(b_scale_ptr), \
                topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                nullptr, nullptr, nullptr, eps); \
    } else { \
        musa_moe_gemv_generic_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, float, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, false, false, true, 1, false, check_bounds, true, _ROUTE_INTERLEAVED> \
            <<<grid_size, block_size, shmem_size, stream>>>( \
                static_cast<_CDTYPE*>(C.data_ptr()), \
                static_cast<_ADTYPE*>(A.data_ptr()), \
                static_cast<_BDTYPE*>(B.data_ptr()), \
                static_cast<int*>(topk_ids_ptr), \
                static_cast<float*>(topk_weights_ptr), \
                static_cast<float*>(a_scale_ptr), \
                static_cast<float*>(b_scale_ptr), \
                topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                nullptr, nullptr, nullptr, eps); \
    } \
    return;

#define RUN_FP8_ROUTE(_ADTYPE, _BDTYPE, _CDTYPE, _ROUTE_INTERLEAVED) \
    if (mul_routed_weight) { \
        if (use_swigelu) { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, true, true, _ROUTE_INTERLEAVED) \
        } else { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, true, false, _ROUTE_INTERLEAVED) \
        } \
    } else { \
        if (use_swigelu) { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, false, true, _ROUTE_INTERLEAVED) \
        } else { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, false, false, _ROUTE_INTERLEAVED) \
        } \
    }

#define GEN_LAUNCH_FP8(_BLK_N, _BLK_K, _CHECK_BOUNDS) \
    { \
        launch_kernel = [&]() { \
            constexpr int block_n = _BLK_N; \
            constexpr int block_k = _BLK_K; \
            constexpr bool check_bounds = _CHECK_BOUNDS; \
            TORCH_CHECK(hidden_size % block_k == 0, "gemv k need align"); \
            dim3 block_size{block_n * block_k, 1, 1}; \
            int shmem_size = block_n * sizeof(float) * block_k; \
            if (topk == 8) { \
                dim3 grid_size{(uint32_t)(ceil_div(reduce_size, block_n) * 8), (uint32_t)bseqlen, 1}; \
                if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                    RUN_FP8_ROUTE(bfloat16_t, __mt_fp8_e4m3, bfloat16_t, true) \
                } else { \
                    RUN_FP8_ROUTE(__mt_fp8_e4m3, __mt_fp8_e4m3, bfloat16_t, true) \
                } \
            } else { \
                dim3 grid_size{(uint32_t)ceil_div(reduce_size, block_n), (uint32_t)bseqlen, (uint32_t)topk}; \
                if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                    RUN_FP8_ROUTE(bfloat16_t, __mt_fp8_e4m3, bfloat16_t, false) \
                } else { \
                    RUN_FP8_ROUTE(__mt_fp8_e4m3, __mt_fp8_e4m3, bfloat16_t, false) \
                } \
            } \
            TORCH_CHECK(false, "no support on fp8 moe gemv"); \
        }; \
    }

#define SELECT_LAUNCH_FP8(_BLK_N, _BLK_K) \
    if (nr_n % _BLK_N == 0) { \
        GEN_LAUNCH_FP8(_BLK_N, _BLK_K, false) \
    } else { \
        GEN_LAUNCH_FP8(_BLK_N, _BLK_K, true) \
    }

void launch_moe_gemv_w8ax_block128(
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
    TORCH_CHECK(has_b_scale, "B_scale is required for fp8 moe gemv");

    const int32_t bseqlen = A.size(0);
    const int32_t hidden_size = A.size(1);
    const int32_t num_experts = B.size(0);
    const int32_t reduce_size = B.size(1);
    const int32_t expert_offset_stride = reduce_size * hidden_size;
    const int32_t half_n_idx = reduce_size / 2;
    const float eps = 1e-6;
    const int nr_n = use_swigelu ? reduce_size / 2 : reduce_size;
    int scale_k_len = B_scale.size(2);
    int scale_k_group_tile = 1;
    bool is_pergroup_scale =
        B_scale.ndim() == 3 && (B_scale.size(1) != 1 || B_scale.size(2) != 1);
    if (is_pergroup_scale) {
        scale_k_group_tile = ceil_div(hidden_size, scale_k_len);
        TORCH_CHECK(scale_k_group_tile == 128 || scale_k_group_tile == 64, "scale_k_group_tile only support 128 or 64");
    }

    ffi::MUSADeviceGuard device_guard(A.device().device_id);
    musaStream_t stream = get_stream(A.device());

    void *topk_ids_ptr = topk_ids.data_ptr();
    void *topk_weights_ptr = topk_weights.data_ptr();
    void *a_scale_ptr = has_a_scale ? A_scale.data_ptr() : nullptr;
    void *b_scale_ptr = B_scale.data_ptr();
    std::function<void()> launch_kernel;

    constexpr int64_t kAutoConfig = -1;
    constexpr int64_t kStaticConfig = 100;
    // Profitability filters belong only to auto dispatch. Keep the explicit
    // static family measurable for every legal shape so the offline exact-
    // shape tuner can promote or reject it without source edits.
    const bool auto_static_profitable =
        (mul_routed_weight || bseqlen >= 16) &&
        (hidden_size != 7168 || bseqlen >= 2) &&
        (bseqlen >= 16 || reduce_size <= 512 || reduce_size >= 1536);
    const bool request_static = config_id == kStaticConfig ||
        (config_id == kAutoConfig && auto_static_profitable);
    if (request_static &&
        dtype_equal(A.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16) &&
        !has_a_scale && use_swigelu &&
        (topk == 8 || topk == 9 || topk == 10 || topk == 11) &&
        scale_k_group_tile == 128 && reduce_size % 2 == 0 &&
        (hidden_size == 2048 || hidden_size == 3072 || hidden_size == 4096 ||
         hidden_size == 7168) &&
        bseqlen >= 1) {
        if (bseqlen <= 4) {
            const int small_block_n = bseqlen <= 2 ? 16 : 8;
            const int small_block_k = bseqlen <= 2 ? 8 : 16;
            dim3 block_size{static_cast<uint32_t>(small_block_n * small_block_k), 1, 1};
            dim3 grid_size{
                static_cast<uint32_t>(ceil_div(nr_n, small_block_n) * topk),
                static_cast<uint32_t>(bseqlen), 1};
#define LAUNCH_FP8_STATIC_K(_BN, _BK, _K, _TOPK) \
            do { \
            if (mul_routed_weight) { \
                musa_moe_gemv_fp8_block128_swiglu_static_kernel<_BN, _BK, _K, _TOPK, true> \
                    <<<grid_size, block_size, 0, stream>>>( \
                        static_cast<bfloat16_t*>(C.data_ptr()), \
                        static_cast<const bfloat16_t*>(A.data_ptr()), \
                        static_cast<const __mt_fp8_e4m3*>(B.data_ptr()), \
                        static_cast<const int*>(topk_ids_ptr), \
                        static_cast<const float*>(topk_weights_ptr), \
                        static_cast<const float*>(b_scale_ptr), num_experts, \
                        reduce_size, B_scale.size(1), scale_k_len); \
            } else { \
                musa_moe_gemv_fp8_block128_swiglu_static_kernel<_BN, _BK, _K, _TOPK, false> \
                    <<<grid_size, block_size, 0, stream>>>( \
                        static_cast<bfloat16_t*>(C.data_ptr()), \
                        static_cast<const bfloat16_t*>(A.data_ptr()), \
                        static_cast<const __mt_fp8_e4m3*>(B.data_ptr()), \
                        static_cast<const int*>(topk_ids_ptr), \
                        static_cast<const float*>(topk_weights_ptr), \
                        static_cast<const float*>(b_scale_ptr), num_experts, \
                        reduce_size, B_scale.size(1), scale_k_len); \
            } \
            } while (0)
#define DISPATCH_FP8_STATIC_K(_BN, _BK, _TOPK) \
            do { \
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(_BN, _BK, 2048, _TOPK); \
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(_BN, _BK, 3072, _TOPK); \
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(_BN, _BK, 4096, _TOPK); \
                else LAUNCH_FP8_STATIC_K(_BN, _BK, 7168, _TOPK); \
            } while (0)
            if (bseqlen <= 2) {
                if (topk == 8) DISPATCH_FP8_STATIC_K(16, 8, 8);
                else if (topk == 9) DISPATCH_FP8_STATIC_K(16, 8, 9);
                else if (topk == 10) DISPATCH_FP8_STATIC_K(16, 8, 10);
                else DISPATCH_FP8_STATIC_K(16, 8, 11);
            } else {
                if (topk == 8) DISPATCH_FP8_STATIC_K(8, 16, 8);
                else if (topk == 9) DISPATCH_FP8_STATIC_K(8, 16, 9);
                else if (topk == 10) DISPATCH_FP8_STATIC_K(8, 16, 10);
                else DISPATCH_FP8_STATIC_K(8, 16, 11);
            }
#undef DISPATCH_FP8_STATIC_K
        } else if (bseqlen == 8) {
            dim3 block_size{8 * 16, 1, 1};
            dim3 grid_size{
                static_cast<uint32_t>(ceil_div(nr_n, 8) * topk),
                static_cast<uint32_t>(bseqlen), 1};
            if (topk == 8) {
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(8, 16, 2048, 8);
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(8, 16, 3072, 8);
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(8, 16, 4096, 8);
                else LAUNCH_FP8_STATIC_K(8, 16, 7168, 8);
            } else if (topk == 9) {
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(8, 16, 2048, 9);
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(8, 16, 3072, 9);
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(8, 16, 4096, 9);
                else LAUNCH_FP8_STATIC_K(8, 16, 7168, 9);
            } else if (topk == 10) {
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(8, 16, 2048, 10);
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(8, 16, 3072, 10);
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(8, 16, 4096, 10);
                else LAUNCH_FP8_STATIC_K(8, 16, 7168, 10);
            } else {
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(8, 16, 2048, 11);
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(8, 16, 3072, 11);
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(8, 16, 4096, 11);
                else LAUNCH_FP8_STATIC_K(8, 16, 7168, 11);
            }
        } else {
            dim3 block_size{32 * 4, 1, 1};
            dim3 grid_size{
                static_cast<uint32_t>(ceil_div(nr_n, 32) * topk),
                static_cast<uint32_t>(bseqlen), 1};
            if (topk == 8) {
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(32, 4, 2048, 8);
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(32, 4, 3072, 8);
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(32, 4, 4096, 8);
                else LAUNCH_FP8_STATIC_K(32, 4, 7168, 8);
            } else if (topk == 9) {
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(32, 4, 2048, 9);
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(32, 4, 3072, 9);
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(32, 4, 4096, 9);
                else LAUNCH_FP8_STATIC_K(32, 4, 7168, 9);
            } else if (topk == 10) {
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(32, 4, 2048, 10);
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(32, 4, 3072, 10);
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(32, 4, 4096, 10);
                else LAUNCH_FP8_STATIC_K(32, 4, 7168, 10);
            } else {
                if (hidden_size == 2048) LAUNCH_FP8_STATIC_K(32, 4, 2048, 11);
                else if (hidden_size == 3072) LAUNCH_FP8_STATIC_K(32, 4, 3072, 11);
                else if (hidden_size == 4096) LAUNCH_FP8_STATIC_K(32, 4, 4096, 11);
                else LAUNCH_FP8_STATIC_K(32, 4, 7168, 11);
            }
#undef LAUNCH_FP8_STATIC_K
        }
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA MoE FP8 GEMV kernel failed: "
            << musaGetErrorString(err);
        return;
    }

    MoeGemvBlockConfig configs[] = {
        {8, 16, 0.f, false},
        {16, 8, 0.f, false},
        {32, 4, 0.f, false},
        {4, 32, 0.f, false},
    };

    constexpr int iobit = 128;
    const int bits_of_byte = 8;
    const int vlen = iobit / (tensor_element_size(B.dtype()) * bits_of_byte);
    float target_ratio = static_cast<float>(reduce_size) / hidden_size;
    // For decode batches, a TP-local projection with a small N is latency
    // limited by CTA count rather than arithmetic throughput. Prefer wider N
    // tiles once enough tokens provide independent CTAs; keep M<=4 on the
    // ratio-based choice because it is more sensitive to per-CTA overhead.
    if (reduce_size <= 512 &&
        ((topk >= 10 && bseqlen >= 8) || bseqlen >= 32)) {
        target_ratio = fmaxf(target_ratio, 8.0f);
    }
    for (auto& config : configs) {
        int load_size = config.block_k * vlen;
        config.valid = (hidden_size % load_size == 0) && (load_size % scale_k_group_tile == 0);
        if (config.valid) {
            float block_ratio = static_cast<float>(config.block_n) / config.block_k;
            config.score = 1.0f / (1.0f + fabsf(block_ratio - target_ratio));
        }
    }

    MoeGemvBlockConfig best_config_storage = {32, 1, -1.0f, false};
    MoeGemvBlockConfig* best_config = &best_config_storage;
    constexpr int kNumGenericConfigs = sizeof(configs) / sizeof(configs[0]);
    if (config_id >= 0 && config_id < kNumGenericConfigs) {
        TORCH_CHECK(configs[config_id].valid,
                    "Requested FP8 MoE GEMV config is invalid for this shape");
        best_config = &configs[config_id];
    } else {
        TORCH_CHECK(config_id == kAutoConfig,
                    "Unsupported FP8 MoE GEMV config_id");
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
    TORCH_CHECK(best_config->valid, "Unsupported MoE FP8 GEMV block configuration");

    switch (best_config->block_n) {
        case 4:
            switch (best_config->block_k) {
                case 32: SELECT_LAUNCH_FP8(4, 32); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=4");
            }
            break;
        case 8:
            switch (best_config->block_k) {
                case 16: SELECT_LAUNCH_FP8(8, 16); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=8");
            }
            break;
        case 16:
            switch (best_config->block_k) {
                case 8: SELECT_LAUNCH_FP8(16, 8); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=16");
            }
            break;
        case 32:
            switch (best_config->block_k) {
                case 4: SELECT_LAUNCH_FP8(32, 4); break;
                case 1: SELECT_LAUNCH_FP8(32, 1); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=32");
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported block configuration");
    }

    launch_kernel();

    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA MoE FP8 GEMV kernel failed: " << musaGetErrorString(err);
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
    TORCH_CHECK(!use_w4a16, "W8AX MoE GEMV does not use the W4A16 path");
    launch_moe_gemv_w8ax_block128(A, B, C, A_scale, B_scale, topk_weights, topk_ids,
                        has_a_scale, has_b_scale, mul_routed_weight, topk,
                        use_swigelu, config_id);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_moe_gemv, sgl_musa_moe_gemv);
