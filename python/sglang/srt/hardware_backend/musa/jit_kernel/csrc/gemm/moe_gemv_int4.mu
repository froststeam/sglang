#include "gemv.muh"

#define CAL_MOE_GEMV_INT4(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU) \
    if (is_pergroup_scale) { \
        if (scale_k_group_tile == 128) { \
            musa_gemv_kernel<_ADTYPE, _BDTYPE, _ADTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, true, true, false, 128, false, check_bounds> \
                <<<grid_size, block_size, shmem_size, stream>>>( \
                    static_cast<_ADTYPE*>(C.data_ptr()), \
                    static_cast<_ADTYPE*>(A.data_ptr()), \
                    static_cast<_BDTYPE*>(B.data_ptr()), \
                    static_cast<int*>(topk_ids_ptr), \
                    static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
                    nullptr, \
                    static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
                    topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                    nullptr, nullptr, nullptr, eps); \
        } else { \
            musa_gemv_kernel<_ADTYPE, _BDTYPE, _ADTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, true, true, false, 64, false, check_bounds> \
                <<<grid_size, block_size, shmem_size, stream>>>( \
                    static_cast<_ADTYPE*>(C.data_ptr()), \
                    static_cast<_ADTYPE*>(A.data_ptr()), \
                    static_cast<_BDTYPE*>(B.data_ptr()), \
                    static_cast<int*>(topk_ids_ptr), \
                    static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
                    nullptr, \
                    static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
                    topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                    nullptr, nullptr, nullptr, eps); \
        } \
    } else { \
        musa_gemv_kernel<_ADTYPE, _BDTYPE, _ADTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, true, false, false, 1, false, check_bounds> \
            <<<grid_size, block_size, shmem_size, stream>>>( \
                static_cast<_ADTYPE*>(C.data_ptr()), \
                static_cast<_ADTYPE*>(A.data_ptr()), \
                static_cast<_BDTYPE*>(B.data_ptr()), \
                static_cast<int*>(topk_ids_ptr), \
                static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
                nullptr, \
                static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
                topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                nullptr, nullptr, nullptr, eps); \
    } \
    return;

#define RUN_INT4_ROUTE(_ADTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE) \
    if (mul_routed_weight) { \
        if (use_swigelu) { \
            CAL_MOE_GEMV_INT4(_ADTYPE, int8_t, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, true, true) \
        } else { \
            CAL_MOE_GEMV_INT4(_ADTYPE, int8_t, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, true, false) \
        } \
    } else { \
        if (use_swigelu) { \
            CAL_MOE_GEMV_INT4(_ADTYPE, int8_t, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, false, true) \
        } else { \
            CAL_MOE_GEMV_INT4(_ADTYPE, int8_t, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, false, false) \
        } \
    }

#define GEN_LAUNCH_INT4(_BLK_N, _BLK_K, _CHECK_BOUNDS) \
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
                    RUN_INT4_ROUTE(bfloat16_t, float, float) \
                } else if (dtype_equal(topk_weights.dtype(), dl_bfloat16)) { \
                    RUN_INT4_ROUTE(bfloat16_t, bfloat16_t, float) \
                } \
            } else if (dtype_equal(A.dtype(), dl_float16)) { \
                if (dtype_equal(topk_weights.dtype(), dl_float32)) { \
                    RUN_INT4_ROUTE(float16_t, float, float) \
                } else if (dtype_equal(topk_weights.dtype(), dl_float16)) { \
                    RUN_INT4_ROUTE(float16_t, float16_t, float) \
                } \
            } \
            TORCH_CHECK(false, "no support on int4 moe gemv"); \
        }; \
    }

#define SELECT_LAUNCH_INT4(_BLK_N, _BLK_K) \
    if (nr_n % _BLK_N == 0) { \
        GEN_LAUNCH_INT4(_BLK_N, _BLK_K, false) \
    } else { \
        GEN_LAUNCH_INT4(_BLK_N, _BLK_K, true) \
    }

void launch_moe_gemv_int4(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView B_scale,
    ffi::TensorView topk_weights,
    ffi::TensorView topk_ids,
    bool has_b_scale,
    bool mul_routed_weight,
    int64_t topk,
    bool use_swigelu) {
    TORCH_CHECK(A.ndim() == 2, "A must be dim 2.")
    TORCH_CHECK(B.ndim() == 3, "B must be dim 3.")
    TORCH_CHECK(C.ndim() == 3, "C must be dim 3.")
    TVM_FFI_ICHECK_EQ(A.device().device_id, B.device().device_id);
    TVM_FFI_ICHECK_EQ(A.device().device_id, C.device().device_id);
    TVM_FFI_ICHECK_EQ(A.device().device_id, topk_weights.device().device_id);
    TVM_FFI_ICHECK_EQ(A.device().device_id, topk_ids.device().device_id);
    TVM_FFI_ICHECK(dtype_equal(topk_ids.dtype(), dl_int32));
    TORCH_CHECK(has_b_scale, "B_scale is required for int4 moe gemv");

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
    void *b_scale_ptr = B_scale.data_ptr();
    std::function<void()> launch_kernel;

    GemvBlockConfig configs[] = {
        {8, 16, 0.f, false},
        {16, 8, 0.f, false},
        {32, 4, 0.f, false},
        {4, 32, 0.f, false},
        {16, 4, 0.f, false},
    };

    constexpr int iobit = 128;
    const int vlen = iobit / 4;
    float target_ratio = static_cast<float>(reduce_size) / hidden_size;

    for (auto& config : configs) {
        int load_size = config.block_k * vlen;
        config.valid = (hidden_size % load_size == 0) && (load_size % scale_k_group_tile == 0);
        if (config.valid) {
            float block_ratio = static_cast<float>(config.block_n) / config.block_k;
            config.score = 1.0f / (1.0f + fabsf(block_ratio - target_ratio));
        }
    }

    GemvBlockConfig best_config_storage = {32, 1, -1.0f, false};
    GemvBlockConfig* best_config = &best_config_storage;
    for (auto& config : configs) {
        if (config.valid && (nr_n % config.block_n == 0) && config.score > best_config->score) {
            best_config = &config;
        }
    }
    for (auto& config : configs) {
        if (!best_config->valid && config.valid && config.score > best_config->score) {
            best_config = &config;
        }
    }
    TORCH_CHECK(best_config->valid, "Unsupported MoE INT4 GEMV block configuration");

    switch (best_config->block_n) {
        case 4:
            switch (best_config->block_k) {
                case 32: SELECT_LAUNCH_INT4(4, 32); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=4");
            }
            break;
        case 8:
            switch (best_config->block_k) {
                case 16: SELECT_LAUNCH_INT4(8, 16); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=8");
            }
            break;
        case 16:
            switch (best_config->block_k) {
                case 8: SELECT_LAUNCH_INT4(16, 8); break;
                case 4: SELECT_LAUNCH_INT4(16, 4); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=16");
            }
            break;
        case 32:
            switch (best_config->block_k) {
                case 4: SELECT_LAUNCH_INT4(32, 4); break;
                case 1: SELECT_LAUNCH_INT4(32, 1); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=32");
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported block configuration");
    }

    launch_kernel();

    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA fused MoE INT4 GEMV kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_fused_moe_gemv(
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
    bool use_int4_w4a16,
    bool use_swigelu) {
    TORCH_CHECK(!has_a_scale, "int4 moe gemv does not use A_scale");
    TORCH_CHECK(use_int4_w4a16, "int4 moe gemv requires int4 path");
    launch_moe_gemv_int4(A, B, C, B_scale, topk_weights, topk_ids, has_b_scale, mul_routed_weight, topk, use_swigelu);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_moe_gemv, sgl_musa_fused_moe_gemv);
