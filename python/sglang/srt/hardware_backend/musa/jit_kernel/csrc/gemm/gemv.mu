#include "gemv.muh"

#define RUN_LINEAR_GEMV_KERNEL(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, _IS_W4A16, _IS_PER_GROUP_SCALE, _IS_FP8) \
    if (use_swigelu) { \
        if constexpr (_IS_FP8) { \
            if (scale_k_group_tile == 128) { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, true, false, _IS_W4A16, true, true, 128, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } else if (scale_k_group_tile == 64) { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, true, false, _IS_W4A16, true, true, 64, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } else { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, true, false, _IS_W4A16, false, true, 1, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } \
        } else { \
            musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, true, false, _IS_W4A16, _IS_PER_GROUP_SCALE, false, 1, false, check_bounds> \
                <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, nullptr, static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
        } \
    } else if (use_silu) { \
        if constexpr (_IS_FP8) { \
            if (scale_k_group_tile == 128) { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, false, true, _IS_W4A16, true, true, 128, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } else if (scale_k_group_tile == 64) { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, false, true, _IS_W4A16, true, true, 64, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } else { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, false, true, _IS_W4A16, false, true, 1, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } \
        } else { \
            musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, false, true, _IS_W4A16, _IS_PER_GROUP_SCALE, false, 1, false, check_bounds> \
                <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, nullptr, static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
        } \
    } else { \
        if constexpr (_IS_FP8) { \
            if (scale_k_group_tile == 128) { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, false, false, _IS_W4A16, true, true, 128, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } else if (scale_k_group_tile == 64) { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, false, false, _IS_W4A16, true, true, 64, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } else { \
                musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, false, false, _IS_W4A16, false, true, 1, false, check_bounds> \
                    <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, static_cast<_SCALE_DTYPE*>(a_scale_ptr), static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
            } \
        } else { \
            musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, float, _SCALE_DTYPE, block_n, block_k, iobit, false, false, false, _IS_W4A16, _IS_PER_GROUP_SCALE, false, 1, false, check_bounds> \
                <<<grid_size, block_size, shmem_size, stream>>>(static_cast<_CDTYPE*>(C.data_ptr()), static_cast<_ADTYPE*>(A.data_ptr()), static_cast<_BDTYPE*>(B.data_ptr()), nullptr, nullptr, nullptr, static_cast<_SCALE_DTYPE*>(b_scale_ptr), topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, nullptr, nullptr, nullptr, eps); \
        } \
    } \
    return;

#define GEN_LAUNCH_KERN_GEMV(_BLK_N, _BLK_K, _CHECK_BOUNDS) \
    { \
        launch_kernel = [&]() { \
            constexpr int block_n = _BLK_N; \
            constexpr int block_k = _BLK_K; \
            constexpr bool check_bounds = _CHECK_BOUNDS; \
            TORCH_CHECK(hidden_size % block_k == 0, "gemv k need align"); \
            dim3 block_size{block_n * block_k, 1, 1}; \
            dim3 grid_size{(uint32_t)ceil_div(reduce_size, block_n), (uint32_t)topk, (uint32_t)bseqlen}; \
            int shmem_size = block_n * sizeof(float) * block_k; \
            if (use_int4_w4a16) { \
                if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                    if (is_pergroup_scale) { \
                        RUN_LINEAR_GEMV_KERNEL(bfloat16_t, int8_t, bfloat16_t, float, true, true, false) \
                    } else { \
                        RUN_LINEAR_GEMV_KERNEL(bfloat16_t, int8_t, bfloat16_t, float, true, false, false) \
                    } \
                } else if (dtype_equal(A.dtype(), dl_float16)) { \
                    if (is_pergroup_scale) { \
                        RUN_LINEAR_GEMV_KERNEL(float16_t, int8_t, float16_t, float, true, true, false) \
                    } else { \
                        RUN_LINEAR_GEMV_KERNEL(float16_t, int8_t, float16_t, float, true, false, false) \
                    } \
                } \
            } else if (is_fp8) { \
                if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                    RUN_LINEAR_GEMV_KERNEL(bfloat16_t, __mt_fp8_e4m3, bfloat16_t, float, false, false, true) \
                } else if (dtype_equal(A.dtype(), dl_float8_e4m3fn)) { \
                    RUN_LINEAR_GEMV_KERNEL(__mt_fp8_e4m3, __mt_fp8_e4m3, bfloat16_t, float, false, false, true) \
                } \
            } else { \
                if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                    RUN_LINEAR_GEMV_KERNEL(bfloat16_t, bfloat16_t, bfloat16_t, float, false, false, false) \
                } else if (dtype_equal(A.dtype(), dl_float16)) { \
                    RUN_LINEAR_GEMV_KERNEL(float16_t, float16_t, float16_t, float, false, false, false) \
                } \
            } \
            TORCH_CHECK(false, "no support on linear gemv"); \
        }; \
    }

#define SELECT_LAUNCH_KERN_GEMV(_BLK_N, _BLK_K) \
    if (nr_n % _BLK_N == 0) { \
        GEN_LAUNCH_KERN_GEMV(_BLK_N, _BLK_K, false) \
    } else { \
        GEN_LAUNCH_KERN_GEMV(_BLK_N, _BLK_K, true) \
    }

void launch_gemv(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView B_scale,
    bool has_b_scale,
    bool use_int4_w4a16,
    bool use_swigelu,
    bool use_silu) {

    TORCH_CHECK(!(use_swigelu && use_silu), "use_swigelu and use_silu cannot both be true");
    TORCH_CHECK(A.ndim() == 2, "A must be dim 2.")
    TORCH_CHECK(B.ndim() == 2, "B must be dim 2.")
    TORCH_CHECK(C.ndim() == 2, "C must be dim 2.")
    TVM_FFI_ICHECK_EQ(A.device().device_id, B.device().device_id);
    TVM_FFI_ICHECK_EQ(A.device().device_id, C.device().device_id);

    const int64_t topk = 1;
    const int32_t bseqlen = A.size(0);
    const int32_t hidden_size = A.size(1);
    const int32_t num_experts = 1;
    const int32_t reduce_size = B.size(0);
    const bool is_fp8 = dtype_equal(B.dtype(), dl_float8_e4m3fn);

    void *a_scale_ptr = nullptr;
    void *b_scale_ptr = nullptr;
    void *rms_gamma_ptr = nullptr;
    void *rms_sum_out_ptr = nullptr;
    void *rms_count_ptr = nullptr;
    const float eps = 1e-6;
    const bool use_rms_norm = false;

    int current_arch = 310;
    if (current_arch < 300 && is_fp8) {
        TORCH_CHECK(false, "gemv not support Float8_e4m3fn on MUSA arch ", current_arch);
    }

    ffi::MUSADeviceGuard device_guard(A.device().device_id);
    musaStream_t stream = get_stream(A.device());

    if (has_b_scale) {
        b_scale_ptr = B_scale.data_ptr();
    }

    int expert_offset_stride = reduce_size * hidden_size;
    int half_n_idx = reduce_size / 2;
    int scale_k_len = 1;
    int scale_k_group_tile = 1;

    bool is_pergroup_scale = false;
    if (use_int4_w4a16 || is_fp8) {
        TORCH_CHECK(has_b_scale, "B_scale is required for int4/fp8 gemv");
        scale_k_len = B_scale.ndim() == 1 ? 1 : B_scale.size(B_scale.ndim() - 1);
        is_pergroup_scale =
            B_scale.ndim() >= 2
            && (B_scale.size(B_scale.ndim() - 2) != 1 || B_scale.size(B_scale.ndim() - 1) != 1);
        if (is_pergroup_scale) {
            scale_k_group_tile = ceil_div(hidden_size, scale_k_len);
            TORCH_CHECK(scale_k_group_tile == 128 || scale_k_group_tile == 64, "scale_k_group_tile only support 128 or 64");
        }
    }

    int nr_n = use_swigelu ? reduce_size / 2 : reduce_size;
    std::function<void()> launch_kernel;

    GemvBlockConfig configs[] = {
        {8, 16, 0.f, false},
        {16, 8, 0.f, false},
        {32, 4, 0.f, false},
        {4, 32, 0.f, false},
    };

    constexpr int iobit = 128;
    const int bits_of_byte = 8;
    const int vlen = use_int4_w4a16 ? (iobit / 4) : (iobit / (tensor_element_size(B.dtype()) * bits_of_byte));
    float target_ratio = static_cast<float>(reduce_size) / hidden_size;

    for (auto& config : configs) {
        int load_size = config.block_k * vlen;
        config.valid = (hidden_size % load_size == 0) && (load_size % scale_k_group_tile == 0);
        if (config.valid) {
            float block_ratio = static_cast<float>(config.block_n) / config.block_k;
            config.score = 1.0f / (1.0f + fabsf(block_ratio - target_ratio));
        }
    }

    GemvBlockConfig best_config_storage;
    if (current_arch < 300) {
        best_config_storage = {128, 1, -1.0f, false};
    } else {
        best_config_storage = {32, 1, -1.0f, false};
    }
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
    TORCH_CHECK(best_config->valid, "Unsupported GEMV block configuration");

    switch (best_config->block_n) {
        case 4:
            switch (best_config->block_k) {
                case 32: SELECT_LAUNCH_KERN_GEMV(4, 32); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=4");
            }
            break;
        case 8:
            switch (best_config->block_k) {
                case 16: SELECT_LAUNCH_KERN_GEMV(8, 16); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=8");
            }
            break;
        case 16:
            switch (best_config->block_k) {
                case 8: SELECT_LAUNCH_KERN_GEMV(16, 8); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=16");
            }
            break;
        case 32:
            switch (best_config->block_k) {
                case 4: SELECT_LAUNCH_KERN_GEMV(32, 4); break;
                case 1: SELECT_LAUNCH_KERN_GEMV(32, 1); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=32");
            }
            break;
        case 128:
            switch (best_config->block_k) {
                case 1: SELECT_LAUNCH_KERN_GEMV(128, 1); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=128");
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported block configuration");
    }

    launch_kernel();

    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA fused GEMV kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_fused_gemv(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView B_scale,
    bool has_b_scale,
    bool use_int4_w4a16,
    bool use_swigelu,
    bool use_silu) {
    launch_gemv(
        A,
        B,
        C,
        B_scale,
        has_b_scale,
        use_int4_w4a16,
        use_swigelu,
        use_silu);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_gemv, sgl_musa_fused_gemv);
