#include "gemv_kernel.muh"
#include <mma.h>

// Shape-agnostic dense FP8 GEMV. This kernel intentionally excludes routing,
// expert lookup, routed weights, and normalization state. N, K, and token count
// remain runtime values; BLOCK_N/BLOCK_K are the autotuned launch configuration.
template <int BLOCK_N, int BLOCK_K, bool CHECK_BOUNDS>
__global__ void musa_gemv_fp8_block128_generic_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const __mt_fp8_e4m3* __restrict__ b_ptr,
    const float* __restrict__ b_scale,
    const bfloat16_t* __restrict__ bias_ptr,
    int n,
    int k,
    int scale_k_len,
    bool fuse_silu) {
    constexpr int VEC = 16;
    constexpr int K_STEP = BLOCK_K * VEC;
    const int token_idx = blockIdx.y;
    const int n_block_idx = blockIdx.x;
    const int t_n_idx = threadIdx.x / BLOCK_K;
    const int t_k_idx = threadIdx.x % BLOCK_K;
    const int n_idx = n_block_idx * BLOCK_N + t_n_idx;
    const bool valid_n = !CHECK_BOUNDS || n_idx < n;
    float partial[4] = {0.f, 0.f, 0.f, 0.f};

    if (valid_n) {
        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += K_STEP) {
            bfloat16_t a_reg[VEC];
            __mt_fp8_e4m3 b_reg[VEC];
            using ALoad = typename VecType<bfloat16_t, 128>::Ttype;
            const int k_offset = k_base + t_k_idx * VEC;
            if (k_offset + VEC <= k) {
                *reinterpret_cast<ALoad*>(a_reg) = *reinterpret_cast<const ALoad*>(
                    a_ptr + token_idx * k + k_offset);
                *reinterpret_cast<ALoad*>(a_reg + 8) = *reinterpret_cast<const ALoad*>(
                    a_ptr + token_idx * k + k_offset + 8);
                *reinterpret_cast<ALoad*>(b_reg) = *reinterpret_cast<const ALoad*>(
                    b_ptr + n_idx * k + k_offset);
            } else {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int idx = k_offset + i;
                    a_reg[i] = idx < k ? a_ptr[token_idx * k + idx] : bfloat16_t(0);
                    b_reg[i] = idx < k ? b_ptr[n_idx * k + idx] : __mt_fp8_e4m3(0);
                }
            }

            const float scale = b_scale[
                (n_idx / 128) * scale_k_len + k_offset / 128];
            using Half4 = _Float16 __attribute__((ext_vector_type(4)));
            using Fp8x4 = unsigned char __attribute__((vector_size(4)));
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const Half4 b4 = __musa_e4m32f16_rn_bst4(
                    reinterpret_cast<const Fp8x4*>(b_reg)[i]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    partial[i] += scale * float(a_reg[i * 4 + j]) * b4[j];
                }
            }
        }
    }

    float sum = partial[0] + partial[1] + partial[2] + partial[3];
    __shared__ float reduction[BLOCK_N * BLOCK_K];
    reduction[threadIdx.x] = sum;
    __SYNCTHREADS_LM;
    if (threadIdx.x < BLOCK_N) {
        sum = 0.f;
        #pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
            sum += reduction[threadIdx.x * BLOCK_K + i];
        }
        const int output_n = n_block_idx * BLOCK_N + threadIdx.x;
        if (!CHECK_BOUNDS || output_n < n) {
            if (bias_ptr != nullptr) sum += float(bias_ptr[output_n]);
            c_ptr[token_idx * n + output_n] =
                fuse_silu ? sum * sigmoid(sum) : sum;
        }
    }
}

// Reuse each FP8 weight vector across a small token tile for dense decode.
// Unlike MoE, every token addresses the same matrix, so this removes repeated
// B loads without any routing or expert-grouping overhead.
template <int BLOCK_N, int BLOCK_K, int TOKEN_TILE, bool CHECK_BOUNDS>
__global__ void musa_gemv_fp8_block128_multitoken_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const __mt_fp8_e4m3* __restrict__ b_ptr,
    const float* __restrict__ b_scale,
    const bfloat16_t* __restrict__ bias_ptr,
    int tokens,
    int n,
    int k,
    int scale_k_len,
    bool fuse_silu) {
    constexpr int VEC = 16;
    constexpr int K_STEP = BLOCK_K * VEC;
    const int token_base = blockIdx.y * TOKEN_TILE;
    const int n_block_idx = blockIdx.x;
    const int t_n_idx = threadIdx.x / BLOCK_K;
    const int t_k_idx = threadIdx.x % BLOCK_K;
    const int n_idx = n_block_idx * BLOCK_N + t_n_idx;
    const bool valid_n = !CHECK_BOUNDS || n_idx < n;
    float partial[TOKEN_TILE] = {};

    if (valid_n) {
        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += K_STEP) {
            __mt_fp8_e4m3 b_reg[VEC];
            const int k_offset = k_base + t_k_idx * VEC;
            if (k_offset + VEC <= k) {
                using WeightLoad = typename VecType<bfloat16_t, 128>::Ttype;
                *reinterpret_cast<WeightLoad*>(b_reg) =
                    *reinterpret_cast<const WeightLoad*>(
                        b_ptr + n_idx * k + k_offset);
            } else {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int idx = k_offset + i;
                    b_reg[i] = idx < k ? b_ptr[n_idx * k + idx] : __mt_fp8_e4m3(0);
                }
            }
            const float scale = b_scale[
                (n_idx / 128) * scale_k_len + k_offset / 128];
            using Half4 = _Float16 __attribute__((ext_vector_type(4)));
            using Fp8x4 = unsigned char __attribute__((vector_size(4)));
            Half4 b4[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                b4[i] = __musa_e4m32f16_rn_bst4(
                    reinterpret_cast<const Fp8x4*>(b_reg)[i]);
            }

            #pragma unroll
            for (int token_i = 0; token_i < TOKEN_TILE; ++token_i) {
                const int token_idx = token_base + token_i;
                if (token_idx < tokens) {
                    bfloat16_t a_reg[VEC];
                    using ALoad = typename VecType<bfloat16_t, 128>::Ttype;
                    if (k_offset + VEC <= k) {
                        *reinterpret_cast<ALoad*>(a_reg) =
                            *reinterpret_cast<const ALoad*>(
                                a_ptr + token_idx * k + k_offset);
                        *reinterpret_cast<ALoad*>(a_reg + 8) =
                            *reinterpret_cast<const ALoad*>(
                                a_ptr + token_idx * k + k_offset + 8);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            const int idx = k_offset + i;
                            a_reg[i] = idx < k
                                ? a_ptr[token_idx * k + idx] : bfloat16_t(0);
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            partial[token_i] +=
                                scale * float(a_reg[i * 4 + j]) * b4[i][j];
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int token_i = 0; token_i < TOKEN_TILE; ++token_i) {
        float sum = partial[token_i];
        #pragma unroll
        for (int offset = BLOCK_K / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset, BLOCK_K);
        }
        if (t_k_idx == 0 && valid_n) {
            const int token_idx = token_base + token_i;
            if (token_idx < tokens) {
                if (bias_ptr != nullptr) sum += float(bias_ptr[n_idx]);
                c_ptr[token_idx * n + n_idx] =
                    fuse_silu ? sum * sigmoid(sum) : sum;
            }
        }
    }
}

// Dense-only BF16 x block-FP8 SwiGLU. Unlike the generic fallback this common
// decode path has no int4, A-scale, routing, or runtime feature branches. Each
// BLOCK_K-wide subwarp accumulates the gate and up rows together.  This reuses
// each activation vector and keeps the paired reductions in one lane, avoiding
// the shared-memory round trip used by the generic fused path.
template <int BLOCK_N, int BLOCK_K, int STATIC_K, bool CHECK_N>
__global__ void musa_gemv_fp8_block128_swiglu_static_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const __mt_fp8_e4m3* __restrict__ b_ptr,
    const float* __restrict__ b_scale,
    int tokens,
    int full_n,
    int scale_k_len) {
    constexpr int VEC = 16;
    constexpr int K_STEP = BLOCK_K * VEC;
    static_assert(BLOCK_K <= 32 && (BLOCK_K & (BLOCK_K - 1)) == 0);

    const int token_idx = blockIdx.y;
    const int t_n_idx = threadIdx.x / BLOCK_K;
    const int t_k_idx = threadIdx.x % BLOCK_K;
    const int out_n_idx = blockIdx.x * BLOCK_N + t_n_idx;
    const int up_n_idx = out_n_idx + full_n / 2;
    const bool valid_n = !CHECK_N || out_n_idx < full_n / 2;
    float gate_partial[4] = {0.f, 0.f, 0.f, 0.f};
    float up_partial[4] = {0.f, 0.f, 0.f, 0.f};

    if (valid_n) {
        #pragma unroll 1
        for (int k_base = 0; k_base < STATIC_K; k_base += K_STEP) {
            bfloat16_t a_reg[VEC];
            __mt_fp8_e4m3 gate_reg[VEC];
            __mt_fp8_e4m3 up_reg[VEC];
            using ALoad = typename VecType<bfloat16_t, 128>::Ttype;
            const int k_offset = k_base + t_k_idx * VEC;
            *reinterpret_cast<ALoad*>(a_reg) = *reinterpret_cast<const ALoad*>(
                a_ptr + token_idx * STATIC_K + k_offset);
            *reinterpret_cast<ALoad*>(a_reg + 8) =
                *reinterpret_cast<const ALoad*>(
                    a_ptr + token_idx * STATIC_K + k_offset + 8);
            *reinterpret_cast<int4*>(gate_reg) = __lsu_ld_cache_hint(
                reinterpret_cast<const int4*>(
                    b_ptr + static_cast<size_t>(out_n_idx) * STATIC_K + k_offset),
                4);
            *reinterpret_cast<int4*>(up_reg) = __lsu_ld_cache_hint(
                reinterpret_cast<const int4*>(
                    b_ptr + static_cast<size_t>(up_n_idx) * STATIC_K + k_offset),
                4);

            const int scale_k_idx = k_base / 128 + t_k_idx / 8;
            const float gate_scale = b_scale[
                (out_n_idx / 128) * scale_k_len + scale_k_idx];
            const float up_scale = b_scale[
                (up_n_idx / 128) * scale_k_len + scale_k_idx];
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
    }

    float gate = gate_partial[0] + gate_partial[1] +
        gate_partial[2] + gate_partial[3];
    float up = up_partial[0] + up_partial[1] +
        up_partial[2] + up_partial[3];
    #pragma unroll
    for (int offset = BLOCK_K / 2; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffff, gate, offset, BLOCK_K);
        up += __shfl_down_sync(0xffffffff, up, offset, BLOCK_K);
    }

    if (t_k_idx == 0) {
        if (!CHECK_N || out_n_idx < full_n / 2) {
            c_ptr[token_idx * (full_n / 2) + out_n_idx] =
                gate * sigmoid(gate) * up;
        }
    }
}

// Batched fused GEMV: one subwarp owns a gate/up row pair and reuses both
// weight vectors across a small token tile.  This combines the two useful
// reuse directions for decode: B is shared by TOKEN_TILE input rows, while A
// is loaded once for the paired gate/up dot products.
template <int BLOCK_N, int BLOCK_K, int STATIC_K, int TOKEN_TILE, bool CHECK_N>
__global__ void musa_gemv_fp8_block128_swiglu_multitoken_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const __mt_fp8_e4m3* __restrict__ b_ptr,
    const float* __restrict__ b_scale,
    int tokens,
    int full_n,
    int scale_k_len) {
    constexpr int VEC = 16;
    constexpr int K_STEP = BLOCK_K * VEC;
    static_assert(BLOCK_K <= 32 && (BLOCK_K & (BLOCK_K - 1)) == 0);

    const int token_base = blockIdx.y * TOKEN_TILE;
    const int t_n_idx = threadIdx.x / BLOCK_K;
    const int t_k_idx = threadIdx.x % BLOCK_K;
    const int out_n_idx = blockIdx.x * BLOCK_N + t_n_idx;
    const int up_n_idx = out_n_idx + full_n / 2;
    const bool valid_n = !CHECK_N || out_n_idx < full_n / 2;
    float gate_partial[TOKEN_TILE][4] = {};
    float up_partial[TOKEN_TILE][4] = {};

    if (valid_n) {
        #pragma unroll 1
        for (int k_base = 0; k_base < STATIC_K; k_base += K_STEP) {
            __mt_fp8_e4m3 gate_reg[VEC];
            __mt_fp8_e4m3 up_reg[VEC];
            const int k_offset = k_base + t_k_idx * VEC;
            *reinterpret_cast<int4*>(gate_reg) = __lsu_ld_cache_hint(
                reinterpret_cast<const int4*>(
                    b_ptr + static_cast<size_t>(out_n_idx) * STATIC_K + k_offset),
                4);
            *reinterpret_cast<int4*>(up_reg) = __lsu_ld_cache_hint(
                reinterpret_cast<const int4*>(
                    b_ptr + static_cast<size_t>(up_n_idx) * STATIC_K + k_offset),
                4);
            const int scale_k_idx = k_base / 128 + t_k_idx / 8;
            const float gate_scale = b_scale[
                (out_n_idx / 128) * scale_k_len + scale_k_idx];
            const float up_scale = b_scale[
                (up_n_idx / 128) * scale_k_len + scale_k_idx];
            using Half4 = _Float16 __attribute__((ext_vector_type(4)));
            using Fp8x4 = unsigned char __attribute__((vector_size(4)));
            Half4 gate4[4];
            Half4 up4[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                gate4[i] = __musa_e4m32f16_rn_bst4(
                    reinterpret_cast<const Fp8x4*>(gate_reg)[i]);
                up4[i] = __musa_e4m32f16_rn_bst4(
                    reinterpret_cast<const Fp8x4*>(up_reg)[i]);
            }

            #pragma unroll
            for (int token_i = 0; token_i < TOKEN_TILE; ++token_i) {
                const int token_idx = token_base + token_i;
                if (token_idx < tokens) {
                    bfloat16_t a_reg[VEC];
                    using ALoad = typename VecType<bfloat16_t, 128>::Ttype;
                    *reinterpret_cast<ALoad*>(a_reg) =
                        *reinterpret_cast<const ALoad*>(
                            a_ptr + token_idx * STATIC_K + k_offset);
                    *reinterpret_cast<ALoad*>(a_reg + 8) =
                        *reinterpret_cast<const ALoad*>(
                            a_ptr + token_idx * STATIC_K + k_offset + 8);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float a = float(a_reg[i * 4 + j]);
                            gate_partial[token_i][i] +=
                                gate_scale * a * gate4[i][j];
                            up_partial[token_i][i] +=
                                up_scale * a * up4[i][j];
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int token_i = 0; token_i < TOKEN_TILE; ++token_i) {
        float gate = gate_partial[token_i][0] + gate_partial[token_i][1] +
            gate_partial[token_i][2] + gate_partial[token_i][3];
        float up = up_partial[token_i][0] + up_partial[token_i][1] +
            up_partial[token_i][2] + up_partial[token_i][3];
        #pragma unroll
        for (int offset = BLOCK_K / 2; offset > 0; offset >>= 1) {
            gate += __shfl_down_sync(0xffffffff, gate, offset, BLOCK_K);
            up += __shfl_down_sync(0xffffffff, up, offset, BLOCK_K);
        }
        if (t_k_idx == 0 && valid_n) {
            const int token_idx = token_base + token_i;
            if (token_idx < tokens) {
                c_ptr[token_idx * (full_n / 2) + out_n_idx] =
                    gate * sigmoid(gate) * up;
            }
        }
    }
}

// Dense BF16 batched SwiGLU GEMV.  Each subwarp owns an output channel and
// accumulates its gate/up rows together.  The two weight vectors stay in
// registers while TOKEN_TILE input rows are consumed, so the dense path gets
// both A reuse across gate/up and B reuse across tokens without routing state.
template <int BLOCK_N, int BLOCK_K, int TOKEN_TILE, bool CHECK_N>
__global__ void musa_gemv_bf16_swiglu_bgemv_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    int tokens,
    int full_n,
    int k) {
    constexpr int VEC = 8;
    constexpr int K_STEP = BLOCK_K * VEC;
    static_assert(BLOCK_K <= 32 && (BLOCK_K & (BLOCK_K - 1)) == 0);

    const int token_base = blockIdx.y * TOKEN_TILE;
    const int t_n_idx = threadIdx.x / BLOCK_K;
    const int t_k_idx = threadIdx.x % BLOCK_K;
    const int out_n_idx = blockIdx.x * BLOCK_N + t_n_idx;
    const int up_n_idx = out_n_idx + full_n / 2;
    const bool valid_n = !CHECK_N || out_n_idx < full_n / 2;
    float gate[TOKEN_TILE] = {};
    float up[TOKEN_TILE] = {};

    if (valid_n) {
        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += K_STEP) {
            const int k_offset = k_base + t_k_idx * VEC;
            bfloat16_t gate_reg[VEC];
            bfloat16_t up_reg[VEC];
            using Load = typename VecType<bfloat16_t, 128>::Ttype;
            if (k_offset + VEC <= k) {
                *reinterpret_cast<Load*>(gate_reg) =
                    *reinterpret_cast<const Load*>(
                        b_ptr + static_cast<size_t>(out_n_idx) * k + k_offset);
                *reinterpret_cast<Load*>(up_reg) =
                    *reinterpret_cast<const Load*>(
                        b_ptr + static_cast<size_t>(up_n_idx) * k + k_offset);
            } else {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int idx = k_offset + i;
                    gate_reg[i] = idx < k
                        ? b_ptr[static_cast<size_t>(out_n_idx) * k + idx]
                        : bfloat16_t(0);
                    up_reg[i] = idx < k
                        ? b_ptr[static_cast<size_t>(up_n_idx) * k + idx]
                        : bfloat16_t(0);
                }
            }

            #pragma unroll
            for (int token_i = 0; token_i < TOKEN_TILE; ++token_i) {
                const int token_idx = token_base + token_i;
                if (token_idx < tokens) {
                    bfloat16_t a_reg[VEC];
                    if (k_offset + VEC <= k) {
                        *reinterpret_cast<Load*>(a_reg) =
                            *reinterpret_cast<const Load*>(
                                a_ptr + token_idx * k + k_offset);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            const int idx = k_offset + i;
                            a_reg[i] = idx < k
                                ? a_ptr[token_idx * k + idx] : bfloat16_t(0);
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < VEC; ++i) {
                        const float a = float(a_reg[i]);
                        gate[token_i] += a * float(gate_reg[i]);
                        up[token_i] += a * float(up_reg[i]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int token_i = 0; token_i < TOKEN_TILE; ++token_i) {
        float gate_sum = gate[token_i];
        float up_sum = up[token_i];
        #pragma unroll
        for (int offset = BLOCK_K / 2; offset > 0; offset >>= 1) {
            gate_sum += __shfl_down_sync(
                0xffffffff, gate_sum, offset, BLOCK_K);
            up_sum += __shfl_down_sync(0xffffffff, up_sum, offset, BLOCK_K);
        }
        if (t_k_idx == 0 && valid_n) {
            const int token_idx = token_base + token_i;
            if (token_idx < tokens) {
                c_ptr[token_idx * (full_n / 2) + out_n_idx] =
                    gate_sum * sigmoid(gate_sum) * up_sum;
            }
        }
    }
}

template <int BLOCK_N, int BLOCK_K, bool CHECK_BOUNDS>
__global__ void musa_gemv_bf16_generic_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    const bfloat16_t* __restrict__ bias_ptr,
    int n,
    int k,
    bool fuse_silu) {
    constexpr int VEC = 8;
    constexpr int K_STEP = BLOCK_K * VEC;
    const int token_idx = blockIdx.y;
    const int n_block_idx = blockIdx.x;
    const int t_n_idx = threadIdx.x / BLOCK_K;
    const int t_k_idx = threadIdx.x % BLOCK_K;
    const int n_idx = n_block_idx * BLOCK_N + t_n_idx;
    const bool valid_n = !CHECK_BOUNDS || n_idx < n;
    float partial[VEC] = {0.f};

    if (valid_n) {
        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += K_STEP) {
            bfloat16_t a_reg[VEC];
            bfloat16_t b_reg[VEC];
            using Load = typename VecType<bfloat16_t, 128>::Ttype;
            const int k_offset = k_base + t_k_idx * VEC;
            if (k_offset + VEC <= k) {
                *reinterpret_cast<Load*>(a_reg) = *reinterpret_cast<const Load*>(
                    a_ptr + token_idx * k + k_offset);
                *reinterpret_cast<Load*>(b_reg) = *reinterpret_cast<const Load*>(
                    b_ptr + n_idx * k + k_offset);
            } else {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int idx = k_offset + i;
                    a_reg[i] = idx < k ? a_ptr[token_idx * k + idx] : bfloat16_t(0);
                    b_reg[i] = idx < k ? b_ptr[n_idx * k + idx] : bfloat16_t(0);
                }
            }
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                partial[i] += float(a_reg[i]) * float(b_reg[i]);
            }
        }
    }

    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        sum += partial[i];
    }
    __shared__ float reduction[BLOCK_N * BLOCK_K];
    reduction[threadIdx.x] = sum;
    __SYNCTHREADS_LM;
    if (threadIdx.x < BLOCK_N) {
        sum = 0.f;
        #pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
            sum += reduction[threadIdx.x * BLOCK_K + i];
        }
        const int output_n = n_block_idx * BLOCK_N + threadIdx.x;
        if (!CHECK_BOUNDS || output_n < n) {
            if (bias_ptr != nullptr) sum += float(bias_ptr[output_n]);
            c_ptr[token_idx * n + output_n] =
                fuse_silu ? sum * sigmoid(sum) : sum;
        }
    }
}

// Each MP31 warp computes a complete 16x16 output tile with tensor cores.
// This is intentionally shape-restricted: smaller decode batches stay on the
// bandwidth-oriented GEMV kernels, while M=16/32 has enough rows to amortize
// the WMMA fragment setup and reuse each weight tile across all input rows.
__global__ void musa_gemv_bf16_wmma_m16_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    const bfloat16_t* __restrict__ bias_ptr,
    int n,
    int k,
    bool fuse_silu) {
    using namespace mtmusa::wmma;
    fragment<matrix_a, 16, 16, 16, __mt_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __mt_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fill_fragment(c_frag, 0.f);

    constexpr int WMMA_WAVE_SIZE = 128;
    const int warp_id = threadIdx.x / WMMA_WAVE_SIZE;
    const int lane_id = threadIdx.x % WMMA_WAVE_SIZE;
    const int token_base = blockIdx.y * 16;
    const int n_base = (blockIdx.x * 2 + warp_id) * 16;
    #pragma unroll 1
    for (int k_base = 0; k_base < k; k_base += 16) {
        load_matrix_sync(a_frag, a_ptr + token_base * k + k_base, k);
        load_matrix_sync(b_frag, b_ptr + n_base * k + k_base, k);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ float output_tile[2][16 * 16];
    store_matrix_sync(output_tile[warp_id], c_frag, 16, mem_row_major);
    __SYNCTHREADS_LM;
    for (int idx = lane_id; idx < 16 * 16; idx += WMMA_WAVE_SIZE) {
        const int token = token_base + idx / 16;
        const int tile_n = idx % 16;
        float sum = output_tile[warp_id][idx];
        if (bias_ptr != nullptr) sum += float(bias_ptr[n_base + tile_n]);
        c_ptr[token * n + n_base + tile_n] =
            bfloat16_t(fuse_silu ? sum * sigmoid(sum) : sum);
    }
}

// Tensor-core SwiGLU for M=16/32. Each warp computes one 16-column output
// tile and accumulates its gate/up projections together, so activations are
// consumed once and no temporary 2N projection is materialized.
template <bool CHECK_M>
__global__ void musa_gemv_bf16_swiglu_wmma_m16_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    int output_n,
    int k,
    int valid_tokens) {
    using namespace mtmusa::wmma;
    fragment<matrix_a, 16, 16, 16, __mt_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __mt_bfloat16, col_major> gate_b_frag;
    fragment<matrix_b, 16, 16, 16, __mt_bfloat16, col_major> up_b_frag;
    fragment<accumulator, 16, 16, 16, float> gate_frag;
    fragment<accumulator, 16, 16, 16, float> up_frag;
    fill_fragment(gate_frag, 0.f);
    fill_fragment(up_frag, 0.f);

    constexpr int WMMA_WAVE_SIZE = 128;
    const int warp_id = threadIdx.x / WMMA_WAVE_SIZE;
    const int lane_id = threadIdx.x % WMMA_WAVE_SIZE;
    const int token_base = blockIdx.y * 16;
    const int n_base = (blockIdx.x * 2 + warp_id) * 16;
    __shared__ bfloat16_t a_tile[16 * 16];
    #pragma unroll 1
    for (int k_base = 0; k_base < k; k_base += 16) {
        if constexpr (CHECK_M) {
            const int row = threadIdx.x / 16;
            const int col = threadIdx.x % 16;
            a_tile[threadIdx.x] = row < valid_tokens
                ? a_ptr[(token_base + row) * k + k_base + col]
                : bfloat16_t(0);
            __SYNCTHREADS_LM;
            load_matrix_sync(a_frag, a_tile, 16);
        } else {
            load_matrix_sync(a_frag, a_ptr + token_base * k + k_base, k);
        }
        load_matrix_sync(gate_b_frag, b_ptr + n_base * k + k_base, k);
        load_matrix_sync(
            up_b_frag, b_ptr + (n_base + output_n) * k + k_base, k);
        mma_sync(gate_frag, a_frag, gate_b_frag, gate_frag);
        mma_sync(up_frag, a_frag, up_b_frag, up_frag);
        if constexpr (CHECK_M) {
            __SYNCTHREADS_LM;
        }
    }

    __shared__ float output_tile[2][2][16 * 16];
    store_matrix_sync(output_tile[warp_id][0], gate_frag, 16, mem_row_major);
    store_matrix_sync(output_tile[warp_id][1], up_frag, 16, mem_row_major);
    __SYNCTHREADS_LM;
    for (int idx = lane_id; idx < 16 * 16; idx += WMMA_WAVE_SIZE) {
        const int token = token_base + idx / 16;
        const int tile_n = idx % 16;
        const float gate = output_tile[warp_id][0][idx];
        const float up = output_tile[warp_id][1][idx];
        if (!CHECK_M || idx / 16 < valid_tokens) {
            c_ptr[token * output_n + n_base + tile_n] =
                bfloat16_t(gate * sigmoid(gate) * up);
        }
    }
}

// Multi-token dense GEMV. A CTA keeps one weight vector in registers and
// reuses it for TOKEN_TILE input rows, reducing weight traffic for decode
// batches without switching to the generic GEMM path.
template <int BLOCK_N, int BLOCK_K, int TOKEN_TILE, bool CHECK_N>
__global__ void musa_gemv_bf16_multitoken_kernel(
    bfloat16_t* __restrict__ c_ptr,
    const bfloat16_t* __restrict__ a_ptr,
    const bfloat16_t* __restrict__ b_ptr,
    const bfloat16_t* __restrict__ bias_ptr,
    int tokens,
    int n,
    int k,
    bool fuse_silu) {
    constexpr int VEC = 8;
    constexpr int K_STEP = BLOCK_K * VEC;
    const int token_base = blockIdx.y * TOKEN_TILE;
    const int t_n_idx = threadIdx.x / BLOCK_K;
    const int t_k_idx = threadIdx.x % BLOCK_K;
    const int n_idx = blockIdx.x * BLOCK_N + t_n_idx;
    const bool valid_n = !CHECK_N || n_idx < n;
    float partial[TOKEN_TILE][VEC] = {0.f};

    if (valid_n) {
        #pragma unroll 1
        for (int k_base = 0; k_base < k; k_base += K_STEP) {
            const int k_offset = k_base + t_k_idx * VEC;
            bfloat16_t b_reg[VEC];
            using Load = typename VecType<bfloat16_t, 128>::Ttype;
            if (k_offset + VEC <= k) {
                *reinterpret_cast<Load*>(b_reg) = *reinterpret_cast<const Load*>(
                    b_ptr + n_idx * k + k_offset);
            } else {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int idx = k_offset + i;
                    b_reg[i] = idx < k ? b_ptr[n_idx * k + idx] : bfloat16_t(0);
                }
            }
            #pragma unroll
            for (int tt = 0; tt < TOKEN_TILE; ++tt) {
                const int token = token_base + tt;
                if (token < tokens) {
                    bfloat16_t a_reg[VEC];
                    if (k_offset + VEC <= k) {
                        *reinterpret_cast<Load*>(a_reg) = *reinterpret_cast<const Load*>(
                            a_ptr + token * k + k_offset);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            const int idx = k_offset + i;
                            a_reg[i] = idx < k ? a_ptr[token * k + idx] : bfloat16_t(0);
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < VEC; ++i) {
                        partial[tt][i] += float(a_reg[i]) * float(b_reg[i]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int tt = 0; tt < TOKEN_TILE; ++tt) {
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) sum += partial[tt][i];
        #pragma unroll
        for (int offset = BLOCK_K / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset, BLOCK_K);
        }
        if (t_k_idx == 0) {
            const int token = token_base + tt;
            const int output_n = blockIdx.x * BLOCK_N + t_n_idx;
            if (token < tokens && (!CHECK_N || output_n < n)) {
                if (bias_ptr != nullptr) sum += float(bias_ptr[output_n]);
                c_ptr[token * n + output_n] =
                    fuse_silu ? sum * sigmoid(sum) : sum;
            }
        }
    }
}

#define LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                               _IS_SWIGLU, _IS_W4A16, _IS_PER_GROUP_SCALE, \
                               _IS_FP8, _SCALE_BLOCK) \
    musa_gemv_fallback_kernel< \
        _ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, block_n, block_k, iobit, \
        _IS_SWIGLU, _IS_W4A16, _IS_PER_GROUP_SCALE, _IS_FP8, _SCALE_BLOCK, \
        check_bounds> \
        <<<grid_size, block_size, shmem_size, stream>>>( \
            static_cast<_CDTYPE*>(C.data_ptr()), \
            static_cast<const _ADTYPE*>(A.data_ptr()), \
            static_cast<const _BDTYPE*>(B.data_ptr()), \
            static_cast<_SCALE_DTYPE*>(a_scale_ptr), \
            static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
            static_cast<const _CDTYPE*>(bias_ptr), \
            nr_n, hidden_size, half_n_idx, scale_k_len, fuse_silu); \

#define RUN_GEMV_KERNEL(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                               _IS_W4A16, _IS_PER_GROUP_SCALE, _IS_FP8) \
    if (fuse_swiglu) { \
        if constexpr (_IS_FP8) { \
            if (scale_k_group_tile == 128) { \
                LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                    true, _IS_W4A16, true, true, 128); \
            } else if (scale_k_group_tile == 64) { \
                LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                    true, _IS_W4A16, true, true, 64); \
            } else { \
                LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                    true, _IS_W4A16, false, true, 1); \
            } \
        } else { \
            LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                true, _IS_W4A16, _IS_PER_GROUP_SCALE, false, 1); \
        } \
    } else { \
        if constexpr (_IS_FP8) { \
            if (scale_k_group_tile == 128) { \
                LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                    false, _IS_W4A16, true, true, 128); \
            } else if (scale_k_group_tile == 64) { \
                LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                    false, _IS_W4A16, true, true, 64); \
            } else { \
                LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                    false, _IS_W4A16, false, true, 1); \
            } \
        } else { \
            LAUNCH_DENSE_FALLBACK(_ADTYPE, _BDTYPE, _CDTYPE, _SCALE_DTYPE, \
                false, _IS_W4A16, _IS_PER_GROUP_SCALE, false, 1); \
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
            dim3 grid_size{(uint32_t)ceil_div(reduce_size, block_n), (uint32_t)bseqlen, 1}; \
            int shmem_size = block_n * sizeof(float) * block_k; \
            if (use_int4_w4a16) { \
                if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                    if (is_pergroup_scale) { \
                        RUN_GEMV_KERNEL(bfloat16_t, int8_t, bfloat16_t, float, true, true, false) \
                    } else { \
                        RUN_GEMV_KERNEL(bfloat16_t, int8_t, bfloat16_t, float, true, false, false) \
                    } \
                } else if (dtype_equal(A.dtype(), dl_float16)) { \
                    if (is_pergroup_scale) { \
                        RUN_GEMV_KERNEL(float16_t, int8_t, float16_t, float, true, true, false) \
                    } else { \
                        RUN_GEMV_KERNEL(float16_t, int8_t, float16_t, float, true, false, false) \
                    } \
                } \
            } else if (is_fp8) { \
                if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                    RUN_GEMV_KERNEL(bfloat16_t, __mt_fp8_e4m3, bfloat16_t, float, false, false, true) \
                } else if (dtype_equal(A.dtype(), dl_float8_e4m3fn)) { \
                    RUN_GEMV_KERNEL(__mt_fp8_e4m3, __mt_fp8_e4m3, bfloat16_t, float, false, false, true) \
                } \
            } else { \
                if (dtype_equal(A.dtype(), dl_bfloat16)) { \
                    RUN_GEMV_KERNEL(bfloat16_t, bfloat16_t, bfloat16_t, float, false, false, false) \
                } else if (dtype_equal(A.dtype(), dl_float16)) { \
                    RUN_GEMV_KERNEL(float16_t, float16_t, float16_t, float, false, false, false) \
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

static int get_musa_arch(int device_id) {
    static thread_local int cached_device_id = -1;
    static thread_local int cached_arch = -1;
    if (cached_device_id != device_id) {
        struct musaDeviceProp device_prop{};
        const musaError_t err = musaGetDeviceProperties(&device_prop, device_id);
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "musaGetDeviceProperties failed: " << musaGetErrorString(err);
        cached_device_id = device_id;
        cached_arch = static_cast<int>(device_prop.major) * 100 +
                      static_cast<int>(device_prop.minor) * 10;
    }
    return cached_arch;
}

void launch_gemv(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView B_scale,
    ffi::TensorView Bias,
    bool has_b_scale,
    bool has_bias,
    bool use_int4_w4a16,
    bool fuse_swiglu,
    bool fuse_silu,
    int forced_config_id) {

    TORCH_CHECK(!(fuse_swiglu && fuse_silu),
                "fuse_swiglu and fuse_silu cannot both be enabled");
    TORCH_CHECK(A.ndim() == 2, "A must be dim 2.")
    TORCH_CHECK(B.ndim() == 2, "B must be dim 2.")
    TORCH_CHECK(C.ndim() == 2, "C must be dim 2.")
    TVM_FFI_ICHECK_EQ(A.device().device_id, B.device().device_id);
    TVM_FFI_ICHECK_EQ(A.device().device_id, C.device().device_id);

    const int32_t bseqlen = A.size(0);
    const int32_t hidden_size = A.size(1);
    const int32_t reduce_size = B.size(0);
    const bool is_fp8 = dtype_equal(B.dtype(), dl_float8_e4m3fn);

    TORCH_CHECK(B.size(1) == hidden_size, "B K must match A K");
    TORCH_CHECK(C.size(0) == bseqlen, "C M must match A M");
    TORCH_CHECK(C.size(1) == (fuse_swiglu ? reduce_size / 2 : reduce_size),
                "C N does not match GEMV output N");
    TORCH_CHECK(A.stride(1) == 1 && A.stride(0) == hidden_size,
                "A must be contiguous");
    TORCH_CHECK(B.stride(1) == 1 && B.stride(0) == hidden_size,
                "B must be contiguous");
    TORCH_CHECK(C.stride(1) == 1 && C.stride(0) == C.size(1),
                "C must be contiguous");
    TORCH_CHECK(use_int4_w4a16 || dtype_equal(A.dtype(), dl_bfloat16),
                "A must be BF16 for non-W4 GEMV");
    TORCH_CHECK(use_int4_w4a16 || dtype_equal(C.dtype(), dl_bfloat16),
                "C must be BF16 for non-W4 GEMV");
    TORCH_CHECK(is_fp8 || use_int4_w4a16 || dtype_equal(B.dtype(), dl_bfloat16),
                "B must be BF16, FP8 E4M3, or packed W4");
    TORCH_CHECK(!fuse_swiglu || !has_bias,
                "fused SwiGLU GEMV does not support bias");
    TORCH_CHECK(!fuse_swiglu || reduce_size % 2 == 0,
                "fused SwiGLU GEMV requires even B N");

    void *a_scale_ptr = nullptr;
    void *b_scale_ptr = nullptr;
    void *bias_ptr = nullptr;

    ffi::MUSADeviceGuard device_guard(A.device().device_id);
    const int current_arch = get_musa_arch(A.device().device_id);
    if (current_arch < 300 && is_fp8) {
        TORCH_CHECK(false, "gemv not support Float8_e4m3fn on MUSA arch ", current_arch);
    }

    musaStream_t stream = get_stream(A.device());

    if (has_b_scale) {
        TVM_FFI_ICHECK_EQ(A.device().device_id, B_scale.device().device_id);
        b_scale_ptr = B_scale.data_ptr();
    }
    if (has_bias) {
        TVM_FFI_ICHECK_EQ(A.device().device_id, Bias.device().device_id);
        TORCH_CHECK(Bias.ndim() == 1 && Bias.size(0) == reduce_size,
                    "GEMV bias must be a 1D tensor matching output N");
        TORCH_CHECK((use_int4_w4a16 || dtype_equal(Bias.dtype(), dl_bfloat16)) &&
                        Bias.stride(0) == 1,
                    "GEMV bias must be contiguous and match activation dtype");
        bias_ptr = Bias.data_ptr();
    }

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
            // Block-FP8 scales are laid out as [ceil(N/128), ceil(K/128)].
            // scale_k_len is the number of K groups, not the tile width.
            scale_k_group_tile = 128;
        }
        if (is_fp8) {
            TORCH_CHECK(dtype_equal(B_scale.dtype(), dl_float32) && B_scale.ndim() == 2,
                        "FP8 B_scale must be a 2D float32 tensor");
            TORCH_CHECK(B_scale.stride(1) == 1 &&
                        B_scale.stride(0) == B_scale.size(1),
                        "FP8 B_scale must be contiguous");
            TORCH_CHECK(B_scale.size(0) == ceil_div(reduce_size, 128) &&
                        B_scale.size(1) == ceil_div(hidden_size, 128),
                        "FP8 B_scale must use block shape 128x128");
        }
    }

    int nr_n = fuse_swiglu ? reduce_size / 2 : reduce_size;
    std::function<void()> launch_kernel;

    // Config ids are consumed by gemv_auto_tune.py. Keep this order stable.
    GemvBlockConfig configs[] = {
        {8, 32, 0.f, false},
        {8, 16, 0.f, false},
        {16, 4, 0.f, false},
        {16, 8, 0.f, false},
        {32, 4, 0.f, false},
        {4, 16, 0.f, false},
        {4, 32, 0.f, false},
        {32, 1, 0.f, false},
        {128, 1, 0.f, false},
    };

    constexpr int iobit = 128;
    const int bits_of_byte = 8;
    const int vlen = use_int4_w4a16 ? (iobit / 4) : (iobit / (tensor_element_size(B.dtype()) * bits_of_byte));
    float target_ratio = static_cast<float>(reduce_size) / hidden_size;

    for (auto& config : configs) {
        int load_size = config.block_k * vlen;
        // The generic kernel performs vector loads without a K-tail mask.
        // A config that only aligns to the FP8 scale block can still read
        // beyond K (for example K=2432 with a 256-element tile).  Reject it
        // here instead of returning numerically invalid output.
        config.valid = load_size % scale_k_group_tile == 0 &&
            hidden_size % load_size == 0;
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
    constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);
    if (forced_config_id >= 0) {
        TORCH_CHECK(forced_config_id < num_configs, "Invalid GEMV config id");
        best_config = &configs[forced_config_id];
        TORCH_CHECK(best_config->valid, "Forced GEMV config is invalid for this shape");
    } else {
        for (int config_id = 0; config_id < num_configs; ++config_id) {
            if (config_id == 2 || config_id == 5) {
                continue;
            }
            auto& config = configs[config_id];
            if (config.valid && (nr_n % config.block_n == 0) && config.score > best_config->score) {
                best_config = &config;
            }
        }
        for (int config_id = 0; config_id < num_configs; ++config_id) {
            if (config_id == 2 || config_id == 5) {
                continue;
            }
            auto& config = configs[config_id];
            if (!best_config->valid && config.valid && config.score > best_config->score) {
                best_config = &config;
            }
        }
        // With multiple input rows the 8x32 FP8 launch under-fills the GPU:
        // halving BLOCK_K creates twice as many independent N tiles and is
        // consistently faster for the decode M=16..32 range. Keep explicit
        // autotune choices untouched and only refine this specific heuristic
        // winner.
        if (is_fp8 && bseqlen >= 16 && best_config == &configs[0]
            && configs[1].valid) {
            best_config = &configs[1];
        }
        // The same smaller K tile also helps fused FP8 at M>=16, where both
        // gate/up projections otherwise serialize behind the wide reduction.
        if (fuse_swiglu && is_fp8 && bseqlen >= 16 && configs[1].valid) {
            best_config = &configs[1];
        }
        // Fused BF16 gate/up reads are register-heavy. The 16x4 launch keeps
        // enough independent output tiles resident for M>=16 and beats the
        // ratio-only winner (8x32) while preserving the scalar GEMV path for
        // the latency-sensitive M<16 cases.
        if (fuse_swiglu && !is_fp8 && bseqlen >= 16 && configs[2].valid) {
            best_config = &configs[2];
        }
        // The narrow 16x4 reduction is also the best scalar BF16 down-proj
        // layout at M=4: it exposes more output tiles than the ratio-selected
        // 16x8 variant without the register cost of batching rows in a CTA.
        if (!fuse_swiglu && !is_fp8 && bseqlen == 4 && configs[2].valid) {
            best_config = &configs[2];
        }
        // At M=8, keeping the wider K reduction in one CTA is faster than
        // the ratio-matched 8x16 launch. It cuts the number of partial
        // output tiles without sacrificing the token-level grid dimension.
        if (!fuse_swiglu && !is_fp8 && bseqlen == 8 &&
            hidden_size == 2048 && reduce_size == 1024 && configs[0].valid) {
            best_config = &configs[0];
        }
    }
    TORCH_CHECK(best_config->valid, "Unsupported GEMV block configuration");

#define LAUNCH_DENSE_FP8_SWIGLU_STATIC(_BN, _BK, _K, _CHECK_N) \
    do { \
        /* M=3 needs one three-row tile; a 2+1 split reloads the complete \
           weight matrix for the tail and loses most of the BGEMV gain. */ \
        if (bseqlen == 3) { \
            musa_gemv_fp8_block128_swiglu_multitoken_kernel< \
                _BN, _BK, _K, 3, _CHECK_N> \
                <<<dim3{static_cast<uint32_t>(ceil_div(nr_n, _BN)), 1, 1}, \
                   dim3{_BN * _BK, 1, 1}, 0, stream>>>( \
                    static_cast<bfloat16_t*>(C.data_ptr()), \
                    static_cast<const bfloat16_t*>(A.data_ptr()), \
                    static_cast<const __mt_fp8_e4m3*>(B.data_ptr()), \
                    static_cast<const float*>(b_scale_ptr), bseqlen, reduce_size, \
                    scale_k_len); \
        } else if (bseqlen >= 2) { \
            musa_gemv_fp8_block128_swiglu_multitoken_kernel< \
                _BN, _BK, _K, 2, _CHECK_N> \
                <<<dim3{static_cast<uint32_t>(ceil_div(nr_n, _BN)), \
                         static_cast<uint32_t>(ceil_div(bseqlen, 2)), 1}, \
                   dim3{_BN * _BK, 1, 1}, 0, stream>>>( \
                    static_cast<bfloat16_t*>(C.data_ptr()), \
                    static_cast<const bfloat16_t*>(A.data_ptr()), \
                    static_cast<const __mt_fp8_e4m3*>(B.data_ptr()), \
                    static_cast<const float*>(b_scale_ptr), bseqlen, reduce_size, \
                    scale_k_len); \
        } else { \
            musa_gemv_fp8_block128_swiglu_static_kernel< \
                _BN, _BK, _K, _CHECK_N> \
                <<<dim3{static_cast<uint32_t>(ceil_div(nr_n, _BN)), \
                         static_cast<uint32_t>(bseqlen), 1}, \
                   dim3{_BN * _BK, 1, 1}, 0, stream>>>( \
                    static_cast<bfloat16_t*>(C.data_ptr()), \
                    static_cast<const bfloat16_t*>(A.data_ptr()), \
                    static_cast<const __mt_fp8_e4m3*>(B.data_ptr()), \
                    static_cast<const float*>(b_scale_ptr), bseqlen, reduce_size, \
                    scale_k_len); \
        } \
    } while (0)

    const bool use_dense_fp8_swiglu_static =
        forced_config_id < 0 && !fuse_silu && current_arch >= 310 && is_fp8 && fuse_swiglu &&
        bseqlen >= 1 &&
        !use_int4_w4a16 && is_pergroup_scale && scale_k_group_tile == 128 &&
        reduce_size % 2 == 0 &&
        (hidden_size == 2048 || hidden_size == 3072 || hidden_size == 4096 ||
         hidden_size == 5120) &&
        dtype_equal(A.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16);
    if (use_dense_fp8_swiglu_static) {
        // M=1 benefits from more output rows per CTA; M=2..31 instead need
        // the wider K reduction, except that M=32 again has enough token
        // parallelism to make the wide N tile profitable.
        const bool use_wide_n_tile = bseqlen == 1 || bseqlen >= 32;
        const int output_tile = use_wide_n_tile ? 16 : 8;
        const bool check_n = nr_n % output_tile != 0;
        if (hidden_size == 2048) {
            if (use_wide_n_tile) {
                if (check_n) { LAUNCH_DENSE_FP8_SWIGLU_STATIC(16, 8, 2048, true); }
                else { LAUNCH_DENSE_FP8_SWIGLU_STATIC(16, 8, 2048, false); }
            } else if (check_n) {
                LAUNCH_DENSE_FP8_SWIGLU_STATIC(8, 16, 2048, true);
            } else {
                LAUNCH_DENSE_FP8_SWIGLU_STATIC(8, 16, 2048, false);
            }
        } else if (hidden_size == 3072) {
            if (use_wide_n_tile) {
                if (check_n) { LAUNCH_DENSE_FP8_SWIGLU_STATIC(16, 8, 3072, true); }
                else { LAUNCH_DENSE_FP8_SWIGLU_STATIC(16, 8, 3072, false); }
            } else if (check_n) {
                LAUNCH_DENSE_FP8_SWIGLU_STATIC(8, 16, 3072, true);
            } else {
                LAUNCH_DENSE_FP8_SWIGLU_STATIC(8, 16, 3072, false);
            }
        } else if (hidden_size == 4096) {
            if (use_wide_n_tile) {
                if (check_n) { LAUNCH_DENSE_FP8_SWIGLU_STATIC(16, 8, 4096, true); }
                else { LAUNCH_DENSE_FP8_SWIGLU_STATIC(16, 8, 4096, false); }
            } else if (check_n) {
                LAUNCH_DENSE_FP8_SWIGLU_STATIC(8, 16, 4096, true);
            } else {
                LAUNCH_DENSE_FP8_SWIGLU_STATIC(8, 16, 4096, false);
            }
        } else {
            if (use_wide_n_tile) {
                if (check_n) { LAUNCH_DENSE_FP8_SWIGLU_STATIC(16, 8, 5120, true); }
                else { LAUNCH_DENSE_FP8_SWIGLU_STATIC(16, 8, 5120, false); }
            } else if (check_n) {
                LAUNCH_DENSE_FP8_SWIGLU_STATIC(8, 16, 5120, true);
            } else {
                LAUNCH_DENSE_FP8_SWIGLU_STATIC(8, 16, 5120, false);
            }
        }
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA static dense FP8 SwiGLU kernel failed: "
            << musaGetErrorString(err);
        return;
    }
#undef LAUNCH_DENSE_FP8_SWIGLU_STATIC

#define LAUNCH_DENSE_BF16_SWIGLU_BGEMV_AT( \
    _BN, _BK, _TT, _CHECK_N, _TOKENS, _A_OFFSET, _C_OFFSET) \
    musa_gemv_bf16_swiglu_bgemv_kernel<_BN, _BK, _TT, _CHECK_N> \
        <<<dim3{static_cast<uint32_t>(ceil_div(nr_n, _BN)), \
                 static_cast<uint32_t>(ceil_div(_TOKENS, _TT)), 1}, \
           dim3{_BN * _BK, 1, 1}, 0, stream>>>( \
            static_cast<bfloat16_t*>(C.data_ptr()) + (_C_OFFSET), \
            static_cast<const bfloat16_t*>(A.data_ptr()) + (_A_OFFSET), \
            static_cast<const bfloat16_t*>(B.data_ptr()), _TOKENS, reduce_size, \
            hidden_size)

#define SELECT_DENSE_BF16_SWIGLU_BGEMV_AT( \
    _BN, _BK, _TT, _TOKENS, _A_OFFSET, _C_OFFSET) \
    do { \
        if (nr_n % _BN == 0) { \
            LAUNCH_DENSE_BF16_SWIGLU_BGEMV_AT( \
                _BN, _BK, _TT, false, _TOKENS, _A_OFFSET, _C_OFFSET); \
        } else { \
            LAUNCH_DENSE_BF16_SWIGLU_BGEMV_AT( \
                _BN, _BK, _TT, true, _TOKENS, _A_OFFSET, _C_OFFSET); \
        } \
    } while (0)

#define SELECT_DENSE_BF16_SWIGLU_BGEMV(_BN, _BK, _TT) \
    SELECT_DENSE_BF16_SWIGLU_BGEMV_AT( \
        _BN, _BK, _TT, bseqlen, 0, 0)

    const bool can_use_dense_bf16_swiglu_wmma_m16 =
        forced_config_id < 0 && !fuse_silu && current_arch >= 310 && !is_fp8 &&
        !use_int4_w4a16 && fuse_swiglu && bseqlen >= 16 && bseqlen <= 32 &&
        nr_n >= 1024 && nr_n % 32 == 0 && hidden_size % 16 == 0 &&
        dtype_equal(A.dtype(), dl_bfloat16) &&
        dtype_equal(B.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16);
    const bool use_dense_bf16_swiglu_wmma_m16 =
        can_use_dense_bf16_swiglu_wmma_m16 && bseqlen % 16 == 0;
    if (use_dense_bf16_swiglu_wmma_m16) {
        musa_gemv_bf16_swiglu_wmma_m16_kernel<false>
            <<<dim3{static_cast<uint32_t>(nr_n / 32),
                     static_cast<uint32_t>(bseqlen / 16), 1},
               dim3{256, 1, 1}, 0, stream>>>(
                static_cast<bfloat16_t*>(C.data_ptr()),
                static_cast<const bfloat16_t*>(A.data_ptr()),
                static_cast<const bfloat16_t*>(B.data_ptr()), nr_n, hidden_size, 16);
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA dense BF16 SwiGLU WMMA kernel failed: "
            << musaGetErrorString(err);
        return;
    }

    // Keep the complete 16-row tile on tensor cores and send only the tail to
    // BGEMV. Both kernels are enqueued by this one FFI call on the same stream;
    // the output ranges are disjoint and require no intermediate or merge.
    const bool use_dense_bf16_swiglu_wmma_tail =
        can_use_dense_bf16_swiglu_wmma_m16 && bseqlen > 16 && bseqlen < 32;
    if (use_dense_bf16_swiglu_wmma_tail) {
        musa_gemv_bf16_swiglu_wmma_m16_kernel<false>
            <<<dim3{static_cast<uint32_t>(nr_n / 32), 1, 1},
               dim3{256, 1, 1}, 0, stream>>>(
                static_cast<bfloat16_t*>(C.data_ptr()),
                static_cast<const bfloat16_t*>(A.data_ptr()),
                static_cast<const bfloat16_t*>(B.data_ptr()), nr_n, hidden_size, 16);
        const int tail_tokens = bseqlen - 16;
        const int a_offset = 16 * hidden_size;
        const int c_offset = 16 * nr_n;
        if (tail_tokens >= 8) {
            musa_gemv_bf16_swiglu_wmma_m16_kernel<true>
                <<<dim3{static_cast<uint32_t>(nr_n / 32), 1, 1},
                   dim3{256, 1, 1}, 0, stream>>>(
                    static_cast<bfloat16_t*>(C.data_ptr()) + c_offset,
                    static_cast<const bfloat16_t*>(A.data_ptr()) + a_offset,
                    static_cast<const bfloat16_t*>(B.data_ptr()), nr_n,
                    hidden_size, tail_tokens);
        } else if (tail_tokens == 1) {
            SELECT_DENSE_BF16_SWIGLU_BGEMV_AT(
                2, 32, 1, tail_tokens, a_offset, c_offset);
        } else if (tail_tokens == 2) {
            SELECT_DENSE_BF16_SWIGLU_BGEMV_AT(
                4, 32, 2, tail_tokens, a_offset, c_offset);
        } else if (tail_tokens <= 4) {
            SELECT_DENSE_BF16_SWIGLU_BGEMV_AT(
                4, 32, 4, tail_tokens, a_offset, c_offset);
        } else {
            SELECT_DENSE_BF16_SWIGLU_BGEMV_AT(
                8, 16, 4, tail_tokens, a_offset, c_offset);
        }
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA dense BF16 SwiGLU WMMA-tail kernel failed: "
            << musaGetErrorString(err);
        return;
    }

    const bool use_gemv_bf16_swiglu_bgemv =
        forced_config_id < 0 && !fuse_silu && current_arch >= 310 && !is_fp8 &&
        !use_int4_w4a16 && fuse_swiglu &&
        bseqlen >= 1 && bseqlen <= 32 && reduce_size % 2 == 0 &&
        dtype_equal(A.dtype(), dl_bfloat16) &&
        dtype_equal(B.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16);
    if (use_gemv_bf16_swiglu_bgemv) {
        if (bseqlen == 1) {
            SELECT_DENSE_BF16_SWIGLU_BGEMV(2, 32, 1);
        } else if (bseqlen == 2) {
            SELECT_DENSE_BF16_SWIGLU_BGEMV(4, 32, 2);
        } else if (bseqlen <= 4) {
            SELECT_DENSE_BF16_SWIGLU_BGEMV(4, 32, 4);
        } else if (bseqlen <= 16) {
            SELECT_DENSE_BF16_SWIGLU_BGEMV(8, 16, 4);
        } else {
            SELECT_DENSE_BF16_SWIGLU_BGEMV(16, 8, 4);
        }
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA dense BF16 SwiGLU BGEMV kernel failed: "
            << musaGetErrorString(err);
        return;
    }
#undef SELECT_DENSE_BF16_SWIGLU_BGEMV
#undef SELECT_DENSE_BF16_SWIGLU_BGEMV_AT
#undef LAUNCH_DENSE_BF16_SWIGLU_BGEMV_AT

#define LAUNCH_DENSE_FP8(_BLK_N, _BLK_K, _CHECK_BOUNDS) \
    do { \
        if (forced_config_id < 0 && bseqlen == 2) { \
            musa_gemv_fp8_block128_multitoken_kernel< \
                _BLK_N, _BLK_K, 2, _CHECK_BOUNDS> \
                <<<dim3{static_cast<uint32_t>(ceil_div(reduce_size, _BLK_N)), \
                         static_cast<uint32_t>(ceil_div(bseqlen, 2)), 1}, \
                   dim3{_BLK_N * _BLK_K, 1, 1}, 0, stream>>>( \
                    static_cast<bfloat16_t*>(C.data_ptr()), \
                    static_cast<const bfloat16_t*>(A.data_ptr()), \
                    static_cast<const __mt_fp8_e4m3*>(B.data_ptr()), \
                    static_cast<const float*>(b_scale_ptr), \
                    static_cast<const bfloat16_t*>(bias_ptr), bseqlen, reduce_size, \
                    hidden_size, scale_k_len, fuse_silu); \
        } else { \
            musa_gemv_fp8_block128_generic_kernel<_BLK_N, _BLK_K, _CHECK_BOUNDS> \
                <<<dim3{static_cast<uint32_t>(ceil_div(reduce_size, _BLK_N)), \
                         static_cast<uint32_t>(bseqlen), 1}, \
                   dim3{_BLK_N * _BLK_K, 1, 1}, 0, stream>>>( \
                    static_cast<bfloat16_t*>(C.data_ptr()), \
                    static_cast<const bfloat16_t*>(A.data_ptr()), \
                    static_cast<const __mt_fp8_e4m3*>(B.data_ptr()), \
                    static_cast<const float*>(b_scale_ptr), \
                    static_cast<const bfloat16_t*>(bias_ptr), reduce_size, hidden_size, \
                    scale_k_len, fuse_silu); \
        } \
    } while (0)

    const bool use_dense_fp8_kernel =
        is_fp8 && dtype_equal(A.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16) && is_pergroup_scale &&
        scale_k_group_tile == 128 && !use_int4_w4a16 && !fuse_swiglu &&
        bseqlen == 2 &&
        static_cast<size_t>(reduce_size) * hidden_size >= 16 * 1024 * 1024;
    if (use_dense_fp8_kernel) {
        const bool check_bounds = reduce_size % best_config->block_n != 0;
        if (best_config->block_n == 4 && best_config->block_k == 16) {
            if (check_bounds) { LAUNCH_DENSE_FP8(4, 16, true); }
            else { LAUNCH_DENSE_FP8(4, 16, false); }
        } else if (best_config->block_n == 4 && best_config->block_k == 32) {
            if (check_bounds) { LAUNCH_DENSE_FP8(4, 32, true); }
            else { LAUNCH_DENSE_FP8(4, 32, false); }
        } else if (best_config->block_n == 8 && best_config->block_k == 16) {
            if (check_bounds) { LAUNCH_DENSE_FP8(8, 16, true); }
            else { LAUNCH_DENSE_FP8(8, 16, false); }
        } else if (best_config->block_n == 8 && best_config->block_k == 32) {
            if (check_bounds) { LAUNCH_DENSE_FP8(8, 32, true); }
            else { LAUNCH_DENSE_FP8(8, 32, false); }
        } else if (best_config->block_n == 16 && best_config->block_k == 4) {
            if (check_bounds) { LAUNCH_DENSE_FP8(16, 4, true); }
            else { LAUNCH_DENSE_FP8(16, 4, false); }
        } else if (best_config->block_n == 16 && best_config->block_k == 8) {
            if (check_bounds) { LAUNCH_DENSE_FP8(16, 8, true); }
            else { LAUNCH_DENSE_FP8(16, 8, false); }
        } else if (best_config->block_n == 32 && best_config->block_k == 4) {
            if (check_bounds) { LAUNCH_DENSE_FP8(32, 4, true); }
            else { LAUNCH_DENSE_FP8(32, 4, false); }
        } else {
            TORCH_CHECK(false, "Unsupported dense FP8 GEMV block configuration");
        }
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA dense FP8 GEMV kernel failed: " << musaGetErrorString(err);
        return;
    }
#undef LAUNCH_DENSE_FP8

    const bool use_dense_bf16_wmma_m16 =
        forced_config_id < 0 && current_arch >= 310 && !is_fp8 &&
        !use_int4_w4a16 && !fuse_swiglu &&
        bseqlen >= 16 && bseqlen <= 32 && bseqlen % 16 == 0 &&
        reduce_size >= 1024 && reduce_size % 32 == 0 &&
        hidden_size % 16 == 0 &&
        dtype_equal(A.dtype(), dl_bfloat16) && dtype_equal(B.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16);
    if (use_dense_bf16_wmma_m16) {
        musa_gemv_bf16_wmma_m16_kernel
            <<<dim3{static_cast<uint32_t>(reduce_size / 32),
                     static_cast<uint32_t>(bseqlen / 16), 1},
               dim3{256, 1, 1}, 0, stream>>>(
                static_cast<bfloat16_t*>(C.data_ptr()),
                static_cast<const bfloat16_t*>(A.data_ptr()),
                static_cast<const bfloat16_t*>(B.data_ptr()),
                static_cast<const bfloat16_t*>(bias_ptr), reduce_size, hidden_size,
                fuse_silu);
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA dense BF16 WMMA kernel failed: " << musaGetErrorString(err);
        return;
    }

#define LAUNCH_DENSE_BF16_MULTI(_BLK_N, _BLK_K, _TOKEN_TILE, _CHECK_BOUNDS) \
    musa_gemv_bf16_multitoken_kernel< \
        _BLK_N, _BLK_K, _TOKEN_TILE, _CHECK_BOUNDS> \
        <<<dim3{static_cast<uint32_t>(ceil_div(reduce_size, _BLK_N)), \
                 static_cast<uint32_t>(ceil_div(bseqlen, _TOKEN_TILE)), 1}, \
           dim3{_BLK_N * _BLK_K, 1, 1}, 0, stream>>>( \
            static_cast<bfloat16_t*>(C.data_ptr()), \
            static_cast<const bfloat16_t*>(A.data_ptr()), \
            static_cast<const bfloat16_t*>(B.data_ptr()), \
            static_cast<const bfloat16_t*>(bias_ptr), bseqlen, reduce_size, \
            hidden_size, fuse_silu)

#define LAUNCH_DENSE_BF16_MULTI_SELECTED(_BLK_N, _BLK_K, _CHECK_BOUNDS) \
    /* Tile-4 is beneficial through M=21 on MTT S5000; M=22..31 is kept on \
       tile-2 because the extra partial-token registers reduce occupancy. */ \
    if (bseqlen >= 4 && bseqlen <= 21) { \
        LAUNCH_DENSE_BF16_MULTI(_BLK_N, _BLK_K, 4, _CHECK_BOUNDS); \
    } else { \
        LAUNCH_DENSE_BF16_MULTI(_BLK_N, _BLK_K, 2, _CHECK_BOUNDS); \
    }

    const bool use_dense_bf16_multitoken_kernel =
        !is_fp8 && !use_int4_w4a16 && !fuse_swiglu && bseqlen >= 8 &&
        dtype_equal(A.dtype(), dl_bfloat16) && dtype_equal(B.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16);
    if (use_dense_bf16_multitoken_kernel) {
        const bool check_bounds = reduce_size % best_config->block_n != 0;
        if (best_config->block_n == 4 && best_config->block_k == 16) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(4, 16, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(4, 16, false); }
        } else if (best_config->block_n == 4 && best_config->block_k == 32) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(4, 32, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(4, 32, false); }
        } else if (best_config->block_n == 8 && best_config->block_k == 16) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(8, 16, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(8, 16, false); }
        } else if (best_config->block_n == 8 && best_config->block_k == 32) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(8, 32, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(8, 32, false); }
        } else if (best_config->block_n == 16 && best_config->block_k == 4) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(16, 4, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(16, 4, false); }
        } else if (best_config->block_n == 16 && best_config->block_k == 8) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(16, 8, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(16, 8, false); }
        } else if (best_config->block_n == 32 && best_config->block_k == 4) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(32, 4, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(32, 4, false); }
        } else if (best_config->block_n == 32 && best_config->block_k == 1) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(32, 1, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(32, 1, false); }
        } else if (best_config->block_n == 128 && best_config->block_k == 1) {
            if (check_bounds) { LAUNCH_DENSE_BF16_MULTI_SELECTED(128, 1, true); }
            else { LAUNCH_DENSE_BF16_MULTI_SELECTED(128, 1, false); }
        } else {
            TORCH_CHECK(false, "Unsupported multi-token BF16 GEMV block configuration");
        }
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA multi-token BF16 GEMV kernel failed: " << musaGetErrorString(err);
        return;
    }
#undef LAUNCH_DENSE_BF16_MULTI
#undef LAUNCH_DENSE_BF16_MULTI_SELECTED

#define LAUNCH_DENSE_BF16(_BLK_N, _BLK_K, _CHECK_BOUNDS) \
    musa_gemv_bf16_generic_kernel<_BLK_N, _BLK_K, _CHECK_BOUNDS> \
        <<<dim3{static_cast<uint32_t>(ceil_div(reduce_size, _BLK_N)), \
                 static_cast<uint32_t>(bseqlen), 1}, \
           dim3{_BLK_N * _BLK_K, 1, 1}, 0, stream>>>( \
            static_cast<bfloat16_t*>(C.data_ptr()), \
            static_cast<const bfloat16_t*>(A.data_ptr()), \
            static_cast<const bfloat16_t*>(B.data_ptr()), \
            static_cast<const bfloat16_t*>(bias_ptr), reduce_size, hidden_size, \
            fuse_silu)

    const bool use_dense_bf16_kernel =
        !is_fp8 && !use_int4_w4a16 && !fuse_swiglu &&
        dtype_equal(A.dtype(), dl_bfloat16) && dtype_equal(B.dtype(), dl_bfloat16) &&
        dtype_equal(C.dtype(), dl_bfloat16);
    if (use_dense_bf16_kernel) {
        const bool check_bounds = reduce_size % best_config->block_n != 0;
        if (best_config->block_n == 4 && best_config->block_k == 16) {
            if (check_bounds) { LAUNCH_DENSE_BF16(4, 16, true); }
            else { LAUNCH_DENSE_BF16(4, 16, false); }
        } else if (best_config->block_n == 4 && best_config->block_k == 32) {
            if (check_bounds) { LAUNCH_DENSE_BF16(4, 32, true); }
            else { LAUNCH_DENSE_BF16(4, 32, false); }
        } else if (best_config->block_n == 8 && best_config->block_k == 16) {
            if (check_bounds) { LAUNCH_DENSE_BF16(8, 16, true); }
            else { LAUNCH_DENSE_BF16(8, 16, false); }
        } else if (best_config->block_n == 8 && best_config->block_k == 32) {
            if (check_bounds) { LAUNCH_DENSE_BF16(8, 32, true); }
            else { LAUNCH_DENSE_BF16(8, 32, false); }
        } else if (best_config->block_n == 16 && best_config->block_k == 4) {
            if (check_bounds) { LAUNCH_DENSE_BF16(16, 4, true); }
            else { LAUNCH_DENSE_BF16(16, 4, false); }
        } else if (best_config->block_n == 16 && best_config->block_k == 8) {
            if (check_bounds) { LAUNCH_DENSE_BF16(16, 8, true); }
            else { LAUNCH_DENSE_BF16(16, 8, false); }
        } else if (best_config->block_n == 32 && best_config->block_k == 4) {
            if (check_bounds) { LAUNCH_DENSE_BF16(32, 4, true); }
            else { LAUNCH_DENSE_BF16(32, 4, false); }
        } else if (best_config->block_n == 32 && best_config->block_k == 1) {
            if (check_bounds) { LAUNCH_DENSE_BF16(32, 1, true); }
            else { LAUNCH_DENSE_BF16(32, 1, false); }
        } else if (best_config->block_n == 128 && best_config->block_k == 1) {
            if (check_bounds) { LAUNCH_DENSE_BF16(128, 1, true); }
            else { LAUNCH_DENSE_BF16(128, 1, false); }
        } else {
            TORCH_CHECK(false, "Unsupported dense BF16 GEMV block configuration");
        }
        const musaError_t err = musaGetLastError();
        TVM_FFI_ICHECK_EQ(err, musaSuccess)
            << "MUSA dense BF16 GEMV kernel failed: " << musaGetErrorString(err);
        return;
    }
#undef LAUNCH_DENSE_BF16

    switch (best_config->block_n) {
        case 4:
            switch (best_config->block_k) {
                case 16: SELECT_LAUNCH_KERN_GEMV(4, 16); break;
                case 32: SELECT_LAUNCH_KERN_GEMV(4, 32); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=4");
            }
            break;
        case 8:
            switch (best_config->block_k) {
                case 16: SELECT_LAUNCH_KERN_GEMV(8, 16); break;
                case 32: SELECT_LAUNCH_KERN_GEMV(8, 32); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=8");
            }
            break;
        case 16:
            switch (best_config->block_k) {
                case 4: SELECT_LAUNCH_KERN_GEMV(16, 4); break;
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
        << "MUSA dense GEMV kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_gemv(
    ffi::TensorView A,
    ffi::TensorView B,
    ffi::TensorView C,
    ffi::TensorView B_scale,
    ffi::TensorView Bias,
    bool has_b_scale,
    bool has_bias,
    bool use_int4_w4a16,
    bool fuse_swiglu,
    bool fuse_silu,
    int forced_config_id) {
    launch_gemv(
        A,
        B,
        C,
        B_scale,
        Bias,
        has_b_scale,
        has_bias,
        use_int4_w4a16,
        fuse_swiglu,
        fuse_silu,
        forced_config_id);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_gemv, sgl_musa_gemv);
