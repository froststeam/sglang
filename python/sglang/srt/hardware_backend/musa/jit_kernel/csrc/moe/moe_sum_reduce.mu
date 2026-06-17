#include <cstdint>
#include <type_traits>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

#include "../common.h"
#include "../device_utils.h"

constexpr int kWarpSize = 32;

template <typename T>
__device__ __forceinline__ T convert_from_float(float value) {
  return from_float<T>(value);
}

template <>
__device__ __forceinline__ float convert_from_float<float>(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ int4 load_vec8(const T* ptr) {
  return *reinterpret_cast<const int4*>(ptr);
}

template <typename T>
__device__ __forceinline__ void store_vec8(T* ptr, const int4& value) {
  *reinterpret_cast<int4*>(ptr) = value;
}

template <typename T>
union Pack16B {
  int4 v;
  T elems[8];
};

template <typename T>
union Pack4B {
  int v;
  T elems[2];
};

template <typename T, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 1)
void moe_sum_reduce_warp_token_vec8_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t topk_num,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    float scale) {
  constexpr int VEC = 8;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t token = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;
  if (token >= token_num) {
    return;
  }

  const int64_t chunks = hidden_dim / VEC;
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * kWarpSize + lane;
       chunk < chunks;
       chunk += static_cast<int64_t>(gridDim.x) * kWarpSize) {
    const int64_t d = chunk * VEC;
    const int64_t base = token * stride_token + d;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc[i] = 0.0f;
    }

#pragma unroll 1
    for (int64_t k = 0; k < topk_num; ++k) {
      Pack16B<T> pack;
      pack.v = load_vec8(input + base + k * stride_topk);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += to_float(pack.elems[i]);
      }
    }

    Pack16B<T> out;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      out.elems[i] = convert_from_float<T>(acc[i] * scale);
    }
    store_vec8(output + token * out_stride_token + d, out.v);
  }
}

template <typename T, int TOPK, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 1)
void moe_sum_reduce_warp_token_vec8_topk_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    float scale) {
  constexpr int VEC = 8;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t token = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;
  if (token >= token_num) {
    return;
  }

  const int64_t chunks = hidden_dim / VEC;
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * kWarpSize + lane;
       chunk < chunks;
       chunk += static_cast<int64_t>(gridDim.x) * kWarpSize) {
    const int64_t d = chunk * VEC;
    const int64_t base = token * stride_token + d;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc[i] = 0.0f;
    }

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      Pack16B<T> pack;
      pack.v = load_vec8(input + base + static_cast<int64_t>(k) * stride_topk);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += to_float(pack.elems[i]);
      }
    }

    Pack16B<T> out;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      out.elems[i] = convert_from_float<T>(acc[i] * scale);
    }
    store_vec8(output + token * out_stride_token + d, out.v);
  }
}

template <typename T>
__global__ __launch_bounds__(256, 1)
void moe_sum_reduce_small_token_vec8_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t topk_num,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    float scale) {
  constexpr int VEC = 8;
  const int64_t token = blockIdx.y;
  if (token >= token_num) {
    return;
  }

  const int64_t chunks = hidden_dim / VEC;
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       chunk < chunks;
       chunk += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t d = chunk * VEC;
    const int64_t base = token * stride_token + d;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc[i] = 0.0f;
    }

#pragma unroll 1
    for (int64_t k = 0; k < topk_num; ++k) {
      Pack16B<T> pack;
      pack.v = load_vec8(input + base + k * stride_topk);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += to_float(pack.elems[i]);
      }
    }

    Pack16B<T> out;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      out.elems[i] = convert_from_float<T>(acc[i] * scale);
    }
    store_vec8(output + token * out_stride_token + d, out.v);
  }
}

template <typename T, int TOPK>
__global__ __launch_bounds__(256, 1)
void moe_sum_reduce_small_token_vec8_topk_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    float scale) {
  constexpr int VEC = 8;
  const int64_t token = blockIdx.y;
  if (token >= token_num) {
    return;
  }

  const int64_t chunks = hidden_dim / VEC;
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       chunk < chunks;
       chunk += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t d = chunk * VEC;
    const int64_t base = token * stride_token + d;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc[i] = 0.0f;
    }

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      Pack16B<T> pack;
      pack.v = load_vec8(input + base + static_cast<int64_t>(k) * stride_topk);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += to_float(pack.elems[i]);
      }
    }

    Pack16B<T> out;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      out.elems[i] = convert_from_float<T>(acc[i] * scale);
    }
    store_vec8(output + token * out_stride_token + d, out.v);
  }
}

template <typename T, int TOPK, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 1)
void moe_sum_reduce_warp_token_topk_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    float scale) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t token = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;
  if (token >= token_num) {
    return;
  }

  for (int64_t d = static_cast<int64_t>(blockIdx.x) * kWarpSize + lane;
       d < hidden_dim;
       d += static_cast<int64_t>(gridDim.x) * kWarpSize) {
    const int64_t base = token * stride_token + d;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      acc += to_float(input[base + static_cast<int64_t>(k) * stride_topk]);
    }
    output[token * out_stride_token + d] = convert_from_float<T>(acc * scale);
  }
}

template <typename T, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 1)
void moe_sum_reduce_warp_token_general_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    int topk_num,
    float scale) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t token = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;
  if (token >= token_num) {
    return;
  }

  for (int64_t d = static_cast<int64_t>(blockIdx.x) * kWarpSize + lane;
       d < hidden_dim;
       d += static_cast<int64_t>(gridDim.x) * kWarpSize) {
    const int64_t base = token * stride_token + d;
    float acc = 0.0f;
#pragma unroll 1
    for (int k = 0; k < topk_num; ++k) {
      acc += to_float(input[base + static_cast<int64_t>(k) * stride_topk]);
    }
    output[token * out_stride_token + d] = convert_from_float<T>(acc * scale);
  }
}

template <typename T, int TOPK>
__global__ void moe_sum_reduce_scalar_topk_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    float scale) {
  for (int64_t token = blockIdx.y; token < token_num; token += gridDim.y) {
    for (int64_t d = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         d < hidden_dim;
         d += static_cast<int64_t>(blockDim.x) * gridDim.x) {
      const int64_t base = token * stride_token + d;
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        acc += to_float(input[base + static_cast<int64_t>(k) * stride_topk]);
      }
      output[token * out_stride_token + d] = convert_from_float<T>(acc * scale);
    }
  }
}

template <typename T, int TOPK>
__global__ __launch_bounds__(256, 1)
void moe_sum_reduce_scalar_vec2_topk_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    float scale) {
  constexpr int VEC = 2;
  for (int64_t token = blockIdx.y; token < token_num; token += gridDim.y) {
    const int64_t chunks = hidden_dim / VEC;
    for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         chunk < chunks;
         chunk += static_cast<int64_t>(blockDim.x) * gridDim.x) {
      const int64_t d = chunk * VEC;
      const int64_t base = token * stride_token + d;
      float acc0 = 0.0f;
      float acc1 = 0.0f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        Pack4B<T> pack;
        pack.v = *reinterpret_cast<const int*>(input + base + static_cast<int64_t>(k) * stride_topk);
        acc0 += to_float(pack.elems[0]);
        acc1 += to_float(pack.elems[1]);
      }
      Pack4B<T> out;
      out.elems[0] = convert_from_float<T>(acc0 * scale);
      out.elems[1] = convert_from_float<T>(acc1 * scale);
      *reinterpret_cast<int*>(output + token * out_stride_token + d) = out.v;
    }
  }
}

template <typename T>
__global__ void moe_sum_reduce_scalar_general_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t token_num,
    int64_t hidden_dim,
    int64_t stride_token,
    int64_t stride_topk,
    int64_t out_stride_token,
    int topk_num,
    float scale) {
  for (int64_t token = blockIdx.y; token < token_num; token += gridDim.y) {
    for (int64_t d = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         d < hidden_dim;
         d += static_cast<int64_t>(blockDim.x) * gridDim.x) {
      const int64_t base = token * stride_token + d;
      float acc = 0.0f;
#pragma unroll 1
      for (int k = 0; k < topk_num; ++k) {
        acc += to_float(input[base + static_cast<int64_t>(k) * stride_topk]);
      }
      output[token * out_stride_token + d] = convert_from_float<T>(acc * scale);
    }
  }
}

void check_moe_sum_reduce_inputs(ffi::TensorView input, ffi::TensorView output) {
  CHECK_MUSA_CONTIGUOUS(input);
  CHECK_MUSA_CONTIGUOUS(output);
  TVM_FFI_ICHECK_EQ(input.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(input.size(2), output.size(1));
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), output.dtype()));
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), dl_float32) ||
                 dtype_equal(input.dtype(), dl_float16) ||
                 dtype_equal(input.dtype(), dl_bfloat16));
}

int small_vec8_threads(int64_t hidden_dim) {
  const int64_t chunks = hidden_dim / 8;
  if (chunks == 352) return 32;  // hidden=2816
  constexpr int candidates[] = {256, 224, 192, 160, 128, 96, 64, 32};
  int best_threads = 256;
  int64_t best_score = INT64_MAX;
  for (int threads : candidates) {
    const int64_t blocks = (chunks + threads - 1) / threads;
    const int64_t padded = blocks * threads;
    const int64_t waste = padded - chunks;
    const int64_t score = waste + blocks * 32;
    if (score < best_score) {
      best_score = score;
      best_threads = threads;
    }
  }
  return best_threads;
}

template <typename T>
void launch_moe_sum_reduce(ffi::TensorView input, ffi::TensorView output, float scale) {
  const int64_t token_num = input.size(0);
  const int64_t topk_num = input.size(1);
  const int64_t hidden_dim = input.size(2);
  const int64_t stride_token = input.stride(0);
  const int64_t stride_topk = input.stride(1);
  const int64_t out_stride_token = output.stride(0);
  musaStream_t stream = get_stream(input.device());

  if (token_num == 0 || hidden_dim == 0) {
    return;
  }

  if (!std::is_same_v<T, float> &&
      ((topk_num == 2 && token_num <= 8) ||
       (topk_num == 4 &&
        (token_num <= 8 || (token_num <= 32 && hidden_dim <= 4096) ||
         (token_num >= 256 && hidden_dim >= 7168))) ||
       ((topk_num == 8 || topk_num == 9) && token_num <= 512)) &&
      (hidden_dim % 8) == 0) {
    const int THREADS =
        (topk_num == 2 && token_num <= 4) ? 128 :
        (token_num <= 4 ? 64 : (token_num <= 8 ? 128 : 256));
    const bool use_vec2 =
        (topk_num == 4) ? token_num >= 256 : token_num >= 64;
    int64_t grid_x =
        ((use_vec2 ? hidden_dim / 2 : hidden_dim) + THREADS - 1) / THREADS;
    grid_x = grid_x > 65535 ? 65535 : grid_x;
    int64_t grid_y = token_num > 65535 ? 65535 : token_num;
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
    dim3 block(THREADS);
#define LAUNCH_TINY_TOPK(TOPK)                                                \
  do {                                                                        \
    if (use_vec2) {                                                           \
      moe_sum_reduce_scalar_vec2_topk_kernel<T, TOPK>                         \
          <<<grid, block, 0, stream>>>(                                       \
              static_cast<const T*>(input.data_ptr()),                        \
              static_cast<T*>(output.data_ptr()),                             \
              token_num, hidden_dim, stride_token, stride_topk,               \
              out_stride_token, scale);                                       \
    } else {                                                                  \
      moe_sum_reduce_scalar_topk_kernel<T, TOPK><<<grid, block, 0, stream>>>( \
          static_cast<const T*>(input.data_ptr()),                            \
          static_cast<T*>(output.data_ptr()),                                 \
          token_num, hidden_dim, stride_token, stride_topk, out_stride_token, \
          scale);                                                             \
    }                                                                         \
  } while (0)
    switch (topk_num) {
      case 2:
        LAUNCH_TINY_TOPK(2);
        break;
      case 4:
        LAUNCH_TINY_TOPK(4);
        break;
      case 8:
        LAUNCH_TINY_TOPK(8);
        break;
      default:
        LAUNCH_TINY_TOPK(9);
        break;
    }
#undef LAUNCH_TINY_TOPK
    return;
  }

  const bool small_vec8_ok =
      !std::is_same_v<T, float> && token_num <= 96 && (hidden_dim % 8) == 0;
  if (small_vec8_ok) {
    const int THREADS = (token_num <= 4 && hidden_dim == 7168) ? 256 : small_vec8_threads(hidden_dim);
    int64_t grid_x = (hidden_dim / 8 + THREADS - 1) / THREADS;
    grid_x = grid_x > 65535 ? 65535 : grid_x;
    int64_t grid_y = token_num > 65535 ? 65535 : token_num;
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
    dim3 block(THREADS);
    if (token_num <= 32) {
#define LAUNCH_SMALL_VEC8_TOPK(TOPK)                                          \
  moe_sum_reduce_small_token_vec8_topk_kernel<T, TOPK><<<grid, block, 0, stream>>>( \
      static_cast<const T*>(input.data_ptr()),                                \
      static_cast<T*>(output.data_ptr()),                                     \
      token_num, hidden_dim, stride_token, stride_topk, out_stride_token,     \
      scale)
      switch (topk_num) {
        case 2:
          LAUNCH_SMALL_VEC8_TOPK(2);
          break;
        case 4:
          LAUNCH_SMALL_VEC8_TOPK(4);
          break;
        case 8:
          LAUNCH_SMALL_VEC8_TOPK(8);
          break;
        case 9:
          LAUNCH_SMALL_VEC8_TOPK(9);
          break;
        default:
          moe_sum_reduce_small_token_vec8_kernel<T><<<grid, block, 0, stream>>>(
              static_cast<const T*>(input.data_ptr()),
              static_cast<T*>(output.data_ptr()),
              token_num,
              hidden_dim,
              topk_num,
              stride_token,
              stride_topk,
              out_stride_token,
              scale);
      }
#undef LAUNCH_SMALL_VEC8_TOPK
    } else {
      moe_sum_reduce_small_token_vec8_kernel<T><<<grid, block, 0, stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<T*>(output.data_ptr()),
          token_num,
          hidden_dim,
          topk_num,
          stride_token,
          stride_topk,
          out_stride_token,
          scale);
    }
    return;
  }

  const bool vec8_ok =
      !std::is_same_v<T, float> && token_num >= 128 && (hidden_dim % 8) == 0;
  if (vec8_ok) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * kWarpSize;
    int64_t grid_x = (hidden_dim / 8 + kWarpSize - 1) / kWarpSize;
    grid_x = grid_x > 65535 ? 65535 : grid_x;
    int64_t grid_y = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    grid_y = grid_y > 65535 ? 65535 : grid_y;
    moe_sum_reduce_warp_token_vec8_kernel<T, WARPS_PER_BLOCK>
        <<<dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y)),
           dim3(THREADS), 0, stream>>>(
            static_cast<const T*>(input.data_ptr()),
            static_cast<T*>(output.data_ptr()),
            token_num,
            hidden_dim,
            topk_num,
            stride_token,
            stride_topk,
            out_stride_token,
            scale);
    return;
  }

  const bool warp_per_token = token_num > 128;
  if (warp_per_token) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * kWarpSize;
    int64_t grid_x = (hidden_dim + kWarpSize - 1) / kWarpSize;
    grid_x = grid_x > 65535 ? 65535 : grid_x;
    int64_t grid_y = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    grid_y = grid_y > 65535 ? 65535 : grid_y;
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
    dim3 block(THREADS);
#define LAUNCH_WARP_TOPK(TOPK)                                                \
  moe_sum_reduce_warp_token_topk_kernel<T, TOPK, WARPS_PER_BLOCK>             \
      <<<grid, block, 0, stream>>>(                                           \
          static_cast<const T*>(input.data_ptr()),                            \
          static_cast<T*>(output.data_ptr()),                                 \
          token_num, hidden_dim, stride_token, stride_topk, out_stride_token, \
          scale)
    switch (topk_num) {
      case 2:
        LAUNCH_WARP_TOPK(2);
        break;
      case 4:
        LAUNCH_WARP_TOPK(4);
        break;
      case 8:
        LAUNCH_WARP_TOPK(8);
        break;
      case 9:
        LAUNCH_WARP_TOPK(9);
        break;
      default:
        moe_sum_reduce_warp_token_general_kernel<T, WARPS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(
                static_cast<const T*>(input.data_ptr()),
                static_cast<T*>(output.data_ptr()),
                token_num, hidden_dim, stride_token, stride_topk,
                out_stride_token, static_cast<int>(topk_num), scale);
    }
#undef LAUNCH_WARP_TOPK
  } else {
    constexpr int THREADS = 256;
    int64_t grid_x = (hidden_dim + THREADS - 1) / THREADS;
    grid_x = grid_x > 65535 ? 65535 : grid_x;
    int64_t grid_y = token_num > 65535 ? 65535 : token_num;
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
    dim3 block(THREADS);
#define LAUNCH_SCALAR_TOPK(TOPK)                                              \
  moe_sum_reduce_scalar_topk_kernel<T, TOPK><<<grid, block, 0, stream>>>(     \
      static_cast<const T*>(input.data_ptr()),                                \
      static_cast<T*>(output.data_ptr()),                                     \
      token_num, hidden_dim, stride_token, stride_topk, out_stride_token,     \
      scale)
    switch (topk_num) {
      case 2:
        LAUNCH_SCALAR_TOPK(2);
        break;
      case 4:
        LAUNCH_SCALAR_TOPK(4);
        break;
      case 8:
        LAUNCH_SCALAR_TOPK(8);
        break;
      case 9:
        LAUNCH_SCALAR_TOPK(9);
        break;
      default:
        moe_sum_reduce_scalar_general_kernel<T><<<grid, block, 0, stream>>>(
            static_cast<const T*>(input.data_ptr()),
            static_cast<T*>(output.data_ptr()),
            token_num, hidden_dim, stride_token, stride_topk, out_stride_token,
            static_cast<int>(topk_num), scale);
    }
#undef LAUNCH_SCALAR_TOPK
  }
}

void sgl_musa_moe_sum_reduce(ffi::TensorView input, ffi::TensorView output,
                             double routed_scaling_factor) {
  check_moe_sum_reduce_inputs(input, output);
  ffi::MUSADeviceGuard device_guard(input.device().device_id);
  const float scale = static_cast<float>(routed_scaling_factor);
  if (dtype_equal(input.dtype(), dl_float32)) {
    launch_moe_sum_reduce<float>(input, output, scale);
  } else if (dtype_equal(input.dtype(), dl_float16)) {
    launch_moe_sum_reduce<half>(input, output, scale);
  } else if (dtype_equal(input.dtype(), dl_bfloat16)) {
    launch_moe_sum_reduce<__mt_bfloat16>(input, output, scale);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported moe_sum_reduce dtype";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA moe_sum_reduce kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_moe_sum_reduce, sgl_musa_moe_sum_reduce);
