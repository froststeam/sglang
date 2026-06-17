#include <cstdint>

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

template <typename T>
struct __align__(16) Vec8Storage {
  T elem[8];
};

struct __align__(32) Float8Storage {
  float elem[8];
};

template <typename T>
struct __align__(16) Vec8 {
  union {
    Vec8Storage<T> storage;
    T elem[8];
  } val;

  __device__ __forceinline__ Vec8() {}

  template <typename Offset>
  static __device__ __forceinline__ Vec8 load(const T* ptr, Offset idx) {
    return *(const Vec8*)(ptr + idx);
  }

  template <typename Offset>
  static __device__ __forceinline__ Vec8 load_byp_slc(const T* ptr, Offset idx) {
#if ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
    Vec8 dst;
    const T* addr = ptr + idx;
    asm volatile(
        "LSU.LD.B128 %0, %1, _, 16, 1, 1, inner_persist=0, outer_persist=2, "
        "chrnt=l2_l3, slc=byp, persist=0, stride_add_first=0"
        : "=R"(dst)
        : "R"(addr));
    return dst;
#else
    return *(const Vec8*)(ptr + idx);
#endif
  }
};

struct __align__(32) Float8 {
  union {
    Float8Storage storage;
    float elem[8];
  } val;

  __device__ __forceinline__ Float8() {}
};

__device__ __forceinline__ int qwen3vl_interleaved_axis(int rot_offset) {
  const int mod = rot_offset % 3;
  return (mod == 1 && rot_offset <= 60) ? 1
                                        : ((mod == 2 && rot_offset <= 60) ? 2 : 0);
}

__device__ __forceinline__ float fast_rsqrt(float value) {
#if ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
  const float half_value = 0.5f * value;
  float y = __frsqrt_rn(value);
  y = y * (1.5f - half_value * y * y);
  return y;
#else
  return rsqrtf(value);
#endif
}

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = ((int)blockDim.x + 31) >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

__device__ __forceinline__ float block_sum_8warps(float value, float* warp_sums) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 8 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

__device__ __forceinline__ float block_sum_4warps(float value, float* warp_sums) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 4 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

template <typename T, bool GEMMA, bool CACHE>
__launch_bounds__(1024, 1)
__global__ void rmsnorm_vec8_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t out_row_stride,
    float inv_hidden,
    float eps) {
  constexpr int kVec = 8;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + (CACHE ? hidden : 0);

  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int vec_count = hidden / kVec;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t out_base = (int64_t)row * out_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load(input + input_base, col);
    Float8 x_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value = to_float<T>(x.val.elem[i]);
      sum += value * value;
      x_float.val.elem[i] = value;
    }
    if constexpr (CACHE) {
      *(Float8*)(cached + col) = x_float;
    }
  }

  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Float8 x_float;
    if constexpr (CACHE) {
      x_float = *(Float8*)(cached + col);
    } else {
      Vec8<T> x = Vec8<T>::load(input + input_base, col);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        x_float.val.elem[i] = to_float<T>(x.val.elem[i]);
      }
    }
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value = to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] = from_float<T>(x_float.val.elem[i] * scale * weight_value);
    }
    *(Vec8<T>*)(out + out_base + col) = dst;
  }
}

template <typename T, bool GEMMA, int H, int WARPS>
__launch_bounds__(256, 1)
__global__ void rmsnorm_small_h_vec8_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t out_row_stride,
    float inv_hidden,
    float eps) {
  constexpr int kVec = 8;
  constexpr int vec_count = H / kVec;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + H;

  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t out_base = (int64_t)row * out_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load(input + input_base, col);
    Float8 x_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value = to_float<T>(x.val.elem[i]);
      sum += value * value;
      x_float.val.elem[i] = value;
    }
    *(Float8*)(cached + col) = x_float;
  }

  if constexpr (WARPS == 4) {
    sum = block_sum_4warps(sum, warp_sums);
  } else {
    sum = block_sum_8warps(sum, warp_sums);
  }

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Float8 x_float = *(Float8*)(cached + col);
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value = to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] = from_float<T>(x_float.val.elem[i] * scale * weight_value);
    }
    *(Vec8<T>*)(out + out_base + col) = dst;
  }
}

template <typename T, bool GEMMA, int H, int WARPS>
__launch_bounds__(256, 1)
__global__ void rmsnorm_small_h_one_vec_register_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t out_row_stride,
    float inv_hidden,
    float eps) {
  constexpr int kVec = 8;
  extern __shared__ float warp_sums[];

  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int col = tid * kVec;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t out_base = (int64_t)row * out_row_stride;
  float sum = 0.0f;
  Float8 x_float;

  Vec8<T> x = Vec8<T>::load(input + input_base, col);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float value = to_float<T>(x.val.elem[i]);
    sum += value * value;
    x_float.val.elem[i] = value;
  }

  if constexpr (WARPS == 4) {
    sum = block_sum_4warps(sum, warp_sums);
  } else {
    sum = block_sum_8warps(sum, warp_sums);
  }

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  Vec8<T> w = Vec8<T>::load(weight, col);
  Vec8<T> dst;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float weight_value = to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
    dst.val.elem[i] = from_float<T>(x_float.val.elem[i] * scale * weight_value);
  }
  *(Vec8<T>*)(out + out_base + col) = dst;
}

template <typename T, bool GEMMA, bool CACHE>
__launch_bounds__(1024, 1)
__global__ void fused_add_rmsnorm_vec8_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t residual_row_stride,
    float inv_hidden,
    float eps) {
  constexpr int kVec = 8;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + (CACHE ? hidden : 0);

  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int vec_count = hidden / kVec;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t residual_base = (int64_t)row * residual_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load_byp_slc(input + input_base, col);
    Vec8<T> r = Vec8<T>::load_byp_slc(residual + residual_base, col);
    Vec8<T> residual_out;
    Float8 sum_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value = to_float<T>(x.val.elem[i]) + to_float<T>(r.val.elem[i]);
      sum += value * value;
      residual_out.val.elem[i] = from_float<T>(value);
      sum_float.val.elem[i] = value;
    }
    *(Vec8<T>*)(residual + residual_base + col) = residual_out;
    if constexpr (CACHE) {
      *(Float8*)(cached + col) = sum_float;
    }
  }

  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Float8 sum_float;
    if constexpr (CACHE) {
      sum_float = *(Float8*)(cached + col);
    } else {
      Vec8<T> r = Vec8<T>::load(residual + residual_base, col);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
      }
    }
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value = to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] = from_float<T>(sum_float.val.elem[i] * scale * weight_value);
    }
    *(Vec8<T>*)(input + input_base + col) = dst;
  }
}

template <typename T, bool GEMMA>
__global__ void rmsnorm_scalar_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t out_row_stride,
    float inv_hidden,
    float eps) {
  extern __shared__ float warp_sums[];
  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t out_base = (int64_t)row * out_row_stride;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += (int)blockDim.x) {
    const float value = to_float<T>(input[input_base + col]);
    sum += value * value;
  }
  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int col = tid; col < hidden; col += (int)blockDim.x) {
    const float weight_value = to_float<T>(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    out[out_base + col] = from_float<T>(to_float<T>(input[input_base + col]) * scale * weight_value);
  }
}

template <typename T, bool GEMMA>
__global__ void fused_add_rmsnorm_scalar_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t residual_row_stride,
    float inv_hidden,
    float eps) {
  extern __shared__ float warp_sums[];
  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t residual_base = (int64_t)row * residual_row_stride;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += (int)blockDim.x) {
    const float value = to_float<T>(input[input_base + col]) + to_float<T>(residual[residual_base + col]);
    residual[residual_base + col] = from_float<T>(value);
    sum += value * value;
  }
  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int col = tid; col < hidden; col += (int)blockDim.x) {
    const float weight_value = to_float<T>(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    input[input_base + col] = from_float<T>(to_float<T>(residual[residual_base + col]) * scale * weight_value);
  }
}

__global__ void fused_qk_rmsnorm_mrope_qwen3vl_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    int batch, int64_t position_stride, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride, float eps) {
  constexpr int q_heads = 32;
  constexpr int k_heads = 4;
  constexpr int hidden = 128;
  constexpr int embed_dim = 64;
  constexpr int rot_dim = 128;
  constexpr int heads_per_block = 8;
  constexpr int groups_per_token = 5;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= q_heads + k_heads) {
    return;
  }

  const bool is_q = global_head < q_heads;
  const int head = is_q ? global_head : global_head - q_heads;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  __mt_bfloat16 *__restrict__ out = is_q ? q_out : k_out;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int64_t out_base =
      is_q ? ((int64_t)token * q_out_batch_stride +
              (int64_t)head * q_out_head_stride)
           : ((int64_t)token * k_out_batch_stride +
              (int64_t)head * k_out_head_stride);

  float sum = 0.0f;
#pragma unroll
  for (int col = lane; col < hidden; col += 32) {
    const float value = __bfloat162float(data[base + col]);
    sum += value * value;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);
#pragma unroll
  for (int rot_offset = lane; rot_offset < embed_dim; rot_offset += 32) {
    const int axis = qwen3vl_interleaved_axis(rot_offset);
    const int64_t pos = positions[(int64_t)axis * position_stride + token];
    const __mt_bfloat16 *__restrict__ cache_ptr =
        cos_sin_cache + pos * rot_dim;
    const float cos_v = __bfloat162float(cache_ptr[rot_offset]);
    const float sin_v = __bfloat162float(cache_ptr[embed_dim + rot_offset]);
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;
    const float x = __bfloat162float(data[base + x_index]) * scale *
                    __bfloat162float(weight[x_index]);
    const float y = __bfloat162float(data[base + y_index]) * scale *
                    __bfloat162float(weight[y_index]);
    out[out_base + x_index] = __float2bfloat16_rn(x * cos_v - y * sin_v);
    out[out_base + y_index] = __float2bfloat16_rn(y * cos_v + x * sin_v);
  }
}

template <typename index_t>
__global__ void fused_qk_rmsnorm_mrope_cache_qwen3vl_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ v,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_cache,
    __mt_bfloat16 *__restrict__ v_cache, const index_t *__restrict__ indices,
    int batch, int64_t position_stride, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t v_batch_stride, int64_t v_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_cache_row_stride, int64_t v_cache_row_stride,
    int64_t indices_stride, float eps) {
  constexpr int q_heads = 32;
  constexpr int k_heads = 4;
  constexpr int hidden = 128;
  constexpr int embed_dim = 64;
  constexpr int rot_dim = 128;
  constexpr int heads_per_block = 8;
  constexpr int groups_per_token = 5;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token * indices_stride]);

  if (group == 0) {
    constexpr int kVec = 8;
    constexpr int row_dim = k_heads * hidden;
    constexpr int vec_count = row_dim / kVec;
    const int64_t v_token_base = (int64_t)token * v_batch_stride;
    const int64_t v_out_base = cache_idx * v_cache_row_stride;
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
      const int head = vec_idx >> 4;
      const int col = (vec_idx & 15) * kVec;
      Vec8<__mt_bfloat16> v_vec = Vec8<__mt_bfloat16>::load(
          v + v_token_base + (int64_t)head * v_head_stride, col);
      *(Vec8<__mt_bfloat16> *)(v_cache + v_out_base + (int64_t)head * hidden +
                               col) = v_vec;
    }
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= q_heads + k_heads) {
    return;
  }

  const bool is_q = global_head < q_heads;
  const int head = is_q ? global_head : global_head - q_heads;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int64_t q_out_base =
      (int64_t)token * q_out_batch_stride + (int64_t)head * q_out_head_stride;
  const int64_t k_cache_base =
      cache_idx * k_cache_row_stride + (int64_t)head * hidden;

  float sum = 0.0f;
#pragma unroll
  for (int col = lane; col < hidden; col += 32) {
    const float value = __bfloat162float(data[base + col]);
    sum += value * value;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);
#pragma unroll
  for (int rot_offset = lane; rot_offset < embed_dim; rot_offset += 32) {
    const int axis = qwen3vl_interleaved_axis(rot_offset);
    const int64_t pos = positions[(int64_t)axis * position_stride + token];
    const __mt_bfloat16 *__restrict__ cache_ptr =
        cos_sin_cache + pos * rot_dim;
    const float cos_v = __bfloat162float(cache_ptr[rot_offset]);
    const float sin_v = __bfloat162float(cache_ptr[embed_dim + rot_offset]);
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;
    const float x = __bfloat162float(data[base + x_index]) * scale *
                    __bfloat162float(weight[x_index]);
    const float y = __bfloat162float(data[base + y_index]) * scale *
                    __bfloat162float(weight[y_index]);
    const __mt_bfloat16 x_rot = __float2bfloat16_rn(x * cos_v - y * sin_v);
    const __mt_bfloat16 y_rot = __float2bfloat16_rn(y * cos_v + x * sin_v);
    if (is_q) {
      q_out[q_out_base + x_index] = x_rot;
      q_out[q_out_base + y_index] = y_rot;
    } else {
      k_cache[k_cache_base + x_index] = x_rot;
      k_cache[k_cache_base + y_index] = y_rot;
    }
  }
}

template <typename index_t, typename vec_t>
__global__ void store_kv_cache_kernel(
    const char *__restrict__ k, const char *__restrict__ v,
    char *__restrict__ k_cache, char *__restrict__ v_cache,
    const index_t *__restrict__ indices, int64_t k_stride_bytes,
    int64_t v_stride_bytes, int64_t k_cache_stride_bytes,
    int64_t v_cache_stride_bytes, int64_t indices_stride,
    int64_t row_bytes, int64_t num_tokens) {
  const int token = (int)blockIdx.x;
  if (token >= num_tokens) {
    return;
  }

  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token * indices_stride]);
  const vec_t *__restrict__ k_src =
      reinterpret_cast<const vec_t *>(k + (int64_t)token * k_stride_bytes);
  const vec_t *__restrict__ v_src =
      reinterpret_cast<const vec_t *>(v + (int64_t)token * v_stride_bytes);
  vec_t *__restrict__ k_dst = reinterpret_cast<vec_t *>(
      k_cache + cache_idx * k_cache_stride_bytes);
  vec_t *__restrict__ v_dst = reinterpret_cast<vec_t *>(
      v_cache + cache_idx * v_cache_stride_bytes);

  const int64_t vec_count = row_bytes / (int64_t)sizeof(vec_t);
  for (int64_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
    k_dst[i] = k_src[i];
    v_dst[i] = v_src[i];
  }
}

template <typename index_t, bool WITH_CACHE>
__global__ void fused_qk_rmsnorm_mrope_generic_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ v,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int batch, int q_heads, int k_heads,
    int hidden, int64_t position_stride, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t v_batch_stride, int64_t v_head_stride, int64_t q_out_batch_stride,
    int64_t q_out_head_stride, int64_t k_out_batch_stride,
    int64_t k_out_head_stride, int64_t k_cache_row_stride,
    int64_t v_cache_row_stride, int64_t indices_stride, int mrope_section_t,
    int mrope_section_h, int mrope_section_w, bool is_interleaved, float eps) {
  constexpr int heads_per_block = 8;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int total_heads = q_heads + k_heads;
  const int groups_per_token =
      (total_heads + heads_per_block - 1) / heads_per_block;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  int64_t cache_idx = 0;
  if constexpr (WITH_CACHE) {
    cache_idx = static_cast<int64_t>(indices[(int64_t)token * indices_stride]);
    if (group == 0) {
      const int row_dim = k_heads * hidden;
      const int64_t v_base = (int64_t)token * v_batch_stride;
      const int64_t v_dst = cache_idx * v_cache_row_stride;
      for (int col = tid; col < row_dim; col += (int)blockDim.x) {
        const int head = col / hidden;
        const int head_col = col - head * hidden;
        v_cache[v_dst + col] =
            v[v_base + (int64_t)head * v_head_stride + head_col];
      }
    }
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= total_heads) {
    return;
  }
  const bool is_q = global_head < q_heads;
  const int head = is_q ? global_head : global_head - q_heads;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int embed_dim = hidden >> 1;

  float sum = 0.0f;
  for (int col = lane; col < hidden; col += 32) {
    const float value = __bfloat162float(data[base + col]);
    sum += value * value;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum / static_cast<float>(hidden) + eps);

  for (int rot_offset = lane; rot_offset < embed_dim; rot_offset += 32) {
    int axis;
    if (is_interleaved) {
      const bool use_h =
          (rot_offset % 3 == 1) && (rot_offset <= 3 * mrope_section_h);
      const bool use_w =
          (rot_offset % 3 == 2) && (rot_offset <= 3 * mrope_section_w);
      axis = use_h ? 1 : (use_w ? 2 : 0);
    } else {
      axis = rot_offset < mrope_section_t
                 ? 0
                 : (rot_offset < mrope_section_t + mrope_section_h ? 1 : 2);
    }
    const int64_t pos = positions[(int64_t)axis * position_stride + token];
    const __mt_bfloat16 *__restrict__ cache_ptr = cos_sin_cache + pos * hidden;
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;
    const float cos_v = __bfloat162float(cache_ptr[x_index]);
    const float sin_v = __bfloat162float(cache_ptr[y_index]);
    const float x = __bfloat162float(data[base + x_index]) * scale *
                    __bfloat162float(weight[x_index]);
    const float y = __bfloat162float(data[base + y_index]) * scale *
                    __bfloat162float(weight[y_index]);
    const __mt_bfloat16 x_rot = __float2bfloat16_rn(x * cos_v - y * sin_v);
    const __mt_bfloat16 y_rot = __float2bfloat16_rn(y * cos_v + x * sin_v);
    if (is_q) {
      const int64_t out_base =
          (int64_t)token * q_out_batch_stride +
          (int64_t)head * q_out_head_stride;
      q_out[out_base + x_index] = x_rot;
      q_out[out_base + y_index] = y_rot;
    } else if constexpr (WITH_CACHE) {
      const int64_t cache_base =
          cache_idx * k_cache_row_stride + (int64_t)head * hidden;
      k_cache[cache_base + x_index] = x_rot;
      k_cache[cache_base + y_index] = y_rot;
    } else {
      const int64_t out_base =
          (int64_t)token * k_out_batch_stride +
          (int64_t)head * k_out_head_stride;
      k_out[out_base + x_index] = x_rot;
      k_out[out_base + y_index] = y_rot;
    }
  }
}

inline int vec8_block_threads(int hidden) {
  const int vec_count = hidden / 8;
  const int rounded = ((vec_count + 31) / 32) * 32;
  return rounded < 1024 ? rounded : 1024;
}

inline int rmsnorm_block_threads(int rows, int hidden) {
  if (hidden <= 512) {
    return 64;
  }
  if (hidden <= 4096) {
    if (rows <= 16) {
      const int threads = vec8_block_threads(hidden);
      return threads < 512 ? threads : 512;
    }
    if (rows <= 256) {
      const int threads = vec8_block_threads(hidden);
      return threads < 256 ? threads : 256;
    }
    return 128;
  }
  if (hidden <= 8192) {
    const int threads = vec8_block_threads(hidden);
    return threads < 512 ? threads : 512;
  }
  const int threads = vec8_block_threads(hidden);
  return threads < 896 ? threads : 896;
}

inline int fused_block_threads(int hidden) {
  const int threads = vec8_block_threads(hidden);
  return threads;
}

inline int cached_vec8_shared_bytes(int hidden, int block_threads, int cache_hidden_limit) {
  const int reduce_floats = (block_threads + 31) / 32;
  const int cached_floats = hidden <= cache_hidden_limit ? hidden : 0;
  return (cached_floats + reduce_floats) * static_cast<int>(sizeof(float));
}

void check_rmsnorm_inputs(ffi::TensorView input, ffi::TensorView weight, ffi::TensorView out) {
  CHECK_MUSA(input);
  CHECK_MUSA(weight);
  CHECK_MUSA(out);
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input.stride(1), 1);
  TVM_FFI_ICHECK_EQ(out.stride(1), 1);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(weight.stride(0), 1);
  TVM_FFI_ICHECK_EQ(input.size(0), out.size(0));
  TVM_FFI_ICHECK_EQ(input.size(1), out.size(1));
  TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
  TVM_FFI_ICHECK_GE(input.stride(0), input.size(1));
  TVM_FFI_ICHECK_GE(out.stride(0), out.size(1));
  TVM_FFI_ICHECK_EQ(input.device().device_id, weight.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, out.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), weight.dtype()));
}

void check_fused_inputs(ffi::TensorView input, ffi::TensorView residual, ffi::TensorView weight) {
  CHECK_MUSA(input);
  CHECK_MUSA(residual);
  CHECK_MUSA(weight);
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  TVM_FFI_ICHECK_EQ(residual.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input.stride(1), 1);
  TVM_FFI_ICHECK_EQ(residual.stride(1), 1);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(weight.stride(0), 1);
  TVM_FFI_ICHECK_EQ(input.size(0), residual.size(0));
  TVM_FFI_ICHECK_EQ(input.size(1), residual.size(1));
  TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
  TVM_FFI_ICHECK_GE(input.stride(0), input.size(1));
  TVM_FFI_ICHECK_GE(residual.stride(0), residual.size(1));
  TVM_FFI_ICHECK_EQ(input.device().device_id, residual.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, weight.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), residual.dtype()));
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), weight.dtype()));
}

void check_fused_qk_mrope_inputs(ffi::TensorView q, ffi::TensorView k,
                                 ffi::TensorView q_weight,
                                 ffi::TensorView k_weight,
                                 ffi::TensorView positions,
                                 ffi::TensorView cos_sin_cache,
                                 ffi::TensorView q_out, ffi::TensorView k_out) {
  CHECK_MUSA(q);
  CHECK_MUSA(k);
  CHECK_MUSA(q_weight);
  CHECK_MUSA(k_weight);
  CHECK_MUSA(positions);
  CHECK_MUSA(cos_sin_cache);
  CHECK_MUSA(q_out);
  CHECK_MUSA(k_out);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q_out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k_out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(positions.ndim(), 2);
  TVM_FFI_ICHECK_EQ(positions.size(0), 3);
  TVM_FFI_ICHECK_EQ(positions.size(1), q.size(0));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
  TVM_FFI_ICHECK_EQ(q.size(2) % 2, 0);
  TVM_FFI_ICHECK_EQ(q_weight.size(0), q.size(2));
  TVM_FFI_ICHECK_EQ(k_weight.size(0), k.size(2));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1), q.size(2));
  TVM_FFI_ICHECK_EQ(q_out.size(0), q.size(0));
  TVM_FFI_ICHECK_EQ(q_out.size(1), q.size(1));
  TVM_FFI_ICHECK_EQ(q_out.size(2), q.size(2));
  TVM_FFI_ICHECK_EQ(k_out.size(0), k.size(0));
  TVM_FFI_ICHECK_EQ(k_out.size(1), k.size(1));
  TVM_FFI_ICHECK_EQ(k_out.size(2), k.size(2));
  TVM_FFI_ICHECK_EQ(q.stride(2), 1);
  TVM_FFI_ICHECK_EQ(k.stride(2), 1);
  TVM_FFI_ICHECK_EQ(q_out.stride(2), 1);
  TVM_FFI_ICHECK_EQ(k_out.stride(2), 1);
  TVM_FFI_ICHECK_EQ(cos_sin_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, q_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, positions.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, cos_sin_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, q_out.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k_out.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), dl_bfloat16));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), cos_sin_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
}

void check_fused_qk_mrope_cache_inputs(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices) {
  CHECK_MUSA(q);
  CHECK_MUSA(k);
  CHECK_MUSA(v);
  CHECK_MUSA(q_weight);
  CHECK_MUSA(k_weight);
  CHECK_MUSA(positions);
  CHECK_MUSA(cos_sin_cache);
  CHECK_MUSA(q_out);
  CHECK_MUSA(k_cache);
  CHECK_MUSA(v_cache);
  CHECK_MUSA(indices);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k.ndim(), 3);
  TVM_FFI_ICHECK_EQ(v.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q_out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(v_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(positions.ndim(), 2);
  TVM_FFI_ICHECK_EQ(indices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(positions.size(0), 3);
  TVM_FFI_ICHECK_EQ(positions.size(1), q.size(0));
  TVM_FFI_ICHECK_EQ(indices.size(0), q.size(0));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(k.size(1), v.size(1));
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
  TVM_FFI_ICHECK_EQ(k.size(2), v.size(2));
  TVM_FFI_ICHECK_EQ(q.size(2) % 2, 0);
  TVM_FFI_ICHECK_EQ(q_weight.size(0), q.size(2));
  TVM_FFI_ICHECK_EQ(k_weight.size(0), k.size(2));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1), q.size(2));
  TVM_FFI_ICHECK_EQ(k_cache.size(1), k.size(1) * k.size(2));
  TVM_FFI_ICHECK_EQ(v_cache.size(1), v.size(1) * v.size(2));
  TVM_FFI_ICHECK_EQ(q_out.size(0), q.size(0));
  TVM_FFI_ICHECK_EQ(q_out.size(1), q.size(1));
  TVM_FFI_ICHECK_EQ(q_out.size(2), q.size(2));
  TVM_FFI_ICHECK_EQ(q.stride(2), 1);
  TVM_FFI_ICHECK_EQ(k.stride(2), 1);
  TVM_FFI_ICHECK_EQ(v.stride(2), 1);
  TVM_FFI_ICHECK_EQ(q_out.stride(2), 1);
  TVM_FFI_ICHECK_EQ(k_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(v_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(cos_sin_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, v.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, q_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, positions.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, cos_sin_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, q_out.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, v_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, indices.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), dl_bfloat16));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), v.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), cos_sin_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), v_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(indices.dtype(), dl_int32) ||
                 dtype_equal(indices.dtype(), dl_int64));
}

void check_store_cache_inputs(ffi::TensorView k, ffi::TensorView v,
                              ffi::TensorView k_cache,
                              ffi::TensorView v_cache,
                              ffi::TensorView indices) {
  CHECK_MUSA(k);
  CHECK_MUSA(v);
  CHECK_MUSA(k_cache);
  CHECK_MUSA(v_cache);
  CHECK_MUSA(indices);
  TVM_FFI_ICHECK_EQ(k.ndim(), 2);
  TVM_FFI_ICHECK_EQ(v.ndim(), 2);
  TVM_FFI_ICHECK_EQ(k_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(v_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(indices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(k.size(0), v.size(0));
  TVM_FFI_ICHECK_EQ(k.size(0), indices.size(0));
  TVM_FFI_ICHECK_EQ(k.size(1), v.size(1));
  TVM_FFI_ICHECK_EQ(k_cache.size(1), k.size(1));
  TVM_FFI_ICHECK_EQ(v_cache.size(1), v.size(1));
  TVM_FFI_ICHECK_EQ(k.stride(1), 1);
  TVM_FFI_ICHECK_EQ(v.stride(1), 1);
  TVM_FFI_ICHECK_EQ(k_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(v_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(k.device().device_id, v.device().device_id);
  TVM_FFI_ICHECK_EQ(k.device().device_id, k_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(k.device().device_id, v_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(k.device().device_id, indices.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(k.dtype(), dl_bfloat16));
  TVM_FFI_ICHECK(dtype_equal(k.dtype(), v.dtype()));
  TVM_FFI_ICHECK(dtype_equal(k.dtype(), k_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(k.dtype(), v_cache.dtype()));
  TVM_FFI_ICHECK_EQ((k.size(1) * k.dtype().bits) % 32, 0);
  TVM_FFI_ICHECK(dtype_equal(indices.dtype(), dl_int32) ||
                 dtype_equal(indices.dtype(), dl_int64));
}

template <typename T, bool GEMMA>
void launch_rmsnorm(ffi::TensorView input, ffi::TensorView weight, ffi::TensorView out, float eps) {
  const int rows = static_cast<int>(input.size(0));
  const int hidden = static_cast<int>(input.size(1));
  const int64_t input_row_stride = static_cast<int64_t>(input.stride(0));
  const int64_t out_row_stride = static_cast<int64_t>(out.stride(0));
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  musaStream_t stream = get_stream(input.device());

  if ((hidden % 8) == 0 && hidden <= 32768) {
    if (rows <= 16 && hidden == 1024) {
      constexpr int threads = 128;
      constexpr int smem_bytes = 4 * static_cast<int>(sizeof(float));
      rmsnorm_small_h_one_vec_register_kernel<T, GEMMA, 1024, 4><<<rows, threads, smem_bytes, stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          out_row_stride,
          inv_hidden,
          eps);
      return;
    }
    if (rows <= 16 && hidden == 2048) {
      constexpr int threads = 256;
      constexpr int smem_bytes = 8 * static_cast<int>(sizeof(float));
      rmsnorm_small_h_one_vec_register_kernel<T, GEMMA, 2048, 8><<<rows, threads, smem_bytes, stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          out_row_stride,
          inv_hidden,
          eps);
      return;
    }
    const int threads = rmsnorm_block_threads(rows, hidden);
    if (hidden <= 8192) {
      rmsnorm_vec8_kernel<T, GEMMA, true><<<rows, threads, cached_vec8_shared_bytes(hidden, threads, 8192), stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          out_row_stride,
          inv_hidden,
          eps);
    } else {
      rmsnorm_vec8_kernel<T, GEMMA, false><<<rows, threads, cached_vec8_shared_bytes(hidden, threads, 8192), stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          out_row_stride,
          inv_hidden,
          eps);
    }
  } else {
    constexpr int threads = 256;
    constexpr int smem_bytes = ((threads + 31) / 32) * static_cast<int>(sizeof(float));
    rmsnorm_scalar_kernel<T, GEMMA><<<rows, threads, smem_bytes, stream>>>(
        static_cast<const T*>(input.data_ptr()),
        static_cast<const T*>(weight.data_ptr()),
        static_cast<T*>(out.data_ptr()),
        rows,
        hidden,
        input_row_stride,
        out_row_stride,
        inv_hidden,
        eps);
  }
}

template <typename T, bool GEMMA>
void launch_fused_add_rmsnorm(ffi::TensorView input, ffi::TensorView residual, ffi::TensorView weight, float eps) {
  const int rows = static_cast<int>(input.size(0));
  const int hidden = static_cast<int>(input.size(1));
  const int64_t input_row_stride = static_cast<int64_t>(input.stride(0));
  const int64_t residual_row_stride = static_cast<int64_t>(residual.stride(0));
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  musaStream_t stream = get_stream(input.device());

  if ((hidden % 8) == 0 && hidden <= 32768) {
    const int threads = fused_block_threads(hidden);
    if (hidden <= 32768) {
      fused_add_rmsnorm_vec8_kernel<T, GEMMA, true><<<rows, threads, cached_vec8_shared_bytes(hidden, threads, 32768), stream>>>(
          static_cast<T*>(input.data_ptr()),
          static_cast<T*>(residual.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          residual_row_stride,
          inv_hidden,
          eps);
    } else {
      fused_add_rmsnorm_vec8_kernel<T, GEMMA, false><<<rows, threads, cached_vec8_shared_bytes(hidden, threads, 32768), stream>>>(
          static_cast<T*>(input.data_ptr()),
          static_cast<T*>(residual.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          residual_row_stride,
          inv_hidden,
          eps);
    }
  } else {
    constexpr int threads = 256;
    constexpr int smem_bytes = ((threads + 31) / 32) * static_cast<int>(sizeof(float));
    fused_add_rmsnorm_scalar_kernel<T, GEMMA><<<rows, threads, smem_bytes, stream>>>(
        static_cast<T*>(input.data_ptr()),
        static_cast<T*>(residual.data_ptr()),
        static_cast<const T*>(weight.data_ptr()),
        rows,
        hidden,
        input_row_stride,
        residual_row_stride,
        inv_hidden,
        eps);
  }
}

void launch_qk_mrope_kernel(ffi::TensorView q, ffi::TensorView k,
                            ffi::TensorView q_weight,
                            ffi::TensorView k_weight,
                            ffi::TensorView positions,
                            ffi::TensorView cos_sin_cache,
                            ffi::TensorView q_out, ffi::TensorView k_out,
                            int batch, float eps, musaStream_t stream) {
  constexpr int threads = 256;
  constexpr int groups_per_token = 5;
  fused_qk_rmsnorm_mrope_qwen3vl_bf16_kernel
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)), eps);
}

void launch_fused_qk_rmsnorm_mrope_qwen3vl_bf16(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView positions,
    ffi::TensorView cos_sin_cache, ffi::TensorView q_out, ffi::TensorView k_out,
    int mrope_section_t, int mrope_section_h, int mrope_section_w,
    bool is_interleaved, float eps) {
  const int batch = static_cast<int>(q.size(0));
  if (batch == 0) {
    return;
  }
  musaStream_t stream = get_stream(q.device());
  const int q_heads = static_cast<int>(q.size(1));
  const int k_heads = static_cast<int>(k.size(1));
  const int hidden = static_cast<int>(q.size(2));
  const bool qwen3vl_shape =
      q_heads == 32 && k_heads == 4 && hidden == 128 &&
      mrope_section_t == 24 && mrope_section_h == 20 &&
      mrope_section_w == 20;
  if (!qwen3vl_shape || !is_interleaved) {
    constexpr int threads = 256;
    constexpr int heads_per_block = 8;
    const int groups_per_token =
        (q_heads + k_heads + heads_per_block - 1) / heads_per_block;
    fused_qk_rmsnorm_mrope_generic_bf16_kernel<int32_t, false>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(nullptr),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(nullptr),
            static_cast<__mt_bfloat16 *>(nullptr),
            static_cast<const int32_t *>(nullptr), batch, q_heads, k_heads,
            hidden, static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(0), static_cast<int64_t>(0),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)),
            static_cast<int64_t>(k_out.stride(0)),
            static_cast<int64_t>(k_out.stride(1)), static_cast<int64_t>(0),
            static_cast<int64_t>(0), static_cast<int64_t>(0), mrope_section_t,
            mrope_section_h, mrope_section_w, is_interleaved, eps);
    return;
  }
  launch_qk_mrope_kernel(q, k, q_weight, k_weight, positions, cos_sin_cache,
                         q_out, k_out, batch, eps, stream);
}

template <typename index_t>
void launch_qk_mrope_cache_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, int batch, float eps, musaStream_t stream) {
  constexpr int threads = 256;
  constexpr int groups_per_token = 5;
  fused_qk_rmsnorm_mrope_cache_qwen3vl_bf16_kernel<index_t>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_cache.stride(0)),
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)), eps);
}

template <typename index_t>
void launch_fused_qk_rmsnorm_mrope_cache_qwen3vl_bf16(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, int mrope_section_t, int mrope_section_h,
    int mrope_section_w, bool is_interleaved, float eps) {
  const int batch = static_cast<int>(q.size(0));
  if (batch == 0) {
    return;
  }
  musaStream_t stream = get_stream(q.device());
  const int q_heads = static_cast<int>(q.size(1));
  const int k_heads = static_cast<int>(k.size(1));
  const int hidden = static_cast<int>(q.size(2));
  const bool qwen3vl_shape =
      q_heads == 32 && k_heads == 4 && hidden == 128 &&
      mrope_section_t == 24 && mrope_section_h == 20 &&
      mrope_section_w == 20;
  if (!qwen3vl_shape || !is_interleaved) {
    constexpr int threads = 256;
    constexpr int heads_per_block = 8;
    const int groups_per_token =
        (q_heads + k_heads + heads_per_block - 1) / heads_per_block;
    fused_qk_rmsnorm_mrope_generic_bf16_kernel<index_t, true>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(v.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(nullptr),
            static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
            static_cast<const index_t *>(indices.data_ptr()), batch, q_heads,
            k_heads, hidden, static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)), static_cast<int64_t>(0),
            static_cast<int64_t>(0), static_cast<int64_t>(k_cache.stride(0)),
            static_cast<int64_t>(v_cache.stride(0)),
            static_cast<int64_t>(indices.stride(0)), mrope_section_t,
            mrope_section_h, mrope_section_w, is_interleaved, eps);
    return;
  }
  launch_qk_mrope_cache_kernel<index_t>(
      q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
      v_cache, indices, batch, eps, stream);
}

template <typename index_t>
void launch_store_cache(ffi::TensorView k, ffi::TensorView v,
                        ffi::TensorView k_cache, ffi::TensorView v_cache,
                        ffi::TensorView indices) {
  const int64_t num_tokens = k.size(0);
  if (num_tokens == 0) {
    return;
  }

  const int64_t dtype_bytes = k.dtype().bits / 8;
  const int64_t row_bytes = k.size(1) * dtype_bytes;
  const int64_t k_stride_bytes = static_cast<int64_t>(k.stride(0)) * dtype_bytes;
  const int64_t v_stride_bytes = static_cast<int64_t>(v.stride(0)) * dtype_bytes;
  const int64_t k_cache_stride_bytes =
      static_cast<int64_t>(k_cache.stride(0)) * dtype_bytes;
  const int64_t v_cache_stride_bytes =
      static_cast<int64_t>(v_cache.stride(0)) * dtype_bytes;
  const int64_t indices_stride = static_cast<int64_t>(indices.stride(0));
  constexpr int threads = 64;
  musaStream_t stream = get_stream(k.device());

  const uintptr_t ptr_or = reinterpret_cast<uintptr_t>(k.data_ptr()) |
                           reinterpret_cast<uintptr_t>(v.data_ptr()) |
                           reinterpret_cast<uintptr_t>(k_cache.data_ptr()) |
                           reinterpret_cast<uintptr_t>(v_cache.data_ptr());
  if ((row_bytes % 16 == 0) &&
      (((ptr_or | k_stride_bytes | v_stride_bytes | k_cache_stride_bytes |
         v_cache_stride_bytes) &
        0xF) == 0)) {
    store_kv_cache_kernel<index_t, uint4><<<num_tokens, threads, 0, stream>>>(
        static_cast<const char *>(k.data_ptr()),
        static_cast<const char *>(v.data_ptr()),
        static_cast<char *>(k_cache.data_ptr()),
        static_cast<char *>(v_cache.data_ptr()),
        static_cast<const index_t *>(indices.data_ptr()), k_stride_bytes,
        v_stride_bytes, k_cache_stride_bytes, v_cache_stride_bytes,
        indices_stride, row_bytes, num_tokens);
  } else {
    store_kv_cache_kernel<index_t, uint32_t><<<num_tokens, threads, 0, stream>>>(
        static_cast<const char *>(k.data_ptr()),
        static_cast<const char *>(v.data_ptr()),
        static_cast<char *>(k_cache.data_ptr()),
        static_cast<char *>(v_cache.data_ptr()),
        static_cast<const index_t *>(indices.data_ptr()), k_stride_bytes,
        v_stride_bytes, k_cache_stride_bytes, v_cache_stride_bytes,
        indices_stride, row_bytes, num_tokens);
  }
}

void sgl_musa_rmsnorm(ffi::TensorView input, ffi::TensorView weight, ffi::TensorView out, double eps, bool gemma) {
  check_rmsnorm_inputs(input, weight, out);
  ffi::MUSADeviceGuard device_guard(input.device().device_id);
  if (dtype_equal(input.dtype(), dl_float16)) {
    if (gemma) {
      launch_rmsnorm<half, true>(input, weight, out, static_cast<float>(eps));
    } else {
      launch_rmsnorm<half, false>(input, weight, out, static_cast<float>(eps));
    }
  } else if (dtype_equal(input.dtype(), dl_bfloat16)) {
    if (gemma) {
      launch_rmsnorm<__mt_bfloat16, true>(input, weight, out, static_cast<float>(eps));
    } else {
      launch_rmsnorm<__mt_bfloat16, false>(input, weight, out, static_cast<float>(eps));
    }
  } else {
    TVM_FFI_THROW(ValueError) << "sgl_musa_rmsnorm only supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA rmsnorm kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_fused_add_rmsnorm(
    ffi::TensorView input, ffi::TensorView residual, ffi::TensorView weight, double eps, bool gemma) {
  check_fused_inputs(input, residual, weight);
  ffi::MUSADeviceGuard device_guard(input.device().device_id);
  if (dtype_equal(input.dtype(), dl_float16)) {
    if (gemma) {
      launch_fused_add_rmsnorm<half, true>(input, residual, weight, static_cast<float>(eps));
    } else {
      launch_fused_add_rmsnorm<half, false>(input, residual, weight, static_cast<float>(eps));
    }
  } else if (dtype_equal(input.dtype(), dl_bfloat16)) {
    if (gemma) {
      launch_fused_add_rmsnorm<__mt_bfloat16, true>(input, residual, weight, static_cast<float>(eps));
    } else {
      launch_fused_add_rmsnorm<__mt_bfloat16, false>(input, residual, weight, static_cast<float>(eps));
    }
  } else {
    TVM_FFI_THROW(ValueError) << "sgl_musa_fused_add_rmsnorm only supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA fused_add_rmsnorm kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_fused_qk_rmsnorm_mrope(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView positions,
    ffi::TensorView cos_sin_cache, ffi::TensorView q_out, ffi::TensorView k_out,
    bool is_neox, int mrope_section_t, int mrope_section_h, int mrope_section_w,
    bool is_interleaved, double eps) {
  check_fused_qk_mrope_inputs(q, k, q_weight, k_weight, positions,
                              cos_sin_cache, q_out, k_out);
  TVM_FFI_ICHECK(is_neox) << "Qwen3-VL fused MRoPE path requires NeoX style";
  TVM_FFI_ICHECK_EQ(mrope_section_t + mrope_section_h + mrope_section_w,
                    q.size(2) / 2);
  ffi::MUSADeviceGuard device_guard(q.device().device_id);
  launch_fused_qk_rmsnorm_mrope_qwen3vl_bf16(
      q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
      mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved,
      static_cast<float>(eps));
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA fused_qk_rmsnorm_mrope kernel failed: "
      << musaGetErrorString(err);
}

void sgl_musa_fused_qk_rmsnorm_mrope_cache(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, bool is_neox, int mrope_section_t,
    int mrope_section_h, int mrope_section_w, bool is_interleaved, double eps) {
  check_fused_qk_mrope_cache_inputs(q, k, v, q_weight, k_weight, positions,
                                    cos_sin_cache, q_out, k_cache, v_cache,
                                    indices);
  TVM_FFI_ICHECK(is_neox)
      << "Qwen3-VL fused MRoPE cache path requires NeoX style";
  TVM_FFI_ICHECK_EQ(mrope_section_t + mrope_section_h + mrope_section_w,
                    q.size(2) / 2);
  ffi::MUSADeviceGuard device_guard(q.device().device_id);
  if (dtype_equal(indices.dtype(), dl_int32)) {
    launch_fused_qk_rmsnorm_mrope_cache_qwen3vl_bf16<int32_t>(
        q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
        v_cache, indices, mrope_section_t, mrope_section_h, mrope_section_w,
        is_interleaved, static_cast<float>(eps));
  } else if (dtype_equal(indices.dtype(), dl_int64)) {
    launch_fused_qk_rmsnorm_mrope_cache_qwen3vl_bf16<int64_t>(
        q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
        v_cache, indices, mrope_section_t, mrope_section_h, mrope_section_w,
        is_interleaved, static_cast<float>(eps));
  } else {
    TVM_FFI_THROW(ValueError) << "indices must be int32 or int64";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA fused_qk_rmsnorm_mrope_cache kernel failed: "
      << musaGetErrorString(err);
}

void sgl_musa_store_cache(ffi::TensorView k, ffi::TensorView v,
                          ffi::TensorView k_cache, ffi::TensorView v_cache,
                          ffi::TensorView indices) {
  check_store_cache_inputs(k, v, k_cache, v_cache, indices);
  ffi::MUSADeviceGuard device_guard(k.device().device_id);
  if (dtype_equal(indices.dtype(), dl_int32)) {
    launch_store_cache<int32_t>(k, v, k_cache, v_cache, indices);
  } else {
    launch_store_cache<int64_t>(k, v, k_cache, v_cache, indices);
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA store_cache kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_rmsnorm, sgl_musa_rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_add_rmsnorm, sgl_musa_fused_add_rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_qk_rmsnorm_mrope,
                              sgl_musa_fused_qk_rmsnorm_mrope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_qk_rmsnorm_mrope_cache,
                              sgl_musa_fused_qk_rmsnorm_mrope_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_store_cache, sgl_musa_store_cache);
