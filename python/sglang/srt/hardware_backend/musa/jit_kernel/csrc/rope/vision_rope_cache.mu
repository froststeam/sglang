#include <algorithm>
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

__device__ __forceinline__ void vision_rope_pair_fp32_bf16(
    const __mt_bfloat162 c, const __mt_bfloat162 s, const __mt_bfloat162 x,
    const __mt_bfloat162 y, __mt_bfloat162 &out_x, __mt_bfloat162 &out_y) {
  const float2 cf = __bfloat1622float2(c);
  const float2 sf = __bfloat1622float2(s);
  const float2 xf = __bfloat1622float2(x);
  const float2 yf = __bfloat1622float2(y);
  float2 oxf, oyf;
  oxf.x = xf.x * cf.x - yf.x * sf.x;
  oxf.y = xf.y * cf.y - yf.y * sf.y;
  oyf.x = yf.x * cf.x + xf.x * sf.x;
  oyf.y = yf.y * cf.y + xf.y * sf.y;
  out_x = __float22bfloat162_rn(oxf);
  out_y = __float22bfloat162_rn(oyf);
}

__device__ __forceinline__ void vision_rope_pair_fp16(
    const half2 c, const half2 s, const half2 x, const half2 y, half2 &out_x,
    half2 &out_y) {
  out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  out_y = __hfma2(x, s, __hmul2(y, c));
}

template <typename scalar_t>
__global__ void qwen3vl_vision_rope_cache_kernel(
    const int64_t *__restrict__ pos_ids, const scalar_t *__restrict__ cos,
    const scalar_t *__restrict__ sin, scalar_t *__restrict__ cos_sin_cache,
    int64_t *__restrict__ positions, int64_t num_tokens, int64_t half_dim,
    int64_t pos_stride0, int64_t pos_stride1, int64_t cos_stride0,
    int64_t cos_stride1, int64_t sin_stride0, int64_t sin_stride1) {
  const int64_t row_width = half_dim * 4;
  const int64_t total = num_tokens * row_width;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t token = idx / row_width;
    const int64_t col = idx - token * row_width;
    const bool use_sin = col >= half_dim * 2;
    const int64_t local_col = use_sin ? col - half_dim * 2 : col;
    const int64_t axis = local_col / half_dim;
    const int64_t dim = local_col - axis * half_dim;
    const int64_t pos = pos_ids[token * pos_stride0 + axis * pos_stride1];
    cos_sin_cache[idx] =
        use_sin ? sin[pos * sin_stride0 + dim * sin_stride1]
                : cos[pos * cos_stride0 + dim * cos_stride1];
  }

  for (int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < num_tokens;
       token += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    positions[token] = token;
  }
}

__global__ void qwen3vl_vision_rope_cache_u16_vec8_flat_kernel(
    const int64_t *__restrict__ pos_ids, const uint16_t *__restrict__ cos,
    const uint16_t *__restrict__ sin, uint16_t *__restrict__ cos_sin_cache,
    int64_t *__restrict__ positions, int64_t num_tokens, int64_t half_dim,
    int64_t pos_stride0, int64_t pos_stride1, int64_t cos_stride0,
    int64_t sin_stride0) {
  const int64_t vecs_per_axis = half_dim / 8;
  const int64_t total_vecs = vecs_per_axis * 4;
  const int64_t total = num_tokens * total_vecs;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t token = idx / total_vecs;
    const int64_t vec = idx - token * total_vecs;
    if (vec == 0) {
      positions[token] = token;
    }
    const int64_t group = vec / vecs_per_axis;
    const int64_t dim_vec = (vec - group * vecs_per_axis) * 8;
    const int64_t pos_h = pos_ids[token * pos_stride0];
    const int64_t pos_w = pos_ids[token * pos_stride0 + pos_stride1];
    const uint16_t *src;
    if (group == 0) {
      src = cos + pos_h * cos_stride0 + dim_vec;
    } else if (group == 1) {
      src = cos + pos_w * cos_stride0 + dim_vec;
    } else if (group == 2) {
      src = sin + pos_h * sin_stride0 + dim_vec;
    } else {
      src = sin + pos_w * sin_stride0 + dim_vec;
    }
    reinterpret_cast<uint4 *>(cos_sin_cache)[idx] =
        *reinterpret_cast<const uint4 *>(src);
  }
}

void sgl_qwen3vl_vision_rope_cache(ffi::TensorView pos_ids,
                                   ffi::TensorView cos, ffi::TensorView sin,
                                   ffi::TensorView cos_sin_cache,
                                   ffi::TensorView positions) {
  CHECK_MUSA(pos_ids);
  CHECK_MUSA(cos);
  CHECK_MUSA(sin);
  CHECK_MUSA_CONTIGUOUS(cos_sin_cache);
  CHECK_MUSA_CONTIGUOUS(positions);
  TVM_FFI_ICHECK_EQ(pos_ids.ndim(), 2);
  TVM_FFI_ICHECK_EQ(pos_ids.size(1), 2);
  TVM_FFI_ICHECK_EQ(cos.ndim(), 2);
  TVM_FFI_ICHECK_EQ(sin.ndim(), 2);
  TVM_FFI_ICHECK_EQ(cos.size(1), sin.size(1));
  TVM_FFI_ICHECK_EQ(pos_ids.size(0), cos_sin_cache.size(0));
  TVM_FFI_ICHECK_EQ(pos_ids.size(0), positions.size(0));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1), cos.size(1) * 4);
  TVM_FFI_ICHECK(dtype_equal(pos_ids.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(cos.dtype(), sin.dtype()));
  TVM_FFI_ICHECK(dtype_equal(cos.dtype(), cos_sin_cache.dtype()));

  ffi::MUSADeviceGuard device_guard(cos_sin_cache.device().device_id);
  const int64_t num_tokens = pos_ids.size(0);
  if (num_tokens == 0) {
    return;
  }
  const int64_t half_dim = cos.size(1);
  constexpr int threads = 256;
  const int64_t total = num_tokens * half_dim * 4;
  const int blocks =
      static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 4096));
  musaStream_t stream = get_stream(cos_sin_cache.device());

  if ((dtype_equal(cos.dtype(), dl_bfloat16) ||
       dtype_equal(cos.dtype(), dl_float16)) &&
      half_dim % 8 == 0 && cos.stride(1) == 1 && sin.stride(1) == 1 &&
      cos_sin_cache.stride(1) == 1) {
    const int64_t total_vecs = num_tokens * (half_dim / 8) * 4;
    const int vec_blocks =
        static_cast<int>(std::min<int64_t>((total_vecs + threads - 1) / threads, 4096));
    qwen3vl_vision_rope_cache_u16_vec8_flat_kernel<<<vec_blocks, threads, 0, stream>>>(
        static_cast<const int64_t *>(pos_ids.data_ptr()),
        static_cast<const uint16_t *>(cos.data_ptr()),
        static_cast<const uint16_t *>(sin.data_ptr()),
        static_cast<uint16_t *>(cos_sin_cache.data_ptr()),
        static_cast<int64_t *>(positions.data_ptr()), num_tokens, half_dim,
        pos_ids.stride(0), pos_ids.stride(1), cos.stride(0), sin.stride(0));
  } else if (dtype_equal(cos.dtype(), dl_bfloat16) ||
             dtype_equal(cos.dtype(), dl_float16)) {
    qwen3vl_vision_rope_cache_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
        static_cast<const int64_t *>(pos_ids.data_ptr()),
        static_cast<const uint16_t *>(cos.data_ptr()),
        static_cast<const uint16_t *>(sin.data_ptr()),
        static_cast<uint16_t *>(cos_sin_cache.data_ptr()),
        static_cast<int64_t *>(positions.data_ptr()), num_tokens, half_dim,
        pos_ids.stride(0), pos_ids.stride(1), cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1));
  } else if (dtype_equal(cos.dtype(), dl_float32)) {
    qwen3vl_vision_rope_cache_kernel<float><<<blocks, threads, 0, stream>>>(
        static_cast<const int64_t *>(pos_ids.data_ptr()),
        static_cast<const float *>(cos.data_ptr()),
        static_cast<const float *>(sin.data_ptr()),
        static_cast<float *>(cos_sin_cache.data_ptr()),
        static_cast<int64_t *>(positions.data_ptr()), num_tokens, half_dim,
        pos_ids.stride(0), pos_ids.stride(1), cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1));
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported dtype for vision rope cache";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "qwen3vl vision rope cache kernel failed: " << musaGetErrorString(err);
}

template <typename scalar_t>
__global__ void vision_qkv_unpack_rope_scalar_kernel(
    const scalar_t *__restrict__ qkv, const int64_t *__restrict__ pos_ids,
    const scalar_t *__restrict__ cos, const scalar_t *__restrict__ sin,
    scalar_t *__restrict__ q_out, scalar_t *__restrict__ k_out,
    scalar_t *__restrict__ v_out, int64_t num_tokens, int64_t num_heads,
    int64_t num_kv_heads, int64_t head_size, int64_t qkv_stride0,
    int64_t qkv_stride1, int64_t qkv_stride2, int64_t pos_stride0,
    int64_t pos_stride1, int64_t cos_stride0, int64_t cos_stride1,
    int64_t sin_stride0, int64_t sin_stride1) {
  const int64_t q_size = num_heads * head_size;
  const int64_t kv_size = num_kv_heads * head_size;
  const int64_t row_size = q_size + kv_size + kv_size;
  const int64_t total = num_tokens * row_size;
  const int64_t half_head = head_size / 2;
  const int64_t base_dim = half_head / 2;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t token = idx / row_size;
    const int64_t row_col = idx - token * row_size;

    scalar_t *dst = nullptr;
    int64_t local = row_col;
    int64_t src_col = row_col;
    bool do_rope = false;
    int64_t heads = num_heads;

    if (row_col < q_size) {
      dst = q_out;
      do_rope = true;
    } else if (row_col < q_size + kv_size) {
      dst = k_out;
      local = row_col - q_size;
      src_col = row_col;
      do_rope = true;
      heads = num_kv_heads;
    } else {
      dst = v_out;
      local = row_col - q_size - kv_size;
      src_col = row_col;
      heads = num_kv_heads;
    }

    const int64_t head_idx = local / head_size;
    const int64_t dim = local - head_idx * head_size;
    const int64_t dst_offset = token * heads * head_size + local;
    const int64_t src_base = token * qkv_stride1 + src_col * qkv_stride2;

    if (!do_rope) {
      dst[dst_offset] = qkv[src_base];
      continue;
    }

    const int64_t x_dim = dim < half_head ? dim : dim - half_head;
    const int64_t y_dim = dim < half_head ? dim + half_head : dim;
    const int64_t pair_dim = dim < half_head ? dim : dim - half_head;
    const int64_t axis = pair_dim / base_dim;
    const int64_t axis_dim = pair_dim - axis * base_dim;
    const int64_t pos = pos_ids[token * pos_stride0 + axis * pos_stride1];

    const int64_t section_base = row_col - dim;
    const float x =
        to_float(qkv[token * qkv_stride1 + (section_base + x_dim) * qkv_stride2]);
    const float y =
        to_float(qkv[token * qkv_stride1 + (section_base + y_dim) * qkv_stride2]);
    const float c = to_float(cos[pos * cos_stride0 + axis_dim * cos_stride1]);
    const float s = to_float(sin[pos * sin_stride0 + axis_dim * sin_stride1]);
    const float out = dim < half_head ? (x * c - y * s) : (y * c + x * s);
    dst[dst_offset] = from_float<scalar_t>(out);
  }
}

template <typename scalar_t, typename scalar2_t>
__global__ void vision_qkv_unpack_rope_vec2_kernel(
    const scalar_t *__restrict__ qkv, const int64_t *__restrict__ pos_ids,
    const scalar_t *__restrict__ cos, const scalar_t *__restrict__ sin,
    scalar_t *__restrict__ q_out, scalar_t *__restrict__ k_out,
    scalar_t *__restrict__ v_out, int64_t num_tokens, int64_t num_heads,
    int64_t num_kv_heads, int64_t head_size, int64_t qkv_stride1,
    int64_t pos_stride0, int64_t pos_stride1, int64_t cos_stride0,
    int64_t sin_stride0) {
  const int64_t q_size = num_heads * head_size;
  const int64_t kv_size = num_kv_heads * head_size;
  const int64_t half_head = head_size / 2;
  const int64_t base_dim = half_head / 2;
  const int64_t rope_vecs_per_token =
      (num_heads + num_kv_heads) * (half_head / 2);
  const int64_t rope_total = num_tokens * rope_vecs_per_token;
  const int64_t v_vecs_per_token = kv_size / 8;
  const int64_t total = rope_total + num_tokens * v_vecs_per_token;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (idx < rope_total) {
      const int64_t token = idx / rope_vecs_per_token;
      const int64_t local_vec = idx - token * rope_vecs_per_token;
      const bool is_k = local_vec >= num_heads * (half_head / 2);
      const int64_t section_vec =
          is_k ? local_vec - num_heads * (half_head / 2) : local_vec;
      const int64_t heads = is_k ? num_kv_heads : num_heads;
      const int64_t head_idx = section_vec / (half_head / 2);
      const int64_t rot = (section_vec - head_idx * (half_head / 2)) * 2;
      if (head_idx >= heads) {
        continue;
      }

      const int64_t axis = rot / base_dim;
      const int64_t axis_dim = rot - axis * base_dim;
      const int64_t pos = pos_ids[token * pos_stride0 + axis * pos_stride1];
      const int64_t src_section = is_k ? q_size : 0;
      const int64_t src_base =
          token * qkv_stride1 + src_section + head_idx * head_size;
      const int64_t dst_base = token * heads * head_size + head_idx * head_size;

      const scalar2_t c =
          *reinterpret_cast<const scalar2_t *>(cos + pos * cos_stride0 + axis_dim);
      const scalar2_t s =
          *reinterpret_cast<const scalar2_t *>(sin + pos * sin_stride0 + axis_dim);
      const scalar2_t x =
          *reinterpret_cast<const scalar2_t *>(qkv + src_base + rot);
      const scalar2_t y =
          *reinterpret_cast<const scalar2_t *>(qkv + src_base + half_head + rot);
      scalar2_t out_x, out_y;
      if constexpr (std::is_same_v<scalar_t, __mt_bfloat16>) {
        vision_rope_pair_fp32_bf16(c, s, x, y, out_x, out_y);
      } else {
        vision_rope_pair_fp16(c, s, x, y, out_x, out_y);
      }

      scalar_t *dst = is_k ? k_out : q_out;
      *reinterpret_cast<scalar2_t *>(dst + dst_base + rot) = out_x;
      *reinterpret_cast<scalar2_t *>(dst + dst_base + half_head + rot) = out_y;
    } else {
      const int64_t v_idx = idx - rope_total;
      const int64_t token = v_idx / v_vecs_per_token;
      const int64_t vec = v_idx - token * v_vecs_per_token;
      const int64_t src = token * qkv_stride1 + q_size + kv_size + vec * 8;
      reinterpret_cast<uint4 *>(v_out)[token * v_vecs_per_token + vec] =
          *reinterpret_cast<const uint4 *>(qkv + src);
    }
  }
}

// Qwen vision heads share one position and rotary table row per token. Group
// several heads under one thread so those loads are reused for both Q and K.
template <int NUM_HEADS, int HEAD_SIZE, int HEADS_PER_THREAD,
          int TOKENS_PER_BLOCK, int TOKEN_THREADS, typename scalar_t,
          typename scalar2_t>
__global__ void vision_qkv_unpack_rope_grouped_kernel(
    const scalar_t *__restrict__ qkv, const int64_t *__restrict__ pos_ids,
    const scalar_t *__restrict__ cos, const scalar_t *__restrict__ sin,
    scalar_t *__restrict__ q_out, scalar_t *__restrict__ k_out,
    scalar_t *__restrict__ v_out, int64_t num_tokens) {
  constexpr int HALF_HEAD = HEAD_SIZE / 2;
  constexpr int BASE_DIM = HALF_HEAD / 2;
  constexpr int PAIRS_PER_HEAD = HALF_HEAD / 2;
  constexpr int HEAD_GROUPS =
      (NUM_HEADS + HEADS_PER_THREAD - 1) / HEADS_PER_THREAD;
  constexpr int WORKERS_PER_TOKEN = HEAD_GROUPS * PAIRS_PER_HEAD;
  constexpr int TOKEN_THREAD_STRIDE =
      TOKEN_THREADS == 0 ? WORKERS_PER_TOKEN : TOKEN_THREADS;
  constexpr int Q_SIZE = NUM_HEADS * HEAD_SIZE;
  constexpr int KV_SIZE = Q_SIZE;
  constexpr int QKV_SIZE = Q_SIZE + KV_SIZE + KV_SIZE;
  constexpr int ROPE_TABLE_SIZE = HEAD_SIZE / 4;
  constexpr int V_VECS_PER_TOKEN = KV_SIZE / 8;

  const int token_in_block = threadIdx.x / TOKEN_THREAD_STRIDE;
  if (token_in_block >= TOKENS_PER_BLOCK) {
    return;
  }
  const int worker = threadIdx.x - token_in_block * TOKEN_THREAD_STRIDE;
  if (worker >= WORKERS_PER_TOKEN) {
    return;
  }
  const int head_group =
      HEAD_GROUPS == 1 ? 0 : worker / PAIRS_PER_HEAD;
  const int pair =
      HEAD_GROUPS == 1 ? worker : worker - head_group * PAIRS_PER_HEAD;
  const int64_t token =
      static_cast<int64_t>(blockIdx.x) * TOKENS_PER_BLOCK + token_in_block;
  if (token >= num_tokens) {
    return;
  }

  const int axis = HEAD_SIZE == 72 ? pair >= 9 : pair >= 8;
  const int axis_dim =
      HEAD_SIZE == 72 ? (pair - axis * 9) * 2 : (pair - axis * 8) * 2;
  const int rot = pair * 2;
  const int64_t pos = pos_ids[token * 2 + axis];
  const scalar2_t c =
      *reinterpret_cast<const scalar2_t *>(cos + pos * ROPE_TABLE_SIZE + axis_dim);
  const scalar2_t s =
      *reinterpret_cast<const scalar2_t *>(sin + pos * ROPE_TABLE_SIZE + axis_dim);
  const int64_t token_src = token * QKV_SIZE;

#pragma unroll
  for (int local_head = 0; local_head < HEADS_PER_THREAD; ++local_head) {
    const int head = head_group * HEADS_PER_THREAD + local_head;
    if (head >= NUM_HEADS) {
      break;
    }
    const int64_t q_src = token_src + head * HEAD_SIZE;
    const int64_t q_dst = token * Q_SIZE + head * HEAD_SIZE;
    scalar2_t x = *reinterpret_cast<const scalar2_t *>(qkv + q_src + rot);
    scalar2_t y =
        *reinterpret_cast<const scalar2_t *>(qkv + q_src + HALF_HEAD + rot);
    scalar2_t out_x, out_y;
    if constexpr (std::is_same_v<scalar_t, __mt_bfloat16>) {
      vision_rope_pair_fp32_bf16(c, s, x, y, out_x, out_y);
    } else {
      vision_rope_pair_fp16(c, s, x, y, out_x, out_y);
    }
    *reinterpret_cast<scalar2_t *>(q_out + q_dst + rot) = out_x;
    *reinterpret_cast<scalar2_t *>(q_out + q_dst + HALF_HEAD + rot) = out_y;

    const int64_t k_src = token_src + Q_SIZE + head * HEAD_SIZE;
    const int64_t k_dst = token * KV_SIZE + head * HEAD_SIZE;
    x = *reinterpret_cast<const scalar2_t *>(qkv + k_src + rot);
    y = *reinterpret_cast<const scalar2_t *>(qkv + k_src + HALF_HEAD + rot);
    if constexpr (std::is_same_v<scalar_t, __mt_bfloat16>) {
      vision_rope_pair_fp32_bf16(c, s, x, y, out_x, out_y);
    } else {
      vision_rope_pair_fp16(c, s, x, y, out_x, out_y);
    }
    *reinterpret_cast<scalar2_t *>(k_out + k_dst + rot) = out_x;
    *reinterpret_cast<scalar2_t *>(k_out + k_dst + HALF_HEAD + rot) = out_y;
  }

  if constexpr (V_VECS_PER_TOKEN == WORKERS_PER_TOKEN) {
    const int64_t v_src = token_src + Q_SIZE + KV_SIZE + worker * 8;
    reinterpret_cast<uint4 *>(v_out)[token * V_VECS_PER_TOKEN + worker] =
        *reinterpret_cast<const uint4 *>(qkv + v_src);
  } else {
    for (int v_vec = worker; v_vec < V_VECS_PER_TOKEN;
         v_vec += WORKERS_PER_TOKEN) {
    const int64_t v_src = token_src + Q_SIZE + KV_SIZE + v_vec * 8;
    reinterpret_cast<uint4 *>(v_out)[token * V_VECS_PER_TOKEN + v_vec] =
        *reinterpret_cast<const uint4 *>(qkv + v_src);
    }
  }
}

template <int NUM_HEADS, int HEAD_SIZE, int HEADS_PER_THREAD,
          int TOKENS_PER_BLOCK, int THREADS = 128, int TOKEN_THREADS = 0>
void launch_vision_qkv_unpack_rope_grouped_bf16(
    ffi::TensorView qkv, ffi::TensorView pos_ids, ffi::TensorView cos,
    ffi::TensorView sin, ffi::TensorView q_out, ffi::TensorView k_out,
    ffi::TensorView v_out, int64_t num_tokens, musaStream_t stream) {
  const int blocks = static_cast<int>(
      (num_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK);
  vision_qkv_unpack_rope_grouped_kernel<
      NUM_HEADS, HEAD_SIZE, HEADS_PER_THREAD, TOKENS_PER_BLOCK,
      TOKEN_THREADS, __mt_bfloat16, __mt_bfloat162>
      <<<blocks, THREADS, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(qkv.data_ptr()),
          static_cast<const int64_t *>(pos_ids.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos.data_ptr()),
          static_cast<const __mt_bfloat16 *>(sin.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_out.data_ptr()), num_tokens);
}

template <int NUM_Q_HEADS, int NUM_KV_HEADS, int HEAD_SIZE, typename scalar_t,
          typename scalar2_t>
__global__ void vision_qkv_unpack_rope_block_kernel(
    const scalar_t *__restrict__ qkv, const int64_t *__restrict__ pos_ids,
    const scalar_t *__restrict__ cos, const scalar_t *__restrict__ sin,
    scalar_t *__restrict__ q_out, scalar_t *__restrict__ k_out,
    scalar_t *__restrict__ v_out, int64_t qkv_stride1, int64_t pos_stride0,
    int64_t pos_stride1, int64_t cos_stride0, int64_t sin_stride0) {
  constexpr int HALF_HEAD = HEAD_SIZE / 2;
  constexpr int BASE_DIM = HALF_HEAD / 2;
  constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_SIZE;
  constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_SIZE;
  constexpr int MAX_HEADS = NUM_Q_HEADS > NUM_KV_HEADS ? NUM_Q_HEADS : NUM_KV_HEADS;
  constexpr int VEC_ROPE_PER_MAX = MAX_HEADS * (HALF_HEAD / 2);
  constexpr int V_VEC_PER_TOKEN = KV_SIZE / 8;

  const int64_t token = blockIdx.x;
  for (int tid = threadIdx.x; tid < VEC_ROPE_PER_MAX; tid += blockDim.x) {
    const int head_idx = tid / (HALF_HEAD / 2);
    const int rot = (tid - head_idx * (HALF_HEAD / 2)) * 2;
    const int axis = rot / BASE_DIM;
    const int axis_dim = rot - axis * BASE_DIM;
    const int64_t pos = pos_ids[token * pos_stride0 + axis * pos_stride1];
    const scalar2_t c =
        *reinterpret_cast<const scalar2_t *>(cos + pos * cos_stride0 + axis_dim);
    const scalar2_t s =
        *reinterpret_cast<const scalar2_t *>(sin + pos * sin_stride0 + axis_dim);

    if (head_idx < NUM_Q_HEADS) {
      const int64_t q_src = token * qkv_stride1 + head_idx * HEAD_SIZE;
      const int64_t q_dst = token * Q_SIZE + head_idx * HEAD_SIZE;
      scalar2_t x = *reinterpret_cast<const scalar2_t *>(qkv + q_src + rot);
      scalar2_t y =
          *reinterpret_cast<const scalar2_t *>(qkv + q_src + HALF_HEAD + rot);
      scalar2_t out_x, out_y;
      if constexpr (std::is_same_v<scalar_t, __mt_bfloat16>) {
        vision_rope_pair_fp32_bf16(c, s, x, y, out_x, out_y);
      } else {
        vision_rope_pair_fp16(c, s, x, y, out_x, out_y);
      }
      *reinterpret_cast<scalar2_t *>(q_out + q_dst + rot) = out_x;
      *reinterpret_cast<scalar2_t *>(q_out + q_dst + HALF_HEAD + rot) = out_y;
    }

    if (head_idx < NUM_KV_HEADS) {
      const int64_t k_src = token * qkv_stride1 + Q_SIZE + head_idx * HEAD_SIZE;
      const int64_t k_dst = token * KV_SIZE + head_idx * HEAD_SIZE;
      scalar2_t x = *reinterpret_cast<const scalar2_t *>(qkv + k_src + rot);
      scalar2_t y =
          *reinterpret_cast<const scalar2_t *>(qkv + k_src + HALF_HEAD + rot);
      scalar2_t out_x, out_y;
      if constexpr (std::is_same_v<scalar_t, __mt_bfloat16>) {
        vision_rope_pair_fp32_bf16(c, s, x, y, out_x, out_y);
      } else {
        vision_rope_pair_fp16(c, s, x, y, out_x, out_y);
      }
      *reinterpret_cast<scalar2_t *>(k_out + k_dst + rot) = out_x;
      *reinterpret_cast<scalar2_t *>(k_out + k_dst + HALF_HEAD + rot) = out_y;
    }
  }

  for (int tid = threadIdx.x; tid < V_VEC_PER_TOKEN; tid += blockDim.x) {
    const int64_t src = token * qkv_stride1 + Q_SIZE + KV_SIZE + tid * 8;
    reinterpret_cast<uint4 *>(v_out)[token * V_VEC_PER_TOKEN + tid] =
        *reinterpret_cast<const uint4 *>(qkv + src);
  }
}

template <int NUM_Q_HEADS, int NUM_KV_HEADS, int HEAD_SIZE, typename scalar_t,
          typename scalar2_t>
void launch_vision_qkv_unpack_rope_block(
    ffi::TensorView qkv, ffi::TensorView pos_ids, ffi::TensorView cos,
    ffi::TensorView sin, ffi::TensorView q_out, ffi::TensorView k_out,
    ffi::TensorView v_out, int64_t num_tokens, int64_t qkv_token_stride,
    musaStream_t stream) {
  constexpr int threads = 256;
  vision_qkv_unpack_rope_block_kernel<NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE,
                                      scalar_t, scalar2_t>
      <<<num_tokens, threads, 0, stream>>>(
          static_cast<const scalar_t *>(qkv.data_ptr()),
          static_cast<const int64_t *>(pos_ids.data_ptr()),
          static_cast<const scalar_t *>(cos.data_ptr()),
          static_cast<const scalar_t *>(sin.data_ptr()),
          static_cast<scalar_t *>(q_out.data_ptr()),
          static_cast<scalar_t *>(k_out.data_ptr()),
          static_cast<scalar_t *>(v_out.data_ptr()), qkv_token_stride,
          pos_ids.stride(0), pos_ids.stride(1), cos.stride(0), sin.stride(0));
}

void sgl_vision_qkv_unpack_rope(ffi::TensorView qkv, ffi::TensorView pos_ids,
                                ffi::TensorView cos, ffi::TensorView sin,
                                ffi::TensorView q_out, ffi::TensorView k_out,
                                ffi::TensorView v_out) {
  CHECK_MUSA(qkv);
  CHECK_MUSA(pos_ids);
  CHECK_MUSA(cos);
  CHECK_MUSA(sin);
  CHECK_MUSA_CONTIGUOUS(q_out);
  CHECK_MUSA_CONTIGUOUS(k_out);
  CHECK_MUSA_CONTIGUOUS(v_out);
  TVM_FFI_ICHECK_EQ(qkv.ndim(), 3);
  TVM_FFI_ICHECK(qkv.size(0) == 1 || qkv.size(1) == 1)
      << "qkv must be shaped as [1, tokens, hidden] or [tokens, 1, hidden]";
  TVM_FFI_ICHECK_EQ(pos_ids.ndim(), 2);
  TVM_FFI_ICHECK_EQ(pos_ids.size(1), 2);
  TVM_FFI_ICHECK_EQ(cos.ndim(), 2);
  TVM_FFI_ICHECK_EQ(sin.ndim(), 2);
  TVM_FFI_ICHECK_EQ(q_out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k_out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(v_out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(pos_ids.size(0), q_out.size(0));
  TVM_FFI_ICHECK_EQ(pos_ids.size(0), k_out.size(0));
  TVM_FFI_ICHECK_EQ(pos_ids.size(0), v_out.size(0));
  TVM_FFI_ICHECK_EQ(q_out.size(2), k_out.size(2));
  TVM_FFI_ICHECK_EQ(q_out.size(2), v_out.size(2));
  TVM_FFI_ICHECK_EQ(cos.size(1) * 4, q_out.size(2));
  TVM_FFI_ICHECK_EQ(sin.size(1), cos.size(1));
  TVM_FFI_ICHECK_EQ(qkv.size(2),
                    q_out.size(1) * q_out.size(2) +
                        k_out.size(1) * k_out.size(2) +
                        v_out.size(1) * v_out.size(2));
  TVM_FFI_ICHECK(dtype_equal(pos_ids.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(qkv.dtype(), q_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(qkv.dtype(), k_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(qkv.dtype(), v_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(qkv.dtype(), cos.dtype()));
  TVM_FFI_ICHECK(dtype_equal(qkv.dtype(), sin.dtype()));

  ffi::MUSADeviceGuard device_guard(qkv.device().device_id);
  const int64_t num_tokens = q_out.size(0);
  if (num_tokens == 0) {
    return;
  }
  TVM_FFI_ICHECK_EQ(qkv.size(0) * qkv.size(1), num_tokens);
  constexpr int threads = 256;
  const int64_t row_size = qkv.size(2);
  const int64_t qkv_token_stride = qkv.size(0) == 1 ? qkv.stride(1) : qkv.stride(0);
  const int blocks = static_cast<int>(
      std::min<int64_t>((num_tokens * row_size + threads - 1) / threads, 4096));
  musaStream_t stream = get_stream(qkv.device());
  const bool can_vec2 =
      qkv.stride(2) == 1 && q_out.stride(2) == 1 && k_out.stride(2) == 1 &&
      v_out.stride(2) == 1 && cos.stride(1) == 1 && sin.stride(1) == 1 &&
      q_out.size(2) % 4 == 0 && cos.size(1) % 2 == 0 &&
      (k_out.size(1) * k_out.size(2)) % 8 == 0;
  const bool can_block =
      can_vec2 && q_out.size(1) == k_out.size(1) &&
      k_out.size(1) == v_out.size(1) &&
      ((q_out.size(1) == 12 && q_out.size(2) == 64) ||
       (q_out.size(1) == 16 &&
        (q_out.size(2) == 64 || q_out.size(2) == 72 || q_out.size(2) == 80))) &&
      cos.size(1) * 4 == q_out.size(2);

  const bool use_grouped_bf16 =
      can_vec2 && dtype_equal(qkv.dtype(), dl_bfloat16) &&
      q_out.size(1) == k_out.size(1) && k_out.size(1) == v_out.size(1) &&
      qkv_token_stride == row_size && pos_ids.stride(0) == 2 &&
      pos_ids.stride(1) == 1 && cos.stride(0) == cos.size(1) &&
      sin.stride(0) == sin.size(1) &&
      q_out.size(2) == 72 &&
      (q_out.size(1) == 2 || q_out.size(1) == 4 || q_out.size(1) == 8);

  if (use_grouped_bf16) {
    if (q_out.size(1) == 2) {
      if (num_tokens >= 32768) {
        launch_vision_qkv_unpack_rope_grouped_bf16<2, 72, 2, 8, 256, 32>(
            qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens, stream);
      } else {
        launch_vision_qkv_unpack_rope_grouped_bf16<2, 72, 2, 14, 256>(
            qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens, stream);
      }
    } else if (q_out.size(1) == 4) {
      launch_vision_qkv_unpack_rope_grouped_bf16<4, 72, 2, 3>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens, stream);
    } else {
      launch_vision_qkv_unpack_rope_grouped_bf16<8, 72, 1, 1, 256>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens, stream);
    }
  } else if (can_block && dtype_equal(qkv.dtype(), dl_bfloat16)) {
    if (q_out.size(1) == 12 && q_out.size(2) == 64) {
      launch_vision_qkv_unpack_rope_block<12, 12, 64, __mt_bfloat16,
                                          __mt_bfloat162>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens,
          qkv_token_stride, stream);
    } else if (q_out.size(2) == 64) {
      launch_vision_qkv_unpack_rope_block<16, 16, 64, __mt_bfloat16,
                                          __mt_bfloat162>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens,
          qkv_token_stride, stream);
    } else if (q_out.size(2) == 72) {
      launch_vision_qkv_unpack_rope_block<16, 16, 72, __mt_bfloat16,
                                          __mt_bfloat162>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens,
          qkv_token_stride, stream);
    } else {
      launch_vision_qkv_unpack_rope_block<16, 16, 80, __mt_bfloat16,
                                          __mt_bfloat162>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens,
          qkv_token_stride, stream);
    }
  } else if (can_block && dtype_equal(qkv.dtype(), dl_float16)) {
    if (q_out.size(1) == 12 && q_out.size(2) == 64) {
      launch_vision_qkv_unpack_rope_block<12, 12, 64, half, half2>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens,
          qkv_token_stride, stream);
    } else if (q_out.size(2) == 64) {
      launch_vision_qkv_unpack_rope_block<16, 16, 64, half, half2>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens,
          qkv_token_stride, stream);
    } else if (q_out.size(2) == 72) {
      launch_vision_qkv_unpack_rope_block<16, 16, 72, half, half2>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens,
          qkv_token_stride, stream);
    } else {
      launch_vision_qkv_unpack_rope_block<16, 16, 80, half, half2>(
          qkv, pos_ids, cos, sin, q_out, k_out, v_out, num_tokens,
          qkv_token_stride, stream);
    }
  } else if (can_vec2 && dtype_equal(qkv.dtype(), dl_bfloat16)) {
    const int64_t half_head = q_out.size(2) / 2;
    const int64_t rope_vecs =
        num_tokens * (q_out.size(1) + k_out.size(1)) * (half_head / 2);
    const int64_t v_vecs = num_tokens * (v_out.size(1) * v_out.size(2) / 8);
    const int vec_blocks = static_cast<int>(
        std::min<int64_t>((rope_vecs + v_vecs + threads - 1) / threads, 4096));
    vision_qkv_unpack_rope_vec2_kernel<__mt_bfloat16, __mt_bfloat162>
        <<<vec_blocks, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(qkv.data_ptr()),
            static_cast<const int64_t *>(pos_ids.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos.data_ptr()),
            static_cast<const __mt_bfloat16 *>(sin.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(v_out.data_ptr()), num_tokens,
            q_out.size(1), k_out.size(1), q_out.size(2), qkv_token_stride,
            pos_ids.stride(0), pos_ids.stride(1), cos.stride(0), sin.stride(0));
  } else if (can_vec2 && dtype_equal(qkv.dtype(), dl_float16)) {
    const int64_t half_head = q_out.size(2) / 2;
    const int64_t rope_vecs =
        num_tokens * (q_out.size(1) + k_out.size(1)) * (half_head / 2);
    const int64_t v_vecs = num_tokens * (v_out.size(1) * v_out.size(2) / 8);
    const int vec_blocks = static_cast<int>(
        std::min<int64_t>((rope_vecs + v_vecs + threads - 1) / threads, 4096));
    vision_qkv_unpack_rope_vec2_kernel<half, half2>
        <<<vec_blocks, threads, 0, stream>>>(
            static_cast<const half *>(qkv.data_ptr()),
            static_cast<const int64_t *>(pos_ids.data_ptr()),
            static_cast<const half *>(cos.data_ptr()),
            static_cast<const half *>(sin.data_ptr()),
            static_cast<half *>(q_out.data_ptr()), static_cast<half *>(k_out.data_ptr()),
            static_cast<half *>(v_out.data_ptr()), num_tokens, q_out.size(1),
            k_out.size(1), q_out.size(2), qkv_token_stride, pos_ids.stride(0),
            pos_ids.stride(1), cos.stride(0), sin.stride(0));
  } else if (dtype_equal(qkv.dtype(), dl_bfloat16)) {
    vision_qkv_unpack_rope_scalar_kernel<__mt_bfloat16>
        <<<blocks, threads, 0, stream>>>(
        static_cast<const __mt_bfloat16 *>(qkv.data_ptr()),
        static_cast<const int64_t *>(pos_ids.data_ptr()),
        static_cast<const __mt_bfloat16 *>(cos.data_ptr()),
        static_cast<const __mt_bfloat16 *>(sin.data_ptr()),
        static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
        static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
        static_cast<__mt_bfloat16 *>(v_out.data_ptr()), num_tokens, q_out.size(1),
        k_out.size(1), q_out.size(2), qkv.stride(0), qkv_token_stride,
        qkv.stride(2), pos_ids.stride(0), pos_ids.stride(1), cos.stride(0),
        cos.stride(1), sin.stride(0), sin.stride(1));
  } else if (dtype_equal(qkv.dtype(), dl_float16)) {
    vision_qkv_unpack_rope_scalar_kernel<half><<<blocks, threads, 0, stream>>>(
        static_cast<const half *>(qkv.data_ptr()),
        static_cast<const int64_t *>(pos_ids.data_ptr()),
        static_cast<const half *>(cos.data_ptr()),
        static_cast<const half *>(sin.data_ptr()),
        static_cast<half *>(q_out.data_ptr()), static_cast<half *>(k_out.data_ptr()),
        static_cast<half *>(v_out.data_ptr()), num_tokens, q_out.size(1),
        k_out.size(1), q_out.size(2), qkv.stride(0), qkv_token_stride,
        qkv.stride(2), pos_ids.stride(0), pos_ids.stride(1), cos.stride(0),
        cos.stride(1), sin.stride(0), sin.stride(1));
  } else if (dtype_equal(qkv.dtype(), dl_float32)) {
    vision_qkv_unpack_rope_scalar_kernel<float><<<blocks, threads, 0, stream>>>(
        static_cast<const float *>(qkv.data_ptr()),
        static_cast<const int64_t *>(pos_ids.data_ptr()),
        static_cast<const float *>(cos.data_ptr()),
        static_cast<const float *>(sin.data_ptr()),
        static_cast<float *>(q_out.data_ptr()), static_cast<float *>(k_out.data_ptr()),
        static_cast<float *>(v_out.data_ptr()), num_tokens, q_out.size(1),
        k_out.size(1), q_out.size(2), qkv.stride(0), qkv_token_stride,
        qkv.stride(2), pos_ids.stride(0), pos_ids.stride(1), cos.stride(0),
        cos.stride(1), sin.stride(0), sin.stride(1));
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported dtype for vision qkv unpack rope";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "vision qkv unpack rope kernel failed: " << musaGetErrorString(err);
}

void sgl_qwen3vl_qkv_unpack_rope(ffi::TensorView qkv, ffi::TensorView pos_ids,
                                 ffi::TensorView cos, ffi::TensorView sin,
                                 ffi::TensorView q_out,
                                 ffi::TensorView k_out,
                                 ffi::TensorView v_out) {
  sgl_vision_qkv_unpack_rope(qkv, pos_ids, cos, sin, q_out, k_out, v_out);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_qwen3vl_vision_rope_cache,
                              sgl_qwen3vl_vision_rope_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_vision_qkv_unpack_rope,
                              sgl_vision_qkv_unpack_rope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_qwen3vl_qkv_unpack_rope,
                              sgl_qwen3vl_qkv_unpack_rope);
