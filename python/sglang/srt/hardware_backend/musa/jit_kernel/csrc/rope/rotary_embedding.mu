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

#ifndef SGLANG_MUSA_ROPE_STORE_HINT
#define SGLANG_MUSA_ROPE_STORE_HINT 0
#endif

#ifndef SGLANG_MUSA_ROPE_ARCH_MP31
#define SGLANG_MUSA_ROPE_ARCH_MP31 0
#endif

__device__ __forceinline__ void store_half2(half2 *ptr, half2 value) {
#if SGLANG_MUSA_ROPE_STORE_HINT == 1
  __stwb(ptr, value);
#elif SGLANG_MUSA_ROPE_STORE_HINT == 2
  __stcg(ptr, value);
#elif SGLANG_MUSA_ROPE_STORE_HINT == 3
  __stcs(ptr, value);
#else
  *ptr = value;
#endif
}

__device__ __forceinline__ void store_bfloat162(__mt_bfloat162 *ptr,
                                                __mt_bfloat162 value) {
#if SGLANG_MUSA_ROPE_STORE_HINT == 1
  __stwb(ptr, value);
#elif SGLANG_MUSA_ROPE_STORE_HINT == 2
  __stcg(ptr, value);
#elif SGLANG_MUSA_ROPE_STORE_HINT == 3
  __stcs(ptr, value);
#else
  *ptr = value;
#endif
}

inline bool supported_h2_rot_dim(int rot_dim) {
  return rot_dim == 64 || rot_dim == 128;
}

inline bool even_stride(int stride) { return (stride & 1) == 0; }

inline bool can_use_neox_h2_q(int seq_len, int rot_dim, int query_stride,
                              int query_head_stride, int query_dim_stride) {
  return seq_len == 0 && query_dim_stride == 1 &&
         supported_h2_rot_dim(rot_dim) && even_stride(query_stride) &&
         even_stride(query_head_stride);
}

inline bool can_use_neox_h2_qk(int seq_len, int rot_dim, int query_stride,
                               int key_stride, int query_head_stride,
                               int key_head_stride, int query_dim_stride,
                               int key_dim_stride) {
  return can_use_neox_h2_q(seq_len, rot_dim, query_stride, query_head_stride,
                           query_dim_stride) &&
         key_dim_stride == 1 && even_stride(key_stride) &&
         even_stride(key_head_stride);
}

inline bool is_32x8_hs128_layout(int num_heads, int num_kv_heads,
                                 int query_stride, int key_stride,
                                 int query_head_stride, int key_head_stride) {
  return SGLANG_MUSA_ROPE_ARCH_MP31 && num_heads == 32 && num_kv_heads == 8 &&
         query_stride == 4096 && key_stride == 1024 &&
         query_head_stride == 128 && key_head_stride == 128;
}

inline bool is_hs128_contiguous_layout(int num_heads, int num_kv_heads,
                                       int query_stride, int key_stride,
                                       int query_head_stride,
                                       int key_head_stride) {
  return SGLANG_MUSA_ROPE_ARCH_MP31 && query_head_stride == 128 &&
         key_head_stride == 128 && query_stride == (num_heads << 7) &&
         key_stride == (num_kv_heads << 7);
}

inline int prefill_block_pairs(int num_tokens, int num_heads,
                               int half_rot_dim) {
  int block_pairs = std::min(512, std::max(32, num_heads * half_rot_dim));
  if (num_tokens > 16384) {
    block_pairs = std::min(256, block_pairs);
  }
  return block_pairs;
}

template <typename T, bool IS_NEOX>
__device__ __forceinline__ void
apply_one(T *data, const T *cache, int base, int head_stride, int dim_stride,
          int head_idx, int rot_offset, int half_rot_dim) {
  const int x_idx = IS_NEOX ? rot_offset : 2 * rot_offset;
  const int y_idx = IS_NEOX ? half_rot_dim + rot_offset : x_idx + 1;
  const int head_base = base + head_idx * head_stride;
  const int x_offset = head_base + x_idx * dim_stride;
  const int y_offset = head_base + y_idx * dim_stride;

  const float x = to_float(data[x_offset]);
  const float y = to_float(data[y_offset]);
  const float c = to_float(cache[rot_offset]);
  const float s = to_float(cache[half_rot_dim + rot_offset]);
  data[x_offset] = from_float<T>(x * c - y * s);
  data[y_offset] = from_float<T>(y * c + x * s);
}

template <typename T, bool IS_NEOX, bool HAS_KEY>
__global__ void rotary_embedding_decode_kernel(
    const int64_t *__restrict__ positions, T *__restrict__ query,
    T *__restrict__ key, const T *__restrict__ cos_sin_cache, int num_heads,
    int num_kv_heads, int rot_dim, int seq_len, int query_batch_stride,
    int key_batch_stride, int query_stride, int key_stride,
    int query_head_stride, int key_head_stride, int query_dim_stride,
    int key_dim_stride) {
  const int token_idx = blockIdx.x;
  const int pair_idx = blockIdx.y * blockDim.x + threadIdx.x;
  const int half_rot_dim = rot_dim / 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const T *cache = cos_sin_cache + pos * rot_dim;

  int q_token_base;
  int k_token_base;
  if (seq_len > 0) {
    const int batch_idx = token_idx / seq_len;
    const int seq_idx = token_idx - batch_idx * seq_len;
    q_token_base = batch_idx * query_batch_stride + seq_idx * query_stride;
    k_token_base = batch_idx * key_batch_stride + seq_idx * key_stride;
  } else {
    q_token_base = token_idx * query_stride;
    k_token_base = token_idx * key_stride;
  }

  if (pair_idx < num_heads * half_rot_dim) {
    const int q_head_idx = pair_idx / half_rot_dim;
    const int q_rot_offset = pair_idx - q_head_idx * half_rot_dim;
    apply_one<T, IS_NEOX>(query, cache, q_token_base, query_head_stride,
                          query_dim_stride, q_head_idx, q_rot_offset,
                          half_rot_dim);
  }
  if constexpr (HAS_KEY) {
    if (pair_idx < num_kv_heads * half_rot_dim) {
      const int k_head_idx = pair_idx / half_rot_dim;
      const int k_rot_offset = pair_idx - k_head_idx * half_rot_dim;
      apply_one<T, IS_NEOX>(key, cache, k_token_base, key_head_stride,
                            key_dim_stride, k_head_idx, k_rot_offset,
                            half_rot_dim);
    }
  }
}

template <typename T, bool IS_NEOX, bool HAS_KEY>
__global__ void rotary_embedding_prefill_direct_kernel(
    const int64_t *__restrict__ positions, T *__restrict__ query,
    T *__restrict__ key, const T *__restrict__ cos_sin_cache, int num_heads,
    int num_kv_heads, int rot_dim, int seq_len, int query_batch_stride,
    int key_batch_stride, int query_stride, int key_stride,
    int query_head_stride, int key_head_stride, int query_dim_stride,
    int key_dim_stride) {
  const int token_idx = blockIdx.x;
  const int half_rot_dim = rot_dim / 2;
  const int max_head_pairs = max(num_heads, num_kv_heads) * half_rot_dim;
  const int pos = static_cast<int>(positions[token_idx]);
  const T *cache = cos_sin_cache + pos * rot_dim;

  int q_token_base;
  int k_token_base;
  if (seq_len > 0) {
    const int batch_idx = token_idx / seq_len;
    const int seq_idx = token_idx - batch_idx * seq_len;
    q_token_base = batch_idx * query_batch_stride + seq_idx * query_stride;
    k_token_base = batch_idx * key_batch_stride + seq_idx * key_stride;
  } else {
    q_token_base = token_idx * query_stride;
    k_token_base = token_idx * key_stride;
  }

  for (int pair_idx = threadIdx.x; pair_idx < max_head_pairs;
       pair_idx += blockDim.x) {
    if (pair_idx < num_heads * half_rot_dim) {
      const int q_head_idx = pair_idx / half_rot_dim;
      const int q_rot_offset = pair_idx - q_head_idx * half_rot_dim;
      apply_one<T, IS_NEOX>(query, cache, q_token_base, query_head_stride,
                            query_dim_stride, q_head_idx, q_rot_offset,
                            half_rot_dim);
    }
    if constexpr (HAS_KEY) {
      if (pair_idx < num_kv_heads * half_rot_dim) {
        const int k_head_idx = pair_idx / half_rot_dim;
        const int k_rot_offset = pair_idx - k_head_idx * half_rot_dim;
        apply_one<T, IS_NEOX>(key, cache, k_token_base, key_head_stride,
                              key_dim_stride, k_head_idx, k_rot_offset,
                              half_rot_dim);
      }
    }
  }
}

template <typename T>
__global__ void rotary_pos_emb_neox_kernel(
    const T *__restrict__ query, const T *__restrict__ key,
    const T *__restrict__ cos, const T *__restrict__ sin,
    T *__restrict__ query_out, T *__restrict__ key_out, int num_tokens,
    int seq_len, int num_heads, int num_kv_heads, int head_size,
    int query_batch_stride, int key_batch_stride, int query_out_batch_stride,
    int key_out_batch_stride, int query_stride, int key_stride,
    int query_out_stride, int key_out_stride, int query_head_stride,
    int key_head_stride, int query_out_head_stride, int key_out_head_stride,
    int query_dim_stride, int key_dim_stride, int query_out_dim_stride,
    int key_out_dim_stride, int cos_batch_stride, int sin_batch_stride,
    int cos_stride, int sin_stride, int cos_dim_stride, int sin_dim_stride) {
  const int token_idx = blockIdx.x;
  const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;
  const int half_head_size = head_size / 2;
  const int max_elems = max(num_heads, num_kv_heads) * head_size;

  if (elem_idx >= max_elems) {
    return;
  }

  const int head_idx = elem_idx / head_size;
  const int dim_idx = elem_idx - head_idx * head_size;
  const int batch_idx = seq_len > 0 ? token_idx / seq_len : 0;
  const int seq_idx = seq_len > 0 ? token_idx - batch_idx * seq_len : token_idx;
  const float c =
      to_float(cos[batch_idx * cos_batch_stride + seq_idx * cos_stride +
                   dim_idx * cos_dim_stride]);
  const float s =
      to_float(sin[batch_idx * sin_batch_stride + seq_idx * sin_stride +
                   dim_idx * sin_dim_stride]);

  if (head_idx < num_heads) {
    const int src_base =
        batch_idx * query_batch_stride + seq_idx * query_stride +
        head_idx * query_head_stride;
    const int dst_offset =
        batch_idx * query_out_batch_stride + seq_idx * query_out_stride +
        head_idx * query_out_head_stride + dim_idx * query_out_dim_stride;
    const int x_idx =
        dim_idx < half_head_size ? dim_idx : dim_idx - half_head_size;
    const int y_idx =
        dim_idx < half_head_size ? dim_idx + half_head_size : dim_idx;
    const float x = to_float(query[src_base + x_idx * query_dim_stride]);
    const float y = to_float(query[src_base + y_idx * query_dim_stride]);
    const float value = dim_idx < half_head_size ? x * c - y * s : y * c + x * s;
    query_out[dst_offset] = from_float<T>(value);
  }

  if (head_idx < num_kv_heads) {
    const int src_base = batch_idx * key_batch_stride + seq_idx * key_stride +
                         head_idx * key_head_stride;
    const int dst_offset =
        batch_idx * key_out_batch_stride + seq_idx * key_out_stride +
        head_idx * key_out_head_stride + dim_idx * key_out_dim_stride;
    const int x_idx =
        dim_idx < half_head_size ? dim_idx : dim_idx - half_head_size;
    const int y_idx =
        dim_idx < half_head_size ? dim_idx + half_head_size : dim_idx;
    const float x = to_float(key[src_base + x_idx * key_dim_stride]);
    const float y = to_float(key[src_base + y_idx * key_dim_stride]);
    const float value = dim_idx < half_head_size ? x * c - y * s : y * c + x * s;
    key_out[dst_offset] = from_float<T>(value);
  }
}

template <typename T, typename T2>
__device__ __forceinline__ void apply_rotary_pos_emb_h2(
    const T *__restrict__ src, T *__restrict__ dst, const T *__restrict__ cos,
    const T *__restrict__ sin, int src_base, int dst_base, int cos_base,
    int sin_base, int head_size, int rot) {
  const int half_head_size = head_size / 2;
  const T2 c0 = *reinterpret_cast<const T2 *>(cos + cos_base + rot);
  const T2 s0 = *reinterpret_cast<const T2 *>(sin + sin_base + rot);
  const T2 c1 =
      *reinterpret_cast<const T2 *>(cos + cos_base + half_head_size + rot);
  const T2 s1 =
      *reinterpret_cast<const T2 *>(sin + sin_base + half_head_size + rot);
  const T2 x = *reinterpret_cast<const T2 *>(src + src_base + rot);
  const T2 y =
      *reinterpret_cast<const T2 *>(src + src_base + half_head_size + rot);
  const T2 out_x = __hfma2(y, __hneg2(s0), __hmul2(x, c0));
  const T2 out_y = __hfma2(x, s1, __hmul2(y, c1));
  if constexpr (std::is_same<T, half>::value) {
    store_half2(reinterpret_cast<half2 *>(dst + dst_base + rot), out_x);
    store_half2(
        reinterpret_cast<half2 *>(dst + dst_base + half_head_size + rot),
        out_y);
  } else {
    store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(dst + dst_base + rot),
                    out_x);
    store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(
                        dst + dst_base + half_head_size + rot),
                    out_y);
  }
}

template <typename T, typename T2>
__global__ void rotary_pos_emb_neox_h2_kernel(
    const T *__restrict__ query, const T *__restrict__ key,
    const T *__restrict__ cos, const T *__restrict__ sin,
    T *__restrict__ query_out, T *__restrict__ key_out, int num_tokens,
    int seq_len, int num_heads, int num_kv_heads, int head_size,
    int query_batch_stride, int key_batch_stride, int query_out_batch_stride,
    int key_out_batch_stride, int query_stride, int key_stride,
    int query_out_stride, int key_out_stride, int query_head_stride,
    int key_head_stride, int query_out_head_stride, int key_out_head_stride,
    int cos_batch_stride, int sin_batch_stride, int cos_stride,
    int sin_stride) {
  const int token_idx = blockIdx.x;
  const int pair2_idx = blockIdx.y * blockDim.x + threadIdx.x;
  const int half_head_size = head_size / 2;
  const int max_pairs = max(num_heads, num_kv_heads) * half_head_size;
  const int pair_idx = pair2_idx * 2;
  if (pair_idx >= max_pairs) {
    return;
  }

  const int batch_idx = token_idx / seq_len;
  const int seq_idx = token_idx - batch_idx * seq_len;
  const int head_idx = pair_idx / half_head_size;
  const int rot = pair_idx - head_idx * half_head_size;
  if (rot + 1 >= half_head_size) {
    return;
  }
  const int cos_base = batch_idx * cos_batch_stride + seq_idx * cos_stride;
  const int sin_base = batch_idx * sin_batch_stride + seq_idx * sin_stride;

  if (head_idx < num_heads) {
    const int src_base =
        batch_idx * query_batch_stride + seq_idx * query_stride +
        head_idx * query_head_stride;
    const int dst_base =
        batch_idx * query_out_batch_stride + seq_idx * query_out_stride +
        head_idx * query_out_head_stride;
    apply_rotary_pos_emb_h2<T, T2>(query, query_out, cos, sin, src_base,
                                   dst_base, cos_base, sin_base, head_size,
                                   rot);
  }

  if (head_idx < num_kv_heads) {
    const int src_base = batch_idx * key_batch_stride + seq_idx * key_stride +
                         head_idx * key_head_stride;
    const int dst_base = batch_idx * key_out_batch_stride +
                         seq_idx * key_out_stride +
                         head_idx * key_out_head_stride;
    apply_rotary_pos_emb_h2<T, T2>(key, key_out, cos, sin, src_base, dst_base,
                                   cos_base, sin_base, head_size, rot);
  }
}

template <typename T, typename T2>
__global__ void rotary_pos_emb_neox_h2_flat_kernel(
    const T *__restrict__ query, const T *__restrict__ key,
    const T *__restrict__ cos, const T *__restrict__ sin,
    T *__restrict__ query_out, T *__restrict__ key_out, int num_tokens,
    int num_heads, int num_kv_heads, int head_size) {
  const int token_idx = blockIdx.x;
  const int pair2_idx = blockIdx.y * blockDim.x + threadIdx.x;
  const int half_head_size = head_size / 2;
  const int max_pairs = max(num_heads, num_kv_heads) * half_head_size;
  const int pair_idx = pair2_idx * 2;
  if (pair_idx >= max_pairs) {
    return;
  }

  const int head_idx = pair_idx / half_head_size;
  const int rot = pair_idx - head_idx * half_head_size;
  if (rot + 1 >= half_head_size) {
    return;
  }

  const int q_stride = num_heads * head_size;
  const int k_stride = num_kv_heads * head_size;
  const int cos_base = token_idx * head_size;
  if (head_idx < num_heads) {
    const int base = token_idx * q_stride + head_idx * head_size;
    apply_rotary_pos_emb_h2<T, T2>(query, query_out, cos, sin, base, base,
                                   cos_base, cos_base, head_size, rot);
  }
  if (head_idx < num_kv_heads) {
    const int base = token_idx * k_stride + head_idx * head_size;
    apply_rotary_pos_emb_h2<T, T2>(key, key_out, cos, sin, base, base,
                                   cos_base, cos_base, head_size, rot);
  }
}

__global__ void rotary_pos_emb_neox_fp16_16h80_flat_kernel(
    const half *__restrict__ query, const half *__restrict__ key,
    const half *__restrict__ cos, const half *__restrict__ sin,
    half *__restrict__ query_out, half *__restrict__ key_out, int num_tokens) {
  const int token_idx = blockIdx.x;
  const int group_idx = blockIdx.y;
  const int lane = threadIdx.x;
  constexpr int HEAD_SIZE = 80;
  constexpr int HALF_HEAD_SIZE = 40;
  constexpr int HEADS_PER_GROUP = 8;
  constexpr int Q_STRIDE = 16 * HEAD_SIZE;
  constexpr int K_STRIDE = 16 * HEAD_SIZE;
  constexpr int WORK_PER_GROUP = HEADS_PER_GROUP * (HALF_HEAD_SIZE / 2);

  if (lane >= WORK_PER_GROUP) {
    return;
  }

  const int head_idx = group_idx * HEADS_PER_GROUP + lane / 20;
  const int rot = (lane - (lane / 20) * 20) * 2;
  const int cos_base = token_idx * HEAD_SIZE;
  const int q_base = token_idx * Q_STRIDE + head_idx * HEAD_SIZE;
  const int k_base = token_idx * K_STRIDE + head_idx * HEAD_SIZE;

  apply_rotary_pos_emb_h2<half, half2>(
      query, query_out, cos, sin, q_base, q_base, cos_base, cos_base,
      HEAD_SIZE, rot);
  apply_rotary_pos_emb_h2<half, half2>(
      key, key_out, cos, sin, k_base, k_base, cos_base, cos_base, HEAD_SIZE,
      rot);
}

template <typename T, typename T2>
__global__ void rotary_pos_emb_neox_h2_flat_hs256_kernel(
    const T *__restrict__ query, const T *__restrict__ key,
    const T *__restrict__ cos, const T *__restrict__ sin,
    T *__restrict__ query_out, T *__restrict__ key_out, int num_tokens,
    int num_heads, int num_kv_heads) {
  const int token_idx = blockIdx.x;
  const int pair2_idx = blockIdx.y * blockDim.x + threadIdx.x;
  constexpr int HEAD_SIZE = 256;
  constexpr int HALF_HEAD_SIZE = 128;
  const int max_pairs = max(num_heads, num_kv_heads) * HALF_HEAD_SIZE;
  const int pair_idx = pair2_idx * 2;
  if (pair_idx >= max_pairs) {
    return;
  }

  const int head_idx = pair_idx >> 7;
  const int rot = pair_idx & (HALF_HEAD_SIZE - 1);
  if (rot + 1 >= HALF_HEAD_SIZE) {
    return;
  }

  const int q_stride = num_heads << 8;
  const int k_stride = num_kv_heads << 8;
  const int cos_base = token_idx << 8;
  if (head_idx < num_heads) {
    const int base = token_idx * q_stride + (head_idx << 8);
    apply_rotary_pos_emb_h2<T, T2>(query, query_out, cos, sin, base, base,
                                   cos_base, cos_base, HEAD_SIZE, rot);
  }
  if (head_idx < num_kv_heads) {
    const int base = token_idx * k_stride + (head_idx << 8);
    apply_rotary_pos_emb_h2<T, T2>(key, key_out, cos, sin, base, base,
                                   cos_base, cos_base, HEAD_SIZE, rot);
  }
}

template <typename T, int ROT_DIM, bool HAS_KEY, bool HAS_SEQ>
__global__ void rotary_embedding_prefill_neox_static_kernel(
    const int64_t *__restrict__ positions, T *__restrict__ query,
    T *__restrict__ key, const T *__restrict__ cos_sin_cache, int num_heads,
    int num_kv_heads, int seq_len, int query_batch_stride, int key_batch_stride,
    int query_stride, int key_stride, int query_head_stride,
    int key_head_stride) {
  constexpr int half_rot_dim = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int max_head_pairs = max(num_heads, num_kv_heads) * half_rot_dim;
  const int pos = static_cast<int>(positions[token_idx]);
  const T *cache = cos_sin_cache + pos * ROT_DIM;

  int q_token_base;
  int k_token_base;
  if constexpr (HAS_SEQ) {
    const int batch_idx = token_idx / seq_len;
    const int seq_idx = token_idx - batch_idx * seq_len;
    q_token_base = batch_idx * query_batch_stride + seq_idx * query_stride;
    k_token_base = batch_idx * key_batch_stride + seq_idx * key_stride;
  } else {
    q_token_base = token_idx * query_stride;
    k_token_base = token_idx * key_stride;
  }

  for (int pair_idx = threadIdx.x; pair_idx < max_head_pairs;
       pair_idx += blockDim.x) {
    if (pair_idx < num_heads * half_rot_dim) {
      const int q_head_idx = pair_idx / half_rot_dim;
      const int q_rot_offset = pair_idx & (half_rot_dim - 1);
      apply_one<T, true>(query, cache, q_token_base, query_head_stride, 1,
                         q_head_idx, q_rot_offset, half_rot_dim);
    }
    if constexpr (HAS_KEY) {
      if (pair_idx < num_kv_heads * half_rot_dim) {
        const int k_head_idx = pair_idx / half_rot_dim;
        const int k_rot_offset = pair_idx & (half_rot_dim - 1);
        apply_one<T, true>(key, cache, k_token_base, key_head_stride, 1,
                           k_head_idx, k_rot_offset, half_rot_dim);
      }
    }
  }
}

template <int ROT_DIM>
__device__ __forceinline__ void
apply_neox_fp16_half2(half *__restrict__ data, const half *__restrict__ cache,
                      int base, int head_stride, int pair_idx) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int head_idx = pair_idx / HALF_ROT_DIM;
  const int rot = pair_idx & (HALF_ROT_DIM - 1);
  const int head_base = base + head_idx * head_stride;
  const half2 c = *reinterpret_cast<const half2 *>(cache + rot);
  const half2 s = *reinterpret_cast<const half2 *>(cache + HALF_ROT_DIM + rot);
  const half2 x = *reinterpret_cast<const half2 *>(data + head_base + rot);
  const half2 y =
      *reinterpret_cast<const half2 *>(data + head_base + HALF_ROT_DIM + rot);
  const half2 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  const half2 out_y = __hfma2(x, s, __hmul2(y, c));
  store_half2(reinterpret_cast<half2 *>(data + head_base + rot), out_x);
  store_half2(reinterpret_cast<half2 *>(data + head_base + HALF_ROT_DIM + rot),
              out_y);
}

template <int ROT_DIM>
__device__ __forceinline__ void
apply_neox_fp16_half2_hs128(half *__restrict__ data,
                            const half *__restrict__ cache, int base,
                            int pair_idx) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int head_idx = pair_idx / HALF_ROT_DIM;
  const int rot = pair_idx & (HALF_ROT_DIM - 1);
  const int head_base = base + (head_idx << 7);
  const half2 c = *reinterpret_cast<const half2 *>(cache + rot);
  const half2 s = *reinterpret_cast<const half2 *>(cache + HALF_ROT_DIM + rot);
  const half2 x = *reinterpret_cast<const half2 *>(data + head_base + rot);
  const half2 y =
      *reinterpret_cast<const half2 *>(data + head_base + HALF_ROT_DIM + rot);
  const half2 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  const half2 out_y = __hfma2(x, s, __hmul2(y, c));
  store_half2(reinterpret_cast<half2 *>(data + head_base + rot), out_x);
  store_half2(reinterpret_cast<half2 *>(data + head_base + HALF_ROT_DIM + rot),
              out_y);
}

template <int ROT_DIM>
__device__ __forceinline__ void
apply_neox_bf16_bfloat162(__mt_bfloat16 *__restrict__ data,
                          const __mt_bfloat16 *__restrict__ cache, int base,
                          int head_stride, int pair_idx) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int head_idx = pair_idx / HALF_ROT_DIM;
  const int rot = pair_idx & (HALF_ROT_DIM - 1);
  const int head_base = base + head_idx * head_stride;
  const __mt_bfloat162 c =
      *reinterpret_cast<const __mt_bfloat162 *>(cache + rot);
  const __mt_bfloat162 s =
      *reinterpret_cast<const __mt_bfloat162 *>(cache + HALF_ROT_DIM + rot);
  const __mt_bfloat162 x =
      *reinterpret_cast<const __mt_bfloat162 *>(data + head_base + rot);
  const __mt_bfloat162 y = *reinterpret_cast<const __mt_bfloat162 *>(
      data + head_base + HALF_ROT_DIM + rot);
  const __mt_bfloat162 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  const __mt_bfloat162 out_y = __hfma2(x, s, __hmul2(y, c));
  store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(data + head_base + rot),
                  out_x);
  store_bfloat162(
      reinterpret_cast<__mt_bfloat162 *>(data + head_base + HALF_ROT_DIM + rot),
      out_y);
}

template <int ROT_DIM>
__device__ __forceinline__ void
apply_neox_bf16_bfloat162_hs128(__mt_bfloat16 *__restrict__ data,
                                const __mt_bfloat16 *__restrict__ cache,
                                int base, int pair_idx) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int head_idx = pair_idx / HALF_ROT_DIM;
  const int rot = pair_idx & (HALF_ROT_DIM - 1);
  const int head_base = base + (head_idx << 7);
  const __mt_bfloat162 c =
      *reinterpret_cast<const __mt_bfloat162 *>(cache + rot);
  const __mt_bfloat162 s =
      *reinterpret_cast<const __mt_bfloat162 *>(cache + HALF_ROT_DIM + rot);
  const __mt_bfloat162 x =
      *reinterpret_cast<const __mt_bfloat162 *>(data + head_base + rot);
  const __mt_bfloat162 y = *reinterpret_cast<const __mt_bfloat162 *>(
      data + head_base + HALF_ROT_DIM + rot);
  const __mt_bfloat162 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  const __mt_bfloat162 out_y = __hfma2(x, s, __hmul2(y, c));
  store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(data + head_base + rot),
                  out_x);
  store_bfloat162(
      reinterpret_cast<__mt_bfloat162 *>(data + head_base + HALF_ROT_DIM + rot),
      out_y);
}

template <int ROT_DIM, bool HAS_KEY>
__global__ void rotary_embedding_prefill_neox_fp16_h2_kernel(
    const int64_t *__restrict__ positions, half *__restrict__ query,
    half *__restrict__ key, const half *__restrict__ cos_sin_cache,
    int num_heads, int num_kv_heads, int query_stride, int key_stride,
    int query_head_stride, int key_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pos = static_cast<int>(positions[token_idx]);
  const half *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int max_pairs = HAS_KEY ? max(q_pairs, k_pairs) : q_pairs;

  for (int pair_idx = threadIdx.x * 2; pair_idx < max_pairs;
       pair_idx += blockDim.x * 2) {
    if (pair_idx < q_pairs) {
      apply_neox_fp16_half2<ROT_DIM>(query, cache, q_token_base,
                                     query_head_stride, pair_idx);
    }
    if constexpr (HAS_KEY) {
      if (pair_idx < k_pairs) {
        apply_neox_fp16_half2<ROT_DIM>(key, cache, k_token_base,
                                       key_head_stride, pair_idx);
      }
    }
  }
}

template <int ROT_DIM, bool HAS_KEY>
__global__ void rotary_embedding_decode_neox_fp16_h2_split_kernel(
    const int64_t *__restrict__ positions, half *__restrict__ query,
    half *__restrict__ key, const half *__restrict__ cos_sin_cache,
    int num_heads, int num_kv_heads, int query_stride, int key_stride,
    int query_head_stride, int key_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int pos = static_cast<int>(positions[token_idx]);
  const half *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;

  if (pair_idx < q_pairs) {
    apply_neox_fp16_half2<ROT_DIM>(query, cache, q_token_base,
                                   query_head_stride, pair_idx);
  }
  if constexpr (HAS_KEY) {
    if (pair_idx < k_pairs) {
      apply_neox_fp16_half2<ROT_DIM>(key, cache, k_token_base, key_head_stride,
                                     pair_idx);
    }
  }
}

template <int ROT_DIM>
__global__ void rotary_embedding_decode_neox_fp16_h2_32x8_hs128_kernel(
    const int64_t *__restrict__ positions, half *__restrict__ query,
    half *__restrict__ key, const half *__restrict__ cos_sin_cache) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  constexpr int Q_PAIRS = 32 * HALF_ROT_DIM;
  constexpr int K_PAIRS = 8 * HALF_ROT_DIM;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const half *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx << 12;
  const int k_token_base = token_idx << 10;

  if (pair_idx < Q_PAIRS) {
    apply_neox_fp16_half2_hs128<ROT_DIM>(query, cache, q_token_base, pair_idx);
  }
  if (pair_idx < K_PAIRS) {
    apply_neox_fp16_half2_hs128<ROT_DIM>(key, cache, k_token_base, pair_idx);
  }
}

template <int ROT_DIM, int Q_HEADS, int KV_HEADS>
__global__ void rotary_embedding_decode_neox_fp16_h2_hs128_fixed_kernel(
    const int64_t *__restrict__ positions, half *__restrict__ query,
    half *__restrict__ key, const half *__restrict__ cos_sin_cache) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  constexpr int Q_PAIRS = Q_HEADS * HALF_ROT_DIM;
  constexpr int K_PAIRS = KV_HEADS * HALF_ROT_DIM;
  constexpr int Q_STRIDE = Q_HEADS << 7;
  constexpr int K_STRIDE = KV_HEADS << 7;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const half *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * Q_STRIDE;
  const int k_token_base = token_idx * K_STRIDE;

  if (pair_idx < Q_PAIRS) {
    apply_neox_fp16_half2_hs128<ROT_DIM>(query, cache, q_token_base, pair_idx);
  }
  if (pair_idx < K_PAIRS) {
    apply_neox_fp16_half2_hs128<ROT_DIM>(key, cache, k_token_base, pair_idx);
  }
}

template <int ROT_DIM, bool HAS_KEY>
__global__ void rotary_embedding_prefill_neox_bf16_h2_kernel(
    const int64_t *__restrict__ positions, __mt_bfloat16 *__restrict__ query,
    __mt_bfloat16 *__restrict__ key,
    const __mt_bfloat16 *__restrict__ cos_sin_cache, int num_heads,
    int num_kv_heads, int query_stride, int key_stride, int query_head_stride,
    int key_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16 *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int max_pairs = HAS_KEY ? max(q_pairs, k_pairs) : q_pairs;

  for (int pair_idx = threadIdx.x * 2; pair_idx < max_pairs;
       pair_idx += blockDim.x * 2) {
    if (pair_idx < q_pairs) {
      apply_neox_bf16_bfloat162<ROT_DIM>(query, cache, q_token_base,
                                         query_head_stride, pair_idx);
    }
    if constexpr (HAS_KEY) {
      if (pair_idx < k_pairs) {
        apply_neox_bf16_bfloat162<ROT_DIM>(key, cache, k_token_base,
                                           key_head_stride, pair_idx);
      }
    }
  }
}

template <int ROT_DIM, bool HAS_KEY>
__global__ void rotary_embedding_decode_neox_bf16_h2_split_kernel(
    const int64_t *__restrict__ positions, __mt_bfloat16 *__restrict__ query,
    __mt_bfloat16 *__restrict__ key,
    const __mt_bfloat16 *__restrict__ cos_sin_cache, int num_heads,
    int num_kv_heads, int query_stride, int key_stride, int query_head_stride,
    int key_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16 *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;

  if (pair_idx < q_pairs) {
    apply_neox_bf16_bfloat162<ROT_DIM>(query, cache, q_token_base,
                                       query_head_stride, pair_idx);
  }
  if constexpr (HAS_KEY) {
    if (pair_idx < k_pairs) {
      apply_neox_bf16_bfloat162<ROT_DIM>(key, cache, k_token_base,
                                         key_head_stride, pair_idx);
    }
  }
}

template <int ROT_DIM>
__global__ void rotary_embedding_decode_neox_bf16_h2_32x8_hs128_kernel(
    const int64_t *__restrict__ positions, __mt_bfloat16 *__restrict__ query,
    __mt_bfloat16 *__restrict__ key,
    const __mt_bfloat16 *__restrict__ cos_sin_cache) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  constexpr int Q_PAIRS = 32 * HALF_ROT_DIM;
  constexpr int K_PAIRS = 8 * HALF_ROT_DIM;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16 *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx << 12;
  const int k_token_base = token_idx << 10;

  if (pair_idx < Q_PAIRS) {
    apply_neox_bf16_bfloat162_hs128<ROT_DIM>(query, cache, q_token_base,
                                             pair_idx);
  }
  if (pair_idx < K_PAIRS) {
    apply_neox_bf16_bfloat162_hs128<ROT_DIM>(key, cache, k_token_base,
                                             pair_idx);
  }
}

template <int ROT_DIM, int Q_HEADS, int KV_HEADS>
__global__ void rotary_embedding_decode_neox_bf16_h2_hs128_fixed_kernel(
    const int64_t *__restrict__ positions, __mt_bfloat16 *__restrict__ query,
    __mt_bfloat16 *__restrict__ key,
    const __mt_bfloat16 *__restrict__ cos_sin_cache) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  constexpr int Q_PAIRS = Q_HEADS * HALF_ROT_DIM;
  constexpr int K_PAIRS = KV_HEADS * HALF_ROT_DIM;
  constexpr int Q_STRIDE = Q_HEADS << 7;
  constexpr int K_STRIDE = KV_HEADS << 7;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16 *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * Q_STRIDE;
  const int k_token_base = token_idx * K_STRIDE;

  if (pair_idx < Q_PAIRS) {
    apply_neox_bf16_bfloat162_hs128<ROT_DIM>(query, cache, q_token_base,
                                             pair_idx);
  }
  if (pair_idx < K_PAIRS) {
    apply_neox_bf16_bfloat162_hs128<ROT_DIM>(key, cache, k_token_base,
                                             pair_idx);
  }
}

template <int ROT_DIM>
__global__ void rotary_embedding_prefill_neox_fp16_q_h2x2_kernel(
    const int64_t *__restrict__ positions, half *__restrict__ query,
    const half *__restrict__ cos_sin_cache, int num_heads, int query_stride,
    int query_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pos = static_cast<int>(positions[token_idx]);
  const half *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int q_pairs = num_heads * HALF_ROT_DIM;

  for (int pair_idx = threadIdx.x * 4; pair_idx < q_pairs;
       pair_idx += blockDim.x * 4) {
    apply_neox_fp16_half2<ROT_DIM>(query, cache, q_token_base,
                                   query_head_stride, pair_idx);
    const int pair_idx_1 = pair_idx + 2;
    if (pair_idx_1 < q_pairs) {
      apply_neox_fp16_half2<ROT_DIM>(query, cache, q_token_base,
                                     query_head_stride, pair_idx_1);
    }
  }
}

template <int ROT_DIM>
inline bool launch_fp16_hs128_fixed(
    int num_heads, int num_kv_heads, dim3 grid, dim3 block,
    musaStream_t stream, const int64_t *positions, half *query, half *key,
    const half *cos_sin_cache) {
#define SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(QH, KVH)                      \
  if (num_heads == QH && num_kv_heads == KVH) {                             \
    rotary_embedding_decode_neox_fp16_h2_hs128_fixed_kernel<ROT_DIM, QH,    \
                                                            KVH>            \
        <<<grid, block, 0, stream>>>(positions, query, key, cos_sin_cache);  \
    return true;                                                            \
  }

  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(16, 2)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(16, 4)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(28, 4)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(28, 8)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(32, 4)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(32, 8)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(40, 8)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(48, 8)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(56, 8)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(64, 4)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(64, 8)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(80, 8)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(96, 8)
  SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED(128, 8)

#undef SGLANG_MUSA_ROPE_TRY_FP16_HS128_FIXED
  return false;
}

template <int ROT_DIM>
inline bool launch_bf16_hs128_fixed(
    int num_heads, int num_kv_heads, dim3 grid, dim3 block,
    musaStream_t stream, const int64_t *positions, __mt_bfloat16 *query,
    __mt_bfloat16 *key, const __mt_bfloat16 *cos_sin_cache) {
#define SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(QH, KVH)                         \
  if (num_heads == QH && num_kv_heads == KVH) {                                \
    rotary_embedding_decode_neox_bf16_h2_hs128_fixed_kernel<ROT_DIM, QH, KVH>  \
        <<<grid, block, 0, stream>>>(positions, query, key, cos_sin_cache);     \
    return true;                                                               \
  }

  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(16, 2)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(16, 4)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(28, 4)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(28, 8)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(32, 4)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(32, 8)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(40, 8)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(48, 8)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(56, 8)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(64, 4)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(64, 8)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(80, 8)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(96, 8)
  SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED(128, 8)

#undef SGLANG_MUSA_ROPE_TRY_BF16_HS128_FIXED
  return false;
}

template <typename T, bool IS_NEOX, bool HAS_KEY>
void launch_rotary_embedding(ffi::TensorView positions, ffi::TensorView query,
                             ffi::TensorView key, ffi::TensorView cos_sin_cache,
                             int num_tokens, int num_heads, int num_kv_heads,
                             int rot_dim, int seq_len, int query_batch_stride,
                             int key_batch_stride, int query_stride,
                             int key_stride, int query_head_stride,
                             int key_head_stride, int query_dim_stride,
                             int key_dim_stride, musaStream_t stream) {
  constexpr int kDecodeThreshold = 64;
  const int half_rot_dim = rot_dim / 2;
  const int max_head_pairs = std::max(num_heads, num_kv_heads) * half_rot_dim;
  if constexpr (IS_NEOX) {
    if constexpr (std::is_same<T, half>::value && HAS_KEY) {
      if (can_use_neox_h2_qk(seq_len, rot_dim, query_stride, key_stride,
                             query_head_stride, key_head_stride,
                             query_dim_stride, key_dim_stride)) {
        if (num_tokens <= 256) {
          const int block_threads = num_tokens < 128 ? 64 : 128;
          const int h2_pairs = (max_head_pairs + 1) / 2;
          const dim3 grid(num_tokens,
                          (h2_pairs + block_threads - 1) / block_threads);
          const dim3 block(block_threads);
          bool launched_hs128 = false;
          if (is_32x8_hs128_layout(num_heads, num_kv_heads, query_stride,
                                   key_stride, query_head_stride,
                                   key_head_stride)) {
            if (rot_dim == 64) {
              rotary_embedding_decode_neox_fp16_h2_32x8_hs128_kernel<64>
                  <<<grid, block, 0, stream>>>(
                      static_cast<const int64_t *>(positions.data_ptr()),
                      static_cast<half *>(query.data_ptr()),
                      static_cast<half *>(key.data_ptr()),
                      static_cast<const half *>(cos_sin_cache.data_ptr()));
            } else {
              rotary_embedding_decode_neox_fp16_h2_32x8_hs128_kernel<128>
                  <<<grid, block, 0, stream>>>(
                      static_cast<const int64_t *>(positions.data_ptr()),
                      static_cast<half *>(query.data_ptr()),
                      static_cast<half *>(key.data_ptr()),
                      static_cast<const half *>(cos_sin_cache.data_ptr()));
            }
            launched_hs128 = true;
          } else if (is_hs128_contiguous_layout(
                         num_heads, num_kv_heads, query_stride, key_stride,
                         query_head_stride, key_head_stride)) {
            launched_hs128 =
                rot_dim == 64
                    ? launch_fp16_hs128_fixed<64>(
                          num_heads, num_kv_heads, grid, block, stream,
                          static_cast<const int64_t *>(positions.data_ptr()),
                          static_cast<half *>(query.data_ptr()),
                          static_cast<half *>(key.data_ptr()),
                          static_cast<const half *>(cos_sin_cache.data_ptr()))
                    : launch_fp16_hs128_fixed<128>(
                          num_heads, num_kv_heads, grid, block, stream,
                          static_cast<const int64_t *>(positions.data_ptr()),
                          static_cast<half *>(query.data_ptr()),
                          static_cast<half *>(key.data_ptr()),
                          static_cast<const half *>(cos_sin_cache.data_ptr()));
          }
          if (!launched_hs128 && rot_dim == 64) {
            rotary_embedding_decode_neox_fp16_h2_split_kernel<64, true>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<half *>(query.data_ptr()),
                    static_cast<half *>(key.data_ptr()),
                    static_cast<const half *>(cos_sin_cache.data_ptr()),
                    num_heads, num_kv_heads, query_stride, key_stride,
                    query_head_stride, key_head_stride);
          } else if (!launched_hs128) {
            rotary_embedding_decode_neox_fp16_h2_split_kernel<128, true>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<half *>(query.data_ptr()),
                    static_cast<half *>(key.data_ptr()),
                    static_cast<const half *>(cos_sin_cache.data_ptr()),
                    num_heads, num_kv_heads, query_stride, key_stride,
                    query_head_stride, key_head_stride);
          }
          return;
        }
        const dim3 grid(num_tokens);
        const bool use_prefill_256_threads =
            (rot_dim == 64 && num_tokens <= 4096) ||
            (rot_dim == 128 && num_tokens >= 2048 && num_tokens <= 4096);
        const dim3 block(use_prefill_256_threads ? 256 : 512);
        if (rot_dim == 64) {
          rotary_embedding_prefill_neox_fp16_h2_kernel<64, true>
              <<<grid, block, 0, stream>>>(
                  static_cast<const int64_t *>(positions.data_ptr()),
                  static_cast<half *>(query.data_ptr()),
                  static_cast<half *>(key.data_ptr()),
                  static_cast<const half *>(cos_sin_cache.data_ptr()),
                  num_heads, num_kv_heads, query_stride, key_stride,
                  query_head_stride, key_head_stride);
        } else {
          rotary_embedding_prefill_neox_fp16_h2_kernel<128, true>
              <<<grid, block, 0, stream>>>(
                  static_cast<const int64_t *>(positions.data_ptr()),
                  static_cast<half *>(query.data_ptr()),
                  static_cast<half *>(key.data_ptr()),
                  static_cast<const half *>(cos_sin_cache.data_ptr()),
                  num_heads, num_kv_heads, query_stride, key_stride,
                  query_head_stride, key_head_stride);
        }
        return;
      }
    }
    if constexpr (std::is_same<T, __mt_bfloat16>::value && HAS_KEY) {
      if (can_use_neox_h2_qk(seq_len, rot_dim, query_stride, key_stride,
                             query_head_stride, key_head_stride,
                             query_dim_stride, key_dim_stride)) {
        if (num_tokens <= 256) {
          const int block_threads = num_tokens < 128 ? 64 : 128;
          const int h2_pairs = (max_head_pairs + 1) / 2;
          const dim3 grid(num_tokens,
                          (h2_pairs + block_threads - 1) / block_threads);
          const dim3 block(block_threads);
          bool launched_hs128 = false;
          if (is_32x8_hs128_layout(num_heads, num_kv_heads, query_stride,
                                   key_stride, query_head_stride,
                                   key_head_stride)) {
            if (rot_dim == 64) {
              rotary_embedding_decode_neox_bf16_h2_32x8_hs128_kernel<64>
                  <<<grid, block, 0, stream>>>(
                      static_cast<const int64_t *>(positions.data_ptr()),
                      static_cast<__mt_bfloat16 *>(query.data_ptr()),
                      static_cast<__mt_bfloat16 *>(key.data_ptr()),
                      static_cast<const __mt_bfloat16 *>(
                          cos_sin_cache.data_ptr()));
            } else {
              rotary_embedding_decode_neox_bf16_h2_32x8_hs128_kernel<128>
                  <<<grid, block, 0, stream>>>(
                      static_cast<const int64_t *>(positions.data_ptr()),
                      static_cast<__mt_bfloat16 *>(query.data_ptr()),
                      static_cast<__mt_bfloat16 *>(key.data_ptr()),
                      static_cast<const __mt_bfloat16 *>(
                          cos_sin_cache.data_ptr()));
            }
            launched_hs128 = true;
          } else if (is_hs128_contiguous_layout(
                         num_heads, num_kv_heads, query_stride, key_stride,
                         query_head_stride, key_head_stride)) {
            launched_hs128 =
                rot_dim == 64
                    ? launch_bf16_hs128_fixed<64>(
                          num_heads, num_kv_heads, grid, block, stream,
                          static_cast<const int64_t *>(positions.data_ptr()),
                          static_cast<__mt_bfloat16 *>(query.data_ptr()),
                          static_cast<__mt_bfloat16 *>(key.data_ptr()),
                          static_cast<const __mt_bfloat16 *>(
                              cos_sin_cache.data_ptr()))
                    : launch_bf16_hs128_fixed<128>(
                          num_heads, num_kv_heads, grid, block, stream,
                          static_cast<const int64_t *>(positions.data_ptr()),
                          static_cast<__mt_bfloat16 *>(query.data_ptr()),
                          static_cast<__mt_bfloat16 *>(key.data_ptr()),
                          static_cast<const __mt_bfloat16 *>(
                              cos_sin_cache.data_ptr()));
          }
          if (!launched_hs128 && rot_dim == 64) {
            rotary_embedding_decode_neox_bf16_h2_split_kernel<64, true>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<__mt_bfloat16 *>(query.data_ptr()),
                    static_cast<__mt_bfloat16 *>(key.data_ptr()),
                    static_cast<const __mt_bfloat16 *>(
                        cos_sin_cache.data_ptr()),
                    num_heads, num_kv_heads, query_stride, key_stride,
                    query_head_stride, key_head_stride);
          } else if (!launched_hs128) {
            rotary_embedding_decode_neox_bf16_h2_split_kernel<128, true>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<__mt_bfloat16 *>(query.data_ptr()),
                    static_cast<__mt_bfloat16 *>(key.data_ptr()),
                    static_cast<const __mt_bfloat16 *>(
                        cos_sin_cache.data_ptr()),
                    num_heads, num_kv_heads, query_stride, key_stride,
                    query_head_stride, key_head_stride);
          }
          return;
        }
        const dim3 grid(num_tokens);
        const bool use_prefill_256_threads =
            (rot_dim == 64 && num_tokens <= 4096) ||
            (rot_dim == 128 && num_tokens >= 2048 && num_tokens <= 4096);
        const dim3 block(use_prefill_256_threads ? 256 : 512);
        if (rot_dim == 64) {
          rotary_embedding_prefill_neox_bf16_h2_kernel<64, true>
              <<<grid, block, 0, stream>>>(
                  static_cast<const int64_t *>(positions.data_ptr()),
                  static_cast<__mt_bfloat16 *>(query.data_ptr()),
                  static_cast<__mt_bfloat16 *>(key.data_ptr()),
                  static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
                  num_heads, num_kv_heads, query_stride, key_stride,
                  query_head_stride, key_head_stride);
        } else {
          rotary_embedding_prefill_neox_bf16_h2_kernel<128, true>
              <<<grid, block, 0, stream>>>(
                  static_cast<const int64_t *>(positions.data_ptr()),
                  static_cast<__mt_bfloat16 *>(query.data_ptr()),
                  static_cast<__mt_bfloat16 *>(key.data_ptr()),
                  static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
                  num_heads, num_kv_heads, query_stride, key_stride,
                  query_head_stride, key_head_stride);
        }
        return;
      }
    }
  }
  if (num_tokens <= kDecodeThreshold) {
    const int block_pairs =
        std::min(512, std::max(32, num_heads * half_rot_dim));
    const dim3 grid(num_tokens,
                    (max_head_pairs + block_pairs - 1) / block_pairs);
    const dim3 block(block_pairs);
    rotary_embedding_decode_kernel<T, IS_NEOX, HAS_KEY>
        <<<grid, block, 0, stream>>>(
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<T *>(query.data_ptr()),
            static_cast<T *>(key.data_ptr()),
            static_cast<const T *>(cos_sin_cache.data_ptr()), num_heads,
            num_kv_heads, rot_dim, seq_len, query_batch_stride,
            key_batch_stride, query_stride, key_stride, query_head_stride,
            key_head_stride, query_dim_stride, key_dim_stride);
  } else {
    if constexpr (IS_NEOX) {
      if constexpr (std::is_same<T, half>::value && HAS_KEY) {
        if (can_use_neox_h2_qk(seq_len, rot_dim, query_stride, key_stride,
                               query_head_stride, key_head_stride,
                               query_dim_stride, key_dim_stride)) {
          const dim3 grid(num_tokens);
          const dim3 block(num_tokens <= 256 ? 256 : 512);
          if (rot_dim == 64) {
            rotary_embedding_prefill_neox_fp16_h2_kernel<64, true>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<half *>(query.data_ptr()),
                    static_cast<half *>(key.data_ptr()),
                    static_cast<const half *>(cos_sin_cache.data_ptr()),
                    num_heads, num_kv_heads, query_stride, key_stride,
                    query_head_stride, key_head_stride);
          } else {
            rotary_embedding_prefill_neox_fp16_h2_kernel<128, true>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<half *>(query.data_ptr()),
                    static_cast<half *>(key.data_ptr()),
                    static_cast<const half *>(cos_sin_cache.data_ptr()),
                    num_heads, num_kv_heads, query_stride, key_stride,
                    query_head_stride, key_head_stride);
          }
          return;
        }
      }
      if constexpr (std::is_same<T, half>::value && !HAS_KEY) {
        if (can_use_neox_h2_q(seq_len, rot_dim, query_stride, query_head_stride,
                              query_dim_stride)) {
          if (rot_dim == 64) {
            const dim3 grid(num_tokens);
            const dim3 block(256);
            rotary_embedding_prefill_neox_fp16_h2_kernel<64, false>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<half *>(query.data_ptr()),
                    static_cast<half *>(key.data_ptr()),
                    static_cast<const half *>(cos_sin_cache.data_ptr()),
                    num_heads, num_kv_heads, query_stride, key_stride,
                    query_head_stride, key_head_stride);
          } else {
            const dim3 grid(num_tokens);
            const dim3 block(512);
            rotary_embedding_prefill_neox_fp16_q_h2x2_kernel<128>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<half *>(query.data_ptr()),
                    static_cast<const half *>(cos_sin_cache.data_ptr()),
                    num_heads, query_stride, query_head_stride);
          }
          return;
        }
      }
      if constexpr (std::is_same<T, __mt_bfloat16>::value) {
        if (can_use_neox_h2_qk(seq_len, rot_dim, query_stride, key_stride,
                               query_head_stride, key_head_stride,
                               query_dim_stride, key_dim_stride)) {
          const dim3 grid(num_tokens);
          const dim3 block(512);
          if constexpr (HAS_KEY) {
            if (rot_dim == 64) {
              rotary_embedding_prefill_neox_bf16_h2_kernel<64, true>
                  <<<grid, block, 0, stream>>>(
                      static_cast<const int64_t *>(positions.data_ptr()),
                      static_cast<__mt_bfloat16 *>(query.data_ptr()),
                      static_cast<__mt_bfloat16 *>(key.data_ptr()),
                      static_cast<const __mt_bfloat16 *>(
                          cos_sin_cache.data_ptr()),
                      num_heads, num_kv_heads, query_stride, key_stride,
                      query_head_stride, key_head_stride);
            } else {
              rotary_embedding_prefill_neox_bf16_h2_kernel<128, true>
                  <<<grid, block, 0, stream>>>(
                      static_cast<const int64_t *>(positions.data_ptr()),
                      static_cast<__mt_bfloat16 *>(query.data_ptr()),
                      static_cast<__mt_bfloat16 *>(key.data_ptr()),
                      static_cast<const __mt_bfloat16 *>(
                          cos_sin_cache.data_ptr()),
                      num_heads, num_kv_heads, query_stride, key_stride,
                      query_head_stride, key_head_stride);
            }
            return;
          } else if (rot_dim == 64) {
            rotary_embedding_prefill_neox_bf16_h2_kernel<64, false>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<__mt_bfloat16 *>(query.data_ptr()),
                    static_cast<__mt_bfloat16 *>(key.data_ptr()),
                    static_cast<const __mt_bfloat16 *>(
                        cos_sin_cache.data_ptr()),
                    num_heads, num_kv_heads, query_stride, key_stride,
                    query_head_stride, key_head_stride);
            return;
          }
        }
      }

      if (query_dim_stride == 1 && key_dim_stride == 1 &&
          (rot_dim == 64 || rot_dim == 128)) {
        const int block_pairs =
            prefill_block_pairs(num_tokens, num_heads, half_rot_dim);
        const dim3 grid(num_tokens);
        const dim3 block(block_pairs);
        if (rot_dim == 64) {
          if (seq_len > 0) {
            rotary_embedding_prefill_neox_static_kernel<T, 64, HAS_KEY, true>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<T *>(query.data_ptr()),
                    static_cast<T *>(key.data_ptr()),
                    static_cast<const T *>(cos_sin_cache.data_ptr()), num_heads,
                    num_kv_heads, seq_len, query_batch_stride, key_batch_stride,
                    query_stride, key_stride, query_head_stride,
                    key_head_stride);
          } else {
            rotary_embedding_prefill_neox_static_kernel<T, 64, HAS_KEY, false>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<T *>(query.data_ptr()),
                    static_cast<T *>(key.data_ptr()),
                    static_cast<const T *>(cos_sin_cache.data_ptr()), num_heads,
                    num_kv_heads, 0, query_batch_stride, key_batch_stride,
                    query_stride, key_stride, query_head_stride,
                    key_head_stride);
          }
        } else {
          if (seq_len > 0) {
            rotary_embedding_prefill_neox_static_kernel<T, 128, HAS_KEY, true>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<T *>(query.data_ptr()),
                    static_cast<T *>(key.data_ptr()),
                    static_cast<const T *>(cos_sin_cache.data_ptr()), num_heads,
                    num_kv_heads, seq_len, query_batch_stride, key_batch_stride,
                    query_stride, key_stride, query_head_stride,
                    key_head_stride);
          } else {
            rotary_embedding_prefill_neox_static_kernel<T, 128, HAS_KEY, false>
                <<<grid, block, 0, stream>>>(
                    static_cast<const int64_t *>(positions.data_ptr()),
                    static_cast<T *>(query.data_ptr()),
                    static_cast<T *>(key.data_ptr()),
                    static_cast<const T *>(cos_sin_cache.data_ptr()), num_heads,
                    num_kv_heads, 0, query_batch_stride, key_batch_stride,
                    query_stride, key_stride, query_head_stride,
                    key_head_stride);
          }
        }
        return;
      }
    }

    const int block_pairs =
        prefill_block_pairs(num_tokens, num_heads, half_rot_dim);
    const dim3 grid(num_tokens);
    const dim3 block(block_pairs);
    rotary_embedding_prefill_direct_kernel<T, IS_NEOX, HAS_KEY>
        <<<grid, block, 0, stream>>>(
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<T *>(query.data_ptr()),
            static_cast<T *>(key.data_ptr()),
            static_cast<const T *>(cos_sin_cache.data_ptr()), num_heads,
            num_kv_heads, rot_dim, seq_len, query_batch_stride,
            key_batch_stride, query_stride, key_stride, query_head_stride,
            key_head_stride, query_dim_stride, key_dim_stride);
  }
}

template <typename T>
void dispatch_rotary_embedding(
    ffi::TensorView positions, ffi::TensorView query, ffi::TensorView key,
    ffi::TensorView cos_sin_cache, int num_tokens, int num_heads,
    int num_kv_heads, int rot_dim, int seq_len, int query_batch_stride,
    int key_batch_stride, int query_stride, int key_stride,
    int query_head_stride, int key_head_stride, int query_dim_stride,
    int key_dim_stride, bool is_neox, bool has_key, musaStream_t stream) {
#define DISPATCH_ROPE(IS_NEOX_VALUE, HAS_KEY_VALUE)                            \
  launch_rotary_embedding<T, IS_NEOX_VALUE, HAS_KEY_VALUE>(                    \
      positions, query, key, cos_sin_cache, num_tokens, num_heads,             \
      num_kv_heads, rot_dim, seq_len, query_batch_stride, key_batch_stride,    \
      query_stride, key_stride, query_head_stride, key_head_stride,            \
      query_dim_stride, key_dim_stride, stream)

  if (is_neox) {
    if (has_key) {
      DISPATCH_ROPE(true, true);
    } else {
      DISPATCH_ROPE(true, false);
    }
  } else {
    if (has_key) {
      DISPATCH_ROPE(false, true);
    } else {
      DISPATCH_ROPE(false, false);
    }
  }
#undef DISPATCH_ROPE
}

void sgl_rotary_embedding(ffi::TensorView positions, ffi::TensorView query,
                          ffi::TensorView key, ffi::TensorView cos_sin_cache,
                          int64_t num_tokens, int64_t head_size,
                          int64_t num_heads, int64_t num_kv_heads,
                          int64_t rot_dim, int64_t seq_len,
                          int64_t query_batch_stride, int64_t key_batch_stride,
                          int64_t query_stride, int64_t key_stride,
                          int64_t query_head_stride, int64_t key_head_stride,
                          int64_t query_dim_stride, int64_t key_dim_stride,
                          bool is_neox, bool has_key) {
  CHECK_MUSA(positions);
  CHECK_MUSA(query);
  CHECK_MUSA(key);
  CHECK_MUSA(cos_sin_cache);
  TVM_FFI_ICHECK(positions.IsContiguous());
  TVM_FFI_ICHECK(cos_sin_cache.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), cos_sin_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(key.dtype(), query.dtype()));
  TVM_FFI_ICHECK_EQ(positions.device().device_id, query.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, key.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id,
                    cos_sin_cache.device().device_id);
  TVM_FFI_ICHECK_GT(num_tokens, 0);
  TVM_FFI_ICHECK_GT(head_size, 0);
  TVM_FFI_ICHECK_GT(num_heads, 0);
  TVM_FFI_ICHECK_GT(num_kv_heads, 0);
  TVM_FFI_ICHECK_GT(rot_dim, 0);
  TVM_FFI_ICHECK_EQ(rot_dim % 2, 0);
  TVM_FFI_ICHECK_LE(rot_dim, head_size);

  ffi::MUSADeviceGuard device_guard(query.device().device_id);
  musaStream_t stream = get_stream(query.device());

#define DISPATCH_DTYPE(T)                                                      \
  dispatch_rotary_embedding<T>(                                                \
      positions, query, key, cos_sin_cache, static_cast<int>(num_tokens),      \
      static_cast<int>(num_heads), static_cast<int>(num_kv_heads),             \
      static_cast<int>(rot_dim), static_cast<int>(seq_len),                    \
      static_cast<int>(query_batch_stride),                                    \
      static_cast<int>(key_batch_stride), static_cast<int>(query_stride),      \
      static_cast<int>(key_stride), static_cast<int>(query_head_stride),       \
      static_cast<int>(key_head_stride), static_cast<int>(query_dim_stride),   \
      static_cast<int>(key_dim_stride), is_neox, has_key, stream)

  if (dtype_equal(query.dtype(), dl_float16)) {
    DISPATCH_DTYPE(half);
  } else if (dtype_equal(query.dtype(), dl_bfloat16)) {
    DISPATCH_DTYPE(__mt_bfloat16);
  } else if (dtype_equal(query.dtype(), dl_float32)) {
    DISPATCH_DTYPE(float);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported rotary_embedding dtype";
  }
#undef DISPATCH_DTYPE
}

template <int ROT_DIM, typename index_t>
__global__ void rotary_embedding_cache_neox_bf16_h2_kernel(
    const int64_t *__restrict__ positions, __mt_bfloat16 *__restrict__ query,
    __mt_bfloat16 *__restrict__ key, const __mt_bfloat16 *__restrict__ value,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int num_heads, int num_kv_heads,
    int query_stride, int key_stride, int value_stride, int query_head_stride,
    int key_head_stride, int64_t k_cache_row_stride,
    int64_t v_cache_row_stride, int64_t indices_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token_idx * indices_stride]);
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16 *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;
  const int v_token_base = token_idx * value_stride;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int row_dim = num_kv_heads * ROT_DIM;
  const int max_pairs = max(q_pairs, k_pairs);

  for (int pair_idx = threadIdx.x * 2; pair_idx < max_pairs;
       pair_idx += blockDim.x * 2) {
    if (pair_idx < q_pairs) {
      apply_neox_bf16_bfloat162<ROT_DIM>(query, cache, q_token_base,
                                         query_head_stride, pair_idx);
    }
    if (pair_idx < k_pairs) {
      const int head_idx = pair_idx / HALF_ROT_DIM;
      const int rot = pair_idx & (HALF_ROT_DIM - 1);
      const int key_head_base = k_token_base + head_idx * key_head_stride;
      const int cache_head_base =
          cache_idx * k_cache_row_stride + head_idx * ROT_DIM;
      const __mt_bfloat162 c =
          *reinterpret_cast<const __mt_bfloat162 *>(cache + rot);
      const __mt_bfloat162 s =
          *reinterpret_cast<const __mt_bfloat162 *>(cache + HALF_ROT_DIM + rot);
      const __mt_bfloat162 x =
          *reinterpret_cast<const __mt_bfloat162 *>(key + key_head_base + rot);
      const __mt_bfloat162 y = *reinterpret_cast<const __mt_bfloat162 *>(
          key + key_head_base + HALF_ROT_DIM + rot);
      const __mt_bfloat162 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
      const __mt_bfloat162 out_y = __hfma2(x, s, __hmul2(y, c));
      store_bfloat162(
          reinterpret_cast<__mt_bfloat162 *>(key + key_head_base + rot), out_x);
      store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(
                          key + key_head_base + HALF_ROT_DIM + rot),
                      out_y);
      store_bfloat162(
          reinterpret_cast<__mt_bfloat162 *>(k_cache + cache_head_base + rot),
          out_x);
      store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(
                          k_cache + cache_head_base + HALF_ROT_DIM + rot),
                      out_y);
    }
  }

  const int64_t v_cache_base = cache_idx * v_cache_row_stride;
  for (int col = threadIdx.x * 2; col < row_dim; col += blockDim.x * 2) {
    const __mt_bfloat162 vv =
        *reinterpret_cast<const __mt_bfloat162 *>(value + v_token_base + col);
    store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(v_cache + v_cache_base + col),
                    vv);
  }
}

template <int ROT_DIM, typename index_t>
__global__ void rotary_embedding_cache_qout_neox_bf16_h2_kernel(
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ query, __mt_bfloat16 *__restrict__ q_out,
    __mt_bfloat16 *__restrict__ key, const __mt_bfloat16 *__restrict__ value,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int num_heads, int num_kv_heads,
    int query_stride, int key_stride, int value_stride, int query_head_stride,
    int key_head_stride, int64_t k_cache_row_stride,
    int64_t v_cache_row_stride, int64_t indices_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token_idx * indices_stride]);
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16 *cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;
  const int v_token_base = token_idx * value_stride;
  const int q_out_token_base = token_idx * num_heads * ROT_DIM;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int row_dim = num_kv_heads * ROT_DIM;
  const int max_pairs = max(q_pairs, k_pairs);

  for (int pair_idx = threadIdx.x * 2; pair_idx < max_pairs;
       pair_idx += blockDim.x * 2) {
    if (pair_idx < q_pairs) {
      const int head_idx = pair_idx / HALF_ROT_DIM;
      const int rot = pair_idx & (HALF_ROT_DIM - 1);
      const int query_head_base = q_token_base + head_idx * query_head_stride;
      const int q_out_head_base = q_out_token_base + head_idx * ROT_DIM;
      const __mt_bfloat162 c =
          *reinterpret_cast<const __mt_bfloat162 *>(cache + rot);
      const __mt_bfloat162 s =
          *reinterpret_cast<const __mt_bfloat162 *>(cache + HALF_ROT_DIM + rot);
      const __mt_bfloat162 x = *reinterpret_cast<const __mt_bfloat162 *>(
          query + query_head_base + rot);
      const __mt_bfloat162 y = *reinterpret_cast<const __mt_bfloat162 *>(
          query + query_head_base + HALF_ROT_DIM + rot);
      const __mt_bfloat162 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
      const __mt_bfloat162 out_y = __hfma2(x, s, __hmul2(y, c));
      store_bfloat162(
          reinterpret_cast<__mt_bfloat162 *>(q_out + q_out_head_base + rot),
          out_x);
      store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(
                          q_out + q_out_head_base + HALF_ROT_DIM + rot),
                      out_y);
    }
    if (pair_idx < k_pairs) {
      const int head_idx = pair_idx / HALF_ROT_DIM;
      const int rot = pair_idx & (HALF_ROT_DIM - 1);
      const int key_head_base = k_token_base + head_idx * key_head_stride;
      const int cache_head_base =
          cache_idx * k_cache_row_stride + head_idx * ROT_DIM;
      const __mt_bfloat162 c =
          *reinterpret_cast<const __mt_bfloat162 *>(cache + rot);
      const __mt_bfloat162 s =
          *reinterpret_cast<const __mt_bfloat162 *>(cache + HALF_ROT_DIM + rot);
      const __mt_bfloat162 x =
          *reinterpret_cast<const __mt_bfloat162 *>(key + key_head_base + rot);
      const __mt_bfloat162 y = *reinterpret_cast<const __mt_bfloat162 *>(
          key + key_head_base + HALF_ROT_DIM + rot);
      const __mt_bfloat162 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
      const __mt_bfloat162 out_y = __hfma2(x, s, __hmul2(y, c));
      store_bfloat162(
          reinterpret_cast<__mt_bfloat162 *>(key + key_head_base + rot), out_x);
      store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(
                          key + key_head_base + HALF_ROT_DIM + rot),
                      out_y);
      store_bfloat162(
          reinterpret_cast<__mt_bfloat162 *>(k_cache + cache_head_base + rot),
          out_x);
      store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(
                          k_cache + cache_head_base + HALF_ROT_DIM + rot),
                      out_y);
    }
  }

  const int64_t v_cache_base = cache_idx * v_cache_row_stride;
  for (int col = threadIdx.x * 2; col < row_dim; col += blockDim.x * 2) {
    const __mt_bfloat162 vv =
        *reinterpret_cast<const __mt_bfloat162 *>(value + v_token_base + col);
    store_bfloat162(reinterpret_cast<__mt_bfloat162 *>(v_cache + v_cache_base + col),
                    vv);
  }
}

template <typename index_t, int ROT_DIM>
void launch_rotary_embedding_cache_neox_bf16_h2(
    ffi::TensorView positions, ffi::TensorView query, ffi::TensorView key,
    ffi::TensorView value, ffi::TensorView cos_sin_cache,
    ffi::TensorView k_cache, ffi::TensorView v_cache, ffi::TensorView indices,
    int64_t num_tokens, int64_t num_heads, int64_t num_kv_heads,
    int64_t query_stride, int64_t key_stride, int64_t query_head_stride,
    int64_t key_head_stride) {
  constexpr int threads = 256;
  musaStream_t stream = get_stream(query.device());
  rotary_embedding_cache_neox_bf16_h2_kernel<ROT_DIM, index_t>
      <<<static_cast<int>(num_tokens), threads, 0, stream>>>(
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<__mt_bfloat16 *>(query.data_ptr()),
          static_cast<__mt_bfloat16 *>(key.data_ptr()),
          static_cast<const __mt_bfloat16 *>(value.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()),
          static_cast<int>(num_heads), static_cast<int>(num_kv_heads),
          static_cast<int>(query_stride), static_cast<int>(key_stride),
          static_cast<int>(value.stride(0)),
          static_cast<int>(query_head_stride),
          static_cast<int>(key_head_stride),
          static_cast<int64_t>(k_cache.stride(0)),
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)));
}

template <typename index_t, int ROT_DIM>
void launch_rotary_embedding_cache_qout_neox_bf16_h2(
    ffi::TensorView positions, ffi::TensorView query, ffi::TensorView q_out,
    ffi::TensorView key, ffi::TensorView value, ffi::TensorView cos_sin_cache,
    ffi::TensorView k_cache, ffi::TensorView v_cache, ffi::TensorView indices,
    int64_t num_tokens, int64_t num_heads, int64_t num_kv_heads,
    int64_t query_stride, int64_t key_stride, int64_t query_head_stride,
    int64_t key_head_stride) {
  constexpr int threads = 256;
  musaStream_t stream = get_stream(query.device());
  rotary_embedding_cache_qout_neox_bf16_h2_kernel<ROT_DIM, index_t>
      <<<static_cast<int>(num_tokens), threads, 0, stream>>>(
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(query.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(key.data_ptr()),
          static_cast<const __mt_bfloat16 *>(value.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()),
          static_cast<int>(num_heads), static_cast<int>(num_kv_heads),
          static_cast<int>(query_stride), static_cast<int>(key_stride),
          static_cast<int>(value.stride(0)),
          static_cast<int>(query_head_stride),
          static_cast<int>(key_head_stride),
          static_cast<int64_t>(k_cache.stride(0)),
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)));
}

template <typename index_t, typename vec_t>
__global__ void store_rotary_kv_cache_kernel(
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

template <typename index_t>
void launch_store_rotary_kv_cache(ffi::TensorView key, ffi::TensorView value,
                                  ffi::TensorView k_cache,
                                  ffi::TensorView v_cache,
                                  ffi::TensorView indices,
                                  int64_t num_tokens, int64_t key_stride,
                                  int64_t row_dim) {
  if (num_tokens == 0) {
    return;
  }

  const int64_t dtype_bytes = key.dtype().bits / 8;
  const int64_t row_bytes = row_dim * dtype_bytes;
  const int64_t k_stride_bytes = key_stride * dtype_bytes;
  const int64_t v_stride_bytes =
      static_cast<int64_t>(value.stride(0)) * dtype_bytes;
  const int64_t k_cache_stride_bytes =
      static_cast<int64_t>(k_cache.stride(0)) * dtype_bytes;
  const int64_t v_cache_stride_bytes =
      static_cast<int64_t>(v_cache.stride(0)) * dtype_bytes;
  const int64_t indices_stride = static_cast<int64_t>(indices.stride(0));
  constexpr int threads = 64;
  musaStream_t stream = get_stream(key.device());

  const uintptr_t ptr_or = reinterpret_cast<uintptr_t>(key.data_ptr()) |
                           reinterpret_cast<uintptr_t>(value.data_ptr()) |
                           reinterpret_cast<uintptr_t>(k_cache.data_ptr()) |
                           reinterpret_cast<uintptr_t>(v_cache.data_ptr());
  if ((row_bytes % 16 == 0) &&
      (((ptr_or | k_stride_bytes | v_stride_bytes | k_cache_stride_bytes |
         v_cache_stride_bytes) &
        0xF) == 0)) {
    store_rotary_kv_cache_kernel<index_t, uint4>
        <<<num_tokens, threads, 0, stream>>>(
            static_cast<const char *>(key.data_ptr()),
            static_cast<const char *>(value.data_ptr()),
            static_cast<char *>(k_cache.data_ptr()),
            static_cast<char *>(v_cache.data_ptr()),
            static_cast<const index_t *>(indices.data_ptr()), k_stride_bytes,
            v_stride_bytes, k_cache_stride_bytes, v_cache_stride_bytes,
            indices_stride, row_bytes, num_tokens);
  } else {
    store_rotary_kv_cache_kernel<index_t, uint32_t>
        <<<num_tokens, threads, 0, stream>>>(
            static_cast<const char *>(key.data_ptr()),
            static_cast<const char *>(value.data_ptr()),
            static_cast<char *>(k_cache.data_ptr()),
            static_cast<char *>(v_cache.data_ptr()),
            static_cast<const index_t *>(indices.data_ptr()), k_stride_bytes,
            v_stride_bytes, k_cache_stride_bytes, v_cache_stride_bytes,
            indices_stride, row_bytes, num_tokens);
  }
}

void sgl_rotary_embedding_cache(
    ffi::TensorView positions, ffi::TensorView query, ffi::TensorView key,
    ffi::TensorView value, ffi::TensorView cos_sin_cache,
    ffi::TensorView k_cache, ffi::TensorView v_cache, ffi::TensorView indices,
    int64_t num_tokens, int64_t head_size, int64_t num_heads,
    int64_t num_kv_heads, int64_t rot_dim, int64_t query_stride,
    int64_t key_stride, int64_t query_head_stride, int64_t key_head_stride,
    int64_t query_dim_stride, int64_t key_dim_stride, bool is_neox) {
  CHECK_MUSA(positions);
  CHECK_MUSA(query);
  CHECK_MUSA(key);
  CHECK_MUSA(value);
  CHECK_MUSA(cos_sin_cache);
  CHECK_MUSA(k_cache);
  CHECK_MUSA(v_cache);
  CHECK_MUSA(indices);
  TVM_FFI_ICHECK(positions.IsContiguous());
  TVM_FFI_ICHECK(cos_sin_cache.IsContiguous());
  TVM_FFI_ICHECK_EQ(value.ndim(), 2);
  TVM_FFI_ICHECK_EQ(k_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(v_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(indices.ndim(), 1);
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), cos_sin_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(key.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(value.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(k_cache.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(v_cache.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(indices.dtype(), dl_int32) ||
                 dtype_equal(indices.dtype(), dl_int64));
  TVM_FFI_ICHECK_EQ(positions.device().device_id, query.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, key.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, value.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id,
                    cos_sin_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, k_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, v_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, indices.device().device_id);
  TVM_FFI_ICHECK_GT(num_tokens, 0);
  TVM_FFI_ICHECK_GT(head_size, 0);
  TVM_FFI_ICHECK_GT(num_heads, 0);
  TVM_FFI_ICHECK_GT(num_kv_heads, 0);
  TVM_FFI_ICHECK_GT(rot_dim, 0);
  TVM_FFI_ICHECK_EQ(rot_dim % 2, 0);
  TVM_FFI_ICHECK_LE(rot_dim, head_size);
  const int64_t row_dim = num_kv_heads * head_size;
  TVM_FFI_ICHECK_EQ(value.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(value.size(1), row_dim);
  TVM_FFI_ICHECK_EQ(value.stride(1), 1);
  TVM_FFI_ICHECK_EQ(k_cache.size(1), row_dim);
  TVM_FFI_ICHECK_EQ(v_cache.size(1), row_dim);
  TVM_FFI_ICHECK_EQ(k_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(v_cache.stride(1), 1);
  TVM_FFI_ICHECK_GE(indices.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ((row_dim * query.dtype().bits) % 32, 0);

  ffi::MUSADeviceGuard device_guard(query.device().device_id);
  musaStream_t stream = get_stream(query.device());

  const bool can_use_fused_cache_kernel =
      dtype_equal(query.dtype(), dl_bfloat16) && is_neox &&
      head_size == rot_dim && query_dim_stride == 1 && key_dim_stride == 1 &&
      value.stride(1) == 1 && (rot_dim == 64 || rot_dim == 128);
  if (can_use_fused_cache_kernel) {
#define DISPATCH_FUSED_CACHE(INDEX_T)                                          \
  do {                                                                         \
    if (rot_dim == 64) {                                                       \
      launch_rotary_embedding_cache_neox_bf16_h2<INDEX_T, 64>(                 \
          positions, query, key, value, cos_sin_cache, k_cache, v_cache,       \
          indices, num_tokens, num_heads, num_kv_heads, query_stride,          \
          key_stride, query_head_stride, key_head_stride);                     \
    } else {                                                                   \
      launch_rotary_embedding_cache_neox_bf16_h2<INDEX_T, 128>(                \
          positions, query, key, value, cos_sin_cache, k_cache, v_cache,       \
          indices, num_tokens, num_heads, num_kv_heads, query_stride,          \
          key_stride, query_head_stride, key_head_stride);                     \
    }                                                                          \
  } while (0)

    if (dtype_equal(indices.dtype(), dl_int32)) {
      DISPATCH_FUSED_CACHE(int32_t);
    } else {
      DISPATCH_FUSED_CACHE(int64_t);
    }
#undef DISPATCH_FUSED_CACHE
    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA rotary_embedding_cache fused kernel failed: "
        << musaGetErrorString(err);
    return;
  }

#define DISPATCH_DTYPE(T)                                                      \
  dispatch_rotary_embedding<T>(                                                \
      positions, query, key, cos_sin_cache, static_cast<int>(num_tokens),      \
      static_cast<int>(num_heads), static_cast<int>(num_kv_heads),             \
      static_cast<int>(rot_dim), 0, 0, 0,                                      \
      static_cast<int>(query_stride), static_cast<int>(key_stride),            \
      static_cast<int>(query_head_stride),                                     \
      static_cast<int>(key_head_stride), static_cast<int>(query_dim_stride),   \
      static_cast<int>(key_dim_stride), is_neox, true, stream)

  if (dtype_equal(query.dtype(), dl_float16)) {
    DISPATCH_DTYPE(half);
  } else if (dtype_equal(query.dtype(), dl_bfloat16)) {
    DISPATCH_DTYPE(__mt_bfloat16);
  } else if (dtype_equal(query.dtype(), dl_float32)) {
    DISPATCH_DTYPE(float);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported rotary_embedding_cache dtype";
  }
#undef DISPATCH_DTYPE

  if (dtype_equal(indices.dtype(), dl_int32)) {
    launch_store_rotary_kv_cache<int32_t>(key, value, k_cache, v_cache,
                                          indices, num_tokens, key_stride,
                                          row_dim);
  } else {
    launch_store_rotary_kv_cache<int64_t>(key, value, k_cache, v_cache,
                                          indices, num_tokens, key_stride,
                                          row_dim);
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA rotary_embedding_cache kernel failed: "
      << musaGetErrorString(err);
}

void sgl_rotary_embedding_cache_qout(
    ffi::TensorView positions, ffi::TensorView query, ffi::TensorView q_out,
    ffi::TensorView key, ffi::TensorView value, ffi::TensorView cos_sin_cache,
    ffi::TensorView k_cache, ffi::TensorView v_cache, ffi::TensorView indices,
    int64_t num_tokens, int64_t head_size, int64_t num_heads,
    int64_t num_kv_heads, int64_t rot_dim, int64_t query_stride,
    int64_t key_stride, int64_t query_head_stride, int64_t key_head_stride,
    int64_t query_dim_stride, int64_t key_dim_stride, bool is_neox) {
  CHECK_MUSA(positions);
  CHECK_MUSA(query);
  CHECK_MUSA(q_out);
  CHECK_MUSA(key);
  CHECK_MUSA(value);
  CHECK_MUSA(cos_sin_cache);
  CHECK_MUSA(k_cache);
  CHECK_MUSA(v_cache);
  CHECK_MUSA(indices);
  TVM_FFI_ICHECK(positions.IsContiguous());
  TVM_FFI_ICHECK(q_out.IsContiguous());
  TVM_FFI_ICHECK(cos_sin_cache.IsContiguous());
  TVM_FFI_ICHECK_EQ(value.ndim(), 2);
  TVM_FFI_ICHECK_EQ(k_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(v_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(indices.ndim(), 1);
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), dl_bfloat16));
  TVM_FFI_ICHECK(dtype_equal(q_out.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(key.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(value.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(cos_sin_cache.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(k_cache.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(v_cache.dtype(), query.dtype()));
  TVM_FFI_ICHECK(dtype_equal(indices.dtype(), dl_int32) ||
                 dtype_equal(indices.dtype(), dl_int64));
  TVM_FFI_ICHECK_EQ(positions.device().device_id, query.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, q_out.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, key.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, value.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id,
                    cos_sin_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, k_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, v_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, indices.device().device_id);
  TVM_FFI_ICHECK_GT(num_tokens, 0);
  TVM_FFI_ICHECK_GT(head_size, 0);
  TVM_FFI_ICHECK_GT(num_heads, 0);
  TVM_FFI_ICHECK_GT(num_kv_heads, 0);
  TVM_FFI_ICHECK_EQ(head_size, rot_dim);
  TVM_FFI_ICHECK(is_neox);
  TVM_FFI_ICHECK(query_dim_stride == 1);
  TVM_FFI_ICHECK(key_dim_stride == 1);
  TVM_FFI_ICHECK(rot_dim == 64 || rot_dim == 128);
  const int64_t q_row_dim = num_heads * head_size;
  const int64_t row_dim = num_kv_heads * head_size;
  TVM_FFI_ICHECK_EQ(q_out.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(q_out.size(1), q_row_dim);
  TVM_FFI_ICHECK_EQ(value.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(value.size(1), row_dim);
  TVM_FFI_ICHECK_EQ(value.stride(1), 1);
  TVM_FFI_ICHECK_EQ(k_cache.size(1), row_dim);
  TVM_FFI_ICHECK_EQ(v_cache.size(1), row_dim);
  TVM_FFI_ICHECK_EQ(k_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(v_cache.stride(1), 1);
  TVM_FFI_ICHECK_GE(indices.size(0), num_tokens);

  ffi::MUSADeviceGuard device_guard(query.device().device_id);

#define DISPATCH_FUSED_CACHE_QOUT(INDEX_T)                                      \
  do {                                                                         \
    if (rot_dim == 64) {                                                       \
      launch_rotary_embedding_cache_qout_neox_bf16_h2<INDEX_T, 64>(            \
          positions, query, q_out, key, value, cos_sin_cache, k_cache,         \
          v_cache, indices, num_tokens, num_heads, num_kv_heads, query_stride, \
          key_stride, query_head_stride, key_head_stride);                     \
    } else {                                                                   \
      launch_rotary_embedding_cache_qout_neox_bf16_h2<INDEX_T, 128>(           \
          positions, query, q_out, key, value, cos_sin_cache, k_cache,         \
          v_cache, indices, num_tokens, num_heads, num_kv_heads, query_stride, \
          key_stride, query_head_stride, key_head_stride);                     \
    }                                                                          \
  } while (0)

  if (dtype_equal(indices.dtype(), dl_int32)) {
    DISPATCH_FUSED_CACHE_QOUT(int32_t);
  } else {
    DISPATCH_FUSED_CACHE_QOUT(int64_t);
  }
#undef DISPATCH_FUSED_CACHE_QOUT
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA rotary_embedding_cache_qout fused kernel failed: "
      << musaGetErrorString(err);
}

void sgl_rotary_pos_emb(ffi::TensorView query, ffi::TensorView key,
                        ffi::TensorView cos, ffi::TensorView sin,
                        ffi::TensorView query_out, ffi::TensorView key_out,
                        int64_t num_tokens, int64_t seq_len,
                        int64_t head_size, int64_t num_heads,
                        int64_t num_kv_heads, int64_t query_batch_stride,
                        int64_t key_batch_stride,
                        int64_t query_out_batch_stride,
                        int64_t key_out_batch_stride, int64_t query_stride,
                        int64_t key_stride, int64_t query_out_stride,
                        int64_t key_out_stride, int64_t query_head_stride,
                        int64_t key_head_stride,
                        int64_t query_out_head_stride,
                        int64_t key_out_head_stride,
                        int64_t query_dim_stride, int64_t key_dim_stride,
                        int64_t query_out_dim_stride,
                        int64_t key_out_dim_stride, int64_t cos_batch_stride,
                        int64_t sin_batch_stride, int64_t cos_stride,
                        int64_t sin_stride, int64_t cos_dim_stride,
                        int64_t sin_dim_stride) {
  CHECK_MUSA(query);
  CHECK_MUSA(key);
  CHECK_MUSA(cos);
  CHECK_MUSA(sin);
  CHECK_MUSA(query_out);
  CHECK_MUSA(key_out);
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), key.dtype()));
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), cos.dtype()));
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), sin.dtype()));
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), query_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), key_out.dtype()));
  TVM_FFI_ICHECK(query.ndim() == 3 || query.ndim() == 4);
  TVM_FFI_ICHECK_EQ(key.ndim(), query.ndim());
  TVM_FFI_ICHECK(cos.ndim() == 2 || cos.ndim() == 3);
  TVM_FFI_ICHECK_EQ(sin.ndim(), cos.ndim());
  TVM_FFI_ICHECK_EQ(query_out.ndim(), query.ndim());
  TVM_FFI_ICHECK_EQ(key_out.ndim(), key.ndim());
  TVM_FFI_ICHECK_GT(num_tokens, 0);
  TVM_FFI_ICHECK_GT(seq_len, 0);
  TVM_FFI_ICHECK_GT(head_size, 0);
  TVM_FFI_ICHECK_EQ(head_size % 2, 0);

  ffi::MUSADeviceGuard device_guard(query.device().device_id);
  musaStream_t stream = get_stream(query.device());
  const int threads = 256;
  const int h2_threads = 256;
  const int max_elems = static_cast<int>(max(num_heads, num_kv_heads) * head_size);
  const dim3 grid(static_cast<unsigned int>(num_tokens),
                  static_cast<unsigned int>((max_elems + threads - 1) / threads));
  const int max_pairs =
      static_cast<int>(max(num_heads, num_kv_heads) * (head_size / 2));
  const bool can_use_h2 =
      head_size % 4 == 0 && head_size <= 256 && query_dim_stride == 1 &&
      key_dim_stride == 1 && query_out_dim_stride == 1 &&
      key_out_dim_stride == 1 &&
      cos_dim_stride == 1 && sin_dim_stride == 1 &&
      even_stride(static_cast<int>(query_batch_stride)) &&
      even_stride(static_cast<int>(key_batch_stride)) &&
      even_stride(static_cast<int>(query_out_batch_stride)) &&
      even_stride(static_cast<int>(key_out_batch_stride)) &&
      even_stride(static_cast<int>(query_stride)) &&
      even_stride(static_cast<int>(key_stride)) &&
      even_stride(static_cast<int>(query_out_stride)) &&
      even_stride(static_cast<int>(key_out_stride)) &&
      even_stride(static_cast<int>(query_head_stride)) &&
      even_stride(static_cast<int>(key_head_stride)) &&
      even_stride(static_cast<int>(query_out_head_stride)) &&
      even_stride(static_cast<int>(key_out_head_stride)) &&
      even_stride(static_cast<int>(cos_batch_stride)) &&
      even_stride(static_cast<int>(sin_batch_stride)) &&
      even_stride(static_cast<int>(cos_stride)) &&
      even_stride(static_cast<int>(sin_stride));
  const bool flat_cos_sin =
      cos_stride == head_size && sin_stride == head_size &&
      ((cos_batch_stride == 0 && sin_batch_stride == 0 &&
        num_tokens == seq_len) ||
       (cos_batch_stride == seq_len * head_size &&
        sin_batch_stride == seq_len * head_size));
  const bool can_use_flat_h2 =
      can_use_h2 && query_stride == num_heads * head_size &&
      key_stride == num_kv_heads * head_size &&
      query_out_stride == num_heads * head_size &&
      key_out_stride == num_kv_heads * head_size &&
      query_head_stride == head_size && key_head_stride == head_size &&
      query_out_head_stride == head_size &&
      key_out_head_stride == head_size &&
      (query_batch_stride == 0 ||
       query_batch_stride == seq_len * query_stride) &&
      (key_batch_stride == 0 || key_batch_stride == seq_len * key_stride) &&
      (query_out_batch_stride == 0 ||
       query_out_batch_stride == seq_len * query_out_stride) &&
      (key_out_batch_stride == 0 ||
       key_out_batch_stride == seq_len * key_out_stride) &&
      flat_cos_sin;

  if (can_use_flat_h2 && dtype_equal(query.dtype(), dl_float16)) {
    if (num_tokens <= 4096 && num_heads == 16 && num_kv_heads == 16 &&
        head_size == 80) {
      const dim3 grid(static_cast<unsigned int>(num_tokens), 2);
      const dim3 block(160);
      rotary_pos_emb_neox_fp16_16h80_flat_kernel<<<grid, block, 0, stream>>>(
          static_cast<const half *>(query.data_ptr()),
          static_cast<const half *>(key.data_ptr()),
          static_cast<const half *>(cos.data_ptr()),
          static_cast<const half *>(sin.data_ptr()),
          static_cast<half *>(query_out.data_ptr()),
          static_cast<half *>(key_out.data_ptr()), static_cast<int>(num_tokens));
      const musaError_t err = musaGetLastError();
      TVM_FFI_ICHECK_EQ(err, musaSuccess)
          << "MUSA rotary_pos_emb fp16 16h80 flat kernel failed: "
          << musaGetErrorString(err);
      return;
    }

    if (head_size == 256) {
      const dim3 h2_grid(
          static_cast<unsigned int>(num_tokens),
          static_cast<unsigned int>(
              ((max_pairs + 1) / 2 + h2_threads - 1) / h2_threads));
      rotary_pos_emb_neox_h2_flat_hs256_kernel<half, half2>
          <<<h2_grid, h2_threads, 0, stream>>>(
              static_cast<const half *>(query.data_ptr()),
              static_cast<const half *>(key.data_ptr()),
              static_cast<const half *>(cos.data_ptr()),
              static_cast<const half *>(sin.data_ptr()),
              static_cast<half *>(query_out.data_ptr()),
              static_cast<half *>(key_out.data_ptr()),
              static_cast<int>(num_tokens), static_cast<int>(num_heads),
              static_cast<int>(num_kv_heads));
      const musaError_t err = musaGetLastError();
      TVM_FFI_ICHECK_EQ(err, musaSuccess)
          << "MUSA rotary_pos_emb hs256 flat h2 kernel failed: "
          << musaGetErrorString(err);
      return;
    }

    const dim3 h2_grid(
        static_cast<unsigned int>(num_tokens),
        static_cast<unsigned int>(
            ((max_pairs + 1) / 2 + h2_threads - 1) / h2_threads));
    rotary_pos_emb_neox_h2_flat_kernel<half, half2>
        <<<h2_grid, h2_threads, 0, stream>>>(
            static_cast<const half *>(query.data_ptr()),
            static_cast<const half *>(key.data_ptr()),
            static_cast<const half *>(cos.data_ptr()),
            static_cast<const half *>(sin.data_ptr()),
            static_cast<half *>(query_out.data_ptr()),
            static_cast<half *>(key_out.data_ptr()),
            static_cast<int>(num_tokens), static_cast<int>(num_heads),
            static_cast<int>(num_kv_heads), static_cast<int>(head_size));
    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA rotary_pos_emb flat h2 kernel failed: "
        << musaGetErrorString(err);
    return;
  }

  if (can_use_flat_h2 && dtype_equal(query.dtype(), dl_bfloat16)) {
    if (head_size == 256) {
      const dim3 h2_grid(
          static_cast<unsigned int>(num_tokens),
          static_cast<unsigned int>(
              ((max_pairs + 1) / 2 + h2_threads - 1) / h2_threads));
      rotary_pos_emb_neox_h2_flat_hs256_kernel<__mt_bfloat16, __mt_bfloat162>
          <<<h2_grid, h2_threads, 0, stream>>>(
              static_cast<const __mt_bfloat16 *>(query.data_ptr()),
              static_cast<const __mt_bfloat16 *>(key.data_ptr()),
              static_cast<const __mt_bfloat16 *>(cos.data_ptr()),
              static_cast<const __mt_bfloat16 *>(sin.data_ptr()),
              static_cast<__mt_bfloat16 *>(query_out.data_ptr()),
              static_cast<__mt_bfloat16 *>(key_out.data_ptr()),
              static_cast<int>(num_tokens), static_cast<int>(num_heads),
              static_cast<int>(num_kv_heads));
      const musaError_t err = musaGetLastError();
      TVM_FFI_ICHECK_EQ(err, musaSuccess)
          << "MUSA rotary_pos_emb hs256 flat h2 kernel failed: "
          << musaGetErrorString(err);
      return;
    }

    const dim3 h2_grid(
        static_cast<unsigned int>(num_tokens),
        static_cast<unsigned int>(
            ((max_pairs + 1) / 2 + h2_threads - 1) / h2_threads));
    rotary_pos_emb_neox_h2_flat_kernel<__mt_bfloat16, __mt_bfloat162>
        <<<h2_grid, h2_threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(query.data_ptr()),
            static_cast<const __mt_bfloat16 *>(key.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos.data_ptr()),
            static_cast<const __mt_bfloat16 *>(sin.data_ptr()),
            static_cast<__mt_bfloat16 *>(query_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(key_out.data_ptr()),
            static_cast<int>(num_tokens), static_cast<int>(num_heads),
            static_cast<int>(num_kv_heads), static_cast<int>(head_size));
    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA rotary_pos_emb flat h2 kernel failed: "
        << musaGetErrorString(err);
    return;
  }

  if (can_use_h2 && dtype_equal(query.dtype(), dl_float16)) {
    const dim3 h2_grid(
        static_cast<unsigned int>(num_tokens),
        static_cast<unsigned int>(
            ((max_pairs + 1) / 2 + h2_threads - 1) / h2_threads));
    rotary_pos_emb_neox_h2_kernel<half, half2>
        <<<h2_grid, h2_threads, 0, stream>>>(
            static_cast<const half *>(query.data_ptr()),
            static_cast<const half *>(key.data_ptr()),
            static_cast<const half *>(cos.data_ptr()),
            static_cast<const half *>(sin.data_ptr()),
            static_cast<half *>(query_out.data_ptr()),
            static_cast<half *>(key_out.data_ptr()),
            static_cast<int>(num_tokens), static_cast<int>(seq_len),
            static_cast<int>(num_heads), static_cast<int>(num_kv_heads),
            static_cast<int>(head_size), static_cast<int>(query_batch_stride),
            static_cast<int>(key_batch_stride),
            static_cast<int>(query_out_batch_stride),
            static_cast<int>(key_out_batch_stride),
            static_cast<int>(query_stride), static_cast<int>(key_stride),
            static_cast<int>(query_out_stride), static_cast<int>(key_out_stride),
            static_cast<int>(query_head_stride),
            static_cast<int>(key_head_stride),
            static_cast<int>(query_out_head_stride),
            static_cast<int>(key_out_head_stride),
            static_cast<int>(cos_batch_stride),
            static_cast<int>(sin_batch_stride), static_cast<int>(cos_stride),
            static_cast<int>(sin_stride));
    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA rotary_pos_emb h2 kernel failed: " << musaGetErrorString(err);
    return;
  }

  if (can_use_h2 && dtype_equal(query.dtype(), dl_bfloat16)) {
    const dim3 h2_grid(
        static_cast<unsigned int>(num_tokens),
        static_cast<unsigned int>(
            ((max_pairs + 1) / 2 + h2_threads - 1) / h2_threads));
    rotary_pos_emb_neox_h2_kernel<__mt_bfloat16, __mt_bfloat162>
        <<<h2_grid, h2_threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(query.data_ptr()),
            static_cast<const __mt_bfloat16 *>(key.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos.data_ptr()),
            static_cast<const __mt_bfloat16 *>(sin.data_ptr()),
            static_cast<__mt_bfloat16 *>(query_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(key_out.data_ptr()),
            static_cast<int>(num_tokens), static_cast<int>(seq_len),
            static_cast<int>(num_heads), static_cast<int>(num_kv_heads),
            static_cast<int>(head_size), static_cast<int>(query_batch_stride),
            static_cast<int>(key_batch_stride),
            static_cast<int>(query_out_batch_stride),
            static_cast<int>(key_out_batch_stride),
            static_cast<int>(query_stride), static_cast<int>(key_stride),
            static_cast<int>(query_out_stride), static_cast<int>(key_out_stride),
            static_cast<int>(query_head_stride),
            static_cast<int>(key_head_stride),
            static_cast<int>(query_out_head_stride),
            static_cast<int>(key_out_head_stride),
            static_cast<int>(cos_batch_stride),
            static_cast<int>(sin_batch_stride), static_cast<int>(cos_stride),
            static_cast<int>(sin_stride));
    const musaError_t err = musaGetLastError();
    TVM_FFI_ICHECK_EQ(err, musaSuccess)
        << "MUSA rotary_pos_emb h2 kernel failed: " << musaGetErrorString(err);
    return;
  }

#define DISPATCH_DTYPE(T)                                                      \
  rotary_pos_emb_neox_kernel<T><<<grid, threads, 0, stream>>>(                 \
      static_cast<const T *>(query.data_ptr()),                                \
      static_cast<const T *>(key.data_ptr()),                                  \
      static_cast<const T *>(cos.data_ptr()),                                  \
      static_cast<const T *>(sin.data_ptr()),                                  \
      static_cast<T *>(query_out.data_ptr()),                                  \
      static_cast<T *>(key_out.data_ptr()), static_cast<int>(num_tokens),      \
      static_cast<int>(seq_len), static_cast<int>(num_heads),                  \
      static_cast<int>(num_kv_heads), static_cast<int>(head_size),             \
      static_cast<int>(query_batch_stride),                                    \
      static_cast<int>(key_batch_stride),                                      \
      static_cast<int>(query_out_batch_stride),                                \
      static_cast<int>(key_out_batch_stride), static_cast<int>(query_stride),  \
      static_cast<int>(key_stride), static_cast<int>(query_out_stride),        \
      static_cast<int>(key_out_stride), static_cast<int>(query_head_stride),    \
      static_cast<int>(key_head_stride),                                       \
      static_cast<int>(query_out_head_stride),                                 \
      static_cast<int>(key_out_head_stride),                                   \
      static_cast<int>(query_dim_stride), static_cast<int>(key_dim_stride),    \
      static_cast<int>(query_out_dim_stride),                                  \
      static_cast<int>(key_out_dim_stride), static_cast<int>(cos_batch_stride), \
      static_cast<int>(sin_batch_stride), static_cast<int>(cos_stride),        \
      static_cast<int>(sin_stride), static_cast<int>(cos_dim_stride),          \
      static_cast<int>(sin_dim_stride))

  if (dtype_equal(query.dtype(), dl_float16)) {
    DISPATCH_DTYPE(half);
  } else if (dtype_equal(query.dtype(), dl_bfloat16)) {
    DISPATCH_DTYPE(__mt_bfloat16);
  } else if (dtype_equal(query.dtype(), dl_float32)) {
    DISPATCH_DTYPE(float);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported rotary_pos_emb dtype";
  }
#undef DISPATCH_DTYPE

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA rotary_pos_emb kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_rotary_embedding, sgl_rotary_embedding);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_rotary_embedding_cache,
                              sgl_rotary_embedding_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_rotary_embedding_cache_qout,
                              sgl_rotary_embedding_cache_qout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_rotary_pos_emb, sgl_rotary_pos_emb);
