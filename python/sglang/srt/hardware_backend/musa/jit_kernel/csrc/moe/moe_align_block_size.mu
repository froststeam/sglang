#include <algorithm>
#include <cstdint>

#include <musa_runtime.h>
#include <tvm/ffi/extra/musa/device_guard.h>

#include "../common.h"

namespace {

constexpr int kVecSize = 4;
constexpr int kWarpSize = 32;

struct Int4 {
  int32_t x;
  int32_t y;
  int32_t z;
  int32_t w;
};

__device__ __forceinline__ int32_t warp_exclusive_scan(int32_t v) {
  const int32_t original = v;
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const int32_t n = __shfl_up_sync(0xffffffff, v, offset);
    if ((threadIdx.x & (kWarpSize - 1)) >= offset) {
      v += n;
    }
  }
  return v - original;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ int32_t ceil_to_block_size_const(int32_t count) {
  if constexpr ((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0) {
    return (count + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
  } else {
    return ((count + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  }
}

template <int BLOCK_SIZE>
__device__ __forceinline__ int32_t block_index_const(int32_t offset) {
  static_assert(BLOCK_SIZE == 8 || BLOCK_SIZE == 16 || BLOCK_SIZE == 32 ||
                    BLOCK_SIZE == 48 || BLOCK_SIZE == 64 ||
                    BLOCK_SIZE == 128 || BLOCK_SIZE == 256 ||
                    BLOCK_SIZE == 512,
                "Unsupported compile-time block size");
  if constexpr (BLOCK_SIZE == 8) {
    return offset >> 3;
  } else if constexpr (BLOCK_SIZE == 16) {
    return offset >> 4;
  } else if constexpr (BLOCK_SIZE == 32) {
    return offset >> 5;
  } else if constexpr (BLOCK_SIZE == 64) {
    return offset >> 6;
  } else if constexpr (BLOCK_SIZE == 128) {
    return offset >> 7;
  } else if constexpr (BLOCK_SIZE == 256) {
    return offset >> 8;
  } else if constexpr (BLOCK_SIZE == 512) {
    return offset >> 9;
  } else {
    return offset / BLOCK_SIZE;
  }
}

inline int64_t ceil_div_host(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int64_t next_pow2_host(int64_t v) {
  int64_t out = 1;
  while (out < v) {
    out <<= 1;
  }
  return out;
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    const int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
  }
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_blocked_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel,
    int32_t num_experts) {
  extern __shared__ int32_t smem[];
  int32_t* counts = smem;
  int32_t* bases = counts + num_experts;

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t idx = static_cast<int32_t>(blockIdx.x) *
                          static_cast<int32_t>(blockDim.x) +
                      tid;
  const int32_t numel_i = static_cast<int32_t>(numel);

  for (int32_t expert = tid; expert < num_experts; expert += blockDim.x) {
    counts[expert] = 0;
  }
  __syncthreads();

  int32_t expert_id = -1;
  if (idx < numel_i) {
    expert_id = static_cast<int32_t>(topk_ids[idx]) + 1;
    atomicAdd(&counts[expert_id], 1);
  }
  __syncthreads();

  for (int32_t expert = tid; expert < num_experts; expert += blockDim.x) {
    const int32_t count = counts[expert];
    if (count > 0) {
      bases[expert] = atomicAdd(&cumsum_buffer[expert], count);
    }
    counts[expert] = 0;
  }
  __syncthreads();

  if (idx < numel_i) {
    const int32_t local_rank = atomicAdd(&counts[expert_id], 1);
    sorted_token_ids[bases[expert_id] + local_rank] =
        static_cast<int32_t>(idx);
  }
}

template <typename scalar_t>
__global__ void count_expert_tokens_by_chunk_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ block_counts,
    int32_t num_experts,
    size_t numel,
    int32_t chunk_size) {
  extern __shared__ int32_t counts[];
  const int32_t chunk = blockIdx.x;
  const int32_t tid = threadIdx.x;

  for (int32_t e = tid; e < num_experts; e += blockDim.x) {
    counts[e] = 0;
  }
  __syncthreads();

  const size_t begin = static_cast<size_t>(chunk) * chunk_size;
  const size_t end = min(begin + static_cast<size_t>(chunk_size), numel);
  for (size_t i = begin + tid; i < end; i += blockDim.x) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    atomicAdd(&counts[expert_id], 1);
  }
  __syncthreads();

  int32_t* out = block_counts + static_cast<size_t>(chunk) * num_experts;
  for (int32_t e = tid; e < num_experts; e += blockDim.x) {
    out[e] = counts[e];
  }
}

__global__ void prefix_expert_tokens_by_chunk_kernel(
    const int32_t* __restrict__ block_counts,
    int32_t* __restrict__ block_offsets,
    int32_t num_experts,
    int32_t num_chunks) {
  const int32_t expert_id = blockIdx.x;
  if (expert_id >= num_experts || threadIdx.x != 0) {
    return;
  }

  int32_t running = 0;
  for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
    const size_t idx = static_cast<size_t>(chunk) * num_experts + expert_id;
    block_offsets[idx] = running;
    running += block_counts[idx];
  }
  block_offsets[static_cast<size_t>(num_chunks) * num_experts + expert_id] =
      running;
}

template <int BLOCK_SIZE>
__global__ void finalize_moe_align_large_sort_kernel(
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ cumsum,
    const int32_t* __restrict__ block_offsets,
    int32_t num_experts,
    int32_t num_chunks,
    int32_t numel,
    int32_t scan_size,
    bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
  extern __shared__ int32_t scan_buf[];
  const int32_t tid = threadIdx.x;

  if (tid < scan_size) {
    if (tid < num_experts) {
      const int32_t count =
          block_offsets[static_cast<size_t>(num_chunks) * num_experts + tid];
      scan_buf[tid] = ceil_to_block_size_const<BLOCK_SIZE>(count);
    } else {
      scan_buf[tid] = 0;
    }
  }
  __syncthreads();

  int offset = 1;
  for (int d = scan_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      const int ai = offset * (2 * tid + 1) - 1;
      const int bi = offset * (2 * tid + 2) - 1;
      scan_buf[bi] += scan_buf[ai];
    }
    offset <<= 1;
    __syncthreads();
  }

  if (tid == 0) {
    *total_tokens_post_pad = scan_buf[scan_size - 1];
    scan_buf[scan_size - 1] = 0;
  }
  __syncthreads();

  for (int d = 1; d < scan_size; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      const int ai = offset * (2 * tid + 1) - 1;
      const int bi = offset * (2 * tid + 2) - 1;
      const int temp = scan_buf[ai];
      scan_buf[ai] = scan_buf[bi];
      scan_buf[bi] += temp;
    }
    __syncthreads();
  }

  if (tid < num_experts) {
    cumsum[tid] = scan_buf[tid];
  }
  if (tid == 0) {
    cumsum[num_experts] = *total_tokens_post_pad;
  }
  __syncthreads();

  for (int32_t e = tid; e < num_experts; e += blockDim.x) {
    for (int32_t block_start = cumsum[e]; block_start < cumsum[e + 1];
         block_start += BLOCK_SIZE) {
      expert_ids[block_index_const<BLOCK_SIZE>(block_start)] = e - 1;
    }
  }
}

__global__ void fill_sorted_token_ids_kernel(int32_t* __restrict__ sorted_token_ids,
                                             int32_t numel,
                                             int32_t max_num_tokens_padded) {
  const Int4 fill_vec{numel, numel, numel, numel};
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride = blockDim.x * gridDim.x;
  const int32_t num_vecs = max_num_tokens_padded / kVecSize;
  Int4* out_vec = reinterpret_cast<Int4*>(sorted_token_ids);
  for (int32_t i = tid; i < num_vecs; i += stride) {
    out_vec[i] = fill_vec;
  }
  for (int32_t i = num_vecs * kVecSize + tid; i < max_num_tokens_padded;
       i += stride) {
    sorted_token_ids[i] = numel;
  }
}

template <int BLOCK_SIZE, bool PAD_SORTED_TOKEN_IDS>
__global__ void finalize_moe_align_chunk_counts_kernel(
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ cumsum,
    const int32_t* __restrict__ block_counts,
    int32_t num_experts,
    int32_t num_chunks,
    int32_t scan_size,
    int32_t max_num_tokens_padded,
    int32_t numel) {
  if (blockIdx.x == 1) {
    if constexpr (PAD_SORTED_TOKEN_IDS) {
      const Int4 fill_vec{numel, numel, numel, numel};
      const int32_t num_vecs = max_num_tokens_padded / kVecSize;
      Int4* out_vec = reinterpret_cast<Int4*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        out_vec[i] = fill_vec;
      }
      for (int32_t i = num_vecs * kVecSize + threadIdx.x;
           i < max_num_tokens_padded; i += blockDim.x) {
        sorted_token_ids[i] = numel;
      }
    }
    return;
  }

  extern __shared__ int32_t scan_buf[];
  __shared__ int32_t s_total_tokens_post_pad;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  if (tid < num_experts) {
    int32_t count = 0;
    for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
      count += block_counts[static_cast<size_t>(chunk) * num_experts + tid];
    }
    scan_buf[tid] = ceil_to_block_size_const<BLOCK_SIZE>(count);
  }
  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }
  __syncthreads();

  int32_t* warp_sums = scan_buf + scan_size;
  const int32_t warp_id = tid / kWarpSize;
  const int32_t lane = tid & (kWarpSize - 1);
  const int32_t num_scan_warps = (scan_size + kWarpSize - 1) / kWarpSize;

  int32_t v = tid < scan_size ? scan_buf[tid] : 0;
  const int32_t pre = warp_exclusive_scan(v);
  if (lane == kWarpSize - 1) {
    warp_sums[warp_id] = pre + v;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int32_t warp_sum = lane < num_scan_warps ? warp_sums[lane] : 0;
    const int32_t warp_offset = warp_exclusive_scan(warp_sum);
    if (lane < num_scan_warps) {
      warp_sums[lane] = warp_offset;
    }
    if (lane == num_scan_warps - 1) {
      s_total_tokens_post_pad = warp_offset + warp_sum;
      cumsum[num_experts] = s_total_tokens_post_pad;
      *total_tokens_post_pad = s_total_tokens_post_pad;
    }
  }
  __syncthreads();

  if (tid < scan_size) {
    scan_buf[tid] = pre + warp_sums[warp_id];
  }
  __syncthreads();

  if (tid < num_experts) {
    const int32_t begin = scan_buf[tid];
    const int32_t end =
        (tid + 1 == num_experts) ? s_total_tokens_post_pad : scan_buf[tid + 1];
    cumsum[tid] = begin;
    for (int32_t i = begin; i < end; i += BLOCK_SIZE) {
      expert_ids[block_index_const<BLOCK_SIZE>(i)] = tid - 1;
    }
  }
}

template <typename scalar_t>
__global__ void scatter_expert_tokens_by_chunk_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ cumsum,
    const int32_t* __restrict__ block_offsets,
    int32_t num_experts,
    size_t numel,
    int32_t chunk_size) {
  extern __shared__ int32_t local_offsets[];
  const int32_t chunk = blockIdx.x;
  const int32_t tid = threadIdx.x;

  for (int32_t e = tid; e < num_experts; e += blockDim.x) {
    local_offsets[e] = 0;
  }
  __syncthreads();

  const size_t begin = static_cast<size_t>(chunk) * chunk_size;
  const size_t end = min(begin + static_cast<size_t>(chunk_size), numel);
  for (size_t i = begin + tid; i < end; i += blockDim.x) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    const int32_t local_rank = atomicAdd(&local_offsets[expert_id], 1);
    const int32_t block_rank =
        block_offsets[static_cast<size_t>(chunk) * num_experts + expert_id];
    const int32_t rank_post_pad = cumsum[expert_id] + block_rank + local_rank;
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
  }
}

template <typename scalar_t>
__global__ void scatter_expert_tokens_by_chunk_direct_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ cumsum,
    const int32_t* __restrict__ block_counts,
    int32_t num_experts,
    size_t numel,
    int32_t chunk_size) {
  extern __shared__ int32_t shared_mem[];
  int32_t* local_offsets = shared_mem;
  int32_t* chunk_bases = local_offsets + num_experts;
  const int32_t chunk = blockIdx.x;
  const int32_t tid = threadIdx.x;

  for (int32_t e = tid; e < num_experts; e += blockDim.x) {
    local_offsets[e] = 0;
    int32_t base = 0;
    for (int32_t prev = 0; prev < chunk; ++prev) {
      base += block_counts[static_cast<size_t>(prev) * num_experts + e];
    }
    chunk_bases[e] = base;
  }
  __syncthreads();

  const size_t begin = static_cast<size_t>(chunk) * chunk_size;
  const size_t end = min(begin + static_cast<size_t>(chunk_size), numel);
  for (size_t i = begin + tid; i < end; i += blockDim.x) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    const int32_t local_rank = atomicAdd(&local_offsets[expert_id], 1);
    const int32_t rank_post_pad =
        cumsum[expert_id] + chunk_bases[expert_id] + local_rank;
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
  }
}

template <typename scalar_t,
          int BLOCK_SIZE,
          bool PAD_SORTED_TOKEN_IDS,
          bool DIRECT_EXPERT_FILL>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    size_t numel,
    int32_t* __restrict__ cumsum,
    int32_t scan_size,
    int32_t max_num_tokens_padded) {
  if (blockIdx.x == 1) {
    if constexpr (PAD_SORTED_TOKEN_IDS) {
      const Int4 fill_vec{static_cast<int32_t>(numel), static_cast<int32_t>(numel),
                          static_cast<int32_t>(numel), static_cast<int32_t>(numel)};
      const int32_t num_vecs = max_num_tokens_padded / kVecSize;
      Int4* out_vec = reinterpret_cast<Int4*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        out_vec[i] = fill_vec;
      }
      for (int32_t i = num_vecs * kVecSize + threadIdx.x;
           i < max_num_tokens_padded; i += blockDim.x) {
        sorted_token_ids[i] = static_cast<int32_t>(numel);
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;
  int32_t* scan_buf = shared_counts + num_experts;
  __shared__ int32_t s_total_tokens_post_pad;

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t stride = static_cast<int32_t>(blockDim.x);
  const int32_t numel_i = static_cast<int32_t>(numel);

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }
  __syncthreads();

  for (int32_t i = tid; i < numel_i; i += stride) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }
  __syncthreads();

  if (tid < num_experts) {
    const int32_t count = shared_counts[tid];
    scan_buf[tid] = ceil_to_block_size_const<BLOCK_SIZE>(count);
  }
  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }
  __syncthreads();

  int32_t* warp_sums = scan_buf + scan_size;
  const int32_t warp_id = static_cast<int32_t>(tid) / kWarpSize;
  const int32_t lane = static_cast<int32_t>(tid) & (kWarpSize - 1);
  const int32_t num_scan_warps = (scan_size + kWarpSize - 1) / kWarpSize;

  int32_t v = tid < scan_size ? scan_buf[tid] : 0;
  const int32_t pre = warp_exclusive_scan(v);
  if (lane == kWarpSize - 1) {
    warp_sums[warp_id] = pre + v;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int32_t warp_sum = lane < num_scan_warps ? warp_sums[lane] : 0;
    const int32_t warp_offset = warp_exclusive_scan(warp_sum);
    if (lane < num_scan_warps) {
      warp_sums[lane] = warp_offset;
    }
    if (lane == num_scan_warps - 1) {
      s_total_tokens_post_pad = warp_offset + warp_sum;
      *total_tokens_post_pad = s_total_tokens_post_pad;
    }
  }
  __syncthreads();

  if (tid < scan_size) {
    scan_buf[tid] = pre + warp_sums[warp_id];
  }
  __syncthreads();

  if (tid < num_experts) {
    cumsum[tid] = scan_buf[tid];
  } else if (tid == num_experts) {
    cumsum[tid] = s_total_tokens_post_pad;
  }

  if constexpr (DIRECT_EXPERT_FILL) {
    if (tid < num_experts) {
      const int32_t begin = scan_buf[tid];
      const int32_t end = (tid + 1 == num_experts)
                              ? s_total_tokens_post_pad
                              : scan_buf[tid + 1];
      for (int32_t block_start = begin; block_start < end;
           block_start += BLOCK_SIZE) {
        expert_ids[block_index_const<BLOCK_SIZE>(block_start)] =
            static_cast<int32_t>(tid) - 1;
      }
    }
  } else {
    const int32_t num_blocks =
        block_index_const<BLOCK_SIZE>(s_total_tokens_post_pad);
    for (int32_t i = static_cast<int32_t>(tid); i < num_blocks; i += stride) {
      const int32_t block_start = i * BLOCK_SIZE;
      int left = 0;
      int right = num_experts;
      while (left < right) {
        const int mid = (left + right) >> 1;
        if (scan_buf[mid] <= block_start) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
      expert_ids[i] = left - 2;
    }
  }
}

template <typename scalar_t, int BLOCK_SIZE, bool PAD_SORTED_TOKEN_IDS>
__global__ void moe_align_block_size_one_cta_atomic_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ cumsum_out,
    int32_t num_experts,
    size_t numel,
    int32_t max_num_tokens_padded,
    int32_t scan_size) {
  extern __shared__ int32_t shared_mem[];
  int32_t* counts = shared_mem;
  int32_t* prefix = counts + num_experts;
  int32_t* scan_buf = prefix + num_experts + 1;
  __shared__ int32_t s_total_tokens_post_pad;

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t stride = static_cast<int32_t>(blockDim.x);

  if constexpr (PAD_SORTED_TOKEN_IDS) {
    const Int4 fill_vec{static_cast<int32_t>(numel),
                        static_cast<int32_t>(numel),
                        static_cast<int32_t>(numel),
                        static_cast<int32_t>(numel)};
    const int32_t num_vecs = max_num_tokens_padded / kVecSize;
    Int4* out_vec = reinterpret_cast<Int4*>(sorted_token_ids);
    for (int32_t i = tid; i < num_vecs; i += stride) {
      out_vec[i] = fill_vec;
    }
    for (int32_t i = num_vecs * kVecSize + tid; i < max_num_tokens_padded;
         i += stride) {
      sorted_token_ids[i] = static_cast<int32_t>(numel);
    }
  }

  for (int32_t e = tid; e < num_experts; e += stride) {
    counts[e] = 0;
  }
  __syncthreads();

  for (size_t i = static_cast<size_t>(tid); i < numel; i += stride) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    atomicAdd(&counts[expert_id], 1);
  }
  __syncthreads();

  if (tid < num_experts) {
    const int32_t count = counts[tid];
    scan_buf[tid] = ceil_to_block_size_const<BLOCK_SIZE>(count);
  }
  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }
  __syncthreads();

  int32_t* warp_sums = scan_buf + scan_size;
  const int32_t warp_id = tid / kWarpSize;
  const int32_t lane = tid & (kWarpSize - 1);
  const int32_t num_scan_warps = (scan_size + kWarpSize - 1) / kWarpSize;

  int32_t v = tid < scan_size ? scan_buf[tid] : 0;
  const int32_t pre = warp_exclusive_scan(v);
  if (lane == kWarpSize - 1) {
    warp_sums[warp_id] = pre + v;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int32_t warp_sum = lane < num_scan_warps ? warp_sums[lane] : 0;
    const int32_t warp_offset = warp_exclusive_scan(warp_sum);
    if (lane < num_scan_warps) {
      warp_sums[lane] = warp_offset;
    }
    if (lane == num_scan_warps - 1) {
      s_total_tokens_post_pad = warp_offset + warp_sum;
      prefix[num_experts] = s_total_tokens_post_pad;
      cumsum_out[num_experts] = s_total_tokens_post_pad;
      *total_tokens_post_pad = s_total_tokens_post_pad;
    }
  }
  __syncthreads();

  if (tid < scan_size) {
    scan_buf[tid] = pre + warp_sums[warp_id];
  }
  __syncthreads();

  if (tid < num_experts) {
    const int32_t begin = scan_buf[tid];
    const int32_t end =
        (tid + 1 == num_experts) ? s_total_tokens_post_pad : scan_buf[tid + 1];
    prefix[tid] = begin;
    counts[tid] = begin;
    cumsum_out[tid] = begin;
    for (int32_t i = begin; i < end; i += BLOCK_SIZE) {
      expert_ids[block_index_const<BLOCK_SIZE>(i)] = tid - 1;
    }
  }
  __syncthreads();

  for (size_t i = static_cast<size_t>(tid); i < numel; i += stride) {
    const int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    const int32_t rank_post_pad = atomicAdd(&counts[expert_id], 1);
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
  }
}

void check_moe_align_inputs(ffi::TensorView topk_ids,
                            ffi::TensorView sorted_token_ids,
                            ffi::TensorView expert_ids,
                            ffi::TensorView num_tokens_post_pad,
                            ffi::TensorView cumsum_buffer,
                            int64_t num_experts) {
  CHECK_MUSA_CONTIGUOUS(topk_ids);
  CHECK_MUSA_CONTIGUOUS(sorted_token_ids);
  CHECK_MUSA_CONTIGUOUS(expert_ids);
  CHECK_MUSA_CONTIGUOUS(num_tokens_post_pad);
  CHECK_MUSA_CONTIGUOUS(cumsum_buffer);
  TVM_FFI_ICHECK(dtype_equal(topk_ids.dtype(), dl_int32) ||
                 dtype_equal(topk_ids.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(sorted_token_ids.dtype(), dl_int32));
  TVM_FFI_ICHECK(dtype_equal(expert_ids.dtype(), dl_int32));
  TVM_FFI_ICHECK(dtype_equal(num_tokens_post_pad.dtype(), dl_int32));
  TVM_FFI_ICHECK(dtype_equal(cumsum_buffer.dtype(), dl_int32));
  TVM_FFI_ICHECK_GE(num_experts, 1);
  TVM_FFI_ICHECK_GE(cumsum_buffer.size(0), num_experts + 1);
  TVM_FFI_ICHECK_EQ(topk_ids.device().device_id,
                    sorted_token_ids.device().device_id);
  TVM_FFI_ICHECK_EQ(topk_ids.device().device_id, expert_ids.device().device_id);
  TVM_FFI_ICHECK_EQ(topk_ids.device().device_id,
                    num_tokens_post_pad.device().device_id);
  TVM_FFI_ICHECK_EQ(topk_ids.device().device_id,
                    cumsum_buffer.device().device_id);
}

template <typename scalar_t, int BLOCK_SIZE>
void launch_moe_align_block(ffi::TensorView topk_ids,
                            int64_t num_experts,
                            ffi::TensorView sorted_token_ids,
                            ffi::TensorView expert_ids,
                            ffi::TensorView num_tokens_post_pad,
                            ffi::TensorView cumsum_buffer,
                            bool pad_sorted_token_ids) {
  const int64_t numel = tensor_numel(topk_ids);
  const int64_t max_num_tokens_padded = sorted_token_ids.size(0);
  musaStream_t stream = get_stream(topk_ids.device());

  const bool one_cta_atomic_mode =
      (numel < 4096 ||
       (numel == 4096 && num_experts == 257 && BLOCK_SIZE <= 256)) &&
      num_experts <= 257 &&
      !(numel == 2048 && num_experts == 129);
  if (one_cta_atomic_mode) {
    const int32_t scan_size = static_cast<int32_t>(next_pow2_host(num_experts));
    const int32_t min_threads = numel >= 4096 ? 512 : 256;
    const int32_t threads = static_cast<int32_t>(
        ceil_div_host(std::max<int64_t>(scan_size, min_threads), kWarpSize) *
        kWarpSize);
    const size_t shared_mem_size =
        (num_experts + (num_experts + 1) + scan_size + kWarpSize) *
        sizeof(int32_t);

    if (pad_sorted_token_ids) {
      moe_align_block_size_one_cta_atomic_kernel<scalar_t, BLOCK_SIZE, true>
          <<<1, threads, shared_mem_size, stream>>>(
              static_cast<const scalar_t*>(topk_ids.data_ptr()),
              static_cast<int32_t*>(sorted_token_ids.data_ptr()),
              static_cast<int32_t*>(expert_ids.data_ptr()),
              static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
              static_cast<int32_t*>(cumsum_buffer.data_ptr()),
              static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
              static_cast<int32_t>(max_num_tokens_padded), scan_size);
    } else {
      moe_align_block_size_one_cta_atomic_kernel<scalar_t, BLOCK_SIZE, false>
          <<<1, threads, shared_mem_size, stream>>>(
              static_cast<const scalar_t*>(topk_ids.data_ptr()),
              static_cast<int32_t*>(sorted_token_ids.data_ptr()),
              static_cast<int32_t*>(expert_ids.data_ptr()),
              static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
              static_cast<int32_t*>(cumsum_buffer.data_ptr()),
              static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
              static_cast<int32_t>(max_num_tokens_padded), scan_size);
    }
    return;
  }

  const bool use_direct_expert_fill = numel <= 32768;
  const bool use_power2_scan =
      (BLOCK_SIZE == 128 && numel == 4096 && num_experts > 65) ||
      numel > 4096;
  const int32_t scan_size = static_cast<int32_t>(
      (use_direct_expert_fill && !use_power2_scan)
          ? num_experts
          : next_pow2_host(num_experts));
  int32_t threads =
      numel == 2048 ? std::max(scan_size, 512)
                    : (numel > 2048 ? 1024 : std::max(scan_size, 256));
  threads = static_cast<int32_t>(ceil_div_host(threads, kWarpSize) * kWarpSize);
  const size_t shared_mem_size =
      (num_experts + scan_size + kWarpSize) * sizeof(int32_t);

  if (pad_sorted_token_ids && use_direct_expert_fill) {
    moe_align_block_size_kernel<scalar_t, BLOCK_SIZE, true, true>
        <<<2, threads, shared_mem_size, stream>>>(
            static_cast<const scalar_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(expert_ids.data_ptr()),
            static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
            static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
            static_cast<int32_t*>(cumsum_buffer.data_ptr()), scan_size,
            static_cast<int32_t>(max_num_tokens_padded));
  } else if (pad_sorted_token_ids) {
    moe_align_block_size_kernel<scalar_t, BLOCK_SIZE, true, false>
        <<<2, threads, shared_mem_size, stream>>>(
            static_cast<const scalar_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(expert_ids.data_ptr()),
            static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
            static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
            static_cast<int32_t*>(cumsum_buffer.data_ptr()), scan_size,
            static_cast<int32_t>(max_num_tokens_padded));
  } else if (use_direct_expert_fill) {
    moe_align_block_size_kernel<scalar_t, BLOCK_SIZE, false, true>
        <<<2, threads, shared_mem_size, stream>>>(
            static_cast<const scalar_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(expert_ids.data_ptr()),
            static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
            static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
            static_cast<int32_t*>(cumsum_buffer.data_ptr()), scan_size,
            static_cast<int32_t>(max_num_tokens_padded));
  } else {
    moe_align_block_size_kernel<scalar_t, BLOCK_SIZE, false, false>
        <<<2, threads, shared_mem_size, stream>>>(
            static_cast<const scalar_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(expert_ids.data_ptr()),
            static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
            static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
            static_cast<int32_t*>(cumsum_buffer.data_ptr()), scan_size,
            static_cast<int32_t>(max_num_tokens_padded));
  }

  const bool use_blocked_sort = numel >= 2048;
  const int block_threads =
      use_blocked_sort
          ? (numel <= 4096 ? 256 : 512)
          : std::min(256, static_cast<int>(threads));
  const int num_blocks = static_cast<int>(ceil_div_host(numel, block_threads));
  const int actual_blocks = std::min(num_blocks, 65535);

  if (use_blocked_sort) {
    const size_t sort_shared_mem_size =
        static_cast<size_t>(2 * num_experts) * sizeof(int32_t);
    count_and_sort_expert_tokens_blocked_kernel<scalar_t>
        <<<actual_blocks, block_threads, sort_shared_mem_size, stream>>>(
            static_cast<const scalar_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(cumsum_buffer.data_ptr()),
            static_cast<size_t>(numel), static_cast<int32_t>(num_experts));
  } else {
    count_and_sort_expert_tokens_kernel<scalar_t>
        <<<actual_blocks, block_threads, 0, stream>>>(
            static_cast<const scalar_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(cumsum_buffer.data_ptr()),
            static_cast<size_t>(numel));
  }
}

template <typename scalar_t>
void launch_moe_align(ffi::TensorView topk_ids,
                      int64_t num_experts,
                      int64_t block_size,
                      ffi::TensorView sorted_token_ids,
                      ffi::TensorView expert_ids,
                      ffi::TensorView num_tokens_post_pad,
                      ffi::TensorView cumsum_buffer,
                      bool pad_sorted_token_ids) {
  switch (block_size) {
    case 8:
      launch_moe_align_block<scalar_t, 8>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids);
      break;
    case 16:
      launch_moe_align_block<scalar_t, 16>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids);
      break;
    case 32:
      launch_moe_align_block<scalar_t, 32>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids);
      break;
    case 48:
      launch_moe_align_block<scalar_t, 48>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids);
      break;
    case 64:
      launch_moe_align_block<scalar_t, 64>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids);
      break;
    case 128:
      launch_moe_align_block<scalar_t, 128>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids);
      break;
    case 256:
      launch_moe_align_block<scalar_t, 256>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids);
      break;
    case 512:
      launch_moe_align_block<scalar_t, 512>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids);
      break;
    default:
      TVM_FFI_THROW(ValueError)
          << "MUSA moe_align_block_size optimized JIT supports "
             "block_size=8/16/32/48/64/128/256/512, got "
          << block_size;
  }
}

template <typename scalar_t, int BLOCK_SIZE>
void launch_moe_align_chunked_direct_block(
    ffi::TensorView topk_ids,
    int64_t num_experts,
    ffi::TensorView sorted_token_ids,
    ffi::TensorView expert_ids,
    ffi::TensorView num_tokens_post_pad,
    ffi::TensorView cumsum_buffer,
    ffi::TensorView block_workspace,
    int64_t sort_chunk_size,
    bool pad_sorted_token_ids) {
  const int64_t numel = tensor_numel(topk_ids);
  const int64_t num_chunks = ceil_div_host(numel, sort_chunk_size);
  const int64_t required_workspace = num_chunks * num_experts;
  TVM_FFI_ICHECK_GE(tensor_numel(block_workspace), required_workspace);

  musaStream_t stream = get_stream(topk_ids.device());
  int32_t* block_counts = static_cast<int32_t*>(block_workspace.data_ptr());
  const int32_t sort_threads = 256;
  const size_t count_smem = static_cast<size_t>(num_experts) * sizeof(int32_t);

  count_expert_tokens_by_chunk_kernel<scalar_t>
      <<<static_cast<int>(num_chunks), sort_threads, count_smem, stream>>>(
          static_cast<const scalar_t*>(topk_ids.data_ptr()), block_counts,
          static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
          static_cast<int32_t>(sort_chunk_size));

  const int32_t scan_size = static_cast<int32_t>(num_experts);
  const int32_t finalize_threads =
      static_cast<int32_t>(ceil_div_host(std::max<int64_t>(scan_size, 256),
                                         kWarpSize) *
                           kWarpSize);
  const size_t finalize_smem =
      static_cast<size_t>(scan_size + kWarpSize) * sizeof(int32_t);
  if (pad_sorted_token_ids) {
    finalize_moe_align_chunk_counts_kernel<BLOCK_SIZE, true>
        <<<2, finalize_threads, finalize_smem, stream>>>(
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(expert_ids.data_ptr()),
            static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
            static_cast<int32_t*>(cumsum_buffer.data_ptr()), block_counts,
            static_cast<int32_t>(num_experts), static_cast<int32_t>(num_chunks),
            scan_size, static_cast<int32_t>(sorted_token_ids.size(0)),
            static_cast<int32_t>(numel));
  } else {
    finalize_moe_align_chunk_counts_kernel<BLOCK_SIZE, false>
        <<<2, finalize_threads, finalize_smem, stream>>>(
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(expert_ids.data_ptr()),
            static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
            static_cast<int32_t*>(cumsum_buffer.data_ptr()), block_counts,
            static_cast<int32_t>(num_experts), static_cast<int32_t>(num_chunks),
            scan_size, static_cast<int32_t>(sorted_token_ids.size(0)),
            static_cast<int32_t>(numel));
  }

  const size_t scatter_smem =
      static_cast<size_t>(2 * num_experts) * sizeof(int32_t);
  scatter_expert_tokens_by_chunk_direct_kernel<scalar_t>
      <<<static_cast<int>(num_chunks), sort_threads, scatter_smem, stream>>>(
          static_cast<const scalar_t*>(topk_ids.data_ptr()),
          static_cast<int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<const int32_t*>(cumsum_buffer.data_ptr()), block_counts,
          static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
          static_cast<int32_t>(sort_chunk_size));
}

template <typename scalar_t>
void launch_moe_align_chunked_direct(ffi::TensorView topk_ids,
                                     int64_t num_experts,
                                     int64_t block_size,
                                     ffi::TensorView sorted_token_ids,
                                     ffi::TensorView expert_ids,
                                     ffi::TensorView num_tokens_post_pad,
                                     ffi::TensorView cumsum_buffer,
                                     ffi::TensorView block_workspace,
                                     int64_t sort_chunk_size,
                                     bool pad_sorted_token_ids) {
  switch (block_size) {
    case 8:
      launch_moe_align_chunked_direct_block<scalar_t, 8>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 16:
      launch_moe_align_chunked_direct_block<scalar_t, 16>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 32:
      launch_moe_align_chunked_direct_block<scalar_t, 32>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 48:
      launch_moe_align_chunked_direct_block<scalar_t, 48>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 64:
      launch_moe_align_chunked_direct_block<scalar_t, 64>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 128:
      launch_moe_align_chunked_direct_block<scalar_t, 128>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 256:
      launch_moe_align_chunked_direct_block<scalar_t, 256>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 512:
      launch_moe_align_chunked_direct_block<scalar_t, 512>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    default:
      TVM_FFI_THROW(ValueError)
          << "MUSA moe_align_block_size chunked-direct optimized JIT supports "
             "block_size=8/16/32/48/64/128/256/512, got "
          << block_size;
  }
}

template <typename scalar_t, int BLOCK_SIZE>
void launch_moe_align_large_sort_block(ffi::TensorView topk_ids,
                                       int64_t num_experts,
                                       ffi::TensorView sorted_token_ids,
                                       ffi::TensorView expert_ids,
                                       ffi::TensorView num_tokens_post_pad,
                                       ffi::TensorView cumsum_buffer,
                                       ffi::TensorView block_workspace,
                                       int64_t sort_chunk_size,
                                       bool pad_sorted_token_ids) {
  const int64_t numel = tensor_numel(topk_ids);
  const int64_t num_chunks = ceil_div_host(numel, sort_chunk_size);
  const int64_t required_workspace = 2 * (num_chunks + 1) * num_experts;
  TVM_FFI_ICHECK_GE(tensor_numel(block_workspace), required_workspace);

  musaStream_t stream = get_stream(topk_ids.device());
  int32_t* workspace = static_cast<int32_t*>(block_workspace.data_ptr());
  int32_t* block_counts = workspace;
  int32_t* block_offsets =
      workspace + static_cast<size_t>(num_chunks + 1) * num_experts;
  const int32_t sort_threads = 256;
  const size_t sort_smem = static_cast<size_t>(num_experts) * sizeof(int32_t);

  count_expert_tokens_by_chunk_kernel<scalar_t>
      <<<static_cast<int>(num_chunks), sort_threads, sort_smem, stream>>>(
          static_cast<const scalar_t*>(topk_ids.data_ptr()), block_counts,
          static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
          static_cast<int32_t>(sort_chunk_size));

  prefix_expert_tokens_by_chunk_kernel<<<static_cast<int>(num_experts), 1, 0,
                                         stream>>>(
      block_counts, block_offsets, static_cast<int32_t>(num_experts),
      static_cast<int32_t>(num_chunks));

  const int32_t finalize_scan_size =
      static_cast<int32_t>(next_pow2_host(num_experts));
  finalize_moe_align_large_sort_kernel<BLOCK_SIZE>
      <<<1, 1024, finalize_scan_size * sizeof(int32_t), stream>>>(
      static_cast<int32_t*>(sorted_token_ids.data_ptr()),
      static_cast<int32_t*>(expert_ids.data_ptr()),
      static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
      static_cast<int32_t*>(cumsum_buffer.data_ptr()), block_offsets,
      static_cast<int32_t>(num_experts), static_cast<int32_t>(num_chunks),
      static_cast<int32_t>(numel), finalize_scan_size, pad_sorted_token_ids,
      static_cast<int32_t>(sorted_token_ids.size(0)));
  if (pad_sorted_token_ids) {
    const int fill_threads = 256;
    const int fill_blocks = std::min<int>(
        1024,
        static_cast<int>(ceil_div_host(
            ceil_div_host(sorted_token_ids.size(0), kVecSize), fill_threads)));
    fill_sorted_token_ids_kernel<<<fill_blocks, fill_threads, 0, stream>>>(
        static_cast<int32_t*>(sorted_token_ids.data_ptr()),
        static_cast<int32_t>(numel),
        static_cast<int32_t>(sorted_token_ids.size(0)));
  }

  scatter_expert_tokens_by_chunk_kernel<scalar_t>
      <<<static_cast<int>(num_chunks), sort_threads, sort_smem, stream>>>(
          static_cast<const scalar_t*>(topk_ids.data_ptr()),
          static_cast<int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<const int32_t*>(cumsum_buffer.data_ptr()), block_offsets,
          static_cast<int32_t>(num_experts), static_cast<size_t>(numel),
          static_cast<int32_t>(sort_chunk_size));
}

template <typename scalar_t>
void launch_moe_align_large_sort(ffi::TensorView topk_ids,
                                 int64_t num_experts,
                                 int64_t block_size,
                                 ffi::TensorView sorted_token_ids,
                                 ffi::TensorView expert_ids,
                                 ffi::TensorView num_tokens_post_pad,
                                 ffi::TensorView cumsum_buffer,
                                 ffi::TensorView block_workspace,
                                 int64_t sort_chunk_size,
                                 bool pad_sorted_token_ids) {
  switch (block_size) {
    case 8:
      launch_moe_align_large_sort_block<scalar_t, 8>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 16:
      launch_moe_align_large_sort_block<scalar_t, 16>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 32:
      launch_moe_align_large_sort_block<scalar_t, 32>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 48:
      launch_moe_align_large_sort_block<scalar_t, 48>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 64:
      launch_moe_align_large_sort_block<scalar_t, 64>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 128:
      launch_moe_align_large_sort_block<scalar_t, 128>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 256:
      launch_moe_align_large_sort_block<scalar_t, 256>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    case 512:
      launch_moe_align_large_sort_block<scalar_t, 512>(
          topk_ids, num_experts, sorted_token_ids, expert_ids,
          num_tokens_post_pad, cumsum_buffer, block_workspace,
          sort_chunk_size, pad_sorted_token_ids);
      break;
    default:
      TVM_FFI_THROW(ValueError)
          << "MUSA moe_align_block_size large-sort optimized JIT supports "
             "block_size=8/16/32/48/64/128/256/512, got "
          << block_size;
  }
}

}  // namespace

void sgl_musa_moe_align_block_size(ffi::TensorView topk_ids,
                                   int64_t num_experts,
                                   int64_t block_size,
                                   ffi::TensorView sorted_token_ids,
                                   ffi::TensorView expert_ids,
                                   ffi::TensorView num_tokens_post_pad,
                                   ffi::TensorView cumsum_buffer,
                                   bool pad_sorted_token_ids) {
  check_moe_align_inputs(topk_ids, sorted_token_ids, expert_ids,
                         num_tokens_post_pad, cumsum_buffer, num_experts);
  ffi::MUSADeviceGuard device_guard(topk_ids.device().device_id);

  if (dtype_equal(topk_ids.dtype(), dl_int32)) {
    launch_moe_align<int32_t>(topk_ids, num_experts, block_size,
                              sorted_token_ids, expert_ids,
                              num_tokens_post_pad, cumsum_buffer,
                              pad_sorted_token_ids);
  } else if (dtype_equal(topk_ids.dtype(), dl_int64)) {
    launch_moe_align<int64_t>(topk_ids, num_experts, block_size,
                              sorted_token_ids, expert_ids,
                              num_tokens_post_pad, cumsum_buffer,
                              pad_sorted_token_ids);
  } else {
    TVM_FFI_THROW(ValueError) << "sgl_musa_moe_align_block_size only supports "
                                 "int32/int64 topk_ids";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA moe_align_block_size kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_moe_align_block_size,
                              sgl_musa_moe_align_block_size);

void sgl_musa_moe_align_block_size_large_sort(
    ffi::TensorView topk_ids, int64_t num_experts, int64_t block_size,
    ffi::TensorView sorted_token_ids, ffi::TensorView expert_ids,
    ffi::TensorView num_tokens_post_pad, ffi::TensorView cumsum_buffer,
    ffi::TensorView block_workspace, int64_t sort_chunk_size,
    bool pad_sorted_token_ids) {
  check_moe_align_inputs(topk_ids, sorted_token_ids, expert_ids,
                         num_tokens_post_pad, cumsum_buffer, num_experts);
  CHECK_MUSA_CONTIGUOUS(block_workspace);
  TVM_FFI_ICHECK(dtype_equal(block_workspace.dtype(), dl_int32));
  TVM_FFI_ICHECK_EQ(block_workspace.device().device_id,
                    topk_ids.device().device_id);
  TVM_FFI_ICHECK_GE(sort_chunk_size, 128);
  ffi::MUSADeviceGuard device_guard(topk_ids.device().device_id);

  if (dtype_equal(topk_ids.dtype(), dl_int32)) {
    launch_moe_align_large_sort<int32_t>(
        topk_ids, num_experts, block_size, sorted_token_ids, expert_ids,
        num_tokens_post_pad, cumsum_buffer, block_workspace,
        sort_chunk_size, pad_sorted_token_ids);
  } else if (dtype_equal(topk_ids.dtype(), dl_int64)) {
    launch_moe_align_large_sort<int64_t>(
        topk_ids, num_experts, block_size, sorted_token_ids, expert_ids,
        num_tokens_post_pad, cumsum_buffer, block_workspace,
        sort_chunk_size, pad_sorted_token_ids);
  } else {
    TVM_FFI_THROW(ValueError) << "sgl_musa_moe_align_block_size_large_sort "
                                 "only supports int32/int64 topk_ids";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA moe_align_block_size large-sort kernel failed: "
      << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_moe_align_block_size_large_sort,
                              sgl_musa_moe_align_block_size_large_sort);

void sgl_musa_moe_align_block_size_chunked_direct(
    ffi::TensorView topk_ids, int64_t num_experts, int64_t block_size,
    ffi::TensorView sorted_token_ids, ffi::TensorView expert_ids,
    ffi::TensorView num_tokens_post_pad, ffi::TensorView cumsum_buffer,
    ffi::TensorView block_workspace, int64_t sort_chunk_size,
    bool pad_sorted_token_ids) {
  check_moe_align_inputs(topk_ids, sorted_token_ids, expert_ids,
                         num_tokens_post_pad, cumsum_buffer, num_experts);
  CHECK_MUSA_CONTIGUOUS(block_workspace);
  TVM_FFI_ICHECK(dtype_equal(block_workspace.dtype(), dl_int32));
  TVM_FFI_ICHECK_EQ(block_workspace.device().device_id,
                    topk_ids.device().device_id);
  TVM_FFI_ICHECK_GE(sort_chunk_size, 128);
  ffi::MUSADeviceGuard device_guard(topk_ids.device().device_id);

  if (dtype_equal(topk_ids.dtype(), dl_int32)) {
    launch_moe_align_chunked_direct<int32_t>(
        topk_ids, num_experts, block_size, sorted_token_ids, expert_ids,
        num_tokens_post_pad, cumsum_buffer, block_workspace, sort_chunk_size,
        pad_sorted_token_ids);
  } else if (dtype_equal(topk_ids.dtype(), dl_int64)) {
    launch_moe_align_chunked_direct<int64_t>(
        topk_ids, num_experts, block_size, sorted_token_ids, expert_ids,
        num_tokens_post_pad, cumsum_buffer, block_workspace, sort_chunk_size,
        pad_sorted_token_ids);
  } else {
    TVM_FFI_THROW(ValueError)
        << "sgl_musa_moe_align_block_size_chunked_direct only supports "
           "int32/int64 topk_ids";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA moe_align_block_size chunked-direct kernel failed: "
      << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_moe_align_block_size_chunked_direct,
                              sgl_musa_moe_align_block_size_chunked_direct);
