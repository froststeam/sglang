#include <ATen/musa/MUSAContext.h>
#include <c10/musa/MUSAGuard.h>
#include <torch/all.h>

#include <cfloat>
#include <cstdint>

#include "sgl_kernel_musa_ops.h"

namespace {

constexpr int kBlock = 1024;
constexpr int kMaxK = 256;

struct TopKRenormStorage {
  int histogram[256];
  float candidate_probs[kMaxK];
  int candidate_ids[kMaxK];
  uint32_t radix_prefix;
  uint32_t radix_mask;
  int radix_rank;
  int radix_confirmed;
  int radix_bucket_count;
  int candidate_count;
  int overflow;
  float selected_mass;
};

__device__ __forceinline__ uint32_t ordered_float_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t mask = (0u - (bits >> 31)) | 0x80000000u;
  return bits ^ mask;
}

__global__ void top_k_renorm_probs_kernel(
    const float* __restrict__ probs,
    float* __restrict__ output,
    const int* __restrict__ top_ks,
    int batch,
    int vocab) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= batch) return;

  const int top_k = top_ks[row];
  const float* row_probs = probs + static_cast<size_t>(row) * vocab;
  float* row_output = output + static_cast<size_t>(row) * vocab;
  if (top_k <= 0 || top_k > kMaxK) {
    for (int idx = tid; idx < vocab; idx += kBlock) row_output[idx] = 0.0f;
    return;
  }

  extern __shared__ unsigned char raw[];
  auto& smem = *reinterpret_cast<TopKRenormStorage*>(raw);
  if (tid == 0) {
    smem.radix_prefix = 0;
    smem.radix_mask = 0;
    smem.radix_rank = top_k;
    smem.radix_confirmed = 0;
    smem.radix_bucket_count = vocab;
  }
  __syncthreads();

  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) smem.histogram[tid] = 0;
    __syncthreads();

    const uint32_t prefix = smem.radix_prefix;
    const uint32_t mask = smem.radix_mask;
    for (int idx = tid; idx < vocab; idx += kBlock) {
      const uint32_t key = ordered_float_key(row_probs[idx]);
      if ((key & mask) == prefix) {
        atomicAdd(&smem.histogram[(key >> shift) & 0xffu], 1);
      }
    }
    __syncthreads();

    if (tid == 0) {
      int greater = 0;
      int chosen = 0;
      for (int bin = 255; bin >= 0; --bin) {
        const int count = smem.histogram[bin];
        if (greater + count >= smem.radix_rank) {
          chosen = bin;
          smem.radix_rank -= greater;
          smem.radix_confirmed += greater;
          smem.radix_bucket_count = count;
          break;
        }
        greater += count;
      }
      smem.radix_prefix |= static_cast<uint32_t>(chosen) << shift;
      smem.radix_mask |= 0xffu << shift;
    }
    __syncthreads();

    if (smem.radix_confirmed + smem.radix_bucket_count <= kMaxK) break;
  }

  if (tid == 0) {
    smem.candidate_count = 0;
    smem.overflow = 0;
  }
  __syncthreads();

  const uint32_t kth_key = smem.radix_prefix;
  const bool exact_tie_overflow =
      smem.radix_mask == 0xffffffffu &&
      smem.radix_confirmed + smem.radix_bucket_count > kMaxK;
  if (exact_tie_overflow) {
    for (int idx = tid; idx < vocab; idx += kBlock) {
      const float value = row_probs[idx];
      if (ordered_float_key(value) > kth_key) {
        const int pos = atomicAdd(&smem.candidate_count, 1);
        if (pos < kMaxK) {
          smem.candidate_probs[pos] = value;
          smem.candidate_ids[pos] = idx;
        }
      }
    }
    __syncthreads();
    for (int idx = tid; idx < vocab; idx += kBlock) {
      const float value = row_probs[idx];
      if (ordered_float_key(value) == kth_key) {
        const int pos = atomicAdd(&smem.candidate_count, 1);
        if (pos < top_k) {
          smem.candidate_probs[pos] = value;
          smem.candidate_ids[pos] = idx;
        }
      }
    }
  } else {
    for (int idx = tid; idx < vocab; idx += kBlock) {
      const float value = row_probs[idx];
      if (ordered_float_key(value) >= kth_key) {
        const int pos = atomicAdd(&smem.candidate_count, 1);
        if (pos < kMaxK) {
          smem.candidate_probs[pos] = value;
          smem.candidate_ids[pos] = idx;
        } else {
          smem.overflow = 1;
        }
      }
    }
  }
  __syncthreads();
  if (exact_tie_overflow && tid == 0) smem.candidate_count = top_k;
  __syncthreads();

  if (smem.overflow || smem.candidate_count > kMaxK) {
    for (int idx = tid; idx < vocab; idx += kBlock) row_output[idx] = 0.0f;
    return;
  }

  const int candidate_count = smem.candidate_count;
  if (top_k <= 20 && candidate_count <= 64) {
    if (tid == 0) {
      for (int i = 0; i < top_k; ++i) {
        int best = i;
        for (int j = i + 1; j < candidate_count; ++j) {
          if (smem.candidate_probs[j] > smem.candidate_probs[best]) best = j;
        }
        if (best != i) {
          const float value = smem.candidate_probs[i];
          smem.candidate_probs[i] = smem.candidate_probs[best];
          smem.candidate_probs[best] = value;
          const int id = smem.candidate_ids[i];
          smem.candidate_ids[i] = smem.candidate_ids[best];
          smem.candidate_ids[best] = id;
        }
      }
    }
    __syncthreads();
  } else {
    if (tid < kMaxK && tid >= candidate_count) {
      smem.candidate_probs[tid] = -FLT_MAX;
      smem.candidate_ids[tid] = -1;
    }
    __syncthreads();
    for (int width = 2; width <= kMaxK; width <<= 1) {
      for (int stride = width >> 1; stride > 0; stride >>= 1) {
        if (tid < kMaxK) {
          const int other = tid ^ stride;
          if (other > tid) {
            const bool descending = (tid & width) == 0;
            const float lhs = smem.candidate_probs[tid];
            const float rhs = smem.candidate_probs[other];
            if ((descending && lhs < rhs) || (!descending && lhs > rhs)) {
              smem.candidate_probs[tid] = rhs;
              smem.candidate_probs[other] = lhs;
              const int lhs_id = smem.candidate_ids[tid];
              smem.candidate_ids[tid] = smem.candidate_ids[other];
              smem.candidate_ids[other] = lhs_id;
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (tid == 0) {
    float mass = 0.0f;
    for (int i = 0; i < top_k; ++i) mass += smem.candidate_probs[i];
    smem.selected_mass = mass;
  }
  __syncthreads();

  for (int idx = tid; idx < vocab; idx += kBlock) row_output[idx] = 0.0f;
  __syncthreads();
  if (tid < top_k) {
    row_output[smem.candidate_ids[tid]] =
        smem.candidate_probs[tid] / smem.selected_mass;
  }
}

}  // namespace

torch::Tensor musa_top_k_renorm_probs(
    torch::Tensor probs,
    torch::Tensor top_ks) {
  TORCH_CHECK(probs.device().is_privateuseone(), "probs must be a MUSA tensor");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32,
              "probs must be float32");
  TORCH_CHECK(probs.dim() == 2 && probs.is_contiguous(),
              "probs must be contiguous [B,V]");
  TORCH_CHECK(top_ks.scalar_type() == torch::kInt32 && top_ks.is_contiguous(),
              "top_ks must be contiguous int32");
  TORCH_CHECK(top_ks.numel() == probs.size(0),
              "top_ks must have one value per row");

  auto output = torch::empty_like(probs);
  const c10::musa::OptionalMUSAGuard guard(probs.device());
  auto stream = at::musa::getCurrentMUSAStream(probs.device().index());
  top_k_renorm_probs_kernel<<<
      probs.size(0), kBlock, sizeof(TopKRenormStorage), stream.stream()>>>(
      probs.data_ptr<float>(), output.data_ptr<float>(), top_ks.data_ptr<int>(),
      probs.size(0), probs.size(1));
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return output;
}
