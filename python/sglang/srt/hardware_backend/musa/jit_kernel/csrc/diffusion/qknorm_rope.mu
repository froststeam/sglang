#include <algorithm>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

#include "../common.h"
#include "../device_utils.h"

constexpr int kHeadDim = 128;
constexpr int kRopePairs = kHeadDim / 2;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;

template <typename T>
struct __align__(8) QKVec4 {
  T elem[4];
};

template <typename T>
__global__ __launch_bounds__(kThreads) void qknorm_rope_kernel(
    T* __restrict__ q, T* __restrict__ k,
    const T* __restrict__ q_weight, const T* __restrict__ k_weight,
    const float* __restrict__ rope_cache,
    const int64_t* __restrict__ positions, int tokens, int heads, float eps) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int total_heads = heads * 2;
  const int total_work = tokens * total_heads;
  const int first_work = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  const int work_stride = static_cast<int>(gridDim.x) * kWarpsPerBlock;

  for (int work = first_work; work < total_work; work += work_stride) {
    const int token = work / total_heads;
    const int combined_head = work - token * total_heads;
    const bool is_q = combined_head < heads;
    const int head = is_q ? combined_head : combined_head - heads;
    T* data = (is_q ? q : k) +
              (static_cast<int64_t>(token) * heads + head) * kHeadDim;
    const T* weight = is_q ? q_weight : k_weight;
    const int col = lane * 4;
    QKVec4<T> values = *reinterpret_cast<const QKVec4<T>*>(data + col);
    const QKVec4<T> weights =
        *reinterpret_cast<const QKVec4<T>*>(weight + col);

    float normalized[4];
    float square_sum = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const int i = pair * 2;
      normalized[i] = to_float(values.elem[i]);
      normalized[i + 1] = to_float(values.elem[i + 1]);
      square_sum += normalized[i] * normalized[i] +
                    normalized[i + 1] * normalized[i + 1];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      square_sum += __shfl_xor_sync(0xffffffff, square_sum, mask, 32);
    }
    const float inv_rms =
        rsqrtf(square_sum / static_cast<float>(kHeadDim) + eps);
    const int64_t cache_offset =
        positions[token] * static_cast<int64_t>(kHeadDim) + lane * 2;

#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const int i = pair * 2;
      const float x =
          normalized[i] * (inv_rms * to_float(weights.elem[i]));
      const float y =
          normalized[i + 1] * (inv_rms * to_float(weights.elem[i + 1]));
      const float cos = rope_cache[cache_offset + pair];
      const float sin = rope_cache[cache_offset + kRopePairs + pair];
      values.elem[i] = from_float<T>(x * cos - y * sin);
      values.elem[i + 1] = from_float<T>(y * cos + x * sin);
    }
    *reinterpret_cast<QKVec4<T>*>(data + col) = values;
  }
}

void check_qknorm_rope_inputs(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView rope_cache,
    ffi::TensorView positions) {
  CHECK_MUSA_CONTIGUOUS(q);
  CHECK_MUSA_CONTIGUOUS(k);
  CHECK_MUSA_CONTIGUOUS(q_weight);
  CHECK_MUSA_CONTIGUOUS(k_weight);
  CHECK_MUSA_CONTIGUOUS(rope_cache);
  CHECK_MUSA_CONTIGUOUS(positions);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q.size(0), k.size(0));
  TVM_FFI_ICHECK_EQ(q.size(1), k.size(1));
  TVM_FFI_ICHECK_EQ(q.size(2), kHeadDim);
  TVM_FFI_ICHECK_EQ(k.size(2), kHeadDim);
  TVM_FFI_ICHECK_EQ(q_weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(k_weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(q_weight.size(0), kHeadDim);
  TVM_FFI_ICHECK_EQ(k_weight.size(0), kHeadDim);
  TVM_FFI_ICHECK_EQ(rope_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(rope_cache.size(1), kHeadDim);
  TVM_FFI_ICHECK_EQ(positions.ndim(), 1);
  TVM_FFI_ICHECK_EQ(positions.size(0), q.size(0));
  for (const auto& tensor :
       {k, q_weight, k_weight, rope_cache, positions}) {
    TVM_FFI_ICHECK_EQ(tensor.device().device_id, q.device().device_id);
  }
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(rope_cache.dtype(), dl_float32));
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
}

template <typename T>
void launch_qknorm_rope(ffi::TensorView q, ffi::TensorView k,
                        ffi::TensorView q_weight, ffi::TensorView k_weight,
                        ffi::TensorView rope_cache, ffi::TensorView positions,
                        float eps) {
  const int tokens = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  if (tokens == 0 || heads == 0) {
    return;
  }
  const int needed_blocks =
      (tokens * heads * 2 + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int blocks = std::min(needed_blocks, 2048);
  musaStream_t stream = get_stream(q.device());
  qknorm_rope_kernel<T><<<blocks, kThreads, 0, stream>>>(
      static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
      static_cast<const T*>(q_weight.data_ptr()),
      static_cast<const T*>(k_weight.data_ptr()),
      static_cast<const float*>(rope_cache.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()), tokens, heads, eps);
}

void sgl_musa_diffusion_qknorm_rope(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView rope_cache,
    ffi::TensorView positions, double eps) {
  check_qknorm_rope_inputs(q, k, q_weight, k_weight, rope_cache, positions);
  ffi::MUSADeviceGuard device_guard(q.device().device_id);
  if (dtype_equal(q.dtype(), dl_float16)) {
    launch_qknorm_rope<half>(q, k, q_weight, k_weight, rope_cache, positions,
                             eps);
  } else if (dtype_equal(q.dtype(), dl_bfloat16)) {
    launch_qknorm_rope<__mt_bfloat16>(q, k, q_weight, k_weight, rope_cache,
                                      positions, eps);
  } else {
    TVM_FFI_THROW(ValueError)
        << "MUSA diffusion QKNorm+RoPE supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA diffusion QKNorm+RoPE failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_diffusion_qknorm_rope,
                              sgl_musa_diffusion_qknorm_rope);
