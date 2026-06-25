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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_store_cache, sgl_musa_store_cache);
