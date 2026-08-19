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
#include "common.muh"

constexpr int kDiffusionHiddenSize = 3072;
constexpr int kVecSize = 8;
constexpr int kThreads = kDiffusionHiddenSize / kVecSize;

void check_mul_add_inputs(ffi::TensorView a, ffi::TensorView b,
                          ffi::TensorView c, ffi::TensorView output) {
  CHECK_MUSA_CONTIGUOUS(a);
  CHECK_MUSA_CONTIGUOUS(b);
  CHECK_MUSA_CONTIGUOUS(c);
  CHECK_MUSA_CONTIGUOUS(output);
  CHECK_CONTIGUOUS_2D(a);
  CHECK_CONTIGUOUS_2D(b);
  CHECK_CONTIGUOUS_2D(c);
  CHECK_CONTIGUOUS_2D(output);
  TVM_FFI_ICHECK_EQ(a.size(1), kDiffusionHiddenSize);
  TVM_FFI_ICHECK_EQ(a.size(0), c.size(0));
  TVM_FFI_ICHECK_EQ(a.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(a.size(1), b.size(1));
  TVM_FFI_ICHECK_GT(b.size(0), 0);
  TVM_FFI_ICHECK_EQ(a.size(0) % b.size(0), 0);
  TVM_FFI_ICHECK_EQ(a.device().device_id, b.device().device_id);
  TVM_FFI_ICHECK_EQ(a.device().device_id, c.device().device_id);
  TVM_FFI_ICHECK_EQ(a.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(a.dtype(), b.dtype()));
  TVM_FFI_ICHECK(dtype_equal(a.dtype(), c.dtype()));
  TVM_FFI_ICHECK(dtype_equal(a.dtype(), output.dtype()));
}

template <typename T>
__global__ __launch_bounds__(kThreads, 1) void mul_add_kernel(
    const T* __restrict__ a, const T* __restrict__ b,
    const T* __restrict__ c, T* __restrict__ output, int rows_per_batch) {
  const int row = static_cast<int>(blockIdx.x);
  const int col = static_cast<int>(threadIdx.x) * kVecSize;

  const int batch = row / rows_per_batch;
  const int64_t row_offset = static_cast<int64_t>(row) * kDiffusionHiddenSize;
  const int64_t batch_offset =
      static_cast<int64_t>(batch) * kDiffusionHiddenSize;
  const DiffusionVec8<T> a_vec =
      DiffusionVec8<T>::load(a + row_offset + col);
  const DiffusionVec8<T> b_vec =
      DiffusionVec8<T>::load(b + batch_offset + col);
  const DiffusionVec8<T> c_vec =
      DiffusionVec8<T>::load(c + row_offset + col);
  DiffusionVec8<T> output_vec;

#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const T product =
        from_float<T>(to_float(a_vec.elem[i]) * to_float(b_vec.elem[i]));
    output_vec.elem[i] =
        from_float<T>(to_float(c_vec.elem[i]) + to_float(product));
  }
  output_vec.store(output + row_offset + col);
}

template <typename T>
void launch_mul_add(ffi::TensorView a, ffi::TensorView b, ffi::TensorView c,
                    ffi::TensorView output) {
  const int rows = static_cast<int>(a.size(0));
  if (rows == 0) {
    return;
  }
  const int rows_per_batch = rows / static_cast<int>(b.size(0));
  musaStream_t stream = get_stream(a.device());
  mul_add_kernel<T><<<rows, kThreads, 0, stream>>>(
      static_cast<const T*>(a.data_ptr()),
      static_cast<const T*>(b.data_ptr()),
      static_cast<const T*>(c.data_ptr()), static_cast<T*>(output.data_ptr()),
      rows_per_batch);
}

void sgl_musa_diffusion_mul_add(ffi::TensorView a, ffi::TensorView b,
                                ffi::TensorView c, ffi::TensorView output) {
  check_mul_add_inputs(a, b, c, output);
  ffi::MUSADeviceGuard device_guard(a.device().device_id);
  if (dtype_equal(a.dtype(), dl_float16)) {
    launch_mul_add<half>(a, b, c, output);
  } else if (dtype_equal(a.dtype(), dl_bfloat16)) {
    launch_mul_add<__mt_bfloat16>(a, b, c, output);
  } else {
    TVM_FFI_THROW(ValueError) << "MUSA diffusion MulAdd supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA diffusion MulAdd failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_diffusion_mul_add,
                              sgl_musa_diffusion_mul_add);
