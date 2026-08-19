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

void check_layernorm_inputs(ffi::TensorView x, ffi::TensorView scale,
                            ffi::TensorView shift, ffi::TensorView output) {
  CHECK_MUSA_CONTIGUOUS(x);
  CHECK_MUSA_CONTIGUOUS(scale);
  CHECK_MUSA_CONTIGUOUS(shift);
  CHECK_MUSA_CONTIGUOUS(output);
  CHECK_CONTIGUOUS_2D(x);
  CHECK_CONTIGUOUS_2D(scale);
  CHECK_CONTIGUOUS_2D(shift);
  CHECK_CONTIGUOUS_2D(output);
  TVM_FFI_ICHECK_EQ(x.size(1), kDiffusionHiddenSize);
  TVM_FFI_ICHECK_EQ(x.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(scale.size(0), shift.size(0));
  TVM_FFI_ICHECK_EQ(scale.size(1), x.size(1));
  TVM_FFI_ICHECK_GT(scale.size(0), 0);
  TVM_FFI_ICHECK_EQ(x.size(0) % scale.size(0), 0);
  TVM_FFI_ICHECK_EQ(x.device().device_id, scale.device().device_id);
  TVM_FFI_ICHECK_EQ(x.device().device_id, shift.device().device_id);
  TVM_FFI_ICHECK_EQ(x.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(x.dtype(), scale.dtype()));
  TVM_FFI_ICHECK(dtype_equal(x.dtype(), shift.dtype()));
  TVM_FFI_ICHECK(dtype_equal(x.dtype(), output.dtype()));
}

template <typename T, bool WithResidual, bool Select01 = false>
__global__ __launch_bounds__(kThreads, 1) void layernorm_scale_shift_kernel(
    const T* __restrict__ x, const T* __restrict__ residual,
    const T* __restrict__ gate, const T* __restrict__ scale,
    const T* __restrict__ shift, const T* __restrict__ scale1,
    const T* __restrict__ shift1, const T* __restrict__ gate0,
    const T* __restrict__ gate1, const int32_t* __restrict__ index,
    T* __restrict__ output, T* __restrict__ residual_output,
    T* __restrict__ gate_output, int rows_per_batch, float eps) {
  const int row = static_cast<int>(blockIdx.x);
  const int col = static_cast<int>(threadIdx.x) * kVecSize;

  const int batch = row / rows_per_batch;
  const int64_t row_offset = static_cast<int64_t>(row) * kDiffusionHiddenSize;
  const int64_t batch_offset =
      static_cast<int64_t>(batch) * kDiffusionHiddenSize;
  const DiffusionVec8<T> x_vec =
      DiffusionVec8<T>::load(x + row_offset + col);
  DiffusionVec8<T> normalized_input;
  float values[kVecSize];

  if constexpr (WithResidual) {
    const DiffusionVec8<T> residual_vec =
        DiffusionVec8<T>::load(residual + row_offset + col);
    const DiffusionVec8<T> gate_vec =
        DiffusionVec8<T>::load(gate +
                               (Select01 ? row_offset : batch_offset) + col);
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const T product =
          from_float<T>(to_float(gate_vec.elem[i]) * to_float(x_vec.elem[i]));
      normalized_input.elem[i] = from_float<T>(
          to_float(residual_vec.elem[i]) + to_float(product));
      values[i] = to_float(normalized_input.elem[i]);
    }
    normalized_input.store(residual_output + row_offset + col);
  } else {
    normalized_input = x_vec;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      values[i] = to_float(x_vec.elem[i]);
    }
  }

  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    local_sum += values[i];
  }
  __shared__ float warp_sums[(kThreads + 31) / 32];
  const float sum = diffusion_block_sum<kThreads>(local_sum, warp_sums);
  const float mean = sum / static_cast<float>(kDiffusionHiddenSize);

  // Compute variance from centered values instead of E[x^2] - E[x]^2.
  // The latter suffers catastrophic cancellation on the residual path when
  // the residual has a large mean, which can produce a near-zero variance,
  // amplify the normalized output, and eventually generate NaN/Inf values.
  float local_centered_square_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const float centered = values[i] - mean;
    local_centered_square_sum += centered * centered;
  }
  const float centered_square_sum =
      diffusion_block_sum<kThreads>(local_centered_square_sum, warp_sums);
  const float variance = centered_square_sum /
                         static_cast<float>(kDiffusionHiddenSize);
  const float inv_std = rsqrtf(variance + eps);

  const bool select1 = Select01 && index[row] != 0;
  const T* selected_scale = select1 ? scale1 : scale;
  const T* selected_shift = select1 ? shift1 : shift;
  const DiffusionVec8<T> scale_vec =
      DiffusionVec8<T>::load(selected_scale + batch_offset + col);
  const DiffusionVec8<T> shift_vec =
      DiffusionVec8<T>::load(selected_shift + batch_offset + col);
  DiffusionVec8<T> output_vec;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const float value = (values[i] - mean) * inv_std *
                            (1.0f + to_float(scale_vec.elem[i])) +
                        to_float(shift_vec.elem[i]);
    output_vec.elem[i] = from_float<T>(value);
  }
  output_vec.store(output + row_offset + col);
  if constexpr (Select01) {
    const T* selected_gate = select1 ? gate1 : gate0;
    DiffusionVec8<T>::load(selected_gate + batch_offset + col)
        .store(gate_output + row_offset + col);
  }
}

template <typename T, bool WithResidual, bool Select01 = false>
void launch_layernorm_scale_shift(
    ffi::TensorView x, ffi::TensorView residual, ffi::TensorView gate,
    ffi::TensorView scale, ffi::TensorView shift, ffi::TensorView output,
    ffi::TensorView residual_output, float eps, ffi::TensorView scale1,
    ffi::TensorView shift1, ffi::TensorView gate0, ffi::TensorView gate1,
    ffi::TensorView index, ffi::TensorView gate_output) {
  const int rows = static_cast<int>(x.size(0));
  if (rows == 0) {
    return;
  }
  const int rows_per_batch = rows / static_cast<int>(scale.size(0));
  musaStream_t stream = get_stream(x.device());
  layernorm_scale_shift_kernel<T, WithResidual, Select01>
      <<<rows, kThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()),
      WithResidual ? static_cast<const T*>(residual.data_ptr()) : nullptr,
      WithResidual ? static_cast<const T*>(gate.data_ptr()) : nullptr,
      static_cast<const T*>(scale.data_ptr()),
      static_cast<const T*>(shift.data_ptr()),
      Select01 ? static_cast<const T*>(scale1.data_ptr()) : nullptr,
      Select01 ? static_cast<const T*>(shift1.data_ptr()) : nullptr,
      Select01 ? static_cast<const T*>(gate0.data_ptr()) : nullptr,
      Select01 ? static_cast<const T*>(gate1.data_ptr()) : nullptr,
      Select01 ? static_cast<const int32_t*>(index.data_ptr()) : nullptr,
      static_cast<T*>(output.data_ptr()),
      WithResidual ? static_cast<T*>(residual_output.data_ptr()) : nullptr,
      Select01 ? static_cast<T*>(gate_output.data_ptr()) : nullptr,
      rows_per_batch, eps);
}

void sgl_musa_diffusion_layernorm_scale_shift(
    ffi::TensorView x, ffi::TensorView scale, ffi::TensorView shift,
    ffi::TensorView output, double eps) {
  check_layernorm_inputs(x, scale, shift, output);
  ffi::MUSADeviceGuard device_guard(x.device().device_id);
  if (dtype_equal(x.dtype(), dl_float16)) {
    launch_layernorm_scale_shift<half, false>(x, x, scale, scale, shift,
                                               output, output, eps, scale, shift,
                                               scale, scale, scale, output);
  } else if (dtype_equal(x.dtype(), dl_bfloat16)) {
    launch_layernorm_scale_shift<__mt_bfloat16, false>(
        x, x, scale, scale, shift, output, output, eps, scale, shift, scale,
        scale, scale, output);
  } else {
    TVM_FFI_THROW(ValueError)
        << "MUSA diffusion LayerNorm+ScaleShift supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA diffusion LayerNorm+ScaleShift failed: "
      << musaGetErrorString(err);
}

void sgl_musa_diffusion_scale_residual_layernorm_scale_shift(
    ffi::TensorView x, ffi::TensorView residual, ffi::TensorView gate,
    ffi::TensorView scale, ffi::TensorView shift, ffi::TensorView output,
    ffi::TensorView residual_output, double eps) {
  check_layernorm_inputs(x, scale, shift, output);
  CHECK_MUSA_CONTIGUOUS(residual);
  CHECK_MUSA_CONTIGUOUS(gate);
  CHECK_MUSA_CONTIGUOUS(residual_output);
  CHECK_CONTIGUOUS_2D(residual);
  CHECK_CONTIGUOUS_2D(gate);
  CHECK_CONTIGUOUS_2D(residual_output);
  TVM_FFI_ICHECK_EQ(residual.size(0), x.size(0));
  TVM_FFI_ICHECK_EQ(residual.size(1), x.size(1));
  TVM_FFI_ICHECK_EQ(gate.size(0), scale.size(0));
  TVM_FFI_ICHECK_EQ(gate.size(1), x.size(1));
  TVM_FFI_ICHECK_EQ(residual_output.size(0), x.size(0));
  TVM_FFI_ICHECK_EQ(residual_output.size(1), x.size(1));
  TVM_FFI_ICHECK_EQ(residual.device().device_id, x.device().device_id);
  TVM_FFI_ICHECK_EQ(gate.device().device_id, x.device().device_id);
  TVM_FFI_ICHECK_EQ(residual_output.device().device_id, x.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(x.dtype(), residual.dtype()));
  TVM_FFI_ICHECK(dtype_equal(x.dtype(), gate.dtype()));
  TVM_FFI_ICHECK(dtype_equal(x.dtype(), residual_output.dtype()));
  ffi::MUSADeviceGuard device_guard(x.device().device_id);
  if (dtype_equal(x.dtype(), dl_float16)) {
    launch_layernorm_scale_shift<half, true>(
        x, residual, gate, scale, shift, output, residual_output, eps, scale,
        shift, gate, gate, gate, output);
  } else if (dtype_equal(x.dtype(), dl_bfloat16)) {
    launch_layernorm_scale_shift<__mt_bfloat16, true>(
        x, residual, gate, scale, shift, output, residual_output, eps, scale,
        shift, gate, gate, gate, output);
  } else {
    TVM_FFI_THROW(ValueError)
        << "MUSA diffusion residual LayerNorm supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA diffusion residual LayerNorm failed: "
      << musaGetErrorString(err);
}

void check_select01_inputs(
    ffi::TensorView x, ffi::TensorView scale0, ffi::TensorView shift0,
    ffi::TensorView gate0, ffi::TensorView scale1, ffi::TensorView shift1,
    ffi::TensorView gate1, ffi::TensorView index, ffi::TensorView output,
    ffi::TensorView gate_output) {
  check_layernorm_inputs(x, scale0, shift0, output);
  const int64_t batches = scale0.size(0);
  for (const auto& tensor : {gate0, scale1, shift1, gate1}) {
    CHECK_MUSA_CONTIGUOUS(tensor);
    CHECK_CONTIGUOUS_2D(tensor);
    TVM_FFI_ICHECK_EQ(tensor.size(0), batches);
    TVM_FFI_ICHECK_EQ(tensor.size(1), kDiffusionHiddenSize);
    TVM_FFI_ICHECK_EQ(tensor.device().device_id, x.device().device_id);
    TVM_FFI_ICHECK(dtype_equal(tensor.dtype(), x.dtype()));
  }
  CHECK_MUSA_CONTIGUOUS(index);
  TVM_FFI_ICHECK_EQ(index.ndim(), 1);
  TVM_FFI_ICHECK_EQ(index.size(0), x.size(0));
  TVM_FFI_ICHECK_EQ(index.device().device_id, x.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(index.dtype(), dl_int32));
  CHECK_MUSA_CONTIGUOUS(gate_output);
  CHECK_CONTIGUOUS_2D(gate_output);
  TVM_FFI_ICHECK_EQ(gate_output.size(0), x.size(0));
  TVM_FFI_ICHECK_EQ(gate_output.size(1), kDiffusionHiddenSize);
  TVM_FFI_ICHECK_EQ(gate_output.device().device_id, x.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(gate_output.dtype(), x.dtype()));
}

void sgl_musa_diffusion_layernorm_scale_shift_gate_select01(
    ffi::TensorView x, ffi::TensorView scale0, ffi::TensorView shift0,
    ffi::TensorView gate0, ffi::TensorView scale1, ffi::TensorView shift1,
    ffi::TensorView gate1, ffi::TensorView index, ffi::TensorView output,
    ffi::TensorView gate_output, double eps) {
  check_select01_inputs(x, scale0, shift0, gate0, scale1, shift1, gate1,
                        index, output, gate_output);
  ffi::MUSADeviceGuard device_guard(x.device().device_id);
  if (dtype_equal(x.dtype(), dl_float16)) {
    launch_layernorm_scale_shift<half, false, true>(
        x, x, x, scale0, shift0, output, output, eps, scale1, shift1, gate0,
        gate1, index, gate_output);
  } else if (dtype_equal(x.dtype(), dl_bfloat16)) {
    launch_layernorm_scale_shift<__mt_bfloat16, false, true>(
        x, x, x, scale0, shift0, output, output, eps, scale1, shift1, gate0,
        gate1, index, gate_output);
  } else {
    TVM_FFI_THROW(ValueError)
        << "MUSA diffusion Select01 LayerNorm supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA diffusion Select01 LayerNorm failed: "
      << musaGetErrorString(err);
}

void sgl_musa_diffusion_residual_layernorm_scale_shift_gate_select01(
    ffi::TensorView x, ffi::TensorView residual,
    ffi::TensorView residual_gate, ffi::TensorView scale0,
    ffi::TensorView shift0, ffi::TensorView gate0, ffi::TensorView scale1,
    ffi::TensorView shift1, ffi::TensorView gate1, ffi::TensorView index,
    ffi::TensorView output, ffi::TensorView residual_output,
    ffi::TensorView gate_output, double eps) {
  check_select01_inputs(x, scale0, shift0, gate0, scale1, shift1, gate1,
                        index, output, gate_output);
  for (const auto& tensor : {residual, residual_gate, residual_output}) {
    CHECK_MUSA_CONTIGUOUS(tensor);
    CHECK_CONTIGUOUS_2D(tensor);
    TVM_FFI_ICHECK_EQ(tensor.size(0), x.size(0));
    TVM_FFI_ICHECK_EQ(tensor.size(1), kDiffusionHiddenSize);
    TVM_FFI_ICHECK_EQ(tensor.device().device_id, x.device().device_id);
    TVM_FFI_ICHECK(dtype_equal(tensor.dtype(), x.dtype()));
  }
  ffi::MUSADeviceGuard device_guard(x.device().device_id);
  if (dtype_equal(x.dtype(), dl_float16)) {
    launch_layernorm_scale_shift<half, true, true>(
        x, residual, residual_gate, scale0, shift0, output, residual_output,
        eps, scale1, shift1, gate0, gate1, index, gate_output);
  } else if (dtype_equal(x.dtype(), dl_bfloat16)) {
    launch_layernorm_scale_shift<__mt_bfloat16, true, true>(
        x, residual, residual_gate, scale0, shift0, output, residual_output,
        eps, scale1, shift1, gate0, gate1, index, gate_output);
  } else {
    TVM_FFI_THROW(ValueError)
        << "MUSA diffusion residual Select01 LayerNorm supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA diffusion residual Select01 LayerNorm failed: "
      << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_diffusion_layernorm_scale_shift,
    sgl_musa_diffusion_layernorm_scale_shift);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_diffusion_scale_residual_layernorm_scale_shift,
    sgl_musa_diffusion_scale_residual_layernorm_scale_shift);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_diffusion_layernorm_scale_shift_gate_select01,
    sgl_musa_diffusion_layernorm_scale_shift_gate_select01);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_diffusion_residual_layernorm_scale_shift_gate_select01,
    sgl_musa_diffusion_residual_layernorm_scale_shift_gate_select01);
