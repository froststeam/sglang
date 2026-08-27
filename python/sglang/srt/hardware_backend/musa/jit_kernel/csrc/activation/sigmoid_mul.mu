#include "common.mu"

template <typename T, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 1)
void sigmoid_mul_vec8_kernel(
    T* __restrict__ output,
    const T* __restrict__ gate,
    const T* __restrict__ value,
    int64_t chunks) {
  const int64_t chunk =
      static_cast<int64_t>(blockIdx.x) * BLOCK_THREADS + threadIdx.x;
  if (chunk >= chunks) {
    return;
  }

  const int64_t offset = chunk * kVecElems;
  Pack16B<T> gate_pack;
  Pack16B<T> value_pack;
  Pack16B<T> output_pack;
  gate_pack.v = load_vec8(gate + offset);
  value_pack.v = load_vec8(value + offset);

#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    const float gate_value = to_float(gate_pack.elems[i]);
    const float input_value = to_float(value_pack.elems[i]);
    const float sigmoid_value = 1.0f / (1.0f + expf(-gate_value));
    output_pack.elems[i] =
        cast_from_float<T>(sigmoid_value * input_value);
  }
  store_vec8(output + offset, output_pack.v);
}

template <typename T, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 1)
void sigmoid_mul_scalar_kernel(
    T* __restrict__ output,
    const T* __restrict__ gate,
    const T* __restrict__ value,
    int64_t numel) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * BLOCK_THREADS + threadIdx.x;
  if (index >= numel) {
    return;
  }
  const float gate_value = to_float(gate[index]);
  const float input_value = to_float(value[index]);
  output[index] = cast_from_float<T>(
      input_value / (1.0f + expf(-gate_value)));
}

void check_sigmoid_mul_inputs(
    ffi::TensorView gate,
    ffi::TensorView value,
    ffi::TensorView output) {
  CHECK_INPUT(gate);
  CHECK_INPUT(value);
  CHECK_INPUT(output);
  TVM_FFI_ICHECK_EQ(tensor_numel(gate), tensor_numel(value));
  TVM_FFI_ICHECK_EQ(tensor_numel(gate), tensor_numel(output));
  TVM_FFI_ICHECK(dtype_equal(gate.dtype(), value.dtype()));
  TVM_FFI_ICHECK(dtype_equal(gate.dtype(), output.dtype()));
  TVM_FFI_ICHECK(
      dtype_equal(gate.dtype(), dl_bfloat16) ||
      dtype_equal(gate.dtype(), dl_float16));
  TVM_FFI_ICHECK_EQ(gate.device().device_id, value.device().device_id);
  TVM_FFI_ICHECK_EQ(gate.device().device_id, output.device().device_id);
}

template <typename T>
void launch_sigmoid_mul(
    ffi::TensorView gate,
    ffi::TensorView value,
    ffi::TensorView output) {
  const int64_t numel = tensor_numel(gate);
  if (numel == 0) {
    return;
  }
  constexpr int threads = 256;
  musaStream_t stream = get_stream(gate.device());
  const uintptr_t pointer_bits =
      reinterpret_cast<uintptr_t>(gate.data_ptr()) |
      reinterpret_cast<uintptr_t>(value.data_ptr()) |
      reinterpret_cast<uintptr_t>(output.data_ptr());
  if (numel % kVecElems == 0 && (pointer_bits & 0xF) == 0) {
    const int64_t chunks = numel / kVecElems;
    sigmoid_mul_vec8_kernel<T, threads>
        <<<static_cast<uint32_t>((chunks + threads - 1) / threads),
           threads, 0, stream>>>(
            static_cast<T*>(output.data_ptr()),
            static_cast<const T*>(gate.data_ptr()),
            static_cast<const T*>(value.data_ptr()), chunks);
  } else {
    sigmoid_mul_scalar_kernel<T, threads>
        <<<static_cast<uint32_t>((numel + threads - 1) / threads),
           threads, 0, stream>>>(
            static_cast<T*>(output.data_ptr()),
            static_cast<const T*>(gate.data_ptr()),
            static_cast<const T*>(value.data_ptr()), numel);
  }
}

void sgl_musa_sigmoid_mul(
    ffi::TensorView gate,
    ffi::TensorView value,
    ffi::TensorView output) {
  check_sigmoid_mul_inputs(gate, value, output);
  ffi::MUSADeviceGuard device_guard(gate.device().device_id);
  if (dtype_equal(gate.dtype(), dl_bfloat16)) {
    launch_sigmoid_mul<__mt_bfloat16>(gate, value, output);
  } else {
    launch_sigmoid_mul<half>(gate, value, output);
  }
  const musaError_t error = musaGetLastError();
  TVM_FFI_ICHECK_EQ(error, musaSuccess)
      << "MUSA sigmoid_mul kernel failed: " << musaGetErrorString(error);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_sigmoid_mul, sgl_musa_sigmoid_mul);
