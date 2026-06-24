#include "common.mu"

template <typename T, int ACTIVATION_TYPE, int ROWS_PER_BLOCK>
__global__ __launch_bounds__(kVecThreads, 1)
void act_and_mul_masked_vec8_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    const int32_t* __restrict__ masked_m,
    int experts,
    int m,
    int half_hidden_size,
    int chunks) {
  const int expert = static_cast<int>(blockIdx.z);
  const int row_base = static_cast<int>(blockIdx.y) * ROWS_PER_BLOCK;
  if (expert >= experts || row_base >= m || row_base >= masked_m[expert]) {
    return;
  }

  const int chunk = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunk >= chunks) {
    return;
  }

  const int col = chunk * kVecElems;
  const int hidden_size = half_hidden_size * 2;
  const int valid_m = masked_m[expert];
#pragma unroll
  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const int row = row_base + r;
    if (row < m && row < valid_m) {
      const int64_t input_offset =
          (static_cast<int64_t>(expert) * m + row) * hidden_size + col;
      const int64_t output_offset =
          (static_cast<int64_t>(expert) * m + row) * half_hidden_size + col;

      Pack16B<T> gate_pack;
      Pack16B<T> up_pack;
      Pack16B<T> out_pack;
      gate_pack.v = load_vec8(gateup_output + input_offset);
      up_pack.v = load_vec8(gateup_output + input_offset + half_hidden_size);

#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        const float gate = to_float(gate_pack.elems[i]);
        const float up = to_float(up_pack.elems[i]);
        out_pack.elems[i] =
            cast_from_float<T>(apply_activation<ACTIVATION_TYPE>(gate) * up);
      }

      store_vec8(down_input + output_offset, out_pack.v);
    }
  }
}

template <typename T, int ACTIVATION_TYPE, int ROWS_PER_BLOCK>
__global__ __launch_bounds__(kThreads, 1)
void act_and_mul_masked_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    const int32_t* __restrict__ masked_m,
    int experts,
    int m,
    int half_hidden_size) {
  const int expert = static_cast<int>(blockIdx.z);
  const int row_base = static_cast<int>(blockIdx.y) * ROWS_PER_BLOCK;
  if (expert >= experts || row_base >= m || row_base >= masked_m[expert]) {
    return;
  }

  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= half_hidden_size) {
    return;
  }

  const int hidden_size = half_hidden_size * 2;
  const int valid_m = masked_m[expert];
#pragma unroll
  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const int row = row_base + r;
    if (row < m && row < valid_m) {
      const int64_t input_offset =
          (static_cast<int64_t>(expert) * m + row) * hidden_size + col;
      const int64_t output_offset =
          (static_cast<int64_t>(expert) * m + row) * half_hidden_size + col;

      const float gate = to_float(gateup_output[input_offset]);
      const float up = to_float(gateup_output[input_offset + half_hidden_size]);
      down_input[output_offset] =
          cast_from_float<T>(apply_activation<ACTIVATION_TYPE>(gate) * up);
    }
  }
}
void check_act_and_mul_masked_inputs(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView masked_m,
    int64_t activation_type) {
  CHECK_INPUT(gateup_output);
  CHECK_INPUT(down_input);
  CHECK_INPUT(masked_m);
  TVM_FFI_ICHECK_EQ(gateup_output.ndim(), 3);
  TVM_FFI_ICHECK_EQ(down_input.ndim(), 3);
  TVM_FFI_ICHECK_EQ(masked_m.ndim(), 1);
  TVM_FFI_ICHECK_EQ(gateup_output.device().device_id, down_input.device().device_id);
  TVM_FFI_ICHECK_EQ(gateup_output.device().device_id, masked_m.device().device_id);
  TVM_FFI_ICHECK_EQ(gateup_output.size(0), down_input.size(0));
  TVM_FFI_ICHECK_EQ(gateup_output.size(1), down_input.size(1));
  TVM_FFI_ICHECK_EQ(gateup_output.size(2), down_input.size(2) * 2);
  TVM_FFI_ICHECK_EQ(gateup_output.size(0), masked_m.size(0));
  TVM_FFI_ICHECK(dtype_equal(gateup_output.dtype(), down_input.dtype()));
  TVM_FFI_ICHECK(
      dtype_equal(gateup_output.dtype(), dl_float16) ||
      dtype_equal(gateup_output.dtype(), dl_bfloat16) ||
      dtype_equal(gateup_output.dtype(), dl_float32));
  TVM_FFI_ICHECK(dtype_equal(masked_m.dtype(), dl_int32));
  TVM_FFI_ICHECK(
      activation_type == kSiluActivation || activation_type == kGeluActivation ||
      activation_type == kGeluTanhActivation);
}
template <typename T, int ACTIVATION_TYPE, int ROWS_PER_BLOCK>
void launch_act_and_mul_masked(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView masked_m,
    musaStream_t stream) {
  const int experts = static_cast<int>(down_input.size(0));
  const int m = static_cast<int>(down_input.size(1));
  const int half_hidden_size = static_cast<int>(down_input.size(2));
  constexpr bool can_vec8 = !std::is_same_v<T, float>;
  const bool use_vec8 = can_vec8 && (half_hidden_size % kVecElems) == 0;

  if (use_vec8) {
    const int chunks = half_hidden_size / kVecElems;
    const int vec_threads = chunks <= 128 ? 128 : kVecThreads;
    const int chunk_tiles = (chunks + vec_threads - 1) / vec_threads;
    const int row_tiles = (m + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(
        static_cast<unsigned>(chunk_tiles),
        static_cast<unsigned>(row_tiles),
        static_cast<unsigned>(experts));
    act_and_mul_masked_vec8_kernel<T, ACTIVATION_TYPE, ROWS_PER_BLOCK>
        <<<grid, dim3(static_cast<unsigned>(vec_threads)), 0, stream>>>(
            static_cast<const T*>(gateup_output.data_ptr()),
            static_cast<T*>(down_input.data_ptr()),
            static_cast<const int32_t*>(masked_m.data_ptr()),
            experts,
            m,
            half_hidden_size,
            chunks);
    return;
  }

  const int tiles = (half_hidden_size + kThreads - 1) / kThreads;
  const int row_tiles = (m + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  dim3 grid(
      static_cast<unsigned>(tiles),
      static_cast<unsigned>(row_tiles),
      static_cast<unsigned>(experts));
  act_and_mul_masked_kernel<T, ACTIVATION_TYPE, ROWS_PER_BLOCK>
      <<<grid, dim3(kThreads), 0, stream>>>(
          static_cast<const T*>(gateup_output.data_ptr()),
          static_cast<T*>(down_input.data_ptr()),
          static_cast<const int32_t*>(masked_m.data_ptr()),
          experts,
          m,
          half_hidden_size);
}

template <typename T, int ACTIVATION_TYPE>
void dispatch_act_and_mul_masked_rows(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView masked_m,
    musaStream_t stream) {
  launch_act_and_mul_masked<T, ACTIVATION_TYPE, 1>(
      gateup_output, down_input, masked_m, stream);
}

template <typename T>
void dispatch_act_and_mul_masked_activation(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView masked_m,
    int activation_type,
    musaStream_t stream) {
  if (activation_type == kGeluActivation) {
    dispatch_act_and_mul_masked_rows<T, kGeluActivation>(
        gateup_output, down_input, masked_m, stream);
  } else if (activation_type == kGeluTanhActivation) {
    dispatch_act_and_mul_masked_rows<T, kGeluTanhActivation>(
        gateup_output, down_input, masked_m, stream);
  } else {
    dispatch_act_and_mul_masked_rows<T, kSiluActivation>(
        gateup_output, down_input, masked_m, stream);
  }
}
void sgl_musa_act_and_mul_masked(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView masked_m,
    int64_t activation_type) {
  check_act_and_mul_masked_inputs(gateup_output, down_input, masked_m, activation_type);
  ffi::MUSADeviceGuard device_guard(gateup_output.device().device_id);
  musaStream_t stream = get_stream(gateup_output.device());
  const int act = static_cast<int>(activation_type);

  if (dtype_equal(gateup_output.dtype(), dl_float16)) {
    dispatch_act_and_mul_masked_activation<half>(
        gateup_output, down_input, masked_m, act, stream);
  } else if (dtype_equal(gateup_output.dtype(), dl_bfloat16)) {
    dispatch_act_and_mul_masked_activation<__mt_bfloat16>(
        gateup_output, down_input, masked_m, act, stream);
  } else if (dtype_equal(gateup_output.dtype(), dl_float32)) {
    dispatch_act_and_mul_masked_activation<float>(
        gateup_output, down_input, masked_m, act, stream);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported act_and_mul_masked dtype";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA act_and_mul_masked kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_act_and_mul_masked, sgl_musa_act_and_mul_masked);
