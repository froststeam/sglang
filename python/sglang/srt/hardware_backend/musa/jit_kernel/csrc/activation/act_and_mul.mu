#include "common.mu"

template <typename T, typename IdT, int ACTIVATION_TYPE, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 1)
void act_and_mul_flat_vec8_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    const IdT* __restrict__ expert_ids,
    int rows,
    int half_hidden_size,
    int expert_step,
    int chunks_per_row,
    int total_chunks) {
  const int base_chunk =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kFlatVecChunksPerThread;
  const int hidden_size = half_hidden_size * 2;
#pragma unroll
  for (int c = 0; c < kFlatVecChunksPerThread; ++c) {
    const int chunk = base_chunk + c;
    if (chunk < total_chunks) {
      const int row = chunk / chunks_per_row;
      if (row < rows) {
        const IdT expert_id = expert_ids[row / expert_step];
        if (expert_id != static_cast<IdT>(-1)) {
          const int chunk_in_row = chunk - row * chunks_per_row;
          const int col = chunk_in_row * kVecElems;
          const T* gate_ptr =
              gateup_output + static_cast<int64_t>(row) * hidden_size + col;
          const T* up_ptr = gate_ptr + half_hidden_size;
          T* out_ptr =
              down_input + static_cast<int64_t>(row) * half_hidden_size + col;

          Pack16B<T> gate_pack;
          Pack16B<T> up_pack;
          Pack16B<T> out_pack;
          gate_pack.v = load_vec8(gate_ptr);
          up_pack.v = load_vec8(up_ptr);

#pragma unroll
          for (int i = 0; i < kVecElems; ++i) {
            const float gate = to_float(gate_pack.elems[i]);
            const float up = to_float(up_pack.elems[i]);
            out_pack.elems[i] =
                cast_from_float<T>(apply_activation<ACTIVATION_TYPE>(gate) * up);
          }

          store_vec8(out_ptr, out_pack.v);
        }
      }
    }
  }
}

template <typename T, typename IdT, int ACTIVATION_TYPE, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 1)
void act_and_mul_flat_vec8_no_filter_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    int rows,
    int half_hidden_size,
    int chunks_per_row,
    int total_chunks) {
  const int chunk = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunk >= total_chunks) {
    return;
  }

  const int row = chunk / chunks_per_row;
  if (row >= rows) {
    return;
  }

  const int chunk_in_row = chunk - row * chunks_per_row;
  const int col = chunk_in_row * kVecElems;
  const int hidden_size = half_hidden_size * 2;
  const T* gate_ptr = gateup_output + static_cast<int64_t>(row) * hidden_size + col;
  const T* up_ptr = gate_ptr + half_hidden_size;
  T* out_ptr = down_input + static_cast<int64_t>(row) * half_hidden_size + col;

  Pack16B<T> gate_pack;
  Pack16B<T> up_pack;
  Pack16B<T> out_pack;
  gate_pack.v = load_vec8(gate_ptr);
  up_pack.v = load_vec8(up_ptr);

#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    const float gate = to_float(gate_pack.elems[i]);
    const float up = to_float(up_pack.elems[i]);
    out_pack.elems[i] = cast_from_float<T>(apply_activation<ACTIVATION_TYPE>(gate) * up);
  }

  store_vec8(out_ptr, out_pack.v);
}

template <typename T, typename IdT, int ACTIVATION_TYPE, bool SKIP_EXPERT_CHECK = false>
__global__ __launch_bounds__(kVecThreads, 1)
void act_and_mul_vec8_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    const IdT* __restrict__ expert_ids,
    int64_t rows,
    int64_t half_hidden_size,
    int expert_step) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows) {
    return;
  }

  if constexpr (!SKIP_EXPERT_CHECK) {
    const IdT expert_id = expert_ids[row / expert_step];
    if (expert_id == static_cast<IdT>(-1)) {
      return;
    }
  }

  const int64_t chunks = half_hidden_size / kVecElems;
  const int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunk >= chunks) {
    return;
  }

  const int64_t col = chunk * kVecElems;
  const int64_t hidden_size = half_hidden_size * 2;
  const T* gate_ptr = gateup_output + row * hidden_size + col;
  const T* up_ptr = gate_ptr + half_hidden_size;
  T* out_ptr = down_input + row * half_hidden_size + col;

  Pack16B<T> gate_pack;
  Pack16B<T> up_pack;
  Pack16B<T> out_pack;
  gate_pack.v = load_vec8(gate_ptr);
  up_pack.v = load_vec8(up_ptr);

#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    const float gate = to_float(gate_pack.elems[i]);
    const float up = to_float(up_pack.elems[i]);
    out_pack.elems[i] = cast_from_float<T>(apply_activation<ACTIVATION_TYPE>(gate) * up);
  }

  store_vec8(out_ptr, out_pack.v);
}

template <typename T, typename IdT, int ACTIVATION_TYPE, bool SKIP_EXPERT_CHECK = false>
__global__ __launch_bounds__(kVecThreads, 1)
void act_and_mul_row_vec8_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    const IdT* __restrict__ expert_ids,
    int64_t rows,
    int64_t half_hidden_size,
    int expert_step) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  if constexpr (!SKIP_EXPERT_CHECK) {
    const IdT expert_id = expert_ids[row / expert_step];
    if (expert_id == static_cast<IdT>(-1)) {
      return;
    }
  }

  const int64_t chunks = half_hidden_size / kVecElems;
  const int64_t hidden_size = half_hidden_size * 2;
  const T* gate_base = gateup_output + row * hidden_size;
  const T* up_base = gate_base + half_hidden_size;
  T* out_base = down_input + row * half_hidden_size;

  for (int64_t chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
    const int64_t col = chunk * kVecElems;
    const T* gate_ptr = gate_base + col;
    const T* up_ptr = up_base + col;
    T* out_ptr = out_base + col;

    Pack16B<T> gate_pack;
    Pack16B<T> up_pack;
    Pack16B<T> out_pack;
    gate_pack.v = load_vec8(gate_ptr);
    up_pack.v = load_vec8(up_ptr);

#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      const float gate = to_float(gate_pack.elems[i]);
      const float up = to_float(up_pack.elems[i]);
      out_pack.elems[i] =
          cast_from_float<T>(apply_activation<ACTIVATION_TYPE>(gate) * up);
    }

    store_vec8(out_ptr, out_pack.v);
  }
}

template <typename T, typename IdT, int ACTIVATION_TYPE, bool SKIP_EXPERT_CHECK = false>
__global__ __launch_bounds__(kThreads, 1)
void act_and_mul_row_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    const IdT* __restrict__ expert_ids,
    int64_t rows,
    int64_t half_hidden_size,
    int expert_step) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  if constexpr (!SKIP_EXPERT_CHECK) {
    const IdT expert_id = expert_ids[row / expert_step];
    if (expert_id == static_cast<IdT>(-1)) {
      return;
    }
  }

  const int64_t hidden_size = half_hidden_size * 2;
  const T* gate_ptr = gateup_output + row * hidden_size;
  const T* up_ptr = gate_ptr + half_hidden_size;
  T* out_ptr = down_input + row * half_hidden_size;

  for (int64_t col = threadIdx.x; col < half_hidden_size; col += blockDim.x) {
    const float gate = to_float(gate_ptr[col]);
    const float up = to_float(up_ptr[col]);
    out_ptr[col] = cast_from_float<T>(apply_activation<ACTIVATION_TYPE>(gate) * up);
  }
}

template <typename T, typename IdT, int ACTIVATION_TYPE, bool SKIP_EXPERT_CHECK = false>
__global__ __launch_bounds__(kThreads, 1)
void act_and_mul_split_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    const IdT* __restrict__ expert_ids,
    int64_t rows,
    int64_t half_hidden_size,
    int expert_step) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows) {
    return;
  }

  if constexpr (!SKIP_EXPERT_CHECK) {
    const IdT expert_id = expert_ids[row / expert_step];
    if (expert_id == static_cast<IdT>(-1)) {
      return;
    }
  }

  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= half_hidden_size) {
    return;
  }

  const int64_t hidden_size = half_hidden_size * 2;
  const T* gate_ptr = gateup_output + row * hidden_size;
  const T* up_ptr = gate_ptr + half_hidden_size;
  T* out_ptr = down_input + row * half_hidden_size;

  const float gate = to_float(gate_ptr[col]);
  const float up = to_float(up_ptr[col]);
  out_ptr[col] = cast_from_float<T>(apply_activation<ACTIVATION_TYPE>(gate) * up);
}

void check_act_and_mul_inputs(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView expert_ids,
    int64_t expert_step,
    int64_t activation_type,
    int64_t skip_expert_check) {
  CHECK_INPUT(gateup_output);
  CHECK_INPUT(down_input);
  CHECK_INPUT(expert_ids);
  CHECK_CONTIGUOUS_2D(gateup_output);
  CHECK_CONTIGUOUS_2D(down_input);
  TVM_FFI_ICHECK_EQ(gateup_output.device().device_id, down_input.device().device_id);
  TVM_FFI_ICHECK_EQ(gateup_output.device().device_id, expert_ids.device().device_id);
  TVM_FFI_ICHECK_EQ(gateup_output.size(0), down_input.size(0));
  TVM_FFI_ICHECK_EQ(gateup_output.size(1), down_input.size(1) * 2);
  TVM_FFI_ICHECK(dtype_equal(gateup_output.dtype(), down_input.dtype()));
  TVM_FFI_ICHECK(
      dtype_equal(gateup_output.dtype(), dl_float16) ||
      dtype_equal(gateup_output.dtype(), dl_bfloat16) ||
      dtype_equal(gateup_output.dtype(), dl_float32));
  TVM_FFI_ICHECK(
      dtype_equal(expert_ids.dtype(), dl_int64) ||
      dtype_equal(expert_ids.dtype(), dl_int32));
  TVM_FFI_ICHECK_GE(expert_step, 1);
  TVM_FFI_ICHECK(skip_expert_check == 0 || skip_expert_check == 1);
  if (skip_expert_check == 0) {
    TVM_FFI_ICHECK_GE(
        expert_ids.size(0), (down_input.size(0) + expert_step - 1) / expert_step);
  }
  TVM_FFI_ICHECK(
      activation_type == kSiluActivation || activation_type == kGeluActivation ||
      activation_type == kGeluTanhActivation);
}

template <typename T, typename IdT, int ACTIVATION_TYPE>
void launch_act_and_mul(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView expert_ids,
    int expert_step,
    bool skip_expert_check,
    musaStream_t stream) {
  const int64_t rows = down_input.size(0);
  const int64_t half_hidden_size = down_input.size(1);
  const int64_t tiles = (half_hidden_size + kThreads - 1) / kThreads;
  constexpr bool can_vec8 = !std::is_same_v<T, float>;
  const bool use_vec8 = can_vec8 && (half_hidden_size % kVecElems) == 0;
  if (use_vec8) {
    const int64_t chunks = half_hidden_size / kVecElems;
    const int vec_threads = chunks <= 128 ? 128 : kVecThreads;
    if (rows > 128) {
      const int64_t total_chunks = rows * chunks;
      if (rows <= static_cast<int64_t>(INT32_MAX) &&
          half_hidden_size <= static_cast<int64_t>(INT32_MAX) &&
          chunks <= static_cast<int64_t>(INT32_MAX) &&
          total_chunks <= static_cast<int64_t>(INT32_MAX)) {
        if (skip_expert_check) {
          if constexpr (std::is_same_v<T, half>) {
            if (rows >= 1024 && rows <= 4096) {
              const int64_t chunk_tiles =
                  (total_chunks + kFlatVecThreadsHalfMedium - 1) /
                  kFlatVecThreadsHalfMedium;
              act_and_mul_flat_vec8_no_filter_kernel<
                  T, IdT, ACTIVATION_TYPE, kFlatVecThreadsHalfMedium>
                  <<<dim3(static_cast<unsigned>(chunk_tiles)),
                     dim3(kFlatVecThreadsHalfMedium), 0, stream>>>(
                      static_cast<const T*>(gateup_output.data_ptr()),
                      static_cast<T*>(down_input.data_ptr()),
                      static_cast<int>(rows),
                      static_cast<int>(half_hidden_size),
                      static_cast<int>(chunks),
                      static_cast<int>(total_chunks));
            } else {
              const int64_t chunk_tiles =
                  (total_chunks + kFlatVecThreads - 1) / kFlatVecThreads;
              act_and_mul_flat_vec8_no_filter_kernel<
                  T, IdT, ACTIVATION_TYPE, kFlatVecThreads>
                  <<<dim3(static_cast<unsigned>(chunk_tiles)),
                     dim3(kFlatVecThreads), 0, stream>>>(
                      static_cast<const T*>(gateup_output.data_ptr()),
                      static_cast<T*>(down_input.data_ptr()),
                      static_cast<int>(rows),
                      static_cast<int>(half_hidden_size),
                      static_cast<int>(chunks),
                      static_cast<int>(total_chunks));
            }
          } else {
            const int64_t chunk_tiles =
                (total_chunks + kFlatVecThreads - 1) / kFlatVecThreads;
            act_and_mul_flat_vec8_no_filter_kernel<
                T, IdT, ACTIVATION_TYPE, kFlatVecThreads>
                <<<dim3(static_cast<unsigned>(chunk_tiles)),
                   dim3(kFlatVecThreads), 0, stream>>>(
                    static_cast<const T*>(gateup_output.data_ptr()),
                    static_cast<T*>(down_input.data_ptr()),
                    static_cast<int>(rows),
                    static_cast<int>(half_hidden_size),
                    static_cast<int>(chunks),
                    static_cast<int>(total_chunks));
          }
          return;
        }
        const int64_t chunk_tiles =
            (total_chunks + kFlatVecThreads * kFlatVecChunksPerThread - 1) /
            (kFlatVecThreads * kFlatVecChunksPerThread);
        if (rows >= 1024 && rows <= 4096) {
          if constexpr (std::is_same_v<T, half>) {
            const int64_t half_medium_chunk_tiles =
                (total_chunks +
                 kFlatVecThreadsHalfMedium * kFlatVecChunksPerThread - 1) /
                (kFlatVecThreadsHalfMedium * kFlatVecChunksPerThread);
            act_and_mul_flat_vec8_kernel<
                T, IdT, ACTIVATION_TYPE, kFlatVecThreadsHalfMedium>
                <<<dim3(static_cast<unsigned>(half_medium_chunk_tiles)),
                   dim3(kFlatVecThreadsHalfMedium), 0, stream>>>(
                    static_cast<const T*>(gateup_output.data_ptr()),
                    static_cast<T*>(down_input.data_ptr()),
                    static_cast<const IdT*>(expert_ids.data_ptr()),
                    static_cast<int>(rows),
                    static_cast<int>(half_hidden_size),
                    expert_step,
                    static_cast<int>(chunks),
                    static_cast<int>(total_chunks));
          } else {
            const int64_t medium_chunk_tiles =
                (total_chunks +
                 kFlatVecThreadsMedium * kFlatVecChunksPerThread - 1) /
                (kFlatVecThreadsMedium * kFlatVecChunksPerThread);
            act_and_mul_flat_vec8_kernel<
                T, IdT, ACTIVATION_TYPE, kFlatVecThreadsMedium>
                <<<dim3(static_cast<unsigned>(medium_chunk_tiles)),
                   dim3(kFlatVecThreadsMedium), 0, stream>>>(
                    static_cast<const T*>(gateup_output.data_ptr()),
                    static_cast<T*>(down_input.data_ptr()),
                    static_cast<const IdT*>(expert_ids.data_ptr()),
                    static_cast<int>(rows),
                    static_cast<int>(half_hidden_size),
                    expert_step,
                    static_cast<int>(chunks),
                    static_cast<int>(total_chunks));
          }
        } else {
          act_and_mul_flat_vec8_kernel<
              T, IdT, ACTIVATION_TYPE, kFlatVecThreads>
              <<<dim3(static_cast<unsigned>(chunk_tiles)),
                 dim3(kFlatVecThreads), 0, stream>>>(
                  static_cast<const T*>(gateup_output.data_ptr()),
                  static_cast<T*>(down_input.data_ptr()),
                  static_cast<const IdT*>(expert_ids.data_ptr()),
                  static_cast<int>(rows),
                  static_cast<int>(half_hidden_size),
                  expert_step,
                  static_cast<int>(chunks),
                  static_cast<int>(total_chunks));
        }
      } else {
        if (skip_expert_check) {
          act_and_mul_row_vec8_kernel<T, IdT, ACTIVATION_TYPE, true>
              <<<dim3(static_cast<unsigned>(rows)),
                 dim3(static_cast<unsigned>(vec_threads)), 0, stream>>>(
                  static_cast<const T*>(gateup_output.data_ptr()),
                  static_cast<T*>(down_input.data_ptr()),
                  static_cast<const IdT*>(expert_ids.data_ptr()),
                  rows,
                  half_hidden_size,
                  expert_step);
        } else {
          act_and_mul_row_vec8_kernel<T, IdT, ACTIVATION_TYPE>
              <<<dim3(static_cast<unsigned>(rows)),
                 dim3(static_cast<unsigned>(vec_threads)), 0, stream>>>(
                  static_cast<const T*>(gateup_output.data_ptr()),
                  static_cast<T*>(down_input.data_ptr()),
                  static_cast<const IdT*>(expert_ids.data_ptr()),
                  rows,
                  half_hidden_size,
                  expert_step);
        }
      }
      return;
    }

    const int64_t chunk_tiles =
        (chunks + vec_threads - 1) / vec_threads;
    dim3 grid(static_cast<unsigned>(chunk_tiles), static_cast<unsigned>(rows));
    if (skip_expert_check) {
      act_and_mul_vec8_kernel<T, IdT, ACTIVATION_TYPE, true>
          <<<grid, dim3(static_cast<unsigned>(vec_threads)), 0, stream>>>(
          static_cast<const T*>(gateup_output.data_ptr()),
          static_cast<T*>(down_input.data_ptr()),
          static_cast<const IdT*>(expert_ids.data_ptr()),
          rows,
          half_hidden_size,
          expert_step);
    } else {
      act_and_mul_vec8_kernel<T, IdT, ACTIVATION_TYPE>
          <<<grid, dim3(static_cast<unsigned>(vec_threads)), 0, stream>>>(
          static_cast<const T*>(gateup_output.data_ptr()),
          static_cast<T*>(down_input.data_ptr()),
          static_cast<const IdT*>(expert_ids.data_ptr()),
          rows,
          half_hidden_size,
          expert_step);
    }
    return;
  }

  const bool use_split = rows <= 128 && half_hidden_size > kThreads;

  if (use_split) {
    dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(rows));
    if (skip_expert_check) {
      act_and_mul_split_kernel<T, IdT, ACTIVATION_TYPE, true>
          <<<grid, dim3(kThreads), 0, stream>>>(
          static_cast<const T*>(gateup_output.data_ptr()),
          static_cast<T*>(down_input.data_ptr()),
          static_cast<const IdT*>(expert_ids.data_ptr()),
          rows,
          half_hidden_size,
          expert_step);
    } else {
      act_and_mul_split_kernel<T, IdT, ACTIVATION_TYPE>
          <<<grid, dim3(kThreads), 0, stream>>>(
          static_cast<const T*>(gateup_output.data_ptr()),
          static_cast<T*>(down_input.data_ptr()),
          static_cast<const IdT*>(expert_ids.data_ptr()),
          rows,
          half_hidden_size,
          expert_step);
    }
    return;
  }

  if (skip_expert_check) {
    act_and_mul_row_kernel<T, IdT, ACTIVATION_TYPE, true>
        <<<dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0, stream>>>(
            static_cast<const T*>(gateup_output.data_ptr()),
            static_cast<T*>(down_input.data_ptr()),
            static_cast<const IdT*>(expert_ids.data_ptr()),
            rows,
            half_hidden_size,
            expert_step);
  } else {
    act_and_mul_row_kernel<T, IdT, ACTIVATION_TYPE>
        <<<dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0, stream>>>(
            static_cast<const T*>(gateup_output.data_ptr()),
            static_cast<T*>(down_input.data_ptr()),
            static_cast<const IdT*>(expert_ids.data_ptr()),
            rows,
            half_hidden_size,
            expert_step);
  }
}

template <typename T, int ACTIVATION_TYPE>
void dispatch_act_and_mul_dtype(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView expert_ids,
    int expert_step,
    int activation_type,
    bool skip_expert_check,
    musaStream_t stream) {
  if (dtype_equal(expert_ids.dtype(), dl_int64)) {
    launch_act_and_mul<T, int64_t, ACTIVATION_TYPE>(
        gateup_output, down_input, expert_ids, expert_step, skip_expert_check, stream);
  } else {
    launch_act_and_mul<T, int32_t, ACTIVATION_TYPE>(
        gateup_output, down_input, expert_ids, expert_step, skip_expert_check, stream);
  }
}

template <typename T>
void dispatch_act_and_mul_activation(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView expert_ids,
    int expert_step,
    int activation_type,
    bool skip_expert_check,
    musaStream_t stream) {
  if (activation_type == kGeluActivation) {
    dispatch_act_and_mul_dtype<T, kGeluActivation>(
        gateup_output, down_input, expert_ids, expert_step, activation_type, skip_expert_check, stream);
  } else if (activation_type == kGeluTanhActivation) {
    dispatch_act_and_mul_dtype<T, kGeluTanhActivation>(
        gateup_output, down_input, expert_ids, expert_step, activation_type, skip_expert_check, stream);
  } else {
    dispatch_act_and_mul_dtype<T, kSiluActivation>(
        gateup_output, down_input, expert_ids, expert_step, activation_type, skip_expert_check, stream);
  }
}

void sgl_musa_act_and_mul(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView expert_ids,
    int64_t expert_step,
    int64_t activation_type,
    int64_t skip_expert_check) {
  check_act_and_mul_inputs(
      gateup_output, down_input, expert_ids, expert_step, activation_type, skip_expert_check);
  ffi::MUSADeviceGuard device_guard(gateup_output.device().device_id);
  musaStream_t stream = get_stream(gateup_output.device());
  const int step = static_cast<int>(expert_step);
  const int act = static_cast<int>(activation_type);
  const bool skip_check = skip_expert_check != 0;

  if (dtype_equal(gateup_output.dtype(), dl_float16)) {
    dispatch_act_and_mul_activation<half>(
        gateup_output, down_input, expert_ids, step, act, skip_check, stream);
  } else if (dtype_equal(gateup_output.dtype(), dl_bfloat16)) {
    dispatch_act_and_mul_activation<__mt_bfloat16>(
        gateup_output, down_input, expert_ids, step, act, skip_check, stream);
  } else if (dtype_equal(gateup_output.dtype(), dl_float32)) {
    dispatch_act_and_mul_activation<float>(
        gateup_output, down_input, expert_ids, step, act, skip_check, stream);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported act_and_mul dtype";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA act_and_mul kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_act_and_mul, sgl_musa_act_and_mul);
