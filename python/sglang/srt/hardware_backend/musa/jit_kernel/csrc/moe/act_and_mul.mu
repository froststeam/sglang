#include <cstdint>
#include <climits>
#include <string>
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

constexpr int kThreads = 512;
constexpr int kVecThreads = 256;
constexpr int kFlatVecThreads = 512;
constexpr int kFlatVecThreadsMedium = 256;
constexpr int kFlatVecThreadsHalfMedium = 384;
constexpr int kFlatVecChunksPerThread = 1;
constexpr int kMaskedRowsPerBlock = 8;
constexpr int kSiluActivation = 0;
constexpr int kGeluActivation = 1;
constexpr int kGeluTanhActivation = 2;
constexpr int kVecElems = 8;

template <typename T>
__device__ __forceinline__ T cast_from_float(float value) {
  return from_float<T>(value);
}

template <>
__device__ __forceinline__ float cast_from_float<float>(float value) {
  return value;
}

__device__ __forceinline__ float fast_erf(float x) {
  constexpr float kP = 0.3275911f;
  constexpr float kA1 = 0.254829592f;
  constexpr float kA2 = -0.284496736f;
  constexpr float kA3 = 1.421413741f;
  constexpr float kA4 = -1.453152027f;
  constexpr float kA5 = 1.061405429f;
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float ax = fabsf(x);
  const float t = 1.0f / (1.0f + kP * ax);
  const float poly =
      (((((kA5 * t + kA4) * t + kA3) * t + kA2) * t + kA1) * t);
  return sign * (1.0f - poly * expf(-ax * ax));
}

__device__ __forceinline__ float gelu_exact(float x) {
  return 0.5f * x * (1.0f + fast_erf(x * 0.7071067811865475f));
}

__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float kGeluTanhSqrt2OverPi = 0.7978845608028654f;
  constexpr float kGeluTanhCoeff = 0.044715f;
  return 0.5f * x *
      (1.0f + tanhf(kGeluTanhSqrt2OverPi * (x + kGeluTanhCoeff * x * x * x)));
}

__device__ __forceinline__ float fast_silu(float x) {
  const float half_x = 0.5f * x;
  return half_x * (1.0f + tanhf(half_x));
}

__device__ __forceinline__ float apply_activation(float x, int activation_type) {
  if (activation_type == kGeluActivation) {
    return gelu_exact(x);
  }
  if (activation_type == kGeluTanhActivation) {
    return gelu_tanh(x);
  }
  return fast_silu(x);
}

template <int ACTIVATION_TYPE>
__device__ __forceinline__ float apply_activation(float x) {
  if constexpr (ACTIVATION_TYPE == kGeluActivation) {
    return gelu_exact(x);
  } else if constexpr (ACTIVATION_TYPE == kGeluTanhActivation) {
    return gelu_tanh(x);
  } else {
    return fast_silu(x);
  }
}

template <typename T>
union Pack16B {
  int4 v;
  T elems[kVecElems];
};

template <typename T>
__device__ __forceinline__ int4 load_vec8(const T* ptr) {
  return *reinterpret_cast<const int4*>(ptr);
}

template <typename T>
__device__ __forceinline__ void store_vec8(T* ptr, const int4& value) {
  *reinterpret_cast<int4*>(ptr) = value;
}

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

template <typename T, int ACTIVATION_TYPE>
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
  const int row_base = static_cast<int>(blockIdx.y) * kMaskedRowsPerBlock;
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
  for (int r = 0; r < kMaskedRowsPerBlock; ++r) {
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

template <typename T, int ACTIVATION_TYPE>
__global__ __launch_bounds__(kThreads, 1)
void act_and_mul_masked_kernel(
    const T* __restrict__ gateup_output,
    T* __restrict__ down_input,
    const int32_t* __restrict__ masked_m,
    int experts,
    int m,
    int half_hidden_size) {
  const int expert = static_cast<int>(blockIdx.z);
  const int row_base = static_cast<int>(blockIdx.y) * kMaskedRowsPerBlock;
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
  for (int r = 0; r < kMaskedRowsPerBlock; ++r) {
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
    const int row_tiles = (m + kMaskedRowsPerBlock - 1) / kMaskedRowsPerBlock;
    dim3 grid(
        static_cast<unsigned>(chunk_tiles),
        static_cast<unsigned>(row_tiles),
        static_cast<unsigned>(experts));
    act_and_mul_masked_vec8_kernel<T, ACTIVATION_TYPE>
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
  const int row_tiles = (m + kMaskedRowsPerBlock - 1) / kMaskedRowsPerBlock;
  dim3 grid(
      static_cast<unsigned>(tiles),
      static_cast<unsigned>(row_tiles),
      static_cast<unsigned>(experts));
  act_and_mul_masked_kernel<T, ACTIVATION_TYPE>
      <<<grid, dim3(kThreads), 0, stream>>>(
          static_cast<const T*>(gateup_output.data_ptr()),
          static_cast<T*>(down_input.data_ptr()),
          static_cast<const int32_t*>(masked_m.data_ptr()),
          experts,
          m,
          half_hidden_size);
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

template <typename T>
void dispatch_act_and_mul_masked_activation(
    ffi::TensorView gateup_output,
    ffi::TensorView down_input,
    ffi::TensorView masked_m,
    int activation_type,
    musaStream_t stream) {
  if (activation_type == kGeluActivation) {
    launch_act_and_mul_masked<T, kGeluActivation>(
        gateup_output, down_input, masked_m, stream);
  } else if (activation_type == kGeluTanhActivation) {
    launch_act_and_mul_masked<T, kGeluTanhActivation>(
        gateup_output, down_input, masked_m, stream);
  } else {
    launch_act_and_mul_masked<T, kSiluActivation>(
        gateup_output, down_input, masked_m, stream);
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
