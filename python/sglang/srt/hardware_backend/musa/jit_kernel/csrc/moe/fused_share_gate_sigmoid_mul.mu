#include <cmath>
#include <cstdint>
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

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

#ifndef SGLANG_FSG_HDIM_WARPS_PER_BLOCK
#define SGLANG_FSG_HDIM_WARPS_PER_BLOCK 8
#endif

#ifndef SGLANG_FSG_GENERIC_FAST_SIGMOID
#define SGLANG_FSG_GENERIC_FAST_SIGMOID 0
#endif

#ifndef SGLANG_FSG_HDIM_FAST_SIGMOID
#define SGLANG_FSG_HDIM_FAST_SIGMOID 0
#endif

#ifndef SGLANG_FSG_HDIM_CACHE_WEIGHT
#define SGLANG_FSG_HDIM_CACHE_WEIGHT 0
#endif

#ifndef SGLANG_FSG_HDIM_MIN_TOKENS
#define SGLANG_FSG_HDIM_MIN_TOKENS 2048
#endif

#ifndef SGLANG_FSG_HDIM_BF16_SMALL_MAX_TOKENS
#define SGLANG_FSG_HDIM_BF16_SMALL_MAX_TOKENS 1024
#endif

#ifndef SGLANG_FSG_HDIM_PAIR_CONVERT
#define SGLANG_FSG_HDIM_PAIR_CONVERT 0
#endif

#ifndef SGLANG_FSG_HDIM_BF16_TPR
#define SGLANG_FSG_HDIM_BF16_TPR 1
#endif

#ifndef SGLANG_FSG_HDIM_BF16_TPR_AUTO
#define SGLANG_FSG_HDIM_BF16_TPR_AUTO 1
#endif

#ifndef SGLANG_FSG_HDIM_BF16_THREADS_PER_ROW
#define SGLANG_FSG_HDIM_BF16_THREADS_PER_ROW 192
#endif

#ifndef SGLANG_FSG_HDIM_BF16_BLOCK_WARPS
#define SGLANG_FSG_HDIM_BF16_BLOCK_WARPS 6
#endif

#ifndef SGLANG_FSG_HDIM_BF16_WARP_LOCAL_GATE
#define SGLANG_FSG_HDIM_BF16_WARP_LOCAL_GATE 1
#endif

template <typename T>
__device__ __forceinline__ int4 load_vec8_fsg(const T* ptr) {
  return *reinterpret_cast<const int4*>(ptr);
}

template <typename T>
__device__ __forceinline__ void store_vec8_fsg(T* ptr, const int4& value) {
  *reinterpret_cast<int4*>(ptr) = value;
}

template <typename T>
union Pack16B {
  int4 v;
  T elems[8];
  __mt_bfloat162 bf16_pairs[4];
};

__device__ __forceinline__ float warp_sum_fsg(float val) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ __forceinline__ float stable_sigmoid_fsg(float x) {
  const bool positive = x >= 0.0f;
  const float z = __expf(positive ? -x : x);
  const float inv = 1.0f / (1.0f + z);
  return positive ? inv : z * inv;
}

__device__ __forceinline__ float fast_sigmoid_fsg(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

template <typename T>
__global__ __launch_bounds__(kWarpsPerBlock * kWarpSize, 1)
void fused_share_gate_sigmoid_mul_kernel(
    T* __restrict__ output,
    const T* __restrict__ hidden_state,
    const T* __restrict__ share_gate_weight,
    const T* __restrict__ share_expert_output,
    int64_t token_num,
    int64_t hidden_dim) {
  constexpr int VEC = 8;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t token =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;

  if (token >= token_num) {
    return;
  }

  float gate_value = 0.0f;
  const int64_t chunks = hidden_dim / VEC;
  for (int64_t chunk = lane; chunk < chunks; chunk += kWarpSize) {
    const int64_t offset = token * hidden_dim + chunk * VEC;
    Pack16B<T> hidden_pack;
    Pack16B<T> weight_pack;
    hidden_pack.v = load_vec8_fsg(hidden_state + offset);
    weight_pack.v = load_vec8_fsg(share_gate_weight + chunk * VEC);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      gate_value += to_float(hidden_pack.elems[i]) * to_float(weight_pack.elems[i]);
    }
  }

  gate_value = warp_sum_fsg(gate_value);
  const float gate_input = __shfl_sync(0xffffffff, gate_value, 0);
  const float gate =
#if SGLANG_FSG_GENERIC_FAST_SIGMOID
      fast_sigmoid_fsg(gate_input);
#else
      stable_sigmoid_fsg(gate_input);
#endif

  for (int64_t chunk = lane; chunk < chunks; chunk += kWarpSize) {
    const int64_t offset = token * hidden_dim + chunk * VEC;
    Pack16B<T> shared_pack;
    Pack16B<T> out_pack;
    shared_pack.v = load_vec8_fsg(share_expert_output + offset);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      out_pack.elems[i] = from_float<T>(to_float(shared_pack.elems[i]) * gate);
    }
    store_vec8_fsg(output + offset, out_pack.v);
  }
}

template <typename T, int HIDDEN_DIM, int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * kWarpSize, 1)
void fused_share_gate_sigmoid_mul_hdim_kernel(
    T* __restrict__ output,
    const T* __restrict__ hidden_state,
    const T* __restrict__ share_gate_weight,
    const T* __restrict__ share_expert_output,
    int token_num) {
  constexpr int VEC = 8;
  constexpr int CHUNKS = HIDDEN_DIM / VEC;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int token = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp_id;

#if SGLANG_FSG_HDIM_CACHE_WEIGHT
  __shared__ int4 weight_chunks[CHUNKS];
  for (int chunk = threadIdx.x; chunk < CHUNKS; chunk += blockDim.x) {
    weight_chunks[chunk] = load_vec8_fsg(share_gate_weight + chunk * VEC);
  }
  __syncthreads();
#endif

  if (token >= token_num) {
    return;
  }

  float gate_value = 0.0f;
  const int token_base = token * HIDDEN_DIM;
  for (int chunk = lane; chunk < CHUNKS; chunk += kWarpSize) {
    const int offset = token_base + chunk * VEC;
    Pack16B<T> hidden_pack;
    Pack16B<T> weight_pack;
    hidden_pack.v = load_vec8_fsg(hidden_state + offset);
#if SGLANG_FSG_HDIM_CACHE_WEIGHT
    weight_pack.v = weight_chunks[chunk];
#else
    weight_pack.v = load_vec8_fsg(share_gate_weight + chunk * VEC);
#endif
#if SGLANG_FSG_HDIM_PAIR_CONVERT
    if constexpr (std::is_same_v<T, __mt_bfloat16>) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 h = __bfloat1622float2(hidden_pack.bf16_pairs[i]);
        const float2 w = __bfloat1622float2(weight_pack.bf16_pairs[i]);
        gate_value += h.x * w.x + h.y * w.y;
      }
    } else
#endif
    {
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        gate_value += to_float(hidden_pack.elems[i]) * to_float(weight_pack.elems[i]);
      }
    }
  }

  gate_value = warp_sum_fsg(gate_value);
  const float gate_input = __shfl_sync(0xffffffff, gate_value, 0);
  const float gate =
#if SGLANG_FSG_HDIM_FAST_SIGMOID
      fast_sigmoid_fsg(gate_input);
#else
      stable_sigmoid_fsg(gate_input);
#endif

  for (int chunk = lane; chunk < CHUNKS; chunk += kWarpSize) {
    const int offset = token_base + chunk * VEC;
    Pack16B<T> shared_pack;
    Pack16B<T> out_pack;
    shared_pack.v = load_vec8_fsg(share_expert_output + offset);
#if SGLANG_FSG_HDIM_PAIR_CONVERT
    if constexpr (std::is_same_v<T, __mt_bfloat16>) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 x = __bfloat1622float2(shared_pack.bf16_pairs[i]);
        float2 y;
        y.x = x.x * gate;
        y.y = x.y * gate;
        out_pack.bf16_pairs[i] = __float22bfloat162_rn(y);
      }
    } else
#endif
    {
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        out_pack.elems[i] = from_float<T>(to_float(shared_pack.elems[i]) * gate);
      }
    }
    store_vec8_fsg(output + offset, out_pack.v);
  }
}

template <int HIDDEN_DIM, int BlockWarps, int ThreadsPerRow>
__global__ __launch_bounds__(BlockWarps * kWarpSize, 1)
void fused_share_gate_sigmoid_mul_hdim_bf16_tpr_kernel(
    __mt_bfloat16* __restrict__ output,
    const __mt_bfloat16* __restrict__ hidden_state,
    const __mt_bfloat16* __restrict__ share_gate_weight,
    const __mt_bfloat16* __restrict__ share_expert_output,
    int token_num) {
  static_assert(ThreadsPerRow % kWarpSize == 0);
  constexpr int VEC = 8;
  constexpr int CHUNKS = HIDDEN_DIM / VEC;
  constexpr int WarpsPerRow = ThreadsPerRow / kWarpSize;
  static_assert(BlockWarps % WarpsPerRow == 0);
  constexpr int RowsPerBlock = BlockWarps / WarpsPerRow;

  const int tid = threadIdx.x;
  const int row_in_block = tid / ThreadsPerRow;
  const int lane_in_row = tid - row_in_block * ThreadsPerRow;
  const int warp_in_row = lane_in_row / kWarpSize;
  const int lane = lane_in_row & (kWarpSize - 1);
  const int token = static_cast<int>(blockIdx.x) * RowsPerBlock + row_in_block;

  __shared__ float partial_sums[RowsPerBlock][WarpsPerRow];
  __shared__ float gate_values[RowsPerBlock];

  if (token < token_num) {
    float gate_value = 0.0f;
    const int token_base = token * HIDDEN_DIM;
    for (int chunk = lane_in_row; chunk < CHUNKS; chunk += ThreadsPerRow) {
      const int offset = token_base + chunk * VEC;
      Pack16B<__mt_bfloat16> hidden_pack;
      Pack16B<__mt_bfloat16> weight_pack;
      hidden_pack.v = load_vec8_fsg(hidden_state + offset);
      weight_pack.v = load_vec8_fsg(share_gate_weight + chunk * VEC);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 h = __bfloat1622float2(hidden_pack.bf16_pairs[i]);
        const float2 w = __bfloat1622float2(weight_pack.bf16_pairs[i]);
        gate_value += h.x * w.x + h.y * w.y;
      }
    }

    gate_value = warp_sum_fsg(gate_value);
    if (lane == 0) {
      partial_sums[row_in_block][warp_in_row] = gate_value;
    }
  }
  __syncthreads();

#if SGLANG_FSG_HDIM_BF16_WARP_LOCAL_GATE
  float gate = 0.0f;
  if (token < token_num) {
    float gate_input = 0.0f;
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < WarpsPerRow; ++i) {
        gate_input += partial_sums[row_in_block][i];
      }
      gate_input =
#if SGLANG_FSG_HDIM_FAST_SIGMOID
          fast_sigmoid_fsg(gate_input);
#else
          stable_sigmoid_fsg(gate_input);
#endif
    }
    gate = __shfl_sync(0xffffffff, gate_input, 0);
  }
#else
  if (token < token_num && lane_in_row == 0) {
    float gate_input = 0.0f;
#pragma unroll
    for (int i = 0; i < WarpsPerRow; ++i) {
      gate_input += partial_sums[row_in_block][i];
    }
    gate_values[row_in_block] =
#if SGLANG_FSG_HDIM_FAST_SIGMOID
        fast_sigmoid_fsg(gate_input);
#else
        stable_sigmoid_fsg(gate_input);
#endif
  }
  __syncthreads();
#endif

  if (token >= token_num) {
    return;
  }

#if !SGLANG_FSG_HDIM_BF16_WARP_LOCAL_GATE
  const float gate = gate_values[row_in_block];
#endif
  const int token_base = token * HIDDEN_DIM;
  for (int chunk = lane_in_row; chunk < CHUNKS; chunk += ThreadsPerRow) {
    const int offset = token_base + chunk * VEC;
    Pack16B<__mt_bfloat16> shared_pack;
    Pack16B<__mt_bfloat16> out_pack;
    shared_pack.v = load_vec8_fsg(share_expert_output + offset);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 x = __bfloat1622float2(shared_pack.bf16_pairs[i]);
      float2 y;
      y.x = x.x * gate;
      y.y = x.y * gate;
      out_pack.bf16_pairs[i] = __float22bfloat162_rn(y);
    }
    store_vec8_fsg(output + offset, out_pack.v);
  }
}

template <int HIDDEN_DIM, int BlockWarps, int ThreadsPerRow>
void launch_fused_share_gate_sigmoid_mul_hdim_bf16_tpr(
    ffi::TensorView output,
    ffi::TensorView hidden_state,
    ffi::TensorView share_gate_weight,
    ffi::TensorView share_expert_output,
    int64_t token_num,
    musaStream_t stream) {
  constexpr int warps_per_row = ThreadsPerRow / kWarpSize;
  constexpr int rows_per_block = BlockWarps / warps_per_row;
  static_assert(ThreadsPerRow % kWarpSize == 0);
  static_assert(BlockWarps % warps_per_row == 0);
  const int64_t grid_x = (token_num + rows_per_block - 1) / rows_per_block;
  fused_share_gate_sigmoid_mul_hdim_bf16_tpr_kernel<
      HIDDEN_DIM, BlockWarps, ThreadsPerRow>
      <<<dim3(static_cast<unsigned>(grid_x)),
         dim3(BlockWarps * kWarpSize), 0, stream>>>(
          static_cast<__mt_bfloat16*>(output.data_ptr()),
          static_cast<const __mt_bfloat16*>(hidden_state.data_ptr()),
          static_cast<const __mt_bfloat16*>(share_gate_weight.data_ptr()),
          static_cast<const __mt_bfloat16*>(share_expert_output.data_ptr()),
          static_cast<int>(token_num));
}

void check_fused_share_gate_sigmoid_mul_inputs(
    ffi::TensorView output,
    ffi::TensorView hidden_state,
    ffi::TensorView share_gate_weight,
    ffi::TensorView share_expert_output) {
  CHECK_MUSA_CONTIGUOUS(output);
  CHECK_MUSA_CONTIGUOUS(hidden_state);
  CHECK_MUSA_CONTIGUOUS(share_gate_weight);
  CHECK_MUSA_CONTIGUOUS(share_expert_output);
  CHECK_CONTIGUOUS_2D(output);
  CHECK_CONTIGUOUS_2D(hidden_state);
  CHECK_CONTIGUOUS_2D(share_gate_weight);
  CHECK_CONTIGUOUS_2D(share_expert_output);
  TVM_FFI_ICHECK_EQ(hidden_state.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(hidden_state.size(1), output.size(1));
  TVM_FFI_ICHECK_EQ(share_expert_output.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(share_expert_output.size(1), output.size(1));
  TVM_FFI_ICHECK_EQ(share_gate_weight.size(0), 1);
  TVM_FFI_ICHECK_EQ(share_gate_weight.size(1), output.size(1));
  TVM_FFI_ICHECK_EQ(output.size(1) % 8, 0);
  TVM_FFI_ICHECK_EQ(output.device().device_id, hidden_state.device().device_id);
  TVM_FFI_ICHECK_EQ(output.device().device_id, share_gate_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(output.device().device_id, share_expert_output.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(output.dtype(), hidden_state.dtype()));
  TVM_FFI_ICHECK(dtype_equal(output.dtype(), share_gate_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(output.dtype(), share_expert_output.dtype()));
  TVM_FFI_ICHECK(dtype_equal(output.dtype(), dl_float16) ||
                 dtype_equal(output.dtype(), dl_bfloat16));
}

template <typename T>
void launch_fused_share_gate_sigmoid_mul(
    ffi::TensorView output,
    ffi::TensorView hidden_state,
    ffi::TensorView share_gate_weight,
    ffi::TensorView share_expert_output) {
  const int64_t token_num = output.size(0);
  const int64_t hidden_dim = output.size(1);
  if (token_num == 0 || hidden_dim == 0) {
    return;
  }

  musaStream_t stream = get_stream(output.device());
#if SGLANG_FSG_HDIM_BF16_TPR
  if (hidden_dim == 3072 &&
      token_num <= SGLANG_FSG_HDIM_BF16_SMALL_MAX_TOKENS &&
      token_num <= INT32_MAX) {
    if constexpr (std::is_same_v<T, __mt_bfloat16>) {
      if (token_num <= 64) {
        launch_fused_share_gate_sigmoid_mul_hdim_bf16_tpr<3072, 7, 224>(
            output,
            hidden_state,
            share_gate_weight,
            share_expert_output,
            token_num,
            stream);
      } else if (token_num <= 128) {
        launch_fused_share_gate_sigmoid_mul_hdim_bf16_tpr<3072, 6, 192>(
            output,
            hidden_state,
            share_gate_weight,
            share_expert_output,
            token_num,
            stream);
      } else {
        launch_fused_share_gate_sigmoid_mul_hdim_bf16_tpr<3072, 3, 96>(
            output,
            hidden_state,
            share_gate_weight,
            share_expert_output,
            token_num,
            stream);
      }
      return;
    }
  }
#endif
  if (hidden_dim == 3072 && token_num >= SGLANG_FSG_HDIM_MIN_TOKENS &&
      token_num <= INT32_MAX) {
    constexpr int hdim_warps_per_block = SGLANG_FSG_HDIM_WARPS_PER_BLOCK;
#if SGLANG_FSG_HDIM_BF16_TPR
    if constexpr (std::is_same_v<T, __mt_bfloat16>) {
#if SGLANG_FSG_HDIM_BF16_TPR_AUTO
      if (token_num >= 24576) {
        launch_fused_share_gate_sigmoid_mul_hdim_bf16_tpr<3072, 14, 224>(
            output,
            hidden_state,
            share_gate_weight,
            share_expert_output,
            token_num,
            stream);
      } else if (token_num >= 12288) {
        launch_fused_share_gate_sigmoid_mul_hdim_bf16_tpr<3072, 6, 192>(
            output,
            hidden_state,
            share_gate_weight,
            share_expert_output,
            token_num,
            stream);
      } else {
        launch_fused_share_gate_sigmoid_mul_hdim_bf16_tpr<3072, 3, 96>(
            output,
            hidden_state,
            share_gate_weight,
            share_expert_output,
            token_num,
            stream);
      }
      return;
#else
      constexpr int threads_per_row = SGLANG_FSG_HDIM_BF16_THREADS_PER_ROW;
      constexpr int warps_per_row = threads_per_row / kWarpSize;
      constexpr int bf16_block_warps = SGLANG_FSG_HDIM_BF16_BLOCK_WARPS;
      constexpr int rows_per_block = bf16_block_warps / warps_per_row;
      static_assert(threads_per_row % kWarpSize == 0);
      static_assert(bf16_block_warps % warps_per_row == 0);
      const int64_t hdim_grid_x =
          (token_num + rows_per_block - 1) / rows_per_block;
      fused_share_gate_sigmoid_mul_hdim_bf16_tpr_kernel<
          3072, bf16_block_warps, threads_per_row>
          <<<dim3(static_cast<unsigned>(hdim_grid_x)),
             dim3(bf16_block_warps * kWarpSize), 0, stream>>>(
              static_cast<__mt_bfloat16*>(output.data_ptr()),
              static_cast<const __mt_bfloat16*>(hidden_state.data_ptr()),
              static_cast<const __mt_bfloat16*>(share_gate_weight.data_ptr()),
              static_cast<const __mt_bfloat16*>(share_expert_output.data_ptr()),
              static_cast<int>(token_num));
      return;
#endif
    }
#endif
    const int64_t hdim_grid_x =
        (token_num + hdim_warps_per_block - 1) / hdim_warps_per_block;
    fused_share_gate_sigmoid_mul_hdim_kernel<T, 3072, hdim_warps_per_block>
        <<<dim3(static_cast<unsigned>(hdim_grid_x)),
           dim3(hdim_warps_per_block * kWarpSize), 0, stream>>>(
            static_cast<T*>(output.data_ptr()),
            static_cast<const T*>(hidden_state.data_ptr()),
            static_cast<const T*>(share_gate_weight.data_ptr()),
            static_cast<const T*>(share_expert_output.data_ptr()),
            static_cast<int>(token_num));
    return;
  }
  const int64_t grid_x =
      (token_num + kWarpsPerBlock - 1) / kWarpsPerBlock;
  fused_share_gate_sigmoid_mul_kernel<T>
        <<<dim3(static_cast<unsigned>(grid_x)),
           dim3(kWarpsPerBlock * kWarpSize), 0, stream>>>(
            static_cast<T*>(output.data_ptr()),
            static_cast<const T*>(hidden_state.data_ptr()),
            static_cast<const T*>(share_gate_weight.data_ptr()),
            static_cast<const T*>(share_expert_output.data_ptr()),
            token_num,
            hidden_dim);
}

void sgl_musa_fused_share_gate_sigmoid_mul(
    ffi::TensorView output,
    ffi::TensorView hidden_state,
    ffi::TensorView share_gate_weight,
    ffi::TensorView share_expert_output) {
  check_fused_share_gate_sigmoid_mul_inputs(
      output, hidden_state, share_gate_weight, share_expert_output);
  ffi::MUSADeviceGuard device_guard(output.device().device_id);
  if (dtype_equal(output.dtype(), dl_float16)) {
    launch_fused_share_gate_sigmoid_mul<half>(
        output, hidden_state, share_gate_weight, share_expert_output);
  } else if (dtype_equal(output.dtype(), dl_bfloat16)) {
    launch_fused_share_gate_sigmoid_mul<__mt_bfloat16>(
        output, hidden_state, share_gate_weight, share_expert_output);
  } else {
    TVM_FFI_THROW(ValueError)
        << "Unsupported fused_share_gate_sigmoid_mul dtype";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA fused_share_gate_sigmoid_mul kernel failed: "
      << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_fused_share_gate_sigmoid_mul,
    sgl_musa_fused_share_gate_sigmoid_mul);
