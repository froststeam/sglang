#include <cmath>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp8.h>
#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

#include "../common.h"
#include "../device_utils.h"

constexpr int kSiluActivation = 0;
constexpr int kGeluActivation = 1;
constexpr int kGeluTanhActivation = 2;
constexpr int kGroupSize = 128;
constexpr int kThreadsPerGroup = 32;
constexpr int kElemsPerThread = 4;
constexpr int kMaskedRowsPerBlock = 4;
constexpr float kFp8E4M3Max = 448.0f;
constexpr float kLocalAbsMaxMin = 1.0e-10f;

__device__ __forceinline__ float fast_erf_musa(float x) {
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

__device__ __forceinline__ float gelu_exact_musa(float x) {
  return 0.5f * x * (1.0f + fast_erf_musa(x * 0.7071067811865475f));
}

__device__ __forceinline__ float gelu_tanh_musa(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;
  return 0.5f * x *
      (1.0f + tanhf(kSqrt2OverPi * (x + kCoeff * x * x * x)));
}

__device__ __forceinline__ float silu_musa(float x) {
  const float half_x = 0.5f * x;
  return half_x * (1.0f + tanhf(half_x));
}

template <int ACTIVATION_TYPE>
__device__ __forceinline__ float apply_activation_musa(float x) {
  if constexpr (ACTIVATION_TYPE == kGeluActivation) {
    return gelu_exact_musa(x);
  } else if constexpr (ACTIVATION_TYPE == kGeluTanhActivation) {
    return gelu_tanh_musa(x);
  } else {
    return silu_musa(x);
  }
}

__device__ __forceinline__ float clamp_gate_musa(float x, float swiglu_limit) {
  return swiglu_limit > 0.0f ? fminf(x, swiglu_limit) : x;
}

__device__ __forceinline__ float clamp_up_musa(float x, float swiglu_limit) {
  return swiglu_limit > 0.0f ? fminf(fmaxf(x, -swiglu_limit), swiglu_limit) : x;
}

template <int SUBWARPS_PER_BLOCK, int ACTIVATION_TYPE, int ROWS_PER_BLOCK>
__global__ void act_and_mul_masked_post_quant_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output,
    float* __restrict__ output_scale,
    const int32_t* __restrict__ masked_m,
    int hidden_groups,
    int tokens_per_expert,
    float swiglu_limit,
    bool swizzle) {
  const int subwarp_id = threadIdx.x / kThreadsPerGroup;
  const int lane_id = threadIdx.x - subwarp_id * kThreadsPerGroup;
  const int hidden_group = blockIdx.x * SUBWARPS_PER_BLOCK + subwarp_id;
  const int token_base = blockIdx.y * ROWS_PER_BLOCK;
  const int expert = blockIdx.z;
  if (hidden_group >= hidden_groups || token_base >= masked_m[expert]) {
    return;
  }

  const int hidden = hidden_groups * kGroupSize;
  const int elem_offset = lane_id * kElemsPerThread;
  const int swizzled_chunk_offset =
      (elem_offset / 8) * 16 + (elem_offset % 8);
  const int valid_m = masked_m[expert];
#pragma unroll
  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const int token = token_base + r;
    if (token < tokens_per_expert && token < valid_m) {
      const int64_t input_base =
          (static_cast<int64_t>(expert) * tokens_per_expert + token) *
              hidden * 2 +
          (swizzle ? hidden_group * kGroupSize * 2 : hidden_group * kGroupSize);
      const int64_t gate_offset =
          input_base + (swizzle ? swizzled_chunk_offset : elem_offset);
      const int64_t up_offset =
          input_base + (swizzle ? swizzled_chunk_offset + 8
                                : hidden + elem_offset);
      const int64_t output_base =
          (static_cast<int64_t>(expert) * tokens_per_expert + token) * hidden +
          hidden_group * kGroupSize + elem_offset;

      uint64_t gate_u64 = *reinterpret_cast<const uint64_t*>(input + gate_offset);
      uint64_t up_u64 = *reinterpret_cast<const uint64_t*>(input + up_offset);
      auto gate_vec = reinterpret_cast<__mt_bfloat16*>(&gate_u64);
      auto up_vec = reinterpret_cast<__mt_bfloat16*>(&up_u64);

      float values[kElemsPerThread];
      float local_absmax = kLocalAbsMaxMin;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        const float gate_value =
            clamp_gate_musa(__bfloat162float(gate_vec[i]), swiglu_limit);
        const float up_value =
            clamp_up_musa(__bfloat162float(up_vec[i]), swiglu_limit);
        __mt_bfloat16 gate_lowprec = __float2bfloat16_rn(
            apply_activation_musa<ACTIVATION_TYPE>(gate_value));
        __mt_bfloat16 up_lowprec = __float2bfloat16_rn(up_value);
        __mt_bfloat16 val_lowprec = gate_lowprec * up_lowprec;
        const float val = __bfloat162float(val_lowprec);
        values[i] = val;
        local_absmax = fmaxf(local_absmax, fabsf(val));
      }

      local_absmax = GroupReduceMax<kThreadsPerGroup>(local_absmax, lane_id);
      const float inv_absmax = 1.0f / local_absmax;
      const float scale_inv = local_absmax * (1.0f / kFp8E4M3Max);
      const float scale = kFp8E4M3Max * inv_absmax;

      if (lane_id == 0) {
        const int64_t scale_offset =
            (static_cast<int64_t>(expert) * tokens_per_expert + token) *
                hidden_groups +
            hidden_group;
        output_scale[scale_offset] = scale_inv;
      }

      float4 out4 = {
          values[0] * scale,
          values[1] * scale,
          values[2] * scale,
          values[3] * scale,
      };
      const uint32_t packed =
          __musa_cvt_float4_to_fp8x4(out4, __MT_SATFINITE, __MT_E4M3);
      *reinterpret_cast<uint32_t*>(output + output_base) = packed;
    }
  }
}

template <int SUBWARPS_PER_BLOCK, int ACTIVATION_TYPE, int ROWS_PER_BLOCK>
__global__ void act_and_mul_masked_post_quant_scale_blocked_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output,
    float* __restrict__ output_scale,
    const int32_t* __restrict__ masked_m,
    int hidden_groups,
    int tokens_per_expert,
    float swiglu_limit,
    bool swizzle) {
  const int subwarp_id = threadIdx.x / kThreadsPerGroup;
  const int lane_id = threadIdx.x - subwarp_id * kThreadsPerGroup;
  const int hidden_group = blockIdx.x * SUBWARPS_PER_BLOCK + subwarp_id;
  const int token_base = blockIdx.y * ROWS_PER_BLOCK;
  const int expert = blockIdx.z;
  if (hidden_group >= hidden_groups || token_base >= masked_m[expert]) {
    return;
  }

  const int hidden = hidden_groups * kGroupSize;
  const int elem_offset = lane_id * kElemsPerThread;
  const int swizzled_chunk_offset =
      (elem_offset / 8) * 16 + (elem_offset % 8);
  const int valid_m = masked_m[expert];
  __shared__ float scale_buffer[SUBWARPS_PER_BLOCK][ROWS_PER_BLOCK];
#pragma unroll
  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const int token = token_base + r;
    if (token < tokens_per_expert && token < valid_m) {
      const int64_t input_base =
          (static_cast<int64_t>(expert) * tokens_per_expert + token) *
              hidden * 2 +
          (swizzle ? hidden_group * kGroupSize * 2 : hidden_group * kGroupSize);
      const int64_t gate_offset =
          input_base + (swizzle ? swizzled_chunk_offset : elem_offset);
      const int64_t up_offset =
          input_base + (swizzle ? swizzled_chunk_offset + 8
                                : hidden + elem_offset);
      const int64_t output_base =
          (static_cast<int64_t>(expert) * tokens_per_expert + token) * hidden +
          hidden_group * kGroupSize + elem_offset;

      uint64_t gate_u64 = *reinterpret_cast<const uint64_t*>(input + gate_offset);
      uint64_t up_u64 = *reinterpret_cast<const uint64_t*>(input + up_offset);
      auto gate_vec = reinterpret_cast<__mt_bfloat16*>(&gate_u64);
      auto up_vec = reinterpret_cast<__mt_bfloat16*>(&up_u64);

      float values[kElemsPerThread];
      float local_absmax = kLocalAbsMaxMin;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        const float gate_value =
            clamp_gate_musa(__bfloat162float(gate_vec[i]), swiglu_limit);
        const float up_value =
            clamp_up_musa(__bfloat162float(up_vec[i]), swiglu_limit);
        __mt_bfloat16 gate_lowprec = __float2bfloat16_rn(
            apply_activation_musa<ACTIVATION_TYPE>(gate_value));
        __mt_bfloat16 up_lowprec = __float2bfloat16_rn(up_value);
        __mt_bfloat16 val_lowprec = gate_lowprec * up_lowprec;
        const float val = __bfloat162float(val_lowprec);
        values[i] = val;
        local_absmax = fmaxf(local_absmax, fabsf(val));
      }

      local_absmax = GroupReduceMax<kThreadsPerGroup>(local_absmax, lane_id);
      const float inv_absmax = 1.0f / local_absmax;
      const float scale_inv = local_absmax * (1.0f / kFp8E4M3Max);
      const float scale = kFp8E4M3Max * inv_absmax;

      if (lane_id == 0) {
        scale_buffer[subwarp_id][r] = scale_inv;
      }

      float4 out4 = {
          values[0] * scale,
          values[1] * scale,
          values[2] * scale,
          values[3] * scale,
      };
      const uint32_t packed =
          __musa_cvt_float4_to_fp8x4(out4, __MT_SATFINITE, __MT_E4M3);
      *reinterpret_cast<uint32_t*>(output + output_base) = packed;
    }
  }

  __syncthreads();
  if constexpr (SUBWARPS_PER_BLOCK == 4) {
    const int hidden_group_base = static_cast<int>(blockIdx.x) * 4;
    if (hidden_group_base + 3 < hidden_groups) {
      for (int r = threadIdx.x; r < ROWS_PER_BLOCK; r += blockDim.x) {
        const int token = token_base + r;
        if (token < tokens_per_expert && token < valid_m) {
          const int64_t scale_offset =
              (static_cast<int64_t>(expert) * tokens_per_expert + token) *
                  hidden_groups +
              hidden_group_base;
          float4 scale_vec = {
              scale_buffer[0][r],
              scale_buffer[1][r],
              scale_buffer[2][r],
              scale_buffer[3][r],
          };
          *reinterpret_cast<float4*>(output_scale + scale_offset) = scale_vec;
        }
      }
    } else {
      for (int idx = threadIdx.x; idx < ROWS_PER_BLOCK * 4; idx += blockDim.x) {
        const int r = idx / 4;
        const int i = idx - r * 4;
        const int token = token_base + r;
        if (token < tokens_per_expert && token < valid_m &&
            hidden_group_base + i < hidden_groups) {
          const int64_t scale_offset =
              (static_cast<int64_t>(expert) * tokens_per_expert + token) *
                  hidden_groups +
              hidden_group_base + i;
          output_scale[scale_offset] = scale_buffer[i][r];
        }
      }
    }
  } else {
    for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
      const int token = token_base + r;
      if (token < tokens_per_expert && token < valid_m) {
        if (lane_id == 0) {
          const int64_t scale_offset =
              (static_cast<int64_t>(expert) * tokens_per_expert + token) *
                  hidden_groups +
              hidden_group;
          output_scale[scale_offset] = scale_buffer[subwarp_id][r];
        }
      }
    }
  }
}

template <int ACTIVATION_TYPE>
// Decode-specialized layout: one 512-thread CTA covers 16 quant groups for
// one expert. The z dimension partitions rows so small m does not serialize
// all tokens behind a single CTA.
__global__ void act_and_mul_masked_post_quant_compact_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output,
    float* __restrict__ output_scale,
    const int32_t* __restrict__ masked_m,
    int hidden_groups, int tokens_per_expert, float swiglu_limit, bool swizzle) {
  const int subwarp_id = threadIdx.x / kThreadsPerGroup;
  const int lane_id = threadIdx.x % kThreadsPerGroup;
  const int hidden_group = blockIdx.y * 16 + subwarp_id;
  const int expert = blockIdx.x;
  const int valid_m = min(masked_m[expert], tokens_per_expert);
  if (hidden_group >= hidden_groups) return;
  const int hidden = hidden_groups * kGroupSize;
  const int elem_offset = lane_id * kElemsPerThread;
  const int swizzled_chunk_offset = (elem_offset / 8) * 16 + (elem_offset % 8);
#pragma unroll 1
  for (int token = blockIdx.z; token < valid_m; token += gridDim.z) {
    const int64_t base =
        (static_cast<int64_t>(expert) * tokens_per_expert + token) * hidden * 2 +
        (swizzle ? hidden_group * kGroupSize * 2 : hidden_group * kGroupSize);
    const int64_t gate_offset =
        base + (swizzle ? swizzled_chunk_offset : elem_offset);
    const int64_t up_offset =
        base + (swizzle ? swizzled_chunk_offset + 8 : hidden + elem_offset);
    const int64_t out_offset =
        (static_cast<int64_t>(expert) * tokens_per_expert + token) * hidden +
        hidden_group * kGroupSize + elem_offset;
    uint64_t gate_u64 = *reinterpret_cast<const uint64_t*>(input + gate_offset);
    uint64_t up_u64 = *reinterpret_cast<const uint64_t*>(input + up_offset);
    auto gate_vec = reinterpret_cast<__mt_bfloat16*>(&gate_u64);
    auto up_vec = reinterpret_cast<__mt_bfloat16*>(&up_u64);
    float values[kElemsPerThread];
    float local_absmax = kLocalAbsMaxMin;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      const float gv =
          clamp_gate_musa(__bfloat162float(gate_vec[i]), swiglu_limit);
      const float uv =
          clamp_up_musa(__bfloat162float(up_vec[i]), swiglu_limit);
      __mt_bfloat16 g = __float2bfloat16_rn(
          apply_activation_musa<ACTIVATION_TYPE>(gv));
      __mt_bfloat16 u = __float2bfloat16_rn(uv);
      const float v = __bfloat162float(g * u);
      values[i] = v;
      local_absmax = fmaxf(local_absmax, fabsf(v));
    }
    local_absmax = GroupReduceMax<kThreadsPerGroup>(local_absmax, lane_id);
    const float scale_inv = local_absmax * (1.0f / kFp8E4M3Max);
    const float scale = kFp8E4M3Max / local_absmax;
    if (lane_id == 0) {
      output_scale[(static_cast<int64_t>(expert) * tokens_per_expert + token) *
                       hidden_groups +
                   hidden_group] = scale_inv;
    }
    float4 out4 = {
        values[0] * scale, values[1] * scale, values[2] * scale,
        values[3] * scale};
    *reinterpret_cast<uint32_t*>(output + out_offset) =
        __musa_cvt_float4_to_fp8x4(out4, __MT_SATFINITE, __MT_E4M3);
  }
}

template <int ACTIVATION_TYPE, int ROWS_PER_BLOCK>
void launch_act_and_mul_masked_post_quant(
    ffi::TensorView input,
    ffi::TensorView output,
    ffi::TensorView output_scale,
    ffi::TensorView masked_m,
    float swiglu_limit,
    bool swizzle,
    musaStream_t stream) {
  const int hidden = static_cast<int>(output.size(2));
  const int hidden_groups = hidden / kGroupSize;
  const int tokens_per_expert = static_cast<int>(output.size(1));
  const int experts = static_cast<int>(output.size(0));
  int subwarps_per_block = 1;
  if (tokens_per_expert >= 8 && tokens_per_expert < 128) {
    subwarps_per_block = 1;
  } else if (tokens_per_expert <= 128 && hidden_groups % 2 == 0) {
    subwarps_per_block = 2;
  } else if (hidden_groups % 8 == 0) {
    subwarps_per_block = 8;
  } else if (hidden_groups % 4 == 0) {
    subwarps_per_block = 4;
  } else if (hidden_groups % 2 == 0) {
    subwarps_per_block = 2;
  }

  const int row_tiles =
      (tokens_per_expert + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  dim3 grid(
      (hidden_groups + subwarps_per_block - 1) / subwarps_per_block,
      row_tiles,
      experts);
  dim3 block(subwarps_per_block * kThreadsPerGroup);

  if (subwarps_per_block == 8) {
    act_and_mul_masked_post_quant_kernel<8, ACTIVATION_TYPE, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(input.data_ptr()),
        static_cast<__mt_fp8_e4m3*>(output.data_ptr()),
        static_cast<float*>(output_scale.data_ptr()),
        static_cast<const int32_t*>(masked_m.data_ptr()),
        hidden_groups,
        tokens_per_expert,
        swiglu_limit,
        swizzle);
  } else if (subwarps_per_block == 4) {
    if (tokens_per_expert >= 4096) {
      act_and_mul_masked_post_quant_scale_blocked_kernel<4, ACTIVATION_TYPE, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(input.data_ptr()),
          static_cast<__mt_fp8_e4m3*>(output.data_ptr()),
          static_cast<float*>(output_scale.data_ptr()),
          static_cast<const int32_t*>(masked_m.data_ptr()),
          hidden_groups,
          tokens_per_expert,
          swiglu_limit,
          swizzle);
    } else {
      act_and_mul_masked_post_quant_kernel<4, ACTIVATION_TYPE, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(input.data_ptr()),
          static_cast<__mt_fp8_e4m3*>(output.data_ptr()),
          static_cast<float*>(output_scale.data_ptr()),
          static_cast<const int32_t*>(masked_m.data_ptr()),
          hidden_groups,
          tokens_per_expert,
          swiglu_limit,
          swizzle);
    }
  } else if (subwarps_per_block == 2) {
    act_and_mul_masked_post_quant_kernel<2, ACTIVATION_TYPE, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(input.data_ptr()),
        static_cast<__mt_fp8_e4m3*>(output.data_ptr()),
        static_cast<float*>(output_scale.data_ptr()),
        static_cast<const int32_t*>(masked_m.data_ptr()),
        hidden_groups,
        tokens_per_expert,
        swiglu_limit,
        swizzle);
  } else {
    act_and_mul_masked_post_quant_kernel<1, ACTIVATION_TYPE, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(input.data_ptr()),
        static_cast<__mt_fp8_e4m3*>(output.data_ptr()),
        static_cast<float*>(output_scale.data_ptr()),
        static_cast<const int32_t*>(masked_m.data_ptr()),
        hidden_groups,
        tokens_per_expert,
        swiglu_limit,
        swizzle);
  }
}

template <int ACTIVATION_TYPE>
void dispatch_act_and_mul_masked_post_quant_rows(
    ffi::TensorView input,
    ffi::TensorView output,
    ffi::TensorView output_scale,
    ffi::TensorView masked_m,
    float swiglu_limit,
    bool swizzle,
    int64_t expected_m,
    musaStream_t stream) {
  if (expected_m > 0 && expected_m <= 128 && output.size(2) % kGroupSize == 0) {
    const int hidden_groups = static_cast<int>(output.size(2) / kGroupSize);
    const int row_partitions = expected_m <= 32 ? 1 : (expected_m <= 64 ? 2 : 4);
    dim3 grid(static_cast<unsigned int>(output.size(0)),
              static_cast<unsigned int>((hidden_groups + 15) / 16),
              static_cast<unsigned int>(row_partitions));
    dim3 block(16 * kThreadsPerGroup);
    act_and_mul_masked_post_quant_compact_kernel<ACTIVATION_TYPE>
        <<<grid, block, 0, stream>>>(
            static_cast<const __mt_bfloat16*>(input.data_ptr()),
            static_cast<__mt_fp8_e4m3*>(output.data_ptr()),
            static_cast<float*>(output_scale.data_ptr()),
            static_cast<const int32_t*>(masked_m.data_ptr()), hidden_groups,
            static_cast<int>(output.size(1)), swiglu_limit, swizzle);
  } else if (output.size(1) <= 4096) {
    launch_act_and_mul_masked_post_quant<ACTIVATION_TYPE, 32>(
        input, output, output_scale, masked_m, swiglu_limit, swizzle, stream);
  } else {
    launch_act_and_mul_masked_post_quant<ACTIVATION_TYPE, kMaskedRowsPerBlock>(
        input, output, output_scale, masked_m, swiglu_limit, swizzle, stream);
  }
}

void sgl_musa_act_and_mul_masked_post_quant(
    ffi::TensorView input,
    ffi::TensorView output,
    ffi::TensorView output_scale,
    ffi::TensorView masked_m,
    int64_t activation_type,
    double swiglu_limit,
    bool swizzle,
    int64_t expected_m) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(output_scale);
  CHECK_INPUT(masked_m);
  TVM_FFI_ICHECK_EQ(input.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output_scale.ndim(), 3);
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), dl_bfloat16));
  TVM_FFI_ICHECK(dtype_equal(output.dtype(), dl_float8_e4m3fn));
  TVM_FFI_ICHECK(dtype_equal(output_scale.dtype(), dl_float32));
  TVM_FFI_ICHECK(dtype_equal(masked_m.dtype(), dl_int32));
  TVM_FFI_ICHECK_EQ(input.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(input.size(1), output.size(1));
  TVM_FFI_ICHECK_EQ(input.size(2), output.size(2) * 2);
  TVM_FFI_ICHECK_EQ(output.size(2) % kGroupSize, 0);
  TVM_FFI_ICHECK_EQ(output_scale.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(output_scale.size(1), output.size(1));
  TVM_FFI_ICHECK_EQ(output_scale.size(2), output.size(2) / kGroupSize);
  TVM_FFI_ICHECK_EQ(masked_m.size(0), output.size(0));

  ffi::MUSADeviceGuard device_guard(input.device().device_id);
  musaStream_t stream = get_stream(input.device());
  const int act = static_cast<int>(activation_type);
  const float swiglu_limit_f = static_cast<float>(swiglu_limit);
  if (act == kGeluActivation) {
    dispatch_act_and_mul_masked_post_quant_rows<kGeluActivation>(
        input, output, output_scale, masked_m, swiglu_limit_f, swizzle,
        expected_m, stream);
  } else if (act == kGeluTanhActivation) {
    dispatch_act_and_mul_masked_post_quant_rows<kGeluTanhActivation>(
        input, output, output_scale, masked_m, swiglu_limit_f, swizzle,
        expected_m, stream);
  } else {
    dispatch_act_and_mul_masked_post_quant_rows<kSiluActivation>(
        input, output, output_scale, masked_m, swiglu_limit_f, swizzle,
        expected_m, stream);
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA act_and_mul_masked_post_quant kernel failed: "
      << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_act_and_mul_masked_post_quant,
    sgl_musa_act_and_mul_masked_post_quant);
