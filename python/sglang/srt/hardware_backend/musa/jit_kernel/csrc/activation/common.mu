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
