#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>

#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#elif defined(USE_MUSA) || defined(__MUSACC__)
#include <musa_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace host::runtime {

// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
#if defined(USE_ROCM)
  RuntimeDeviceCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
#elif defined(USE_MUSA) || defined(__MUSACC__)
  RuntimeDeviceCheck(
      musaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
#else
  RuntimeDeviceCheck(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
#endif
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of SMs for the given device
inline auto get_sm_count(int device_id) -> uint32_t {
  int sm_count;
#if defined(USE_ROCM)
  RuntimeDeviceCheck(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, device_id));
#elif defined(USE_MUSA) || defined(__MUSACC__)
  RuntimeDeviceCheck(musaDeviceGetAttribute(&sm_count, musaDevAttrMultiProcessorCount, device_id));
#else
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
#endif
  return static_cast<uint32_t>(sm_count);
}

// Return the Major compute capability for the given device
inline auto get_cc_major(int device_id) -> int {
  int cc_major;
#if defined(USE_ROCM)
  RuntimeDeviceCheck(hipDeviceGetAttribute(&cc_major, hipDeviceAttributeComputeCapabilityMajor, device_id));
#elif defined(USE_MUSA) || defined(__MUSACC__)
  RuntimeDeviceCheck(musaDeviceGetAttribute(&cc_major, musaDevAttrComputeCapabilityMajor, device_id));
#else
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id));
#endif
  return cc_major;
}

// Return the runtime version
inline auto get_runtime_version() -> int {
  int runtime_version;
#if defined(USE_ROCM)
  RuntimeDeviceCheck(hipRuntimeGetVersion(&runtime_version));
#elif defined(USE_MUSA) || defined(__MUSACC__)
  RuntimeDeviceCheck(musaRuntimeGetVersion(&runtime_version));
#else
  RuntimeDeviceCheck(cudaRuntimeGetVersion(&runtime_version));
#endif
  return runtime_version;
}

// Return the maximum dynamic shared memory per block for the given kernel
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
#if defined(USE_ROCM)
  RuntimeDeviceCheck(hipOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
#elif defined(USE_MUSA) || defined(__MUSACC__)
  RuntimeDeviceCheck(musaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
#else
  RuntimeDeviceCheck(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
#endif
  return smem_size;
}

}  // namespace host::runtime
