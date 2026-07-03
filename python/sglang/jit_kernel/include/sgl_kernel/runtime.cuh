#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <musa_runtime.h>

namespace host::runtime {

// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  RuntimeDeviceCheck(
      musaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of SMs for the given device
inline auto get_sm_count(int device_id) -> uint32_t {
  int sm_count;
  RuntimeDeviceCheck(musaDeviceGetAttribute(&sm_count, musaDevAttrMultiProcessorCount, device_id));
  return static_cast<uint32_t>(sm_count);
}

// Return the Major compute capability for the given device
inline auto get_cc_major(int device_id) -> int {
  int cc_major;
  RuntimeDeviceCheck(musaDeviceGetAttribute(&cc_major, musaDevAttrComputeCapabilityMajor, device_id));
  return cc_major;
}

// Return the runtime version
inline auto get_runtime_version() -> int {
  int runtime_version;
  RuntimeDeviceCheck(musaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

// Return the maximum dynamic shared memory per block for the given kernel
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  RuntimeDeviceCheck(musaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
  return smem_size;
}

}  // namespace host::runtime
