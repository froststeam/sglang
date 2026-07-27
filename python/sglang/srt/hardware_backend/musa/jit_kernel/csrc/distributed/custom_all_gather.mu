#include "../common.h"
#include "../device_utils.h"

#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

#ifndef SGL_CUSTOM_AG_THREADS
#define SGL_CUSTOM_AG_THREADS 512
#endif

#ifndef SGL_CUSTOM_AG_BLOCKS
#define SGL_CUSTOM_AG_BLOCKS 48
#endif

#ifndef SGL_CUSTOM_AG_MAX_BLOCKS
#define SGL_CUSTOM_AG_MAX_BLOCKS 120
#endif

#ifndef SGL_CUSTOM_AG_ATOMIC_BARRIER
#define SGL_CUSTOM_AG_ATOMIC_BARRIER 1
#endif

#ifndef SGL_CUSTOM_AG_DYNAMIC_BLOCKS
#define SGL_CUSTOM_AG_DYNAMIC_BLOCKS 1
#endif

#ifndef SGL_CUSTOM_AG_PARALLEL_SRC_MIN_BYTES
#define SGL_CUSTOM_AG_PARALLEL_SRC_MIN_BYTES 2097152
#endif

constexpr int kMaxBlocks = SGL_CUSTOM_AG_MAX_BLOCKS;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kDefaultThreads = SGL_CUSTOM_AG_THREADS;
constexpr int kDefaultBlockLimit = SGL_CUSTOM_AG_BLOCKS;
constexpr int64_t kParallelSrcMinBytes = SGL_CUSTOM_AG_PARALLEL_SRC_MIN_BYTES;
constexpr int64_t kMiB = 1024 * 1024;
constexpr int kMaxRanks = 8;
using FlagType = uint32_t;

struct alignas(128) Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks];
};

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

__device__ __forceinline__ void signal_store(FlagType* ptr, FlagType value) {
  volatile_store(static_cast<uint32_t>(value), reinterpret_cast<uint32_t*>(ptr));
}

__device__ __forceinline__ FlagType signal_load(FlagType* ptr) {
  flushInv_byp();
  return static_cast<uint32_t>(volatile_load(reinterpret_cast<uint32_t*>(ptr)));
}

template <int nranks, bool fence = false>
__device__ __forceinline__ void multi_rank_barrier(
    const RankSignals& sg,
    Signal* self_sg,
    int rank) {
  __syncthreads_lm();
  if (threadIdx.x < nranks) {
#if SGL_CUSTOM_AG_ATOMIC_BARRIER
    auto flag = atomicAdd(&self_sg->self_counter[blockIdx.x][threadIdx.x], 1) + 1;
    auto* peer = &sg.signals[threadIdx.x]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][threadIdx.x];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
#else
    auto flag = self_sg->self_counter[blockIdx.x][threadIdx.x] + 1;
    self_sg->self_counter[blockIdx.x][threadIdx.x] = flag;
    auto* peer = &sg.signals[threadIdx.x]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][threadIdx.x];
    signal_store(peer, flag);
    while (signal_load(local) != flag) {
    }
#endif
  }
  if constexpr (fence) {
    __syncthreads_lm();
  }
}

template <int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) custom_all_gather_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    void* __restrict__ out,
    int64_t input_packets,
    int rank) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  auto* output = reinterpret_cast<uint4*>(out);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  flushInv_byp();
#pragma unroll
  for (int step = 0; step < nranks; ++step) {
    const int src_rank = (rank + step) % nranks;
    const auto* src = reinterpret_cast<const uint4*>(data.ptrs[src_rank]);
    auto* dst = output + src_rank * input_packets;
    for (int64_t idx = tid; idx < input_packets; idx += stride) {
      dst[idx] = src[idx];
    }
  }
  multi_rank_barrier<nranks>(sg, self_sg, rank);
}

template <int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_gather_parallel_src_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    void* __restrict__ out,
    int64_t input_packets,
    int blocks_per_rank,
    int rank) {
  const int local_block = blockIdx.x / nranks;
  const int src_slot = blockIdx.x - local_block * nranks;
  const int src_rank = (rank + src_slot) % nranks;
  const int tid = local_block * blockDim.x + threadIdx.x;
  const int stride = blocks_per_rank * blockDim.x;
  const auto* src = reinterpret_cast<const uint4*>(data.ptrs[src_rank]);
  auto* dst = reinterpret_cast<uint4*>(out) + src_rank * input_packets;

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  flushInv_byp();
  for (int64_t idx = tid; idx < input_packets; idx += stride) {
    dst[idx] = src[idx];
  }
  multi_rank_barrier<nranks>(sg, self_sg, rank);
}

template <int nranks>
int64_t select_block_limit(int64_t input_nbytes) {
  int64_t block_limit = std::min<int64_t>(kMaxBlocks, kDefaultBlockLimit);
#if SGL_CUSTOM_AG_DYNAMIC_BLOCKS
  if constexpr (nranks == 4) {
    block_limit = std::min<int64_t>(
        block_limit, input_nbytes >= 16 * kMiB ? 80 : 48);
  } else if constexpr (nranks == 8) {
    block_limit = std::min<int64_t>(
        block_limit, input_nbytes >= 64 * kMiB ? 64 : 48);
  }
#endif
  return block_limit;
}

template <int nranks>
void launch_ag(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    void* out,
    int64_t input_nbytes,
    int rank,
    musaStream_t stream) {
  TVM_FFI_ICHECK_EQ(input_nbytes % 16, 0);
  const int64_t packets = input_nbytes / 16;
  const int64_t needed_blocks =
      (packets + kDefaultThreads - 1) / kDefaultThreads;
  const int64_t block_limit = select_block_limit<nranks>(input_nbytes);
  const bool use_parallel_src =
      nranks > 2 &&
      input_nbytes >= kParallelSrcMinBytes &&
      block_limit >= nranks;
  const int64_t blocks_per_rank_limit =
      use_parallel_src ? std::max<int64_t>(1, block_limit / nranks) : block_limit;
  const int blocks = static_cast<int>(
      std::min<int64_t>(blocks_per_rank_limit, needed_blocks));
  if (blocks <= 0) {
    return;
  }
  if (use_parallel_src) {
    custom_all_gather_parallel_src_kernel<nranks>
        <<<blocks * nranks, kDefaultThreads, 0, stream>>>(
            data, sg, self_sg, out, packets, blocks, rank);
  } else {
    custom_all_gather_kernel<nranks><<<blocks, kDefaultThreads, 0, stream>>>(
        data, sg, self_sg, out, packets, rank);
  }
}

void dispatch_world_size(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    void* out,
    int64_t input_nbytes,
    int rank,
    int world_size,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_ag<2>(data, sg, self_sg, out, input_nbytes, rank, stream);
      break;
    case 4:
      launch_ag<4>(data, sg, self_sg, out, input_nbytes, rank, stream);
      break;
    case 6:
      launch_ag<6>(data, sg, self_sg, out, input_nbytes, rank, stream);
      break;
    case 8:
      launch_ag<8>(data, sg, self_sg, out, input_nbytes, rank, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

RankSignals make_rank_signals(ffi::TensorView signal_ptrs_cpu, int64_t world_size) {
  RankSignals sg{};
  const auto* ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(ptrs[i]);
  }
  return sg;
}

RankData make_rank_data(ffi::TensorView rank_data) {
  RankData data{};
  const auto* ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(ptrs[i]);
  }
  return data;
}

void check_common_args(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t input_nbytes,
    int64_t rank,
    int64_t world_size) {
  CHECK_MUSA_CONTIGUOUS(out);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK(self_signal_ptr != 0);
  TVM_FFI_ICHECK(input_nbytes > 0);
  TVM_FFI_ICHECK_EQ(input_nbytes % 16, 0);
  const int64_t out_nbytes =
      tensor_numel(out) *
      ((static_cast<int64_t>(out.dtype().bits) * out.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_EQ(
      out_nbytes, input_nbytes * world_size);
}

}  // namespace

int64_t sgl_musa_custom_ag_meta_size() {
  return static_cast<int64_t>(sizeof(Signal));
}

void sgl_musa_custom_ag_launch(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t input_nbytes,
    int64_t rank,
    int64_t world_size) {
  // Registered-input launch path. This is currently not used by the normal
  // eager custom all-gather path; it is kept for explicit benchmark/test usage
  // and graph capture when input buffers have been registered ahead of time.
  check_common_args(
      rank_data, signal_ptrs_cpu, out, self_signal_ptr, input_nbytes, rank,
      world_size);
  auto sg = make_rank_signals(signal_ptrs_cpu, world_size);
  auto data = make_rank_data(rank_data);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(out.device());
  dispatch_world_size(
      data, sg, self_sg, out.data_ptr(), input_nbytes, static_cast<int>(rank),
      static_cast<int>(world_size), stream);
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AG kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_custom_ag_launch_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size) {
  TVM_FFI_ICHECK_EQ(inp.device().device_type, kDLExtDev);
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), out.dtype()));
  const int64_t elem_bytes =
      (static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8;
  const int64_t input_nbytes = tensor_numel(inp) * elem_bytes;
  TVM_FFI_ICHECK_LE(input_nbytes, max_size_bytes);
  TVM_FFI_ICHECK(self_buffer_ptr != 0);
  check_common_args(
      rank_data, signal_ptrs_cpu, out, self_signal_ptr, input_nbytes, rank,
      world_size);

  auto stream = get_stream(out.device());
  const musaError_t copy_err = musaMemcpyAsync(
      reinterpret_cast<void*>(self_buffer_ptr),
      inp.data_ptr(),
      static_cast<size_t>(input_nbytes),
      musaMemcpyDeviceToDevice,
      stream);
  TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
      << "MUSA custom AG copy failed: " << musaGetErrorString(copy_err);

  auto sg = make_rank_signals(signal_ptrs_cpu, world_size);
  auto data = make_rank_data(rank_data);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  dispatch_world_size(
      data, sg, self_sg, out.data_ptr(), input_nbytes, static_cast<int>(rank),
      static_cast<int>(world_size), stream);
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AG kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ag_meta_size, sgl_musa_custom_ag_meta_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ag_launch, sgl_musa_custom_ag_launch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ag_launch_unregistered,
    sgl_musa_custom_ag_launch_unregistered);
