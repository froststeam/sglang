#include "../common.h"
#include "../device_utils.h"

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

#ifndef SGL_CUSTOM_RS_THREADS
#define SGL_CUSTOM_RS_THREADS 512
#endif

#ifndef SGL_CUSTOM_RS_BLOCKS
#define SGL_CUSTOM_RS_BLOCKS 56
#endif

#ifndef SGL_CUSTOM_RS_MAX_BLOCKS
#define SGL_CUSTOM_RS_MAX_BLOCKS 120
#endif

#ifndef SGL_CUSTOM_RS_DYNAMIC_BLOCKS
#define SGL_CUSTOM_RS_DYNAMIC_BLOCKS 1
#endif

constexpr int kMaxBlocks = SGL_CUSTOM_RS_MAX_BLOCKS;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kDefaultThreads = SGL_CUSTOM_RS_THREADS;
constexpr int kDefaultBlockLimit = SGL_CUSTOM_RS_BLOCKS;
constexpr int kMaxRanks = 8;
using FlagType = uint32_t;

struct alignas(128) Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks];
  alignas(128) FlagType push_epoch[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

__device__ __forceinline__ float upcast_s(half value) {
  return __half2float(value);
}

__device__ __forceinline__ float upcast_s(__mt_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float upcast_s(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T downcast_s(float value);

template <>
__device__ __forceinline__ half downcast_s<half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16 downcast_s<__mt_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ float downcast_s<float>(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T& assign_add(T& a, T b) {
  a = downcast_s<T>(upcast_s(a) + upcast_s(b));
  return a;
}

template <>
__device__ __forceinline__ float& assign_add<float>(float& a, float b) {
  a += b;
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<float, N> upcast(array_t<T, N> value) {
  if constexpr (std::is_same<T, float>::value) {
    return value;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out.data[i] = upcast_s(value.data[i]);
    }
    return out;
  }
}

template <typename O>
__device__ __forceinline__ O downcast(array_t<float, O::size> value) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return value;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; ++i) {
      out.data[i] = downcast_s<typename O::type>(value.data[i]);
    }
    return out;
  }
}

__device__ __forceinline__ void signal_store(FlagType* ptr, FlagType value) {
  volatile_store(static_cast<uint32_t>(value), reinterpret_cast<uint32_t*>(ptr));
}

__device__ __forceinline__ FlagType signal_load(FlagType* ptr) {
  flushInv_byp();
  return static_cast<uint32_t>(volatile_load(reinterpret_cast<uint32_t*>(ptr)));
}

template <int nranks, bool start>
__device__ __forceinline__ void multi_rank_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!start) {
    __syncthreads_lm();
  }
  if (threadIdx.x < nranks) {
    auto flag = atomicAdd(&self_sg->self_counter[blockIdx.x][threadIdx.x], 1) + 1;
    auto* peer = &sg.signals[threadIdx.x]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][threadIdx.x];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
  }
  if constexpr (start) {
    __syncthreads_lm();
  }
}

template <typename P, int nranks, typename A>
__device__ __forceinline__ P packed_reduce_scatter(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < nranks; ++i) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename P, int nranks>
__device__ __forceinline__ P packed_reduce_scatter_staged_lowp(const P* ptrs[], int idx) {
  P tmp = ptrs[0][idx];
  if constexpr (nranks == 8) {
    packed_assign_add(tmp, ptrs[1][idx]);
    packed_assign_add(tmp, ptrs[3][idx]);
    packed_assign_add(tmp, ptrs[2][idx]);
    packed_assign_add(tmp, ptrs[4][idx]);
    packed_assign_add(tmp, ptrs[5][idx]);
    packed_assign_add(tmp, ptrs[6][idx]);
    packed_assign_add(tmp, ptrs[7][idx]);
  } else {
#pragma unroll
    for (int i = 1; i < nranks; ++i) {
      packed_assign_add(tmp, ptrs[i][idx]);
    }
  }
  return tmp;
}

template <typename P, int nranks>
__device__ __forceinline__ P packed_reduce_scatter_staged_lowp_rank_aware(const P* ptrs[], int idx, int rank) {
  P tmp = ptrs[0][idx];
  if constexpr (nranks == 8) {
    packed_assign_add(tmp, ptrs[1][idx]);
    if (rank == 3) {
      packed_assign_add(tmp, ptrs[3][idx]);
      packed_assign_add(tmp, ptrs[4][idx]);
      packed_assign_add(tmp, ptrs[5][idx]);
      packed_assign_add(tmp, ptrs[6][idx]);
      packed_assign_add(tmp, ptrs[2][idx]);
      packed_assign_add(tmp, ptrs[7][idx]);
    } else if (rank == 6) {
      packed_assign_add(tmp, ptrs[3][idx]);
      packed_assign_add(tmp, ptrs[2][idx]);
      packed_assign_add(tmp, ptrs[4][idx]);
      packed_assign_add(tmp, ptrs[5][idx]);
      packed_assign_add(tmp, ptrs[6][idx]);
      packed_assign_add(tmp, ptrs[7][idx]);
    } else if (rank == 7) {
      packed_assign_add(tmp, ptrs[2][idx]);
      packed_assign_add(tmp, ptrs[3][idx]);
      packed_assign_add(tmp, ptrs[4][idx]);
      packed_assign_add(tmp, ptrs[5][idx]);
      packed_assign_add(tmp, ptrs[6][idx]);
      packed_assign_add(tmp, ptrs[7][idx]);
    } else {
      packed_assign_add(tmp, ptrs[2][idx]);
      packed_assign_add(tmp, ptrs[3][idx]);
      packed_assign_add(tmp, ptrs[4][idx]);
      packed_assign_add(tmp, ptrs[5][idx]);
      packed_assign_add(tmp, ptrs[6][idx]);
      packed_assign_add(tmp, ptrs[7][idx]);
    }
  } else {
#pragma unroll
    for (int i = 1; i < nranks; ++i) {
      packed_assign_add(tmp, ptrs[i][idx]);
    }
  }
  return tmp;
}

template <typename P, int nranks>
__device__ __forceinline__ P packed_reduce_scatter_staged_lowp_rank_rotated(const P* ptrs[], int idx, int rank) {
  int peer = (rank + 1) % nranks;
  P tmp = ptrs[peer][idx];
#pragma unroll
  for (int step = 1; step < nranks; ++step) {
    peer = (rank + 1 + step) % nranks;
    packed_assign_add(tmp, ptrs[peer][idx]);
  }
  return tmp;
}

template <typename P, int nranks>
__device__ __forceinline__ P packed_reduce_scatter_staged_lowp_rank_chunked(const P* ptrs[], int idx, int rank) {
  if constexpr (nranks == 8) {
    constexpr int orders[8][3][8] = {
        {{1, 2, 3, 4, 5, 6, 7, 0}, {2, 4, 6, 1, 3, 7, 5, 0}, {3, 5, 7, 1, 4, 2, 6, 0}},
        {{2, 3, 4, 5, 6, 7, 0, 1}, {3, 7, 5, 0, 2, 4, 6, 1}, {2, 4, 6, 0, 3, 5, 7, 1}},
        {{3, 4, 5, 6, 7, 0, 1, 2}, {4, 6, 1, 3, 7, 5, 0, 2}, {0, 6, 3, 5, 7, 1, 4, 2}},
        {{4, 5, 6, 7, 0, 1, 2, 3}, {5, 7, 0, 2, 4, 6, 1, 3}, {5, 7, 1, 4, 2, 6, 0, 3}},
        {{5, 6, 7, 0, 1, 2, 3, 4}, {1, 6, 3, 7, 5, 0, 2, 4}, {2, 6, 0, 3, 5, 7, 1, 4}},
        {{6, 7, 0, 1, 2, 3, 4, 5}, {0, 2, 4, 6, 1, 3, 7, 5}, {1, 7, 4, 2, 6, 0, 3, 5}},
        {{0, 7, 1, 2, 3, 4, 5, 6}, {1, 3, 7, 5, 0, 2, 4, 6}, {0, 3, 5, 7, 1, 4, 2, 6}},
        {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 5, 2, 4, 6, 1, 3, 7}, {1, 4, 2, 6, 0, 3, 5, 7}},
    };
    int chunk = (idx * P::size) / 2048;
    chunk = chunk > 2 ? 2 : chunk;
    P tmp = ptrs[orders[rank][chunk][0]][idx];
#pragma unroll
    for (int step = 1; step < 8; ++step) {
      packed_assign_add(tmp, ptrs[orders[rank][chunk][step]][idx]);
    }
    return tmp;
  } else {
    return packed_reduce_scatter_staged_lowp_rank_rotated<P, nranks>(ptrs, idx, rank);
  }
}

template <typename P, int nranks, typename A>
__device__ __forceinline__ P packed_reduce_scatter_mccl_compatible(const P* ptrs[], int idx, int rank, int mode) {
  if (mode == 4) {
    return packed_reduce_scatter_staged_lowp_rank_chunked<P, nranks>(ptrs, idx, rank);
  }
  if (mode == 3) {
    return packed_reduce_scatter_staged_lowp_rank_rotated<P, nranks>(ptrs, idx, rank);
  }
  if constexpr (std::is_same<typename P::type, float>::value) {
    return packed_reduce_scatter<P, nranks, A>(ptrs, idx);
  } else {
    if (mode == 2) {
      return packed_reduce_scatter_staged_lowp_rank_aware<P, nranks>(ptrs, idx, rank);
    }
    return packed_reduce_scatter_staged_lowp<P, nranks>(ptrs, idx);
  }
}

template <typename T, int nranks, bool mccl_compatible>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_scatter(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ out,
    int rank,
    int shard_size,
    int mode) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const int pack = packed_t<T>::P::size;
  const int packed_shard_size = shard_size / pack;
  const P* ptrs[nranks];
#pragma unroll
  for (int peer = 0; peer < nranks; ++peer) {
    ptrs[peer] = reinterpret_cast<const P*>(data.ptrs[peer]) + rank * packed_shard_size;
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packed_shard_size; idx += gridDim.x * blockDim.x) {
    if constexpr (mccl_compatible) {
      reinterpret_cast<P*>(out)[idx] = packed_reduce_scatter_mccl_compatible<P, nranks, A>(ptrs, idx, rank, mode);
    } else {
      reinterpret_cast<P*>(out)[idx] = packed_reduce_scatter<P, nranks, A>(ptrs, idx);
    }
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
int select_block_limit(int packed_shard_size) {
  int limit = kDefaultBlockLimit;
#if SGL_CUSTOM_RS_DYNAMIC_BLOCKS
  if (!std::is_same<T, float>::value) {
    if (packed_shard_size <= 128 * 1024) {
      limit = std::min(limit, 60);
    }
  }
#endif
  return std::min(limit, kMaxBlocks);
}

template <typename T, int nranks, bool mccl_compatible>
void launch_rs(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    T* out,
    int rank,
    int shard_size,
    int mode,
    musaStream_t stream) {
  const int pack = packed_t<T>::P::size;
  TVM_FFI_ICHECK_EQ(shard_size % pack, 0);
  const int packed_shard_size = shard_size / pack;
  const int block_limit = select_block_limit<T, nranks>(packed_shard_size);
  const int blocks = std::min(block_limit, (packed_shard_size + kDefaultThreads - 1) / kDefaultThreads);
  if (blocks <= 0) {
    return;
  }
  cross_device_reduce_scatter<T, nranks, mccl_compatible><<<blocks, kDefaultThreads, 0, stream>>>(
      data, sg, self_sg, out, rank, shard_size, mode);
}

template <typename T>
void dispatch_world_size(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    T* out,
    int rank,
    int world_size,
    int shard_size,
    bool mccl_compatible,
    int mode,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      if (mccl_compatible) {
        launch_rs<T, 2, true>(data, sg, self_sg, out, rank, shard_size, mode, stream);
      } else {
        launch_rs<T, 2, false>(data, sg, self_sg, out, rank, shard_size, mode, stream);
      }
      break;
    case 4:
      if (mccl_compatible) {
        launch_rs<T, 4, true>(data, sg, self_sg, out, rank, shard_size, mode, stream);
      } else {
        launch_rs<T, 4, false>(data, sg, self_sg, out, rank, shard_size, mode, stream);
      }
      break;
    case 6:
      if (mccl_compatible) {
        launch_rs<T, 6, true>(data, sg, self_sg, out, rank, shard_size, mode, stream);
      } else {
        launch_rs<T, 6, false>(data, sg, self_sg, out, rank, shard_size, mode, stream);
      }
      break;
    case 8:
      if (mccl_compatible) {
        launch_rs<T, 8, true>(data, sg, self_sg, out, rank, shard_size, mode, stream);
      } else {
        launch_rs<T, 8, false>(data, sg, self_sg, out, rank, shard_size, mode, stream);
      }
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

}  // namespace

int64_t sgl_musa_custom_rs_meta_size() {
  return static_cast<int64_t>(sizeof(Signal));
}

void sgl_musa_custom_rs_launch_unregistered_impl(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size,
    bool mccl_compatible,
    int mode) {
  CHECK_MUSA_CONTIGUOUS(inp);
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
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), out.dtype()));
  TVM_FFI_ICHECK_EQ(tensor_numel(inp), tensor_numel(out) * world_size);

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }
  RankData data{};
  const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(out.device());
  const int64_t shard_numel64 = tensor_numel(out);
  TVM_FFI_ICHECK_LE(shard_numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t elem_bytes = (static_cast<int64_t>(out.dtype().bits) * out.dtype().lanes + 7) / 8;
  const int64_t input_bytes = tensor_numel(inp) * elem_bytes;
  TVM_FFI_ICHECK_LE(input_bytes, max_size_bytes);
  const musaError_t copy_err = musaMemcpyAsync(
      reinterpret_cast<void*>(self_buffer_ptr),
      inp.data_ptr(),
      static_cast<size_t>(input_bytes),
      musaMemcpyDeviceToDevice,
      stream);
  TVM_FFI_ICHECK_EQ(copy_err, musaSuccess) << "MUSA custom RS copy failed: " << musaGetErrorString(copy_err);
  const int shard_size = static_cast<int>(shard_numel64);

  if (dtype_equal(out.dtype(), dl_float16)) {
    dispatch_world_size(data, sg, self_sg, static_cast<half*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), shard_size, mccl_compatible, mode, stream);
  } else if (dtype_equal(out.dtype(), dl_bfloat16)) {
    dispatch_world_size(data, sg, self_sg, static_cast<__mt_bfloat16*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), shard_size, mccl_compatible, mode, stream);
  } else if (dtype_equal(out.dtype(), dl_float32)) {
    dispatch_world_size(data, sg, self_sg, static_cast<float*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), shard_size, mccl_compatible, mode, stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom rs only supports fp16/bf16/fp32";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA custom RS kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_custom_rs_launch_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size) {
  sgl_musa_custom_rs_launch_unregistered_impl(
      rank_data, signal_ptrs_cpu, inp, out, self_signal_ptr, self_buffer_ptr, max_size_bytes, rank, world_size, false, 0);
}

void sgl_musa_custom_rs_launch_unregistered_mccl_compatible(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size) {
  sgl_musa_custom_rs_launch_unregistered_impl(
      rank_data, signal_ptrs_cpu, inp, out, self_signal_ptr, self_buffer_ptr, max_size_bytes, rank, world_size, true, 1);
}

void sgl_musa_custom_rs_launch_unregistered_empirical(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size) {
  sgl_musa_custom_rs_launch_unregistered_impl(
      rank_data, signal_ptrs_cpu, inp, out, self_signal_ptr, self_buffer_ptr, max_size_bytes, rank, world_size, true, 2);
}

void sgl_musa_custom_rs_launch_unregistered_rotated(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size) {
  sgl_musa_custom_rs_launch_unregistered_impl(
      rank_data, signal_ptrs_cpu, inp, out, self_signal_ptr, self_buffer_ptr, max_size_bytes, rank, world_size, true, 3);
}

void sgl_musa_custom_rs_launch_unregistered_chunked(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size) {
  sgl_musa_custom_rs_launch_unregistered_impl(
      rank_data, signal_ptrs_cpu, inp, out, self_signal_ptr, self_buffer_ptr, max_size_bytes, rank, world_size, true, 4);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_rs_meta_size, sgl_musa_custom_rs_meta_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_rs_launch_unregistered, sgl_musa_custom_rs_launch_unregistered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_rs_launch_unregistered_mccl_compatible,
    sgl_musa_custom_rs_launch_unregistered_mccl_compatible);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_rs_launch_unregistered_empirical,
    sgl_musa_custom_rs_launch_unregistered_empirical);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_rs_launch_unregistered_rotated,
    sgl_musa_custom_rs_launch_unregistered_rotated);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_rs_launch_unregistered_chunked,
    sgl_musa_custom_rs_launch_unregistered_chunked);
