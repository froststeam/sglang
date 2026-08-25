#include "../common.h"
#include "../device_utils.h"

#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kMaxRanks = 8;
constexpr int kMaxBlocks = 120;
constexpr int kThreads = 512;
constexpr int kBlockLimit = 120;
constexpr int kQwenGlobalHeads = 24;
constexpr int kQwenHeadDim = 128;
constexpr int kQwenHeadDimPackets = kQwenHeadDim / 8;
using FlagType = uint32_t;

struct alignas(128) Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks];
};

struct __align__(16) RankData {
  void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

__device__ __forceinline__ void signal_store(FlagType* ptr, FlagType value) {
  volatile_store(static_cast<uint32_t>(value),
                 reinterpret_cast<uint32_t*>(ptr));
}

__device__ __forceinline__ FlagType signal_load(FlagType* ptr) {
  flushInv_byp();
  return static_cast<uint32_t>(
      volatile_load(reinterpret_cast<uint32_t*>(ptr)));
}

template <int nranks>
__device__ __forceinline__ void multi_rank_barrier(
    const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads_lm();
  if (threadIdx.x < nranks) {
    auto flag =
        atomicAdd(&self_sg->self_counter[blockIdx.x][threadIdx.x], 1) + 1;
    auto* peer =
        &sg.signals[threadIdx.x]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local =
        &self_sg->peer_counter[flag & 1][blockIdx.x][threadIdx.x];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
  }
  __syncthreads_lm();
}

template <int nranks>
__global__ void __launch_bounds__(kThreads, 1) custom_all_to_all_kernel(
    RankData outputs,
    RankSignals sg,
    Signal* self_sg,
    const uint4* __restrict__ input,
    int64_t packets,
    int64_t slot_stride_packets,
    int slot,
    int rank) {
  // A slot is reused cyclically. Wait until every rank has reached this
  // collective before allowing a peer to overwrite that slot.
  multi_rank_barrier<nranks>(sg, self_sg, rank);
  const int64_t chunk_packets = packets / nranks;
  const int dst_rank = blockIdx.x % nranks;
  const int rank_block = blockIdx.x / nranks;
  const int rank_blocks = gridDim.x / nranks;
  const int64_t tid =
      static_cast<int64_t>(rank_block) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(rank_blocks) * blockDim.x;
  const int64_t output_offset =
      static_cast<int64_t>(slot) * slot_stride_packets +
      static_cast<int64_t>(rank) * chunk_packets;
  auto* dst = reinterpret_cast<uint4*>(outputs.ptrs[dst_rank]) +
              output_offset;

  for (int64_t chunk_idx = tid; chunk_idx < chunk_packets;
       chunk_idx += stride) {
    dst[chunk_idx] = input[dst_rank * chunk_packets + chunk_idx];
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks>(sg, self_sg, rank);
}

template <int nranks, bool input_layout>
__global__ void __launch_bounds__(kThreads, 1) custom_ulysses_kernel(
    RankData outputs,
    RankSignals sg,
    Signal* self_sg,
    const uint4* __restrict__ input,
    int64_t slot_stride_packets,
    int64_t slot,
    int64_t local_sequence,
    int rank) {
  multi_rank_barrier<nranks>(sg, self_sg, rank);
  constexpr int local_heads = kQwenGlobalHeads / nranks;
  const int64_t rows = local_sequence * local_heads;
  const int dst_rank = blockIdx.x % nranks;
  const int rank_block = blockIdx.x / nranks;
  const int rank_blocks = gridDim.x / nranks;
  const int64_t tid =
      static_cast<int64_t>(rank_block) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(rank_blocks) * blockDim.x;
  const int64_t packet = tid & (kQwenHeadDimPackets - 1);
  const int64_t row_stride = stride / kQwenHeadDimPackets;
  const int64_t slot_offset = slot * slot_stride_packets;
  auto* output = reinterpret_cast<uint4*>(outputs.ptrs[dst_rank]) +
                 slot_offset;

  for (int64_t row = tid / kQwenHeadDimPackets; row < rows;
       row += row_stride) {
    const int64_t sequence = row / local_heads;
    const int64_t head = row - sequence * local_heads;

    int64_t dst;
    if constexpr (input_layout) {
      dst =
          ((rank * local_sequence + sequence) * local_heads + head) *
              kQwenHeadDimPackets +
          packet;
    } else {
      dst =
          (sequence * kQwenGlobalHeads + rank * local_heads + head) *
              kQwenHeadDimPackets +
          packet;
    }
    int64_t src;
    if constexpr (input_layout) {
      src =
          (sequence * kQwenGlobalHeads + dst_rank * local_heads + head) *
              kQwenHeadDimPackets +
          packet;
    } else {
      src =
          ((dst_rank * local_sequence + sequence) * local_heads + head) *
              kQwenHeadDimPackets +
          packet;
    }
    output[dst] = input[src];
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks>(sg, self_sg, rank);
}

template <int nranks>
__global__ void __launch_bounds__(kThreads, 1) custom_qkv_ulysses_kernel(
    RankData outputs,
    RankSignals sg,
    Signal* self_sg,
    const uint4* __restrict__ query,
    const uint4* __restrict__ key,
    const uint4* __restrict__ value,
    int64_t slot_stride_packets,
    int64_t query_slot,
    int64_t key_slot,
    int64_t value_slot,
    int64_t local_sequence,
    int rank) {
  multi_rank_barrier<nranks>(sg, self_sg, rank);
  constexpr int local_heads = kQwenGlobalHeads / nranks;
  const int64_t rows = local_sequence * local_heads;
  const int dst_rank = blockIdx.x % nranks;
  const int rank_block = blockIdx.x / nranks;
  const int rank_blocks = gridDim.x / nranks;
  const int64_t tid =
      static_cast<int64_t>(rank_block) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(rank_blocks) * blockDim.x;
  const int64_t packet = tid & (kQwenHeadDimPackets - 1);
  const int64_t row_stride = stride / kQwenHeadDimPackets;
  auto* query_output = reinterpret_cast<uint4*>(outputs.ptrs[dst_rank]) +
                       query_slot * slot_stride_packets;
  auto* key_output = reinterpret_cast<uint4*>(outputs.ptrs[dst_rank]) +
                     key_slot * slot_stride_packets;
  auto* value_output = reinterpret_cast<uint4*>(outputs.ptrs[dst_rank]) +
                       value_slot * slot_stride_packets;

  for (int64_t row = tid / kQwenHeadDimPackets; row < rows;
       row += row_stride) {
    const int64_t sequence = row / local_heads;
    const int64_t head = row - sequence * local_heads;
    const int64_t src =
        (sequence * kQwenGlobalHeads + dst_rank * local_heads + head) *
            kQwenHeadDimPackets +
        packet;
    const int64_t dst =
        ((rank * local_sequence + sequence) * local_heads + head) *
            kQwenHeadDimPackets +
        packet;
    query_output[dst] = query[src];
    key_output[dst] = key[src];
    value_output[dst] = value[src];
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks>(sg, self_sg, rank);
}

template <int nranks>
__global__ void __launch_bounds__(kThreads, 1)
    custom_ulysses_prefix_output_kernel(
        RankData outputs,
        RankSignals sg,
        Signal* self_sg,
        const uint4* __restrict__ prefix,
        const uint4* __restrict__ sharded,
        int64_t slot_stride_packets,
        int64_t slot,
        int64_t prefix_sequence,
        int64_t local_sequence,
        int rank) {
  multi_rank_barrier<nranks>(sg, self_sg, rank);
  constexpr int local_heads = kQwenGlobalHeads / nranks;
  const int dst_rank = blockIdx.x % nranks;
  const int rank_block = blockIdx.x / nranks;
  const int rank_blocks = gridDim.x / nranks;
  const int64_t tid =
      static_cast<int64_t>(rank_block) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(rank_blocks) * blockDim.x;
  const int64_t packet = tid & (kQwenHeadDimPackets - 1);
  const int64_t row_stride = stride / kQwenHeadDimPackets;
  auto* output = reinterpret_cast<uint4*>(outputs.ptrs[dst_rank]) +
                 slot * slot_stride_packets;

  const int64_t prefix_rows = prefix_sequence * local_heads;
  for (int64_t row = tid / kQwenHeadDimPackets; row < prefix_rows;
       row += row_stride) {
    const int64_t sequence = row / local_heads;
    const int64_t head = row - sequence * local_heads;
    const int64_t dst =
        (sequence * kQwenGlobalHeads + rank * local_heads + head) *
            kQwenHeadDimPackets +
        packet;
    output[dst] = prefix[row * kQwenHeadDimPackets + packet];
  }

  const int64_t sharded_rows = local_sequence * local_heads;
  const int64_t output_shard_offset =
      prefix_sequence * kQwenGlobalHeads * kQwenHeadDimPackets;
  for (int64_t row = tid / kQwenHeadDimPackets; row < sharded_rows;
       row += row_stride) {
    const int64_t sequence = row / local_heads;
    const int64_t head = row - sequence * local_heads;
    const int64_t src =
        ((dst_rank * local_sequence + sequence) * local_heads + head) *
            kQwenHeadDimPackets +
        packet;
    const int64_t dst =
        output_shard_offset +
        (sequence * kQwenGlobalHeads + rank * local_heads + head) *
            kQwenHeadDimPackets +
        packet;
    output[dst] = sharded[src];
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks>(sg, self_sg, rank);
}

RankSignals make_rank_signals(ffi::TensorView signal_ptrs_cpu,
                              int64_t world_size) {
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
    data.ptrs[i] = reinterpret_cast<void*>(ptrs[i]);
  }
  return data;
}

int64_t tensor_nbytes(ffi::TensorView tensor) {
  const int64_t elem_bytes =
      (static_cast<int64_t>(tensor.dtype().bits) * tensor.dtype().lanes + 7) /
      8;
  return tensor_numel(tensor) * elem_bytes;
}

void check_args(ffi::TensorView rank_data,
                ffi::TensorView signal_ptrs_cpu,
                ffi::TensorView input,
                ffi::TensorView output,
                int64_t self_signal_ptr,
                int64_t slot_stride_bytes,
                int64_t slot,
                int64_t slots,
                int64_t rank,
                int64_t world_size) {
  CHECK_MUSA_CONTIGUOUS(input);
  CHECK_MUSA_CONTIGUOUS(output);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), output.dtype()));
  TVM_FFI_ICHECK_EQ(tensor_numel(input), tensor_numel(output));
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 8);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK(slots > 0);
  TVM_FFI_ICHECK(slot >= 0 && slot < slots);
  TVM_FFI_ICHECK(self_signal_ptr != 0);
  TVM_FFI_ICHECK(slot_stride_bytes > 0);
  TVM_FFI_ICHECK_EQ(slot_stride_bytes % 16, 0);
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16, 0);
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(output.data_ptr()) % 16, 0);
  const auto* output_ptrs =
      static_cast<const int64_t*>(rank_data.data_ptr());
  const auto* signal_ptrs =
      static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    TVM_FFI_ICHECK(output_ptrs[i] != 0);
    TVM_FFI_ICHECK(signal_ptrs[i] != 0);
  }
  const int64_t input_nbytes = tensor_nbytes(input);
  TVM_FFI_ICHECK(input_nbytes > 0);
  TVM_FFI_ICHECK_LE(input_nbytes, slot_stride_bytes);
  TVM_FFI_ICHECK_EQ(input_nbytes % (world_size * 16), 0);
}

}  // namespace

int64_t sgl_musa_custom_a2a_meta_size() {
  return static_cast<int64_t>(sizeof(Signal));
}

void sgl_musa_custom_a2a_launch(ffi::TensorView rank_data,
                                ffi::TensorView signal_ptrs_cpu,
                                ffi::TensorView input,
                                ffi::TensorView output,
                                int64_t self_signal_ptr,
                                int64_t slot_stride_bytes,
                                int64_t slot,
                                int64_t slots,
                                int64_t rank,
                                int64_t world_size) {
  check_args(rank_data, signal_ptrs_cpu, input, output, self_signal_ptr,
             slot_stride_bytes, slot, slots, rank, world_size);
  const int64_t input_nbytes = tensor_nbytes(input);
  const int64_t packets = input_nbytes / 16;
  const int64_t chunk_packets = packets / world_size;
  const int blocks_per_rank = static_cast<int>(std::min<int64_t>(
      kBlockLimit / world_size,
      (chunk_packets + kThreads - 1) / kThreads));
  const int blocks = blocks_per_rank * world_size;
  auto data = make_rank_data(rank_data);
  auto sg = make_rank_signals(signal_ptrs_cpu, world_size);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(input.device());
  switch (world_size) {
    case 2:
      custom_all_to_all_kernel<2><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          packets, slot_stride_bytes / 16, static_cast<int>(slot),
          static_cast<int>(rank));
      break;
    case 4:
      custom_all_to_all_kernel<4><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          packets, slot_stride_bytes / 16, static_cast<int>(slot),
          static_cast<int>(rank));
      break;
    case 8:
      custom_all_to_all_kernel<8><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          packets, slot_stride_bytes / 16, static_cast<int>(slot),
          static_cast<int>(rank));
      break;
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom All-to-All kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_custom_ulysses_launch(ffi::TensorView rank_data,
                                    ffi::TensorView signal_ptrs_cpu,
                                    ffi::TensorView input,
                                    ffi::TensorView output,
                                    int64_t self_signal_ptr,
                                    int64_t slot_stride_bytes,
                                    int64_t slot,
                                    int64_t slots,
                                    int64_t local_sequence,
                                    int64_t rank,
                                    int64_t world_size,
                                    int64_t input_layout) {
  check_args(rank_data, signal_ptrs_cpu, input, output, self_signal_ptr,
             slot_stride_bytes, slot, slots, rank, world_size);
  TVM_FFI_ICHECK(local_sequence > 0);
  TVM_FFI_ICHECK_EQ(tensor_numel(input),
                    local_sequence * kQwenGlobalHeads * kQwenHeadDim);
  TVM_FFI_ICHECK(input_layout == 0 || input_layout == 1);
  const int64_t packets_per_rank =
      tensor_nbytes(input) / (world_size * 16);
  const int blocks_per_rank = static_cast<int>(std::min<int64_t>(
      kBlockLimit / world_size,
      (packets_per_rank + kThreads - 1) / kThreads));
  const int blocks = blocks_per_rank * world_size;
  auto data = make_rank_data(rank_data);
  auto sg = make_rank_signals(signal_ptrs_cpu, world_size);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(input.device());
  if (world_size == 2) {
    if (input_layout) {
      custom_ulysses_kernel<2, true><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          slot_stride_bytes / 16, slot, local_sequence, rank);
    } else {
      custom_ulysses_kernel<2, false><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          slot_stride_bytes / 16, slot, local_sequence, rank);
    }
  } else if (world_size == 4) {
    if (input_layout) {
      custom_ulysses_kernel<4, true><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          slot_stride_bytes / 16, slot, local_sequence, rank);
    } else {
      custom_ulysses_kernel<4, false><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          slot_stride_bytes / 16, slot, local_sequence, rank);
    }
  } else {
    if (input_layout) {
      custom_ulysses_kernel<8, true><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          slot_stride_bytes / 16, slot, local_sequence, rank);
    } else {
      custom_ulysses_kernel<8, false><<<blocks, kThreads, 0, stream>>>(
          data, sg, self_sg, static_cast<const uint4*>(input.data_ptr()),
          slot_stride_bytes / 16, slot, local_sequence, rank);
    }
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom Ulysses kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_custom_qkv_ulysses_launch(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView query,
    ffi::TensorView key,
    ffi::TensorView value,
    ffi::TensorView query_output,
    ffi::TensorView key_output,
    ffi::TensorView value_output,
    int64_t self_signal_ptr,
    int64_t slot_stride_bytes,
    int64_t query_slot,
    int64_t key_slot,
    int64_t value_slot,
    int64_t slots,
    int64_t local_sequence,
    int64_t rank,
    int64_t world_size) {
  check_args(rank_data, signal_ptrs_cpu, query, query_output,
             self_signal_ptr, slot_stride_bytes, query_slot, slots, rank,
             world_size);
  check_args(rank_data, signal_ptrs_cpu, key, key_output, self_signal_ptr,
             slot_stride_bytes, key_slot, slots, rank, world_size);
  check_args(rank_data, signal_ptrs_cpu, value, value_output,
             self_signal_ptr, slot_stride_bytes, value_slot, slots, rank,
             world_size);
  TVM_FFI_ICHECK(local_sequence > 0);
  const int64_t expected_numel =
      local_sequence * kQwenGlobalHeads * kQwenHeadDim;
  TVM_FFI_ICHECK_EQ(tensor_numel(query), expected_numel);
  TVM_FFI_ICHECK_EQ(tensor_numel(key), expected_numel);
  TVM_FFI_ICHECK_EQ(tensor_numel(value), expected_numel);
  const int64_t packets_per_rank =
      tensor_nbytes(query) / (world_size * 16);
  const int blocks_per_rank = static_cast<int>(std::min<int64_t>(
      kBlockLimit / world_size,
      (packets_per_rank + kThreads - 1) / kThreads));
  const int blocks = blocks_per_rank * world_size;
  auto data = make_rank_data(rank_data);
  auto sg = make_rank_signals(signal_ptrs_cpu, world_size);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(query.device());

  if (world_size == 2) {
    custom_qkv_ulysses_kernel<2><<<blocks, kThreads, 0, stream>>>(
        data, sg, self_sg, static_cast<const uint4*>(query.data_ptr()),
        static_cast<const uint4*>(key.data_ptr()),
        static_cast<const uint4*>(value.data_ptr()), slot_stride_bytes / 16,
        query_slot, key_slot, value_slot, local_sequence, rank);
  } else if (world_size == 4) {
    custom_qkv_ulysses_kernel<4><<<blocks, kThreads, 0, stream>>>(
        data, sg, self_sg, static_cast<const uint4*>(query.data_ptr()),
        static_cast<const uint4*>(key.data_ptr()),
        static_cast<const uint4*>(value.data_ptr()), slot_stride_bytes / 16,
        query_slot, key_slot, value_slot, local_sequence, rank);
  } else {
    custom_qkv_ulysses_kernel<8><<<blocks, kThreads, 0, stream>>>(
        data, sg, self_sg, static_cast<const uint4*>(query.data_ptr()),
        static_cast<const uint4*>(key.data_ptr()),
        static_cast<const uint4*>(value.data_ptr()), slot_stride_bytes / 16,
        query_slot, key_slot, value_slot, local_sequence, rank);
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom QKV Ulysses kernel failed: "
      << musaGetErrorString(err);
}

void sgl_musa_custom_ulysses_prefix_output_launch(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView prefix,
    ffi::TensorView sharded,
    ffi::TensorView output,
    int64_t self_signal_ptr,
    int64_t slot_stride_bytes,
    int64_t slot,
    int64_t slots,
    int64_t prefix_sequence,
    int64_t local_sequence,
    int64_t rank,
    int64_t world_size) {
  check_args(rank_data, signal_ptrs_cpu, sharded, sharded,
             self_signal_ptr, slot_stride_bytes, slot, slots, rank,
             world_size);
  CHECK_MUSA_CONTIGUOUS(prefix);
  CHECK_MUSA_CONTIGUOUS(output);
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(prefix.data_ptr()) % 16, 0);
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(output.data_ptr()) % 16, 0);
  TVM_FFI_ICHECK_EQ(prefix.device().device_id, sharded.device().device_id);
  TVM_FFI_ICHECK_EQ(output.device().device_id, sharded.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(prefix.dtype(), sharded.dtype()));
  TVM_FFI_ICHECK(dtype_equal(output.dtype(), sharded.dtype()));
  TVM_FFI_ICHECK(prefix_sequence > 0);
  TVM_FFI_ICHECK(local_sequence > 0);
  const int64_t local_heads = kQwenGlobalHeads / world_size;
  TVM_FFI_ICHECK_EQ(
      tensor_numel(prefix),
      prefix_sequence * local_heads * kQwenHeadDim);
  TVM_FFI_ICHECK_EQ(
      tensor_numel(sharded),
      local_sequence * world_size * local_heads * kQwenHeadDim);
  TVM_FFI_ICHECK_EQ(
      tensor_numel(output),
      (prefix_sequence + local_sequence) * kQwenGlobalHeads * kQwenHeadDim);
  TVM_FFI_ICHECK_LE(tensor_nbytes(output), slot_stride_bytes);

  const int64_t rows =
      std::max(prefix_sequence * local_heads, local_sequence * local_heads);
  const int64_t packets_per_rank = rows * kQwenHeadDimPackets;
  const int blocks_per_rank = static_cast<int>(std::min<int64_t>(
      kBlockLimit / world_size,
      (packets_per_rank + kThreads - 1) / kThreads));
  const int blocks = blocks_per_rank * world_size;
  auto data = make_rank_data(rank_data);
  auto sg = make_rank_signals(signal_ptrs_cpu, world_size);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(prefix.device());

  if (world_size == 2) {
    custom_ulysses_prefix_output_kernel<2><<<blocks, kThreads, 0, stream>>>(
        data, sg, self_sg, static_cast<const uint4*>(prefix.data_ptr()),
        static_cast<const uint4*>(sharded.data_ptr()),
        slot_stride_bytes / 16, slot, prefix_sequence, local_sequence, rank);
  } else if (world_size == 4) {
    custom_ulysses_prefix_output_kernel<4><<<blocks, kThreads, 0, stream>>>(
        data, sg, self_sg, static_cast<const uint4*>(prefix.data_ptr()),
        static_cast<const uint4*>(sharded.data_ptr()),
        slot_stride_bytes / 16, slot, prefix_sequence, local_sequence, rank);
  } else {
    custom_ulysses_prefix_output_kernel<8><<<blocks, kThreads, 0, stream>>>(
        data, sg, self_sg, static_cast<const uint4*>(prefix.data_ptr()),
        static_cast<const uint4*>(sharded.data_ptr()),
        slot_stride_bytes / 16, slot, prefix_sequence, local_sequence, rank);
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom Ulysses prefix output kernel failed: "
      << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_a2a_meta_size,
                              sgl_musa_custom_a2a_meta_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_a2a_launch,
                              sgl_musa_custom_a2a_launch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ulysses_launch,
                              sgl_musa_custom_ulysses_launch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_qkv_ulysses_launch,
                              sgl_musa_custom_qkv_ulysses_launch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ulysses_prefix_output_launch,
                              sgl_musa_custom_ulysses_prefix_output_launch);
