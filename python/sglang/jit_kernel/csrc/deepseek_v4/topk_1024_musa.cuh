#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kTopK = 1024;
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);  // 64KB dynamic candidate storage.
constexpr uint32_t kRadix = 256;
constexpr uint32_t kHistGroups = 4;
constexpr uint32_t kHistGroupThreads = kBlockSize / kHistGroups;

struct MusaTopK1024Params {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t page_bits;
};

SGL_DEVICE uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, int32_t raw_index, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  const uint32_t i = static_cast<uint32_t>(raw_index);
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

SGL_DEVICE void naive_transform_musa(
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ raw_indices,
    const uint32_t length,
    const uint32_t page_bits) {
  const uint32_t tx = threadIdx.x;
  if (tx < length) {
    indices[tx] = page_to_indices(page_table, static_cast<int32_t>(tx), page_bits);
    if (raw_indices != nullptr) raw_indices[tx] = static_cast<int32_t>(tx);
  } else if (tx < kTopK) {
    indices[tx] = -1;
    if (raw_indices != nullptr) raw_indices[tx] = -1;
  }
}

SGL_DEVICE void radix_topk_musa_split_hist(const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length) {
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][kRadix + 32];
  alignas(128) __shared__ uint32_t s_group_hist[kHistGroups][kRadix + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ uint32_t s_sub_bin24_min;
  alignas(128) __shared__ uint32_t s_sub_bin24_max;
  alignas(128) __shared__ int32_t s_last_remain;

  extern __shared__ uint32_t s_input_idx[][kSMEM / (2 * sizeof(int32_t))];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = kTopK;
  auto& s_histogram = _s_histogram_buf[0];

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      if (tx < kRadix) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = _s_histogram_buf[k][tx];
        if (tx + j < kRadix) value += _s_histogram_buf[k][tx + j];
        _s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads_lm();
    }
  };

  for (uint32_t i = tx; i < kHistGroups * (kRadix + 32); i += kBlockSize) {
    reinterpret_cast<uint32_t*>(s_group_hist)[i] = 0;
  }
  __syncthreads_lm();

  const uint32_t hist_group = tx / kHistGroupThreads;
  for (uint32_t idx = tx; idx < length; idx += kBlockSize) {
    const uint32_t bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_group_hist[hist_group][bin], 1);
  }
  __syncthreads_lm();

  if (tx < kRadix + 1) {
    uint32_t sum = 0;
#pragma unroll
    for (uint32_t g = 0; g < kHistGroups; ++g) sum += s_group_hist[g][tx];
    s_histogram[tx] = sum;
  }
  __syncthreads_lm();

  run_cumsum();
  if (tx < kRadix && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads_lm();

  const auto threshold_bin_0 = s_threshold_bin_id;
  remain_topk -= s_histogram[threshold_bin_0 + 1];
  if (remain_topk == 0) {
    for (uint32_t idx = tx; idx < length; idx += kBlockSize) {
      const uint32_t bin = convert_to_uint8(input[idx]);
      if (bin > threshold_bin_0) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      }
    }
    __syncthreads_lm();
    return;
  }

  __syncthreads_lm();
  if (tx < kRadix + 1) s_histogram[tx] = 0;
  if (tx == 0) {
    s_num_input[1] = 0;
    s_sub_bin24_min = 0xFFFFFFFFu;
    s_sub_bin24_max = 0;
  }
  __syncthreads_lm();

  for (uint32_t idx = tx; idx < length; idx += kBlockSize) {
    const float raw_input = input[idx];
    const uint32_t bin = convert_to_uint8(raw_input);
    if (bin > threshold_bin_0) {
      const auto pos = ::atomicAdd(&s_counter, 1);
      output[pos] = idx;
    } else if (bin == threshold_bin_0) {
      const auto pos = ::atomicAdd(&s_num_input[1], 1);
      if (pos < SMEM_INPUT_SIZE) {
        s_input_idx[1][pos] = idx;
        const auto key = convert_to_uint32(raw_input);
        const auto sub_bin24 = (key >> 24) & 0xFF;
        const auto sub_bin16 = (key >> 16) & 0xFF;
        ::atomicMin(&s_sub_bin24_min, sub_bin24);
        ::atomicMax(&s_sub_bin24_max, sub_bin24);
        ::atomicAdd(&s_histogram[sub_bin16], 1);
      }
    }
  }
  __syncthreads_lm();

  int start_round = 1;
  if (s_sub_bin24_min != s_sub_bin24_max) {
    if (tx < kRadix + 1) s_histogram[tx] = 0;
    if (tx == 0) s_num_input[0] = 0;
    __syncthreads_lm();
    const auto raw_num_input = s_num_input[1];
    const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;
    for (uint32_t i = tx; i < num_input; i += kBlockSize) {
      const auto idx = s_input_idx[1][i];
      const auto key = convert_to_uint32(input[idx]);
      const auto pos = ::atomicAdd(&s_num_input[0], 1);
      if (pos < SMEM_INPUT_SIZE) {
        s_input_idx[0][pos] = idx;
        const auto sub_bin24 = (key >> 24) & 0xFF;
        ::atomicAdd(&s_histogram[sub_bin24], 1);
      }
    }
    start_round = 0;
    __syncthreads_lm();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    if (round < start_round) continue;
    const auto r_idx = round % 2;
    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    run_cumsum();
    if (tx < kRadix && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = remain_topk - s_histogram[tx + 1];
    }
    __syncthreads_lm();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += kBlockSize) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        }
      }
      __syncthreads_lm();
      break;
    }

    __syncthreads_lm();
    if (tx < kRadix + 1) s_histogram[tx] = 0;
    __syncthreads_lm();
    for (uint32_t i = tx; i < num_input; i += kBlockSize) {
      const auto idx = s_input_idx[r_idx][i];
      const auto raw_input = input[idx];
      const auto offset = 24 - round * 8;
      const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const auto pos = ::atomicAdd(&s_last_remain, -1);
          if (pos > 0) output[kTopK - pos] = idx;
        } else {
          const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
          if (pos < SMEM_INPUT_SIZE) {
            s_input_idx[r_idx ^ 1][pos] = idx;
            const auto sub_bin = (convert_to_uint32(raw_input) >> (offset - 8)) & 0xFF;
            ::atomicAdd(&s_histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads_lm();
  }
}

__global__ void topk_1024_transform_musa_candidate(const MusaTopK1024Params params) {
  const uint32_t tx = threadIdx.x;
  const uint32_t work_id = blockIdx.x;
  const uint32_t seq_len = static_cast<uint32_t>(params.seq_lens[work_id]);
  const float* score_ptr = params.scores + work_id * params.score_stride;
  const int32_t* page_ptr = params.page_table + work_id * params.page_table_stride;
  int32_t* page_indices_ptr = params.page_indices + work_id * kTopK;
  int32_t* raw_indices_ptr = params.raw_indices == nullptr ? nullptr : params.raw_indices + work_id * kTopK;

  if (seq_len <= kTopK) {
    naive_transform_musa(page_ptr, page_indices_ptr, raw_indices_ptr, seq_len, params.page_bits);
    return;
  }

  __shared__ int32_t s_topk_indices[kTopK];
  radix_topk_musa_split_hist(score_ptr, s_topk_indices, seq_len);
  if (tx < kTopK) {
    const int32_t raw_index = s_topk_indices[tx];
    page_indices_ptr[tx] = page_to_indices(page_ptr, raw_index, params.page_bits);
    if (raw_indices_ptr != nullptr) raw_indices_ptr[tx] = raw_index;
  }
}

template <auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once(host::DebugInfo where = {}) {
  [[maybe_unused]]
  static const auto result = [] {
    const void* fptr = reinterpret_cast<const void*>(f);
    return ::musaFuncSetAttribute(fptr, ::musaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
  host::RuntimeDeviceCheck(result, where);
}

struct MusaTopK1024Kernel {
  static constexpr auto kernel = topk_1024_transform_musa_candidate;

  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({host::details::SizeRef(B), host::details::SizeRef(-1)})
        .with_strides({host::details::SizeRef(S), host::details::SizeRef(1)})
        .with_dtype<float>()
        .with_device(host::details::DeviceRef(device))
        .verify(scores);
    TensorMatcher({host::details::SizeRef(B)})
        .with_dtype<int32_t>()
        .with_device(host::details::DeviceRef(device))
        .verify(seq_lens);
    TensorMatcher({host::details::SizeRef(B), host::details::SizeRef(-1)})
        .with_strides({host::details::SizeRef(P), host::details::SizeRef(1)})
        .with_dtype<int32_t>()
        .with_device(host::details::DeviceRef(device))
        .verify(page_table);
    TensorMatcher({host::details::SizeRef(B), host::details::SizeRef(1024)})
        .with_dtype<int32_t>()
        .with_device(host::details::DeviceRef(device))
        .verify(page_indices);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({host::details::SizeRef(B), host::details::SizeRef(1024)})
          .with_dtype<int32_t>()
          .with_device(host::details::DeviceRef(device))
          .verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be power of 2");
    const auto params = MusaTopK1024Params{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = static_cast<uint32_t>(__builtin_ctz(page_size)),
    };
    constexpr auto kSMEM_ = kSMEM + sizeof(int32_t);
    setup_kernel_smem_once<kernel, kSMEM_>();
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), kBlockSize, device.unwrap(), kSMEM_)(kernel, params);
  }
};

}  // namespace
