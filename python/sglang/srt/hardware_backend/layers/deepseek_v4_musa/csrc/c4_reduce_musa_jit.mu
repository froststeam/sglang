#include <torch/all.h>
#include <torch/extension.h>
#include "musa.h"
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <cstdint>
#include <cfloat>
#include <musa_runtime.h>

namespace {

constexpr int kPageSize = 4;
constexpr int kVec = 4;
constexpr int kWarp = 32;
constexpr int kThreads = 128;
constexpr int kRowsPerBlock = 4;
constexpr int kColsPerTile = kWarp * kVec;

__device__ __forceinline__ float4 load4(const float* ptr) {
  return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store4(float* ptr, const float (&v)[kVec]) {
  float4 out;
  out.x = v[0];
  out.y = v[1];
  out.z = v[2];
  out.w = v[3];
  *reinterpret_cast<float4*>(ptr) = out;
}

__device__ __forceinline__ void unpack4(float4 v, float (&out)[kVec]) {
  out[0] = v.x;
  out[1] = v.y;
  out[2] = v.z;
  out[3] = v.w;
}

__device__ __forceinline__ void unpack4_add(float4 v, float4 b, float (&out)[kVec]) {
  out[0] = v.x + b.x;
  out[1] = v.y + b.y;
  out[2] = v.z + b.z;
  out[3] = v.w + b.w;
}

template <int kHeadDim>
__device__ __forceinline__ void load_slot(
    const float* __restrict__ kv_score_buffer,
    const float* __restrict__ kv_score_input,
    const float* __restrict__ ape,
    int64_t buffer_stride0,
    int ragged_id,
    int window_len,
    int slot,
    int col_base,
    int read_page_0,
    int read_page_1,
    bool seq_len_eq_4,
    float (&vals)[kVec],
    float (&logits)[kVec]) {
  constexpr int kWidth = kHeadDim * 4;
  if (seq_len_eq_4 && slot < 4) {
    const float4 bias = load4(ape + slot * kHeadDim + col_base);
    vals[0] = 0.0f;
    vals[1] = 0.0f;
    vals[2] = 0.0f;
    vals[3] = 0.0f;
    logits[0] = -1.0e9f + bias.x;
    logits[1] = -1.0e9f + bias.y;
    logits[2] = -1.0e9f + bias.z;
    logits[3] = -1.0e9f + bias.w;
    return;
  }

  if (slot < window_len) {
    int read_page = read_page_1;
    if (window_len > 4 && slot < 4) {
      read_page = read_page_0;
    }
    const float* src = kv_score_buffer + static_cast<int64_t>(read_page) * buffer_stride0 + (slot & 3) * kWidth;
    const int value_offset = slot < 4 ? 0 : kHeadDim;
    const int score_offset = slot < 4 ? kHeadDim * 2 : kHeadDim * 3;
    unpack4(load4(src + value_offset + col_base), vals);
    unpack4_add(load4(src + score_offset + col_base), load4(ape + slot * kHeadDim + col_base), logits);
  } else {
    const int input_row = ragged_id + slot - 7;
    const float* src = kv_score_input + static_cast<int64_t>(input_row) * kWidth;
    const int value_offset = slot < 4 ? 0 : kHeadDim;
    const int score_offset = slot < 4 ? kHeadDim * 2 : kHeadDim * 3;
    unpack4(load4(src + value_offset + col_base), vals);
    unpack4_add(load4(src + score_offset + col_base), load4(ape + slot * kHeadDim + col_base), logits);
  }
}

template <int kHeadDim>
__global__ __launch_bounds__(kThreads, 4) void c4_page_reduce_float_kernel(
    const float* __restrict__ kv_score_buffer,
    const float* __restrict__ kv_score_input,
    const float* __restrict__ ape,
    const int32_t* __restrict__ extra_data,
    const int32_t* __restrict__ compress_rows,
    float* __restrict__ out,
    int64_t buffer_stride0,
    int64_t rows_stride0,
    int64_t extra_stride0,
    int num_rows) {
  const int tx = threadIdx.x;
  const int warp_id = tx / kWarp;
  const int lane = tx & (kWarp - 1);
  const int row_id = blockIdx.x * kRowsPerBlock + warp_id;
  const int col_base = blockIdx.y * kColsPerTile + lane * kVec;
  if (row_id >= num_rows) {
    return;
  }
  if (col_base >= kHeadDim) {
    return;
  }

  const int32_t* row = compress_rows + static_cast<int64_t>(row_id) * rows_stride0;
  const int ragged_id = row[0];
  const int batch_id = row[1];
  const int position = row[2];
  const int window_len = row[3];
  if (window_len < 0) {
    return;
  }

  const int32_t* extra = extra_data + static_cast<int64_t>(batch_id) * extra_stride0;
  const int read_page_0 = extra[0];
  const int read_page_1 = extra[1];
  const bool seq_len_eq_4 = (position + 1) == 4;

  float vals[8][kVec];
  float logits[8][kVec];
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    load_slot<kHeadDim>(
        kv_score_buffer,
        kv_score_input,
        ape,
        buffer_stride0,
        ragged_id,
        window_len,
        slot,
        col_base,
        read_page_0,
        read_page_1,
        seq_len_eq_4,
        vals[slot],
        logits[slot]);
  }

  float max_v[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    max_v[i] = logits[0][i];
  }
#pragma unroll
  for (int slot = 1; slot < 8; ++slot) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      max_v[i] = fmaxf(max_v[i], logits[slot][i]);
    }
  }

  float denom[kVec] = {0.f, 0.f, 0.f, 0.f};
  float acc[kVec] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float w = __expf(logits[slot][i] - max_v[i]);
      denom[i] += w;
      acc[i] += vals[slot][i] * w;
    }
  }

  float* dst = out + static_cast<int64_t>(ragged_id) * kHeadDim + col_base;
  float result[kVec];
  result[0] = acc[0] / denom[0];
  result[1] = acc[1] / denom[1];
  result[2] = acc[2] / denom[2];
  result[3] = acc[3] / denom[3];
  store4(dst, result);
}

template <int kHeadDim>
__global__ __launch_bounds__(kThreads, 16) void c4_page_write_float_kernel(
    float* __restrict__ kv_score_buffer,
    const float* __restrict__ kv_score_input,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ extra_data,
    const int32_t* __restrict__ write_rows,
    int64_t buffer_stride0,
    int64_t input_stride0,
    int64_t indices_stride0,
    int64_t extra_stride0,
    int64_t rows_stride0,
    int num_rows) {
  const int tx = threadIdx.x;
  const int warp_id = tx / kWarp;
  const int lane = tx & (kWarp - 1);
  const int row_id = blockIdx.x * kRowsPerBlock + warp_id;
  constexpr int kWidth = kHeadDim * 4;
  const int col_base = blockIdx.y * kColsPerTile + lane * kVec;
  if (row_id >= num_rows) {
    return;
  }
  if (col_base >= kWidth) {
    return;
  }

  const int32_t* row = write_rows + static_cast<int64_t>(row_id) * rows_stride0;
  const int ragged_id = row[0];
  const int batch_id = row[1];
  const int position = row[2];
  const int window_len = row[3];
  if (window_len < 0) {
    return;
  }

  int block_id = indices[static_cast<int64_t>(batch_id) * indices_stride0];
  const int32_t* extra = extra_data + static_cast<int64_t>(batch_id) * extra_stride0;
  if (position < extra[3]) {
    block_id = extra[2];
  }

  const float* src = kv_score_input + static_cast<int64_t>(ragged_id) * input_stride0 + col_base;
  float* dst = kv_score_buffer + static_cast<int64_t>(block_id) * buffer_stride0 + (position & 3) * kWidth + col_base;
  *reinterpret_cast<float4*>(dst) = load4(src);
}

int infer_c4_head_dim(
    const torch::Tensor& kv_score_buffer,
    const torch::Tensor& kv_score_input,
    const torch::Tensor& ape,
    const torch::Tensor& out) {
  TORCH_CHECK(kv_score_input.dim() == 2, "kv_score_input must be 2D");
  TORCH_CHECK(ape.dim() == 2 && ape.size(0) == 8, "ape must be [8,H]");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  const int64_t head_dim = ape.size(1);
  TORCH_CHECK(out.size(1) == head_dim, "out last dim must match ape head_dim");
  TORCH_CHECK(kv_score_input.size(1) == head_dim * 4, "kv_score_input last dim must be 4*head_dim");
  TORCH_CHECK(kv_score_buffer.dim() == 3 && kv_score_buffer.size(1) == kPageSize,
              "kv_score_buffer must be [B,4,4*H]");
  TORCH_CHECK(kv_score_buffer.size(2) == head_dim * 4, "kv_score_buffer last dim must be 4*head_dim");
  TORCH_CHECK(head_dim == 128 || head_dim == 512, "C4 MUSA JIT currently supports head_dim 128 or 512");
  return static_cast<int>(head_dim);
}

template <int kHeadDim>
void launch_c4_page_reduce_float(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor ape,
    torch::Tensor extra_data,
    torch::Tensor compress_rows,
    torch::Tensor out,
    musaStream_t stream) {
  const int num_rows = static_cast<int>(compress_rows.size(0));
  if (num_rows == 0) {
    return;
  }
  const dim3 blocks((num_rows + kRowsPerBlock - 1) / kRowsPerBlock,
                    (kHeadDim + kColsPerTile - 1) / kColsPerTile);
  c4_page_reduce_float_kernel<kHeadDim><<<blocks, kThreads, 0, stream>>>(
      static_cast<const float*>(kv_score_buffer.data_ptr()),
      static_cast<const float*>(kv_score_input.data_ptr()),
      static_cast<const float*>(ape.data_ptr()),
      static_cast<const int32_t*>(extra_data.data_ptr()),
      static_cast<const int32_t*>(compress_rows.data_ptr()),
      static_cast<float*>(out.data_ptr()),
      kv_score_buffer.stride(0),
      compress_rows.stride(0),
      extra_data.stride(0),
      num_rows);
}

template <int kHeadDim>
void launch_c4_page_write_float(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor indices,
    torch::Tensor extra_data,
    torch::Tensor write_rows,
    musaStream_t stream) {
  const int num_rows = static_cast<int>(write_rows.size(0));
  if (num_rows == 0) {
    return;
  }
  constexpr int kWidth = kHeadDim * 4;
  const dim3 blocks((num_rows + kRowsPerBlock - 1) / kRowsPerBlock,
                    (kWidth + kColsPerTile - 1) / kColsPerTile);
  c4_page_write_float_kernel<kHeadDim><<<blocks, kThreads, 0, stream>>>(
      static_cast<float*>(kv_score_buffer.data_ptr()),
      static_cast<const float*>(kv_score_input.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<const int32_t*>(extra_data.data_ptr()),
      static_cast<const int32_t*>(write_rows.data_ptr()),
      kv_score_buffer.stride(0),
      kv_score_input.stride(0),
      indices.stride(0),
      extra_data.stride(0),
      write_rows.stride(0),
      num_rows);
}

}  // namespace

void c4_page_reduce_float(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor ape,
    torch::Tensor extra_data,
    torch::Tensor compress_rows,
    torch::Tensor out) {
  TORCH_CHECK(kv_score_buffer.is_musa(), "kv_score_buffer must be on MUSA");
  TORCH_CHECK(kv_score_input.is_musa(), "kv_score_input must be on MUSA");
  TORCH_CHECK(ape.is_musa(), "ape must be on MUSA");
  TORCH_CHECK(extra_data.is_musa(), "extra_data must be on MUSA");
  TORCH_CHECK(compress_rows.is_musa(), "compress_rows must be on MUSA");
  TORCH_CHECK(out.is_musa(), "out must be on MUSA");
  TORCH_CHECK(kv_score_buffer.scalar_type() == at::ScalarType::Float, "kv_score_buffer must be float32");
  TORCH_CHECK(kv_score_input.scalar_type() == at::ScalarType::Float, "kv_score_input must be float32");
  TORCH_CHECK(ape.scalar_type() == at::ScalarType::Float, "ape must be float32");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Float, "out must be float32");
  TORCH_CHECK(extra_data.scalar_type() == at::ScalarType::Int, "extra_data must be int32");
  TORCH_CHECK(compress_rows.scalar_type() == at::ScalarType::Int, "compress_rows must be int32");
  const int head_dim = infer_c4_head_dim(kv_score_buffer, kv_score_input, ape, out);
  TORCH_CHECK(extra_data.dim() == 2 && extra_data.size(1) >= 2, "extra_data must be [B,>=2]");
  TORCH_CHECK(compress_rows.dim() == 2 && compress_rows.size(1) == 4, "compress_rows must be [R,4]");

  const c10::musa::OptionalMUSAGuard device_guard(kv_score_input.device());
  const musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  if (head_dim == 128) {
    launch_c4_page_reduce_float<128>(kv_score_buffer, kv_score_input, ape, extra_data, compress_rows, out, stream);
  } else {
    launch_c4_page_reduce_float<512>(kv_score_buffer, kv_score_input, ape, extra_data, compress_rows, out, stream);
  }
}

void c4_page_prefill_float(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor ape,
    torch::Tensor indices,
    torch::Tensor extra_data,
    torch::Tensor compress_rows,
    torch::Tensor write_rows,
    torch::Tensor out) {
  TORCH_CHECK(kv_score_buffer.is_musa(), "kv_score_buffer must be on MUSA");
  TORCH_CHECK(kv_score_input.is_musa(), "kv_score_input must be on MUSA");
  TORCH_CHECK(ape.is_musa(), "ape must be on MUSA");
  TORCH_CHECK(indices.is_musa(), "indices must be on MUSA");
  TORCH_CHECK(extra_data.is_musa(), "extra_data must be on MUSA");
  TORCH_CHECK(compress_rows.is_musa(), "compress_rows must be on MUSA");
  TORCH_CHECK(write_rows.is_musa(), "write_rows must be on MUSA");
  TORCH_CHECK(out.is_musa(), "out must be on MUSA");
  TORCH_CHECK(kv_score_buffer.scalar_type() == at::ScalarType::Float, "kv_score_buffer must be float32");
  TORCH_CHECK(kv_score_input.scalar_type() == at::ScalarType::Float, "kv_score_input must be float32");
  TORCH_CHECK(ape.scalar_type() == at::ScalarType::Float, "ape must be float32");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Float, "out must be float32");
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int, "indices must be int32");
  TORCH_CHECK(extra_data.scalar_type() == at::ScalarType::Int, "extra_data must be int32");
  TORCH_CHECK(compress_rows.scalar_type() == at::ScalarType::Int, "compress_rows must be int32");
  TORCH_CHECK(write_rows.scalar_type() == at::ScalarType::Int, "write_rows must be int32");
  const int head_dim = infer_c4_head_dim(kv_score_buffer, kv_score_input, ape, out);
  TORCH_CHECK(extra_data.dim() == 2 && extra_data.size(1) >= 4, "extra_data must be [B,>=4]");
  TORCH_CHECK(compress_rows.dim() == 2 && compress_rows.size(1) == 4, "compress_rows must be [R,4]");
  TORCH_CHECK(write_rows.dim() == 2 && write_rows.size(1) == 4, "write_rows must be [R,4]");

  const c10::musa::OptionalMUSAGuard device_guard(kv_score_input.device());
  const musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  musaMemsetAsync(out.data_ptr(), 0, out.numel() * out.element_size(), stream);

  if (head_dim == 128) {
    launch_c4_page_reduce_float<128>(kv_score_buffer, kv_score_input, ape, extra_data, compress_rows, out, stream);
    launch_c4_page_write_float<128>(kv_score_buffer, kv_score_input, indices, extra_data, write_rows, stream);
  } else {
    launch_c4_page_reduce_float<512>(kv_score_buffer, kv_score_input, ape, extra_data, compress_rows, out, stream);
    launch_c4_page_write_float<512>(kv_score_buffer, kv_score_input, indices, extra_data, write_rows, stream);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("c4_page_reduce_float", &c4_page_reduce_float, "C4 page4 reduce float32 MUSA prototype");
  m.def("c4_page_prefill_float", &c4_page_prefill_float, "C4 page4 prefill float32 MUSA prototype");
}
