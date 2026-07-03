#include <torch/all.h>

#include "sgl_kernel_ops.h"

namespace {

bool is_power_of_two(int64_t x) { return x > 0 && (x & (x - 1)) == 0; }

torch::Tensor hadamard_transform_power2(const torch::Tensor& x, double scale) {
  TORCH_CHECK(x.dim() >= 1, "fast_hadamard_transform expects at least 1D input");

  const int64_t dim = x.size(-1);
  TORCH_CHECK(
      is_power_of_two(dim),
      "fast_hadamard_transform only supports power-of-two hidden dim in the built-in MUSA implementation, got ",
      dim);

  std::vector<int64_t> original_shape = x.sizes().vec();
  torch::Tensor y = x.contiguous().reshape({-1, dim});
  const int64_t batch = y.size(0);

  for (int64_t h = 1; h < dim; h <<= 1) {
    torch::Tensor y_view = y.reshape({batch, dim / (2 * h), 2, h});
    torch::Tensor a = y_view.select(2, 0);
    torch::Tensor b = y_view.select(2, 1);
    y = torch::stack({a + b, a - b}, 2).reshape({batch, dim});
  }

  if (scale != 1.0) {
    y = y * scale;
  }
  return y.reshape(original_shape);
}

}  // namespace

torch::Tensor fast_hadamard_transform(torch::Tensor& x, double scale) {
  return hadamard_transform_power2(x, scale);
}

torch::Tensor fast_hadamard_transform_12N(torch::Tensor& x, double scale) {
  TORCH_CHECK(
      false,
      "fast_hadamard_transform_12N is not available in the built-in MUSA implementation");
  return torch::Tensor();
}

torch::Tensor fast_hadamard_transform_20N(torch::Tensor& x, double scale) {
  TORCH_CHECK(
      false,
      "fast_hadamard_transform_20N is not available in the built-in MUSA implementation");
  return torch::Tensor();
}

torch::Tensor fast_hadamard_transform_28N(torch::Tensor& x, double scale) {
  TORCH_CHECK(
      false,
      "fast_hadamard_transform_28N is not available in the built-in MUSA implementation");
  return torch::Tensor();
}

torch::Tensor fast_hadamard_transform_40N(torch::Tensor& x, double scale) {
  TORCH_CHECK(
      false,
      "fast_hadamard_transform_40N is not available in the built-in MUSA implementation");
  return torch::Tensor();
}
