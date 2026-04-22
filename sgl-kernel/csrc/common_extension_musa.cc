/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  /*
   * From FlashInfer
   */
  m.def(
      "min_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? maybe_min_p_arr, float "
      "min_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("min_p_sampling_from_probs", torch::kMUSA, &min_p_sampling_from_probs);

  m.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  m.impl("top_k_renorm_probs", torch::kMUSA, &top_k_renorm_probs);

  m.def("top_p_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_p_arr, float top_p_val) -> ()");
  m.impl("top_p_renorm_probs", torch::kMUSA, &top_p_renorm_probs);

  m.def(
      "top_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? "
      "maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("top_p_sampling_from_probs", torch::kMUSA, &top_p_sampling_from_probs);

  m.def(
      "top_k_top_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? maybe_top_k_arr, "
      "float top_k_val, Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("top_k_top_p_sampling_from_probs", torch::kMUSA, &top_k_top_p_sampling_from_probs);

  /*
   * From csrc/mamba
   */
  m.def(
      "causal_conv1d_update(Tensor! x,"
      "Tensor! conv_state,"
      "Tensor! weight,"
      "Tensor? bias_,"
      "bool silu_activation,"
      "Tensor? cache_seqlens_,"
      "Tensor? conv_state_indices,"
      "int pad_slot_id) -> ()");
  m.impl("causal_conv1d_update", torch::kMUSA, &causal_conv1d_update);

  m.def(
      "causal_conv1d_fwd(Tensor! x, Tensor! weight,"
      "Tensor? bias_,"
      "Tensor!? conv_states,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "bool silu_activation,"
      "int pad_slot_id) -> ()");
  m.impl("causal_conv1d_fwd", torch::kMUSA, &causal_conv1d_fwd);
}

REGISTER_EXTENSION(common_ops)
