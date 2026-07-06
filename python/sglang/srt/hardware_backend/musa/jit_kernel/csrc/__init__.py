"""MUSA TVM-FFI/C++ JIT kernels."""

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.activation import (
    act_and_mul,
    act_and_mul_masked,
    act_and_mul_masked_post_quant_fwd,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemm import musa_fused_moe_gemv
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.moe import (
    deep_gemm_contig_preprocess,
    deep_gemm_ep_preprocess,
    fused_share_gate_sigmoid_mul,
    moe_align_block_size,
    moe_sum_reduce,
    topk_sigmoid,
    topk_softmax,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.norm import (
    fused_add_rmsnorm,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.quant import (
    per_token_group_quant_8bit,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.rope import rotary_embedding

__all__ = [
    "fused_add_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "deep_gemm_contig_preprocess",
    "deep_gemm_ep_preprocess",
    "fused_share_gate_sigmoid_mul",
    "musa_fused_moe_gemv",
    "moe_align_block_size",
    "act_and_mul",
    "act_and_mul_masked",
    "act_and_mul_masked_post_quant_fwd",
    "moe_sum_reduce",
    "per_token_group_quant_8bit",
    "rmsnorm",
    "rotary_embedding",
    "topk_sigmoid",
    "topk_softmax",
]
