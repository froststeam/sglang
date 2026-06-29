# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only MiniMax-M3 (model_type=minimax_m3_vl) for SGLang / MUSA.

MiniMax-M3 fuses:
  * GPT-OSS-style feed-forward     : ``swigluoai`` clamped-swiglu (alpha/limit) + gemma RMSNorm (1+w)
  * DeepSeek-style sparse MoE       : sigmoid routing + e_score_correction_bias + routed_scaling + 1 shared expert
  * DeepSeek-V3.2-style attention   : per-head QK gemma-norm, partial RoPE, and a native sparse
                                      "lightning indexer" (the ``index_*`` projections)

Layers 0..(num_dense-1) are dense (``mlp``); the rest are MoE (``block_sparse_moe``),
selected by ``moe_layer_freq``.

PHASE 1 (this file): text decoder that LOADS (mxfp8) and GENERATES with FULL attention on every
layer. The sparse ``index_*`` indexer weights are skipped (deferred to Phase 2 / NSA backend), and
the vision tower + multimodal projector are skipped (text-only serving). The published architecture
``MiniMaxM3SparseForConditionalGeneration`` is a thin wrapper that builds + serves the text path.
"""

import logging
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    """MiniMax-M3: first `num_dense` layers are dense, the rest MoE (moe_layer_freq)."""
    freq = getattr(config, "moe_layer_freq", 1)
    if isinstance(freq, (list, tuple)):
        return bool(freq[layer_id]) if layer_id < len(freq) else True
    # int fallback: 1 => every layer MoE
    return (layer_id % max(int(freq), 1)) == 0


def _msa_runtime_enabled() -> bool:
    """Phase-2 MSA (native block-sparse attention) is OFF by default. When off,
    attention is identical to the validated Phase-1 full-attention path (the index
    branch is neither built, loaded, nor used).

    Enable by EITHER:
      * env ``SGLANG_MUSA_M3_MSA=1`` (works when the model process inherits it), OR
      * a marker file at ``/tmp/sglang_musa_m3_msa.on`` (override path via
        ``SGLANG_MUSA_M3_MSA_FILE``). The marker file is the reliable switch on the
        MUSA stack, where sglang **scheduler workers do not inherit shell env vars**,
        so an env-only gate would silently stay off in the model process.

    The MSA path uses the mate kernels (tilelang indexer + torch block-sparse attention).
    """
    import os

    if os.environ.get("SGLANG_MUSA_M3_MSA", "0").lower() not in ("0", "", "false", "no"):
        return True
    marker = os.environ.get("SGLANG_MUSA_M3_MSA_FILE", "/tmp/sglang_musa_m3_msa.on")
    try:
        return os.path.exists(marker)
    except Exception:  # noqa: BLE001
        return False


def _msa_single_seq_prefill(forward_batch: "ForwardBatch", num_tokens: int) -> bool:
    """MSA (this first integration) needs the FULL K/V of the sequence present in
    THIS forward — i.e. a single-sequence prefill with no prefix cache / chunking.
    Otherwise return False so the caller falls back to full attention (RadixAttention).
    The KV-cache-integrated MSA decode path is the remaining Phase-2e work."""
    fm = getattr(forward_batch, "forward_mode", None)
    try:
        if fm is None or not fm.is_extend():
            return False
    except Exception:  # noqa: BLE001
        return False
    if getattr(forward_batch, "batch_size", 1) != 1:
        return False
    prefix = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if prefix is not None and sum(int(x) for x in prefix) != 0:
        return False
    return True


def _msa_no_prefix_extend(forward_batch: "ForwardBatch") -> bool:
    """True for an extend/prefill forward where EVERY request is a fresh full-seq
    prefill (no prefix cache hit, no chunked prefill) — i.e. ``sum(extend_prefix_lens_cpu)
    == 0``. This is the regime where each request's full K/V lives contiguously in THIS
    forward, so we can slice per request and run the dense whole-seq block-sparse MSA on
    each. Covers both bs==1 (the single-seq case) and bs>1 (batched prefill). The
    chunked/prefix-prefill case (``sum != 0``) is excluded -> caller falls back to full
    attention (idx_k is still written unconditionally so decode history stays intact)."""
    fm = getattr(forward_batch, "forward_mode", None)
    try:
        if fm is None or not fm.is_extend():
            return False
    except Exception:  # noqa: BLE001
        return False
    prefix = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if prefix is not None and sum(int(x) for x in prefix) != 0:
        return False
    return True


def _fm_is_decode(forward_batch: "ForwardBatch") -> bool:
    """True for a pure decode forward (one query token per request)."""
    fm = getattr(forward_batch, "forward_mode", None)
    try:
        return fm is not None and fm.is_decode()
    except Exception:  # noqa: BLE001
        return False


def _is_sparse_attention_layer(config: PretrainedConfig, layer_id: int) -> bool:
    """M3 ``sparse_attention_config.sparse_attention_freq[layer_id]`` (1 = sparse).
    Sparse on layers 3..59, dense on 0..2."""
    scfg = getattr(config, "sparse_attention_config", None)
    if not scfg or not scfg.get("use_sparse_attention", False):
        return False
    freq = scfg.get("sparse_attention_freq")
    if isinstance(freq, (list, tuple)):
        return layer_id < len(freq) and bool(freq[layer_id])
    return False


def swiglu_oai(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """GPT-OSS / MiniMax-M3 ``swigluoai`` activation on a fused [.., 2*inter] tensor.

    gate = first half, up = second half (MergedColumnParallelLinear order).
        out = clamp(gate, max=limit) * sigmoid(alpha * clamp(gate, max=limit)) * (clamp(up, +-limit) + 1)
    """
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1.0)


class MiniMaxM3MLP(nn.Module):
    """Dense (layers 0..num_dense-1) and shared-expert MLP with swigluoai activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_alpha: float,
        swiglu_limit: float,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        act = swiglu_oai(gate_up, self.swiglu_alpha, self.swiglu_limit)
        out, _ = self.down_proj(act)
        return out


class MiniMaxM3SparseMoE(nn.Module):
    """Sigmoid-routed MoE (+ correction bias, routed scaling) with one shared expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"TP size {self.tp_size} > num_local_experts {config.num_local_experts}."
            )

        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = self._ebias_weight_loader
        else:
            self.e_score_correction_bias = None

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        swiglu_alpha = getattr(config, "swiglu_alpha", 1.702)
        swiglu_limit = getattr(config, "swiglu_limit", 7.0)
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_local_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            activation="silu",  # gemm1_alpha triggers the OAI swiglu in the kernel
            gemm1_alpha=swiglu_alpha,
            gemm1_clamp_limit=swiglu_limit,
            prefix=add_prefix("experts", prefix),
        )

        # routed_scaling is applied manually on the combined routed output below
        # (this fork's TopK asserts apply_routed_scaling_factor_on_output is unimplemented).
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            scoring_func=getattr(config, "scoring_func", "sigmoid"),
            correction_bias=self.e_score_correction_bias,
            routed_scaling_factor=1.0,
        )

        n_shared = getattr(config, "n_shared_experts", 0) or 0
        if n_shared > 0:
            shared_inter = getattr(
                config, "shared_intermediate_size", config.intermediate_size * n_shared
            )
            self.shared_experts = MiniMaxM3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_inter,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,  # single all-reduce after combining with routed experts
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

    @staticmethod
    def _ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_out = None
        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states)

        if hidden_states.shape[0] > 0:
            router_logits, _ = self.gate(hidden_states.to(torch.float32))
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(hidden_states, topk_output)
        if self.routed_scaling_factor != 1.0:
            final_hidden_states = final_hidden_states * self.routed_scaling_factor

        if shared_out is not None:
            final_hidden_states = final_hidden_states + shared_out

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM3Attention(nn.Module):
    """Full attention with per-head gemma QK-norm and partial RoPE.

    PHASE 1: the native sparse ``index_*`` indexer is NOT built/used (full attention).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.rope_theta, self.rope_scaling = get_rope_config(config)
        self.max_position_embeddings = getattr(
            config, "max_position_embeddings", 8192
        )
        self.rotary_dim = getattr(config, "rotary_dim", self.head_dim)

        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.qk_norm_type = getattr(config, "qk_norm_type", "per_head")

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )

        if self.use_qk_norm:
            if self.qk_norm_type != "per_head":
                raise ValueError(
                    f"MiniMaxM3 only supports per_head qk_norm, got {self.qk_norm_type}"
                )
            # gemma-style (1+w) per-head RMSNorm over head_dim; weight is replicated.
            self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        # ---- Phase-2 MSA index branch (gated; default OFF -> not built) ----
        # The native sparse-attention "lightning indexer": a small per-head Q/K
        # projection whose scores select the top-k KV blocks each query attends to.
        # Selector-only on M3 (value/output disabled). Built only on sparse layers
        # when SGLANG_MUSA_M3_MSA=1. See _msa_runtime_enabled().
        self.is_sparse_attention_layer = (
            _is_sparse_attention_layer(config, layer_id) and _msa_runtime_enabled()
        )
        if self.is_sparse_attention_layer:
            scfg = config.sparse_attention_config
            self.idx_num_heads = int(scfg["sparse_num_index_heads"])  # 4 (== num_kv_heads)
            self.idx_head_dim = int(scfg["sparse_index_dim"])         # 128
            self.msa_block_size = int(scfg["sparse_block_size"])      # 128 (== page size)
            self.msa_topk = int(scfg["sparse_topk_blocks"])           # 16
            # sparse_init_block names the sink block index (0) -> always 1 init block;
            # sparse_local_block is the count of trailing local blocks.
            self.msa_init_blocks = 1
            self.msa_local_blocks = int(scfg.get("sparse_local_block", 1))
            # index_q: hidden -> idx_num_heads*idx_head_dim ; index_k: hidden -> idx_head_dim
            # (replicated across TP: only 4 idx heads, not evenly shardable over tp=8).
            self.index_q_proj = ReplicatedLinear(
                self.hidden_size,
                self.idx_num_heads * self.idx_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("index_q_proj", prefix),
            )
            self.index_k_proj = ReplicatedLinear(
                self.hidden_size,
                self.idx_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("index_k_proj", prefix),
            )
            # gemma (1+w) RMSNorm over idx_head_dim; partial RoPE reuses self.rotary_emb
            # (idx_head_dim == head_dim == 128, same rotary_dim 64).
            self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
            self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
            # Model-owned per-layer idx_k aux cache: [kv_capacity, idx_head_dim] bf16,
            # lazily allocated on first forward (needs the runtime KV-pool slot count).
            # Keyed by the SAME physical slots (out_cache_loc / req_to_token) as the main
            # K/V so decode can gather the full idx_k history. NOT sglang's KV pool.
            self._idx_k_buffer = None

    def _per_head_norm(
        self, x: torch.Tensor, norm: GemmaRMSNorm, num_heads: int, head_dim: int = None
    ) -> torch.Tensor:
        hd = head_dim or self.head_dim
        x = x.view(-1, num_heads, hd)
        x = norm(x)
        return x.view(-1, num_heads * hd)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q = self._per_head_norm(q, self.q_norm, self.num_heads)
            k = self._per_head_norm(k, self.k_norm, self.num_kv_heads)
        q, k = self.rotary_emb(positions, q, k)
        if self.is_sparse_attention_layer and _fm_is_decode(forward_batch):
            # Sparse-layer DECODE fast path. The full RadixAttention output would be
            # DISCARDED (MSA overrides it), and at long context that full O(S) attention
            # is the dominant decode cost on all 57 sparse layers. So write K/V into the
            # paged cache directly (what RadixAttention would have done) and SKIP the full
            # attention — MSA attends only the ~<=18 selected blocks (cost ~constant in
            # context length). If MSA is unavailable, fall back to full attention (K/V is
            # already stored, so save_kv_cache=False to avoid a redundant write).
            self._store_kv_only(k, v, forward_batch)
            msa_output = self._msa_attention(
                positions, hidden_states, q, k, v, forward_batch
            )
            if msa_output is not None:
                attn_output = msa_output
            else:
                attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=False)
        else:
            # Non-sparse layers and ALL prefill: RadixAttention (stores K/V + full attn).
            # For sparse prefill, override with the block-sparse output when available
            # (the single-/batched-no-prefix case; chunked/prefix prefill keeps full attn).
            attn_output = self.attn(q, k, v, forward_batch)
            if self.is_sparse_attention_layer:
                msa_output = self._msa_attention(
                    positions, hidden_states, q, k, v, forward_batch
                )
                if msa_output is not None:
                    attn_output = msa_output
        output, _ = self.o_proj(attn_output)
        return output

    def _msa_attention(
        self, positions, hidden_states, q, k, v, forward_batch
    ):
        """Phase-2 MSA: index branch -> top-k block selection -> block-sparse attention.
        Returns the [T, num_heads*head_dim] output, or None to fall back to full attn.

        The index branch (idx_q/idx_k proj + per-head gemma-norm + partial RoPE) ALWAYS
        runs for sparse layers, and idx_k is written UNCONDITIONALLY to the model-owned
        aux cache (keyed by out_cache_loc, same slots as the main K/V) — even when this
        forward's output falls back to full attention — so decode has the full idx_k
        history. MSA output paths: no-prefix prefill (single-seq AND batched bs>1: dense
        whole-seq block-sparse, per-request slice) and decode (per-request gather of cached
        K/V/idx_k -> q=1 block-sparse). The chunked/prefix-prefill case (sum(prefix)!=0)
        returns None -> RadixAttention full attn. At ctx <= ~2048 selection covers all
        blocks so MSA == full attention (the correctness gate); the exact TP idx-head<->
        kv-head map is now applied per rank (see _select_local_index_heads)."""
        nt = q.shape[0]
        fm = getattr(forward_batch, "forward_mode", None)
        try:
            is_decode = fm is not None and fm.is_decode()
        except Exception:  # noqa: BLE001
            is_decode = False
        # Prefill MSA fires for any no-prefix extend (bs==1 single-seq OR bs>1 batched);
        # each request's full K/V is contiguous in this forward. Chunked/prefix prefill is
        # excluded -> history-only write + fall back.
        is_prefill = _msa_no_prefix_extend(forward_batch)
        # If we won't use an MSA output AND the idx_k history is already complete for this
        # forward's own tokens (it isn't — decode needs the WRITE), skip cheaply. We must
        # still WRITE idx_k whenever we can (any mode) so decode has history; but if no MSA
        # output path applies and we cannot import the kernels, the write alone is enough.
        if not (is_decode or is_prefill):
            # Chunked/prefix prefill: still WRITE idx_k for history, then fall back.
            self._write_idx_k(positions, hidden_states, nt, forward_batch)
            return None

        # index branch: project, per-head gemma norm, partial RoPE (reuse rotary_emb).
        idx_q, idx_k = self._index_branch(positions, hidden_states, nt)
        # UNCONDITIONAL idx_k write (keyed by out_cache_loc) BEFORE the branch decision.
        self._store_idx_k(idx_k, forward_batch)

        idx_q_local = self._select_local_index_heads(idx_q)  # [T, num_kv_heads, idx_head_dim]

        # Separate indexer / attention backends, each selected by
        # SGLANG_MUSA_M3_MSA_{INDEXER,ATTN}_BACKEND (umbrella SGLANG_MUSA_M3_MSA_BACKEND).
        # Default = "triton" (the in-sglang Triton MSA): mate v0.2.2 ships no MSA kernels
        # and sglang workers don't inherit shell env, so the default (not an env) must
        # select the available path. Set =torch for the dense fp32 reference, =tilelang
        # for the mate tilelang kernels.
        import os as _os
        _umbrella = _os.environ.get("SGLANG_MUSA_M3_MSA_BACKEND")
        _idx_be = _os.environ.get("SGLANG_MUSA_M3_MSA_INDEXER_BACKEND", _umbrella or "triton")
        _attn_be = _os.environ.get("SGLANG_MUSA_M3_MSA_ATTN_BACKEND", _umbrella or "triton")

        if is_decode:
            out = self._msa_decode(q, idx_q_local, forward_batch)
            if out is None:
                return None
            self._log_msa_fired("decode", nt)
            return out

        # No-prefix prefill (dense whole-seq block-sparse). bs==1 -> one request spanning
        # all nt tokens; bs>1 -> per-request contiguous slices of the flattened [nt, ...]
        # tensors. The same per-slice MSA core runs each request independently (correct
        # because there is no cross-request attention and no prefix to gather).
        try:
            # backend="triton" sources the kernels from the in-sglang Triton MSA module
            # (mate-independent); else the mate (torch/tilelang) path. Indexer + attention
            # backends are independent (SGLANG_MUSA_M3_MSA_{INDEXER,ATTN}_BACKEND).
            if _idx_be == "triton":
                from sglang.srt.layers.attention.msa_triton import msa_block_topk_indices
            else:
                from mate.msa_indexer import msa_block_topk_indices
            if _attn_be == "triton":
                from sglang.srt.layers.attention.msa_triton import (
                    msa_block_sparse_attention,
                )
            else:
                from mate.msa_attention import msa_block_sparse_attention
        except Exception:  # noqa: BLE001 — MSA kernels unavailable -> full attn
            return None

        starts, lens = self._extend_slices(forward_batch, nt)
        if starts is None:
            return None  # could not determine per-request layout -> full-attn fallback

        out = torch.empty(
            (nt, self.num_heads * self.head_dim), dtype=q.dtype, device=q.device
        )
        for s, L in zip(starts, lens):
            e = s + L
            out[s:e] = self._msa_prefill_slice(
                q[s:e], k[s:e], v[s:e], idx_q_local[s:e], idx_k[s:e],
                msa_block_topk_indices, msa_block_sparse_attention, _idx_be, _attn_be,
            )
        self._log_msa_fired("prefill" if len(starts) == 1 else "batched_prefill", nt)
        return out

    def _extend_slices(self, forward_batch, nt):
        """Per-request (start, length) slices into this forward's flattened [nt, ...]
        token axis, for a no-prefix extend. Uses ``extend_start_loc`` (start offsets) and
        ``extend_seq_lens`` / ``extend_seq_lens_cpu`` (per-request token counts). Returns
        (starts, lens) as python int lists, or (None, None) if the layout is unavailable.

        forward_batch fields (forward_batch_info.py):
          * extend_seq_lens      (L334) Tensor [bs] — per-request extend token counts
          * extend_seq_lens_cpu  (L338) List[int]   — cpu copy (preferred, avoids D2H)
          * extend_start_loc     (L336) Tensor [bs] — per-request start offset in the
                                  flattened token axis (cumsum of extend_seq_lens)."""
        seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if seq_lens_cpu is not None:
            lens = [int(x) for x in seq_lens_cpu]
        elif seq_lens is not None:
            lens = [int(x) for x in seq_lens.tolist()]
        else:
            return None, None

        start_loc = getattr(forward_batch, "extend_start_loc", None)
        if start_loc is not None:
            starts = [int(x) for x in start_loc.tolist()]
        else:
            # derive contiguous starts from the lengths (prefix sum)
            starts = []
            acc = 0
            for L in lens:
                starts.append(acc)
                acc += L
        # sanity: the slices must tile the whole [nt] token axis exactly.
        if sum(lens) != nt or (starts and starts[0] != 0):
            return None, None
        return starts, lens

    def _msa_prefill_slice(
        self, q, k, v, idx_q_local, idx_k,
        msa_block_topk_indices, msa_block_sparse_attention, idx_be, attn_be,
    ):
        """Dense whole-seq block-sparse MSA for ONE request's contiguous tokens.
        q/k/v are [L, *]; idx_q_local [L, num_kv_heads, D]; idx_k [L, D]. Returns
        [L, num_heads*head_dim]. Identical math to the single-seq prefill path —
        batched prefill is just this run independently per request."""
        L = q.shape[0]
        q3 = q.reshape(L, self.num_heads, self.head_dim)
        k3 = k.reshape(L, self.num_kv_heads, self.head_dim)
        v3 = v.reshape(L, self.num_kv_heads, self.head_dim)
        indices, counts = msa_block_topk_indices(
            idx_q_local, idx_k,
            block_size=self.msa_block_size, topk=self.msa_topk,
            init_blocks=self.msa_init_blocks, local_blocks=self.msa_local_blocks,
            backend=idx_be,
        )
        out = msa_block_sparse_attention(
            q3, k3, v3, indices, counts,
            block_size=self.msa_block_size, scale=self.scaling, backend=attn_be,
        )
        return out.reshape(L, self.num_heads * self.head_dim).to(q.dtype)

    def _index_branch(self, positions, hidden_states, nt):
        """Run the index branch on `nt` tokens -> (idx_q [nt,Hk,D], idx_k [nt,D]).

        Identical for prefill and decode (at decode nt = 1 token/request)."""
        idx_q, _ = self.index_q_proj(hidden_states)  # [T, idx_num_heads*idx_head_dim]
        idx_k, _ = self.index_k_proj(hidden_states)  # [T, idx_head_dim]
        idx_q = self._per_head_norm(
            idx_q, self.index_q_norm, self.idx_num_heads, self.idx_head_dim
        )
        idx_k = self._per_head_norm(idx_k, self.index_k_norm, 1, self.idx_head_dim)
        idx_q, idx_k = self.rotary_emb(positions, idx_q, idx_k)
        idx_q = idx_q.view(nt, self.idx_num_heads, self.idx_head_dim)
        idx_k = idx_k.view(nt, self.idx_head_dim)
        return idx_q, idx_k

    def _write_idx_k(self, positions, hidden_states, nt, forward_batch):
        """Compute + store idx_k only (no MSA output) — the fallback-mode history write."""
        _, idx_k = self._index_branch(positions, hidden_states, nt)
        self._store_idx_k(idx_k, forward_batch)

    def _ensure_idx_k_buffer(self, forward_batch, ref_dtype, device):
        """Lazily allocate the per-layer idx_k aux cache [kv_capacity, idx_head_dim] bf16,
        sized to the SAME slot space as the main K/V (so out_cache_loc / req_to_token slots
        index into it identically). Reused across forwards."""
        if self._idx_k_buffer is not None:
            return
        kv_capacity = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn.layer_id
        ).shape[0]
        self._idx_k_buffer = torch.zeros(
            (kv_capacity, self.idx_head_dim), dtype=torch.bfloat16, device=device
        )

    def _store_idx_k(self, idx_k, forward_batch):
        """Write this forward's tokens' idx_k into the aux cache at their physical KV
        slots (out_cache_loc), same keying as set_kv_buffer."""
        self._ensure_idx_k_buffer(forward_batch, idx_k.dtype, idx_k.device)
        loc = forward_batch.out_cache_loc
        self._idx_k_buffer[loc] = idx_k.to(torch.bfloat16)

    def _store_kv_only(self, k, v, forward_batch):
        """Write this step's K/V into the paged cache WITHOUT running the full attention
        (used on sparse decode layers, where the full-attention output is discarded).
        Mirrors the fa3 backend's decode K/V store (set_kv_buffer at out_cache_loc),
        including the reshape RadixAttention.forward applies and the k/v scales. Capturable
        (a device scatter), so the sparse decode forward stays cuda-graph-capturable."""
        layer = self.attn  # the RadixAttention for this layer (owns layer_id + kv scales)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
        )

    def _msa_decode(self, q, idx_q_local, forward_batch):
        """Decode MSA: per request, gather cached K/V/idx_k for the full seq and run the
        q=1 block-sparse decode kernel. Returns [bs, num_heads*head_dim] or None on
        kernel-import failure. q is [bs, q_size]; idx_q_local is [bs, num_kv_heads, D]."""
        import os as _os
        _attn_be = _os.environ.get(
            "SGLANG_MUSA_M3_MSA_ATTN_BACKEND",
            _os.environ.get("SGLANG_MUSA_M3_MSA_BACKEND") or "triton",
        )
        if self._idx_k_buffer is None:  # no idx_k history yet -> full-attn fallback
            return None
        bs = forward_batch.batch_size
        layer_id = self.attn.layer_id
        K_all = forward_batch.token_to_kv_pool.get_key_buffer(layer_id)    # [cap,Hk,D]
        V_all = forward_batch.token_to_kv_pool.get_value_buffer(layer_id)  # [cap,Hk,D]
        req_to_token = forward_batch.req_to_token_pool.req_to_token        # [Nreq, max_ctx]
        q3 = q.view(bs, self.num_heads, self.head_dim)                     # [bs,Hq,D]

        # Decode-kernel selection:
        #  - cuda-graph CAPTURE (get_is_capture_mode()==True): use the BATCHED + PAGED Triton
        #    path (static shapes, all-device-tensor inputs, fixed grid, no host syncs) so the
        #    whole decode forward is capturable. Real serving then REPLAYS these captured
        #    kernels — this Python does NOT execute on replay, so the per-request branch below
        #    is never hit under a captured graph.
        #  - eager (cuda-graph disabled, or a batch size not captured): the paged fn is heavy
        #    run LIVE (an NB-wide block-score grid + a topk-iteration selection, x57 sparse
        #    layers per token) -> use the lightweight per-request loop. mate/torch path too.
        _capturing = False
        if _attn_be == "triton":
            try:
                from sglang.srt.model_executor.cuda_graph_runner import (
                    get_is_capture_mode,
                )
                _capturing = bool(get_is_capture_mode())
            except Exception:  # noqa: BLE001
                _capturing = False
        if _attn_be == "triton" and _capturing:
            try:
                from sglang.srt.layers.attention.msa_triton import (
                    msa_decode_attention_paged,
                )
            except Exception:  # noqa: BLE001 — kernel unavailable -> full attn fallback
                return None
            out = msa_decode_attention_paged(
                q3, idx_q_local, K_all, V_all, self._idx_k_buffer,
                req_to_token, forward_batch.req_pool_indices, forward_batch.seq_lens,
                block_size=self.msa_block_size, topk=self.msa_topk,
                init_blocks=self.msa_init_blocks, local_blocks=self.msa_local_blocks,
                scale=self.scaling,
            )
            return out.reshape(bs, self.num_heads * self.head_dim).to(q.dtype)

        # Per-request loop (eager). Uses host syncs (int(...)) + dynamic gathers, so it is
        # NOT cuda-graph-capturable — which is fine: it never runs under a captured/replayed
        # graph (capture takes the paged branch; replay skips Python entirely).
        try:
            if _attn_be == "triton":
                from sglang.srt.layers.attention.msa_triton import msa_decode_attention
            else:
                from mate.msa_attention import msa_decode_attention
        except Exception:  # noqa: BLE001
            return None
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        out = torch.empty(
            (bs, self.num_heads, self.head_dim), dtype=q.dtype, device=q.device
        )
        for i in range(bs):
            req = int(forward_batch.req_pool_indices[i])
            if seq_lens_cpu is not None:
                slen = int(seq_lens_cpu[i])
            else:
                slen = int(forward_batch.seq_lens[i])
            slots = req_to_token[req, :slen]                  # [slen] physical KV slots
            K_seq = K_all[slots]                              # [slen,Hk,D]
            V_seq = V_all[slots]                              # [slen,Hk,D]
            idxk_seq = self._idx_k_buffer[slots]              # [slen,D]
            out[i] = msa_decode_attention(
                q3[i], idx_q_local[i], K_seq, V_seq, idxk_seq,
                block_size=self.msa_block_size, topk=self.msa_topk,
                init_blocks=self.msa_init_blocks, local_blocks=self.msa_local_blocks,
                scale=self.scaling,
            )
        return out.reshape(bs, self.num_heads * self.head_dim).to(q.dtype)

    def _log_msa_fired(self, mode: str, nt: int):
        if not getattr(self, "_msa_fired_logged", False):
            logger.info(
                "[MSA] block-sparse attention FIRED (mode=%s, layer=%s, T=%d, kv_heads=%d, "
                "topk=%d, block=%d) — index branch + indexer + block-sparse attn active",
                mode, getattr(self.attn, "layer_id", "?"), nt, self.num_kv_heads,
                self.msa_topk, self.msa_block_size,
            )
            self._msa_fired_logged = True

    def _select_local_index_heads(self, idx_q: torch.Tensor) -> torch.Tensor:
        """Pick the idx heads matching THIS rank's local kv head(s). idx_q is
        [T, idx_num_heads, idx_head_dim] and is REPLICATED on every rank (the index
        projections are ReplicatedLinear: all `idx_num_heads` global index heads are
        present on each rank). The indexer/attention GQA grouping uses `num_kv_heads`,
        so we must hand it exactly THIS rank's kv head(s)' matching index head(s) —
        otherwise selection is coherent-but-wrong on ranks whose kv head isn't head 0.

        The M3 index head <-> kv head correspondence is 1:1 by global head id
        (idx_num_heads == total_num_kv_heads == 4). Two TP regimes:

        * SHARDED (total_num_kv_heads >= tp_size): this rank owns the contiguous global
          kv heads [tp_rank*num_kv_heads : (tp_rank+1)*num_kv_heads] -> select those
          index heads (`num_kv_heads` of them).
        * REPLICATED (total_num_kv_heads < tp_size, the M3 @ tp8 case: 4 < 8): each
          global kv head is replicated across `tp_size // total_num_kv_heads` ranks, so
          num_kv_heads == 1 and this rank's single global kv head is
          `tp_rank // (tp_size // total_num_kv_heads)` -> select that one index head.

        The selected index-head count is asserted == num_kv_heads (the indexer's Hk)."""
        if self.total_num_kv_heads >= self.tp_size:
            lo = self.tp_rank * self.num_kv_heads
            hi = lo + self.num_kv_heads
            sel = idx_q[:, lo:hi, :]
        else:
            ranks_per_kv_head = self.tp_size // self.total_num_kv_heads
            kvh = self.tp_rank // ranks_per_kv_head  # this rank's single global kv head
            sel = idx_q[:, kvh : kvh + 1, :]
        assert sel.shape[1] == self.num_kv_heads, (
            sel.shape,
            self.num_kv_heads,
            self.tp_rank,
            self.tp_size,
            self.total_num_kv_heads,
        )
        return sel.contiguous()


class MiniMaxM3DecoderLayer(nn.Module):
    """Pre-norm residual layer (gemma norms); dense MLP or MoE by ``moe_layer_freq``."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = MiniMaxM3Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.is_moe = _is_moe_layer(config, layer_id)
        if self.is_moe:
            self.block_sparse_moe = MiniMaxM3SparseMoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("block_sparse_moe", prefix),
            )
        else:
            self.mlp = MiniMaxM3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=getattr(
                    config, "dense_intermediate_size", config.intermediate_size
                ),
                swiglu_alpha=getattr(config, "swiglu_alpha", 1.702),
                swiglu_limit=getattr(config, "swiglu_limit", 7.0),
                quant_config=quant_config,
                reduce_results=True,
                prefix=add_prefix("mlp", prefix),
            )

        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        if self.is_moe:
            hidden_states = self.block_sparse_moe(hidden_states, forward_batch)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiniMaxM3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        def layer_fn(idx, prefix: str) -> nn.Module:
            return MiniMaxM3DecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            layer_fn,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # EAGLE3: decoder-layer indices whose residual-stream *input* is captured
        # and concatenated as the speculative draft's auxiliary hidden input.
        # Populated by MiniMaxM3SparseForCausalLM.set_eagle3_layers_to_capture().
        self.layers_to_capture: list[int] = []

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.get_input_embeddings(input_ids)
                if input_embeds is None
                else input_embeds
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                # Residual stream entering layer i (== output of layer i-1). EAGLE3
                # draft consumes the concatenation of these taps. Matches the canonical
                # llama.py / minimax_m2.py capture convention.
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual = self.layers[i](
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is not None:
                hidden_states, _ = self.norm(hidden_states, residual)
            else:
                hidden_states = self.norm(hidden_states)

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class MiniMaxM3SparseForCausalLM(nn.Module):
    """Text causal-LM backbone of MiniMax-M3 (consumes ``config.text_config``)."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Accept either the top-level VL config or the text sub-config directly, so
        # this class works both standalone (architectures override) and under the
        # MiniMaxM3SparseForConditionalGeneration wrapper.
        config = getattr(config, "text_config", config)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = MiniMaxM3Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)

        # For EAGLE3 speculative decoding: when True, forward returns auxiliary
        # hidden states (tapped intermediate layers) for the draft model.
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        """Configure which decoder layers feed the EAGLE3 draft (target-side hook).

        Called by ModelRunner.init_aux_hidden_state_capture(). The draft model
        (Inferact/MiniMax-M3-EAGLE3) ships no ``eagle_aux_hidden_state_layer_ids``
        in its config, so ``layer_ids`` arrives as ``None`` and we fall back to the
        canonical EAGLE3 default [2, num//2, num-3] == [2, 30, 57] for M3's 60
        layers, matching the taps the draft was trained against.
        """
        if not self.pp_group.is_last_rank:
            return
        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def get_embed_and_head(self):
        """EAGLE3 weight-sharing: hand the draft the target's embed + LM head."""
        return self.model.embed_tokens.weight, self.lm_head.weight

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            # VL checkpoint: the text weights live under language_model.*; the vision
            # tower + multimodal projector are skipped (PHASE 1 text-only).
            if name.startswith("vision_tower.") or name.startswith(
                "multi_modal_projector."
            ):
                continue
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            if "rotary_emb.inv_freq" in name:
                continue
            # Native sparse-attention indexer (index_q/k_proj/_norm). Phase-1 (MSA off):
            # skip — the branch isn't built. Phase-2 (SGLANG_MUSA_M3_MSA=1): load directly
            # into the index branch params, BYPASSING the qkv stacked mapping (whose
            # "q_proj"/"k_proj" weight_names would false-match inside "index_q_proj").
            if ".self_attn.index_" in name:
                if _msa_runtime_enabled() and name in params_dict:
                    param = params_dict[name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
                    loaded_params.add(name)
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            _is_kv_scale = name.endswith(".k_scale") or name.endswith(".v_scale")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if _is_kv_scale:
                    continue
                # experts handled below; don't let gate_proj/up_proj inside experts
                # be remapped here.
                if ("block_sparse_moe.experts." in name) and name not in params_dict:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name.endswith(".bias") or new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(new_name)
                break
            else:
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    new_name = name.replace(weight_name, param_name)
                    if new_name not in params_dict:
                        continue
                    param = params_dict[new_name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        new_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(new_name)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    remapped = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped is None:
                        continue
                    name = remapped
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        self._interleave_moe_w13_for_oai_swiglu()
        return loaded_params

    def _interleave_moe_w13_for_oai_swiglu(self) -> None:
        """Reorder fused MoE w13 rows from concatenated [gate; up] to interleaved.

        MiniMax-M3 ships separate w1(gate)/w3(up) experts, so FusedMoE stacks w13 as
        concatenated halves. But the swigluoai path (``gemm1_alpha`` set) in the triton
        fused_moe reads gate/up INTERLEAVED (``x[...,::2], x[...,1::2]``, matching the
        gpt-oss fused gate_up layout). Interleave w13 (+ its ue8m0 scale) so the
        activation separates gate/up correctly. Run once, after load.
        """
        # MiniMax-M3 ships separate w1(gate)/w3(up) experts, so FusedMoE stacks w13 as
        # concatenated halves. But the swigluoai path (gemm1_alpha) in this fork's triton
        # fused_moe reads gate/up INTERLEAVED (x[...,::2], x[...,1::2], matching gpt-oss's
        # fused gate_up). Interleave w13 (+ its ue8m0 scale) so the activation separates
        # gate/up correctly — equivalent to upstream PR #27944's interleaved=False.
        layers = getattr(self.model, "layers", None)
        if layers is None:
            return
        start = getattr(self.model, "start_layer", 0)
        end = getattr(self.model, "end_layer", len(layers))
        for i in range(start, end):
            moe = getattr(layers[i], "block_sparse_moe", None)
            if moe is None:
                continue
            experts = getattr(moe, "experts", None)
            if experts is None:
                continue
            for wname in ("w13_weight", "w13_weight_scale_inv", "w13_weight_scale"):
                w = getattr(experts, wname, None)
                if w is None or w.dim() != 3 or (w.shape[1] % 2) != 0:
                    continue
                two_n = w.shape[1]
                n = two_n // 2
                # MUSA-safe interleave (index_select unsupported on fp8): stack halves.
                gate = w.data[:, :n, :]
                up = w.data[:, n:, :]
                w.data = (
                    torch.stack([gate, up], dim=2)
                    .reshape(w.shape[0], two_n, w.shape[2])
                    .contiguous()
                )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )


class MiniMaxM3SparseForConditionalGeneration(nn.Module):
    """Published architecture: VL wrapper around the MiniMax-M3 text model.

    PHASE 1: text-only serving. The vision tower + multimodal projector weights are
    skipped at load time; only the ``language_model.*`` weights are consumed. Multimodal
    inputs are not yet supported.
    """

    packed_modules_mapping = MiniMaxM3SparseForCausalLM.packed_modules_mapping

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        text_config = getattr(config, "text_config", config)
        # Carry the quant ignore-list (lm_head/embed/vision/projector/gate) which lives
        # on the top-level config so the quant method skips those modules.
        self.language_model = MiniMaxM3SparseForCausalLM(
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    # EAGLE3 target-side hooks — delegate to the text backbone. The served entry is
    # this VL wrapper, so ModelRunner / EagleWorker reach these on the top-level model.
    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        return self.language_model.set_eagle3_layers_to_capture(layer_ids)

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # The CausalLM strips the ``language_model.`` prefix and skips vision/projector.
        loaded = self.language_model.load_weights(weights)
        return {f"language_model.{n}" for n in loaded}

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        return MiniMaxM3SparseForCausalLM.get_model_config_for_expert_location(
            text_config
        )


EntryClass = [
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
]
