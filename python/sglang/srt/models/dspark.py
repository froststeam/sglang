from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _require_unquantized_vocab_layer(layer: nn.Module, name: str) -> None:
    quant_method = getattr(layer, "quant_method", None)
    if not isinstance(quant_method, UnquantizedEmbeddingMethod):
        raise RuntimeError(
            "DSPARK replicated vocab weights require unquantized weights, but "
            f"{name} uses {type(quant_method).__name__}. Use TP=1 or an "
            "unquantized DSPARK checkpoint."
        )


def gather_and_crop_vocab(
    local_logits: torch.Tensor, lm_head: nn.Module
) -> torch.Tensor:
    full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    return full_logits[..., : int(lm_head.org_vocab_size)]


class DSparkAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_tensor_model_parallel_world_size())
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(getattr(config, "num_key_value_heads", total_num_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta = float(getattr(config, "rope_theta", 1000000))
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.scaling = head_dim**-0.5
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DSparkMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
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
            prefix=add_prefix("down_proj", prefix),
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DSpark activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch = None) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, forward_batch=forward_batch)
        return x


class DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.self_attn = DSparkAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = DSparkMLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

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
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        return hidden_states, residual


class DSparkVanillaMarkov(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {self.markov_rank}.")
        self.markov_w1 = VocabParallelEmbedding(
            self.vocab_size,
            self.markov_rank,
            quant_config=quant_config,
            prefix=add_prefix("markov_w1", prefix),
        )
        self.markov_w2 = ParallelLMHead(
            self.vocab_size,
            self.markov_rank,
            quant_config=quant_config,
            prefix=add_prefix("markov_w2", prefix),
        )

    def get_prev_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(tokens.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return F.linear(latent_states, self.markov_w2.weight)

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        return logits + self.project_bias(self.get_prev_embeddings(tokens))


class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


_DSPARK_SKIPPED_WEIGHT_PREFIXES = (
    "embed_tokens.",
    "lm_head.",
    "rotary_emb.",
)


class DSparkDraftModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        if draft_config.target_layer_ids is None:
            raise ValueError(
                "config.target_layer_ids or dflash_config.target_layer_ids "
                "must be provided for DSparkDraftModel."
            )
        if draft_config.mask_token_id is None:
            raise ValueError(
                "config.mask_token_id or dflash_config.mask_token_id "
                "must be provided for DSparkDraftModel."
            )
        if not hasattr(config, "markov_rank"):
            raise ValueError("config.markov_rank must be provided for DSparkDraftModel.")
        if int(config.markov_rank) > 0 and not hasattr(config, "markov_head_type"):
            raise ValueError(
                "config.markov_head_type must be provided when markov_rank > 0."
            )
        use_confidence_head = str(envs.SGLANG_RAGGED_VERIFY_MODE.get()) != "static"
        enable_confidence_head = use_confidence_head and bool(
            getattr(config, "enable_confidence_head", False)
        )
        if enable_confidence_head and not hasattr(
            config, "confidence_head_with_markov"
        ):
            raise ValueError(
                "config.confidence_head_with_markov must be provided when enable_confidence_head is true."
            )

        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.target_layer_ids = [
            int(layer_id) for layer_id in draft_config.target_layer_ids
        ]
        block_size = draft_config.resolve_block_size()
        if block_size is None:
            raise ValueError("config.block_size must be provided for DSparkDraftModel.")
        self.block_size = int(block_size)
        self.mask_token_id = int(draft_config.mask_token_id)
        self.start_layer = 0
        self.end_layer = int(config.num_hidden_layers)

        self.layers = nn.ModuleList(
            [
                DSparkDecoderLayer(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * hidden_size,
            hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self._shared_embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self.replicated_lm_head_weight: Optional[torch.Tensor] = None
        self.replicated_markov_w1_weight: Optional[torch.Tensor] = None
        self.replicated_markov_w2_weight: Optional[torch.Tensor] = None

        self.markov_head = None
        markov_rank = int(config.markov_rank)
        if markov_rank > 0:
            markov_head_type = str(config.markov_head_type).lower()
            if markov_head_type != "vanilla":
                raise ValueError(
                    f"Unsupported DSpark markov_head_type={markov_head_type!r}. Only 'vanilla' is supported for now."
                )
            self.markov_head = DSparkVanillaMarkov(
                vocab_size=int(config.vocab_size),
                markov_rank=markov_rank,
                quant_config=quant_config,
                prefix=add_prefix("markov_head", prefix),
            )

        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = False
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = hidden_size
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "DSpark confidence_head_with_markov requires markov_head."
                    )
                input_dim += markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

    def attach_shared_modules(self, *, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        embed_weight = getattr(embed_tokens, "weight", None)
        lm_head_weight = getattr(lm_head, "weight", None)
        if not isinstance(embed_weight, torch.Tensor):
            raise ValueError("DSPARK requires the target embedding to expose weight.")
        if not isinstance(lm_head_weight, torch.Tensor):
            raise ValueError("DSPARK requires the target lm_head to expose weight.")

        hidden_size = int(self.config.hidden_size)
        if int(embed_weight.shape[-1]) != hidden_size:
            raise ValueError(
                "DSPARK target embedding hidden size mismatch: "
                f"expected {hidden_size}, got {tuple(embed_weight.shape)}."
            )
        if int(lm_head_weight.shape[-1]) != hidden_size:
            raise ValueError(
                "DSPARK target lm_head hidden size mismatch: "
                f"expected {hidden_size}, got {tuple(lm_head_weight.shape)}."
            )
        target_vocab_size = int(
            getattr(lm_head, "org_vocab_size", self.config.vocab_size)
        )
        if target_vocab_size != int(self.config.vocab_size):
            raise ValueError(
                "DSPARK target/draft vocab size mismatch: "
                f"target={target_vocab_size}, draft={int(self.config.vocab_size)}."
            )
        object.__setattr__(self, "_shared_embed_tokens", embed_tokens)
        object.__setattr__(self, "lm_head", lm_head)

    @torch.no_grad()
    def configure_replicated_vocab_weights(self, *, replicate_lm_head: bool) -> None:
        """Replicate unquantized vocab weights across TP ranks.

        The replicas are startup snapshots, not registered buffers. They increase
        per-rank model memory and are not refreshed by online target/draft weight
        updates; workers must be restarted after such an update.
        """
        if get_tensor_model_parallel_world_size() == 1:
            return
        if self.lm_head is None:
            raise RuntimeError("DSPARK target lm_head is not attached.")

        if replicate_lm_head:
            _require_unquantized_vocab_layer(self.lm_head, "target lm_head")
        if self.markov_head is not None:
            _require_unquantized_vocab_layer(
                self.markov_head.markov_w1, "Markov embedding"
            )
            _require_unquantized_vocab_layer(
                self.markov_head.markov_w2, "Markov projection"
            )

        vocab_size = int(self.config.vocab_size)
        if replicate_lm_head and self.replicated_lm_head_weight is None:
            self.replicated_lm_head_weight = tensor_model_parallel_all_gather(
                self.lm_head.weight, dim=0
            )[:vocab_size].contiguous()
        if (
            self.markov_head is not None
            and self.replicated_markov_w1_weight is None
        ):
            self.replicated_markov_w1_weight = tensor_model_parallel_all_gather(
                self.markov_head.markov_w1.weight, dim=0
            )[:vocab_size].contiguous()
            self.replicated_markov_w2_weight = tensor_model_parallel_all_gather(
                self.markov_head.markov_w2.weight, dim=0
            )[:vocab_size].contiguous()
        replicated_weights = (
            self.replicated_lm_head_weight,
            self.replicated_markov_w1_weight,
            self.replicated_markov_w2_weight,
        )
        replicated_mib = sum(
            weight.numel() * weight.element_size()
            for weight in replicated_weights
            if weight is not None
        ) / (1024**2)
        logger.warning(
            "DSPARK replicated vocab weights configured: lm_head=%s markov=%s, "
            "extra_weight_memory=%.1f MiB/rank. These replicas are startup "
            "snapshots; restart workers after online target/draft weight updates.",
            (
                tuple(self.replicated_lm_head_weight.shape)
                if self.replicated_lm_head_weight is not None
                else False
            ),
            self.replicated_markov_w1_weight is not None,
            replicated_mib,
        )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DSPARK target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}], got shape={tuple(target_hidden.shape)}."
            )
        return self.hidden_norm(self.fc(target_hidden))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise ValueError(
                "DSPARK dense draft requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        weight = (
            self.replicated_lm_head_weight
            if self.replicated_lm_head_weight is not None
            else self.lm_head.weight
        )
        if hidden_states.dtype != weight.dtype:
            hidden_states = hidden_states.to(weight.dtype)
        return F.linear(hidden_states, weight)

    def gather_vocab_logits(self, local_logits: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise ValueError("DSPARK target lm_head is not attached.")
        return gather_and_crop_vocab(local_logits, self.lm_head)

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.markov_head is None:
            return logits
        if self.replicated_markov_w1_weight is not None:
            latent = F.embedding(prev_tokens.long(), self.replicated_markov_w1_weight)
            return logits + F.linear(latent, self.replicated_markov_w2_weight)
        return self.markov_head.apply_step_logits(
            logits,
            tokens=prev_tokens,
            hidden_states=hidden_states,
        )

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            if self.markov_head is None:
                raise RuntimeError("DSpark confidence head requires markov_head.")
            if self.replicated_markov_w1_weight is not None:
                prev_embeddings = F.embedding(
                    prev_tokens.long(), self.replicated_markov_w1_weight
                ).to(dtype=hidden_states.dtype)
            else:
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_tokens).to(
                    dtype=hidden_states.dtype
                )
            hidden_states = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(hidden_states).float()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        del get_embedding, pp_proxy_tensors
        if input_embeds is None:
            if self._shared_embed_tokens is None:
                raise ValueError(
                    "DSparkDraftModel requires the target embedding "
                    "(call attach_shared_modules first)."
                )
            input_embeds = self._shared_embed_tokens(input_ids)
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            return None

        for name, loaded_weight in weights:
            normalized_name = (
                name[len("model.") :] if name.startswith("model.") else name
            )
            if any(
                normalized_name.startswith(prefix)
                for prefix in _DSPARK_SKIPPED_WEIGHT_PREFIXES
            ):
                continue
            if (
                normalized_name.startswith("confidence_head.")
                and self.confidence_head is None
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    logger.warning("Parameter %s not found in DSparkDraftModel", name)
                    continue
                param = params_dict[resolved_name]
                if resolved_name == "fc.weight" and tuple(loaded_weight.shape) != tuple(
                    param.shape
                ):
                    raise ValueError(
                        "DSPARK fc.weight shape mismatch. "
                        f"Expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)} for weight {name!r}."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [Qwen3DSparkModel, DSparkDraftModel]
