from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from sglang.srt.speculative.dflash_utils import (
    build_target_layer_ids,
    parse_dflash_draft_config,
)

logger = logging.getLogger(__name__)

DEFAULT_DSPARK_GAMMA = 7
SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla",)


def dspark_gamma_from_num_draft_tokens(num_draft_tokens: int) -> int:
    gamma = int(num_draft_tokens) - 1
    if gamma < 1:
        raise ValueError(
            "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1), "
            f"got {num_draft_tokens}."
        )
    return gamma


@dataclass(frozen=True)
class DSparkDraftConfig:
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    gamma: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]
    markov_rank: int
    markov_head_type: Optional[str]

    def resolve_gamma(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.gamma if self.gamma is not None else default

    def require_markov(self) -> bool:
        return int(self.markov_rank) > 0


@dataclass(frozen=True)
class DSparkRuntimeConfig:
    gamma: int
    verify_num_draft_tokens: int
    mask_token_id: int


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dspark_config(config: Any) -> dict:
    cfg = _cfg_get(config, "dspark_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


def _parse_layer_ids_csv(raw_layer_ids: str) -> List[int]:
    parts = [part.strip() for part in raw_layer_ids.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError(
            "--speculative-dspark-target-layer-ids must be a non-empty "
            "comma-separated list of integers."
        )
    try:
        return [int(part) for part in parts]
    except ValueError as e:
        raise ValueError(
            "--speculative-dspark-target-layer-ids must contain only integers, "
            f"got {raw_layer_ids!r}."
        ) from e


def _validate_target_layer_ids(
    *, layer_ids: List[int], target_num_layers: int
) -> List[int]:
    previous = None
    for index, layer_id in enumerate(layer_ids):
        if layer_id < -1 or layer_id >= target_num_layers - 1:
            raise ValueError(
                "DSPARK target_layer_ids must be in {-1} U "
                f"[0, {target_num_layers - 2}], got "
                f"layer_ids[{index}]={layer_id}. -1 denotes the embedding "
                "output; the final target layer cannot be captured by "
                "the current before-layer capture adapters."
            )
        if previous is not None and layer_id <= previous:
            raise ValueError(
                "DSPARK target_layer_ids must be strictly increasing, "
                f"got {layer_ids}."
            )
        previous = layer_id
    return layer_ids


def parse_dspark_draft_config(*, draft_hf_config: Any) -> DSparkDraftConfig:
    """Parse DSpark-specific draft config fields from HF config/dict.

    The parser keeps compatibility with the current local checkpoints
    (top-level fields), nested ``dspark_config``/``dflash_config`` fields,
    and the newer official prefixed ``dspark_*`` convention.
    """
    base = parse_dflash_draft_config(draft_hf_config=draft_hf_config)
    dspark_cfg = _get_dspark_config(draft_hf_config)
    text_config = _get_text_config(draft_hf_config)

    prefixed_block_size = _cfg_get(draft_hf_config, "dspark_block_size", None)
    prefixed_markov_rank = _cfg_get(draft_hf_config, "dspark_markov_rank", None)
    prefixed_markov_head_type = _cfg_get(
        draft_hf_config, "dspark_markov_head_type", None
    )
    prefixed_noise_token_id = _cfg_get(draft_hf_config, "dspark_noise_token_id", None)
    prefixed_target_layer_ids = _cfg_get(
        draft_hf_config, "dspark_target_layer_ids", None
    )
    uses_prefixed = any(
        value is not None
        for value in (
            prefixed_block_size,
            prefixed_markov_rank,
            prefixed_noise_token_id,
            prefixed_target_layer_ids,
        )
    )

    raw_markov_rank = (
        prefixed_markov_rank
        if prefixed_markov_rank is not None
        else dspark_cfg.get(
            "markov_rank",
            _cfg_get(
                text_config, "markov_rank", _cfg_get(draft_hf_config, "markov_rank", 0)
            ),
        )
    )
    markov_rank = _parse_optional_int(
        raw_markov_rank, field_name="DSpark markov_rank", min_value=0
    )
    markov_rank = 0 if markov_rank is None else markov_rank

    markov_head_type = (
        prefixed_markov_head_type
        if prefixed_markov_head_type is not None
        else dspark_cfg.get(
            "markov_head_type",
            _cfg_get(
                text_config,
                "markov_head_type",
                _cfg_get(draft_hf_config, "markov_head_type", None),
            ),
        )
    )
    if markov_rank > 0 and markov_head_type is None and not uses_prefixed:
        raise ValueError(
            "DSpark requires markov_head_type when markov_rank > 0, got None."
        )
    if markov_head_type is not None:
        markov_head_type = str(markov_head_type).lower()
        if markov_head_type not in SUPPORTED_DSPARK_MARKOV_HEAD_TYPES:
            raise ValueError(
                f"Unsupported DSpark markov_head_type={markov_head_type!r}. "
                f"Supported: {SUPPORTED_DSPARK_MARKOV_HEAD_TYPES}."
            )

    raw_mask_token_id = (
        prefixed_noise_token_id
        if prefixed_noise_token_id is not None
        else dspark_cfg.get(
            "mask_token_id",
            _cfg_get(
                text_config,
                "mask_token_id",
                _cfg_get(draft_hf_config, "mask_token_id", base.mask_token_id),
            ),
        )
    )
    mask_token_id = _parse_optional_int(
        raw_mask_token_id, field_name="DSpark mask_token_id", min_value=0
    )

    gamma = _parse_optional_int(
        prefixed_block_size if prefixed_block_size is not None else base.block_size,
        field_name="DSpark block_size",
        min_value=1,
    )

    if prefixed_target_layer_ids is not None:
        if not isinstance(prefixed_target_layer_ids, (list, tuple)) or not len(
            prefixed_target_layer_ids
        ):
            raise ValueError(
                "DSpark dspark_target_layer_ids must be a non-empty list of ints, "
                f"got {prefixed_target_layer_ids!r}."
            )
        target_layer_ids: Optional[List[int]] = [
            int(layer_id) for layer_id in prefixed_target_layer_ids
        ]
    else:
        target_layer_ids = base.target_layer_ids

    return DSparkDraftConfig(
        num_hidden_layers=base.num_hidden_layers,
        num_target_layers=base.num_target_layers,
        gamma=gamma,
        target_layer_ids=target_layer_ids,
        mask_token=base.mask_token,
        mask_token_id=mask_token_id,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
    )


def resolve_runtime_config(
    *,
    draft_hf_config: Any,
    speculative_num_draft_tokens: Optional[int],
    target_vocab_size: int,
) -> DSparkRuntimeConfig:
    draft_config = parse_dspark_draft_config(draft_hf_config=draft_hf_config)
    if not draft_config.require_markov():
        raise ValueError(
            "DSpark draft requires markov_rank > 0; got "
            f"markov_rank={draft_config.markov_rank}."
        )

    if speculative_num_draft_tokens is None:
        gamma = int(draft_config.resolve_gamma(default=None) or 0)
        if gamma < 1:
            raise ValueError(
                "DSpark could not resolve gamma from the draft config and "
                "speculative_num_draft_tokens is unset."
            )
    else:
        gamma = dspark_gamma_from_num_draft_tokens(
            int(speculative_num_draft_tokens)
        )
        config_gamma = draft_config.resolve_gamma(default=None)
        if config_gamma is not None and int(config_gamma) != gamma:
            logger.warning(
                "DSpark gamma mismatch: using gamma=%s from "
                "speculative_num_draft_tokens=%s but draft config block_size=%s.",
                gamma,
                speculative_num_draft_tokens,
                config_gamma,
            )

    if draft_config.mask_token_id is None:
        raise ValueError(
            "DSpark requires mask_token_id to be set in the draft model config."
        )
    mask_token_id = int(draft_config.mask_token_id)
    if mask_token_id >= int(target_vocab_size):
        raise ValueError(
            f"DSpark mask_token_id={mask_token_id} is outside the target "
            f"vocab size {target_vocab_size}."
        )

    return DSparkRuntimeConfig(
        gamma=gamma,
        verify_num_draft_tokens=gamma + 1,
        mask_token_id=mask_token_id,
    )


def read_draft_checkpoint_gamma(*, server_args: Any) -> Optional[int]:
    from sglang.srt.utils.hf_transformers_utils import get_config

    draft_hf_config = get_config(
        server_args.speculative_draft_model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.speculative_draft_model_revision,
        model_override_args=json.loads(server_args.json_model_override_args),
    )
    return parse_dspark_draft_config(draft_hf_config=draft_hf_config).resolve_gamma(
        default=None
    )


def resolve_target_layer_ids(
    *,
    raw_layer_ids: Optional[str],
    target_num_layers: int,
    draft_hf_config: Any,
    draft_num_layers: Optional[int] = None,
) -> List[int]:
    """Resolve target layer ids used to build DSpark context features."""
    target_num_layers = int(target_num_layers)
    if target_num_layers <= 1:
        raise ValueError(
            "DSPARK requires target num_hidden_layers > 1, "
            f"got {target_num_layers}."
        )

    if raw_layer_ids is None:
        draft_config = parse_dspark_draft_config(draft_hf_config=draft_hf_config)
        if draft_config.target_layer_ids is not None:
            if len(draft_config.target_layer_ids) == 0:
                raise ValueError("DSPARK draft config target_layer_ids is empty.")
            layer_ids = [int(layer_id) for layer_id in draft_config.target_layer_ids]
        else:
            if draft_num_layers is None:
                draft_num_layers = draft_config.num_hidden_layers
            if draft_num_layers is None:
                raise ValueError(
                    "DSPARK draft config must provide dspark_target_layer_ids "
                    "or num_hidden_layers so target layers can be inferred."
                )
            layer_ids = build_target_layer_ids(
                target_num_layers, int(draft_num_layers)
            )
    else:
        layer_ids = _parse_layer_ids_csv(raw_layer_ids)

    return _validate_target_layer_ids(
        layer_ids=layer_ids, target_num_layers=target_num_layers
    )


def checkpoint_bundles_dspark_draft(hf_config: Any) -> bool:
    dspark_cfg = _get_dspark_config(hf_config)
    return any(
        _cfg_get(hf_config, key, None) is not None
        for key in (
            "dspark_block_size",
            "dspark_markov_rank",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
        )
    ) or any(
        dspark_cfg.get(key) is not None
        for key in (
            "block_size",
            "markov_rank",
            "mask_token_id",
            "target_layer_ids",
        )
    )
