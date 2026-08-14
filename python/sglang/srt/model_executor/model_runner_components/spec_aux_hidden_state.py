from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
    resolve_target_layer_ids as resolve_dspark_target_layer_ids,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecAuxHiddenStateConfig:
    dflash_use_aux_hidden_state: bool = False
    dflash_draft_num_layers: Optional[int] = None
    dflash_target_layer_ids: Optional[List[int]] = None
    dspark_use_aux_hidden_state: bool = False
    dspark_target_layer_ids: Optional[List[int]] = None


def resolve_spec_aux_hidden_state_config(
    *,
    server_args: "ServerArgs",
    model_config: ModelConfig,
    spec_algorithm: "SpeculativeAlgorithm",
    is_draft_worker: bool,
) -> SpecAuxHiddenStateConfig:
    if is_draft_worker:
        return SpecAuxHiddenStateConfig()

    if spec_algorithm.is_dflash():
        return _resolve_dflash_aux_hidden_state_config(
            server_args=server_args,
            model_config=model_config,
        )

    if spec_algorithm.is_dspark():
        return _resolve_dspark_aux_hidden_state_config(
            server_args=server_args,
            model_config=model_config,
        )

    return SpecAuxHiddenStateConfig()


def _load_draft_model_config(*, server_args: "ServerArgs") -> ModelConfig:
    return ModelConfig.from_server_args(
        server_args,
        model_path=server_args.speculative_draft_model_path,
        model_revision=server_args.speculative_draft_model_revision,
        is_draft_model=True,
    )


def _get_target_num_layers(*, model_config: ModelConfig, algorithm: str) -> int:
    target_num_layers = getattr(model_config.hf_text_config, "num_hidden_layers", None)
    if target_num_layers is None:
        raise ValueError(
            f"{algorithm} requires target num_hidden_layers in config. "
            f"Got target={target_num_layers}."
        )
    return int(target_num_layers)


def _resolve_dflash_aux_hidden_state_config(
    *,
    server_args: "ServerArgs",
    model_config: ModelConfig,
) -> SpecAuxHiddenStateConfig:
    draft_model_config = _load_draft_model_config(server_args=server_args)
    dflash_draft_config = parse_dflash_draft_config(
        draft_hf_config=draft_model_config.hf_config
    )
    draft_num_layers = dflash_draft_config.require_num_layers()
    trained_target_layers = dflash_draft_config.num_target_layers
    target_num_layers = _get_target_num_layers(
        model_config=model_config, algorithm="DFLASH"
    )

    if (
        trained_target_layers is not None
        and int(trained_target_layers) != target_num_layers
    ):
        logger.warning(
            "DFLASH draft config num_target_layers=%s differs from runtime "
            "target num_hidden_layers=%s; selecting capture layers based on "
            "the runtime target model.",
            trained_target_layers,
            target_num_layers,
        )

    return SpecAuxHiddenStateConfig(
        dflash_use_aux_hidden_state=True,
        dflash_draft_num_layers=int(draft_num_layers),
        dflash_target_layer_ids=dflash_draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers,
            draft_num_layers=int(draft_num_layers),
        ),
    )


def _resolve_dspark_aux_hidden_state_config(
    *,
    server_args: "ServerArgs",
    model_config: ModelConfig,
) -> SpecAuxHiddenStateConfig:
    draft_model_config = _load_draft_model_config(server_args=server_args)
    draft_hf_config = draft_model_config.hf_config
    dspark_draft_config = parse_dspark_draft_config(
        draft_hf_config=draft_hf_config
    )
    if not dspark_draft_config.require_markov():
        raise ValueError(
            "DSPARK requires markov_rank > 0 in the draft config, "
            f"got markov_rank={dspark_draft_config.markov_rank}."
        )

    target_num_layers = _get_target_num_layers(
        model_config=model_config, algorithm="DSPARK"
    )
    trained_target_layers = dspark_draft_config.num_target_layers
    if (
        trained_target_layers is not None
        and int(trained_target_layers) != target_num_layers
    ):
        logger.warning(
            "DSPARK draft config num_target_layers=%s differs from runtime "
            "target num_hidden_layers=%s; selecting capture layers based on "
            "the runtime target model.",
            trained_target_layers,
            target_num_layers,
        )

    draft_num_layers = getattr(
        getattr(draft_model_config, "hf_text_config", None), "num_hidden_layers", None
    )
    target_layer_ids = resolve_dspark_target_layer_ids(
        raw_layer_ids=server_args.speculative_dspark_target_layer_ids,
        target_num_layers=target_num_layers,
        draft_hf_config=draft_hf_config,
        draft_num_layers=draft_num_layers,
    )
    logger.info("DSPARK target hidden capture layers: %s", target_layer_ids)

    return SpecAuxHiddenStateConfig(
        dflash_use_aux_hidden_state=True,
        dflash_draft_num_layers=(
            int(draft_num_layers) if draft_num_layers is not None else None
        ),
        dflash_target_layer_ids=target_layer_ids,
        dspark_use_aux_hidden_state=True,
        dspark_target_layer_ids=target_layer_ids,
    )
