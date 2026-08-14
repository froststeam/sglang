from __future__ import annotations

from typing import Any


def configure_aux_hidden_state_capture(
    *,
    model: Any,
    eagle_use_aux_hidden_state: bool,
    eagle_aux_hidden_state_layer_ids: Any,
    dflash_use_aux_hidden_state: bool,
    dflash_target_layer_ids: Any,
    is_dspark: bool,
) -> None:
    """Configure model-side aux hidden-state capture before graph capture."""
    if eagle_use_aux_hidden_state:
        model.set_eagle3_layers_to_capture(eagle_aux_hidden_state_layer_ids)

    if not dflash_use_aux_hidden_state:
        return

    if is_dspark and hasattr(model, "set_dspark_layers_to_capture"):
        model.set_dspark_layers_to_capture(dflash_target_layer_ids)
        return

    if hasattr(model, "set_dflash_layers_to_capture"):
        model.set_dflash_layers_to_capture(dflash_target_layer_ids)
        return

    raise ValueError(
        f"Model {model.__class__.__name__} implements neither "
        "set_dspark_layers_to_capture nor set_dflash_layers_to_capture, "
        "one of which is required for DFLASH/DSPARK."
    )
