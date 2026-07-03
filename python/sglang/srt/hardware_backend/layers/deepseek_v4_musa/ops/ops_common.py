import logging
from typing import Optional

import torch

_MUSA_FALLBACK_LOGGER = logging.getLogger("sglang.musa.fallback")

def _debug_musa_allow_torch_fallback() -> bool:
    # Torch fallbacks are intentionally disabled on MUSA production paths: they
    # are slow and can introduce graph-capture host syncs. Keep call sites
    # fail-closed instead of honoring debug opt-ins.
    return False

def _debug_musa_torch_fallback(reason: str) -> None:
    _MUSA_FALLBACK_LOGGER.warning("MUSA torch fallback: %s", reason)

def _is_musa_tensor(tensor: object) -> bool:
    return getattr(getattr(tensor, "device", None), "type", None) == "musa"

def _has_musa_tensor(*tensors: Optional[torch.Tensor]) -> bool:
    return any(tensor is not None and _is_musa_tensor(tensor) for tensor in tensors)

def _musa_graph_capture_enabled() -> bool:
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if get_is_capture_mode():
            return True
    except Exception:
        pass
    try:
        from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph

        if is_in_piecewise_cuda_graph():
            return True
    except Exception:
        pass
    return False

__all__ = [
    '_debug_musa_allow_torch_fallback',
    '_debug_musa_torch_fallback',
    '_is_musa_tensor',
    '_has_musa_tensor',
    '_musa_graph_capture_enabled',
]
