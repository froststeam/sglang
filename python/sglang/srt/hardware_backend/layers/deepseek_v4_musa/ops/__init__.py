"""DeepSeekV4 MUSA operator wrappers and dispatch logic."""

from .. import kernels as _musa_kernels
from ..kernels import *  # noqa: F403,F401
from . import (
    cache_ops,
    compress_ops,
    hc_head_ops,
    hisparse_ops,
    mhc_ops,
    mhc_prenorm_ops,
    moe_prefill_ops,
    norm_rope_ops,
    routing_ops,
    swiglu_quant_ops,
    wo_a_ops,
)
from .ops_common import (
    _debug_musa_allow_torch_fallback,
    _debug_musa_torch_fallback,
    _has_musa_tensor,
    _is_musa_tensor,
    _musa_graph_capture_enabled,
)

# Kept for the legacy deepseek_v4_musa_ops facade and tests that monkeypatch
# importlib on the facade; _sync_patch_to_domain forwards updates to domains.
importlib = swiglu_quant_ops.importlib

_DOMAIN_MODULES = [
    routing_ops,
    swiglu_quant_ops,
    norm_rope_ops,
    cache_ops,
    compress_ops,
    hisparse_ops,
    hc_head_ops,
    mhc_ops,
    mhc_prenorm_ops,
    wo_a_ops,
    moe_prefill_ops,
]

for _module in _DOMAIN_MODULES:
    for _name in getattr(_module, "__all__", []):
        globals()[_name] = getattr(_module, _name)


def _sync_patch_to_domain(name: str, value) -> None:
    for module in _DOMAIN_MODULES:
        if hasattr(module, name):
            setattr(module, name, value)


__all__ = sorted(
    {
        "_musa_kernels",
        "_debug_musa_allow_torch_fallback",
        "_debug_musa_torch_fallback",
        "_is_musa_tensor",
        "_has_musa_tensor",
        "_musa_graph_capture_enabled",
        *_musa_kernels.__all__,
        *(
            name
            for module in _DOMAIN_MODULES
            for name in getattr(module, "__all__", [])
        ),
    }
)
