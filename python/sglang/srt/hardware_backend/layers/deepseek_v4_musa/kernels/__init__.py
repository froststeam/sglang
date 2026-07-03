"""DeepSeekV4 MUSA TileLang kernel factories."""

from . import (
    cache_kernels,
    compress_kernels,
    hc_head_kernels,
    hisparse_kernels,
    kernel_common,
    mhc_kernels,
    moe_prefill_kernels,
    norm_rope_kernels,
    routing_kernels,
    wo_a_kernels,
)

_DOMAIN_MODULES = [
    kernel_common,
    cache_kernels,
    compress_kernels,
    hc_head_kernels,
    hisparse_kernels,
    mhc_kernels,
    moe_prefill_kernels,
    norm_rope_kernels,
    routing_kernels,
    wo_a_kernels,
]

for _module in _DOMAIN_MODULES:
    for _name in getattr(_module, "__all__", []):
        globals()[_name] = getattr(_module, _name)

__all__ = sorted(
    {
        name
        for module in _DOMAIN_MODULES
        for name in getattr(module, "__all__", [])
    }
)
