"""Compatibility facade for DeepSeekV4 MUSA operator wrappers."""

from sglang.srt.hardware_backend.layers.deepseek_v4_musa import ops as _ops
from sglang.srt.hardware_backend.layers.deepseek_v4_musa._forwarding import (
    install_forwarding_module,
)
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops import *  # noqa: F403

install_forwarding_module(__name__, _ops, _ops.__all__)
