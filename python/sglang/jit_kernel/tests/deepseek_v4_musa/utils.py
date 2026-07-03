from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[5]
_MUSA_OPS_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "hardware_backend" / "layers" / "deepseek_v4_musa_ops.py"
_MUSA_OPS_SPEC = importlib.util.spec_from_file_location("deepseek_v4_musa_ops", _MUSA_OPS_PATH)
assert _MUSA_OPS_SPEC is not None and _MUSA_OPS_SPEC.loader is not None
MUSA_OPS = importlib.util.module_from_spec(_MUSA_OPS_SPEC)
sys.modules[_MUSA_OPS_SPEC.name] = MUSA_OPS
_MUSA_OPS_SPEC.loader.exec_module(MUSA_OPS)


def require_musa_or_skip() -> None:
    musa = getattr(torch, "musa", None)
    if musa is None:
        pytest.skip("MUSA runtime is not available")
    if not musa.is_available():
        pytest.skip("MUSA device is not available")


def get_musa_device() -> torch.device:
    require_musa_or_skip()
    return torch.device("musa")


def assert_sm90_aligned_scale_contract(
    scale: torch.Tensor,
    expected_shape: Optional[tuple[int, ...]] = None,
) -> None:
    assert scale.dtype == torch.float32
    assert scale.dtype != torch.int32
    if expected_shape is not None:
        assert tuple(scale.shape) == expected_shape


def reference_swiglu(input: torch.Tensor, swiglu_limit: Optional[float] = None) -> torch.Tensor:
    gate, up = input.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


def reference_grouped_fp8_quant(
    value: torch.Tensor,
    quant_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reshaped = value.float().reshape(value.shape[0], -1, quant_group_size)
    scale = reshaped.abs().amax(dim=-1).clamp(min=1e-4) / 448.0
    quantized = torch.clamp(reshaped / scale.unsqueeze(-1), -448.0, 448.0)
    return quantized.reshape(value.shape).to(torch.float8_e4m3fn), scale.to(torch.float32)
