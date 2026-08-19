import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.fused_scale_shift_gate import (
    FusedLayerNormScaleShiftGateSelect01,
    FusedResidualLayerNormScaleShiftGateSelect01,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.diffusion import (
    musa_layernorm_scale_shift_gate_select01,
    musa_residual_layernorm_scale_shift_gate_select01,
)

_musa_available = hasattr(torch, "musa") and torch.musa.is_available()
pytestmark = pytest.mark.skipif(not _musa_available, reason="MUSA device not available")


def _inputs(batch_size: int, seq_len: int, dtype: torch.dtype):
    device = torch.device("musa:0")
    hidden_size = 3072
    x = torch.randn((batch_size, seq_len, hidden_size), device=device, dtype=dtype)
    modulation = tuple(
        torch.randn((batch_size, hidden_size), device=device, dtype=dtype) * 0.1
        for _ in range(6)
    )
    index = torch.randint(0, 2, (batch_size, seq_len), device=device, dtype=torch.int32)
    return x, modulation, index


def _reference(
    x: torch.Tensor,
    modulation: tuple[torch.Tensor, ...],
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale0, shift0, gate0, scale1, shift1, gate1 = modulation
    select1 = index.bool().unsqueeze(-1)
    scale = torch.where(select1, scale1.unsqueeze(1), scale0.unsqueeze(1))
    shift = torch.where(select1, shift1.unsqueeze(1), shift0.unsqueeze(1))
    gate = torch.where(select1, gate1.unsqueeze(1), gate0.unsqueeze(1))
    normalized = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
    return normalized * (1 + scale) + shift, gate


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 6), (2, 257)])
def test_musa_layernorm_scale_shift_gate_select01(
    dtype: torch.dtype, batch_size: int, seq_len: int
) -> None:
    torch.manual_seed(1234)
    x, modulation, index = _inputs(batch_size, seq_len, dtype)
    expected, expected_gate = _reference(x, modulation, index)

    actual, actual_gate = musa_layernorm_scale_shift_gate_select01(
        x, *modulation, index, 1e-6
    )
    dispatched, dispatched_gate = FusedLayerNormScaleShiftGateSelect01()(
        x, None, None, *modulation, index, 1e-6
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)
    torch.testing.assert_close(dispatched, expected, rtol=0.02, atol=0.0625)
    torch.testing.assert_close(actual_gate, expected_gate)
    torch.testing.assert_close(dispatched_gate, expected_gate)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 6), (2, 257)])
def test_musa_residual_layernorm_scale_shift_gate_select01(
    dtype: torch.dtype, batch_size: int, seq_len: int
) -> None:
    torch.manual_seed(5678)
    x, modulation, index = _inputs(batch_size, seq_len, dtype)
    residual = torch.randn_like(x)
    residual_gate = torch.randn_like(x)
    expected_residual = residual + residual_gate * x
    expected, expected_gate = _reference(expected_residual, modulation, index)

    actual, actual_residual, actual_gate = (
        musa_residual_layernorm_scale_shift_gate_select01(
            x, residual, residual_gate, *modulation, index, 1e-6
        )
    )
    dispatched, dispatched_residual, dispatched_gate = (
        FusedResidualLayerNormScaleShiftGateSelect01()(
            x,
            residual,
            residual_gate,
            None,
            None,
            *modulation,
            index,
            1e-6,
        )
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)
    torch.testing.assert_close(dispatched, expected, rtol=0.02, atol=0.0625)
    torch.testing.assert_close(actual_residual, expected_residual)
    torch.testing.assert_close(dispatched_residual, expected_residual)
    torch.testing.assert_close(actual_gate, expected_gate)
    torch.testing.assert_close(dispatched_gate, expected_gate)


def test_musa_residual_select01_large_mean() -> None:
    torch.manual_seed(9012)
    x, modulation, index = _inputs(1, 6, torch.bfloat16)
    residual = torch.full_like(x, 32768.0) + torch.randn_like(x) * 512
    residual_gate = torch.ones_like(x)
    expected_residual = residual + residual_gate * x
    expected, expected_gate = _reference(expected_residual, modulation, index)

    actual, actual_residual, actual_gate = (
        musa_residual_layernorm_scale_shift_gate_select01(
            x, residual, residual_gate, *modulation, index, 1e-6
        )
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)
    torch.testing.assert_close(actual_residual, expected_residual)
    torch.testing.assert_close(actual_gate, expected_gate)


def test_musa_residual_select01_is_deterministic() -> None:
    torch.manual_seed(42)
    x, modulation, index = _inputs(1, 4096, torch.bfloat16)
    residual = torch.randn_like(x)
    residual_gate = torch.randn_like(x)

    reference = musa_residual_layernorm_scale_shift_gate_select01(
        x, residual, residual_gate, *modulation, index, 1e-6
    )
    torch.musa.synchronize()
    reference = tuple(output.clone() for output in reference)

    for _ in range(5):
        actual = musa_residual_layernorm_scale_shift_gate_select01(
            x, residual, residual_gate, *modulation, index, 1e-6
        )
        torch.musa.synchronize()
        for actual_output, reference_output in zip(actual, reference):
            assert torch.equal(actual_output, reference_output)
