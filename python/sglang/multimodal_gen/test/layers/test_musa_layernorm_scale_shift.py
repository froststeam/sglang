import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.diffusion import (
    musa_layernorm_scale_shift,
    musa_scale_residual_layernorm_scale_shift,
)


_musa_available = hasattr(torch, "musa") and torch.musa.is_available()
pytestmark = pytest.mark.skipif(not _musa_available, reason="MUSA device not available")


@pytest.mark.parametrize(
    "batch_size,seq_len,modulation_has_sequence_dim",
    [
        (1, 4, False),
        (1, 17, True),
        (2, 17, False),
        (1, 1024, True),
    ],
)
def test_musa_fused_layernorm_scale_shift(
    batch_size: int,
    seq_len: int,
    modulation_has_sequence_dim: bool,
) -> None:
    torch.manual_seed(1234)
    device = torch.device("musa:0")
    dtype = torch.bfloat16
    hidden_size = 3072
    shape = (batch_size, seq_len, hidden_size)
    modulation_shape = (
        (batch_size, 1, hidden_size)
        if modulation_has_sequence_dim
        else (batch_size, hidden_size)
    )
    gate_shape = (batch_size, 1, hidden_size)

    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype)
    gate = torch.randn(gate_shape, device=device, dtype=dtype)
    scale = torch.randn(modulation_shape, device=device, dtype=dtype) * 0.1
    shift = torch.randn(modulation_shape, device=device, dtype=dtype) * 0.1

    norm = LayerNormScaleShift(hidden_size, eps=1e-6, elementwise_affine=False).to(
        device
    )
    residual_norm = ScaleResidualLayerNormScaleShift(
        hidden_size, eps=1e-6, elementwise_affine=False
    ).to(device)

    expected = norm.forward_native(x, shift, scale)
    actual = musa_layernorm_scale_shift(x, scale, shift, 1e-6)
    expected_residual, expected_residual_out = residual_norm.forward_native(
        residual, x, gate, shift, scale
    )
    actual_residual, actual_residual_out = musa_scale_residual_layernorm_scale_shift(
        residual, x, gate, scale, shift, 1e-6
    )
    dispatched = norm(x=x, shift=shift, scale=scale)
    dispatched_residual, dispatched_residual_out = residual_norm(
        residual=residual, x=x, gate=gate, shift=shift, scale=scale
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03125)
    torch.testing.assert_close(dispatched, expected, rtol=0.02, atol=0.03125)
    torch.testing.assert_close(
        actual_residual, expected_residual, rtol=0.02, atol=0.0625
    )
    torch.testing.assert_close(
        dispatched_residual, expected_residual, rtol=0.02, atol=0.0625
    )
    torch.testing.assert_close(actual_residual_out, expected_residual_out)
    torch.testing.assert_close(dispatched_residual_out, expected_residual_out)


def test_musa_fused_residual_layernorm_large_mean() -> None:
    """The residual path must remain stable for large-mean activations."""
    torch.manual_seed(4321)
    device = torch.device("musa:0")
    dtype = torch.bfloat16
    hidden_size = 3072
    shape = (1, 4, hidden_size)

    # This exposes cancellation in E[x^2] - E[x]^2 while keeping the
    # residual and gate values representable in bf16.
    residual = torch.full(shape, 32768.0, device=device, dtype=dtype)
    residual = residual + torch.randn(shape, device=device, dtype=dtype) * 512
    x = torch.randn(shape, device=device, dtype=dtype)
    gate = torch.ones((1, 1, hidden_size), device=device, dtype=dtype)
    scale = torch.randn((1, 1, hidden_size), device=device, dtype=dtype) * 0.1
    shift = torch.randn((1, 1, hidden_size), device=device, dtype=dtype) * 0.1

    norm = ScaleResidualLayerNormScaleShift(
        hidden_size, eps=1e-6, elementwise_affine=False
    ).to(device)
    expected, expected_residual = norm.forward_native(residual, x, gate, shift, scale)
    actual, actual_residual = musa_scale_residual_layernorm_scale_shift(
        residual, x, gate, scale, shift, 1e-6
    )

    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_residual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)
    torch.testing.assert_close(actual_residual, expected_residual)


@pytest.mark.parametrize("frame_modulation", [False, True])
def test_musa_layernorm_fallback_does_not_require_triton(
    frame_modulation: bool,
) -> None:
    torch.manual_seed(2468)
    device = torch.device("musa:0")
    dtype = torch.bfloat16
    batch_size, seq_len, hidden_size = 1, 8, 32
    shape = (batch_size, seq_len, hidden_size)
    modulation_shape = (
        (batch_size, 2, 1, hidden_size)
        if frame_modulation
        else (batch_size, 1, hidden_size)
    )

    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    gate = torch.randn(modulation_shape, device=device, dtype=dtype)
    scale = torch.randn(modulation_shape, device=device, dtype=dtype) * 0.1
    shift = torch.randn(modulation_shape, device=device, dtype=dtype) * 0.1

    def scale_shift(value: torch.Tensor) -> torch.Tensor:
        if not frame_modulation:
            return value * (1 + scale) + shift
        return (value.unflatten(1, (2, seq_len // 2)) * (1 + scale) + shift).flatten(
            1, 2
        )

    if frame_modulation:
        expected_residual = residual + (
            x.unflatten(1, (2, seq_len // 2)) * gate
        ).flatten(1, 2)
    else:
        expected_residual = residual + x * gate

    expected = scale_shift(F.layer_norm(x.float(), (hidden_size,), eps=1e-6).to(dtype))
    expected_with_residual = scale_shift(
        F.layer_norm(expected_residual.float(), (hidden_size,), eps=1e-6).to(dtype)
    )

    norm = LayerNormScaleShift(hidden_size, eps=1e-6, elementwise_affine=False).to(
        device
    )
    residual_norm = ScaleResidualLayerNormScaleShift(
        hidden_size, eps=1e-6, elementwise_affine=False
    ).to(device)
    actual = norm(x=x, shift=shift, scale=scale)
    actual_with_residual, actual_residual = residual_norm(
        residual=residual, x=x, gate=gate, shift=shift, scale=scale
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_residual, expected_residual)
    torch.testing.assert_close(actual_with_residual, expected_with_residual)
