import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.diffusion import musa_mul_add


_musa_available = hasattr(torch, "musa") and torch.musa.is_available()


@pytest.mark.skipif(not _musa_available, reason="MUSA device not available")
@pytest.mark.parametrize("batch_size,seq_len", [(1, 4), (2, 17), (1, 4096)])
def test_musa_fused_mul_add(batch_size: int, seq_len: int) -> None:
    torch.manual_seed(1234)
    device = torch.device("musa:0")
    dtype = torch.bfloat16
    hidden_size = 3072
    shape = (batch_size, seq_len, hidden_size)
    gate_shape = (batch_size, 1, hidden_size)

    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(gate_shape, device=device, dtype=dtype)
    c = torch.randn(shape, device=device, dtype=dtype)

    expected = MulAdd().forward_native(a, b, c)
    actual = musa_mul_add(a, b, c)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_musa_mul_add_fake_output() -> None:
    with FakeTensorMode():
        a = torch.empty((1, 4, 3072))
        b = torch.empty((1, 1, 3072))
        c = torch.empty_like(a)
        output = musa_mul_add(a, b, c)

    assert output.shape == a.shape
    assert output.dtype == a.dtype
