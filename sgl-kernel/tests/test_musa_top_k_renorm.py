"""Correctness tests for the standalone MUSA TopK renorm operator."""

import pytest
import torch

import sgl_kernel  # noqa: F401


def _require_musa():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA is not available")


def _reference_top_k_probs(probs, top_ks):
    output = torch.zeros_like(probs)
    for row, top_k in enumerate(top_ks.tolist()):
        values, indices = torch.topk(probs[row], top_k)
        output[row, indices] = values / values.sum()
    return output


def _run_musa_top_k(probs, top_ks):
    return torch.ops.sgl_kernel.musa_top_k_renorm_probs.default(probs, top_ks)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("vocab", [1024, 8192, 151936])
@pytest.mark.parametrize("top_k", [1, 20, 40, 256])
def test_musa_top_k_renorm_correctness(batch, vocab, top_k):
    _require_musa()
    torch.manual_seed(1357)
    logits = torch.randn(batch, vocab, device="musa", dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    top_ks = torch.full((batch,), top_k, device="musa", dtype=torch.int32)

    expected = _reference_top_k_probs(probs.cpu(), top_ks.cpu())
    actual = _run_musa_top_k(probs, top_ks)
    torch.musa.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-7)


def test_musa_top_k_renorm_per_row_parameters():
    _require_musa()
    torch.manual_seed(9753)
    logits = torch.randn(8, 8192, device="musa", dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    top_ks = torch.tensor(
        [1, 4, 20, 40, 64, 128, 256, 40], device="musa", dtype=torch.int32
    )

    expected = _reference_top_k_probs(probs.cpu(), top_ks.cpu())
    actual = _run_musa_top_k(probs, top_ks)
    torch.musa.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-7)
