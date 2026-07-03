# Migrated from the former monolithic DeepSeekV4 MUSA test file.

from __future__ import annotations

import importlib.util

import os

import sys

import types

from dataclasses import dataclass

from pathlib import Path

import pytest

import torch

from sglang.test.ci.ci_register import register_musa_ci
from ..utils import MUSA_OPS, get_musa_device

register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="DeepSeek V4 MUSA operator test is opt-in outside smoke CI",
)

def _load_deepseek_v4_module_for_validator():
    module_name = "deepseek_v4_test_module"
    path = Path(__file__).resolve().parents[3] / "deepseek_v4.py"

    triton_module = types.ModuleType("triton")
    triton_module.__dict__["jit"] = lambda fn=None, **_: fn if fn is not None else (lambda inner: inner)
    triton_language_module = types.ModuleType("triton.language")
    triton_module.__dict__["language"] = triton_language_module

    def load_jit_stub() -> None:
        return None

    def make_cpp_args_stub() -> str:
        return ""

    utils_module = types.ModuleType("sglang.jit_kernel.utils")
    utils_module.__dict__["cache_once"] = lambda fn: fn
    utils_module.__dict__["is_arch_support_pdl"] = lambda: False
    utils_module.__dict__["load_jit"] = load_jit_stub
    utils_module.__dict__["make_cpp_args"] = make_cpp_args_stub

    debug_module = types.ModuleType("sglang.srt.debug_utils.deepseek_v4_debug_utils")
    debug_module.__dict__["deepseek_v4_moe_code_path_checker"] = types.SimpleNamespace(observed=0)

    saved_modules = {
        name: sys.modules.get(name)
        for name in [
            module_name,
            "triton",
            "triton.language",
            "sglang.jit_kernel.utils",
            "sglang.srt.debug_utils.deepseek_v4_debug_utils",
        ]
    }
    sys.modules["triton"] = triton_module
    sys.modules["triton.language"] = triton_language_module
    sys.modules["sglang.jit_kernel.utils"] = utils_module
    sys.modules["sglang.srt.debug_utils.deepseek_v4_debug_utils"] = debug_module

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

compress_forward_musa = MUSA_OPS.compress_forward_musa

compress_fused_norm_rope_inplace_musa = MUSA_OPS.compress_fused_norm_rope_inplace_musa

fused_rope_musa = MUSA_OPS.fused_rope_musa

fused_store_cache_musa = MUSA_OPS.fused_store_cache_musa

get_paged_mqa_logits_metadata_musa = MUSA_OPS.get_paged_mqa_logits_metadata_musa

rmsnorm_self_musa = MUSA_OPS.rmsnorm_self_musa

def _require_real_tilelang_musa_test() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")

def _assert_repeat_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch_count = int((actual != expected).sum().item())
    if actual.dtype.is_floating_point:
        max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
        detail = f", max_abs_diff={max_abs_diff:.6g}"
    else:
        detail = ""
    pytest.fail(
        f"{name} is not repeat-deterministic: "
        f"mismatch_count={mismatch_count}/{actual.numel()}{detail}"
    )

@dataclass
class _DecodePlan:
    seq_lens: torch.Tensor
    compress_ratio: int

@dataclass
class _PrefillPlan:
    compress_ratio: int
    compress_plan: torch.Tensor
    write_plan: torch.Tensor

def _pack_prefill_rows(rows: list[tuple[int, int, int, int]]) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 16), dtype=torch.uint8)
    return torch.tensor(rows, dtype=torch.int32).view(torch.uint8).reshape(len(rows), 16)

def _prefill_rows(plan: torch.Tensor) -> list[tuple[int, int, int, int]]:
    return [tuple(row) for row in plan.contiguous().view(torch.int32).reshape(-1, 4).tolist()]

def _prefill_plan_ref(
    compress_ratio: int,
    num_q_tokens: int,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    is_overlap = compress_ratio == 4
    effective_ratio = compress_ratio * (1 + int(is_overlap))
    compress_rows = []
    write_rows = []
    counter = 0
    for batch_id, (seq_len_tensor, extend_len_tensor) in enumerate(zip(seq_lens, extend_lens)):
        seq_len = int(seq_len_tensor.item())
        extend_len = int(extend_len_tensor.item())
        prefix_len = seq_len - extend_len
        base_pos = seq_len // compress_ratio * compress_ratio
        start_write_pos = base_pos
        if is_overlap:
            start_write_pos = base_pos - compress_ratio if base_pos >= compress_ratio else 0
        for j in range(extend_len):
            position = prefix_len + j
            row = (counter + j, batch_id, position, effective_ratio - min(j + 1, effective_ratio))
            if (position + 1) % compress_ratio == 0:
                compress_rows.append(row)
            if position >= start_write_pos:
                write_rows.append(row)
        counter += extend_len
    assert counter == num_q_tokens
    return compress_rows, write_rows

def _ratio4_prefill_ref(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_rows: list[tuple[int, int, int, int]],
    write_rows: list[tuple[int, int, int, int]],
    head_dim: int,
    extra_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    buffer = kv_score_buffer.clone()
    out = torch.zeros((kv_score_input.shape[0], head_dim), dtype=kv_score_input.dtype)

    def read_slot(block_id: int, slot: int) -> torch.Tensor:
        if buffer.dim() == 2:
            return buffer[block_id * 8 + slot]
        return buffer[block_id, slot]

    def write_slot(block_id: int, slot: int, value: torch.Tensor) -> None:
        if buffer.dim() == 2:
            buffer[block_id * 8 + slot] = value
        else:
            buffer[block_id, slot] = value

    def run_write_rows() -> None:
        for ragged_id, batch_id, position, _window_len in write_rows:
            if extra_data is not None:
                block_id = int(indices[batch_id].item())
                if position < int(extra_data[batch_id, 3].item()):
                    block_id = int(extra_data[batch_id, 2].item())
                buffer[block_id, position % 4] = kv_score_input[ragged_id]
            else:
                write_slot(int(indices[batch_id].item()), position % 8, kv_score_input[ragged_id])

    def run_compress_rows() -> None:
        for ragged_id, batch_id, position, window_len in compress_rows:
            seq_len = position + 1
            kv_window = []
            score_window = []
            for i in range(8):
                if extra_data is not None and i < window_len:
                    source_block = int(extra_data[batch_id, 1].item())
                    if window_len > 4 and i < 4:
                        source_block = int(extra_data[batch_id, 0].item())
                    src = buffer[source_block, i % 4]
                elif i < window_len:
                    index = int(indices[batch_id].item())
                    src = read_slot(index, (seq_len + i) % 8)
                else:
                    src = kv_score_input[ragged_id + i - 7]
                src = src.reshape(4, head_dim)
                kv_window.append(src[0] if i < 4 else src[1])
                score_window.append(src[2] if i < 4 else src[3])
            if seq_len == 4:
                for i in range(4):
                    kv_window[i] = torch.zeros_like(kv_window[i])
                    score_window[i] = torch.full_like(score_window[i], -1e9)
            out[ragged_id] = _compress_ref(torch.stack(kv_window), torch.stack(score_window), ape)

    if extra_data is None:
        run_write_rows()
        run_compress_rows()
    else:
        run_compress_rows()
        run_write_rows()
    return out, buffer

def _ratio128_prefill_ref(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_rows: list[tuple[int, int, int, int]],
    write_rows: list[tuple[int, int, int, int]],
    head_dim: int,
    extra_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    buffer = kv_score_buffer.clone()
    out = torch.zeros((kv_score_input.shape[0], head_dim), dtype=kv_score_input.dtype)
    if extra_data is None:
        load_indices = indices
    else:
        load_indices = extra_data.reshape(-1)
    for ragged_id, batch_id, position, _window_len in write_rows:
        buffer[int(indices[batch_id].item()), position % 128] = kv_score_input[ragged_id]
    for ragged_id, batch_id, _position, window_len in compress_rows:
        index = int(load_indices[batch_id].item())
        kv_window = []
        score_window = []
        for i in range(128):
            src = buffer[index, i] if i < window_len else kv_score_input[ragged_id + i - 127]
            src = src.reshape(2, head_dim)
            kv_window.append(src[0])
            score_window.append(src[1])
        out[ragged_id] = _compress_ref(torch.stack(kv_window), torch.stack(score_window), ape)
    return out, buffer

def _compress_ref(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(score.float() + ape.float(), dim=0)
    return torch.sum(kv.float() * weights, dim=0)

def _fused_norm_rope_ref(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freqs: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    normalized = (kv.float() * torch.rsqrt(kv.float().pow(2).mean(dim=-1, keepdim=True) + eps)) * weight.float()
    rope_dim = freqs.shape[-1] * 2
    rope_part = normalized if normalized.shape[-1] == rope_dim else normalized[..., -rope_dim:]
    reshaped = rope_part.reshape(rope_part.shape[0], -1, 2)
    rotated = torch.view_as_real(
        torch.view_as_complex(reshaped.contiguous()) * freqs[positions.long()].to(torch.complex64)
    ).flatten(-2)
    if normalized.shape[-1] == rope_dim:
        normalized = rotated
    else:
        normalized[..., -rope_dim:] = rotated
    return normalized.to(kv.dtype)

def _rope_reference(x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(x.shape[0], x.shape[1], x.shape[2] // 2, 2).contiguous())
    freqs_for_pos = freqs[positions.long()].unsqueeze(1).to(torch.complex64)
    rotated = x_complex / freqs_for_pos if inverse else x_complex * freqs_for_pos
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)

def _hadamard_reference(x: torch.Tensor, scale: float) -> torch.Tensor:
    out = x.float().clone()
    step = 1
    while step < out.shape[-1]:
        reshaped = out.reshape(*out.shape[:-1], -1, step * 2)
        left = reshaped[..., :step].clone()
        right = reshaped[..., step : step * 2].clone()
        reshaped[..., :step] = left + right
        reshaped[..., step : step * 2] = left - right
        step *= 2
    return out.mul_(scale).to(x.dtype)

def _rope_hadamard_reference(
    q: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    scale: float = 128.0 ** -0.5,
) -> torch.Tensor:
    rotated = q.float().clone()
    rotated[..., -64:] = _rope_reference(q[..., -64:], freqs, positions).float()
    return _hadamard_reference(rotated, scale).to(q.dtype)

def test_rmsnorm_self_musa_matches_reference() -> None:
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    eps = 1e-5

    got = rmsnorm_self_musa(q, eps)

    ref = (q.float() * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + eps)).to(q.dtype)
    torch.testing.assert_close(got, ref)

def test_rmsnorm_self_musa_handles_zero_and_negative_rows() -> None:
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0], [-1.0, 2.0, -3.0, 4.0]], dtype=torch.bfloat16)
    eps = 1e-3

    got = rmsnorm_self_musa(q, eps)

    ref = (q.float() * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + eps)).to(q.dtype)
    torch.testing.assert_close(got, ref)

def test_rmsnorm_self_musa_invokes_tilelang_fast_path(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_rmsnorm_self_musa(q: torch.Tensor, eps: float) -> torch.Tensor:
        calls.append((q, eps))
        return torch.full_like(q, 0.25)

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_rmsnorm_self_musa", fake_try_tilelang_rmsnorm_self_musa)

    q = torch.ones((2, 4), dtype=torch.bfloat16)
    got = rmsnorm_self_musa(q, 1e-5)

    assert len(calls) == 1
    assert calls[0][1] == 1e-5
    torch.testing.assert_close(got, torch.full_like(q, 0.25), rtol=0, atol=0)

def test_rmsnorm_self_musa_tilelang_miss_requires_fallback_env(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMusaTensor(torch.Tensor):
        @property
        def device(self) -> torch.device:
            return torch.device("musa")

    q = torch.ones((2, 4), dtype=torch.bfloat16).as_subclass(FakeMusaTensor)
    monkeypatch.setitem(rmsnorm_self_musa.__globals__, "_try_tilelang_rmsnorm_self_musa", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(rmsnorm_self_musa.__globals__, "_musa_graph_capture_enabled", lambda: False)
    monkeypatch.delenv("SGLANG_MUSA_ALLOW_TORCH_FALLBACK", raising=False)
    monkeypatch.delenv("SGLANG_DEBUG_MUSA_ALLOW_TORCH_FALLBACK", raising=False)

    with pytest.raises(NotImplementedError, match="rmsnorm_self has no torch fallback by default"):
        rmsnorm_self_musa(q, 1e-5)

    monkeypatch.setenv("SGLANG_MUSA_ALLOW_TORCH_FALLBACK", "1")
    with pytest.raises(NotImplementedError, match="rmsnorm_self has no torch fallback by default"):
        rmsnorm_self_musa(q, 1e-5)

def test_rmsnorm_self_musa_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 6.0]], device=device, dtype=torch.bfloat16)

    got = MUSA_OPS._try_tilelang_rmsnorm_self_musa(q, 1e-5)

    assert got is not None
    ref = (q.float() * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)).to(q.dtype)
    torch.testing.assert_close(got.cpu(), ref.cpu())

def test_rmsnorm_self_musa_real_tilelang_handles_strided_rows_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    base = torch.arange(2 * 8, device=device, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16) / 17
    q = base[:, :4]
    assert q.stride(-1) == 1
    assert not q.is_contiguous()

    got = MUSA_OPS._try_tilelang_rmsnorm_self_musa(q, 1e-5)

    assert got is not None
    ref = (q.float() * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)).to(q.dtype)
    torch.testing.assert_close(got.cpu(), ref.cpu())

@pytest.mark.parametrize("num_tokens", [1, 32, 65])
def test_rmsnorm_self_musa_real_tilelang_is_repeat_deterministic_on_musa(
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20262025 + num_tokens)
    q = torch.randn((num_tokens, 128), device=device, dtype=torch.bfloat16)

    expected = MUSA_OPS._try_tilelang_rmsnorm_self_musa(q, 1e-5)
    assert expected is not None
    torch.musa.synchronize()
    for repeat_idx in range(20):
        actual = MUSA_OPS._try_tilelang_rmsnorm_self_musa(q, 1e-5)
        assert actual is not None
        torch.musa.synchronize()
        _assert_repeat_exact(
            f"rmsnorm_self repeat={repeat_idx}, num_tokens={num_tokens}",
            actual,
            expected,
        )

@pytest.mark.parametrize("num_tokens", [32, 65])
def test_rmsnorm_self_musa_real_tilelang_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20262125 + num_tokens)
    probe = torch.randn((128,), device=device, dtype=torch.bfloat16)

    baseline = MUSA_OPS._try_tilelang_rmsnorm_self_musa(probe.view(1, 128), 1e-5)
    assert baseline is not None
    torch.musa.synchronize()
    expected = baseline[0].detach().clone()

    q = torch.randn((num_tokens, 128), device=device, dtype=torch.bfloat16)
    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    for pos in probe_positions:
        q[pos].copy_(probe)

    actual = MUSA_OPS._try_tilelang_rmsnorm_self_musa(q, 1e-5)
    assert actual is not None
    torch.musa.synchronize()
    for pos in probe_positions:
        _assert_repeat_exact(
            f"rmsnorm_self num_tokens={num_tokens}, pos={pos}",
            actual[pos],
            expected,
        )

def test_fused_rope_musa_matches_complex_reference() -> None:
    q = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16)
    freqs = torch.tensor([[1.0 + 0.0j, 0.0 + 1.0j]], dtype=torch.complex64)
    positions = torch.tensor([0], dtype=torch.int64)

    expected_out = _rope_reference(q.clone(), freqs, positions)

    fused_rope_musa(q, None, freqs, positions)
    torch.testing.assert_close(q, expected_out)

def test_fused_rope_musa_inverse_matches_complex_reference() -> None:
    q = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, -0.25]]], dtype=torch.bfloat16)
    original = q.clone()
    freqs = torch.tensor([[0.0 + 1.0j, -1.0 + 0.0j]], dtype=torch.complex64)
    positions = torch.tensor([0], dtype=torch.int64)

    expected_complex = torch.view_as_complex(original.float().reshape(1, 2, 2, 2).contiguous())
    expected_out = torch.view_as_real(expected_complex / freqs[positions].unsqueeze(1)).flatten(-2).to(q.dtype)

    fused_rope_musa(q, None, freqs, positions, inverse=True)

    torch.testing.assert_close(q, expected_out)

def test_fused_rope_musa_invokes_tilelang_fast_path(monkeypatch) -> None:
    calls = []

    def fake_tilelang_rope_inplace_musa_guard(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        inverse: bool,
        name: str,
    ) -> tuple[bool, str]:
        return True, ""

    def fake_tilelang_rope_inplace_musa_result(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        inverse: bool,
        name: str,
    ) -> tuple[bool, str]:
        calls.append((x, freqs_cis, positions, inverse, name))
        x.fill_(3.0)
        return True, ""

    monkeypatch.setattr(
        MUSA_OPS,
        "_tilelang_rope_inplace_musa_guard",
        fake_tilelang_rope_inplace_musa_guard,
    )
    monkeypatch.setattr(MUSA_OPS, "_tilelang_rope_inplace_musa_result", fake_tilelang_rope_inplace_musa_result)

    q = torch.zeros((2, 1, 4), dtype=torch.bfloat16)
    freqs = torch.ones((2, 2), dtype=torch.complex64)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    fused_rope_musa(q, None, freqs, positions, inverse=True)

    assert len(calls) == 1
    assert calls[0][3] is True
    assert calls[0][4] == "q"
    torch.testing.assert_close(q, torch.full_like(q, 3.0), rtol=0, atol=0)

def test_fused_rope_musa_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    q = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 6.0]],
            [[-1.0, 3.0, 4.0, -2.0], [0.25, 0.5, -0.75, 1.25]],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    freqs = torch.tensor(
        [
            [1.0 + 0.0j, 0.0 + 1.0j],
            [0.0 + 1.0j, -1.0 + 0.0j],
        ],
        device=device,
        dtype=torch.complex64,
    )
    positions = torch.tensor([0, 1], device=device, dtype=torch.int64)

    got = q.clone()
    ok = MUSA_OPS._try_tilelang_rope_inplace_musa(got, freqs, positions, inverse=False)

    assert ok
    expected_complex = torch.view_as_complex(q.cpu().float().reshape(2, 2, 2, 2).contiguous())
    expected_freqs = freqs.cpu()[positions.cpu()].unsqueeze(1)
    expected = torch.view_as_real(expected_complex * expected_freqs).flatten(-2).to(q.dtype)
    torch.testing.assert_close(got.cpu(), expected)

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_fused_rope_musa_real_tilelang_is_repeat_deterministic_on_musa(
    dtype: torch.dtype,
    inverse: bool,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    num_tokens = 3
    rope_dim = 64
    torch.manual_seed(20262225 + int(inverse) + (0 if dtype == torch.float32 else 100))
    q = torch.randn((num_tokens, 2, rope_dim), device=device, dtype=dtype)
    freqs = torch.polar(
        torch.ones((32, rope_dim // 2), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.25, steps=32 * (rope_dim // 2), device=device, dtype=torch.float32).reshape(
            32, rope_dim // 2
        ),
    )
    positions = torch.tensor([3, 7, 11], device=device, dtype=torch.int32)

    expected = q.clone()
    assert MUSA_OPS._try_tilelang_rope_inplace_musa(expected, freqs, positions, inverse=inverse)
    torch.musa.synchronize()
    for repeat_idx in range(20):
        actual = q.clone()
        assert MUSA_OPS._try_tilelang_rope_inplace_musa(actual, freqs, positions, inverse=inverse)
        torch.musa.synchronize()
        _assert_repeat_exact(
            f"rope repeat={repeat_idx}, dtype={dtype}, inverse={inverse}",
            actual,
            expected,
        )

@pytest.mark.parametrize("num_tokens", [1, 3, 17])
def test_fused_rope_musa_real_tilelang_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    rope_dim = 64
    probe_position = 13
    torch.manual_seed(20262325 + num_tokens)
    probe = torch.randn((2, rope_dim), device=device, dtype=torch.bfloat16)
    freqs = torch.polar(
        torch.ones((64, rope_dim // 2), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.31, steps=64 * (rope_dim // 2), device=device, dtype=torch.float32).reshape(
            64, rope_dim // 2
        ),
    )

    baseline = probe.view(1, 2, rope_dim).clone()
    assert MUSA_OPS._try_tilelang_rope_inplace_musa(
        baseline,
        freqs,
        torch.tensor([probe_position], device=device, dtype=torch.int32),
        inverse=False,
    )
    torch.musa.synchronize()
    expected = baseline[0].detach().clone()

    q = torch.randn((num_tokens, 2, rope_dim), device=device, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int32) + 1
    probe_positions = sorted({0, min(5, num_tokens - 1), num_tokens - 1})
    for pos in probe_positions:
        q[pos].copy_(probe)
        positions[pos] = probe_position

    assert MUSA_OPS._try_tilelang_rope_inplace_musa(q, freqs, positions, inverse=False)
    torch.musa.synchronize()
    for pos in probe_positions:
        _assert_repeat_exact(
            f"rope num_tokens={num_tokens}, pos={pos}",
            q[pos],
            expected,
        )

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("heads_per_block", [1, 2, 4])
@pytest.mark.parametrize("pingpong", [False, True])
def test_fused_rope_hadamard_musa_real_tilelang_matches_reference_on_musa(
    dtype: torch.dtype,
    positions_dtype: torch.dtype,
    num_tokens: int,
    heads_per_block: int,
    pingpong: bool,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_heads = 64
    torch.manual_seed(0)
    q = torch.randn((num_tokens, num_heads, 128), device=device, dtype=dtype)
    angles = torch.linspace(
        0.0,
        0.25,
        steps=max(num_tokens, 1) * 32,
        device=device,
        dtype=torch.float32,
    ).reshape(max(num_tokens, 1), 32)
    freqs = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)
    positions = torch.arange(num_tokens, device=device, dtype=positions_dtype)

    got = q.clone()
    ok, reason = MUSA_OPS._try_tilelang_rope_hadamard_inplace_musa(
        got,
        freqs,
        positions,
        heads_per_block=heads_per_block,
        pingpong=pingpong,
    )

    assert ok, reason
    expected = _rope_hadamard_reference(q.cpu(), freqs.cpu(), positions.cpu())
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(got.cpu(), expected, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("heads_per_block", [1, 4, 8])
def test_fused_rope_hadamard_fast_musa_real_tilelang_matches_reference_on_musa(
    dtype: torch.dtype,
    positions_dtype: torch.dtype,
    num_tokens: int,
    heads_per_block: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_heads = 64
    torch.manual_seed(0)
    q = torch.randn((num_tokens, num_heads, 128), device=device, dtype=dtype)
    angles = torch.linspace(
        0.0,
        0.25,
        steps=max(num_tokens, 1) * 32,
        device=device,
        dtype=torch.float32,
    ).reshape(max(num_tokens, 1), 32)
    freqs = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)
    positions = torch.arange(num_tokens, device=device, dtype=positions_dtype)

    got = q.clone()
    ok, reason = MUSA_OPS._try_tilelang_rope_hadamard_inplace_musa(
        got,
        freqs,
        positions,
        heads_per_block=heads_per_block,
        pingpong=False,
    )

    assert ok, reason
    expected = _rope_hadamard_reference(q.cpu(), freqs.cpu(), positions.cpu())
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(got.cpu(), expected, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 128), (16, 128), (128, 64, 128)])
@pytest.mark.parametrize("threads", [16, 32])
def test_hadamard128_musa_real_tilelang_matches_reference_on_musa(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    threads: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(0)
    x = torch.randn(shape, device=device, dtype=dtype)

    got = x.clone()
    ok, reason = MUSA_OPS._try_tilelang_hadamard128_inplace_musa(got, threads=threads)

    assert ok, reason
    expected = _hadamard_reference(x.cpu(), 128.0 ** -0.5)
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(got.cpu(), expected, rtol=rtol, atol=atol)

@pytest.mark.parametrize("threads", [16, 32])
def test_hadamard128_musa_real_tilelang_is_repeat_deterministic_on_musa(
    threads: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20262425 + threads)
    x = torch.randn((17, 128), device=device, dtype=torch.bfloat16)

    expected = x.clone()
    ok, reason = MUSA_OPS._try_tilelang_hadamard128_inplace_musa(expected, threads=threads)
    assert ok, reason
    torch.musa.synchronize()
    for repeat_idx in range(20):
        actual = x.clone()
        ok, reason = MUSA_OPS._try_tilelang_hadamard128_inplace_musa(actual, threads=threads)
        assert ok, reason
        torch.musa.synchronize()
        _assert_repeat_exact(
            f"hadamard128 repeat={repeat_idx}, threads={threads}",
            actual,
            expected,
        )

def test_hadamard128_musa_real_tilelang_probe_row_is_batch_shape_invariant_on_musa() -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20262525)
    probe = torch.randn((128,), device=device, dtype=torch.bfloat16)

    baseline = probe.view(1, 128).clone()
    ok, reason = MUSA_OPS._try_tilelang_hadamard128_inplace_musa(baseline, threads=16)
    assert ok, reason
    torch.musa.synchronize()
    expected = baseline[0].detach().clone()

    x = torch.randn((65, 128), device=device, dtype=torch.bfloat16)
    probe_positions = [0, 17, 64]
    for pos in probe_positions:
        x[pos].copy_(probe)

    ok, reason = MUSA_OPS._try_tilelang_hadamard128_inplace_musa(x, threads=16)
    assert ok, reason
    torch.musa.synchronize()
    for pos in probe_positions:
        _assert_repeat_exact(f"hadamard128 pos={pos}", x[pos], expected)

@pytest.mark.parametrize("num_tokens", [16, 128])
def test_fused_rope_hadamard_musa_real_tilelang_is_repeat_deterministic_on_musa(
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    num_heads = 64
    torch.manual_seed(20262625 + num_tokens)
    q = torch.randn((num_tokens, num_heads, 128), device=device, dtype=torch.bfloat16)
    freqs = torch.polar(
        torch.ones((num_tokens + 8, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.19, steps=(num_tokens + 8) * 32, device=device, dtype=torch.float32).reshape(
            num_tokens + 8, 32
        ),
    )
    positions = torch.arange(num_tokens, device=device, dtype=torch.int32)

    expected = q.clone()
    ok, reason = MUSA_OPS._try_tilelang_rope_hadamard_inplace_musa(
        expected,
        freqs,
        positions,
        heads_per_block=1 if num_tokens < 128 else 4,
        pingpong=False,
    )
    assert ok, reason
    torch.musa.synchronize()
    for repeat_idx in range(10):
        actual = q.clone()
        ok, reason = MUSA_OPS._try_tilelang_rope_hadamard_inplace_musa(
            actual,
            freqs,
            positions,
            heads_per_block=1 if num_tokens < 128 else 4,
            pingpong=False,
        )
        assert ok, reason
        torch.musa.synchronize()
        _assert_repeat_exact(
            f"rope_hadamard repeat={repeat_idx}, num_tokens={num_tokens}",
            actual,
            expected,
        )

def test_fused_rope_musa_real_tilelang_handles_production_tail_views_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    rope_dim = 64
    q_full = torch.arange(1 * 8 * 512, device=device, dtype=torch.float32).reshape(1, 8, 512).to(torch.bfloat16) / 997
    kv_full = torch.arange(1 * 512, device=device, dtype=torch.float32).reshape(1, 512).to(torch.bfloat16) / 257
    freqs = torch.ones((1048576, rope_dim // 2), device=device, dtype=torch.complex64)
    positions = torch.zeros((1,), device=device, dtype=torch.int64)

    q = q_full[..., -rope_dim:]
    k = kv_full[..., -rope_dim:].unsqueeze(1)
    assert tuple(q.shape) == (1, 8, 64)
    assert tuple(q.stride()) == (4096, 512, 1)
    assert tuple(k.shape) == (1, 1, 64)
    assert tuple(k.stride()) == (512, 64, 1)
    assert not q.is_contiguous()
    assert k.is_contiguous()

    got_q_full = q_full.clone()
    got_kv_full = kv_full.clone()
    got_q = got_q_full[..., -rope_dim:]
    got_k = got_kv_full[..., -rope_dim:].unsqueeze(1)

    MUSA_OPS.fused_rope_musa(got_q, got_k, freqs, positions)

    expected_q = q.cpu().float().reshape(1, 8, rope_dim // 2, 2)
    expected_k = k.cpu().float().reshape(1, 1, rope_dim // 2, 2)
    expected_q_full = q_full.cpu().clone()
    expected_kv_full = kv_full.cpu().clone()
    expected_q_full[..., -rope_dim:] = torch.view_as_real(
        torch.view_as_complex(expected_q.contiguous()) * freqs.cpu()[positions.cpu()].unsqueeze(1)
    ).flatten(-2).to(expected_q_full.dtype)
    expected_kv_full[..., -rope_dim:] = torch.view_as_real(
        torch.view_as_complex(expected_k.contiguous()) * freqs.cpu()[positions.cpu()].unsqueeze(1)
    ).flatten(-2).to(expected_kv_full.dtype).squeeze(1)

    torch.testing.assert_close(got_q_full.cpu(), expected_q_full, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(got_kv_full.cpu(), expected_kv_full, rtol=1e-2, atol=1e-2)

def test_compress_fused_norm_rope_prefill_inplace_musa_real_tilelang_handles_int32_plan_positions_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [2.0, -1.0, 0.5, 6.0],
            [-3.0, 1.0, 2.0, -0.5],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([1.0, 1.5, 0.5, 2.0], device=device, dtype=torch.bfloat16)
    freqs = torch.tensor(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 1.0j, 1.0 + 0.0j],
            [-1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 - 1.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, -1.0 + 0.0j],
            [0.0 + 1.0j, -1.0 + 0.0j],
        ],
        device=device,
        dtype=torch.complex64,
    )
    rows = [(2, 0, 1, 7), (0, 1, 4, 4)]
    compress_plan = _pack_prefill_rows(rows).to(device)

    got = kv.clone()
    MUSA_OPS.compress_fused_norm_rope_prefill_inplace_musa(got, weight, 1e-5, freqs, compress_plan)

    expected = kv.cpu().clone()
    ragged_ids = torch.tensor([2, 0], dtype=torch.int64)
    positions = torch.tensor([1, 4], dtype=torch.int32)
    expected[ragged_ids] = _fused_norm_rope_ref(
        kv.cpu().index_select(0, ragged_ids),
        weight.cpu(),
        1e-5,
        freqs.cpu(),
        positions,
    )
    torch.testing.assert_close(got.cpu(), expected)

def test_compress_fused_norm_rope_prefill_inplace_musa_real_tilelang_handles_fp32_production_shape_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.arange(64 * 128, device=device, dtype=torch.float32).reshape(64, 128) / 4099
    weight = torch.linspace(0.75, 1.25, steps=128, device=device, dtype=torch.float32)
    freqs = torch.polar(
        torch.ones((1048576, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.25, steps=1048576 * 32, device=device, dtype=torch.float32).reshape(1048576, 32),
    )
    rows = [(i * 2, 0, i, 128 - i) for i in range(32)]
    compress_plan = _pack_prefill_rows(rows).to(device)

    got = kv.clone()
    MUSA_OPS.compress_fused_norm_rope_prefill_inplace_musa(got, weight, 1e-5, freqs, compress_plan)

    expected = kv.cpu().clone()
    ragged_ids = torch.tensor([row[0] for row in rows], dtype=torch.int64)
    positions = torch.tensor([row[2] for row in rows], dtype=torch.int32)
    selected = kv.cpu().index_select(0, ragged_ids)
    normalized = (
        selected.float()
        * torch.rsqrt(selected.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ) * weight.cpu().float()
    rope_part = normalized[..., -64:]
    rotated = torch.view_as_real(
        torch.view_as_complex(rope_part.reshape(rope_part.shape[0], -1, 2).contiguous())
        * freqs.cpu()[positions.long()].to(torch.complex64)
    ).flatten(-2)
    normalized[..., -64:] = rotated
    expected[ragged_ids] = normalized.to(kv.dtype)
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)

def test_compress_fused_norm_rope_prefill_inplace_musa_real_tilelang_handles_fp32_production_shape_nontrivial_freqs_on_musa(
    monkeypatch,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.arange(12 * 512, device=device, dtype=torch.float32).reshape(12, 512) / 4099
    weight = torch.linspace(0.75, 1.25, steps=512, device=device, dtype=torch.float32)
    freqs = torch.polar(
        torch.ones((1048576, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.13, steps=1048576 * 32, device=device, dtype=torch.float32).reshape(1048576, 32),
    )
    rows = [(3, 0, 3, 4), (7, 1, 3, 4), (11, 2, 3, 4)]
    compress_plan = _pack_prefill_rows(rows).to(device)

    def fail_fused(*_args, **_kwargs):
        raise AssertionError("h512 compress prefill should not use the slow fused kernel")

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_fused_norm_rope_inplace_musa", fail_fused)

    got = kv.clone()
    MUSA_OPS.compress_fused_norm_rope_prefill_inplace_musa(got, weight, 1e-5, freqs, compress_plan)

    expected = kv.cpu().clone()
    ragged_ids = torch.tensor([row[0] for row in rows], dtype=torch.int64)
    positions = torch.tensor([row[2] for row in rows], dtype=torch.int32)
    selected = kv.cpu().index_select(0, ragged_ids)
    normalized = (
        selected.float()
        * torch.rsqrt(selected.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ) * weight.cpu().float()
    rope_part = normalized[..., -64:]
    rotated = torch.view_as_real(
        torch.view_as_complex(rope_part.reshape(rope_part.shape[0], -1, 2).contiguous())
        * freqs.cpu()[positions.long()].to(torch.complex64)
    ).flatten(-2)
    normalized[..., -64:] = rotated
    expected[ragged_ids] = normalized.to(kv.dtype)
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)

def test_compress_fused_norm_rope_prefill_inplace_musa_real_tilelang_fallbacks_h256_r64_on_musa(
    monkeypatch,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(2026060402)
    num_tokens = 768
    hidden_size = 256
    rope_dim = 64
    kv = (torch.randn((num_tokens, hidden_size), device=device) * 0.2).to(torch.bfloat16)
    weight = (torch.randn((hidden_size,), device=device) * 0.2 + 1.0).to(torch.bfloat16)
    freqs = torch.polar(
        torch.ones((num_tokens + 4, rope_dim // 2), device=device, dtype=torch.float32),
        torch.linspace(
            0.0,
            0.25,
            steps=(num_tokens + 4) * (rope_dim // 2),
            device=device,
            dtype=torch.float32,
        ).reshape(num_tokens + 4, rope_dim // 2),
    )
    rows = [(i, 0, i, 0) for i in range(3, num_tokens, 4)]
    compress_plan = _pack_prefill_rows(rows).to(device)

    def fail_direct(*_args, **_kwargs):
        raise AssertionError("h256/r64 must not use the direct prefill TileLang path")

    monkeypatch.setattr(
        MUSA_OPS,
        "_tilelang_compress_fused_norm_rope_prefill_inplace_kernel",
        fail_direct,
    )

    got = kv.clone()
    MUSA_OPS.compress_fused_norm_rope_prefill_inplace_musa(
        got, weight, 1e-5, freqs, compress_plan
    )

    expected = kv.clone()
    ragged_ids = torch.tensor([row[0] for row in rows], device=device, dtype=torch.long)
    positions = torch.tensor([row[2] for row in rows], device=device, dtype=torch.int32)
    transformed = expected.index_select(0, ragged_ids).clone()
    MUSA_OPS.fused_norm_rope_inplace_musa(transformed, weight, 1e-5, freqs, positions)
    expected.index_copy_(0, ragged_ids, transformed)
    torch.musa.synchronize()
    torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=0, atol=0)

def test_fused_rope_musa_real_tilelang_handles_p8_int32_positions_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    rope_dim = 64
    q_full = torch.arange(1 * 64 * 128, device=device, dtype=torch.float32).reshape(1, 64, 128).to(torch.bfloat16) / 4099
    freqs = torch.ones((1048576, rope_dim // 2), device=device, dtype=torch.complex64)
    positions = torch.zeros((1,), device=device, dtype=torch.int32)

    q = q_full[..., 64:128]
    assert tuple(q.shape) == (1, 64, 64)
    assert tuple(q.stride()) == (8192, 128, 1)
    assert q.storage_offset() == 64
    assert not q.is_contiguous()

    got_q_full = q_full.clone()
    got_q = got_q_full[..., 64:128]

    MUSA_OPS.fused_rope_musa(got_q, None, freqs, positions)

    expected_q = q.cpu().float().reshape(1, 64, rope_dim // 2, 2)
    expected_q_full = q_full.cpu().clone()
    expected_q_full[..., 64:128] = torch.view_as_real(
        torch.view_as_complex(expected_q.contiguous()) * freqs.cpu()[positions.cpu().long()].unsqueeze(1)
    ).flatten(-2).to(expected_q_full.dtype)

    torch.testing.assert_close(got_q_full.cpu(), expected_q_full, rtol=1e-2, atol=1e-2)

def test_fused_rope_musa_real_tilelang_handles_float32_strided_positions_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    rope_dim = 64
    q_full = torch.randn((3, 2, rope_dim), device=device, dtype=torch.float32)
    q = q_full[:, 1:2, :]
    freqs = torch.polar(
        torch.ones((1048576, rope_dim // 2), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.25, steps=1048576 * (rope_dim // 2), device=device, dtype=torch.float32).reshape(
            1048576, rope_dim // 2
        ),
    )
    positions_storage = torch.arange(16, device=device, dtype=torch.int32)
    positions = positions_storage.as_strided((3,), (4,), storage_offset=2)

    assert tuple(q.shape) == (3, 1, 64)
    assert tuple(q.stride()) == (128, 64, 1)
    assert q.dtype == torch.float32
    assert tuple(positions.stride()) == (4,)
    assert positions.storage_offset() == 2

    got_q_full = q_full.clone()
    got_q = got_q_full[:, 1:2, :]
    ok, reason = MUSA_OPS._tilelang_rope_inplace_musa_result(got_q, freqs, positions, False, "q")

    assert ok, reason
    expected_q_full = q_full.cpu().clone()
    expected_q_full[:, 1:2, :] = _rope_reference(q.cpu(), freqs.cpu(), positions.cpu())
    torch.testing.assert_close(got_q_full.cpu(), expected_q_full, rtol=1e-5, atol=1e-5)

def test_fused_rope_musa_real_tilelang_float32_does_not_use_reference_fallback_on_musa(monkeypatch) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    q = torch.randn((3, 1, 64), device=device, dtype=torch.float32)
    freqs = torch.ones((1048576, 32), device=device, dtype=torch.complex64)
    positions_storage = torch.arange(16, device=device, dtype=torch.int32)
    positions = positions_storage.as_strided((3,), (4,), storage_offset=2)

    def fail_reference(*_args, **_kwargs):
        raise AssertionError("float32 RoPE should use TileLang, not reference fallback")

    monkeypatch.setattr(MUSA_OPS, "_apply_rope_inplace_real_imag", fail_reference)
    MUSA_OPS.fused_rope_musa(q, None, freqs, positions)

def test_fused_rope_musa_real_tilelang_rejects_half_dim_over_32_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    q = torch.zeros((1, 1, 66), device=device, dtype=torch.bfloat16)
    freqs = torch.ones((8, 33), device=device, dtype=torch.complex64)
    positions = torch.zeros((1,), device=device, dtype=torch.int64)

    ok, reason = MUSA_OPS._tilelang_rope_inplace_musa_result(q, freqs, positions, False, "q")

    assert not ok
    assert "half_dim=33 exceeds TileLang rope thread limit (32)" in reason

@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "backing_dim", "rope_dim", "position_values", "position_dtype"),
    [
        (1, 8, 512, 64, [0], torch.int64),
        (2, 4, 128, 32, [3, 1024], torch.int64),
        (3, 1, 96, 48, [7, 31, 4095], torch.int32),
    ],
)
def test_fused_rope_musa_real_tilelang_tail_view_shape_matrix_on_musa(
    num_tokens: int,
    num_heads: int,
    backing_dim: int,
    rope_dim: int,
    position_values: list[int],
    position_dtype: torch.dtype,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    q_full = (
        torch.arange(num_tokens * num_heads * backing_dim, device=device, dtype=torch.float32)
        .reshape(num_tokens, num_heads, backing_dim)
        .to(torch.bfloat16)
        / 4099
    )
    k_full = (
        torch.arange(num_tokens * backing_dim, device=device, dtype=torch.float32)
        .reshape(num_tokens, backing_dim)
        .to(torch.bfloat16)
        / 8191
    )
    freqs = torch.polar(
        torch.ones((1048576, rope_dim // 2), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.25, steps=1048576 * (rope_dim // 2), device=device, dtype=torch.float32).reshape(1048576, rope_dim // 2),
    )
    positions = torch.tensor(position_values, device=device, dtype=position_dtype)
    q = q_full[..., backing_dim - rope_dim :]
    k = k_full[..., backing_dim - rope_dim :].unsqueeze(1)

    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert tuple(positions.shape) == (num_tokens,)
    assert freqs.shape[-1] == rope_dim // 2
    assert not q.is_contiguous()

    got_q_full = q_full.clone()
    got_k_full = k_full.clone()
    got_q = got_q_full[..., backing_dim - rope_dim :]
    got_k = got_k_full[..., backing_dim - rope_dim :].unsqueeze(1)

    fused_rope_musa(got_q, got_k, freqs, positions)

    expected_q_full = q_full.cpu().clone()
    expected_k_full = k_full.cpu().clone()
    expected_q_full[..., backing_dim - rope_dim :] = _rope_reference(q.cpu(), freqs.cpu(), positions.cpu())
    expected_k_full[..., backing_dim - rope_dim :] = _rope_reference(k.cpu(), freqs.cpu(), positions.cpu()).squeeze(1)
    torch.testing.assert_close(got_q_full.cpu(), expected_q_full, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(got_k_full.cpu(), expected_k_full, rtol=1e-2, atol=1e-2)

@pytest.mark.parametrize(
    ("case", "q_factory", "k_factory", "freqs_factory", "positions_factory", "match"),
    [
        (
            "q dtype",
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.float16),
            lambda device: None,
            lambda device: torch.ones((4, 32), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "q dtype",
        ),
        (
            "q dim",
            lambda device: torch.zeros((1, 64), device=device, dtype=torch.bfloat16),
            lambda device: None,
            lambda device: torch.ones((4, 32), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "q dim",
        ),
        (
            "q last stride",
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.bfloat16).as_strided((1, 1, 32), (64, 64, 2)),
            lambda device: None,
            lambda device: torch.ones((4, 16), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "last-dim stride",
        ),
        (
            "freq device",
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.bfloat16),
            lambda device: None,
            lambda device: torch.ones((4, 32), dtype=torch.complex64),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "freqs_cis device",
        ),
        (
            "position device",
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.bfloat16),
            lambda device: None,
            lambda device: torch.ones((4, 32), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1,), dtype=torch.int64),
            "positions device",
        ),
        (
            "freq dtype",
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.bfloat16),
            lambda device: None,
            lambda device: torch.ones((4, 32), device=device, dtype=torch.complex128),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "freqs_cis dtype",
        ),
        (
            "position dim",
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.bfloat16),
            lambda device: None,
            lambda device: torch.ones((4, 32), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1, 1), device=device, dtype=torch.int64),
            "positions dim",
        ),
        (
            "position length",
            lambda device: torch.zeros((2, 1, 64), device=device, dtype=torch.bfloat16),
            lambda device: None,
            lambda device: torch.ones((4, 32), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "positions length",
        ),
        (
            "odd head dim",
            lambda device: torch.zeros((1, 1, 63), device=device, dtype=torch.bfloat16),
            lambda device: None,
            lambda device: torch.ones((4, 31), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "head_dim is odd",
        ),
        (
            "freq half dim",
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.bfloat16),
            lambda device: None,
            lambda device: torch.ones((4, 31), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "freqs_cis half dim",
        ),
        (
            "k stride",
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.bfloat16),
            lambda device: torch.zeros((1, 1, 64), device=device, dtype=torch.bfloat16).as_strided((1, 1, 32), (64, 64, 2)),
            lambda device: torch.ones((4, 32), device=device, dtype=torch.complex64),
            lambda device: torch.zeros((1,), device=device, dtype=torch.int64),
            "k last-dim stride",
        ),
    ],
)
def test_fused_rope_musa_real_tilelang_rejects_unsupported_shapes_without_torch_fallback(
    case: str,
    q_factory,
    k_factory,
    freqs_factory,
    positions_factory,
    match: str,
) -> None:
    del case
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA fail-closed validation")
    device = get_musa_device()
    q = q_factory(device)
    k = k_factory(device)
    freqs = freqs_factory(device)
    positions = positions_factory(device)

    with pytest.raises(NotImplementedError, match=match):
        fused_rope_musa(q, k, freqs, positions)

def test_fused_norm_rope_inplace_musa_invokes_tilelang_fast_path(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_fused_norm_rope_inplace_musa(
        kv: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        freq_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[bool, str | None]:
        calls.append((kv, weight, eps, freq_cis, positions))
        kv.fill_(4.0)
        return True, None

    monkeypatch.setattr(
        MUSA_OPS,
        "_try_tilelang_fused_norm_rope_inplace_musa",
        fake_try_tilelang_fused_norm_rope_inplace_musa,
    )

    kv = torch.zeros((2, 4), dtype=torch.bfloat16)
    weight = torch.ones((4,), dtype=torch.bfloat16)
    freqs = torch.ones((2, 2), dtype=torch.complex64)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    MUSA_OPS.fused_norm_rope_inplace_musa(kv, weight, 1e-5, freqs, positions)

    assert len(calls) == 1
    assert calls[0][2] == 1e-5
    torch.testing.assert_close(kv, torch.full_like(kv, 4.0), rtol=0, atol=0)

def test_fused_norm_rope_inplace_musa_skips_slow_fused_kernel_for_hidden_gt_128(monkeypatch) -> None:
    calls = {"rmsnorm": 0, "rope": 0}

    def fail_try_fused(*_args, **_kwargs):
        raise AssertionError("h512 public path should use decomposed TileLang RMSNorm + RoPE until fused kernel is faster")

    def fake_rmsnorm_self_musa(kv: torch.Tensor, eps: float) -> torch.Tensor:
        calls["rmsnorm"] += 1
        assert eps == 1e-5
        return torch.ones_like(kv)

    def fake_fused_rope_musa(
        q: torch.Tensor,
        k: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        inverse: bool = False,
    ) -> None:
        calls["rope"] += 1
        assert k is None
        assert q.shape == (2, 1, 64)
        q.fill_(7.0)

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_fused_norm_rope_inplace_musa", fail_try_fused)
    monkeypatch.setattr(MUSA_OPS, "rmsnorm_self_musa", fake_rmsnorm_self_musa)
    monkeypatch.setattr(MUSA_OPS, "fused_rope_musa", fake_fused_rope_musa)
    monkeypatch.setenv("SGLANG_DEEPSEEK_V4_MUSA_FUSED_NORM_ROPE_H512", "0")

    kv = torch.zeros((2, 512), dtype=torch.bfloat16)
    weight = torch.full((512,), 3.0, dtype=torch.bfloat16)
    freqs = torch.ones((16, 32), dtype=torch.complex64)
    positions = torch.tensor([0, 1], dtype=torch.int32)

    MUSA_OPS.fused_norm_rope_inplace_musa(kv, weight, 1e-5, freqs, positions)

    assert calls == {"rmsnorm": 1, "rope": 1}
    torch.testing.assert_close(kv[:, :-64], torch.full_like(kv[:, :-64], 3.0), rtol=0, atol=0)
    torch.testing.assert_close(kv[:, -64:], torch.full_like(kv[:, -64:], 7.0), rtol=0, atol=0)

def test_fused_norm_rope_inplace_musa_real_tilelang_public_wrapper_handles_float32_strided_positions_on_musa(
    monkeypatch,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.arange(3 * 1 * 64, device=device, dtype=torch.float32).reshape(3, 64) / 4099
    weight = torch.linspace(0.75, 1.25, steps=64, device=device, dtype=torch.float32)
    freqs = torch.polar(
        torch.ones((1048576, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.25, steps=1048576 * 32, device=device, dtype=torch.float32).reshape(1048576, 32),
    )
    positions_storage = torch.arange(16, device=device, dtype=torch.int32)
    positions = positions_storage.as_strided((3,), (4,), storage_offset=2)

    def fail_reference(*_args, **_kwargs):
        raise AssertionError("public fused_norm_rope should use TileLang fast path on supported MUSA input")

    monkeypatch.setattr(MUSA_OPS, "_apply_rope_inplace_real_imag", fail_reference)

    got = kv.clone()
    MUSA_OPS.fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, positions)

    expected = _fused_norm_rope_ref(kv.cpu(), weight.cpu(), 1e-5, freqs.cpu(), positions.cpu())
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)

def test_fused_norm_rope_inplace_musa_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 6.0]],
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([1.0, 1.5, 0.5, 2.0], device=device, dtype=torch.bfloat16)
    freqs = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 1.0j], [0.0 + 1.0j, -1.0 + 0.0j]],
        device=device,
        dtype=torch.complex64,
    )
    positions = torch.tensor([0, 1], device=device, dtype=torch.int64)

    got = kv.clone()
    ok, reason = MUSA_OPS._try_tilelang_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, positions)

    assert ok, reason
    expected = _fused_norm_rope_ref(kv.cpu(), weight.cpu(), 1e-5, freqs.cpu(), positions.cpu())
    torch.testing.assert_close(got.cpu(), expected)

@pytest.mark.parametrize("hidden_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_norm_rope_inplace_musa_real_tilelang_is_repeat_deterministic_on_musa(
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    num_tokens = 17
    torch.manual_seed(20262725 + hidden_size + (0 if dtype == torch.float32 else 1000))
    kv = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype)
    weight = torch.linspace(0.75, 1.25, steps=hidden_size, device=device, dtype=torch.float32).to(dtype)
    freqs = torch.polar(
        torch.ones((64, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.17, steps=64 * 32, device=device, dtype=torch.float32).reshape(64, 32),
    )
    positions = torch.arange(num_tokens, device=device, dtype=torch.int32) + 3

    expected = kv.clone()
    ok, reason = MUSA_OPS._try_tilelang_fused_norm_rope_inplace_musa(
        expected,
        weight,
        1e-5,
        freqs,
        positions,
    )
    assert ok, reason
    torch.musa.synchronize()
    for repeat_idx in range(20):
        actual = kv.clone()
        ok, reason = MUSA_OPS._try_tilelang_fused_norm_rope_inplace_musa(
            actual,
            weight,
            1e-5,
            freqs,
            positions,
        )
        assert ok, reason
        torch.musa.synchronize()
        _assert_repeat_exact(
            f"fused_norm_rope repeat={repeat_idx}, hidden_size={hidden_size}, dtype={dtype}",
            actual,
            expected,
        )

@pytest.mark.parametrize("num_tokens", [1, 32, 65])
def test_fused_norm_rope_inplace_musa_real_tilelang_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    hidden_size = 128
    probe_position = 19
    torch.manual_seed(20262825 + num_tokens)
    probe = torch.randn((hidden_size,), device=device, dtype=torch.bfloat16)
    weight = torch.linspace(0.75, 1.25, steps=hidden_size, device=device, dtype=torch.bfloat16)
    freqs = torch.polar(
        torch.ones((64, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.23, steps=64 * 32, device=device, dtype=torch.float32).reshape(64, 32),
    )

    baseline = probe.view(1, hidden_size).clone()
    ok, reason = MUSA_OPS._try_tilelang_fused_norm_rope_inplace_musa(
        baseline,
        weight,
        1e-5,
        freqs,
        torch.tensor([probe_position], device=device, dtype=torch.int32),
    )
    assert ok, reason
    torch.musa.synchronize()
    expected = baseline[0].detach().clone()

    kv = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int32) + 1
    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    for pos in probe_positions:
        kv[pos].copy_(probe)
        positions[pos] = probe_position

    ok, reason = MUSA_OPS._try_tilelang_fused_norm_rope_inplace_musa(
        kv,
        weight,
        1e-5,
        freqs,
        positions,
    )
    assert ok, reason
    torch.musa.synchronize()
    for pos in probe_positions:
        _assert_repeat_exact(
            f"fused_norm_rope num_tokens={num_tokens}, pos={pos}",
            kv[pos],
            expected,
        )

def test_fused_norm_rope_inplace_musa_real_tilelang_handles_size_one_strided_positions_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.arange(512, device=device, dtype=torch.float32).reshape(1, 512) / 4099
    weight = torch.linspace(0.75, 1.25, steps=512, device=device, dtype=torch.float32)
    freqs = torch.ones((1048576, 32), device=device, dtype=torch.complex64)
    positions_backing = torch.zeros((4,), device=device, dtype=torch.int32)
    positions = positions_backing.as_strided((1,), (4,))
    assert tuple(positions.shape) == (1,)
    assert tuple(positions.stride()) == (4,)
    assert positions.is_contiguous()

    got = kv.clone()
    ok, reason = MUSA_OPS._try_tilelang_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, positions)

    assert ok, reason
    expected = _fused_norm_rope_ref(kv.cpu(), weight.cpu(), 1e-5, freqs.cpu(), positions.cpu())
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_norm_rope_inplace_musa_real_tilelang_handles_strided_int32_positions_on_musa(dtype: torch.dtype) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = (torch.arange(32 * 128, device=device, dtype=torch.float32).reshape(32, 128) / 4099).to(dtype)
    weight = torch.linspace(0.75, 1.25, steps=128, device=device, dtype=torch.float32).to(dtype)
    freqs = torch.ones((1048576, 32), device=device, dtype=torch.complex64)
    positions_backing = torch.zeros((128,), device=device, dtype=torch.int32)
    positions_backing[3::4] = torch.arange(32, device=device, dtype=torch.int32)
    positions = positions_backing.as_strided((32,), (4,), storage_offset=3)
    assert positions.storage_offset() == 3
    assert tuple(positions.shape) == (32,)
    assert tuple(positions.stride()) == (4,)
    assert not positions.is_contiguous()

    got = kv.clone()
    ok, reason = MUSA_OPS._try_tilelang_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, positions)

    assert ok, reason
    expected = _fused_norm_rope_ref(kv.cpu(), weight.cpu(), 1e-5, freqs.cpu(), positions.cpu())
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_norm_rope_inplace_musa_real_tilelang_handles_strided_weight_and_freqs_on_musa(
    dtype: torch.dtype,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = (torch.arange(6 * 128, device=device, dtype=torch.float32).reshape(6, 128) / 4099).to(dtype)
    weight_backing = torch.empty((256,), device=device, dtype=dtype)
    weight_backing[1::2] = torch.linspace(0.75, 1.25, steps=128, device=device, dtype=torch.float32).to(dtype)
    weight = weight_backing.as_strided((128,), (2,), storage_offset=1)
    freqs_backing = torch.polar(
        torch.ones((32, 3, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.25, steps=32 * 3 * 32, device=device, dtype=torch.float32).reshape(32, 3, 32),
    )
    freqs = freqs_backing[:, 2, :]
    positions = torch.tensor([0, 1, 2, 3, 4, 5], device=device, dtype=torch.int32)
    assert tuple(weight.stride()) == (2,)
    assert not weight.is_contiguous()
    assert not freqs.is_contiguous()

    got = kv.clone()
    ok, reason = MUSA_OPS._try_tilelang_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, positions)

    assert ok, reason
    expected = _fused_norm_rope_ref(kv.cpu(), weight.cpu(), 1e-5, freqs.cpu(), positions.cpu())
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)

def test_compress_fused_norm_rope_inplace_musa_invokes_tilelang_fast_path(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_compress_fused_norm_rope_inplace_musa(
        kv: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        freq_cis: torch.Tensor,
        seq_lens: torch.Tensor,
        compress_ratio: int,
    ) -> tuple[bool, str | None]:
        calls.append((kv, weight, eps, freq_cis, seq_lens, compress_ratio))
        kv.fill_(5.0)
        return True, None

    monkeypatch.setattr(
        MUSA_OPS,
        "_try_tilelang_compress_fused_norm_rope_inplace_musa",
        fake_try_tilelang_compress_fused_norm_rope_inplace_musa,
    )

    kv = torch.zeros((2, 4), dtype=torch.bfloat16)
    weight = torch.ones((4,), dtype=torch.bfloat16)
    freqs = torch.ones((2, 2), dtype=torch.complex64)
    seq_lens = torch.tensor([4, 8], dtype=torch.int32)
    compress_fused_norm_rope_inplace_musa(kv, weight, 1e-5, freqs, seq_lens, compress_ratio=4)

    assert len(calls) == 1
    assert calls[0][5] == 4
    torch.testing.assert_close(kv, torch.full_like(kv, 5.0), rtol=0, atol=0)

def test_compress_fused_norm_rope_inplace_musa_ratio4_decode_failure_reports_metadata_on_musa(monkeypatch) -> None:
    device = get_musa_device()
    kv = torch.zeros((3, 128), device=device, dtype=torch.float32)
    weight = torch.ones((128,), device=device, dtype=torch.float32)
    freqs = torch.ones((16, 32), device=device, dtype=torch.complex64)
    seq_lens = torch.tensor([5, 4, 8], device=device, dtype=torch.int32)

    def fake_tilelang_kernel(*_args, **_kwargs):
        def kernel(*_args, **_kwargs) -> None:
            raise RuntimeError("worker35021 ratio4 decode repro")

        return kernel

    monkeypatch.setattr(MUSA_OPS, "_tilelang_compress_fused_norm_rope_inplace_kernel", fake_tilelang_kernel)

    with pytest.raises(NotImplementedError) as exc_info:
        compress_fused_norm_rope_inplace_musa(kv, weight, 1e-5, freqs, seq_lens, compress_ratio=4)

    message = str(exc_info.value)
    assert "DeepSeekV4 MUSA compress_fused_norm_rope has no torch fallback for supported MUSA input" in message
    assert "kernel exception RuntimeError: worker35021 ratio4 decode repro" in message
    assert f"kv=device:{kv.device},dtype:{kv.dtype},shape:{tuple(kv.shape)},stride:{kv.stride()},contiguous:{kv.is_contiguous()}" in message
    assert f"weight=device:{weight.device},dtype:{weight.dtype},shape:{tuple(weight.shape)},stride:{weight.stride()},contiguous:{weight.is_contiguous()}" in message
    assert f"freq_cis=device:{freqs.device},dtype:{freqs.dtype},shape:{tuple(freqs.shape)},stride:{freqs.stride()},contiguous:{freqs.is_contiguous()}" in message
    assert f"seq_lens=device:{seq_lens.device},dtype:{seq_lens.dtype},shape:{tuple(seq_lens.shape)},stride:{seq_lens.stride()},contiguous:{seq_lens.is_contiguous()}" in message
    assert "compress_ratio=4" in message

    del kv, weight, freqs, seq_lens
    torch.musa.synchronize()
    torch.musa.empty_cache()

def test_compress_fused_norm_rope_inplace_musa_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [2.0, -1.0, 0.5, 6.0]],
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([1.0, 1.5, 0.5, 2.0], device=device, dtype=torch.bfloat16)
    freqs = torch.tensor(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 1.0j, 1.0 + 0.0j],
            [-1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 - 1.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, -1.0 + 0.0j],
        ],
        device=device,
        dtype=torch.complex64,
    )
    seq_lens = torch.tensor([5, 4, 8], device=device, dtype=torch.int32)

    got = kv.clone()
    ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, seq_lens, 4)

    assert ok, failure
    expected = kv.cpu().clone()
    boundary_rows = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([0, 4], dtype=torch.int32)
    expected[boundary_rows] = _fused_norm_rope_ref(
        kv.cpu().index_select(0, boundary_rows),
        weight.cpu(),
        1e-5,
        freqs.cpu(),
        positions,
    )
    torch.testing.assert_close(got.cpu(), expected)

@pytest.mark.parametrize("hidden_size", [128, 512])
def test_compress_fused_norm_rope_inplace_musa_real_tilelang_is_repeat_deterministic_on_musa(
    hidden_size: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20263925 + hidden_size)
    kv = torch.randn((17, hidden_size), device=device, dtype=torch.float32)
    weight = torch.linspace(0.75, 1.25, steps=hidden_size, device=device, dtype=torch.float32)
    freqs = torch.polar(
        torch.ones((16, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.11, steps=16 * 32, device=device, dtype=torch.float32).reshape(16, 32),
    )
    seq_lens = torch.full((17,), 4, device=device, dtype=torch.int32)

    expected = kv.clone()
    ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(
        expected,
        weight,
        1e-5,
        freqs,
        seq_lens,
        4,
    )
    assert ok, failure
    torch.musa.synchronize()
    for repeat_idx in range(20):
        actual = kv.clone()
        ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(
            actual,
            weight,
            1e-5,
            freqs,
            seq_lens,
            4,
        )
        assert ok, failure
        torch.musa.synchronize()
        _assert_repeat_exact(
            f"compress_fused_norm_rope repeat={repeat_idx}, hidden_size={hidden_size}",
            actual,
            expected,
        )

@pytest.mark.parametrize("num_tokens", [1, 32, 65])
def test_compress_fused_norm_rope_inplace_musa_real_tilelang_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    hidden_size = 128
    torch.manual_seed(20264025 + num_tokens)
    probe = torch.randn((hidden_size,), device=device, dtype=torch.float32)
    weight = torch.linspace(0.75, 1.25, steps=hidden_size, device=device, dtype=torch.float32)
    freqs = torch.polar(
        torch.ones((16, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.13, steps=16 * 32, device=device, dtype=torch.float32).reshape(16, 32),
    )

    baseline = probe.view(1, hidden_size).clone()
    ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(
        baseline,
        weight,
        1e-5,
        freqs,
        torch.tensor([4], device=device, dtype=torch.int32),
        4,
    )
    assert ok, failure
    torch.musa.synchronize()
    expected = baseline[0].detach().clone()

    kv = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float32)
    seq_lens = torch.full((num_tokens,), 5, device=device, dtype=torch.int32)
    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    for pos in probe_positions:
        kv[pos].copy_(probe)
        seq_lens[pos] = 4

    ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(
        kv,
        weight,
        1e-5,
        freqs,
        seq_lens,
        4,
    )
    assert ok, failure
    torch.musa.synchronize()
    for pos in probe_positions:
        _assert_repeat_exact(
            f"compress_fused_norm_rope num_tokens={num_tokens}, pos={pos}",
            kv[pos],
            expected,
        )

def test_compress_fused_norm_rope_inplace_musa_real_tilelang_handles_strided_inputs_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.arange(3 * 128, device=device, dtype=torch.float32).reshape(3, 128) / 4099
    weight_backing = torch.empty((256,), device=device, dtype=torch.float32)
    weight_backing[1::2] = torch.linspace(0.75, 1.25, steps=128, device=device, dtype=torch.float32)
    weight = weight_backing.as_strided((128,), (2,), storage_offset=1)
    freqs_backing = torch.polar(
        torch.ones((16, 3, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.25, steps=16 * 3 * 32, device=device, dtype=torch.float32).reshape(16, 3, 32),
    )
    freqs = freqs_backing[:, 1, :]
    seq_lens_backing = torch.zeros((10,), device=device, dtype=torch.int64)
    seq_lens_backing[2::3] = torch.tensor([5, 4, 8], device=device, dtype=torch.int64)
    seq_lens = seq_lens_backing.as_strided((3,), (3,), storage_offset=2)
    assert not weight.is_contiguous()
    assert not freqs.is_contiguous()
    assert not seq_lens.is_contiguous()

    got = kv.clone()
    ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, seq_lens, 4)

    assert ok, failure
    expected = kv.cpu().clone()
    boundary_rows = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([0, 4], dtype=torch.int64)
    expected[boundary_rows] = _fused_norm_rope_ref(
        kv.cpu().index_select(0, boundary_rows),
        weight.cpu(),
        1e-5,
        freqs.cpu(),
        positions,
    )
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)

def test_compress_fused_norm_rope_inplace_musa_real_tilelang_handles_fp32_decode_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = torch.arange(3 * 128, device=device, dtype=torch.float32).reshape(3, 128) / 4099
    weight = torch.linspace(0.75, 1.25, steps=128, device=device, dtype=torch.float32)
    freqs = torch.polar(
        torch.ones((16, 32), device=device, dtype=torch.float32),
        torch.linspace(0.0, 0.25, steps=16 * 32, device=device, dtype=torch.float32).reshape(16, 32),
    )
    seq_lens = torch.tensor([5, 4, 8], device=device, dtype=torch.int32)

    got = kv.clone()
    ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, seq_lens, 4)

    assert ok, failure
    expected = kv.cpu().clone()
    boundary_rows = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([0, 4], dtype=torch.int32)
    selected = kv.cpu().index_select(0, boundary_rows)
    normalized = (
        selected.float()
        * torch.rsqrt(selected.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ) * weight.cpu().float()
    rope_part = normalized[..., -64:]
    rotated = torch.view_as_real(
        torch.view_as_complex(rope_part.reshape(rope_part.shape[0], -1, 2).contiguous())
        * freqs.cpu()[positions.long()].to(torch.complex64)
    ).flatten(-2)
    normalized[..., -64:] = rotated
    expected[boundary_rows] = normalized.to(kv.dtype)
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)

    del kv, weight, freqs, seq_lens, got
    torch.musa.synchronize()
    torch.musa.empty_cache()

def test_compress_fused_norm_rope_inplace_musa_real_tilelang_handles_strict_fp32_decode_shape_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = (torch.arange(128, device=device, dtype=torch.float32).reshape(1, 128) - 64) / 257
    weight = torch.linspace(0.5, 1.5, steps=128, device=device, dtype=torch.float32)
    freqs = torch.empty((1048576, 32), device=device, dtype=torch.complex64)
    freqs[0] = 1.0 + 0.0j
    seq_lens = torch.tensor([4], device=device, dtype=torch.int32)

    got = kv.clone()
    ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, seq_lens, 4)

    assert ok, failure
    normalized = (kv.cpu().float() * torch.rsqrt(kv.cpu().float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)) * weight.cpu().float()
    rope_part = normalized[..., -64:]
    rotated = torch.view_as_real(
        torch.view_as_complex(rope_part.reshape(rope_part.shape[0], -1, 2).contiguous())
        * freqs.cpu()[0:1].to(torch.complex64)
    ).flatten(-2)
    normalized[..., -64:] = rotated
    torch.testing.assert_close(got.cpu(), normalized.to(kv.dtype), rtol=1e-4, atol=1e-4)

    del kv, weight, freqs, seq_lens, got
    torch.musa.synchronize()
    torch.musa.empty_cache()

def test_compress_fused_norm_rope_inplace_musa_real_tilelang_handles_strict_fp32_decode_hidden512_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    kv = (torch.arange(512, device=device, dtype=torch.float32).reshape(1, 512) - 256) / 1021
    weight = torch.linspace(0.5, 1.5, steps=512, device=device, dtype=torch.float32)
    freqs = torch.empty((1048576, 32), device=device, dtype=torch.complex64)
    freqs[0] = 1.0 + 0.0j
    seq_lens = torch.tensor([4], device=device, dtype=torch.int32)

    got = kv.clone()
    ok, failure = MUSA_OPS._try_tilelang_compress_fused_norm_rope_inplace_musa(got, weight, 1e-5, freqs, seq_lens, 4)

    assert ok, failure
    normalized = (kv.cpu().float() * torch.rsqrt(kv.cpu().float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)) * weight.cpu().float()
    rope_part = normalized[..., -64:]
    rotated = torch.view_as_real(
        torch.view_as_complex(rope_part.reshape(rope_part.shape[0], -1, 2).contiguous())
        * freqs.cpu()[0:1].to(torch.complex64)
    ).flatten(-2)
    normalized[..., -64:] = rotated
    torch.testing.assert_close(got.cpu(), normalized.to(kv.dtype), rtol=1e-4, atol=1e-4)

    del kv, weight, freqs, seq_lens, got
    torch.musa.synchronize()
    torch.musa.empty_cache()
