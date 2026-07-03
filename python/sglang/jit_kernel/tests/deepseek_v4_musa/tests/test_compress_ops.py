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

@dataclass
class _DecodePlan:
    seq_lens: torch.Tensor
    compress_ratio: int

@dataclass
class _PrefillPlan:
    compress_ratio: int
    compress_plan: torch.Tensor
    write_plan: torch.Tensor

_PROD_COMPRESS_SHAPES_B16_I128_O128 = [
    pytest.param("ratio4_decode", 4, "decode", 11452, 128, 16, id="ratio4-decode"),
    pytest.param("ratio4_prefill", 4, "prefill", 11452, 128, 16, id="ratio4-prefill"),
    pytest.param("ratio128_decode", 128, "decode", 183296, 512, 16, id="ratio128-decode"),
    pytest.param("ratio128_prefill", 128, "prefill", 183296, 512, 16, id="ratio128-prefill"),
]

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

def _pack_prefill_rows(rows: list[tuple[int, int, int, int]]) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 16), dtype=torch.uint8)
    return torch.tensor(rows, dtype=torch.int32).view(torch.uint8).reshape(len(rows), 16)

def _pack_row_strided_prefill_rows(rows: list[tuple[int, int, int, int]]) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 16), dtype=torch.uint8)
    packed = _pack_prefill_rows(rows)
    backing = torch.empty((len(rows) * 2, 16), dtype=torch.uint8)
    backing[1::2].copy_(packed)
    return backing[1::2]

def _prefill_rows(plan: torch.Tensor) -> list[tuple[int, int, int, int]]:
    return [tuple(row) for row in plan.contiguous().view(torch.int32).reshape(-1, 4).tolist()]

def _prod_compress_prefill_plan(
    compress_ratio: int,
    batch_size: int,
    extend_len: int,
) -> _PrefillPlan:
    prefix_len = compress_ratio * 8
    seq_lens = torch.full((batch_size,), prefix_len + extend_len, dtype=torch.int32)
    extend_lens = torch.full((batch_size,), extend_len, dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(
        compress_ratio,
        batch_size * extend_len,
        seq_lens,
        extend_lens,
    )
    return _PrefillPlan(
        compress_ratio,
        _pack_prefill_rows(compress_rows),
        _pack_prefill_rows(write_rows),
    )

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

def test_compress_forward_musa_prefill_generates_plan_without_cuda_jit(monkeypatch) -> None:
    deepseek_v4_module = _load_deepseek_v4_module_for_validator()
    head_dim = 128
    kv_score_buffer = torch.zeros((1, 8, head_dim * 4), dtype=torch.float32)
    kv_score_input = torch.zeros((4, head_dim * 4), dtype=torch.float32)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.zeros((1,), dtype=torch.int32)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    extend_lens = torch.tensor([4], dtype=torch.int32)
    calls = []

    def fake_is_musa_tensor(tensor: torch.Tensor) -> bool:
        return tensor is kv_score_input

    def fail_common_module():
        raise AssertionError("MUSA compress_forward prefill must not load CUDA JIT")

    def fake_compress_forward_musa(*args, **kwargs):
        plan = args[4]
        calls.append(plan)
        return kwargs["out"]

    musa_ops_module = types.ModuleType("sglang.srt.hardware_backend.layers.deepseek_v4_musa_ops")
    musa_ops_module.compress_forward_musa = fake_compress_forward_musa
    monkeypatch.setitem(sys.modules, "sglang.srt.hardware_backend.layers.deepseek_v4_musa_ops", musa_ops_module)
    monkeypatch.setattr(deepseek_v4_module, "_is_musa_tensor", fake_is_musa_tensor)
    monkeypatch.setattr(deepseek_v4_module, "_jit_common_module", fail_common_module)

    deepseek_v4_module.compress_forward(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        plan=None,
        head_dim=head_dim,
        compress_ratio=4,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
    )

    assert len(calls) == 1
    compress_rows, write_rows = _prefill_plan_ref(4, 4, seq_lens, extend_lens)
    assert _prefill_rows(calls[0].compress_plan.cpu()) == compress_rows
    assert _prefill_rows(calls[0].write_plan.cpu()) == write_rows

def test_compressor_prefill_plan_generate_musa_avoids_cuda_jit(monkeypatch) -> None:
    deepseek_v4_module = _load_deepseek_v4_module_for_validator()
    seq_lens = torch.tensor([4], dtype=torch.int32)
    extend_lens = torch.tensor([4], dtype=torch.int32)
    calls = []

    def fail_common_module():
        raise AssertionError("MUSA CompressorPrefillPlan.generate must not load CUDA JIT")

    def fake_musa_prefill_plan(compress_ratio, num_q_tokens, got_seq_lens, got_extend_lens, device):
        calls.append((compress_ratio, num_q_tokens, got_seq_lens, got_extend_lens, device))
        compress_rows, write_rows = _prefill_plan_ref(
            compress_ratio,
            num_q_tokens,
            got_seq_lens,
            got_extend_lens,
        )
        return _PrefillPlan(
            compress_ratio,
            _pack_prefill_rows(compress_rows),
            _pack_prefill_rows(write_rows),
        )

    monkeypatch.setattr(deepseek_v4_module, "_jit_common_module", fail_common_module)
    monkeypatch.setattr(deepseek_v4_module, "_musa_compress_prefill_plan", fake_musa_prefill_plan)

    plan = deepseek_v4_module.CompressorPrefillPlan.generate(
        4,
        4,
        seq_lens,
        extend_lens,
        torch.device("musa"),
    )

    assert len(calls) == 1
    assert calls[0][4] == torch.device("musa")
    assert _prefill_rows(plan.compress_plan) == [(3, 0, 3, 4)]

def test_musa_prefill_plan_matches_reference_ratio4() -> None:
    deepseek_v4_module = _load_deepseek_v4_module_for_validator()
    seq_lens = torch.tensor([4, 6], dtype=torch.int32)
    extend_lens = torch.tensor([4, 2], dtype=torch.int32)

    got = deepseek_v4_module._musa_compress_prefill_plan(4, 6, seq_lens, extend_lens, torch.device("cpu"))
    compress_rows, write_rows = _prefill_plan_ref(4, 6, seq_lens, extend_lens)

    assert _prefill_rows(got.compress_plan) == compress_rows
    assert _prefill_rows(got.write_plan) == write_rows

def test_musa_prefill_plan_matches_reference_ratio128() -> None:
    deepseek_v4_module = _load_deepseek_v4_module_for_validator()
    seq_lens = torch.tensor([128, 130], dtype=torch.int32)
    extend_lens = torch.tensor([128, 2], dtype=torch.int32)

    got = deepseek_v4_module._musa_compress_prefill_plan(128, 130, seq_lens, extend_lens, torch.device("cpu"))
    compress_rows, write_rows = _prefill_plan_ref(128, 130, seq_lens, extend_lens)

    assert _prefill_rows(got.compress_plan) == compress_rows
    assert _prefill_rows(got.write_plan) == write_rows

def test_compress_forward_musa_ratio4_prefill_matches_reference() -> None:
    head_dim = 128
    seq_lens = torch.tensor([4], dtype=torch.int32)
    extend_lens = torch.tensor([4], dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(4, 4, seq_lens, extend_lens)
    kv_score_buffer = torch.zeros((1, 8, head_dim * 4), dtype=torch.float32)
    values = []
    for row in range(4):
        values.append(torch.cat([
            torch.full((head_dim,), 10.0 + row),
            torch.full((head_dim,), 20.0 + row),
            torch.full((head_dim,), -3.0 + row),
            torch.full((head_dim,), 4.0 + row),
        ]))
    kv_score_input = torch.stack(values)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32)
    plan = _PrefillPlan(4, _pack_prefill_rows(compress_rows), _pack_prefill_rows(write_rows))

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer.clone(),
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=4,
    )

    ref, _ref_buffer = _ratio4_prefill_ref(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        compress_rows,
        write_rows,
        head_dim,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

def test_compress_forward_musa_ratio4_prefill_page4_matches_reference() -> None:
    head_dim = 128
    compress_rows = [(3, 0, 7, 8)]
    write_rows = [(3, 0, 7, 0)]
    kv_score_buffer = torch.zeros((3, 4, head_dim * 4), dtype=torch.float32)
    for slot in range(4):
        kv_score_buffer[0, slot, :head_dim] = 10.0 + slot
        kv_score_buffer[0, slot, head_dim : head_dim * 2] = 20.0 + slot
        kv_score_buffer[1, slot, :head_dim] = 30.0 + slot
        kv_score_buffer[1, slot, head_dim : head_dim * 2] = 40.0 + slot
    kv_score_input = torch.zeros((4, head_dim * 4), dtype=torch.float32)
    kv_score_input[3, :head_dim] = 900.0
    kv_score_input[3, head_dim : head_dim * 2] = 901.0
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.tensor([1], dtype=torch.int32)
    extra_data = torch.tensor([[0, 1, 2, 0]], dtype=torch.int32)
    plan = _PrefillPlan(4, _pack_prefill_rows(compress_rows), _pack_prefill_rows(write_rows))
    got_buffer = kv_score_buffer.clone()

    got = compress_forward_musa(
        kv_score_buffer=got_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=extra_data,
        head_dim=head_dim,
        compress_ratio=4,
    )

    ref, ref_buffer = _ratio4_prefill_ref(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        compress_rows,
        write_rows,
        head_dim,
        extra_data,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_buffer, ref_buffer, rtol=0, atol=0)

def test_compress_forward_musa_ratio4_prefill_page4_writes_overlap_block_before_boundary() -> None:
    head_dim = 128
    write_rows = [(0, 0, 3, 4)]
    kv_score_buffer = torch.zeros((3, 4, head_dim * 4), dtype=torch.float32)
    kv_score_input = torch.full((1, head_dim * 4), 17.0, dtype=torch.float32)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.tensor([1], dtype=torch.int32)
    extra_data = torch.tensor([[0, 1, 2, 4]], dtype=torch.int32)
    plan = _PrefillPlan(4, _pack_prefill_rows([]), _pack_prefill_rows(write_rows))

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=extra_data,
        head_dim=head_dim,
        compress_ratio=4,
    )

    torch.testing.assert_close(got, torch.zeros((1, head_dim), dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(kv_score_buffer[2, 3], kv_score_input[0], rtol=0, atol=0)
    torch.testing.assert_close(kv_score_buffer[1, 3], torch.zeros_like(kv_score_buffer[1, 3]), rtol=0, atol=0)

def test_compress_forward_musa_ratio128_prefill_matches_reference() -> None:
    head_dim = 128
    seq_lens = torch.tensor([128], dtype=torch.int32)
    extend_lens = torch.tensor([128], dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(128, 128, seq_lens, extend_lens)
    kv_score_buffer = torch.zeros((1, 128, head_dim * 2), dtype=torch.float32)
    kv_score_input = torch.stack(
        [torch.cat([torch.full((head_dim,), float(row)), torch.full((head_dim,), float(row - 64))]) for row in range(128)]
    )
    ape = torch.zeros((128, head_dim), dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32)
    plan = _PrefillPlan(128, _pack_prefill_rows(compress_rows), _pack_prefill_rows(write_rows))

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer.clone(),
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=128,
    )

    ref, _ref_buffer = _ratio128_prefill_ref(kv_score_buffer, kv_score_input, ape, indices, compress_rows, write_rows, head_dim)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

def test_compress_forward_musa_ratio128_prefill_load_indices_matches_reference() -> None:
    head_dim = 128
    seq_lens = torch.tensor([256], dtype=torch.int32)
    extend_lens = torch.tensor([1], dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(128, 1, seq_lens, extend_lens)
    kv_score_buffer = torch.zeros((2, 128, head_dim * 2), dtype=torch.float32)
    for block in range(2):
        for slot in range(128):
            kv_score_buffer[block, slot, :head_dim] = float(block * 1000 + slot + 1)
            kv_score_buffer[block, slot, head_dim:] = float(slot - 32)
    kv_score_input = torch.cat([torch.full((head_dim,), 999.0), torch.full((head_dim,), 12.0)]).reshape(1, -1)
    ape = torch.zeros((128, head_dim), dtype=torch.float32)
    indices = torch.tensor([1], dtype=torch.int32)
    load_indices = torch.tensor([0], dtype=torch.int32)
    plan = _PrefillPlan(128, _pack_prefill_rows(compress_rows), _pack_prefill_rows(write_rows))

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer.clone(),
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=load_indices,
        head_dim=head_dim,
        compress_ratio=128,
    )

    ref, _ref_buffer = _ratio128_prefill_ref(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        compress_rows,
        write_rows,
        head_dim,
        extra_data=load_indices,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize(
    ("compress_ratio", "kv_score_buffer", "kv_score_input", "ape", "helper_name", "fill_value"),
    [
        (4, torch.zeros((1, 8, 512), dtype=torch.float32), torch.zeros((4, 512), dtype=torch.float32), torch.zeros((8, 128), dtype=torch.float32), "_try_tilelang_compress_forward_ratio4_prefill_musa", 8.0),
        (128, torch.zeros((1, 128, 256), dtype=torch.float32), torch.zeros((128, 256), dtype=torch.float32), torch.zeros((128, 128), dtype=torch.float32), "_try_tilelang_compress_forward_ratio128_prefill_musa", 9.0),
    ],
)
def test_compress_forward_musa_prefill_invokes_tilelang_fast_path(
    monkeypatch,
    compress_ratio: int,
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    helper_name: str,
    fill_value: float,
) -> None:
    calls = []

    def fake_try_tilelang_prefill_musa(
        got_kv_score_buffer: torch.Tensor,
        got_kv_score_input: torch.Tensor,
        got_ape: torch.Tensor,
        got_indices: torch.Tensor,
        got_compress_plan: torch.Tensor,
        got_write_plan: torch.Tensor,
        got_extra_data: torch.Tensor | None,
        got_out: torch.Tensor,
        got_head_dim: int,
    ) -> bool:
        calls.append(
            (
                got_kv_score_buffer,
                got_kv_score_input,
                got_ape,
                got_indices,
                got_compress_plan,
                got_write_plan,
                got_extra_data,
                got_out,
                got_head_dim,
            )
        )
        got_out.fill_(fill_value)
        return True

    monkeypatch.setattr(MUSA_OPS, helper_name, fake_try_tilelang_prefill_musa)
    monkeypatch.setattr(MUSA_OPS, "_compress_forward_ratio4_prefill", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prefill fallback must not run")))
    monkeypatch.setattr(MUSA_OPS, "_compress_forward_ratio128_prefill", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prefill fallback must not run")))

    head_dim = 128
    plan = _PrefillPlan(compress_ratio, _pack_prefill_rows([]), _pack_prefill_rows([]))
    indices = torch.tensor([0], dtype=torch.int32)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
    )

    assert len(calls) == 1
    assert calls[0][6] is None
    assert calls[0][8] == head_dim
    torch.testing.assert_close(got, torch.full_like(got, fill_value), rtol=0, atol=0)

def test_compress_forward_musa_ratio4_prefill_page4_invokes_tilelang_fast_path(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_prefill_musa(
        got_kv_score_buffer: torch.Tensor,
        got_kv_score_input: torch.Tensor,
        got_ape: torch.Tensor,
        got_indices: torch.Tensor,
        got_compress_plan: torch.Tensor,
        got_write_plan: torch.Tensor,
        got_extra_data: torch.Tensor | None,
        got_out: torch.Tensor,
        got_head_dim: int,
    ) -> bool:
        calls.append(
            (
                got_kv_score_buffer,
                got_kv_score_input,
                got_ape,
                got_indices,
                got_compress_plan,
                got_write_plan,
                got_extra_data,
                got_out,
                got_head_dim,
            )
        )
        got_out.fill_(10.0)
        return True

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_compress_forward_ratio4_prefill_musa", fake_try_tilelang_prefill_musa)
    monkeypatch.setattr(MUSA_OPS, "_compress_forward_ratio4_prefill", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prefill fallback must not run")))

    head_dim = 128
    kv_score_buffer = torch.zeros((3, 4, head_dim * 4), dtype=torch.float32)
    kv_score_input = torch.zeros((4, head_dim * 4), dtype=torch.float32)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.tensor([1], dtype=torch.int32)
    extra_data = torch.tensor([[0, 1, 2, 0]], dtype=torch.int32)
    plan = _PrefillPlan(4, _pack_prefill_rows([(3, 0, 7, 8)]), _pack_prefill_rows([(3, 0, 7, 0)]))

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=extra_data,
        head_dim=head_dim,
        compress_ratio=4,
    )

    assert len(calls) == 1
    assert calls[0][6] is extra_data
    assert calls[0][8] == head_dim
    torch.testing.assert_close(got, torch.full_like(got, 10.0), rtol=0, atol=0)

def test_compress_forward_musa_ratio128_prefill_load_indices_invokes_tilelang_fast_path(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_prefill_musa(
        got_kv_score_buffer: torch.Tensor,
        got_kv_score_input: torch.Tensor,
        got_ape: torch.Tensor,
        got_indices: torch.Tensor,
        got_compress_plan: torch.Tensor,
        got_write_plan: torch.Tensor,
        got_extra_data: torch.Tensor | None,
        got_out: torch.Tensor,
        got_head_dim: int,
    ) -> bool:
        calls.append(
            (
                got_kv_score_buffer,
                got_kv_score_input,
                got_ape,
                got_indices,
                got_compress_plan,
                got_write_plan,
                got_extra_data,
                got_out,
                got_head_dim,
            )
        )
        got_out.fill_(11.0)
        return True

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_compress_forward_ratio128_prefill_musa", fake_try_tilelang_prefill_musa)
    monkeypatch.setattr(MUSA_OPS, "_compress_forward_ratio128_prefill", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prefill fallback must not run")))

    head_dim = 128
    load_indices = torch.tensor([0], dtype=torch.int32)
    got = compress_forward_musa(
        kv_score_buffer=torch.zeros((2, 128, head_dim * 2), dtype=torch.float32),
        kv_score_input=torch.zeros((1, head_dim * 2), dtype=torch.float32),
        ape=torch.zeros((128, head_dim), dtype=torch.float32),
        indices=torch.tensor([1], dtype=torch.int32),
        plan=_PrefillPlan(128, _pack_prefill_rows([(0, 0, 255, 127)]), _pack_prefill_rows([(0, 0, 255, 127)])),
        extra_data=load_indices,
        head_dim=head_dim,
        compress_ratio=128,
    )

    assert len(calls) == 1
    assert calls[0][6] is load_indices
    assert calls[0][8] == head_dim
    torch.testing.assert_close(got, torch.full_like(got, 11.0), rtol=0, atol=0)

def test_compress_forward_musa_ratio4_prefill_rejects_musa_extra_data_without_fallback(monkeypatch) -> None:
    head_dim = 128
    extra_data = torch.tensor([[0, 1, 2, 0]], dtype=torch.int32)
    monkeypatch.setattr(MUSA_OPS, "_is_musa_tensor", lambda tensor: tensor is extra_data)
    monkeypatch.setattr(MUSA_OPS, "_musa_graph_capture_enabled", lambda: True)

    with pytest.raises(NotImplementedError, match="no torch fallback"):
        compress_forward_musa(
            kv_score_buffer=torch.zeros((3, 4, head_dim * 4), dtype=torch.float32),
            kv_score_input=torch.zeros((4, head_dim * 4), dtype=torch.float32),
            ape=torch.zeros((8, head_dim), dtype=torch.float32),
            indices=torch.tensor([1], dtype=torch.int32),
            plan=_PrefillPlan(4, _pack_prefill_rows([(3, 0, 7, 8)]), _pack_prefill_rows([])),
            extra_data=extra_data,
            head_dim=head_dim,
            compress_ratio=4,
        )

@pytest.mark.parametrize(
    ("kv_score_buffer", "extra_data", "match"),
    [
        (torch.zeros((3, 8, 512), dtype=torch.float32), torch.zeros((1, 4), dtype=torch.int32), r"shape \[N,4,512\]"),
        (torch.zeros((3, 4, 512), dtype=torch.float32), torch.zeros((1, 3), dtype=torch.int32), r"extra_data shape \[N,4\]"),
    ],
)
def test_compress_forward_musa_ratio4_prefill_page4_rejects_malformed_layout(
    kv_score_buffer: torch.Tensor,
    extra_data: torch.Tensor,
    match: str,
) -> None:
    head_dim = 128
    with pytest.raises(ValueError, match=match):
        compress_forward_musa(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=torch.zeros((4, head_dim * 4), dtype=torch.float32),
            ape=torch.zeros((8, head_dim), dtype=torch.float32),
            indices=torch.tensor([1], dtype=torch.int32),
            plan=_PrefillPlan(4, _pack_prefill_rows([(3, 0, 7, 8)]), _pack_prefill_rows([])),
            extra_data=extra_data,
            head_dim=head_dim,
            compress_ratio=4,
        )

def test_compress_forward_musa_ratio4_prefill_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    seq_lens = torch.tensor([4], dtype=torch.int32)
    extend_lens = torch.tensor([4], dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(4, 4, seq_lens, extend_lens)
    kv_score_buffer_cpu = torch.zeros((1, 8, head_dim * 4), dtype=torch.float32)
    values = []
    for row in range(4):
        values.append(torch.cat([
            torch.full((head_dim,), 10.0 + row),
            torch.full((head_dim,), 20.0 + row),
            torch.full((head_dim,), -3.0 + row),
            torch.full((head_dim,), 4.0 + row),
        ]))
    kv_score_input_cpu = torch.stack(values)
    ape_cpu = torch.zeros((8, head_dim), dtype=torch.float32)
    indices_cpu = torch.tensor([0], dtype=torch.int32)
    plan = _PrefillPlan(4, _pack_prefill_rows(compress_rows).to(device), _pack_prefill_rows(write_rows).to(device))
    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input_cpu.to(device),
        ape=ape_cpu.to(device),
        indices=indices_cpu.to(device),
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=4,
    )

    ref, ref_buffer = _ratio4_prefill_ref(kv_score_buffer_cpu, kv_score_input_cpu, ape_cpu, indices_cpu, compress_rows, write_rows, head_dim)
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer.cpu(), ref_buffer, rtol=0, atol=0)

def test_compress_forward_musa_ratio4_prefill_real_tilelang_row_strided_plan_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    seq_lens = torch.tensor([8], dtype=torch.int32)
    extend_lens = torch.tensor([8], dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(4, 8, seq_lens, extend_lens)
    kv_score_buffer_cpu = torch.zeros((1, 8, head_dim * 4), dtype=torch.float32)
    values = []
    for row in range(8):
        values.append(torch.cat([
            torch.full((head_dim,), 10.0 + row),
            torch.full((head_dim,), 20.0 + row),
            torch.full((head_dim,), -3.0 + row),
            torch.full((head_dim,), 4.0 + row),
        ]))
    kv_score_input_cpu = torch.stack(values)
    ape_cpu = torch.zeros((8, head_dim), dtype=torch.float32)
    indices_cpu = torch.tensor([0], dtype=torch.int32)
    compress_plan_packed = _pack_prefill_rows(compress_rows).to(device)
    compress_plan_backing = torch.empty((len(compress_rows) * 2, 16), device=device, dtype=torch.uint8)
    compress_plan_backing[1::2].copy_(compress_plan_packed)
    compress_plan = compress_plan_backing[1::2]
    write_plan_packed = _pack_prefill_rows(write_rows).to(device)
    write_plan_backing = torch.empty((len(write_rows) * 2, 16), device=device, dtype=torch.uint8)
    write_plan_backing[1::2].copy_(write_plan_packed)
    write_plan = write_plan_backing[1::2]
    plan = _PrefillPlan(4, compress_plan, write_plan)
    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)

    assert not plan.compress_plan.is_contiguous()
    assert not plan.write_plan.is_contiguous()
    assert plan.compress_plan.stride() == (32, 1)
    assert plan.write_plan.stride() == (32, 1)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input_cpu.to(device),
        ape=ape_cpu.to(device),
        indices=indices_cpu.to(device),
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=4,
    )

    ref, ref_buffer = _ratio4_prefill_ref(kv_score_buffer_cpu, kv_score_input_cpu, ape_cpu, indices_cpu, compress_rows, write_rows, head_dim)
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer.cpu(), ref_buffer, rtol=0, atol=0)

def test_compress_forward_musa_ratio4_prefill_real_tilelang_flat_state_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    seq_lens = torch.tensor([4], dtype=torch.int32)
    extend_lens = torch.tensor([4], dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(4, 4, seq_lens, extend_lens)
    kv_score_buffer_cpu = torch.zeros((12, head_dim * 4), dtype=torch.float32)
    values = []
    for row in range(4):
        values.append(torch.cat([
            torch.full((head_dim,), 10.0 + row),
            torch.full((head_dim,), 20.0 + row),
            torch.full((head_dim,), -3.0 + row),
            torch.full((head_dim,), 4.0 + row),
        ]))
    kv_score_input_cpu = torch.stack(values)
    ape_cpu = torch.zeros((8, head_dim), dtype=torch.float32)
    indices_cpu = torch.tensor([0], dtype=torch.int32)
    plan = _PrefillPlan(4, _pack_prefill_rows(compress_rows).to(device), _pack_prefill_rows(write_rows).to(device))
    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input_cpu.to(device),
        ape=ape_cpu.to(device),
        indices=indices_cpu.to(device),
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=4,
    )

    ref, ref_buffer = _ratio4_prefill_ref(kv_score_buffer_cpu, kv_score_input_cpu, ape_cpu, indices_cpu, compress_rows, write_rows, head_dim)
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer.cpu(), ref_buffer, rtol=0, atol=0)

def test_compress_forward_musa_ratio128_prefill_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    seq_lens = torch.tensor([128], dtype=torch.int32)
    extend_lens = torch.tensor([128], dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(128, 128, seq_lens, extend_lens)
    kv_score_buffer_cpu = torch.zeros((1, 128, head_dim * 2), dtype=torch.float32)
    kv_score_input_cpu = torch.stack(
        [torch.cat([torch.full((head_dim,), float(row)), torch.full((head_dim,), float(row - 64))]) for row in range(128)]
    )
    ape_cpu = torch.zeros((128, head_dim), dtype=torch.float32)
    indices_cpu = torch.tensor([0], dtype=torch.int32)
    plan = _PrefillPlan(128, _pack_prefill_rows(compress_rows).to(device), _pack_prefill_rows(write_rows).to(device))
    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input_cpu.to(device),
        ape=ape_cpu.to(device),
        indices=indices_cpu.to(device),
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=128,
    )

    ref, ref_buffer = _ratio128_prefill_ref(kv_score_buffer_cpu, kv_score_input_cpu, ape_cpu, indices_cpu, compress_rows, write_rows, head_dim)
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer.cpu(), ref_buffer, rtol=0, atol=0)

def test_compress_forward_musa_ratio128_prefill_load_indices_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    seq_lens = torch.tensor([256], dtype=torch.int32)
    extend_lens = torch.tensor([1], dtype=torch.int32)
    compress_rows, write_rows = _prefill_plan_ref(128, 1, seq_lens, extend_lens)
    kv_score_buffer_cpu = torch.zeros((2, 128, head_dim * 2), dtype=torch.float32)
    for block in range(2):
        for slot in range(128):
            kv_score_buffer_cpu[block, slot, :head_dim] = float(block * 1000 + slot + 1)
            kv_score_buffer_cpu[block, slot, head_dim:] = float(slot - 32)
    kv_score_input_cpu = torch.cat(
        [torch.full((head_dim,), 999.0), torch.full((head_dim,), 12.0)]
    ).reshape(1, -1)
    ape_cpu = torch.zeros((128, head_dim), dtype=torch.float32)
    indices_cpu = torch.tensor([1], dtype=torch.int32)
    load_indices_cpu = torch.tensor([0], dtype=torch.int32)
    plan = _PrefillPlan(128, _pack_prefill_rows(compress_rows).to(device), _pack_prefill_rows(write_rows).to(device))
    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input_cpu.to(device),
        ape=ape_cpu.to(device),
        indices=indices_cpu.to(device),
        plan=plan,
        extra_data=load_indices_cpu.to(device),
        head_dim=head_dim,
        compress_ratio=128,
    )

    ref, ref_buffer = _ratio128_prefill_ref(
        kv_score_buffer_cpu,
        kv_score_input_cpu,
        ape_cpu,
        indices_cpu,
        compress_rows,
        write_rows,
        head_dim,
        extra_data=load_indices_cpu,
    )
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer.cpu(), ref_buffer, rtol=0, atol=0)

@pytest.mark.parametrize(
    ("compress_ratio", "seq_lens", "extend_lens", "head_dim", "indices"),
    [
        (4, [4], [4], 128, [0]),
        (4, [11, 8], [7, 4], 128, [0, 1]),
        (128, [128], [128], 128, [0]),
        (128, [257, 129], [129, 1], 128, [0, 1]),
    ],
)
def test_compress_forward_musa_prefill_shape_matrix_matches_reference(
    compress_ratio: int,
    seq_lens: list[int],
    extend_lens: list[int],
    head_dim: int,
    indices: list[int],
) -> None:
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    extend_lens_tensor = torch.tensor(extend_lens, dtype=torch.int32)
    num_q_tokens = sum(extend_lens)
    compress_rows, write_rows = _prefill_plan_ref(compress_ratio, num_q_tokens, seq_lens_tensor, extend_lens_tensor)
    ring_size = 8 if compress_ratio == 4 else 128
    width = head_dim * (4 if compress_ratio == 4 else 2)
    kv_score_buffer = torch.arange(len(indices) * ring_size * width, dtype=torch.float32).reshape(len(indices), ring_size, width) / 997
    kv_score_input = torch.arange(num_q_tokens * width, dtype=torch.float32).reshape(num_q_tokens, width) / 257
    ape = torch.arange(ring_size * head_dim, dtype=torch.float32).reshape(ring_size, head_dim) / 4099
    index_tensor = torch.tensor(indices, dtype=torch.int32)
    plan = _PrefillPlan(compress_ratio, _pack_prefill_rows(compress_rows), _pack_prefill_rows(write_rows))

    got = compress_forward_musa(kv_score_buffer, kv_score_input, ape, index_tensor, plan, None, head_dim=head_dim, compress_ratio=compress_ratio)

    if compress_ratio == 4:
        ref, ref_buffer = _ratio4_prefill_ref(kv_score_buffer.clone(), kv_score_input, ape, index_tensor, compress_rows, write_rows, head_dim)
    else:
        ref, ref_buffer = _ratio128_prefill_ref(kv_score_buffer.clone(), kv_score_input, ape, index_tensor, compress_rows, write_rows, head_dim)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer, ref_buffer, rtol=0, atol=0)

@pytest.mark.parametrize("bad_plan", [torch.zeros((1, 15), dtype=torch.uint8), torch.zeros((1, 16), dtype=torch.int32)])
def test_compress_forward_musa_prefill_rejects_malformed_plan_edges(bad_plan: torch.Tensor) -> None:
    head_dim = 128
    plan = _PrefillPlan(4, bad_plan, _pack_prefill_rows([]))
    with pytest.raises(ValueError, match="prefill plan expects uint8 shape"):
        compress_forward_musa(
            kv_score_buffer=torch.zeros((1, 8, head_dim * 4), dtype=torch.float32),
            kv_score_input=torch.zeros((4, head_dim * 4), dtype=torch.float32),
            ape=torch.zeros((8, head_dim), dtype=torch.float32),
            indices=torch.tensor([0], dtype=torch.int32),
            plan=plan,
            extra_data=None,
            head_dim=head_dim,
            compress_ratio=4,
        )

def test_compress_forward_musa_ratio128_prefill_rejects_malformed_load_indices() -> None:
    head_dim = 128
    plan = _PrefillPlan(128, _pack_prefill_rows([]), _pack_prefill_rows([]))

    with pytest.raises(ValueError, match="load_indices"):
        compress_forward_musa(
            kv_score_buffer=torch.zeros((1, 128, head_dim * 2), dtype=torch.float32),
            kv_score_input=torch.zeros((1, head_dim * 2), dtype=torch.float32),
            ape=torch.zeros((128, head_dim), dtype=torch.float32),
            indices=torch.tensor([0], dtype=torch.int32),
            plan=plan,
            extra_data=torch.zeros((1, 4), dtype=torch.int32),
            head_dim=head_dim,
            compress_ratio=128,
        )

def test_compress_forward_musa_prefill_rejects_malformed_plan() -> None:
    head_dim = 128
    plan = _PrefillPlan(4, torch.empty((1, 15), dtype=torch.uint8), _pack_prefill_rows([]))

    with pytest.raises(ValueError, match=r"uint8 shape \[N,16\]"):
        compress_forward_musa(
            kv_score_buffer=torch.zeros((1, 8, head_dim * 4), dtype=torch.float32),
            kv_score_input=torch.zeros((4, head_dim * 4), dtype=torch.float32),
            ape=torch.zeros((8, head_dim), dtype=torch.float32),
            indices=torch.tensor([0], dtype=torch.int32),
            plan=plan,
            extra_data=None,
            head_dim=head_dim,
            compress_ratio=4,
        )

def test_compress_forward_musa_prefill_accepts_row_strided_plan_rows() -> None:
    rows = [(3, 0, 7, 8), (7, 1, 11, 4)]
    plan = _pack_row_strided_prefill_rows(rows)

    assert not plan.is_contiguous()
    assert plan.stride() == (32, 1)

    got = MUSA_OPS._prefill_plan_rows(plan)
    assert got.shape == (2, 4)
    assert got.stride() == (8, 1)
    assert got.data_ptr() == plan.data_ptr()
    assert [tuple(row) for row in got.tolist()] == rows

def test_compress_forward_musa_prefill_rejects_musa_without_cpu_or_torch_fallback(monkeypatch) -> None:
    head_dim = 128
    plan = _PrefillPlan(4, _pack_prefill_rows([]), _pack_prefill_rows([]))
    kv_score_buffer = torch.zeros((1, 8, head_dim * 4), dtype=torch.float32)
    kv_score_input = torch.zeros((4, head_dim * 4), dtype=torch.float32)

    monkeypatch.setattr(MUSA_OPS, "_is_musa_tensor", lambda tensor: tensor is kv_score_input)
    monkeypatch.setattr(MUSA_OPS, "_musa_graph_capture_enabled", lambda: True)

    with pytest.raises(NotImplementedError, match="no torch fallback"):
        compress_forward_musa(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=torch.zeros((8, head_dim), dtype=torch.float32),
            indices=torch.tensor([0], dtype=torch.int32),
            plan=plan,
            extra_data=None,
            head_dim=head_dim,
            compress_ratio=4,
        )

@pytest.mark.parametrize(
    ("compress_ratio", "seq_lens", "head_dim"),
    [
        (4, [1], 128),
        (4, [4, 7], 128),
        (128, [1], 128),
        (128, [128, 255], 128),
    ],
)
def test_compress_forward_musa_decode_shape_matrix_matches_reference(
    compress_ratio: int,
    seq_lens: list[int],
    head_dim: int,
) -> None:
    batch = len(seq_lens)
    ring_size = 8 if compress_ratio == 4 else 128
    width = head_dim * (4 if compress_ratio == 4 else 2)
    kv_score_buffer = torch.arange(batch * ring_size * width, dtype=torch.float32).reshape(batch, ring_size, width) / 997
    kv_score_input = torch.arange(batch * width, dtype=torch.float32).reshape(batch, width) / 257
    ape = torch.arange(ring_size * head_dim, dtype=torch.float32).reshape(ring_size, head_dim) / 4099
    indices = torch.arange(batch, dtype=torch.int32)
    plan = _DecodePlan(torch.tensor(seq_lens, dtype=torch.int32), compress_ratio)

    original_buffer = kv_score_buffer.clone()
    got = compress_forward_musa(kv_score_buffer, kv_score_input, ape, indices, plan, None, head_dim=head_dim, compress_ratio=compress_ratio)

    ref_buffer = original_buffer.clone()
    refs = torch.zeros((batch, head_dim), dtype=kv_score_input.dtype)
    for row, seq_len in enumerate(seq_lens):
        write_pos = (seq_len - 1) % ring_size
        ref_buffer[row, write_pos] = kv_score_input[row]
        if seq_len % compress_ratio != 0:
            continue
        if compress_ratio == 4:
            kv_window = []
            score_window = []
            for slot in range(8):
                src = ref_buffer[row, (seq_len + slot) % 8].reshape(4, head_dim)
                kv_window.append(src[0] if slot < 4 else src[1])
                score_window.append(src[2] if slot < 4 else src[3])
            if seq_len == 4:
                for slot in range(4):
                    kv_window[slot] = torch.zeros_like(kv_window[slot])
                    score_window[slot] = torch.full_like(score_window[slot], -1e9)
            refs[row] = _compress_ref(torch.stack(kv_window), torch.stack(score_window), ape)
        else:
            window = torch.stack([ref_buffer[row, (seq_len + slot) % 128] for slot in range(128)]).reshape(128, 2, head_dim)
            refs[row] = _compress_ref(window[:, 0, :], window[:, 1, :], ape)
    torch.testing.assert_close(got, refs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer, ref_buffer, rtol=0, atol=0)

def _run_compress_decode_tilelang(
    compress_ratio: int,
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    out = torch.empty((kv_score_input.shape[0], head_dim), device=kv_score_input.device, dtype=torch.float32)
    if compress_ratio == 4:
        ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio4_decode_musa(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            seq_lens,
            None,
            out,
            head_dim,
        )
    else:
        ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio128_decode_musa(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            seq_lens,
            out,
            head_dim,
        )
    assert ok, failure
    torch.musa.synchronize()
    return out

@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("num_tokens", [1, 65])
def test_compress_forward_musa_decode_real_tilelang_is_repeat_deterministic_on_musa(
    compress_ratio: int,
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    head_dim = 128
    ring_size = 8 if compress_ratio == 4 else 128
    width = head_dim * (4 if compress_ratio == 4 else 2)
    torch.manual_seed(20262925 + compress_ratio + num_tokens)
    kv_score_buffer = torch.randn((num_tokens, ring_size, width), device=device, dtype=torch.float32)
    kv_score_input = torch.randn((num_tokens, width), device=device, dtype=torch.float32)
    ape = torch.randn((ring_size, head_dim), device=device, dtype=torch.float32)
    indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
    seq_lens = torch.full((num_tokens,), compress_ratio, device=device, dtype=torch.int32)

    expected_buffer = kv_score_buffer.clone()
    expected = _run_compress_decode_tilelang(
        compress_ratio,
        expected_buffer,
        kv_score_input,
        ape,
        indices,
        seq_lens,
        head_dim,
    )
    for repeat_idx in range(10):
        actual_buffer = kv_score_buffer.clone()
        actual = _run_compress_decode_tilelang(
            compress_ratio,
            actual_buffer,
            kv_score_input,
            ape,
            indices,
            seq_lens,
            head_dim,
        )
        _assert_repeat_exact(
            f"compress_decode out repeat={repeat_idx}, ratio={compress_ratio}, num_tokens={num_tokens}",
            actual,
            expected,
        )
        _assert_repeat_exact(
            f"compress_decode buffer repeat={repeat_idx}, ratio={compress_ratio}, num_tokens={num_tokens}",
            actual_buffer,
            expected_buffer,
        )

@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("num_tokens", [3, 65])
def test_compress_forward_musa_decode_real_tilelang_probe_row_is_batch_shape_invariant_on_musa(
    compress_ratio: int,
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    head_dim = 128
    ring_size = 8 if compress_ratio == 4 else 128
    width = head_dim * (4 if compress_ratio == 4 else 2)
    torch.manual_seed(20263025 + compress_ratio + num_tokens)
    probe_buffer = torch.randn((1, ring_size, width), device=device, dtype=torch.float32)
    probe_input = torch.randn((1, width), device=device, dtype=torch.float32)
    ape = torch.randn((ring_size, head_dim), device=device, dtype=torch.float32)

    expected_buffer = probe_buffer.clone()
    expected = _run_compress_decode_tilelang(
        compress_ratio,
        expected_buffer,
        probe_input,
        ape,
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.tensor([compress_ratio], device=device, dtype=torch.int32),
        head_dim,
    )

    kv_score_buffer = torch.randn((num_tokens, ring_size, width), device=device, dtype=torch.float32)
    kv_score_input = torch.randn((num_tokens, width), device=device, dtype=torch.float32)
    indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
    seq_lens = torch.full((num_tokens,), compress_ratio, device=device, dtype=torch.int32)
    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    for pos in probe_positions:
        kv_score_buffer[pos].copy_(probe_buffer[0])
        kv_score_input[pos].copy_(probe_input[0])

    actual = _run_compress_decode_tilelang(
        compress_ratio,
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        seq_lens,
        head_dim,
    )
    for pos in probe_positions:
        _assert_repeat_exact(
            f"compress_decode out ratio={compress_ratio}, num_tokens={num_tokens}, pos={pos}",
            actual[pos],
            expected[0],
        )
        _assert_repeat_exact(
            f"compress_decode buffer ratio={compress_ratio}, num_tokens={num_tokens}, pos={pos}",
            kv_score_buffer[pos],
            expected_buffer[0],
        )

@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_compress_forward_musa_prefill_real_tilelang_is_repeat_deterministic_on_musa(
    compress_ratio: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    head_dim = 128
    batch_size = 2
    extend_len = 8 if compress_ratio == 4 else 128
    ring_size = 8 if compress_ratio == 4 else 128
    width = head_dim * (4 if compress_ratio == 4 else 2)
    torch.manual_seed(20263125 + compress_ratio)
    kv_score_buffer = torch.randn((batch_size, ring_size, width), device=device, dtype=torch.float32)
    kv_score_input = torch.randn((batch_size * extend_len, width), device=device, dtype=torch.float32)
    ape = torch.randn((ring_size, head_dim), device=device, dtype=torch.float32)
    indices = torch.arange(batch_size, device=device, dtype=torch.int32)
    plan_cpu = _prod_compress_prefill_plan(compress_ratio, batch_size, extend_len)
    compress_plan = plan_cpu.compress_plan.to(device=device)
    write_plan = plan_cpu.write_plan.to(device=device)

    def run_once(buffer: torch.Tensor) -> torch.Tensor:
        out = torch.empty((kv_score_input.shape[0], head_dim), device=device, dtype=torch.float32)
        if compress_ratio == 4:
            ok = MUSA_OPS._try_tilelang_compress_forward_ratio4_prefill_musa(
                buffer,
                kv_score_input,
                ape,
                indices,
                compress_plan,
                write_plan,
                None,
                out,
                head_dim,
            )
        else:
            ok = MUSA_OPS._try_tilelang_compress_forward_ratio128_prefill_musa(
                buffer,
                kv_score_input,
                ape,
                indices,
                compress_plan,
                write_plan,
                None,
                out,
                head_dim,
            )
        assert ok
        torch.musa.synchronize()
        return out

    expected_buffer = kv_score_buffer.clone()
    expected = run_once(expected_buffer)
    for repeat_idx in range(10):
        actual_buffer = kv_score_buffer.clone()
        actual = run_once(actual_buffer)
        _assert_repeat_exact(
            f"compress_prefill out repeat={repeat_idx}, ratio={compress_ratio}",
            actual,
            expected,
        )
        _assert_repeat_exact(
            f"compress_prefill buffer repeat={repeat_idx}, ratio={compress_ratio}",
            actual_buffer,
            expected_buffer,
        )

@pytest.mark.parametrize(
    ("name", "compress_ratio", "mode", "buffer_rows", "head_dim", "batch_size"),
    _PROD_COMPRESS_SHAPES_B16_I128_O128,
)
def test_compress_forward_musa_prod_b16_i128_o128_shapes_on_musa(
    name: str,
    compress_ratio: int,
    mode: str,
    buffer_rows: int,
    head_dim: int,
    batch_size: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run production-shape MUSA validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    width = head_dim * (4 if compress_ratio == 4 else 2)
    kv_score_buffer = torch.zeros((buffer_rows, width), device=device, dtype=torch.float32)
    ape = torch.zeros((8 if compress_ratio == 4 else 128, head_dim), device=device, dtype=torch.float32)
    indices = torch.arange(batch_size, device=device, dtype=torch.int32)

    if mode == "decode":
        kv_score_input = torch.zeros((batch_size, width), device=device, dtype=torch.float32)
        seq_len = compress_ratio
        plan = _DecodePlan(
            torch.full((batch_size,), seq_len, device=device, dtype=torch.int32),
            compress_ratio,
        )
    else:
        extend_len = 128
        kv_score_input = torch.zeros((batch_size * extend_len, width), device=device, dtype=torch.float32)
        plan_cpu = _prod_compress_prefill_plan(compress_ratio, batch_size, extend_len)
        plan = _PrefillPlan(
            compress_ratio,
            plan_cpu.compress_plan.to(device=device),
            plan_cpu.write_plan.to(device=device),
        )
        expected_compress_rows = 512 if compress_ratio == 4 else 16
        expected_write_rows = 64 if compress_ratio == 4 else 0
        assert plan.compress_plan.shape == (expected_compress_rows, 16)
        assert plan.write_plan.shape == (expected_write_rows, 16)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
    )

    assert name
    assert kv_score_buffer.shape == (buffer_rows, width)
    assert got.shape == (kv_score_input.shape[0], head_dim)
    torch.testing.assert_close(got.cpu(), torch.zeros_like(got.cpu()), rtol=0, atol=0)

@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_compress_forward_musa_decode_rejects_fake_musa_without_torch_fallback(monkeypatch, compress_ratio: int) -> None:
    head_dim = 128
    width = head_dim * (4 if compress_ratio == 4 else 2)
    kv_score_buffer = torch.zeros((1, compress_ratio, width), dtype=torch.float32)
    kv_score_input = torch.zeros((1, width), dtype=torch.float32)
    plan = _DecodePlan(torch.tensor([compress_ratio], dtype=torch.int32), compress_ratio)
    monkeypatch.setattr(MUSA_OPS, "_is_musa_tensor", lambda tensor: tensor is kv_score_input)
    monkeypatch.setattr(MUSA_OPS, "_musa_graph_capture_enabled", lambda: True)

    with pytest.raises(NotImplementedError, match="no torch fallback"):
        compress_forward_musa(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=torch.zeros((compress_ratio, head_dim), dtype=torch.float32),
            indices=torch.tensor([0], dtype=torch.int32),
            plan=plan,
            extra_data=None,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
        )

def test_compress_forward_musa_decode_env_allows_torch_fallback(monkeypatch) -> None:
    head_dim = 128
    width = head_dim * 4
    kv_score_buffer = torch.zeros((1, 8, width), dtype=torch.float32)
    kv_score_input = torch.zeros((1, width), dtype=torch.float32)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    plan = _DecodePlan(seq_lens, 4)

    monkeypatch.setenv("SGLANG_MUSA_ALLOW_TORCH_FALLBACK", "1")
    monkeypatch.setattr(MUSA_OPS, "_is_musa_tensor", lambda tensor: tensor is kv_score_input)
    monkeypatch.setattr(MUSA_OPS, "_musa_graph_capture_enabled", lambda: False)
    monkeypatch.setattr(
        MUSA_OPS,
        "_try_tilelang_compress_forward_ratio4_decode_musa",
        lambda *_args, **_kwargs: (False, "forced miss"),
    )

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=4,
    )

    assert got.shape == (1, head_dim)

def test_compress_forward_musa_ratio4_flat_decode_reports_strict_e2e_metadata(monkeypatch) -> None:
    head_dim = 128
    kv_score_buffer = torch.zeros((11452, head_dim * 4), dtype=torch.float32)
    kv_score_input = torch.zeros((1, head_dim * 4), dtype=torch.float32)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32)
    plan = _DecodePlan(torch.tensor([4], dtype=torch.int32), compress_ratio=4)

    def fake_try_tilelang_compress_forward_ratio4_decode_musa(
        kv_score_buffer_arg: torch.Tensor,
        kv_score_input_arg: torch.Tensor,
        ape_arg: torch.Tensor,
        indices_arg: torch.Tensor,
        seq_lens_arg: torch.Tensor,
        extra_data_arg: torch.Tensor | None,
        out_arg: torch.Tensor,
        head_dim_arg: int,
    ) -> tuple[bool, str | None]:
        assert kv_score_buffer_arg is kv_score_buffer
        assert kv_score_input_arg is kv_score_input
        assert ape_arg is ape
        assert indices_arg is indices
        assert tuple(seq_lens_arg.shape) == (1,)
        assert seq_lens_arg.dtype == torch.int32
        assert seq_lens_arg.is_contiguous()
        torch.testing.assert_close(seq_lens_arg, plan.seq_lens, rtol=0, atol=0)
        assert extra_data_arg is None
        assert tuple(out_arg.shape) == (1, head_dim)
        assert head_dim_arg == head_dim
        return False, (
            "kernel exception TypeError: "
            "_tilelang_compress_forward_ratio4_decode_flat_kernel() "
            "missing 1 required positional argument: 'head_dim'; "
            "kv_score_buffer=device:musa:0,dtype:torch.float32,shape:(11452, 512),stride:(512, 1),contiguous:True; "
            "kv_score_input=device:musa:0,dtype:torch.float32,shape:(1, 512),stride:(512, 1),contiguous:True; "
            "ape=device:musa:0,dtype:torch.float32,shape:(8, 128),stride:(128, 1),contiguous:True; "
            "indices=device:musa:0,dtype:torch.int32,shape:(1,),stride:(1,),contiguous:True; "
            "seq_lens=device:musa:0,dtype:torch.int32,shape:(1,),stride:(1,),contiguous:True; "
            "extra_data=None; "
            "out=device:musa:0,dtype:torch.float32,shape:(1, 128),stride:(128, 1),contiguous:True; "
            "head_dim=128; compress_ratio=4"
        )

    monkeypatch.setattr(MUSA_OPS, "_is_musa_tensor", lambda tensor: tensor is kv_score_input)
    monkeypatch.setattr(
        MUSA_OPS,
        "_try_tilelang_compress_forward_ratio4_decode_musa",
        fake_try_tilelang_compress_forward_ratio4_decode_musa,
    )
    monkeypatch.setattr(MUSA_OPS, "_musa_graph_capture_enabled", lambda: True)

    with pytest.raises(NotImplementedError) as exc_info:
        compress_forward_musa(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape,
            indices=indices,
            plan=plan,
            extra_data=None,
            head_dim=head_dim,
            compress_ratio=4,
        )

    message = str(exc_info.value)
    assert "DeepSeekV4 MUSA compress_forward ratio4 decode has no torch fallback" in message
    assert "kernel exception TypeError" in message
    assert "missing 1 required positional argument: 'head_dim'" in message
    assert "kv_score_buffer" in message
    assert "shape:(11452, 512)" in message
    assert "kv_score_input" in message
    assert "shape:(1, 512)" in message
    assert "ape" in message
    assert "shape:(8, 128)" in message
    assert "indices" in message
    assert "seq_lens" in message
    assert "extra_data=None" in message
    assert "out" in message
    assert "shape:(1, 128)" in message
    assert "head_dim=128" in message
    assert "compress_ratio=4" in message

def test_compress_forward_musa_ratio4_invokes_tilelang_fast_path(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_compress_forward_ratio4_decode_musa(
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extra_data: torch.Tensor | None,
        out: torch.Tensor,
        head_dim: int,
    ) -> tuple[bool, str | None]:
        calls.append((kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out, head_dim))
        out.fill_(6.0)
        return True, None

    monkeypatch.setattr(
        MUSA_OPS,
        "_try_tilelang_compress_forward_ratio4_decode_musa",
        fake_try_tilelang_compress_forward_ratio4_decode_musa,
    )

    head_dim = 128
    got = compress_forward_musa(
        kv_score_buffer=torch.zeros((1, 8, head_dim * 4), dtype=torch.float32),
        kv_score_input=torch.zeros((1, head_dim * 4), dtype=torch.float32),
        ape=torch.zeros((8, head_dim), dtype=torch.float32),
        indices=torch.tensor([0], dtype=torch.int32),
        plan=_DecodePlan(torch.tensor([4], dtype=torch.int32), compress_ratio=4),
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=4,
    )

    assert len(calls) == 1
    assert calls[0][5] is None
    assert calls[0][7] == head_dim
    torch.testing.assert_close(got, torch.full_like(got, 6.0), rtol=0, atol=0)

def test_compress_forward_musa_ratio4_decode_matches_reference() -> None:
    head_dim = 128
    kv_score_buffer = torch.zeros((1, 8, head_dim * 4), dtype=torch.float32)
    kv_score_input = torch.cat(
        [
            torch.full((head_dim,), 10.0),
            torch.full((head_dim,), 20.0),
            torch.full((head_dim,), -50.0),
            torch.full((head_dim,), 3.0),
        ],
        dim=0,
    ).reshape(1, -1)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=torch.tensor([0], dtype=torch.int32),
        plan=_DecodePlan(torch.tensor([4], dtype=torch.int32), compress_ratio=4),
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=4,
    )

    kv_window = torch.stack([torch.zeros(head_dim)] * 7 + [torch.full((head_dim,), 20.0)], dim=0)
    score_window = torch.stack(
        [torch.full((head_dim,), -1e9)] * 4
        + [torch.zeros(head_dim)] * 3
        + [torch.full((head_dim,), 3.0)],
        dim=0,
    )
    ref = _compress_ref(kv_window, score_window, ape)
    torch.testing.assert_close(got[0], ref.to(got.dtype))

def test_compress_forward_musa_ratio4_decode_page4_matches_reference() -> None:
    head_dim = 128
    kv_score_buffer = torch.zeros((2, 4, head_dim * 4), dtype=torch.float32)
    for block in range(2):
        for slot in range(4):
            kv_score_buffer[block, slot, :head_dim] = float(block * 100 + slot + 1)
            kv_score_buffer[block, slot, head_dim : head_dim * 2] = float(block * 100 + slot + 11)
            kv_score_buffer[block, slot, head_dim * 2 : head_dim * 3] = float(slot - 3)
            kv_score_buffer[block, slot, head_dim * 3 :] = float(slot + 5)
    kv_score_input = torch.cat(
        [
            torch.full((head_dim,), 101.0),
            torch.full((head_dim,), 202.0),
            torch.full((head_dim,), -7.0),
            torch.full((head_dim,), 13.0),
        ],
        dim=0,
    ).reshape(1, -1)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.tensor([1], dtype=torch.int32)
    extra_data = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([8], dtype=torch.int32)

    got_buffer = kv_score_buffer.clone()
    got = compress_forward_musa(
        kv_score_buffer=got_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=_DecodePlan(seq_lens, compress_ratio=4),
        extra_data=extra_data,
        head_dim=head_dim,
        compress_ratio=4,
    )

    ref_buffer = kv_score_buffer.clone()
    ref_buffer[1, 3] = kv_score_input[0]
    kv_window = []
    score_window = []
    for slot in range(8):
        if slot < 4:
            src = ref_buffer[0, slot].reshape(4, head_dim)
            kv_window.append(src[0])
            score_window.append(src[2])
        else:
            src = ref_buffer[1, slot - 4].reshape(4, head_dim)
            kv_window.append(src[1])
            score_window.append(src[3])
    ref = _compress_ref(torch.stack(kv_window), torch.stack(score_window), ape)
    torch.testing.assert_close(got[0], ref.to(got.dtype), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_buffer, ref_buffer, rtol=0, atol=0)


def test_compress_forward_musa_ratio4_decode_page4_single_column_extra_data_matches_reference() -> None:
    head_dim = 128
    kv_score_buffer = torch.zeros((2, 4, head_dim * 4), dtype=torch.float32)
    for block in range(2):
        for slot in range(4):
            kv_score_buffer[block, slot, :head_dim] = float(block * 100 + slot + 1)
            kv_score_buffer[block, slot, head_dim : head_dim * 2] = float(block * 100 + slot + 11)
            kv_score_buffer[block, slot, head_dim * 2 : head_dim * 3] = float(slot - 3)
            kv_score_buffer[block, slot, head_dim * 3 :] = float(slot + 5)
    kv_score_input = torch.cat(
        [
            torch.full((head_dim,), 101.0),
            torch.full((head_dim,), 202.0),
            torch.full((head_dim,), -7.0),
            torch.full((head_dim,), 13.0),
        ],
        dim=0,
    ).reshape(1, -1)
    ape = torch.zeros((8, head_dim), dtype=torch.float32)
    indices = torch.tensor([1], dtype=torch.int32)
    extra_data = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([8], dtype=torch.int32)

    got_buffer = kv_score_buffer.clone()
    got = compress_forward_musa(
        kv_score_buffer=got_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=_DecodePlan(seq_lens, compress_ratio=4),
        extra_data=extra_data,
        head_dim=head_dim,
        compress_ratio=4,
    )

    ref_buffer = kv_score_buffer.clone()
    ref_buffer[1, 3] = kv_score_input[0]
    kv_window = []
    score_window = []
    for slot in range(8):
        if slot < 4:
            src = ref_buffer[0, slot].reshape(4, head_dim)
            kv_window.append(src[0])
            score_window.append(src[2])
        else:
            src = ref_buffer[1, slot - 4].reshape(4, head_dim)
            kv_window.append(src[1])
            score_window.append(src[3])
    ref = _compress_ref(torch.stack(kv_window), torch.stack(score_window), ape)
    torch.testing.assert_close(got[0], ref.to(got.dtype), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_buffer, ref_buffer, rtol=0, atol=0)


def test_compress_forward_musa_ratio4_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    kv_score_buffer_cpu = torch.zeros((1, 8, head_dim * 4), dtype=torch.float32)
    for slot in range(8):
        kv_score_buffer_cpu[0, slot, :head_dim] = float(slot + 1)
        kv_score_buffer_cpu[0, slot, head_dim : head_dim * 2] = float(slot + 11)
        kv_score_buffer_cpu[0, slot, head_dim * 2 : head_dim * 3] = float(slot - 3)
        kv_score_buffer_cpu[0, slot, head_dim * 3 :] = float(slot + 5)
    kv_score_input_cpu = torch.cat(
        [
            torch.full((head_dim,), 101.0),
            torch.full((head_dim,), 202.0),
            torch.full((head_dim,), -7.0),
            torch.full((head_dim,), 13.0),
        ],
        dim=0,
    ).reshape(1, -1)
    ape_cpu = torch.zeros((8, head_dim), dtype=torch.float32)

    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)
    kv_score_input = kv_score_input_cpu.to(device)
    ape = ape_cpu.to(device)
    indices = torch.tensor([0], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([8], device=device, dtype=torch.int32)
    out = torch.empty((1, head_dim), device=device, dtype=torch.float32)

    ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio4_decode_musa(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, None, out, head_dim
    )

    assert ok, failure
    ref_buffer = kv_score_buffer_cpu.clone()
    ref_buffer[0, 7] = kv_score_input_cpu[0]
    kv_window = []
    score_window = []
    for i in range(8):
        src = ref_buffer[0, (8 + i) % 8].reshape(4, head_dim)
        kv_window.append(src[0] if i < 4 else src[1])
        score_window.append(src[2] if i < 4 else src[3])
    ref = _compress_ref(torch.stack(kv_window, dim=0), torch.stack(score_window, dim=0), ape_cpu)
    torch.testing.assert_close(out.cpu()[0], ref, rtol=1e-5, atol=1e-5)

    del kv_score_buffer, kv_score_input, ape, indices, seq_lens, out
    torch.musa.synchronize()
    torch.musa.empty_cache()

def test_compress_forward_musa_ratio4_real_tilelang_page4_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    kv_score_buffer_cpu = torch.zeros((2, 4, head_dim * 4), dtype=torch.float32)
    for block in range(2):
        for slot in range(4):
            kv_score_buffer_cpu[block, slot, :head_dim] = float(block * 100 + slot + 1)
            kv_score_buffer_cpu[block, slot, head_dim : head_dim * 2] = float(block * 100 + slot + 11)
            kv_score_buffer_cpu[block, slot, head_dim * 2 : head_dim * 3] = float(slot - 3)
            kv_score_buffer_cpu[block, slot, head_dim * 3 :] = float(slot + 5)
    kv_score_input_cpu = torch.cat(
        [
            torch.full((head_dim,), 101.0),
            torch.full((head_dim,), 202.0),
            torch.full((head_dim,), -7.0),
            torch.full((head_dim,), 13.0),
        ],
        dim=0,
    ).reshape(1, -1)
    ape_cpu = torch.zeros((8, head_dim), dtype=torch.float32)

    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)
    kv_score_input = kv_score_input_cpu.to(device)
    ape = ape_cpu.to(device)
    indices = torch.tensor([1], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([8], device=device, dtype=torch.int32)
    extra_data = torch.tensor([[0, 0, 0, 0]], device=device, dtype=torch.int32)
    out = torch.empty((1, head_dim), device=device, dtype=torch.float32)

    ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio4_decode_musa(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out, head_dim
    )

    assert ok, failure
    ref_buffer = kv_score_buffer_cpu.clone()
    ref_buffer[1, 3] = kv_score_input_cpu[0]
    kv_window = []
    score_window = []
    for slot in range(8):
        if slot < 4:
            src = ref_buffer[0, slot].reshape(4, head_dim)
            kv_window.append(src[0])
            score_window.append(src[2])
        else:
            src = ref_buffer[1, slot - 4].reshape(4, head_dim)
            kv_window.append(src[1])
            score_window.append(src[3])
    ref = _compress_ref(torch.stack(kv_window), torch.stack(score_window), ape_cpu)
    torch.testing.assert_close(out.cpu()[0], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer.cpu(), ref_buffer, rtol=0, atol=0)

    del kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out
    torch.musa.synchronize()
    torch.musa.empty_cache()


def test_compress_forward_musa_ratio4_real_tilelang_page4_single_column_extra_data_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    kv_score_buffer_cpu = torch.zeros((2, 4, head_dim * 4), dtype=torch.float32)
    for block in range(2):
        for slot in range(4):
            kv_score_buffer_cpu[block, slot, :head_dim] = float(block * 100 + slot + 1)
            kv_score_buffer_cpu[block, slot, head_dim : head_dim * 2] = float(block * 100 + slot + 11)
            kv_score_buffer_cpu[block, slot, head_dim * 2 : head_dim * 3] = float(slot - 3)
            kv_score_buffer_cpu[block, slot, head_dim * 3 :] = float(slot + 5)
    kv_score_input_cpu = torch.cat(
        [
            torch.full((head_dim,), 101.0),
            torch.full((head_dim,), 202.0),
            torch.full((head_dim,), -7.0),
            torch.full((head_dim,), 13.0),
        ],
        dim=0,
    ).reshape(1, -1)
    ape_cpu = torch.zeros((8, head_dim), dtype=torch.float32)

    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)
    kv_score_input = kv_score_input_cpu.to(device)
    ape = ape_cpu.to(device)
    indices = torch.tensor([1], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([8], device=device, dtype=torch.int32)
    extra_data = torch.tensor([[0]], device=device, dtype=torch.int32)
    out = torch.empty((1, head_dim), device=device, dtype=torch.float32)

    ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio4_decode_musa(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out, head_dim
    )

    assert ok, failure
    ref_buffer = kv_score_buffer_cpu.clone()
    ref_buffer[1, 3] = kv_score_input_cpu[0]
    kv_window = []
    score_window = []
    for slot in range(8):
        if slot < 4:
            src = ref_buffer[0, slot].reshape(4, head_dim)
            kv_window.append(src[0])
            score_window.append(src[2])
        else:
            src = ref_buffer[1, slot - 4].reshape(4, head_dim)
            kv_window.append(src[1])
            score_window.append(src[3])
    ref = _compress_ref(torch.stack(kv_window), torch.stack(score_window), ape_cpu)
    torch.testing.assert_close(out.cpu()[0], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer.cpu(), ref_buffer, rtol=0, atol=0)

    del kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out
    torch.musa.synchronize()
    torch.musa.empty_cache()


def test_compress_forward_musa_ratio4_real_tilelang_page4_strided_tensors_match_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    kv_score_buffer_cpu = torch.zeros((2, 4, head_dim * 4), dtype=torch.float32)
    for block in range(2):
        for slot in range(4):
            kv_score_buffer_cpu[block, slot, :head_dim] = float(block * 100 + slot + 1)
            kv_score_buffer_cpu[block, slot, head_dim : head_dim * 2] = float(block * 100 + slot + 11)
            kv_score_buffer_cpu[block, slot, head_dim * 2 : head_dim * 3] = float(slot - 3)
            kv_score_buffer_cpu[block, slot, head_dim * 3 :] = float(slot + 5)
    kv_score_input_cpu = torch.cat(
        [
            torch.full((head_dim,), 101.0),
            torch.full((head_dim,), 202.0),
            torch.full((head_dim,), -7.0),
            torch.full((head_dim,), 13.0),
        ],
        dim=0,
    ).reshape(1, -1)
    ape_cpu = torch.zeros((8, head_dim), dtype=torch.float32)

    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)
    kv_score_input_backing = torch.empty((1, 2, head_dim * 4), device=device, dtype=torch.float32)
    kv_score_input_backing[:, 1, :].copy_(kv_score_input_cpu.to(device))
    kv_score_input = kv_score_input_backing[:, 1, :]
    ape_backing = torch.empty((8, 2, head_dim), device=device, dtype=torch.float32)
    ape_backing[:, 1, :].copy_(ape_cpu.to(device))
    ape = ape_backing[:, 1, :]
    indices_backing = torch.tensor([99, 1], device=device, dtype=torch.int32)
    seq_lens_backing = torch.tensor([99, 8], device=device, dtype=torch.int32)
    indices = indices_backing.as_strided((1,), (2,), storage_offset=1)
    seq_lens = seq_lens_backing.as_strided((1,), (2,), storage_offset=1)
    extra_data_backing = torch.tensor([[[99, 99, 99, 99], [0, 0, 0, 0]]], device=device, dtype=torch.int32)
    extra_data = extra_data_backing[:, 1, :]
    out_backing = torch.empty((1, 2, head_dim), device=device, dtype=torch.float32)
    out = out_backing[:, 1, :]

    assert kv_score_input.stride(0) != head_dim * 4
    assert not ape.is_contiguous()
    assert indices.stride(0) != 1
    assert seq_lens.stride(0) != 1
    assert extra_data.stride(0) != 4
    assert out.stride(0) != head_dim

    ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio4_decode_musa(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out, head_dim
    )

    assert ok, failure
    ref_buffer = kv_score_buffer_cpu.clone()
    ref_buffer[1, 3] = kv_score_input_cpu[0]
    kv_window = []
    score_window = []
    for slot in range(8):
        if slot < 4:
            src = ref_buffer[0, slot].reshape(4, head_dim)
            kv_window.append(src[0])
            score_window.append(src[2])
        else:
            src = ref_buffer[1, slot - 4].reshape(4, head_dim)
            kv_window.append(src[1])
            score_window.append(src[3])
    ref = _compress_ref(torch.stack(kv_window), torch.stack(score_window), ape_cpu)
    torch.testing.assert_close(out.cpu()[0], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv_score_buffer.cpu(), ref_buffer, rtol=0, atol=0)

    del kv_score_buffer, kv_score_input_backing, kv_score_input, ape_backing, ape, indices_backing, seq_lens_backing, indices, seq_lens, extra_data_backing, extra_data, out_backing, out
    torch.musa.synchronize()
    torch.musa.empty_cache()


def test_compress_forward_musa_ratio4_real_tilelang_flat_state_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    kv_score_buffer_cpu = torch.zeros((8, head_dim * 4), dtype=torch.float32)
    for slot in range(8):
        kv_score_buffer_cpu[slot, :head_dim] = float(slot + 1)
        kv_score_buffer_cpu[slot, head_dim : head_dim * 2] = float(slot + 11)
        kv_score_buffer_cpu[slot, head_dim * 2 : head_dim * 3] = float(slot - 3)
        kv_score_buffer_cpu[slot, head_dim * 3 :] = float(slot + 5)
    kv_score_input_cpu = torch.cat(
        [
            torch.full((head_dim,), 101.0),
            torch.full((head_dim,), 202.0),
            torch.full((head_dim,), -7.0),
            torch.full((head_dim,), 13.0),
        ],
        dim=0,
    ).reshape(1, -1)
    ape_cpu = torch.zeros((8, head_dim), dtype=torch.float32)

    kv_score_buffer = kv_score_buffer_cpu.clone().to(device)
    kv_score_input = kv_score_input_cpu.to(device)
    ape = ape_cpu.to(device)
    indices = torch.tensor([0], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([8], device=device, dtype=torch.int32)
    out = torch.empty((1, head_dim), device=device, dtype=torch.float32)

    ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio4_decode_musa(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, None, out, head_dim
    )

    assert ok, failure
    ref_buffer = kv_score_buffer_cpu.clone()
    ref_buffer[7] = kv_score_input_cpu[0]
    kv_window = []
    score_window = []
    for i in range(8):
        src = ref_buffer[(8 + i) % 8].reshape(4, head_dim)
        kv_window.append(src[0] if i < 4 else src[1])
        score_window.append(src[2] if i < 4 else src[3])
    ref = _compress_ref(torch.stack(kv_window, dim=0), torch.stack(score_window, dim=0), ape_cpu)
    torch.testing.assert_close(out.cpu()[0], ref, rtol=1e-5, atol=1e-5)

    del kv_score_buffer, kv_score_input, ape, indices, seq_lens, out
    torch.musa.synchronize()
    torch.musa.empty_cache()

def test_compress_forward_musa_ratio128_invokes_tilelang_fast_path(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_compress_forward_ratio128_decode_musa(
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        head_dim: int,
    ) -> tuple[bool, str | None]:
        calls.append((kv_score_buffer, kv_score_input, ape, indices, seq_lens, out, head_dim))
        out.fill_(7.0)
        return True, None

    monkeypatch.setattr(
        MUSA_OPS,
        "_try_tilelang_compress_forward_ratio128_decode_musa",
        fake_try_tilelang_compress_forward_ratio128_decode_musa,
    )

    head_dim = 128
    got = compress_forward_musa(
        kv_score_buffer=torch.zeros((1, 128, head_dim * 2), dtype=torch.float32),
        kv_score_input=torch.zeros((1, head_dim * 2), dtype=torch.float32),
        ape=torch.zeros((128, head_dim), dtype=torch.float32),
        indices=torch.tensor([0], dtype=torch.int32),
        plan=_DecodePlan(torch.tensor([128], dtype=torch.int32), compress_ratio=128),
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=128,
    )

    assert len(calls) == 1
    assert calls[0][6] == head_dim
    torch.testing.assert_close(got, torch.full_like(got, 7.0), rtol=0, atol=0)

def test_compress_forward_musa_ratio128_decode_matches_reference() -> None:
    head_dim = 128
    kv = torch.stack([torch.full((head_dim,), float(i + 1)) for i in range(128)], dim=0)
    score = torch.stack([torch.full((head_dim,), float(i - 64)) for i in range(128)], dim=0)
    kv_score_buffer = torch.stack([torch.cat([kv[i], score[i]], dim=0) for i in range(128)], dim=0).unsqueeze(0)
    kv_score_input = torch.cat([torch.full((head_dim,), 999.0), torch.full((head_dim,), -999.0)], dim=0).reshape(1, -1)
    ape = torch.zeros((128, head_dim), dtype=torch.float32)

    got = compress_forward_musa(
        kv_score_buffer=kv_score_buffer.clone(),
        kv_score_input=kv_score_input,
        ape=ape,
        indices=torch.tensor([0], dtype=torch.int32),
        plan=_DecodePlan(torch.tensor([128], dtype=torch.int32), compress_ratio=128),
        extra_data=None,
        head_dim=head_dim,
        compress_ratio=128,
    )

    ref_buffer = kv_score_buffer.clone()
    ref_buffer[0, 127] = kv_score_input[0]
    window = ref_buffer[0].reshape(128, 2, head_dim)
    ref = _compress_ref(window[:, 0, :], window[:, 1, :], ape)
    torch.testing.assert_close(got[0], ref.to(got.dtype))

def test_compress_forward_musa_ratio128_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    kv = torch.stack([torch.full((head_dim,), float(i + 1)) for i in range(128)], dim=0)
    score = torch.stack([torch.full((head_dim,), float(i - 64)) for i in range(128)], dim=0)
    kv_score_buffer_cpu = torch.stack([torch.cat([kv[i], score[i]], dim=0) for i in range(128)], dim=0).unsqueeze(0)
    kv_score_input_cpu = torch.cat(
        [torch.full((head_dim,), 999.0), torch.full((head_dim,), -999.0)], dim=0
    ).reshape(1, -1)
    ape_cpu = torch.zeros((128, head_dim), dtype=torch.float32)

    kv_score_buffer = kv_score_buffer_cpu.to(device)
    kv_score_input = kv_score_input_cpu.to(device)
    ape = ape_cpu.to(device)
    indices = torch.tensor([0], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([128], device=device, dtype=torch.int32)
    out = torch.empty((1, head_dim), device=device, dtype=torch.float32)

    ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio128_decode_musa(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, out, head_dim
    )

    assert ok, failure
    ref_buffer = kv_score_buffer_cpu.clone()
    ref_buffer[0, 127] = kv_score_input_cpu[0]
    window = ref_buffer[0].reshape(128, 2, head_dim)
    ref = _compress_ref(window[:, 0, :], window[:, 1, :], ape_cpu)
    torch.testing.assert_close(out.cpu()[0], ref, rtol=1e-5, atol=1e-5)

def test_compress_forward_musa_ratio128_real_tilelang_strided_tensors_match_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 128
    kv = torch.stack([torch.full((head_dim,), float(i + 1)) for i in range(128)], dim=0)
    score = torch.stack([torch.full((head_dim,), float(i - 64)) for i in range(128)], dim=0)
    kv_score_buffer_cpu = torch.stack([torch.cat([kv[i], score[i]], dim=0) for i in range(128)], dim=0).unsqueeze(0)
    kv_score_input_cpu = torch.cat(
        [torch.full((head_dim,), 999.0), torch.full((head_dim,), -999.0)], dim=0
    ).reshape(1, -1)
    ape_cpu = torch.zeros((128, head_dim), dtype=torch.float32)

    kv_score_buffer = kv_score_buffer_cpu.to(device)
    kv_score_input_backing = torch.empty((1, 2, head_dim * 2), device=device, dtype=torch.float32)
    kv_score_input_backing[:, 1, :].copy_(kv_score_input_cpu.to(device))
    kv_score_input = kv_score_input_backing[:, 1, :]
    ape_backing = torch.empty((128, 2, head_dim), device=device, dtype=torch.float32)
    ape_backing[:, 1, :].copy_(ape_cpu.to(device))
    ape = ape_backing[:, 1, :]
    indices_backing = torch.tensor([99, 0], device=device, dtype=torch.int32)
    seq_lens_backing = torch.tensor([99, 128], device=device, dtype=torch.int32)
    indices = indices_backing.as_strided((1,), (2,), storage_offset=1)
    seq_lens = seq_lens_backing.as_strided((1,), (2,), storage_offset=1)
    out_backing = torch.empty((1, 2, head_dim), device=device, dtype=torch.float32)
    out = out_backing[:, 1, :]

    assert kv_score_input.stride(0) != head_dim * 2
    assert not ape.is_contiguous()
    assert indices.stride(0) != 1
    assert seq_lens.stride(0) != 1
    assert out.stride(0) != head_dim

    ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio128_decode_musa(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, out, head_dim
    )

    assert ok, failure
    ref_buffer = kv_score_buffer_cpu.clone()
    ref_buffer[0, 127] = kv_score_input_cpu[0]
    window = ref_buffer[0].reshape(128, 2, head_dim)
    ref = _compress_ref(window[:, 0, :], window[:, 1, :], ape_cpu)
    torch.testing.assert_close(out.cpu()[0], ref, rtol=1e-5, atol=1e-5)

    del kv_score_buffer, kv_score_input_backing, kv_score_input, ape_backing, ape, indices_backing, seq_lens_backing, indices, seq_lens, out_backing, out
    torch.musa.synchronize()
    torch.musa.empty_cache()

def test_compress_forward_musa_ratio128_flat_real_tilelang_path_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    head_dim = 512
    block_id = 3
    num_rows = (block_id + 1) * 128
    kv = torch.stack([torch.full((head_dim,), float(i + 1)) for i in range(128)], dim=0)
    score = torch.stack([torch.full((head_dim,), float(i - 64)) for i in range(128)], dim=0)
    kv_score_buffer_cpu = torch.zeros((num_rows, head_dim * 2), dtype=torch.float32)
    kv_score_buffer_cpu[block_id * 128 : (block_id + 1) * 128] = torch.stack(
        [torch.cat([kv[i], score[i]], dim=0) for i in range(128)], dim=0
    )
    kv_score_input_cpu = torch.cat(
        [torch.full((head_dim,), 999.0), torch.full((head_dim,), -999.0)], dim=0
    ).reshape(1, -1)
    ape_cpu = torch.zeros((128, head_dim), dtype=torch.float32)

    kv_score_buffer = kv_score_buffer_cpu.to(device)
    kv_score_input = kv_score_input_cpu.to(device)
    ape = ape_cpu.to(device)
    indices = torch.tensor([block_id], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([128], device=device, dtype=torch.int32)
    out = torch.empty((1, head_dim), device=device, dtype=torch.float32)

    ok, failure = MUSA_OPS._try_tilelang_compress_forward_ratio128_decode_musa(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, out, head_dim
    )

    assert ok, failure
    ref_buffer = kv_score_buffer_cpu.clone()
    ref_buffer[block_id * 128 + 127] = kv_score_input_cpu[0]
    window = ref_buffer[block_id * 128 : (block_id + 1) * 128].reshape(128, 2, head_dim)
    ref = _compress_ref(window[:, 0, :], window[:, 1, :], ape_cpu)
    torch.testing.assert_close(out.cpu()[0], ref, rtol=1e-5, atol=1e-5)
