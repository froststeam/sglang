from __future__ import annotations

import os

import pytest
import torch

import sglang.jit_kernel.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_musa_ci

from ..utils import get_musa_device

register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="DeepSeek V4 MUSA operator test is opt-in outside smoke CI",
)


class _FakeTopKModule:
    def __init__(self) -> None:
        self.calls = []

    def topk_transform(self, *args) -> None:
        self.calls.append(args)


def test_topk_transform_512_musa_jit_env_dispatch(monkeypatch) -> None:
    module = _FakeTopKModule()
    monkeypatch.setenv("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK512", "1")
    monkeypatch.setattr(deepseek_v4, "_is_musa_tensor", lambda _: True)
    monkeypatch.setattr(deepseek_v4, "_jit_topk_musa_module", lambda: module)
    monkeypatch.setattr(
        deepseek_v4,
        "_jit_topk_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected CUDA topk path")),
    )

    scores = torch.empty((2, 1024), dtype=torch.float32)
    seq_lens = torch.tensor([1024, 512], dtype=torch.int32)
    page_tables = torch.empty((2, 16), dtype=torch.int32)
    out_page_indices = torch.empty((2, 512), dtype=torch.int32)
    out_raw_indices = torch.empty((2, 512), dtype=torch.int32)

    deepseek_v4.topk_transform_512(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        64,
        out_raw_indices,
    )

    assert len(module.calls) == 1
    assert module.calls[0] == (
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        64,
        out_raw_indices,
    )


def test_topk_transform_512_musa_jit_is_opt_in(monkeypatch) -> None:
    module = _FakeTopKModule()
    monkeypatch.delenv("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK512", raising=False)
    monkeypatch.setattr(deepseek_v4, "_is_musa_tensor", lambda _: True)
    monkeypatch.setattr(
        deepseek_v4,
        "_jit_topk_musa_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected MUSA JIT topk path")),
    )
    monkeypatch.setattr(deepseek_v4, "_jit_topk_module", lambda: module)

    scores = torch.empty((1, 1024), dtype=torch.float32)
    seq_lens = torch.tensor([1024], dtype=torch.int32)
    page_tables = torch.empty((1, 16), dtype=torch.int32)
    out_page_indices = torch.empty((1, 512), dtype=torch.int32)

    deepseek_v4.topk_transform_512(scores, seq_lens, page_tables, out_page_indices, 64)

    assert len(module.calls) == 1



def test_topk_transform_1024_musa_jit_env_dispatch(monkeypatch) -> None:
    module = _FakeTopKModule()
    monkeypatch.setenv("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK1024", "1")
    monkeypatch.setattr(deepseek_v4, "_is_musa_tensor", lambda _: True)
    monkeypatch.setattr(deepseek_v4, "_jit_topk1024_musa_module", lambda: module)
    monkeypatch.setattr(
        deepseek_v4,
        "_jit_topk1024_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected generic topk1024 path")),
    )

    scores = torch.empty((2, 2048), dtype=torch.float32)
    seq_lens = torch.tensor([2048, 1024], dtype=torch.int32)
    page_tables = torch.empty((2, 32), dtype=torch.int32)
    out_page_indices = torch.empty((2, 1024), dtype=torch.int32)
    out_raw_indices = torch.empty((2, 1024), dtype=torch.int32)

    deepseek_v4.topk_transform_512(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        64,
        out_raw_indices,
    )

    assert len(module.calls) == 1
    assert module.calls[0] == (
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        64,
        out_raw_indices,
    )


def test_topk_transform_1024_musa_jit_is_opt_in(monkeypatch) -> None:
    module = _FakeTopKModule()
    monkeypatch.delenv("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK1024", raising=False)
    monkeypatch.setattr(deepseek_v4, "_is_musa_tensor", lambda _: True)
    monkeypatch.setattr(
        deepseek_v4,
        "_jit_topk1024_musa_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected MUSA JIT topk1024 path")),
    )
    monkeypatch.setattr(deepseek_v4, "_jit_topk1024_module", lambda: module)

    scores = torch.empty((1, 2048), dtype=torch.float32)
    seq_lens = torch.tensor([2048], dtype=torch.int32)
    page_tables = torch.empty((1, 32), dtype=torch.int32)
    out_page_indices = torch.empty((1, 1024), dtype=torch.int32)

    deepseek_v4.topk_transform_512(scores, seq_lens, page_tables, out_page_indices, 64)

    assert len(module.calls) == 1


def _require_real_musa_topk_test() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real MUSA topk validation")
    get_musa_device()


def _assert_repeat_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch_count = int((actual != expected).sum().item())
    pytest.fail(
        f"{name} is not repeat-deterministic: "
        f"mismatch_count={mismatch_count}/{actual.numel()}"
    )


def _assert_same_index_set(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    _assert_repeat_exact(name, torch.sort(actual).values, torch.sort(expected).values)


def _run_topk_transform_512_musa(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_page_indices = torch.empty((scores.shape[0], 512), device=scores.device, dtype=torch.int32)
    out_raw_indices = torch.empty((scores.shape[0], 512), device=scores.device, dtype=torch.int32)
    deepseek_v4.topk_transform_512(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        64,
        out_raw_indices,
    )
    torch.musa.synchronize()
    return out_page_indices, out_raw_indices


def test_topk_transform_512_musa_jit_selected_set_is_repeat_deterministic(monkeypatch) -> None:
    _require_real_musa_topk_test()
    monkeypatch.setenv("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK512", "1")
    device = get_musa_device()
    batch = 17
    max_seq_len = 1024
    torch.manual_seed(20263725)
    scores = torch.randn((batch, max_seq_len), device=device, dtype=torch.float32)
    seq_lens = torch.full((batch,), max_seq_len, device=device, dtype=torch.int32)
    page_tables = torch.arange(batch * (max_seq_len // 64), device=device, dtype=torch.int32).reshape(batch, -1)

    expected_pages, expected_raw = _run_topk_transform_512_musa(scores, seq_lens, page_tables)
    for repeat_idx in range(20):
        actual_pages, actual_raw = _run_topk_transform_512_musa(scores, seq_lens, page_tables)
        _assert_same_index_set(
            f"topk_transform_512 page set repeat={repeat_idx}",
            actual_pages,
            expected_pages,
        )
        _assert_same_index_set(
            f"topk_transform_512 raw set repeat={repeat_idx}",
            actual_raw,
            expected_raw,
        )


def test_topk_transform_512_musa_jit_probe_row_selected_set_is_batch_shape_invariant(monkeypatch) -> None:
    _require_real_musa_topk_test()
    monkeypatch.setenv("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK512", "1")
    device = get_musa_device()
    max_seq_len = 1024
    torch.manual_seed(20263825)
    probe_scores = torch.randn((1, max_seq_len), device=device, dtype=torch.float32)
    probe_seq_lens = torch.tensor([max_seq_len], device=device, dtype=torch.int32)
    probe_page_tables = torch.arange(max_seq_len // 64, device=device, dtype=torch.int32).view(1, -1)
    expected_pages, expected_raw = _run_topk_transform_512_musa(
        probe_scores,
        probe_seq_lens,
        probe_page_tables,
    )

    batch = 17
    scores = torch.randn((batch, max_seq_len), device=device, dtype=torch.float32)
    seq_lens = torch.full((batch,), max_seq_len, device=device, dtype=torch.int32)
    page_tables = torch.arange(batch * (max_seq_len // 64), device=device, dtype=torch.int32).reshape(batch, -1)
    probe_positions = [0, 5, 16]
    for pos in probe_positions:
        scores[pos].copy_(probe_scores[0])
        seq_lens[pos] = probe_seq_lens[0]
        page_tables[pos].copy_(probe_page_tables[0])

    actual_pages, actual_raw = _run_topk_transform_512_musa(scores, seq_lens, page_tables)
    for pos in probe_positions:
        _assert_same_index_set(
            f"topk_transform_512 page set pos={pos}",
            actual_pages[pos],
            expected_pages[0],
        )
        _assert_same_index_set(
            f"topk_transform_512 raw set pos={pos}",
            actual_raw[pos],
            expected_raw[0],
        )
