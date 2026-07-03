"""DeepSeek V4 MUSA operator-only acceptance benchmarks.

These benchmarks are intentionally opt-in and are the first gate for the
operator-side 2.5x-H200 work.  They measure focused operators before any E2E
serving run is needed.
"""

from __future__ import annotations

import json
import os
import statistics as stats
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

import sglang.jit_kernel.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_musa_ci
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.cache_ops import (
    fused_store_cache_musa,
)
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.compress_ops import (
    compress_forward_musa,
)
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.mhc_ops import (
    mhc_post,
    mhc_pre_big_fuse,
)
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.norm_rope_ops import (
    _try_tilelang_rope_hadamard_inplace_musa,
    compress_fused_norm_rope_prefill_inplace_musa,
    fused_rope_musa,
    rmsnorm_self_musa,
)

from ..utils import get_musa_device


register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="opt-in benchmark; set SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH=1 manually",
)

_DEFAULT_WARMUP = 5
_DEFAULT_ITERS = 20
_HIDDEN_SIZE = 4096
_MHC_MULT = 4
_MHC_MULT3 = _MHC_MULT * 2 + _MHC_MULT * _MHC_MULT
_PREFILL_TOKENS = 8192
_PREFILL_LAYERS = 61
_FLASHMLA_CACHE_CALLS = 84


@dataclass(frozen=True)
class BenchStats:
    median_ms: float
    min_ms: float
    p95_ms: float
    mean_ms: float
    samples: int


@dataclass(frozen=True)
class BenchResult:
    name: str
    budget_ms: float
    passed_budget: bool
    stats: BenchStats
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _PrefillCompressPlan:
    compress_ratio: int
    compress_plan: torch.Tensor
    write_plan: torch.Tensor


def _require_operator_benchmark_env() -> torch.device:
    if os.environ.get("SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH") != "1":
        pytest.skip(
            "set SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH=1 to run "
            "DeepSeek V4 MUSA operator acceptance benchmarks"
        )
    pytest.importorskip("tilelang")
    return get_musa_device()


def _configure_production_like_env() -> None:
    # Match the validated no-MTP B1 8K prefill service path as closely as a
    # single-process operator benchmark can.  Callers can still override these.
    os.environ.setdefault("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK512", "1")
    os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "1")
    os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "1")
    os.environ.setdefault("SGLANG_OPT_MHC_PRENORM_BACKEND", "deepgemm")
    os.environ.setdefault("SGLANG_OPT_MHC_PRENORM_SPLIT_K", "32")
    os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM_SPLIT_K", "32")
    os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "1")
    os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "1")
    os.environ.setdefault("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_VECTOR_WRITE", "1")
    os.environ.setdefault("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_C128_PARALLEL_REDUCE", "1")
    os.environ.setdefault("SGLANG_OPT_USE_TILEKERNELS_FP8_QUANT", "1")


def _sync() -> None:
    torch.musa.synchronize()


def _bench_device_ms(
    fn: Callable[[], object],
    *,
    warmup: int | None = None,
    iters: int | None = None,
) -> BenchStats:
    warmup = int(os.environ.get("SGLANG_DSV4_OPERATOR_BENCH_WARMUP", warmup or _DEFAULT_WARMUP))
    iters = int(os.environ.get("SGLANG_DSV4_OPERATOR_BENCH_ITERS", iters or _DEFAULT_ITERS))
    for _ in range(warmup):
        fn()
    _sync()

    values: list[float] = []
    for _ in range(iters):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        values.append(float(start.elapsed_time(end)))
    values_sorted = sorted(values)
    p95_idx = min(len(values_sorted) - 1, int(round((len(values_sorted) - 1) * 0.95)))
    return BenchStats(
        median_ms=values_sorted[len(values_sorted) // 2],
        min_ms=values_sorted[0],
        p95_ms=values_sorted[p95_idx],
        mean_ms=float(stats.fmean(values_sorted)),
        samples=len(values_sorted),
    )


def _record_result(
    name: str,
    stats_obj: BenchStats,
    budget_ms: float,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    result = BenchResult(
        name=name,
        budget_ms=budget_ms,
        passed_budget=stats_obj.median_ms <= budget_ms,
        stats=stats_obj,
        metadata=metadata,
    )
    print("OPERATOR_BENCH_RESULT " + json.dumps(asdict(result), sort_keys=True))
    output_path = os.environ.get("SGLANG_DSV4_OPERATOR_BENCH_JSONL")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")
    if os.environ.get("SGLANG_DSV4_OPERATOR_BENCH_ENFORCE") == "1":
        assert result.passed_budget, (
            f"{name} median {stats_obj.median_ms:.3f} ms exceeds "
            f"budget {budget_ms:.3f} ms"
        )


def _logical_gbps(logical_bytes: int, median_ms: float) -> float:
    if median_ms <= 0.0:
        return 0.0
    return float(logical_bytes) / (median_ms * 1.0e6)


def _shape_metadata(
    *,
    shape: dict[str, int | str | bool],
    dispatch_branch: str,
    logical_bytes: int,
    trace_kernel: str,
    stats_obj: BenchStats,
    config: dict[str, int | str | bool] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "shape": shape,
        "dispatch_branch": dispatch_branch,
        "logical_bytes": int(logical_bytes),
        "gbps": _logical_gbps(logical_bytes, stats_obj.median_ms),
        "trace_kernel": trace_kernel,
    }
    if config is not None:
        metadata["config"] = config
    return metadata


def _compress_zero_logical_bytes(num_tokens: int, head_dim: int) -> int:
    return num_tokens * head_dim * 4


def _compress_c4_reduce_logical_bytes(num_rows: int, head_dim: int) -> int:
    # Per output element: 8 kv + 8 score + 8 ape loads, plus one fp32 output store.
    return num_rows * head_dim * 25 * 4


def _compress_c4_write_logical_bytes(num_rows: int, head_dim: int) -> int:
    # C4 write copies four fp32 head_dim segments from input to cache.
    return num_rows * head_dim * 4 * 2 * 4


def _compress_c128_reduce_logical_bytes(num_rows: int, head_dim: int) -> int:
    # Per output element: 128 kv + 128 score + 128 ape loads, plus one fp32 output store.
    return num_rows * head_dim * 385 * 4


def _compress_c128_write_logical_bytes(num_rows: int, head_dim: int) -> int:
    # C128 write copies two fp32 head_dim segments from input to cache.
    return num_rows * head_dim * 2 * 2 * 4


def _make_mhc_inputs(device: torch.device, num_tokens: int = _PREFILL_TOKENS):
    torch.manual_seed(2026052901)
    residual = (torch.randn((num_tokens, _MHC_MULT, _HIDDEN_SIZE), device=device) * 0.2).to(
        torch.bfloat16
    )
    x = (torch.randn((num_tokens, _HIDDEN_SIZE), device=device) * 0.2).to(torch.bfloat16)
    post = torch.randn((num_tokens, _MHC_MULT, 1), device=device, dtype=torch.float32)
    comb = torch.randn((num_tokens, _MHC_MULT, _MHC_MULT), device=device, dtype=torch.float32)
    fn = torch.randn((_MHC_MULT3, _MHC_MULT * _HIDDEN_SIZE), device=device, dtype=torch.float32)
    mhc_scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    mhc_base = torch.zeros((_MHC_MULT3,), device=device, dtype=torch.float32)
    return residual, x, post, comb, fn, mhc_scale, mhc_base


def test_operator_bench_mhc_prefill_acceptance() -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()
    residual, x, post, comb, fn, mhc_scale, mhc_base = _make_mhc_inputs(device)

    post_out = mhc_post(x, residual, post, comb)
    pre_post, pre_comb, pre_layer = mhc_pre_big_fuse(
        residual,
        fn,
        mhc_scale,
        mhc_base,
        rms_eps=1e-6,
        mhc_pre_eps=1e-6,
        mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=1.0,
        sinkhorn_repeat=2,
    )
    _sync()
    assert post_out.shape == residual.shape
    assert pre_post.shape == (_PREFILL_TOKENS, _MHC_MULT, 1)
    assert pre_comb.shape == (_PREFILL_TOKENS, _MHC_MULT, _MHC_MULT)
    assert pre_layer.shape == (_PREFILL_TOKENS, _HIDDEN_SIZE)

    post_stats = _bench_device_ms(lambda: mhc_post(x, residual, post, comb))
    _record_result("mhc_post_2d_b1_8192", post_stats, budget_ms=30.0)

    pre_stats = _bench_device_ms(
        lambda: mhc_pre_big_fuse(
            residual,
            fn,
            mhc_scale,
            mhc_base,
            rms_eps=1e-6,
            mhc_pre_eps=1e-6,
            mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=1.0,
            sinkhorn_repeat=2,
        ),
        warmup=3,
        iters=10,
    )
    _record_result("mhc_pre_big_fuse_b1_8192", pre_stats, budget_ms=40.0)


def test_operator_bench_topk512_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()
    monkeypatch.setenv("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK512", "1")
    torch.manual_seed(2026052902)
    batch = _PREFILL_TOKENS
    max_seq_len = 1024
    scores = torch.randn((batch, max_seq_len), device=device, dtype=torch.float32)
    seq_lens = torch.full((batch,), max_seq_len, device=device, dtype=torch.int32)
    page_tables = torch.arange(batch * (max_seq_len // 64), device=device, dtype=torch.int32).reshape(
        batch, -1
    )
    out_page = torch.empty((batch, 512), device=device, dtype=torch.int32)
    out_raw = torch.empty((batch, 512), device=device, dtype=torch.int32)

    def run() -> None:
        deepseek_v4.topk_transform_512(scores, seq_lens, page_tables, out_page, 64, out_raw)

    run()
    _sync()
    assert out_page.shape == out_raw.shape == (batch, 512)
    topk_stats = _bench_device_ms(run)
    _record_result("topk_transform_512_b1_8192", topk_stats, budget_ms=10.0)


def _make_freqs(device: torch.device, max_pos: int, rope_dim: int) -> torch.Tensor:
    torch.manual_seed(2026052903)
    angles = torch.randn((max_pos, rope_dim // 2), device=device, dtype=torch.float32)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


def test_operator_bench_norm_rope_acceptance() -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()
    torch.manual_seed(2026052904)
    q512 = (torch.randn((_PREFILL_TOKENS, 512), device=device) * 0.2).to(torch.bfloat16)
    q_rope = (torch.randn((_PREFILL_TOKENS, 8, 64), device=device) * 0.2).to(torch.bfloat16)
    k_rope = (torch.randn((_PREFILL_TOKENS, 1, 64), device=device) * 0.2).to(torch.bfloat16)
    positions = torch.arange(_PREFILL_TOKENS, device=device, dtype=torch.int64)
    freqs = _make_freqs(device, _PREFILL_TOKENS, 64)

    rms = rmsnorm_self_musa(q512, 1e-5)
    fused_rope_musa(q_rope, k_rope, freqs, positions)
    _sync()
    assert rms.shape == q512.shape

    rms_stats = _bench_device_ms(lambda: rmsnorm_self_musa(q512, 1e-5))
    _record_result("rmsnorm_self_512_b1_8192", rms_stats, budget_ms=4.0)

    rope_stats = _bench_device_ms(lambda: fused_rope_musa(q_rope, k_rope, freqs, positions))
    _record_result("fused_rope_qk_64_b1_8192", rope_stats, budget_ms=6.0)


def test_operator_bench_compress_fused_norm_rope_prefill_acceptance() -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()
    torch.manual_seed(2026060401)
    num_tokens = int(os.environ.get("SGLANG_DSV4_OPERATOR_BENCH_NORM_ROPE_TOKENS", _PREFILL_TOKENS))
    compress_ratio = int(os.environ.get("SGLANG_DSV4_OPERATOR_BENCH_NORM_ROPE_COMPRESS_RATIO", "4"))
    seq_len_target = int(os.environ.get("SGLANG_DSV4_OPERATOR_BENCH_NORM_ROPE_SEQ_LEN", "32768"))
    prefix_len = max(0, seq_len_target - num_tokens)
    hidden_size = 512
    rope_dim = 64
    kv = (torch.randn((num_tokens, hidden_size), device=device) * 0.2).to(torch.bfloat16)
    weight = (torch.randn((hidden_size,), device=device) * 0.2 + 1.0).to(torch.bfloat16)
    freqs = _make_freqs(device, max(seq_len_target + 1, num_tokens + 1), rope_dim)
    rows = []
    effective_window = compress_ratio * (2 if compress_ratio == 4 else 1)
    for ragged_id in range(num_tokens):
        position = prefix_len + ragged_id
        if (position + 1) % compress_ratio == 0:
            rows.append(
                (
                    ragged_id,
                    0,
                    position,
                    effective_window - min(ragged_id + 1, effective_window),
                )
            )
    compress_plan = _pack_prefill_rows(rows, device)

    def run() -> None:
        work = kv.clone()
        compress_fused_norm_rope_prefill_inplace_musa(
            work, weight, 1e-5, freqs, compress_plan
        )

    run()
    _sync()
    stats_obj = _bench_device_ms(run)
    logical_bytes = len(rows) * (hidden_size * 2 * 4 + (rope_dim // 2) * 8 + 16)
    _record_result(
        f"compress_fused_norm_rope_prefill_h512_r64_c{compress_ratio}_b1_{num_tokens}",
        stats_obj,
        budget_ms=0.30,
        metadata=_shape_metadata(
            shape={
                "tokens": num_tokens,
                "compress_rows": len(rows),
                "hidden_size": hidden_size,
                "rope_dim": rope_dim,
                "compress_ratio": compress_ratio,
                "seq_len": seq_len_target,
            },
            dispatch_branch="compress_fused_norm_rope_prefill_direct_tilelang",
            logical_bytes=logical_bytes,
            trace_kernel="dsv4_compress_norm_rope_prefill_warp",
            stats_obj=stats_obj,
        ),
    )


def test_operator_bench_c4_indexer_rope_hadamard_acceptance() -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()
    torch.manual_seed(2026052909)
    q = (torch.randn((_PREFILL_TOKENS, 64, 128), device=device) * 0.2).to(torch.bfloat16)
    positions = torch.arange(_PREFILL_TOKENS, device=device, dtype=torch.int64)
    freqs = _make_freqs(device, _PREFILL_TOKENS, 64)

    def run() -> None:
        ok, reason = _try_tilelang_rope_hadamard_inplace_musa(q, freqs, positions)
        assert ok, reason

    run()
    _sync()
    stats_obj = _bench_device_ms(run)
    _record_result("c4_indexer_rope_hadamard_h64_b1_8192", stats_obj, budget_ms=2.0)


def _group_count(env_name: str, default: int) -> int:
    return max(1, int(os.environ.get(env_name, default)))


def test_operator_bench_norm_rope_quant_group_acceptance() -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()
    from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    torch.manual_seed(2026052907)
    layer_count = _group_count("SGLANG_DSV4_OPERATOR_BENCH_NORM_QUANT_LAYERS", _PREFILL_LAYERS)
    x512 = (torch.randn((_PREFILL_TOKENS, 512), device=device) * 0.2).to(torch.bfloat16)
    q_rope = (torch.randn((_PREFILL_TOKENS, 8, 64), device=device) * 0.2).to(torch.bfloat16)
    k_rope = (torch.randn((_PREFILL_TOKENS, 1, 64), device=device) * 0.2).to(torch.bfloat16)
    quant_in = (torch.randn((_PREFILL_TOKENS, _HIDDEN_SIZE), device=device) * 0.2).to(
        torch.bfloat16
    )
    positions = torch.arange(_PREFILL_TOKENS, device=device, dtype=torch.int64)
    freqs = _make_freqs(device, _PREFILL_TOKENS, 64)

    def run_layer_chain() -> None:
        rmsnorm_self_musa(x512, 1e-5)
        fused_rope_musa(q_rope, k_rope, freqs, positions)
        sglang_per_token_group_quant_fp8(quant_in, 128)

    def run_group() -> None:
        for _ in range(layer_count):
            run_layer_chain()

    run_layer_chain()
    _sync()
    group_stats = _bench_device_ms(run_group, warmup=1, iters=5)
    _record_result(f"norm_rope_quant_group_b1_8192_x{layer_count}", group_stats, budget_ms=120.0)


def _pack_prefill_rows(rows: list[tuple[int, int, int, int]], device: torch.device) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 16), device=device, dtype=torch.uint8)
    return torch.tensor(rows, device=device, dtype=torch.int32).view(torch.uint8).reshape(len(rows), 16)


def _make_prefill_compress_plan(
    *,
    compress_ratio: int,
    num_tokens: int,
    device: torch.device,
) -> _PrefillCompressPlan:
    effective_ratio = compress_ratio * (2 if compress_ratio == 4 else 1)
    prefix_len = effective_ratio
    compress_rows: list[tuple[int, int, int, int]] = []
    write_rows: list[tuple[int, int, int, int]] = []
    base_pos = (prefix_len + num_tokens) // compress_ratio * compress_ratio
    start_write_pos = base_pos - compress_ratio if compress_ratio == 4 and base_pos >= compress_ratio else base_pos
    for ragged_id in range(num_tokens):
        position = prefix_len + ragged_id
        window_len = effective_ratio - min(ragged_id + 1, effective_ratio)
        row = (ragged_id, 0, position, window_len)
        if (position + 1) % compress_ratio == 0:
            compress_rows.append(row)
        if position >= start_write_pos:
            write_rows.append(row)
    return _PrefillCompressPlan(
        compress_ratio=compress_ratio,
        compress_plan=_pack_prefill_rows(compress_rows, device),
        write_plan=_pack_prefill_rows(write_rows, device),
    )


def _make_compress_inputs(device: torch.device, compress_ratio: int, num_tokens: int = _PREFILL_TOKENS):
    torch.manual_seed(2026052906 + compress_ratio)
    head_dim = 128
    ring_size = 8 if compress_ratio == 4 else 128
    width = head_dim * (4 if compress_ratio == 4 else 2)
    kv_score_buffer = torch.randn((1, ring_size, width), device=device, dtype=torch.float32) * 0.01
    kv_score_input = torch.randn((num_tokens, width), device=device, dtype=torch.float32) * 0.01
    ape = torch.randn((ring_size, head_dim), device=device, dtype=torch.float32) * 0.01
    indices = torch.zeros((1,), device=device, dtype=torch.int32)
    plan = _make_prefill_compress_plan(
        compress_ratio=compress_ratio,
        num_tokens=num_tokens,
        device=device,
    )
    return kv_score_buffer, kv_score_input, ape, indices, plan, head_dim


def test_operator_bench_compress_prefill_acceptance() -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()

    c4_args = _make_compress_inputs(device, 4)
    c128_args = _make_compress_inputs(device, 128)

    def run_c4() -> None:
        compress_forward_musa(*c4_args[:4], c4_args[4], None, head_dim=c4_args[5], compress_ratio=4)

    def run_c128() -> None:
        compress_forward_musa(*c128_args[:4], c128_args[4], None, head_dim=c128_args[5], compress_ratio=128)

    run_c4()
    run_c128()
    _sync()

    c4_stats = _bench_device_ms(run_c4)
    c4_rows = int(c4_args[4].compress_plan.shape[0])
    c4_write_rows = int(c4_args[4].write_plan.shape[0])
    c4_logical_bytes = (
        _compress_zero_logical_bytes(_PREFILL_TOKENS, c4_args[5])
        + _compress_c4_write_logical_bytes(c4_write_rows, c4_args[5])
        + _compress_c4_reduce_logical_bytes(c4_rows, c4_args[5])
    )
    _record_result(
        "compress_ratio4_prefill_b1_8192",
        c4_stats,
        budget_ms=6.0,
        metadata=_shape_metadata(
            shape={"tokens": _PREFILL_TOKENS, "head_dim": c4_args[5], "compress_ratio": 4},
            dispatch_branch="compress_ratio4_prefill_auto",
            logical_bytes=c4_logical_bytes,
            trace_kernel="dsv4_c4_prefill_page_reduce/write",
            stats_obj=c4_stats,
        ),
    )

    c128_stats = _bench_device_ms(run_c128)
    c128_rows = int(c128_args[4].compress_plan.shape[0])
    c128_write_rows = int(c128_args[4].write_plan.shape[0])
    c128_logical_bytes = (
        _compress_zero_logical_bytes(_PREFILL_TOKENS, c128_args[5])
        + _compress_c128_write_logical_bytes(c128_write_rows, c128_args[5])
        + _compress_c128_reduce_logical_bytes(c128_rows, c128_args[5])
    )
    _record_result(
        "compress_ratio128_prefill_b1_8192",
        c128_stats,
        budget_ms=10.0,
        metadata=_shape_metadata(
            shape={"tokens": _PREFILL_TOKENS, "head_dim": c128_args[5], "compress_ratio": 128},
            dispatch_branch="compress_ratio128_prefill_auto",
            logical_bytes=c128_logical_bytes,
            trace_kernel="dsv4_c128_prefill_parallel_reduce/write",
            stats_obj=c128_stats,
        ),
    )


def test_operator_bench_flashmla_cache_pack_acceptance() -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()
    torch.manual_seed(2026052905)
    page_size = 256
    num_tokens = _PREFILL_TOKENS
    bytes_per_token = 584
    page_bytes = ((page_size * bytes_per_token + 575) // 576) * 576
    num_pages = (num_tokens + page_size - 1) // page_size + 1
    inp = (torch.randn((num_tokens, 512), device=device) * 0.2).to(torch.bfloat16)
    cache = torch.empty((num_pages, page_bytes), device=device, dtype=torch.uint8)
    indices = torch.arange(num_tokens, device=device, dtype=torch.int32)

    def run() -> None:
        fused_store_cache_musa(inp, cache, indices, page_size=page_size, type="flashmla")

    run()
    _sync()
    cache_stats = _bench_device_ms(run)
    _record_result("flashmla_cache_pack_swa_b1_8192", cache_stats, budget_ms=5.0)


def _make_flashmla_cache_inputs(
    device: torch.device,
    *,
    page_size: int,
    num_tokens: int = _PREFILL_TOKENS,
):
    bytes_per_token = 584
    page_bytes = ((page_size * bytes_per_token + 575) // 576) * 576
    num_pages = (num_tokens + page_size - 1) // page_size + 1
    inp = (torch.randn((num_tokens, 512), device=device) * 0.2).to(torch.bfloat16)
    cache = torch.empty((num_pages, page_bytes), device=device, dtype=torch.uint8)
    indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
    return inp, cache, indices


def test_operator_bench_flashmla_cache_pack_group_acceptance() -> None:
    device = _require_operator_benchmark_env()
    _configure_production_like_env()
    torch.manual_seed(2026052908)
    call_count = _group_count("SGLANG_DSV4_OPERATOR_BENCH_CACHE_CALLS", _FLASHMLA_CACHE_CALLS)
    swa_inp, swa_cache, swa_indices = _make_flashmla_cache_inputs(device, page_size=256)
    c128_inp, c128_cache, c128_indices = _make_flashmla_cache_inputs(device, page_size=2)

    def run_swa() -> None:
        fused_store_cache_musa(swa_inp, swa_cache, swa_indices, page_size=256, type="flashmla")

    def run_c128() -> None:
        fused_store_cache_musa(c128_inp, c128_cache, c128_indices, page_size=2, type="flashmla")

    def run_swa_group() -> None:
        for _ in range(call_count):
            run_swa()

    def run_c128_group() -> None:
        for _ in range(call_count):
            run_c128()

    run_swa()
    run_c128()
    _sync()
    swa_stats = _bench_device_ms(run_swa_group, warmup=1, iters=5)
    _record_result(f"flashmla_cache_pack_swa_group_b1_8192_x{call_count}", swa_stats, budget_ms=20.0)

    c128_stats = _bench_device_ms(run_c128_group, warmup=1, iters=5)
    _record_result(f"flashmla_cache_pack_c128_group_b1_8192_x{call_count}", c128_stats, budget_ms=20.0)
