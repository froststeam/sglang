"""Benchmark standalone MUSA TopK renorm against the existing implementation.

Softmax is intentionally outside the timed region. Set
``SGLANG_MUSA_LOGITS_FILE`` to a captured ``{"logits": Tensor}`` file; without
it the benchmark generates a Qwen-like, top-heavy logit distribution.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, dataclass

import torch

import sgl_kernel


CASES = tuple(
    (batch, 151936, top_k)
    for batch in (1, 8, 16)
    for top_k in (20, 40, 256)
)


@dataclass(frozen=True)
class BenchStats:
    median_ms: float
    min_ms: float
    p95_ms: float
    mean_ms: float
    samples: int


def _bench(fn, warmup: int, iters: int) -> BenchStats:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()

    values = []
    for _ in range(iters):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.musa.synchronize()
        values.append(float(start.elapsed_time(end)))

    values.sort()
    p95_index = min(len(values) - 1, int(round((len(values) - 1) * 0.95)))
    return BenchStats(
        median_ms=values[len(values) // 2],
        min_ms=values[0],
        p95_ms=values[p95_index],
        mean_ms=statistics.fmean(values),
        samples=len(values),
    )


def _generate_model_like_logits(rows: int, vocab: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    ranks = torch.arange(1, vocab + 1, dtype=torch.float32)
    base = 22.5 - 2.4 * ranks.log()
    generated = []
    for _ in range(rows):
        values = base + 0.15 * torch.randn(vocab, generator=generator)
        values = values.to(torch.bfloat16).float()
        generated.append(values[torch.randperm(vocab, generator=generator)])
    return torch.stack(generated)


def _load_logits(max_batch: int, vocab: int, seed: int):
    logits_file = os.getenv("SGLANG_MUSA_LOGITS_FILE")
    if logits_file:
        payload = torch.load(logits_file, map_location="cpu", weights_only=True)
        logits = payload["logits"].float().contiguous()
        distribution = logits_file
    else:
        logits = _generate_model_like_logits(max_batch, vocab, seed)
        distribution = "model_like"
    if logits.shape[1] != vocab:
        raise ValueError(f"expected vocab={vocab}, got {logits.shape[1]}")
    repeats = (max_batch + logits.shape[0] - 1) // logits.shape[0]
    return logits.repeat(repeats, 1)[:max_batch], distribution


def main() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    warmup = int(os.getenv("SGLANG_MUSA_PERF_WARMUP", "20"))
    iters = int(os.getenv("SGLANG_MUSA_PERF_ITERS", "100"))
    seed = int(os.getenv("SGLANG_MUSA_PERF_SEED", "2026"))
    max_batch = max(case[0] for case in CASES)
    host_logits, distribution = _load_logits(max_batch, 151936, seed)

    for batch, vocab, top_k in CASES:
        logits = host_logits[:batch].to("musa")
        probs = torch.softmax(logits, dim=-1)
        top_ks = torch.full((batch,), top_k, device="musa", dtype=torch.int32)

        baseline = _bench(
            lambda: sgl_kernel.top_k_renorm_prob(probs, top_ks), warmup, iters
        )
        optimized = _bench(
            lambda: torch.ops.sgl_kernel.musa_top_k_renorm_probs.default(
                probs, top_ks
            ),
            warmup,
            iters,
        )
        result = {
            "distribution": distribution,
            "batch": batch,
            "vocab": vocab,
            "top_k": top_k,
            "baseline": asdict(baseline),
            "optimized": asdict(optimized),
            "speedup": baseline.median_ms / optimized.median_ms,
        }
        print("MUSA_TOP_K_RENORM_BENCH " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
