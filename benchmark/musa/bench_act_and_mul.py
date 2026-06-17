#!/usr/bin/env python3
"""Benchmark and validate MUSA act_and_mul accuracy.

Run on a MUSA host from the SGLang repo root:
  PYTHONPATH=python python benchmark/musa/bench_act_and_mul.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from mate.testing.utils import bench_kineto

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_musa_moe_kernels():
    root = REPO_ROOT / "python"

    def pkg(name: str, relpath: str):
        mod = sys.modules.setdefault(name, types.ModuleType(name))
        mod.__path__ = [str(root / relpath)]
        return mod

    def load_module(name: str, relpath: str):
        path = root / relpath
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    pkg("sglang", "sglang")
    pkg("sglang.jit_kernel", "sglang/jit_kernel")
    pkg("sglang.srt", "sglang/srt")
    pkg("sglang.srt.hardware_backend", "sglang/srt/hardware_backend")
    pkg("sglang.srt.hardware_backend.musa", "sglang/srt/hardware_backend/musa")
    pkg(
        "sglang.srt.hardware_backend.musa.jit_kernel",
        "sglang/srt/hardware_backend/musa/jit_kernel",
    )
    pkg(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc",
        "sglang/srt/hardware_backend/musa/jit_kernel/csrc",
    )

    fake_jit_utils = types.ModuleType("sglang.jit_kernel.utils")
    fake_jit_utils.cache_once = lambda fn: fn
    sys.modules["sglang.jit_kernel.utils"] = fake_jit_utils

    fake_custom_op = types.ModuleType("sglang.srt.utils.custom_op")

    def register_custom_op(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def deco(fn):
            return fn

        return deco

    fake_custom_op.register_custom_op = register_custom_op
    sys.modules["sglang.srt.utils.custom_op"] = fake_custom_op

    load_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit",
        "sglang/srt/hardware_backend/musa/jit_kernel/csrc/jit.py",
    )
    return load_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc.moe",
        "sglang/srt/hardware_backend/musa/jit_kernel/csrc/moe.py",
    )


@dataclass
class CaseResult:
    activation: str
    dtype: str
    rows: int
    hidden: int
    filtered: bool
    max_abs: float
    mean_abs: float
    p99_abs: float
    max_rel: float
    passed: bool
    kernel_ms: float
    torch_ms: float
    speedup: float
    kernel_gbs: float
    torch_gbs: float


def device_name() -> str:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("torch.musa is not available on this host")
    return "musa"


def sync() -> None:
    torch.musa.synchronize()


def ref_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "silu":
        return lambda x: F.silu(x[..., : x.shape[-1] // 2]) * x[..., x.shape[-1] // 2 :]
    if activation == "gelu":
        return (
            lambda x: F.gelu(x[..., : x.shape[-1] // 2], approximate="none")
            * x[..., x.shape[-1] // 2 :]
        )
    if activation == "gelu_tanh":
        return (
            lambda x: F.gelu(x[..., : x.shape[-1] // 2], approximate="tanh")
            * x[..., x.shape[-1] // 2 :]
        )
    raise ValueError(f"unsupported activation {activation}")


def max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    denom = expected_f.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item() if diff.numel() else 0.0
    return max_abs, max_rel


def error_stats(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs().flatten()
    if diff.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    denom = expected_f.abs().flatten().clamp_min(1e-6)
    rel = diff / denom
    if diff.numel() > 10_000_000:
        # torch.quantile can reject very large tensors on some backends.
        p99 = diff[:: max(1, diff.numel() // 1_000_000)].quantile(0.99).item()
    else:
        p99 = torch.quantile(diff, 0.99).item()
    return (diff.max().item(), diff.mean().item(), p99, rel.max().item())


def time_ms(fn: Callable[[], None], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    sync()

    if hasattr(torch.musa, "Event"):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        sync()
        return float(start.elapsed_time(end)) / repeat

    start_time = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync()
    return (time.perf_counter() - start_time) * 1000.0 / repeat


def expected_kernel_name(
    dtype: torch.dtype, rows: int, hidden: int, filtered: bool
) -> str:
    if dtype != torch.float32 and hidden % 8 == 0:
        if rows > 128:
            return (
                "act_and_mul_flat_vec8_kernel"
                if filtered
                else "act_and_mul_flat_vec8_no_filter_kernel"
            )
        return "act_and_mul_vec8_kernel"
    if rows <= 128 and hidden > 512:
        return "act_and_mul_split_kernel"
    return "act_and_mul_row_kernel"


def mate_kineto_ms(fn: Callable[[], None], kernel_name: str, repeat: int) -> float:
    seconds = bench_kineto(
        fn,
        kernel_names=kernel_name,
        num_tests=repeat,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if seconds <= 0:
        raise RuntimeError(
            f"MATE bench_kineto did not capture kernel {kernel_name!r}. "
            "The profiler may be reporting only musaLaunchKernel; rerun with "
            "a MUSA/Torch profiler setup that exposes device kernel symbols, "
            "or use --event only for a non-kineto smoke benchmark."
        )
    return seconds * 1000.0


def print_markdown(results: list[CaseResult]) -> None:
    print(
        "| act | dtype | rows | hidden | filtered | max_abs | mean_abs | p99_abs | kernel_us | torch_us | speedup | kernel_GB/s | pass |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r.activation} | {r.dtype} | {r.rows} | {r.hidden} | "
            f"{str(r.filtered).lower()} | {r.max_abs:.3g} | "
            f"{r.mean_abs:.3g} | {r.p99_abs:.3g} | "
            f"{r.kernel_ms * 1000.0:.3f} | {r.torch_ms * 1000.0:.3f} | "
            f"{r.speedup:.2f} | {r.kernel_gbs:.2f} | {r.passed} |"
        )


def run_case(
    *,
    activation: str,
    dtype: torch.dtype,
    rows: int,
    hidden: int,
    filtered: bool,
    warmup: int,
    repeat: int,
    use_event: bool,
    kernel_name_override: str | None,
) -> CaseResult:
    device = device_name()
    torch.manual_seed(1234 + rows + hidden)
    x = (torch.randn(rows, hidden * 2, device=device, dtype=torch.float32) * 2).to(
        dtype
    )
    expected = ref_fn(activation)(x)

    if filtered:
        topk_ids = torch.arange(rows, device=device, dtype=torch.int32)
        topk_ids[::7] = -1
        out = torch.full((rows, hidden), 17.0, device=device, dtype=dtype)
        actual = out
        act_and_mul(
            x,
            out,
            topk_ids=topk_ids,
            activation=activation,
            filter_expert=True,
        )
        active = topk_ids != -1
        skipped = ~active
        max_abs, mean_abs, p99_abs, max_rel = error_stats(
            actual[active], expected[active]
        )
        skipped_ok = torch.all(
            actual[skipped] == torch.tensor(17.0, device=device, dtype=dtype)
        ).item()
    else:
        actual = act_and_mul(x, activation=activation)
        max_abs, mean_abs, p99_abs, max_rel = error_stats(actual, expected)
        skipped_ok = True

    if dtype == torch.float32:
        atol = 2e-5
    elif dtype == torch.float16:
        atol = 8e-2
    else:
        atol = 3e-1
    passed = bool(skipped_ok and max_abs <= atol)

    if filtered:
        bench_ids = topk_ids
        bench_out = torch.empty((rows, hidden), device=device, dtype=dtype)

        def kernel_call() -> None:
            act_and_mul(
                x,
                bench_out,
                topk_ids=bench_ids,
                activation=activation,
                filter_expert=True,
            )

        def torch_call() -> None:
            y = ref_fn(activation)(x)
            bench_out[bench_ids != -1].copy_(y[bench_ids != -1])

    else:

        def kernel_call() -> None:
            act_and_mul(x, activation=activation)

        def torch_call() -> None:
            ref_fn(activation)(x)

    kernel_name = kernel_name_override or expected_kernel_name(
        dtype, rows, hidden, filtered
    )
    kernel_ms = (
        time_ms(kernel_call, warmup, repeat)
        if use_event
        else mate_kineto_ms(kernel_call, kernel_name, repeat)
    )
    torch_ms = time_ms(torch_call, warmup, repeat)
    speedup = torch_ms / kernel_ms if kernel_ms > 0 else math.inf
    active_rows = rows
    # Effective memory traffic: read gate+up and write output.
    bytes_per_elem = torch.empty((), dtype=dtype).element_size()
    traffic_bytes = active_rows * hidden * 3 * bytes_per_elem
    kernel_gbs = traffic_bytes / (kernel_ms * 1.0e6) if kernel_ms > 0 else math.inf
    torch_gbs = traffic_bytes / (torch_ms * 1.0e6) if torch_ms > 0 else math.inf

    return CaseResult(
        activation=activation,
        dtype=str(dtype).replace("torch.", ""),
        rows=rows,
        hidden=hidden,
        filtered=filtered,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        max_rel=max_rel,
        passed=passed,
        kernel_ms=kernel_ms,
        torch_ms=torch_ms,
        speedup=speedup,
        kernel_gbs=kernel_gbs,
        torch_gbs=torch_gbs,
    )


def main() -> None:
    global act_and_mul
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--event",
        action="store_true",
        help="Use simple MUSA event timing instead of MATE bench_kineto.",
    )
    parser.add_argument(
        "--kernel-name",
        default=None,
        help="Override the kernel name passed to MATE bench_kineto.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument(
        "--m1-32k",
        action="store_true",
        help="Run rows=1..32768 power-of-two bandwidth cases.",
    )
    parser.add_argument(
        "--row-sweep-1-32k",
        action="store_true",
        help="Run rows=1..32768 power-of-two bandwidth sweep.",
    )
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke set for quick accuracy/performance checks.",
    )
    args = parser.parse_args()
    act_and_mul = load_musa_moe_kernels().act_and_mul
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtypes = tuple(dtype_map[name] for name in args.dtypes)

    cases = []
    if args.quick:
        for dtype in dtypes:
            for activation in ("silu", "gelu", "gelu_tanh"):
                for rows, hidden in ((128, 1024), (1024, 2048), (4096, 3584)):
                    cases.append((activation, dtype, rows, hidden, False))
                cases.append((activation, dtype, 1024, 2048, True))
    elif args.m1_32k:
        rows_list = [1 << i for i in range(16)]
        for dtype in dtypes:
            for activation in ("silu", "gelu", "gelu_tanh"):
                for rows in rows_list:
                    cases.append((activation, dtype, rows, args.hidden, False))
    elif args.row_sweep_1_32k:
        rows_list = [1 << i for i in range(16)]
        for dtype in dtypes:
            for activation in ("silu", "gelu", "gelu_tanh"):
                for rows in rows_list:
                    cases.append((activation, dtype, rows, args.hidden, False))
    else:
        for dtype in dtypes:
            for activation in ("silu", "gelu", "gelu_tanh"):
                for rows, hidden in (
                    (1, 128),
                    (17, 512),
                    (128, 1024),
                    (1024, 2048),
                    (4096, 3584),
                ):
                    cases.append((activation, dtype, rows, hidden, False))
                for rows, hidden in ((128, 1024), (1024, 2048), (4096, 3584)):
                    cases.append((activation, dtype, rows, hidden, True))

    results = [
        run_case(
            activation=activation,
            dtype=dtype,
            rows=rows,
            hidden=hidden,
            filtered=filtered,
            warmup=args.warmup,
            repeat=args.repeat,
            use_event=args.event,
            kernel_name_override=args.kernel_name,
        )
        for activation, dtype, rows, hidden, filtered in cases
    ]

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    elif args.markdown:
        print_markdown(results)
    else:
        header = "act dtype rows hidden filt max_abs mean_abs p99_abs max_rel kernel_ms torch_ms speedup kernel_GB/s torch_GB/s pass"
        print(header)
        for r in results:
            print(
                f"{r.activation:9s} {r.dtype:8s} {r.rows:5d} {r.hidden:6d} "
                f"{str(r.filtered):5s} {r.max_abs:.3e} {r.mean_abs:.3e} "
                f"{r.p99_abs:.3e} {r.max_rel:.3e} "
                f"{r.kernel_ms:9.4f} {r.torch_ms:8.4f} {r.speedup:7.2f} "
                f"{r.kernel_gbs:11.2f} {r.torch_gbs:10.2f} {r.passed}"
            )

    failed = [r for r in results if not r.passed]
    if failed:
        raise SystemExit(f"{len(failed)} cases failed")


if __name__ == "__main__":
    main()
