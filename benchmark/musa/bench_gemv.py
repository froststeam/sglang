"""Benchmark MUSA JIT GEMV and MoE GEMV kernels.

Examples:
  PYTHONPATH=python python benchmark/musa/bench_gemv.py
  PYTHONPATH=python python benchmark/musa/bench_gemv.py --ops linear --dtypes bf16 --timing both
  PYTHONPATH=python python benchmark/musa/bench_gemv.py --trace-dir /tmp/gemv_traces
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemv import (
    musa_gemv as jit_linear_gemv,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemv import (
    musa_moe_gemv as jit_moe_gemv,
)

KERNEL_NAME = "gemv"


def parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def parse_csv_strs(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def require_musa() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("torch.musa is not available")


def sync() -> None:
    torch.musa.synchronize()


def event_us(fn: Callable[[], None], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    sync()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    sync()
    return float(start.elapsed_time(end)) * 1000.0 / repeat


def kineto_us(
    fn: Callable[[], None],
    reps: int,
    *,
    flush_l2: bool,
    trace_path: Path | None,
) -> float:
    seconds = bench_kineto(
        fn,
        kernel_names=KERNEL_NAME,
        num_tests=reps,
        suppress_kineto_output=True,
        trace_path=str(trace_path) if trace_path is not None else None,
        flush_l2=flush_l2,
    )
    return seconds * 1e6


def make_linear_case(
    *,
    dtype: str,
    tokens: int,
    n: int,
    k: int,
    seed: int,
) -> tuple[Callable[[], None], Callable[[], torch.Tensor], torch.Tensor]:
    torch.manual_seed(seed)
    A = torch.randn(tokens, k, device="musa", dtype=torch.bfloat16)
    if dtype == "fp8":
        B = torch.randn(n, k, device="musa", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        B_scale = torch.ones(
            (triton_ceil_div(n, 128), triton_ceil_div(k, 128)),
            device="musa",
            dtype=torch.float32,
        )
    elif dtype == "bf16":
        B = torch.randn(n, k, device="musa", dtype=torch.bfloat16)
        B_scale = None
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    out_ref = [torch.empty(tokens, n, device="musa", dtype=torch.bfloat16)]

    def fn() -> None:
        out_ref[0] = jit_linear_gemv(A, B, B_scale)

    return fn, lambda: out_ref[0], A.float() @ B.to(torch.bfloat16).float().t()


def make_moe_case(
    *,
    dtype: str,
    tokens: int,
    n: int,
    k: int,
    experts: int,
    topk: int,
    seed: int,
) -> tuple[Callable[[], None], Callable[[], torch.Tensor], torch.Tensor]:
    if n % 2 != 0:
        raise ValueError("MoE swiglu benchmark requires an even n")

    torch.manual_seed(seed)
    A = torch.randn(tokens, k, device="musa", dtype=torch.bfloat16)
    topk_ids = torch.randint(
        0, experts, (tokens, topk), device="musa", dtype=torch.int32
    )
    topk_weights = torch.rand(tokens, topk, device="musa", dtype=torch.float32)
    C = torch.empty(tokens, topk, n // 2, device="musa", dtype=torch.bfloat16)

    if dtype == "fp8":
        B = torch.randn(experts, n, k, device="musa", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        B_scale = torch.ones(
            (experts, triton_ceil_div(n, 128), triton_ceil_div(k, 128)),
            device="musa",
            dtype=torch.float32,
        )
    elif dtype == "bf16":
        B = torch.randn(experts, n, k, device="musa", dtype=torch.bfloat16)
        B_scale = None
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    def reference() -> torch.Tensor:
        bref = B.to(torch.bfloat16).float()
        rows = []
        for t in range(tokens):
            per_token = []
            for i in range(topk):
                expert = int(topk_ids[t, i].item())
                x = A[t].float() @ bref[expert].t()
                gate = x[: n // 2]
                up = x[n // 2 :]
                per_token.append(gate * torch.sigmoid(gate) * up)
            rows.append(torch.stack(per_token))
        return torch.stack(rows)

    def fn() -> None:
        jit_moe_gemv(
            A,
            B,
            C,
            None,
            B_scale,
            topk_weights,
            topk_ids,
            False,
            topk,
            False,
            True,
        )

    return fn, lambda: C, reference()


def triton_ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def max_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def run_one(
    *,
    op: str,
    dtype: str,
    tokens: int,
    n: int,
    k: int,
    experts: int,
    topk: int,
    args: argparse.Namespace,
) -> dict[str, str]:
    seed = args.seed + tokens * 17 + n * 3 + k
    if op == "linear":
        fn, get_output, ref = make_linear_case(
            dtype=dtype, tokens=tokens, n=n, k=k, seed=seed
        )
    elif op == "moe":
        fn, get_output, ref = make_moe_case(
            dtype=dtype,
            tokens=tokens,
            n=n,
            k=k,
            experts=experts,
            topk=topk,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported op: {op}")

    fn()
    sync()
    diff = 0.0
    if not args.skip_correctness:
        actual = get_output().float()
        expected = ref.float()
        diff = max_diff(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1.0)

    event = None
    kineto = None
    if args.timing in ("event", "both"):
        event = event_us(fn, args.warmup, args.repeat)
    if args.timing in ("kineto", "both"):
        trace_path = None
        if args.trace_dir:
            trace_dir = Path(args.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{op}_{dtype}_t{tokens}_n{n}_k{k}.trace.json"
        kineto = kineto_us(
            fn,
            args.reps,
            flush_l2=args.flush_l2,
            trace_path=trace_path,
        )

    return {
        "op": op,
        "dtype": dtype,
        "tokens": str(tokens),
        "n": str(n),
        "k": str(k),
        "experts": str(experts if op == "moe" else 1),
        "topk": str(topk if op == "moe" else 1),
        "event_us": "" if event is None else f"{event:.3f}",
        "kineto_us": "" if kineto is None else f"{kineto:.3f}",
        "max_diff": f"{diff:.6g}",
    }


def print_rows(rows: list[dict[str, str]]) -> None:
    columns = [
        "op",
        "dtype",
        "tokens",
        "n",
        "k",
        "experts",
        "topk",
        "event_us",
        "kineto_us",
        "max_diff",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(row[c] for c in columns), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MUSA JIT GEMV and MoE GEMV kernels."
    )
    parser.add_argument("--ops", default="linear,moe", help="linear,moe or both")
    parser.add_argument("--dtypes", default="fp8,bf16", help="fp8,bf16 or both")
    parser.add_argument("--tokens", default="1,2,3,4,8,16,32")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=257)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--timing",
        choices=("kineto", "event", "both"),
        default="kineto",
    )
    parser.add_argument("--reps", type=int, default=20, help="Kineto repetitions")
    parser.add_argument("--warmup", type=int, default=10, help="Event warmup")
    parser.add_argument("--repeat", type=int, default=80, help="Event repetitions")
    parser.add_argument("--flush-l2", action="store_true")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    require_musa()
    torch.musa.set_device(0)

    ops = parse_csv_strs(args.ops)
    if ops == ["both"]:
        ops = ["linear", "moe"]
    dtypes = parse_csv_strs(args.dtypes)
    if dtypes == ["both"]:
        dtypes = ["fp8", "bf16"]
    tokens = parse_csv_ints(args.tokens)

    print("device", torch.musa.current_device(), torch.musa.get_device_name())
    print(
        f"timing={args.timing} kineto_reps={args.reps} "
        f"flush_l2={args.flush_l2} kernel={KERNEL_NAME}"
    )

    rows = []
    for op in ops:
        for dtype in dtypes:
            for token in tokens:
                row = run_one(
                    op=op,
                    dtype=dtype,
                    tokens=token,
                    n=args.n,
                    k=args.k,
                    experts=args.experts,
                    topk=args.topk,
                    args=args,
                )
                rows.append(row)
    print_rows(rows)


if __name__ == "__main__":
    main()
