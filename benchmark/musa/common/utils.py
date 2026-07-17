from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
from mate.testing.utils import bench_kineto

DEFAULT_TOKEN_VALUES = (
    "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072"
)


def parse_csv(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in parse_csv(text)]


def dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def sync() -> None:
    torch.musa.synchronize()


def elem_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def bench_mate(
    fn: Callable[[], object],
    kernel_names: str | tuple[str, ...],
    num_tests: int,
    flush_l2: bool,
    with_multiple_kernels: bool = False,
) -> float:
    seconds = bench_kineto(
        fn,
        kernel_names=kernel_names,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
        with_multiple_kernels=with_multiple_kernels,
    )
    if isinstance(seconds, tuple):
        return float(sum(seconds))
    return float(seconds)


def bench_mate_times(
    fn: Callable[[], object],
    kernel_names: str | tuple[str, ...],
    num_tests: int,
    flush_l2: bool,
) -> tuple[float, ...]:
    seconds = bench_kineto(
        fn,
        kernel_names=kernel_names,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
        with_multiple_kernels=True,
    )
    if isinstance(seconds, tuple):
        return tuple(float(value) for value in seconds)
    return (float(seconds),)


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs().flatten()
    if diff.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0, "max_rel": 0.0}
    sample = (
        diff[:: max(1, diff.numel() // 1_000_000)]
        if diff.numel() > 10_000_000
        else diff
    )
    rel = diff / expected.float().abs().flatten().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(sample, 0.99).item()),
        "max_rel": float(rel.max().item()),
    }


def is_close(
    actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float
) -> bool:
    limit = atol + rtol * float(expected.float().abs().max().item())
    return error_stats(actual, expected)["max_abs"] <= limit


def print_rows(rows: list[dict[str, object]], columns: Iterable[str]) -> None:
    columns = tuple(columns)
    if not rows:
        print("no rows")
        return
    text_rows = [
        {column: str(row.get(column, "")) for column in columns} for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in text_rows))
        for column in columns
    }
    print(" ".join(column.rjust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for row in text_rows:
        print(" ".join(row[column].rjust(widths[column]) for column in columns))


def write_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    import csv

    if not rows:
        return
    columns = list(rows[0].keys())
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


NORM_LAYOUTS = ("compact", "strided")


def get_flashinfer_norm():
    try:
        import flashinfer.norm as flashinfer_norm
    except Exception:
        return None
    return flashinfer_norm


def norm_provider_module(provider: str):
    if provider == "csrc":
        from sglang.srt.hardware_backend.musa.jit_kernel.csrc import norm as csrc_norm

        return csrc_norm
    if provider == "flashinfer":
        mod = get_flashinfer_norm()
        if mod is None:
            raise RuntimeError("flashinfer.norm is not available")
        return mod
    raise ValueError(provider)


def strided_hidden(hidden: int, stride_padding: int) -> int:
    padding = stride_padding if stride_padding > 0 else max(1, hidden // 8)
    return hidden + padding


def make_strided_2d(m: int, hidden: int, stride0: int, dtype: torch.dtype):
    return torch.randn((m, stride0), device="musa", dtype=dtype)[:, :hidden]


def make_norm_inputs(
    m: int,
    hidden: int,
    dtype: torch.dtype,
    layout: str,
    stride_padding: int,
):
    if layout == "compact":
        x = torch.randn((m, hidden), device="musa", dtype=dtype)
        residual = torch.randn_like(x)
        stride0 = hidden
    elif layout == "strided":
        stride0 = strided_hidden(hidden, stride_padding)
        x = make_strided_2d(m, hidden, stride0, dtype)
        residual = make_strided_2d(m, hidden, stride0, dtype)
    else:
        raise ValueError(f"unsupported layout: {layout}")
    weight = torch.randn((hidden,), device="musa", dtype=dtype)
    out = torch.empty_like(x)
    return x, residual, weight, out, stride0


def rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> torch.Tensor:
    y = x.float()
    scale = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + eps)
    w = weight.float() + 1.0 if gemma else weight.float()
    return (y * scale * w).to(dtype=x.dtype)


def run_rmsnorm(
    op_name: str,
    provider: str,
    m: int,
    hidden: int,
    dtype: torch.dtype,
    layout: str,
    stride_padding: int,
    num_tests: int,
    flush_l2: bool,
    correctness: bool,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    x, _, weight, out, stride0 = make_norm_inputs(
        m, hidden, dtype, layout, stride_padding
    )
    op = getattr(norm_provider_module(provider), op_name)
    kernel_names = (
        "rmsnorm_h192_vec8row_kernel",
        "rmsnorm_h64_warp_kernel",
        "rmsnorm_h128_warp_kernel",
        "rmsnorm_small_h_halfwarp_kernel",
        "rmsnorm_small_h_one_vec_register_kernel",
        "rmsnorm_vec8_kernel",
        "rmsnorm_scalar_kernel",
    )
    kernel_label = "+".join(kernel_names)

    def fn():
        op(x, weight, eps=1e-6, out=out)

    fn()
    sync()
    correct = True
    max_abs = 0.0
    if correctness:
        ref = rmsnorm_ref(x, weight, 1e-6, gemma=op_name.startswith("gemma"))
        stats = error_stats(out, ref)
        max_abs = stats["max_abs"]
        limit = atol + rtol * float(ref.float().abs().max().item())
        correct = max_abs <= limit

    seconds = bench_mate(
        fn, kernel_names, num_tests, flush_l2, with_multiple_kernels=True
    )
    bytes_ = 2 * m * hidden * elem_size(dtype) + hidden * elem_size(dtype)
    return {
        "op": op_name,
        "provider": provider,
        "m": m,
        "hidden": hidden,
        "stride0": stride0,
        "layout": layout,
        "dtype": dtype_name(dtype),
        "kernel": kernel_label,
        "correct": correct,
        "max_abs": f"{max_abs:.3g}",
        "latency_us": f"{seconds * 1e6:.3f}",
        "logical_GBps": f"{bytes_ / seconds / 1e9:.1f}" if seconds > 0 else "0.0",
    }


def run_fused_add_rmsnorm(
    op_name: str,
    provider: str,
    m: int,
    hidden: int,
    dtype: torch.dtype,
    layout: str,
    stride_padding: int,
    num_tests: int,
    flush_l2: bool,
    correctness: bool,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    x, residual, weight, _, stride0 = make_norm_inputs(
        m, hidden, dtype, layout, stride_padding
    )
    x_before = x.clone()
    residual_before = residual.clone()
    op = getattr(norm_provider_module(provider), op_name)
    kernel_names = (
        "fused_add_rmsnorm_h64_vec8row_kernel",
        "fused_add_rmsnorm_h128_vec8row_kernel",
        "fused_add_rmsnorm_h192_vec8row_kernel",
        "fused_add_rmsnorm_h64_w16_kernel",
        "fused_add_rmsnorm_h64_warp_kernel",
        "fused_add_rmsnorm_vec8_kernel",
        "fused_add_rmsnorm_vec8_tcache_kernel",
        "fused_add_rmsnorm_scalar_kernel",
    )
    kernel_label = "+".join(kernel_names)

    def fn():
        op(x, residual, weight, 1e-6)

    fn()
    sync()
    correct = True
    x_max_abs = 0.0
    residual_max_abs = 0.0
    if correctness:
        residual_ref = x_before + residual_before
        x_ref = rmsnorm_ref(
            residual_ref, weight, 1e-6, gemma=op_name.startswith("gemma")
        )
        x_stats = error_stats(x, x_ref)
        residual_stats = error_stats(residual, residual_ref)
        x_max_abs = x_stats["max_abs"]
        residual_max_abs = residual_stats["max_abs"]
        x_limit = atol + rtol * float(x_ref.float().abs().max().item())
        r_limit = atol + rtol * float(residual_ref.float().abs().max().item())
        correct = x_max_abs <= x_limit and residual_max_abs <= r_limit

    x, residual, weight, _, stride0 = make_norm_inputs(
        m, hidden, dtype, layout, stride_padding
    )

    def bench_fn():
        op(x, residual, weight, 1e-6)

    seconds = bench_mate(
        bench_fn, kernel_names, num_tests, flush_l2, with_multiple_kernels=True
    )
    bytes_ = 4 * m * hidden * elem_size(dtype) + hidden * elem_size(dtype)
    return {
        "op": op_name,
        "provider": provider,
        "m": m,
        "hidden": hidden,
        "stride0": stride0,
        "layout": layout,
        "dtype": dtype_name(dtype),
        "kernel": kernel_label,
        "correct": correct,
        "x_max_abs": f"{x_max_abs:.3g}",
        "residual_max_abs": f"{residual_max_abs:.3g}",
        "latency_us": f"{seconds * 1e6:.3f}",
        "logical_GBps": f"{bytes_ / seconds / 1e9:.1f}" if seconds > 0 else "0.0",
    }
