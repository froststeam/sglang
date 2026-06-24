"""Benchmark TileLang MUSA GDN fused projection against the Triton baseline."""

from __future__ import annotations

import argparse

import torch
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

from sglang.jit_kernel.triton.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous as triton_gdn,
)
from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.fla.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous as tilelang_gdn,
)


def elem_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def bench_one(m: int, num_tests: int) -> None:
    num_heads_qk, num_heads_v, head_qk, head_v = 4, 8, 128, 128
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    total_qkvz = qkv_dim + num_heads_v * head_v
    mixed_qkvz = torch.randn(m, total_qkvz, device="musa", dtype=torch.float16)
    mixed_ba = torch.randn(m, num_heads_v * 2, device="musa", dtype=torch.float16)

    tilelang_out = tilelang_gdn(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )
    triton_out = triton_gdn(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )
    torch.musa.synchronize()
    diffs = [
        float((actual - expected).abs().max().item())
        for actual, expected in zip(tilelang_out, triton_out)
    ]

    def run_tilelang() -> None:
        tilelang_gdn(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)

    def run_triton() -> None:
        triton_gdn(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)

    if m >= 128:
        tilelang_kernel = "qkvzba_contiguous_vec_kernel"
    else:
        tilelang_kernel = "qkvzba_contiguous_row_kernel"
    tilelang_s = bench_kineto(
        run_tilelang,
        tilelang_kernel,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    triton_s = bench_kineto(
        run_triton,
        "fused_qkvzba_split_reshape_cat_contiguous_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    bytes_lb = (
        m * total_qkvz * elem_size(torch.float16)
        + m * qkv_dim * elem_size(torch.float16)
        + m * num_heads_v * head_v * elem_size(torch.float16)
        + m * num_heads_v * 2 * elem_size(torch.float16)
        + m * num_heads_v * 2 * elem_size(torch.float16)
    )
    print(f"M={m} diffs_tilelang_vs_triton={diffs}")
    print(
        f"  TileLang  {tilelang_s * 1e3:8.4f} ms  "
        f"BW_lb={bytes_lb / tilelang_s / 1e9:8.1f} GB/s"
    )
    print(
        f"  Triton    {triton_s * 1e3:8.4f} ms  "
        f"BW_lb={bytes_lb / triton_s / 1e9:8.1f} GB/s"
    )
    speedup = triton_s / tilelang_s if tilelang_s > 0 else 0.0
    print(f"  speedup_vs_triton={speedup:.2f}x kernel={tilelang_kernel}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tests", type=int, default=7)
    parser.add_argument("--m", type=int, nargs="*", default=[4096, 32768])
    args = parser.parse_args()

    for m in args.m:
        bench_one(m, args.num_tests)


if __name__ == "__main__":
    main()
