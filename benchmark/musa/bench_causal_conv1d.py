"""Cold-cache MUSA causal conv1d benchmark using MATE bench_kineto."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Union

import torch
from mate.testing.utils import bench_kineto

from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.causal_conv1d import (
    _ENABLE_WIDTH4_PREFILL_SPLIT,
)
from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.causal_conv1d import (
    causal_conv1d_fwd as tilelang_causal_conv1d_fwd,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn as triton_causal_conv1d_fwd,
)


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    seq_len: int
    dim: int


DEFAULT_CASES = (
    Case("decode", 1, 1, 4096),
    Case("decode", 16, 1, 4096),
    Case("decode", 128, 1, 4096),
    Case("decode", 512, 1, 4096),
    Case("decode", 1024, 1, 4096),
    Case("prefill", 1, 8, 4096),
    Case("prefill", 1, 128, 4096),
    Case("prefill", 1, 1024, 4096),
    Case("prefill", 1, 8, 8192),
    Case("prefill", 1, 128, 8192),
    Case("prefill", 1, 1024, 8192),
)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _tilelang_kernel_name(
    batch: int, seq_len: int, dim: int, width: int
) -> Union[str, tuple[str, ...]]:
    if _ENABLE_WIDTH4_PREFILL_SPLIT and width == 4 and seq_len >= 128 and batch > 1:
        return (
            "sglang_musa_causal_conv1d_prefill_width4_kernel",
            "sglang_musa_causal_conv1d_prefill_width4_body_kernel",
        )
    query_len = batch + 1
    if width == 4 and seq_len == 1 and query_len > 256 and dim >= 4096:
        return "sglang_musa_causal_conv1d_decode_width4_batched"
    if width == 4 and seq_len == 1 and query_len == 2 and dim >= 4096:
        return "sglang_musa_causal_conv1d_fwd_width4_vec"
    return "sglang_musa_causal_conv1d_fwd"


def _tilelang_block_m(batch: int, seq_len: int, width: int) -> int:
    if _ENABLE_WIDTH4_PREFILL_SPLIT and width == 4 and seq_len >= 128 and batch > 1:
        return 28
    if width == 4 and seq_len >= 128:
        if seq_len <= 512:
            return 4
        if seq_len < 4096:
            return 28 if batch >= 4 else 12
        return 28
    return 8


def _make_inputs(case: Case, dtype: torch.dtype, width: int):
    device = "musa"
    seq_lens = [case.seq_len] * case.batch
    total_tokens = case.batch * case.seq_len
    query_start_loc = torch.arange(
        0,
        total_tokens + 1,
        case.seq_len,
        dtype=torch.int32,
        device=device,
    )
    x = (
        torch.randn(
            total_tokens,
            case.dim,
            dtype=dtype,
            device=device,
        )
        .contiguous()
        .t()
    )
    weight = torch.randn((case.dim, width), dtype=dtype, device=device)
    bias = torch.randn((case.dim,), dtype=dtype, device=device)
    conv_states = torch.randn(
        (case.batch, case.dim, width - 1),
        dtype=dtype,
        device=device,
    )
    cache_indices = torch.arange(case.batch, dtype=torch.int32, device=device)
    has_initial_state = torch.ones(case.batch, dtype=torch.bool, device=device)
    return (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens,
        cache_indices,
        has_initial_state,
    )


def _lower_bound_bytes(case: Case, dtype: torch.dtype, width: int) -> int:
    elem_size = torch.empty((), dtype=dtype).element_size()
    state_len = width - 1
    tokens = case.batch * case.seq_len
    token_elems = tokens * case.dim
    block_m = _tilelang_block_m(case.batch, case.seq_len, width)
    chunks_per_seq = (case.seq_len + block_m - 1) // block_m
    chunk_elems = case.batch * chunks_per_seq * case.dim

    # The forward kernels process up to 8 tokens per feature block. For prefill,
    # weights/bias and prior-token window loads are amortized across that chunk;
    # treating them as per-token traffic overestimates bandwidth by several x.
    x_bytes = token_elems * elem_size
    out_bytes = token_elems * elem_size
    weight_bytes = chunk_elems * width * elem_size
    bias_bytes = chunk_elems * elem_size
    prior_x_bytes = (
        case.batch * max(chunks_per_seq - 1, 0) * case.dim * state_len * elem_size
    )
    state_bytes = case.batch * case.dim * state_len * elem_size * 2
    index_bytes = case.batch * (4 + 1)
    return (
        x_bytes
        + out_bytes
        + weight_bytes
        + bias_bytes
        + prior_x_bytes
        + state_bytes
        + index_bytes
    )


def _bench_case(
    case: Case,
    dtype: torch.dtype,
    width: int,
    num_tests: int,
) -> str:
    (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens,
        cache_indices,
        has_initial_state,
    ) = _make_inputs(case, dtype, width)

    ref_state = conv_states.clone()
    out_tilelang = tilelang_causal_conv1d_fwd(
        x,
        weight,
        bias,
        ref_state,
        query_start_loc,
        seq_lens,
        cache_indices,
        has_initial_state,
        "silu",
    )
    out_triton = triton_causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states.clone(),
        query_start_loc,
        seq_lens,
        cache_indices,
        has_initial_state,
        "silu",
    )
    torch.musa.synchronize()
    max_diff = (out_tilelang - out_triton).abs().max().item()

    tilelang_state = conv_states.clone()
    triton_state = conv_states.clone()

    def run_tilelang():
        tilelang_causal_conv1d_fwd(
            x,
            weight,
            bias,
            tilelang_state,
            query_start_loc,
            seq_lens,
            cache_indices,
            has_initial_state,
            "silu",
        )

    def run_triton():
        triton_causal_conv1d_fwd(
            x,
            weight,
            bias,
            triton_state,
            query_start_loc,
            seq_lens,
            cache_indices,
            has_initial_state,
            "silu",
        )

    tilelang_kernel = _tilelang_kernel_name(
        case.batch,
        case.seq_len,
        case.dim,
        width,
    )
    tilelang_t = bench_kineto(
        run_tilelang,
        kernel_names=tilelang_kernel,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=isinstance(tilelang_kernel, tuple),
    )
    triton_s = bench_kineto(
        run_triton,
        kernel_names="_causal_conv1d_fwd_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    tilelang_s = sum(tilelang_t) if isinstance(tilelang_t, tuple) else tilelang_t
    tilelang_ms = float(tilelang_s) * 1e3
    triton_ms = float(triton_s) * 1e3
    bytes_lb = _lower_bound_bytes(case, dtype, width)
    tilelang_bw = bytes_lb / tilelang_s / 1e9 if tilelang_s > 0 else 0.0
    triton_bw = bytes_lb / triton_s / 1e9 if triton_s > 0 else 0.0
    return (
        f"{case.name:7s} B={case.batch:<4d} L={case.seq_len:<4d} "
        f"D={case.dim:<5d} diff={max_diff:<9g} "
        f"TL={tilelang_ms:8.4f} ms TR={triton_ms:8.4f} ms "
        f"speedup={triton_ms / tilelang_ms:6.2f}x "
        f"BW_lb TL/TR={tilelang_bw:7.1f}/{triton_bw:7.1f} GB/s "
        f"kernel={tilelang_kernel}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--num-tests", type=int, default=5)
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")

    dtype = _dtype_from_name(args.dtype)
    print(
        "MATE bench_kineto cold-cache causal_conv1d MUSA path: "
        f"dtype={args.dtype}, width={args.width}, num_tests={args.num_tests}"
    )
    print("Comparator: dispatched MUSA path vs original Triton causal_conv1d_fn")
    rows = []
    for case in DEFAULT_CASES:
        rows.append(_bench_case(case, dtype, args.width, args.num_tests))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
