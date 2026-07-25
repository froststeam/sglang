"""Cold-cache MUSA JIT kernel benchmark using MATE bench_kineto."""

from __future__ import annotations

import argparse
import os

import torch
from mate.testing.utils import bench_kineto

from sglang.jit_kernel.triton.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous as triton_gdn,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc import (
    per_token_group_quant_8bit,
    rotary_embedding,
    topk_sigmoid,
    topk_softmax,
)
from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.causal_conv1d import (
    causal_conv1d_fwd as tilelang_causal_conv1d_fwd,
)
from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.fla.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous as tilelang_gdn,
)
from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.fla.layernorm_gated import (
    rms_norm_gated as tilelang_rms_norm_gated,
)
from sglang.srt.layers.attention.fla.layernorm_gated import (
    rms_norm_gated as triton_rms_norm_gated,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn as triton_causal_conv1d_fwd,
)


def _set_default_musa_arch() -> None:
    if os.environ.get("SGLANG_MUSA_ARCH_LIST") or os.environ.get("MATE_MUSA_ARCH_LIST"):
        return
    try:
        major, minor = torch.musa.get_device_capability()
        os.environ["SGLANG_MUSA_ARCH_LIST"] = f"mp_{int(major)}{int(minor)}"
    except Exception:
        os.environ["SGLANG_MUSA_ARCH_LIST"] = "mp_31"


def _elem_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _print_row(name: str, seconds: float, bytes_lb: int = 0) -> None:
    if bytes_lb:
        bw = bytes_lb / seconds / 1e9 if seconds > 0 else 0.0
        print(f"{name:44s} {seconds * 1e3:8.4f} ms  BW_lb={bw:8.1f} GB/s")
    else:
        print(f"{name:44s} {seconds * 1e3:8.4f} ms")


def _print_compare(
    name: str,
    candidate_s: float,
    baseline_s: float,
    candidate_label: str,
    baseline_label: str,
    bytes_lb: int = 0,
    diff: float | None = None,
) -> None:
    diff_text = "" if diff is None else f" diff={diff:g}"
    speedup = baseline_s / candidate_s if candidate_s > 0 else 0.0
    if bytes_lb:
        candidate_bw = bytes_lb / candidate_s / 1e9 if candidate_s > 0 else 0.0
        baseline_bw = bytes_lb / baseline_s / 1e9 if baseline_s > 0 else 0.0
        print(
            f"{name:30s} {candidate_label}={candidate_s * 1e3:8.4f} ms "
            f"{baseline_label}={baseline_s * 1e3:8.4f} ms "
            f"speedup={speedup:6.2f}x "
            f"BW_lb {candidate_label}/{baseline_label}="
            f"{candidate_bw:7.1f}/{baseline_bw:7.1f} GB/s{diff_text}"
        )
    else:
        print(
            f"{name:30s} {candidate_label}={candidate_s * 1e3:8.4f} ms "
            f"{baseline_label}={baseline_s * 1e3:8.4f} ms "
            f"speedup={speedup:6.2f}x{diff_text}"
        )


def bench_causal_conv1d(num_tests: int) -> None:
    batch, seq_len, dim, width = 16, 2500, 8192, 4
    dtype = torch.float16
    total = batch * seq_len
    query_start_loc = torch.arange(
        0, total + 1, seq_len, device="musa", dtype=torch.int32
    )
    x = torch.randn(total, dim, device="musa", dtype=dtype).contiguous().t()
    weight = torch.randn(dim, width, device="musa", dtype=dtype)
    bias = torch.randn(dim, device="musa", dtype=dtype)
    state = torch.randn(batch, dim, width - 1, device="musa", dtype=dtype)
    cache_indices = torch.arange(batch, device="musa", dtype=torch.int32)
    has_initial_state = torch.ones(batch, device="musa", dtype=torch.bool)

    out_tilelang = tilelang_causal_conv1d_fwd(
        x,
        weight,
        bias,
        state.clone(),
        query_start_loc,
        [seq_len] * batch,
        cache_indices,
        has_initial_state,
        "silu",
    )
    out_triton = triton_causal_conv1d_fwd(
        x,
        weight,
        bias,
        state.clone(),
        query_start_loc,
        [seq_len] * batch,
        cache_indices,
        has_initial_state,
        "silu",
    )
    torch.musa.synchronize()
    max_diff = float((out_tilelang - out_triton).abs().max().item())

    tilelang_state = state.clone()
    triton_state = state.clone()

    def run_tilelang() -> None:
        tilelang_causal_conv1d_fwd(
            x,
            weight,
            bias,
            tilelang_state,
            query_start_loc,
            [seq_len] * batch,
            cache_indices,
            has_initial_state,
            "silu",
        )

    def run_triton() -> None:
        triton_causal_conv1d_fwd(
            x,
            weight,
            bias,
            triton_state,
            query_start_loc,
            [seq_len] * batch,
            cache_indices,
            has_initial_state,
            "silu",
        )

    head_s, body_s = bench_kineto(
        run_tilelang,
        (
            "sglang_musa_causal_conv1d_prefill_width4_kernel",
            "sglang_musa_causal_conv1d_prefill_width4_body_kernel",
        ),
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=True,
    )
    triton_s = bench_kineto(
        run_triton,
        "_causal_conv1d_fwd_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    block_m = 28
    chunks = (seq_len + block_m - 1) // block_m
    bytes_lb = (
        total * dim * _elem_size(dtype) * 2
        + batch * chunks * dim * width * _elem_size(dtype)
        + batch * chunks * dim * _elem_size(dtype)
        + batch * max(chunks - 1, 0) * dim * (width - 1) * _elem_size(dtype)
        + batch * dim * (width - 1) * _elem_size(dtype) * 2
        + batch * 5
    )
    _print_compare(
        "causal_conv1d B16 L2500 D8192",
        head_s + body_s,
        triton_s,
        "TL",
        "TR",
        bytes_lb,
        max_diff,
    )


def bench_gdn_qkvzba(num_tests: int) -> None:
    num_heads_qk, num_heads_v, head_qk, head_v = 4, 8, 128, 128
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    total_qkvz = qkv_dim + num_heads_v * head_v
    for m, kernel_name in (
        (4096, "qkvzba_contiguous_row_kernel"),
        (32768, "qkvzba_contiguous_vec_kernel"),
    ):
        mixed_qkvz = torch.randn(m, total_qkvz, device="musa", dtype=torch.float16)
        mixed_ba = torch.randn(m, num_heads_v * 2, device="musa", dtype=torch.float16)

        out_tilelang = tilelang_gdn(
            mixed_qkvz,
            mixed_ba,
            num_heads_qk,
            num_heads_v,
            head_qk,
            head_v,
        )
        out_triton = triton_gdn(
            mixed_qkvz,
            mixed_ba,
            num_heads_qk,
            num_heads_v,
            head_qk,
            head_v,
        )
        torch.musa.synchronize()
        max_diff = max(
            float((x - y).abs().max().item()) for x, y in zip(out_tilelang, out_triton)
        )

        def run_tilelang() -> None:
            tilelang_gdn(
                mixed_qkvz,
                mixed_ba,
                num_heads_qk,
                num_heads_v,
                head_qk,
                head_v,
            )

        def run_triton() -> None:
            triton_gdn(
                mixed_qkvz,
                mixed_ba,
                num_heads_qk,
                num_heads_v,
                head_qk,
                head_v,
            )

        tilelang_s = bench_kineto(
            run_tilelang,
            kernel_name,
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
            m * total_qkvz * 2
            + m * qkv_dim * 2
            + m * num_heads_v * head_v * 2
            + m * num_heads_v * 2 * 2
            + m * num_heads_v * 2 * 2
        )
        _print_compare(
            f"gdn_qkvzba M{m}",
            tilelang_s,
            triton_s,
            "TL",
            "TR",
            bytes_lb,
            max_diff,
        )


def bench_rms_norm_gated(num_tests: int) -> None:
    m, n = 4096, 8192
    x = torch.randn(m, n, device="musa", dtype=torch.float16)
    z = torch.randn_like(x)
    weight = torch.randn(n, device="musa", dtype=torch.float16)

    out_tilelang = tilelang_rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="swish",
    )
    out_triton = triton_rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="swish",
    )
    torch.musa.synchronize()
    max_diff = float((out_tilelang - out_triton).abs().max().item())

    def run_tilelang() -> None:
        tilelang_rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=z,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="swish",
        )

    def run_triton() -> None:
        triton_rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=z,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="swish",
        )

    tilelang_s = bench_kineto(
        run_tilelang,
        "sglang_musa_rms_norm_gated_cta",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    triton_s = bench_kineto(
        run_triton,
        "_layer_norm_fwd_1pass_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    bytes_lb = m * n * 2 * 2 + n * 2 + m * n * 2 + m * 4
    _print_compare(
        "rms_norm_gated M4096 N8192",
        tilelang_s,
        triton_s,
        "TL",
        "TR",
        bytes_lb,
        max_diff,
    )


def bench_quant(num_tests: int) -> None:
    m, n = 4096, 8192
    x = torch.randn(m, n, device="musa", dtype=torch.float16)
    output_q = torch.empty(m, n, device="musa", dtype=torch.int8)
    output_s = torch.empty(m, n // 128, device="musa", dtype=torch.float32)

    def run_row() -> None:
        per_token_group_quant_8bit(x, output_q, output_s, 128, 1e-10, -128.0, 127.0)

    seconds = bench_kineto(
        run_row,
        "per_token_group_quant_8bit_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    bytes_lb = m * n * 2 + m * n + m * (n // 128) * 4
    _print_row("quant_int8 group128 M4096 N8192", seconds, bytes_lb)

    x_fused = torch.randn(m, n * 2, device="musa", dtype=torch.float16)
    output_q_fused = torch.empty(m, n, device="musa", dtype=torch.int8)
    output_s_fused = torch.empty(m, n // 64, device="musa", dtype=torch.float32)

    def run_fused() -> None:
        per_token_group_quant_8bit(
            x_fused,
            output_q_fused,
            output_s_fused,
            64,
            1e-10,
            -128.0,
            127.0,
            fuse_silu_and_mul=True,
        )

    seconds = bench_kineto(
        run_fused,
        "per_token_group_quant_8bit_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    bytes_lb = m * n * 2 * 2 + m * n + m * (n // 64) * 4
    _print_row("quant_silu_mul group64 M4096 N8192", seconds, bytes_lb)


def bench_rope(num_tests: int) -> None:
    num_tokens, num_heads, num_kv_heads, head_size, rot_dim = 4096, 32, 8, 128, 128
    query = torch.randn(
        num_tokens, num_heads, head_size, device="musa", dtype=torch.float16
    )
    key = torch.randn(
        num_tokens, num_kv_heads, head_size, device="musa", dtype=torch.float16
    )
    positions = torch.arange(num_tokens, device="musa", dtype=torch.long)
    cache = torch.randn(num_tokens + 16, rot_dim, device="musa", dtype=torch.float16)

    def run_prefill() -> None:
        rotary_embedding(positions, query, key, head_size, cache, True)

    seconds = bench_kineto(
        run_prefill,
        "rotary_embedding_prefill_neox_fp16_h2_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    bytes_lb = (
        num_tokens * (num_heads + num_kv_heads) * rot_dim * 2 * 2
        + num_tokens * rot_dim * 2
        + num_tokens * 8
    )
    _print_row("rope_prefill T4096 H32 KV8", seconds, bytes_lb)


def bench_rope_bf16(num_tests: int) -> None:
    """bf16 neox rotary prefill (the bf16 path that uses rope_pair_fp32_bf16)."""
    num_tokens, num_heads, num_kv_heads, head_size, rot_dim = 4096, 32, 8, 128, 128
    query = torch.randn(
        num_tokens, num_heads, head_size, device="musa", dtype=torch.bfloat16
    )
    key = torch.randn(
        num_tokens, num_kv_heads, head_size, device="musa", dtype=torch.bfloat16
    )
    positions = torch.arange(num_tokens, device="musa", dtype=torch.long)
    cache = torch.randn(num_tokens + 16, rot_dim, device="musa", dtype=torch.bfloat16)

    def run_prefill() -> None:
        rotary_embedding(positions, query, key, head_size, cache, True)

    seconds = bench_kineto(
        run_prefill,
        "rotary_embedding_prefill_neox_bf16_h2_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    bytes_lb = (
        num_tokens * (num_heads + num_kv_heads) * rot_dim * 2 * 2
        + num_tokens * rot_dim * 2
        + num_tokens * 8
    )
    _print_row("rope_prefill_bf16 T4096 H32 KV8", seconds, bytes_lb)


def bench_topk(num_tests: int) -> None:
    for experts in (256, 512):
        num_tokens, topk = 4096, 8
        gating_output = torch.randn(
            num_tokens, experts, device="musa", dtype=torch.float16
        )
        topk_weights = torch.empty(num_tokens, topk, device="musa", dtype=torch.float32)
        topk_ids = torch.empty(num_tokens, topk, device="musa", dtype=torch.int32)
        bytes_lb = num_tokens * experts * 2 + num_tokens * topk * (4 + 4) * 2

        def run_softmax() -> None:
            topk_softmax(topk_weights, topk_ids, gating_output, True)

        seconds = bench_kineto(
            run_softmax,
            "topk_softmax_no_bias_renorm_halfwarp_kernel_fixed_k",
            num_tests=num_tests,
            suppress_kineto_output=True,
            flush_l2=True,
        )
        _print_row(f"topk_softmax E{experts} T4096 K8", seconds, bytes_lb)

        def run_sigmoid() -> None:
            topk_sigmoid(topk_weights, topk_ids, gating_output, True)

        seconds = bench_kineto(
            run_sigmoid,
            "topk_sigmoid_no_bias_warp_kernel",
            num_tests=num_tests,
            suppress_kineto_output=True,
            flush_l2=True,
        )
        _print_row(f"topk_sigmoid E{experts} T4096 K8", seconds, bytes_lb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tests", type=int, default=3)
    parser.add_argument(
        "--only",
        choices=[
            "causal_conv1d",
            "gdn_qkvzba",
            "rms_norm_gated",
            "quant",
            "rope",
            "rope_bf16",
            "topk",
        ],
        help="run only one bench section (default: all)",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")

    _set_default_musa_arch()
    arch_list = os.environ.get("SGLANG_MUSA_ARCH_LIST") or os.environ.get(
        "MATE_MUSA_ARCH_LIST", ""
    )
    print(
        f"MUSA JIT kernel cold-cache bench, num_tests={args.num_tests}, "
        f"arch={arch_list}"
    )
    sections = {
        "causal_conv1d": bench_causal_conv1d,
        "gdn_qkvzba": bench_gdn_qkvzba,
        "rms_norm_gated": bench_rms_norm_gated,
        "quant": bench_quant,
        "rope": bench_rope,
        "rope_bf16": bench_rope_bf16,
        "topk": bench_topk,
    }
    if args.only:
        sections[args.only](args.num_tests)
    else:
        for fn in sections.values():
            fn(args.num_tests)


if __name__ == "__main__":
    main()
