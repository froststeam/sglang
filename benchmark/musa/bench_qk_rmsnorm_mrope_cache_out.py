#!/usr/bin/env python3
"""Benchmark fused_qk_rmsnorm_mrope_cache_out."""

from __future__ import annotations

import argparse
import math
import sys
import types
from dataclasses import dataclass
from types import ModuleType

import torch
import torch_musa  # noqa: F401
from common.modelscope_shapes import (
    add_modelscope_config_args,
    attention_profiles,
    config_for_args,
)
from common.utils import (
    DEFAULT_TOKEN_VALUES,
    bench_mate,
    dtype_from_name,
    error_stats,
    is_close,
    parse_ints,
    print_rows,
    sync,
)

server_args = ModuleType("sglang.srt.server_args")
server_args._global_server_args = types.SimpleNamespace(rl_on_policy_target=None)
server_args.get_global_server_args = lambda: server_args._global_server_args
sys.modules["sglang.srt.server_args"] = server_args

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.norm import (
    fused_qk_rmsnorm_mrope_cache_out,
)
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

QK_NORM_GEMMA = True
QK_NORM_EPS = 1e-6


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


@dataclass(frozen=True)
class BenchProfile:
    name: str
    family: str
    tp: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rot_dim: int
    has_qk_norm: bool
    mrope_section: tuple[int, int, int]
    is_interleaved: bool
    dtype_name: str
    qk_norm_gemma: bool = QK_NORM_GEMMA
    source_models: tuple[str, ...] = ()


def select_modelscope_profiles(config_path) -> list[BenchProfile]:
    return [
        BenchProfile(
            p.name,
            p.family,
            p.tp,
            p.q_heads,
            p.kv_heads,
            p.head_dim,
            p.rot_dim,
            p.has_qk_norm,
            p.mrope_section,
            p.is_interleaved,
            p.dtype_name,
            getattr(p, "qk_norm_gemma", QK_NORM_GEMMA),
            p.source_models,
        )
        for p in attention_profiles(config_path)
    ]


def parse_triple(value: str) -> tuple[int, int, int]:
    parts = parse_ints(value)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected three comma-separated integers, got {value!r}"
        )
    return tuple(parts)


def manual_profile_from_args(args: argparse.Namespace) -> BenchProfile:
    return BenchProfile(
        args.manual_name,
        args.manual_family,
        args.manual_tp,
        args.manual_q_heads,
        args.manual_kv_heads,
        args.manual_head_dim,
        args.manual_rot_dim,
        True,
        args.manual_mrope_section,
        args.manual_interleaved,
        args.manual_dtype,
        args.qk_norm_gemma,
        (),
    )


def kernel_name() -> str:
    return "fused_qk_rmsnorm"


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, gemma: bool) -> torch.Tensor:
    y = x.float()
    scale = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + QK_NORM_EPS)
    weight_f = weight.float() + (1.0 if gemma else 0.0)
    return (y * scale * weight_f).to(dtype=x.dtype)


def make_cos_sin_cache(profile: BenchProfile, dtype: torch.dtype) -> torch.Tensor:
    dim = torch.arange(0, profile.rot_dim, 2, dtype=torch.float32, device="musa")
    inv_freq = torch.exp((-math.log(1000000.0) / profile.rot_dim) * dim)
    positions = torch.arange(8192, dtype=torch.float32, device="musa")
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype=dtype)


def rope_ref(
    profile: BenchProfile,
    positions_3d: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope = MRotaryEmbedding(
        profile.head_dim,
        profile.rot_dim,
        8192,
        1000000,
        True,
        q.dtype,
        list(profile.mrope_section),
        profile.is_interleaved,
    )
    rope.cos_sin_cache = cos_sin_cache
    return rope.forward_native(positions_3d, q, k)


def make_inputs(
    profile: BenchProfile,
    m: int,
    real_cache: bool = True,
    distinct_mrope_positions: bool = False,
) -> tuple[torch.Tensor, ...]:
    dtype = dtype_from_name(profile.dtype_name)
    torch.manual_seed(20260624 + m + profile.q_heads * 100 + profile.kv_heads)
    q = torch.randn((m, profile.q_heads, profile.head_dim), device="musa", dtype=dtype)
    k = torch.randn((m, profile.kv_heads, profile.head_dim), device="musa", dtype=dtype)
    v = torch.randn((m, profile.kv_heads, profile.head_dim), device="musa", dtype=dtype)
    q_weight = torch.randn((profile.head_dim,), device="musa", dtype=dtype)
    k_weight = torch.randn((profile.head_dim,), device="musa", dtype=dtype)
    if distinct_mrope_positions:
        positions = torch.randint(0, 4096, (3, m), device="musa", dtype=torch.int64)
    else:
        positions = (
            torch.randint(0, 4096, (m,), device="musa", dtype=torch.int64)
            .unsqueeze(0)
            .expand(3, -1)
        )
    if real_cache:
        cos_sin_cache = make_cos_sin_cache(profile, dtype)
    else:
        cos_sin_cache = torch.empty((8192, profile.rot_dim), device="musa", dtype=dtype)
    k_cache = torch.empty(
        (m + 16, profile.kv_heads * profile.head_dim), device="musa", dtype=dtype
    )
    v_cache = torch.empty_like(k_cache)
    indices = torch.arange(m, device="musa", dtype=torch.int64) + 8
    return (
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        k_cache,
        v_cache,
        indices,
    )


def logical_bytes(profile: BenchProfile, m: int) -> int:
    elem_size = torch.empty(
        (), dtype=dtype_from_name(profile.dtype_name)
    ).element_size()
    q_elems = m * profile.q_heads * profile.head_dim
    kv_elems = m * profile.kv_heads * profile.head_dim
    rope_elems = m * (profile.q_heads + profile.kv_heads) * profile.rot_dim
    return elem_size * (
        q_elems
        + kv_elems
        + kv_elems
        + q_elems
        + 3 * kv_elems
        + rope_elems
        + 2 * profile.head_dim
    )


def estimated_alloc_bytes(profile: BenchProfile, m: int, correctness: bool) -> int:
    multiplier = 4 if correctness else 2
    return logical_bytes(profile, m) * multiplier


def skipped_row(
    profile: BenchProfile,
    m: int,
    reason: str,
    distinct_mrope_positions: bool = False,
) -> dict[str, str]:
    return {
        "model": profile.name,
        "family": profile.family,
        "m": str(m),
        "tp": str(profile.tp),
        "q": str(profile.q_heads),
        "kv": str(profile.kv_heads),
        "hd": str(profile.head_dim),
        "rot": str(profile.rot_dim),
        "dtype": profile.dtype_name,
        "gemma": str(profile.qk_norm_gemma),
        "mrope_positions": "distinct" if distinct_mrope_positions else "broadcast",
        "status": "skipped",
        "error": reason,
        "correct": "",
        "q_max_abs": "",
        "k_max_abs": "",
        "kc_max_abs": "",
        "vc_max_abs": "",
        "latency_us": "",
        "logical_TBps": "0.000",
        "kernel": kernel_name(),
    }


def print_profiles(profiles: list[BenchProfile]) -> None:
    print_rows(
        [
            {
                "model": p.name,
                "family": p.family,
                "tp": p.tp,
                "q": p.q_heads,
                "kv": p.kv_heads,
                "hd": p.head_dim,
                "rot": p.rot_dim,
                "qk_norm": p.has_qk_norm,
                "mrope": p.mrope_section,
                "inter": p.is_interleaved,
                "gemma": p.qk_norm_gemma,
                "dtype": p.dtype_name,
            }
            for p in profiles
        ],
        (
            "model",
            "family",
            "tp",
            "q",
            "kv",
            "hd",
            "rot",
            "qk_norm",
            "mrope",
            "inter",
            "gemma",
            "dtype",
        ),
    )


def check_correctness(
    profile: BenchProfile, inputs: tuple, atol: float, rtol: float
) -> dict[str, object]:
    q, k, v, q_weight, k_weight, positions, cos_sin_cache, k_cache, v_cache, indices = (
        inputs
    )
    q_ref, k_ref = rope_ref(
        profile,
        positions,
        rmsnorm_ref(q, q_weight, profile.qk_norm_gemma),
        rmsnorm_ref(k, k_weight, profile.qk_norm_gemma),
        cos_sin_cache,
    )
    v_ref = v.reshape(v.shape[0], -1)
    k_cache.zero_()
    v_cache.zero_()
    section_t, section_h, section_w = profile.mrope_section
    q_out, k_out = fused_qk_rmsnorm_mrope_cache_out(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        k_cache,
        v_cache,
        indices,
        True,
        section_t,
        section_h,
        section_w,
        profile.is_interleaved,
        QK_NORM_EPS,
        profile.qk_norm_gemma,
    )
    k_cache_ref = k_ref.reshape(k_ref.shape[0], -1)
    q_stats = error_stats(q_out, q_ref)
    k_stats = error_stats(k_out, k_ref)
    kc_stats = error_stats(k_cache[indices], k_cache_ref)
    vc_stats = error_stats(v_cache[indices], v_ref)
    return {
        "passed": is_close(q_out, q_ref, atol, rtol)
        and is_close(k_out, k_ref, atol, rtol)
        and is_close(k_cache[indices], k_cache_ref, atol, rtol)
        and is_close(v_cache[indices], v_ref, atol, rtol),
        "q_max_abs": q_stats["max_abs"],
        "k_max_abs": k_stats["max_abs"],
        "kc_max_abs": kc_stats["max_abs"],
        "vc_max_abs": vc_stats["max_abs"],
    }


def bench_one(
    profile: BenchProfile,
    m: int,
    num_tests: int,
    flush_l2: bool,
    correctness: bool = True,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    distinct_mrope_positions: bool = False,
) -> dict[str, str] | None:
    if not profile.has_qk_norm or profile.dtype_name != "bf16":
        return None
    correct: dict[str, object] = {"passed": True}
    if correctness:
        correct = check_correctness(
            profile,
            make_inputs(
                profile,
                m,
                distinct_mrope_positions=distinct_mrope_positions,
            ),
            atol,
            rtol,
        )
        sync()
    q, k, v, q_weight, k_weight, positions, cos_sin_cache, k_cache, v_cache, indices = (
        make_inputs(
            profile,
            m,
            real_cache=correctness,
            distinct_mrope_positions=distinct_mrope_positions,
        )
    )
    section_t, section_h, section_w = profile.mrope_section

    def run() -> None:
        fused_qk_rmsnorm_mrope_cache_out(
            q,
            k,
            v,
            q_weight,
            k_weight,
            positions,
            cos_sin_cache,
            k_cache,
            v_cache,
            indices,
            True,
            section_t,
            section_h,
            section_w,
            profile.is_interleaved,
            QK_NORM_EPS,
            profile.qk_norm_gemma,
        )

    run()
    sync()
    seconds = bench_mate(run, kernel_name(), num_tests, flush_l2)
    return {
        "model": profile.name,
        "family": profile.family,
        "m": str(m),
        "tp": str(profile.tp),
        "q": str(profile.q_heads),
        "kv": str(profile.kv_heads),
        "hd": str(profile.head_dim),
        "rot": str(profile.rot_dim),
        "dtype": profile.dtype_name,
        "gemma": str(profile.qk_norm_gemma),
        "qk_norm_gemma": str(profile.qk_norm_gemma),
        "mrope_positions": "distinct" if distinct_mrope_positions else "broadcast",
        "kernel": kernel_name(),
        "correct": str(bool(correct["passed"])),
        "q_max_abs": f"{float(correct.get('q_max_abs', 0.0)):.3g}",
        "k_max_abs": f"{float(correct.get('k_max_abs', 0.0)):.3g}",
        "kc_max_abs": f"{float(correct.get('kc_max_abs', 0.0)):.3g}",
        "vc_max_abs": f"{float(correct.get('vc_max_abs', 0.0)):.3g}",
        "latency_us": f"{seconds * 1e6:.3f}",
        "logical_TBps": (
            f"{logical_bytes(profile, m) / seconds / 1e12:.3f}"
            if seconds > 0
            else "0.000"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default=DEFAULT_TOKEN_VALUES)
    parser.add_argument("--num-tests", type=int, default=8)
    parser.add_argument("--no-flush-l2", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--qk-norm-gemma", type=str_to_bool, default=QK_NORM_GEMMA)
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--max-alloc-gb", type=float, default=24.0)
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--modelscope-shapes", action="store_true")
    parser.add_argument(
        "--distinct-mrope-positions",
        action="store_true",
        help="Use independent 3D MRoPE positions instead of broadcast positions.",
    )
    parser.add_argument("--manual-profile", action="store_true")
    parser.add_argument("--manual-name", default="manual")
    parser.add_argument("--manual-family", default="manual")
    parser.add_argument("--manual-tp", type=int, default=1)
    parser.add_argument("--manual-q-heads", type=int, default=8)
    parser.add_argument("--manual-kv-heads", type=int, default=1)
    parser.add_argument("--manual-head-dim", type=int, default=128)
    parser.add_argument("--manual-rot-dim", type=int, default=128)
    parser.add_argument(
        "--manual-mrope-section", type=parse_triple, default=(24, 20, 20)
    )
    parser.add_argument("--manual-interleaved", type=str_to_bool, default=True)
    parser.add_argument("--manual-dtype", default="bf16")
    add_modelscope_config_args(parser)
    args = parser.parse_args()
    if args.manual_profile:
        profiles = [manual_profile_from_args(args)]
    else:
        profiles = select_modelscope_profiles(config_for_args(args))
        profiles = [
            BenchProfile(
                p.name,
                p.family,
                p.tp,
                p.q_heads,
                p.kv_heads,
                p.head_dim,
                p.rot_dim,
                p.has_qk_norm,
                p.mrope_section,
                p.is_interleaved,
                p.dtype_name,
                args.qk_norm_gemma,
                p.source_models,
            )
            for p in profiles
        ]
    if args.list_profiles:
        print_profiles(profiles)
        return
    rows = []
    for profile in profiles:
        for m in parse_ints(args.m_values):
            estimated_gb = (
                estimated_alloc_bytes(profile, m, not args.skip_correctness) / 1024**3
            )
            if estimated_gb > args.max_alloc_gb:
                rows.append(
                    skipped_row(
                        profile,
                        m,
                        f"estimated allocation {estimated_gb:.1f} GiB exceeds "
                        f"--max-alloc-gb={args.max_alloc_gb:.1f}",
                        args.distinct_mrope_positions,
                    )
                )
                continue
            row = bench_one(
                profile,
                m,
                args.num_tests,
                not args.no_flush_l2,
                not args.skip_correctness,
                args.atol,
                args.rtol,
                args.distinct_mrope_positions,
            )
            if row is not None:
                rows.append(row)
    print_rows(
        rows,
        (
            "model",
            "m",
            "tp",
            "q",
            "kv",
            "hd",
            "rot",
            "dtype",
            "gemma",
            "mrope_positions",
            "status",
            "error",
            "correct",
            "q_max_abs",
            "k_max_abs",
            "kc_max_abs",
            "vc_max_abs",
            "latency_us",
            "logical_TBps",
            "kernel",
        ),
    )


if __name__ == "__main__":
    main()
