from __future__ import annotations

import argparse
import types
from dataclasses import dataclass

import torch
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

import sglang.srt.server_args as server_args
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.norm import (
    fused_qk_rmsnorm_mrope,
    fused_qk_rmsnorm_mrope_cache,
    fused_qk_rmsnorm_mrope_cache_out,
)
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

server_args._global_server_args = types.SimpleNamespace(rl_on_policy_target=None)

DEFAULT_M_VALUES = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536"


@dataclass(frozen=True)
class Profile:
    name: str
    q_heads: int
    kv_heads: int
    head_dim: int
    rot_dim: int
    mrope_section: tuple[int, int, int]


PROFILES = (
    Profile("qwen3.5-35b-a3b", 16, 2, 256, 64, (11, 11, 10)),
    Profile("qwen3.5-27b", 24, 4, 256, 64, (11, 11, 10)),
    Profile("qwen3.5-122b-a10b", 32, 2, 256, 64, (11, 11, 10)),
    Profile("qwen3.5-397b-a17b", 32, 2, 256, 64, (11, 11, 10)),
    Profile("h256r64-4q1kv", 4, 1, 256, 64, (11, 11, 10)),
    Profile("h256r64-8q1kv", 8, 1, 256, 64, (11, 11, 10)),
    Profile("h256r64-16q1kv", 16, 1, 256, 64, (11, 11, 10)),
    Profile("h256r64-16q4kv", 16, 4, 256, 64, (11, 11, 10)),
    Profile("h256r64-32q4kv", 32, 4, 256, 64, (11, 11, 10)),
    Profile("h256r64-64q4kv", 64, 4, 256, 64, (11, 11, 10)),
    Profile("h256r64-64q8kv", 64, 8, 256, 64, (11, 11, 10)),
    Profile("qwen3-vl-32q4kv", 32, 4, 128, 128, (24, 20, 20)),
    Profile("qwen3-vl-32q32kv", 32, 32, 128, 128, (24, 20, 20)),
    Profile("qwen3-vl-moe-30b-a3b", 16, 16, 128, 128, (24, 20, 20)),
    Profile("h128full-2q2kv", 2, 2, 128, 128, (24, 20, 20)),
    Profile("h128full-4q4kv", 4, 4, 128, 128, (24, 20, 20)),
    Profile("h128full-8q8kv", 8, 8, 128, 128, (24, 20, 20)),
    Profile("h64full-32q8kv", 32, 8, 64, 64, (12, 10, 10)),
    Profile("h96full-32q8kv", 32, 8, 96, 96, (16, 16, 16)),
    Profile("h128full-32q8kv", 32, 8, 128, 128, (24, 20, 20)),
    Profile("h128r64-32q8kv", 32, 8, 128, 64, (12, 10, 10)),
    Profile("h192full-32q8kv", 32, 8, 192, 192, (32, 32, 32)),
    Profile("h256full-32q8kv", 32, 8, 256, 256, (48, 40, 40)),
    Profile("h256r128-32q8kv", 32, 8, 256, 128, (24, 20, 20)),
)

PROFILE_GROUPS = {
    "mainstream-hidden": (
        "h64full-32q8kv",
        "h96full-32q8kv",
        "h128full-32q8kv",
        "h128r64-32q8kv",
        "h192full-32q8kv",
        "h256full-32q8kv",
        "h256r128-32q8kv",
    ),
    "qwen-mrope": (
        "qwen3.5-27b",
        "qwen3-vl-32q4kv",
        "qwen3-vl-32q32kv",
    ),
}


def parse_m_values(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v]


def kernel_name(profile: Profile, mode: str, m: int | None = None) -> str:
    suffix = "_cache" if mode in ("cache", "cache_out") else ""
    h128_full = (
        profile.head_dim == 128
        and profile.rot_dim == 128
        and profile.mrope_section == (24, 20, 20)
    )
    if (
        profile.head_dim == 256
        and profile.rot_dim == 64
        and profile.mrope_section == (11, 11, 10)
    ):
        return f"fused_qk_rmsnorm_mrope{suffix}_h256r64_bf16_kernel"
    if (
        mode == "qk"
        and m is not None
        and m < 128
        and profile.q_heads == 32
        and profile.kv_heads == 4
        and h128_full
    ):
        return "fused_qk_rmsnorm_mrope_32q4kv_h128_bf16_kernel"
    if mode == "qk" and h128_full and profile.q_heads == 2 and profile.kv_heads == 2:
        return "fused_qk_rmsnorm_mrope_h128_full_bf16_kernel"
    if (
        mode == "qk"
        and h128_full
        and (profile.q_heads, profile.kv_heads)
        in ((4, 4), (8, 8), (16, 16), (32, 32), (32, 4))
    ):
        return "fused_qk_rmsnorm_mrope_h128_full_halfwarp_bf16_kernel"
    if profile.q_heads == 32 and profile.kv_heads == 4 and h128_full:
        return f"fused_qk_rmsnorm_mrope{suffix}_32q4kv_h128_bf16_kernel"
    if h128_full and (profile.q_heads, profile.kv_heads) in (
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
    ):
        return f"fused_qk_rmsnorm_mrope{suffix}_h128_full_bf16_kernel"
    return "fused_qk_rmsnorm_mrope_generic_bf16_kernel"


def path_name(profile: Profile, mode: str, m: int | None = None) -> str:
    name = kernel_name(profile, mode, m)
    if "h256r64" in name:
        return "h256r64"
    if "32q4kv_h128" in name:
        return "fast32q4"
    if "h128_full" in name:
        return "h128full"
    return "generic"


def logical_bytes(profile: Profile, m: int, mode: str) -> int:
    elem_size = 2
    q_elems = m * profile.q_heads * profile.head_dim
    kv_elems = m * profile.kv_heads * profile.head_dim
    rope_elems = m * (profile.q_heads + profile.kv_heads) * profile.rot_dim
    if mode == "qk":
        weights = 2 * profile.head_dim
        return elem_size * (
            q_elems + kv_elems + weights + rope_elems + q_elems + kv_elems
        )

    qk_elems = q_elems + kv_elems
    out_elems = q_elems + 2 * kv_elems
    return elem_size * (q_elems + 2 * kv_elems + qk_elems + out_elems + rope_elems)


def make_inputs(profile: Profile, m: int) -> tuple[torch.Tensor, ...]:
    device = "musa"
    dtype = torch.bfloat16
    torch.manual_seed(20260623 + m + profile.q_heads * 100 + profile.kv_heads)
    q = torch.randn((m, profile.q_heads, profile.head_dim), device=device, dtype=dtype)
    k = torch.randn((m, profile.kv_heads, profile.head_dim), device=device, dtype=dtype)
    v = torch.randn((m, profile.kv_heads, profile.head_dim), device=device, dtype=dtype)
    q_weight = torch.randn((profile.head_dim,), device=device, dtype=dtype)
    k_weight = torch.randn((profile.head_dim,), device=device, dtype=dtype)
    positions = torch.randint(0, 4096, (3, m), device=device, dtype=torch.int64)
    rope = MRotaryEmbedding(
        profile.head_dim,
        profile.rot_dim,
        8192,
        1000000,
        True,
        dtype,
        list(profile.mrope_section),
        True,
    ).to(device)
    cos_sin_cache = rope.cos_sin_cache.to(device=device, dtype=dtype)
    k_cache = torch.empty(
        (m + 16, profile.kv_heads * profile.head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)
    indices = torch.arange(m, device=device, dtype=torch.int64) + 8
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


def bench_one(
    profile: Profile, m: int, mode: str, num_tests: int, flush_l2: bool
) -> dict[str, str]:
    args = make_inputs(profile, m)
    section_t, section_h, section_w = profile.mrope_section

    def run() -> None:
        if mode == "cache":
            fused_qk_rmsnorm_mrope_cache(
                *args,
                True,
                section_t,
                section_h,
                section_w,
                True,
                1e-6,
                True,
            )
        elif mode == "cache_out":
            fused_qk_rmsnorm_mrope_cache_out(
                *args,
                True,
                section_t,
                section_h,
                section_w,
                True,
                1e-6,
                True,
            )
        else:
            fused_qk_rmsnorm_mrope(
                args[0],
                args[1],
                args[3],
                args[4],
                args[5],
                args[6],
                True,
                section_t,
                section_h,
                section_w,
                True,
                1e-6,
                True,
            )

    run()
    torch.musa.synchronize()
    seconds = bench_kineto(
        run,
        kernel_names=kernel_name(profile, mode, m),
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    latency_us = seconds * 1e6
    tbps = logical_bytes(profile, m, mode) / seconds / 1e12
    return {
        "model": profile.name,
        "path": path_name(profile, mode, m),
        "m": str(m),
        "q": str(profile.q_heads),
        "kv": str(profile.kv_heads),
        "hd": str(profile.head_dim),
        "rot": str(profile.rot_dim),
        "latency_us": f"{latency_us:.3f}",
        "logical_TBps": f"{tbps:.3f}",
    }


def print_rows(rows: list[dict[str, str]]) -> None:
    columns = (
        "model",
        "path",
        "m",
        "q",
        "kv",
        "hd",
        "rot",
        "latency_us",
        "logical_TBps",
    )
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    print(" ".join(column.rjust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" ".join(row[column].rjust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default=DEFAULT_M_VALUES)
    parser.add_argument("--mode", choices=("cache", "cache_out", "qk"), default="cache")
    parser.add_argument("--num-tests", type=int, default=8)
    parser.add_argument("--no-flush-l2", action="store_true")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma separated profile names, or all.",
    )
    args = parser.parse_args()

    wanted = set(p.name for p in PROFILES)
    if args.models != "all":
        wanted = set()
        for name in args.models.split(","):
            wanted.update(PROFILE_GROUPS.get(name, (name,)))
    profiles = [profile for profile in PROFILES if profile.name in wanted]
    missing = wanted - {profile.name for profile in profiles}
    if missing:
        raise ValueError(f"Unknown model profiles: {sorted(missing)}")

    print("device", torch.musa.current_device(), torch.musa.get_device_name())
    print(
        f"fused qk rmsnorm + mrope mode={args.mode}: "
        f"num_tests={args.num_tests}, flush_l2={not args.no_flush_l2}"
    )
    print("Units: latency_us and logical_TBps.")

    rows: list[dict[str, str]] = []
    for profile in profiles:
        for m in parse_m_values(args.m_values):
            rows.append(
                bench_one(
                    profile,
                    m,
                    args.mode,
                    args.num_tests,
                    flush_l2=not args.no_flush_l2,
                )
            )
    print_rows(rows)


if __name__ == "__main__":
    main()
