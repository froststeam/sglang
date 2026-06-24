from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.deep_gemm_ep_preprocess import (
    _MAX_TOKENS_PER_ASSIGN_LAUNCH,
    _impl_bf16,
    _impl_fp8,
)

KERNELS = (
    "deep_gemm_contig_preprocess_clear_i32_kernel",
    "deep_gemm_ep_preprocess_bf16_assign_masked",
    "deep_gemm_ep_preprocess_fp8_assign_masked",
)

DEFAULT_M_LIST = "1,16,64,128,256,511,512,1024,2048,4096,8192,16384,32768,65536"


@dataclass(frozen=True)
class Profile:
    key: str
    title: str
    hidden_size: int
    num_local_experts: int
    topk: int


PROFILES = {
    "ep8-h3072-topk9": Profile(
        "ep8-h3072-topk9", "EP local E8 / topk9 / hidden3072", 3072, 8, 9
    ),
    "ep16-h4096-topk11": Profile(
        "ep16-h4096-topk11", "EP local E16 / topk11 / hidden4096", 4096, 16, 11
    ),
    "ep32-h4096-topk11": Profile(
        "ep32-h4096-topk11", "EP local E32 / topk11 / hidden4096", 4096, 32, 11
    ),
}


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_int_list(value: str) -> list[int]:
    result = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not result:
        raise ValueError(f"empty int list: {value!r}")
    return result


def dtype_list(value: str) -> list[str]:
    if value == "all":
        return ["bf16", "fp8"]
    if value in ("bf16", "fp8"):
        return [value]
    raise ValueError(f"unsupported dtype: {value}")


def byte_estimates(m: int, hidden_size: int, topk: int, dtype: str) -> tuple[int, int]:
    src2dst_numel = m * topk
    metadata = 16 * src2dst_numel
    if dtype == "bf16":
        logical = 4 * src2dst_numel * hidden_size
        dram = 2 * m * hidden_size + 2 * src2dst_numel * hidden_size
    elif dtype == "fp8":
        hidden_groups = hidden_size // 128
        logical = (
            2 * m * hidden_size
            + src2dst_numel * hidden_size
            + 4 * src2dst_numel * hidden_groups
        )
        dram = logical
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    return logical + metadata, dram + metadata


def bench_one(
    profile: Profile,
    m: int,
    dtype: str,
    num_tests: int,
    seed: int,
) -> dict:
    hidden_size = profile.hidden_size
    num_local_experts = profile.num_local_experts
    topk = profile.topk
    expected_m = cdiv(m * topk, num_local_experts)
    m_max = cdiv(expected_m, 256) * 256

    torch.manual_seed(seed + m + hidden_size + num_local_experts + topk)
    hidden_states = torch.randn((m, hidden_size), device="musa", dtype=torch.bfloat16)
    slots = torch.arange(m * topk, device="musa", dtype=torch.int32)
    topk_ids = (slots % num_local_experts).reshape(m, topk)
    masked_m = torch.empty((num_local_experts,), device="musa", dtype=torch.int32)
    src2dst = torch.empty((m * topk,), device="musa", dtype=torch.int32)
    if dtype == "bf16":
        output = torch.empty(
            (num_local_experts, m_max, hidden_size),
            device="musa",
            dtype=torch.bfloat16,
        )
        output_scale = None
    elif dtype == "fp8":
        output = torch.empty(
            (num_local_experts, m_max, hidden_size),
            device="musa",
            dtype=torch.float8_e4m3fn,
        )
        output_scale = torch.empty(
            (num_local_experts, m_max, hidden_size // 128),
            device="musa",
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    def fn() -> None:
        if dtype == "bf16":
            _impl_bf16(
                topk_ids,
                hidden_states,
                masked_m,
                src2dst,
                output,
                num_local_experts,
            )
        else:
            _impl_fp8(
                topk_ids,
                hidden_states,
                masked_m,
                src2dst,
                output,
                output_scale,
                num_local_experts,
            )

    fn()
    torch.musa.synchronize()
    times = bench_kineto(
        fn,
        KERNELS,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=False,
    )
    torch.musa.synchronize()

    clear_s = float(times[0])
    assign_launches = cdiv(m, _MAX_TOKENS_PER_ASSIGN_LAUNCH)
    assign_s = (float(times[1]) + float(times[2])) * assign_launches
    total_s = clear_s + assign_s
    logical_bytes, dram_bytes = byte_estimates(m, hidden_size, topk, dtype)
    record = {
        "profile": profile.key,
        "title": profile.title,
        "dtype": dtype,
        "m": m,
        "hidden_size": hidden_size,
        "num_local_experts": num_local_experts,
        "topk": topk,
        "expected_m": expected_m,
        "m_max": m_max,
        "assign_launches": assign_launches,
        "total_us": total_s * 1e6,
        "clear_us": clear_s * 1e6,
        "assign_us": assign_s * 1e6,
        "assign_pct": assign_s / total_s * 100.0 if total_s > 0 else 0.0,
        "logical_bw_gbps": logical_bytes / total_s / 1e9 if total_s > 0 else 0.0,
        "dram_bw_gbps": dram_bytes / total_s / 1e9 if total_s > 0 else 0.0,
    }

    hidden_states = None
    topk_ids = None
    masked_m = None
    src2dst = None
    output = None
    output_scale = None
    gc.collect()
    torch.musa.empty_cache()
    return record


def print_record(record: dict) -> None:
    print(
        f"{record['profile']:>18} {record['dtype']:>4} "
        f"m={record['m']:>6} m_max={record['m_max']:>6} "
        f"total={record['total_us']:>9.2f}us "
        f"clear={record['clear_us']:>6.2f}us "
        f"assign={record['assign_us']:>9.2f}us "
        f"dram={record['dram_bw_gbps']:>8.1f}GB/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MUSA TileLang DeepGEMM EP preprocess."
    )
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES))
    parser.add_argument("--dtype", choices=["bf16", "fp8", "all"], default="all")
    parser.add_argument("--m-list", default=DEFAULT_M_LIST)
    parser.add_argument("--num-tests", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jsonl", type=Path, default=None)
    args = parser.parse_args()

    m_list = parse_int_list(args.m_list)
    dtypes = dtype_list(args.dtype)
    records: list[dict] = []
    jsonl_file = None
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = args.jsonl.open("w", encoding="utf-8")

    print("MUSA TileLang DeepGEMM EP preprocess benchmark")
    print(
        f"profiles={','.join(args.profiles)} dtype={args.dtype} "
        f"m_list={','.join(str(m) for m in m_list)} num_tests={args.num_tests}"
    )
    try:
        for profile_key in args.profiles:
            profile = PROFILES[profile_key]
            for dtype in dtypes:
                for m in m_list:
                    record = bench_one(profile, m, dtype, args.num_tests, args.seed)
                    records.append(record)
                    print_record(record)
                    if jsonl_file is not None:
                        jsonl_file.write(json.dumps(record) + "\n")
                        jsonl_file.flush()
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    print("\nBest total by profile/dtype:")
    for profile_key in args.profiles:
        for dtype in dtypes:
            subset = [
                r
                for r in records
                if r["profile"] == profile_key and r["dtype"] == dtype
            ]
            if not subset:
                continue
            small_candidates = [r for r in subset if r["m"] <= 511]
            small = min(small_candidates or subset, key=lambda r: r["total_us"])
            large = max(subset, key=lambda r: r["m"])
            print(
                f"{profile_key:>18} {dtype:>4}: "
                f"best<=511 m={small['m']} {small['total_us']:.2f}us, "
                f"max-m m={large['m']} {large['total_us']:.2f}us"
            )


if __name__ == "__main__":
    main()
