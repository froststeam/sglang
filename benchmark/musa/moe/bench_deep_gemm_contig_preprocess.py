from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.deep_gemm_contig_preprocess import (
    _impl_bf16,
    _impl_fp8,
)

FP8_KERNELS = (
    "deep_gemm_contig_preprocess_clear_i32_kernel",
    "deep_gemm_contig_preprocess_count_topk_block_hist_kernel",
    "deep_gemm_contig_preprocess_count_topk_single_block_kernel",
    "deep_gemm_contig_preprocess_count_prefix_topk_single_block_kernel",
    "deep_gemm_contig_preprocess_fill_i32_kernel",
    "deep_gemm_contig_preprocess_prefix_counts_scan",
    "deep_gemm_contig_preprocess_prefix_counts_tree",
    "deep_gemm_contig_preprocess_fp8_assign_compact",
)
BF16_KERNELS = (
    "deep_gemm_contig_preprocess_clear_i32_kernel",
    "deep_gemm_contig_preprocess_count_topk_block_hist_kernel",
    "deep_gemm_contig_preprocess_count_topk_single_block_kernel",
    "deep_gemm_contig_preprocess_count_prefix_topk_single_block_kernel",
    "deep_gemm_contig_preprocess_fill_i32_kernel",
    "deep_gemm_contig_preprocess_prefix_counts_scan",
    "deep_gemm_contig_preprocess_prefix_counts_tree",
    "deep_gemm_contig_preprocess_bf16_assign_compact",
)

DEFAULT_M_LIST = "1,64,1024,4096,16384,65536"
DEFAULT_BLOCK_M_LIST = "128,256,512"


@dataclass(frozen=True)
class Profile:
    key: str
    title: str
    hidden_size: int
    num_experts: int
    topk: int


PROFILES = {
    # Synthetic FP8 preprocess profile for Gemma/JoyAI shapes; local model profiles
    # currently use BF16 weights, but this isolates the FP8 contiguous preprocess path.
    "gemma-e128": Profile(
        "gemma-e128", "Gemma E128 / topk8 / hidden2816", 2816, 128, 8
    ),
    "joyai-e256": Profile(
        "joyai-e256", "JoyAI E256 / topk8 / hidden2048", 2048, 256, 8
    ),
    "qwen122-e257-fused": Profile(
        "qwen122-e257-fused",
        "Qwen122 fused E257 / topk9 / hidden3072",
        3072,
        257,
        9,
    ),
    "qwen397-e513-fused": Profile(
        "qwen397-e513-fused",
        "Qwen397 fused E513 / topk11 / hidden4096",
        4096,
        513,
        11,
    ),
}


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_int_list(value: str) -> list[int]:
    result = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not result:
        raise ValueError(f"empty int list: {value!r}")
    return result


def all_tokens(m: int, topk: int, num_experts: int, block_m: int) -> int:
    src2dst_numel = m * topk
    return cdiv(src2dst_numel + num_experts * (block_m - 1), block_m) * block_m


def byte_estimates(
    m: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    block_m: int,
    all_tokens_value: int,
    dtype: str,
) -> tuple[float, float]:
    hidden_groups = hidden_size // 128
    src2dst_numel = m * topk

    clear_counts = 4 * num_experts
    # topk_ids read + approximate atomic count read/write.
    count_topk = 12 * src2dst_numel
    fill_m_indices = 4 * all_tokens_value
    # counts read + cursor write + average padding writes.
    prefix = 8 * num_experts + 4 * num_experts * (block_m - 1) / 2
    compact_meta = 16 * src2dst_numel
    if dtype == "fp8":
        logical_payload = (
            2 * m * hidden_size
            + src2dst_numel * hidden_size
            + 4 * src2dst_numel * hidden_groups
        )
        dram_payload = logical_payload
    elif dtype == "bf16":
        logical_payload = 4 * src2dst_numel * hidden_size
        dram_payload = 2 * m * hidden_size + 2 * src2dst_numel * hidden_size
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    meta = clear_counts + count_topk + fill_m_indices + prefix + compact_meta
    return meta + logical_payload, meta + dram_payload


def bench_one(
    profile: Profile,
    m: int,
    block_m: int,
    num_tests: int,
    seed: int,
    dtype: str,
) -> dict:
    num_experts = profile.num_experts
    topk = profile.topk
    hidden_size = profile.hidden_size
    all_tokens_value = all_tokens(m, topk, num_experts, block_m)

    torch.manual_seed(seed + m + num_experts + block_m)
    hidden_states = torch.randn((m, hidden_size), device="musa", dtype=torch.bfloat16)
    topk_ids = torch.randint(
        0, num_experts, (m, topk), device="musa", dtype=torch.int32
    )
    if dtype == "fp8":
        output = torch.empty(
            (all_tokens_value, hidden_size),
            device="musa",
            dtype=torch.float8_e4m3fn,
        )
        output_scale = torch.empty(
            (all_tokens_value, hidden_size // 128),
            device="musa",
            dtype=torch.float32,
        )
        kernels = FP8_KERNELS
    elif dtype == "bf16":
        output = torch.empty(
            (all_tokens_value, hidden_size),
            device="musa",
            dtype=torch.bfloat16,
        )
        output_scale = None
        kernels = BF16_KERNELS
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    m_indices = torch.empty((all_tokens_value,), device="musa", dtype=torch.int32)
    src2dst = torch.empty((m * topk,), device="musa", dtype=torch.int32)
    topk_ids_for_combine = torch.empty_like(topk_ids)
    counts = torch.empty((num_experts,), device="musa", dtype=torch.int32)
    cursor = torch.empty_like(counts)

    def fn() -> None:
        if dtype == "fp8":
            _impl_fp8(
                hidden_states,
                topk_ids,
                output,
                output_scale,
                m_indices,
                src2dst,
                topk_ids_for_combine,
                counts,
                cursor,
                num_experts,
                block_m,
            )
        else:
            _impl_bf16(
                hidden_states,
                topk_ids,
                output,
                m_indices,
                src2dst,
                topk_ids_for_combine,
                counts,
                cursor,
                num_experts,
                block_m,
            )

    fn()
    torch.musa.synchronize()
    times = bench_kineto(
        fn,
        kernels,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=False,
    )
    torch.musa.synchronize()

    total_s = sum(float(v) for v in times)
    count_s = float(times[1]) + float(times[2]) + float(times[3])
    prefix_s = float(times[5]) + float(times[6])
    compact_s = float(times[7])
    logical_bytes, dram_bytes = byte_estimates(
        m, hidden_size, num_experts, topk, block_m, all_tokens_value, dtype
    )
    record = {
        "profile": profile.key,
        "dtype": dtype,
        "title": profile.title,
        "m": m,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "topk": topk,
        "block_m": block_m,
        "all_tokens": all_tokens_value,
        "total_us": total_s * 1e6,
        "effective_bw_gbps": logical_bytes / total_s / 1e9 if total_s > 0 else 0.0,
        "logical_bw_gbps": logical_bytes / total_s / 1e9 if total_s > 0 else 0.0,
        "dram_bw_gbps": dram_bytes / total_s / 1e9 if total_s > 0 else 0.0,
        "compact_pct": compact_s / total_s * 100.0 if total_s > 0 else 0.0,
        "clear_us": float(times[0]) * 1e6,
        "count_us": count_s * 1e6,
        "count_block_us": float(times[1]) * 1e6,
        "count_single_block_us": float(times[2]) * 1e6,
        "count_prefix_single_block_us": float(times[3]) * 1e6,
        "fill_us": float(times[4]) * 1e6,
        "prefix_us": prefix_s * 1e6,
        "prefix_scan_us": float(times[5]) * 1e6,
        "prefix_tree_us": float(times[6]) * 1e6,
        "compact_us": compact_s * 1e6,
    }

    hidden_states = None
    topk_ids = None
    output = None
    output_scale = None
    m_indices = None
    src2dst = None
    topk_ids_for_combine = None
    counts = None
    cursor = None
    gc.collect()
    torch.musa.empty_cache()
    return record


def format_table(headers: list[str], body: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in body:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    lines = [
        "  ".join(headers[i].rjust(widths[i]) for i in range(len(headers))),
        "  ".join("-" * width for width in widths),
    ]
    for row in body:
        lines.append("  ".join(row[i].rjust(widths[i]) for i in range(len(row))))
    return "\n".join(lines)


def print_profile_matrix(
    profile: Profile, rows: list[dict], block_ms: list[int]
) -> None:
    by_m = {}
    for row in rows:
        by_m.setdefault(row["m"], {})[row["block_m"]] = row

    print(flush=True)
    print(f"Profile: {profile.key}", flush=True)
    print(
        f"Shape: E={profile.num_experts}, topk={profile.topk}, hidden={profile.hidden_size}",
        flush=True,
    )
    print("Full Matrix", flush=True)

    headers = ["m"]
    for block_m in block_ms:
        headers.extend(
            [f"b{block_m}_us", f"b{block_m}_logGB/s", f"b{block_m}_dramGB/s"]
        )
    body = []
    for m in sorted(by_m):
        per_block = by_m[m]
        values = [str(m)]
        for block_m in block_ms:
            row = per_block.get(block_m)
            if row is None:
                values.extend(["-", "-", "-"])
            else:
                values.extend(
                    [
                        f"{row['total_us']:.2f}",
                        f"{row['logical_bw_gbps']:.2f}",
                        f"{row['dram_bw_gbps']:.2f}",
                    ]
                )
        body.append(values)
    print(format_table(headers, body), flush=True)

    print(flush=True)
    print("Best Block Summary", flush=True)
    headers = ["m", "best_block_m", "best_us", "logGB/s", "dramGB/s", "compact%"]
    body = []
    for m in sorted(by_m):
        best = min(by_m[m].values(), key=lambda row: row["total_us"])
        body.append(
            [
                str(m),
                str(best["block_m"]),
                f"{best['total_us']:.2f}",
                f"{best['logical_bw_gbps']:.2f}",
                f"{best['dram_bw_gbps']:.2f}",
                f"{best['compact_pct']:.2f}",
            ]
        )
    print(format_table(headers, body), flush=True)


def print_cross_profile_summary(
    profiles: list[Profile], rows_by_profile: dict[str, list[dict]]
) -> None:
    print(flush=True)
    print("Cross-Profile Best Summary", flush=True)
    ms = sorted({row["m"] for rows in rows_by_profile.values() for row in rows})
    headers = ["m"] + [profile.key for profile in profiles]
    body = []
    for m in ms:
        values = [str(m)]
        for profile in profiles:
            candidates = [row for row in rows_by_profile[profile.key] if row["m"] == m]
            if not candidates:
                values.append("-")
                continue
            best = min(candidates, key=lambda row: row["total_us"])
            values.append(
                f"{best['total_us']:.2f}us@b{best['block_m']}/"
                f"{best['dram_bw_gbps']:.2f}GB/s"
            )
        body.append(values)
    print(format_table(headers, body), flush=True)


def print_run_header(
    dtype: str, args: argparse.Namespace, ms: list[int], block_ms: list[int]
) -> None:
    print("DeepGEMM Contiguous Preprocess Benchmark", flush=True)
    print("=" * 42, flush=True)
    rows = [
        ["device", f"{torch.musa.current_device()} {torch.musa.get_device_name()}"],
        ["dtype", dtype],
        ["path", f"clear + count + fill + prefix + {dtype}_assign_compact"],
        ["timing", f"MATE bench_kineto, num_tests={args.num_tests}, flush_l2=False"],
        ["m_list", ",".join(str(m) for m in ms)],
        ["block_m", ",".join(str(block_m) for block_m in block_ms)],
        ["bandwidth", "logGB/s=logical traffic, dramGB/s=conservative DRAM estimate"],
    ]
    print(format_table(["field", "value"], rows), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full MUSA TileLang DeepGEMM contiguous FP8 preprocess."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILES),
        choices=sorted(PROFILES),
        help="Profiles to benchmark.",
    )
    parser.add_argument("--m-list", default=DEFAULT_M_LIST)
    parser.add_argument("--block-ms", default=DEFAULT_BLOCK_M_LIST)
    parser.add_argument(
        "--dtype",
        choices=("all", "fp8", "bf16"),
        default="all",
        help="Default runs both fp8 and bf16 full matrices.",
    )
    parser.add_argument("--num-tests", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one progress line after each measured shape.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print available profiles and exit.",
    )
    return parser.parse_args()


def open_jsonl(path: Path | None, dtype: str, dtype_count: int):
    if path is None:
        return None
    if dtype_count == 1:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8")
    path.mkdir(parents=True, exist_ok=True)
    return (path / f"deep_gemm_contig_preprocess_{dtype}.jsonl").open(
        "w", encoding="utf-8"
    )


def run_dtype(
    dtype: str,
    args: argparse.Namespace,
    ms: list[int],
    block_ms: list[int],
    selected_profiles: list[Profile],
    dtype_count: int,
) -> None:
    if not args.no_table:
        print_run_header(dtype, args, ms, block_ms)

    jsonl_file = open_jsonl(args.jsonl, dtype, dtype_count)
    rows_by_profile = {}
    try:
        for profile in selected_profiles:
            rows = []
            for m in ms:
                for block_m in block_ms:
                    record = bench_one(
                        profile=profile,
                        m=m,
                        block_m=block_m,
                        num_tests=args.num_tests,
                        seed=args.seed,
                        dtype=dtype,
                    )
                    rows.append(record)
                    if jsonl_file is not None:
                        jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
                        jsonl_file.flush()
                    if args.progress:
                        print(
                            f"[done] {profile.key} dtype={dtype} "
                            f"m={m} block_m={block_m} "
                            f"{record['total_us']:.2f} us, "
                            f"{record['logical_bw_gbps']:.2f} logGB/s, "
                            f"{record['dram_bw_gbps']:.2f} dramGB/s",
                            flush=True,
                        )
            rows_by_profile[profile.key] = rows
            if not args.no_table:
                print_profile_matrix(profile, rows, block_ms)
        if not args.no_table and len(selected_profiles) > 1:
            print_cross_profile_summary(selected_profiles, rows_by_profile)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()


def main() -> None:
    args = parse_args()
    if args.list_profiles:
        for profile in PROFILES.values():
            print(
                f"{profile.key}: {profile.title} "
                f"(hidden={profile.hidden_size}, E={profile.num_experts}, topk={profile.topk})",
                flush=True,
            )
        return

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("torch.musa is not available")
    torch.musa.set_device(0)

    ms = parse_int_list(args.m_list)
    block_ms = parse_int_list(args.block_ms)
    dtypes = ["fp8", "bf16"] if args.dtype == "all" else [args.dtype]
    selected_profiles = [PROFILES[profile_key] for profile_key in args.profiles]
    for dtype in dtypes:
        run_dtype(dtype, args, ms, block_ms, selected_profiles, len(dtypes))


if __name__ == "__main__":
    main()
