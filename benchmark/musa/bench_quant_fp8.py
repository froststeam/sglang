#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import triton
from mate.testing.utils import bench_kineto

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.quant import (
    per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    _per_token_group_quant_8bit,
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
)


def sync() -> None:
    torch.musa.synchronize()


def alloc_outputs(x: torch.Tensor, group_size: int):
    q = torch.empty_like(x, dtype=fp8_dtype)
    s = create_per_token_group_quant_fp8_output_scale(
        x_shape=x.shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )
    return q, s


def make_ref(x: torch.Tensor, group_size: int):
    x_f = x.float().reshape(x.shape[0], x.shape[1] // group_size, group_size)
    amax = torch.amax(torch.abs(x_f), dim=-1).clamp(min=1e-10)
    scale = fp8_max / amax
    q = (x_f * scale.unsqueeze(-1)).clamp(fp8_min, fp8_max).to(fp8_dtype)
    return q.reshape_as(x), (amax / fp8_max).float()


def assert_fp8_quant_close(
    q: torch.Tensor,
    s: torch.Tensor,
    ref_q: torch.Tensor,
    ref_s: torch.Tensor,
    provider: str,
    m: int,
    hidden: int,
) -> None:
    torch.testing.assert_close(s, ref_s, rtol=1e-4, atol=1e-6)

    q_f = q.float()
    ref_q_f = ref_q.float()
    diff = torch.abs(q_f - ref_q_f)
    mismatch = diff != 0
    mismatch_count = int(mismatch.sum().item())
    if mismatch_count == 0:
        return

    max_abs_diff = float(diff.max().item())
    mismatch_ratio = mismatch_count / q.numel()
    max_allowed_abs_diff = 32.0
    max_allowed_mismatch_ratio = 0.01

    if (
        max_abs_diff > max_allowed_abs_diff
        or mismatch_ratio > max_allowed_mismatch_ratio
    ):
        mismatch_idx = mismatch.nonzero()[0].tolist()
        raise AssertionError(
            "FP8 quantization output differs from reference beyond rounding "
            f"tolerance for provider={provider}, m={m}, hidden={hidden}: "
            f"mismatches={mismatch_count}/{q.numel()} "
            f"({mismatch_ratio:.4%}), max_abs_diff={max_abs_diff}, "
            f"first_mismatch_index={mismatch_idx}, "
            f"q={float(q_f[tuple(mismatch_idx)].item())}, "
            f"ref_q={float(ref_q_f[tuple(mismatch_idx)].item())}"
        )


def make_fn(
    provider: str, x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, group_size: int
):
    if provider == "triton":
        total_groups = x.numel() // group_size
        block = triton.next_power_of_2(group_size)
        num_warps = min(max(block // 256, 1), 8)

        def fn():
            _per_token_group_quant_8bit[(total_groups,)](
                x,
                q,
                s,
                group_size,
                group_size,
                1e-10,
                bit8_min=fp8_min,
                bit8_max=fp8_max,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=1,
            )

        return fn, "_per_token_group_quant_8bit"

    if provider == "csrc":

        def fn():
            per_token_group_quant_8bit(
                input=x,
                output_q=q,
                output_s=s,
                group_size=group_size,
                eps=1e-10,
                min_8bit=fp8_min,
                max_8bit=fp8_max,
                scale_ue8m0=False,
                fuse_silu_and_mul=False,
                enable_v2=True,
            )

        return fn, "per_token_group_quant"

    raise ValueError(f"unsupported provider: {provider}")


def run_one(
    provider: str,
    m: int,
    hidden: int,
    reps: int,
    correctness: bool,
):
    group_size = 128
    x = torch.randn((m, hidden), device="musa", dtype=torch.bfloat16)
    q, s = alloc_outputs(x, group_size)
    fn, kernel_name = make_fn(provider, x, q, s, group_size)

    fn()
    sync()

    if correctness:
        ref_q, ref_s = make_ref(x, group_size)
        sync()
        assert_fp8_quant_close(q, s, ref_q, ref_s, provider, m, hidden)

    latency_s = bench_kineto(
        fn,
        kernel_names=kernel_name,
        num_tests=reps,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=True,
    )
    latency_us = latency_s * 1e6
    logical_bytes = m * hidden * 2 + m * hidden + m * (hidden // group_size) * 4
    gbps = logical_bytes / latency_s / 1e9
    return latency_us, gbps


def write_single_provider_markdown(rows: list[dict[str, str]], md_path: Path) -> None:
    with md_path.open("w") as f:
        f.write(
            "Units: `latency_us / logical_GBps`; logical bytes = bf16 input + fp8 output + fp32 scales, group_size=128.\n\n"
        )
        for hidden in sorted({int(r["hidden"]) for r in rows}):
            f.write(f"### hidden={hidden}\n\n")
            f.write("| m | latency_us | logical_GBps |\n")
            f.write("|---:|---:|---:|\n")
            for row in rows:
                if int(row["hidden"]) != hidden:
                    continue
                f.write(
                    f"| {row['m']} | {float(row['latency_us']):.3f} | {float(row['logical_GBps']):.1f} |\n"
                )
            f.write("\n")


def write_compare_markdown(
    rows: list[dict[str, str]], providers: list[str], md_path: Path
) -> None:
    with md_path.open("w") as f:
        f.write(
            "Units: `latency_us / logical_GBps`; logical bytes = bf16 input + fp8 output + fp32 scales, group_size=128.\n\n"
        )
        for m in sorted({int(r["m"]) for r in rows}):
            f.write(f"### m={m}\n\n")
            header = "| hidden |" + "".join(f" {p} us | {p} GB/s |" for p in providers)
            if "triton" in providers and "csrc" in providers:
                header += " csrc speedup |"
            f.write(header + "\n")
            sep = "|---:|" + "---:|---:|" * len(providers)
            if "triton" in providers and "csrc" in providers:
                sep += "---:|"
            f.write(sep + "\n")
            for hidden in sorted({int(r["hidden"]) for r in rows}):
                cells = [f"| {hidden} |"]
                by_provider = {}
                for provider in providers:
                    row = next(
                        r
                        for r in rows
                        if int(r["m"]) == m
                        and int(r["hidden"]) == hidden
                        and r["provider"] == provider
                    )
                    by_provider[provider] = row
                    cells.append(
                        f" {float(row['latency_us']):.3f} | {float(row['logical_GBps']):.1f} |"
                    )
                if "triton" in providers and "csrc" in providers:
                    tr_us = float(by_provider["triton"]["latency_us"])
                    cs_us = float(by_provider["csrc"]["latency_us"])
                    cells.append(f" {tr_us / cs_us:.2f}x |")
                f.write("".join(cells) + "\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="1,16,256,4096,32768")
    parser.add_argument(
        "--hidden-values",
        default="128,256,384,512,640,768,896,1024,1152,1408,1536,1920,2048,2304,2944,4096",
    )
    parser.add_argument("--providers", default="csrc")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", default="quant_fp8_group128_bf16.md")
    args = parser.parse_args()

    m_values = [int(x) for x in args.m_values.split(",") if x]
    hidden_values = [int(x) for x in args.hidden_values.split(",") if x]
    providers = [x for x in args.providers.split(",") if x]
    if not providers:
        raise ValueError("at least one provider is required")

    rows = []
    for hidden in hidden_values:
        if hidden % 128 != 0:
            raise ValueError(f"hidden must be divisible by 128, got {hidden}")
        for m in m_values:
            for provider in providers:
                latency_us, gbps = run_one(
                    provider,
                    m,
                    hidden,
                    args.reps,
                    correctness=not args.skip_correctness,
                )
                row = {
                    "provider": provider,
                    "dtype": "bf16_to_fp8_e4m3",
                    "group_size": "128",
                    "m": str(m),
                    "hidden": str(hidden),
                    "latency_us": f"{latency_us:.3f}",
                    "logical_GBps": f"{gbps:.1f}",
                }
                rows.append(row)
                print(row, flush=True)

    md_path = Path(args.output).resolve()
    if len(providers) == 1:
        write_single_provider_markdown(rows, md_path)
    else:
        write_compare_markdown(rows, providers, md_path)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
