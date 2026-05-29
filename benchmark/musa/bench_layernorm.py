#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from sglang.srt.hardware_backend.musa.jit_kernel.csrc import norm as csrc_norm
from sglang.srt.utils.bench_utils import bench_kineto

OPS = ("rmsnorm", "gemma_rmsnorm", "fused_add_rmsnorm", "gemma_fused_add_rmsnorm")
FUSED_OPS = {"fused_add_rmsnorm", "gemma_fused_add_rmsnorm"}
LAYOUTS = ("compact", "strided")


def sync() -> None:
    torch.musa.synchronize()


def get_flashinfer_norm():
    try:
        import flashinfer.norm as flashinfer_norm
    except Exception:
        return None
    return flashinfer_norm


def provider_module(provider: str):
    if provider == "csrc":
        return csrc_norm
    if provider == "flashinfer":
        mod = get_flashinfer_norm()
        if mod is None:
            raise RuntimeError("flashinfer.norm is not available")
        return mod
    raise ValueError(provider)


def effective_bytes(m: int, hidden: int, elem_bytes: int, fused: bool) -> int:
    data = m * hidden * elem_bytes
    weight = hidden * elem_bytes
    if fused:
        return 4 * data + weight
    return 2 * data + weight


def strided_hidden(hidden: int, stride_padding: int) -> int:
    padding = stride_padding if stride_padding > 0 else max(1, hidden // 8)
    return hidden + padding


def make_strided_2d(m: int, hidden: int, stride0: int, dtype: torch.dtype):
    return torch.randn((m, stride0), device="musa", dtype=dtype)[:, :hidden]


def make_inputs(
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


def make_fn(provider: str, op_name: str, x, residual, weight, out, eps: float):
    mod = provider_module(provider)
    op = getattr(mod, op_name)
    if op_name in FUSED_OPS:

        def fn():
            op(x, residual, weight, eps)

        kernel_name = "fused_add_rmsnorm"
    else:

        def fn():
            op(x, weight, eps=eps, out=out)

        kernel_name = "rmsnorm"
    return fn, kernel_name


def run_one(
    provider: str,
    op_name: str,
    m: int,
    hidden: int,
    dtype: torch.dtype,
    layout: str,
    stride_padding: int,
    reps: int,
    flush_l2: bool,
):
    x, residual, weight, out, stride0 = make_inputs(
        m, hidden, dtype, layout, stride_padding
    )
    fn, kernel_name = make_fn(provider, op_name, x, residual, weight, out, 1e-6)
    fn()
    sync()

    seconds = bench_kineto(
        fn,
        kernel_names=kernel_name,
        num_tests=reps,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
        with_multiple_kernels=True,
    )
    latency_us = seconds * 1e6
    bytes_ = effective_bytes(
        m, hidden, torch.empty((), dtype=dtype).element_size(), op_name in FUSED_OPS
    )
    gbps = bytes_ / seconds / 1e9
    return latency_us, gbps, stride0


def write_markdown(rows: list[dict[str, object]], md_path: Path) -> None:
    providers = []
    for row in rows:
        provider = row["provider"]
        if provider not in providers:
            providers.append(provider)

    with md_path.open("w") as f:
        f.write(
            "Units: `latency_us / logical_GBps`; norm logical bytes count input/output/residual traffic plus one weight read.\n\n"
        )
        for op in sorted({r["op"] for r in rows}):
            f.write(f"## {op}\n\n")
            for layout in sorted({r["layout"] for r in rows if r["op"] == op}):
                f.write(f"### layout={layout}\n\n")
                for m in sorted(
                    {
                        int(r["m"])
                        for r in rows
                        if r["op"] == op and r["layout"] == layout
                    }
                ):
                    f.write(f"#### m={m}\n\n")
                    header = "| hidden | stride0 |" + "".join(
                        f" {p} us | {p} GB/s |" for p in providers
                    )
                    sep = "|---:|---:|" + "---:|---:|" * len(providers)
                    f.write(header + "\n")
                    f.write(sep + "\n")
                    for hidden in sorted(
                        {
                            int(r["hidden"])
                            for r in rows
                            if r["op"] == op
                            and r["layout"] == layout
                            and int(r["m"]) == m
                        }
                    ):
                        row_for_stride = next(
                            r
                            for r in rows
                            if r["op"] == op
                            and r["layout"] == layout
                            and int(r["m"]) == m
                            and int(r["hidden"]) == hidden
                        )
                        cells = [f"| {hidden} | {row_for_stride['stride0']} |"]
                        for provider in providers:
                            row = next(
                                (
                                    r
                                    for r in rows
                                    if r["op"] == op
                                    and r["layout"] == layout
                                    and int(r["m"]) == m
                                    and int(r["hidden"]) == hidden
                                    and r["provider"] == provider
                                ),
                                None,
                            )
                            if row is None or row["status"] != "ok":
                                cells.append(" N/A | N/A |")
                            else:
                                cells.append(
                                    f" {float(row['latency_us']):.3f} | {float(row['logical_GBps']):.1f} |"
                                )
                        f.write("".join(cells) + "\n")
                    f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="1,16,256,4096,32768")
    parser.add_argument(
        "--hidden-values",
        default="128,256,384,512,640,768,896,1024,1152,1408,1536,1920,2048,4096",
    )
    parser.add_argument(
        "--ops",
        default="rmsnorm,gemma_rmsnorm,fused_add_rmsnorm,gemma_fused_add_rmsnorm",
    )
    parser.add_argument("--providers", default="csrc,flashinfer")
    parser.add_argument(
        "--layouts",
        default="compact",
        help="comma-separated input layouts: compact,strided",
    )
    parser.add_argument(
        "--stride-padding",
        type=int,
        default=0,
        help="extra columns for strided layout; default is hidden//8",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--no-flush-l2", action="store_true")
    parser.add_argument("--output", default="layernorm_musa_compare.md")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    m_values = [int(x) for x in args.m_values.split(",") if x]
    hidden_values = [int(x) for x in args.hidden_values.split(",") if x]
    ops = [x for x in args.ops.split(",") if x]
    providers = [x for x in args.providers.split(",") if x]
    layouts = [x for x in args.layouts.split(",") if x]
    for op in ops:
        if op not in OPS:
            raise ValueError(f"unsupported op: {op}")
    for layout in layouts:
        if layout not in LAYOUTS:
            raise ValueError(f"unsupported layout: {layout}")

    rows = []
    for op in ops:
        for hidden in hidden_values:
            for m in m_values:
                for layout in layouts:
                    stride0 = (
                        hidden
                        if layout == "compact"
                        else strided_hidden(hidden, args.stride_padding)
                    )
                    for provider in providers:
                        row = {
                            "op": op,
                            "provider": provider,
                            "layout": layout,
                            "dtype": args.dtype,
                            "m": m,
                            "hidden": hidden,
                            "stride0": stride0,
                            "latency_us": "",
                            "logical_GBps": "",
                            "status": "ok",
                        }
                        try:
                            latency_us, gbps, stride0 = run_one(
                                provider,
                                op,
                                m,
                                hidden,
                                dtype,
                                layout,
                                args.stride_padding,
                                args.reps,
                                not args.no_flush_l2,
                            )
                            row["stride0"] = stride0
                            row["latency_us"] = f"{latency_us:.3f}"
                            row["logical_GBps"] = f"{gbps:.1f}"
                        except Exception as exc:
                            row["status"] = type(exc).__name__
                        rows.append(row)
                        print(row, flush=True)

    md_path = Path(args.output).resolve()
    write_markdown(rows, md_path)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
