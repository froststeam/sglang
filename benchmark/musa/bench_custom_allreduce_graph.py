#!/usr/bin/env python3
"""Kineto graph benchmark for MUSA custom all-reduce + RMSNorm fusion.

Example:
    PYTHONPATH=/data/sglang_musa_0.5.12_moe_port/python MUSA_VISIBLE_DEVICES=0,1 \
    python benchmark/musa/bench_custom_allreduce_graph.py \
      --world-size 2 --hidden-values 4096 --tokens 512,1024,4096,8192
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import socket
from contextlib import nullcontext
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

from sglang.srt.distributed import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    MusaJitCustomAllreduce,
)
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.norm import rmsnorm

KERNEL_SUBSTRINGS = (
    "custom_all_reduce_residual_rmsnorm",
    "custom_all_reduce_residual_2shot_rmsnorm_row",
    "custom_all_reduce_residual",
    "custom_all_reduce_2shot",
    "rmsnorm",
)


def fmt_us(value: str | float | None) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.3f}us"


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def open_port() -> int:
    for _ in range(32):
        port = random.randint(20000, 60000)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def sync() -> None:
    torch.musa.synchronize()


def all_ok(ok: bool, group) -> bool:
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def make_inputs(
    rank: int, rows: int, hidden: int, dtype: torch.dtype, device, call_idx: int = 0
):
    inp = torch.full(
        (rows, hidden), rank + 1 + call_idx * 0.125, dtype=dtype, device=device
    )
    residual = torch.full(
        (rows, hidden), 0.25 + call_idx * 0.03125, dtype=dtype, device=device
    )
    weight = torch.linspace(
        0.5 + call_idx * 0.01, 1.5 + call_idx * 0.01, hidden, dtype=dtype, device=device
    )
    return inp, residual, weight


def make_replay_graph_fn(
    fn,
    check_group,
    check_fn=None,
    reset_fn=None,
    capture_ctx_fn=None,
    register_graph_buffers_fn=None,
):
    result = {}
    capture_ctx_fn = capture_ctx_fn or (lambda: nullcontext())
    graph = None
    for _ in range(3):
        graph = torch.musa.MUSAGraph()
        with capture_ctx_fn():
            if reset_fn is not None:
                reset_fn()
            fn()
            sync()
            if reset_fn is not None:
                reset_fn()
            sync()
            with torch.musa.graph(graph):
                result["value"] = fn()
        registered = (
            register_graph_buffers_fn() if register_graph_buffers_fn is not None else 0
        )
        if not registered:
            break
        del graph
        graph = None
        sync()
    if graph is None:
        return None
    sync()
    if reset_fn is not None:
        reset_fn()
        sync()
    graph.replay()
    sync()
    if check_fn is not None and not all_ok(
        bool(check_fn(result["value"])), check_group
    ):
        return None
    return graph.replay


def load_trace_events(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    else:
        data = json.loads(path.read_text())
    return data.get("traceEvents", data if isinstance(data, list) else [])


def kernel_category(name: str) -> str:
    if (
        "custom_all_reduce_residual_rmsnorm" in name
        or "custom_all_reduce_residual_2shot_rmsnorm_row" in name
    ):
        return "fused"
    if "custom_all_reduce_residual" in name or "custom_all_reduce_2shot" in name:
        return "ar"
    if "rmsnorm" in name:
        return "rms"
    return ""


def summarize_trace_by_category(path: Path) -> dict[str, tuple[float, float, int, str]]:
    by_category: dict[str, list[tuple[float, str]]] = {}
    for event in load_trace_events(path):
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name = str(event.get("name", ""))
        category = kernel_category(name)
        if not category:
            continue
        by_category.setdefault(category, []).append((float(event["dur"]), name))

    result: dict[str, tuple[float, float, int, str]] = {}
    for category, entries in by_category.items():
        values = [duration for duration, _ in entries]
        best_duration, best_name = min(entries, key=lambda item: item[0])
        result[category] = (best_duration, median(values), len(values), best_name)
    return result


def summarize_trace(path: Path) -> tuple[str, float, float, int, str]:
    by_name: dict[str, list[float]] = {}
    for event in load_trace_events(path):
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name = str(event.get("name", ""))
        if not any(s in name for s in KERNEL_SUBSTRINGS):
            continue
        by_name.setdefault(name, []).append(float(event["dur"]))
    if not by_name:
        return "", 0.0, 0.0, 0, ""

    best_name = ""
    best_min = float("inf")
    best_values: list[float] = []
    for name, values in by_name.items():
        name_min = min(values)
        if name_min < best_min:
            best_name = name
            best_min = name_min
            best_values = values
    top = sorted(
        (
            (min(values), median(values), len(values), name)
            for name, values in by_name.items()
        ),
        key=lambda item: item[0],
    )[:6]
    top_text = "; ".join(
        f"{name}:min={mn:.3f}us,med={md:.3f}us,n={cnt}" for mn, md, cnt, name in top
    )
    return best_name, best_min, median(best_values), len(best_values), top_text


def bench_graph_case(
    comm,
    rank: int,
    rows: int,
    hidden: int,
    args,
    device,
    data_group,
    check_group,
    case: str,
    trace_path: Path,
) -> dict:
    dtype = torch.bfloat16
    graph_calls = int(args.graph_calls)
    inputs = []
    residuals = []
    weights = []
    input_inits = []
    residual_inits = []
    residual_refs = []
    norm_refs = []
    norm_outs = []
    for call_idx in range(graph_calls):
        inp, residual, weight = make_inputs(rank, rows, hidden, dtype, device, call_idx)
        inp_init = inp.clone()
        residual_init = residual.clone()
        reduced_ref = inp_init.clone()
        dist.all_reduce(reduced_ref, group=data_group)
        residual_ref = reduced_ref + residual_init
        norm_ref = torch.nn.functional.rms_norm(
            residual_ref.float(), (hidden,), weight.float(), 1e-6
        ).to(dtype)
        inputs.append(inp)
        residuals.append(residual)
        weights.append(weight)
        input_inits.append(inp_init)
        residual_inits.append(residual_init)
        residual_refs.append(residual_ref)
        norm_refs.append(norm_ref)
        norm_outs.append(torch.empty_like(inp))

    def reset_inputs():
        for inp, inp_init, residual, residual_init in zip(
            inputs, input_inits, residuals, residual_inits
        ):
            inp.copy_(inp_init)
            residual.copy_(residual_init)

    def normalize_outputs(out):
        if graph_calls == 1:
            return [out]
        return out

    if case == "fused":

        def fn():
            outputs = [
                comm.fused_allreduce_rmsnorm(inp, residual, weight, 1e-6)
                for inp, residual, weight in zip(inputs, residuals, weights)
            ]
            return outputs[0] if graph_calls == 1 else outputs

        def check_fn(out):
            outputs = normalize_outputs(out)
            if outputs is None or len(outputs) != graph_calls:
                return False
            return all(
                one is not None
                and torch.allclose(one[1], residual_ref, rtol=1e-2, atol=1e-2)
                and torch.allclose(one[0], norm_ref, rtol=1e-2, atol=1e-2)
                for one, residual_ref, norm_ref in zip(
                    outputs, residual_refs, norm_refs
                )
            )

        reset_fn = reset_inputs
        use_comm_capture = True
    elif case == "unfused":

        def fn():
            outputs = []
            for inp, residual, weight, norm_out in zip(
                inputs, residuals, weights, norm_outs
            ):
                residual_out = comm.fused_allreduce_residual(inp, residual)
                if residual_out is None:
                    outputs.append(None)
                    continue
                outputs.append(
                    (residual_out, rmsnorm(residual_out, weight, 1e-6, out=norm_out))
                )
            return outputs[0] if graph_calls == 1 else outputs

        def check_fn(out):
            outputs = normalize_outputs(out)
            if outputs is None or len(outputs) != graph_calls:
                return False
            return all(
                one is not None
                and isinstance(one[0], torch.Tensor)
                and isinstance(one[1], torch.Tensor)
                and torch.allclose(one[0], residual_ref, rtol=1e-2, atol=1e-2)
                and torch.allclose(one[1], norm_ref, rtol=1e-2, atol=1e-2)
                for one, residual_ref, norm_ref in zip(
                    outputs, residual_refs, norm_refs
                )
            )

        reset_fn = reset_inputs
        use_comm_capture = True
    else:
        raise ValueError(case)

    graph_fn = make_replay_graph_fn(
        fn,
        check_group,
        check_fn=check_fn,
        reset_fn=reset_fn,
        capture_ctx_fn=comm.capture if use_comm_capture else None,
        register_graph_buffers_fn=(
            comm.register_graph_buffers if use_comm_capture else None
        ),
    )
    if graph_fn is None:
        return {
            "tp": args.world_size,
            "rank": rank,
            "hidden": hidden,
            "token": rows,
            "graph_calls": graph_calls,
            "case": case,
            "status": "graph_check_failed",
            "avg_kernel_us": "",
            "best_kernel_us": "",
            "best_kernel_median_us": "",
            "best_kernel_count": "",
            "best_kernel": "",
            "top_kernels": "",
            "trace": str(trace_path),
        }

    dist.barrier(group=data_group)
    avg_s = bench_kineto(
        graph_fn,
        kernel_names=KERNEL_SUBSTRINGS,
        num_tests=args.num_tests,
        suppress_kineto_output=True,
        trace_path=str(trace_path),
        flush_l2=False,
        with_multiple_kernels=True,
    )
    sync()
    dist.barrier(group=data_group)

    if rank != 0:
        return {}
    if isinstance(avg_s, tuple):
        avg_kernel_us = min((x for x in avg_s if x > 0), default=0.0) * 1e6
    else:
        avg_kernel_us = float(avg_s) * 1e6
    best_name, best_min, best_median, best_count, top_text = summarize_trace(trace_path)
    category_stats = summarize_trace_by_category(trace_path)
    fused_stats = category_stats.get("fused")
    ar_stats = category_stats.get("ar")
    rms_stats = category_stats.get("rms")
    unfused_sum_min = ar_stats[0] + rms_stats[0] if ar_stats and rms_stats else 0.0
    return {
        "tp": args.world_size,
        "rank": rank,
        "hidden": hidden,
        "token": rows,
        "graph_calls": graph_calls,
        "case": case,
        "status": "ok" if best_name else "no_kernel_events",
        "avg_kernel_us": f"{avg_kernel_us:.3f}" if avg_kernel_us else "",
        "best_kernel_us": f"{best_min:.3f}" if best_name else "",
        "best_kernel_median_us": f"{best_median:.3f}" if best_name else "",
        "best_kernel_count": str(best_count) if best_name else "",
        "best_kernel": best_name,
        "top_kernels": top_text,
        "fused_min_us": f"{fused_stats[0]:.3f}" if fused_stats else "",
        "fused_median_us": f"{fused_stats[1]:.3f}" if fused_stats else "",
        "fused_kernel": fused_stats[3] if fused_stats else "",
        "ar_min_us": f"{ar_stats[0]:.3f}" if ar_stats else "",
        "ar_median_us": f"{ar_stats[1]:.3f}" if ar_stats else "",
        "ar_kernel": ar_stats[3] if ar_stats else "",
        "rms_min_us": f"{rms_stats[0]:.3f}" if rms_stats else "",
        "rms_median_us": f"{rms_stats[1]:.3f}" if rms_stats else "",
        "rms_kernel": rms_stats[3] if rms_stats else "",
        "unfused_sum_min_us": f"{unfused_sum_min:.3f}" if unfused_sum_min else "",
        "trace": str(trace_path),
    }


SUMMARY_COLUMNS = (
    "tp",
    "rank",
    "hidden",
    "token",
    "graph_calls",
    "case",
    "status",
    "avg_kernel_us",
    "best_kernel_us",
    "best_kernel_median_us",
    "best_kernel_count",
    "best_kernel",
    "top_kernels",
    "fused_min_us",
    "fused_median_us",
    "fused_kernel",
    "ar_min_us",
    "ar_median_us",
    "ar_kernel",
    "rms_min_us",
    "rms_median_us",
    "rms_kernel",
    "unfused_sum_min_us",
    "trace",
)

COMPARE_COLUMNS = (
    "tp",
    "hidden",
    "token",
    "graph_calls",
    "status",
    "fused_min_us",
    "unfused_ar_min_us",
    "unfused_rms_min_us",
    "unfused_sum_min_us",
    "saving_us",
    "rms_saved_pct",
    "speedup",
    "winner",
    "fused_median_us",
    "unfused_ar_median_us",
    "unfused_rms_median_us",
    "fused_trace",
    "unfused_trace",
)


def append_row(path: Path, row: dict, columns=SUMMARY_COLUMNS) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_compare_row(fused: dict, unfused: dict) -> dict:
    status = (
        "ok" if fused.get("status") == "ok" and unfused.get("status") == "ok" else "bad"
    )
    fused_min = float(fused.get("fused_min_us") or 0.0)
    ar_min = float(unfused.get("ar_min_us") or 0.0)
    rms_min = float(unfused.get("rms_min_us") or 0.0)
    unfused_sum = ar_min + rms_min
    saving = unfused_sum - fused_min
    rms_saved_pct = saving / rms_min * 100.0 if rms_min > 0 else 0.0
    speedup = unfused_sum / fused_min if fused_min > 0 else 0.0
    winner = "fused" if saving > 0 else "unfused"
    if status != "ok" or fused_min <= 0 or unfused_sum <= 0:
        winner = "n/a"
    return {
        "tp": fused["tp"],
        "hidden": fused["hidden"],
        "token": fused["token"],
        "graph_calls": fused.get("graph_calls", ""),
        "status": status,
        "fused_min_us": f"{fused_min:.3f}" if fused_min else "",
        "unfused_ar_min_us": f"{ar_min:.3f}" if ar_min else "",
        "unfused_rms_min_us": f"{rms_min:.3f}" if rms_min else "",
        "unfused_sum_min_us": f"{unfused_sum:.3f}" if unfused_sum else "",
        "saving_us": f"{saving:.3f}" if fused_min and unfused_sum else "",
        "rms_saved_pct": f"{rms_saved_pct:.1f}" if fused_min and unfused_sum else "",
        "speedup": f"{speedup:.3f}" if speedup else "",
        "winner": winner,
        "fused_median_us": fused.get("fused_median_us", ""),
        "unfused_ar_median_us": unfused.get("ar_median_us", ""),
        "unfused_rms_median_us": unfused.get("rms_median_us", ""),
        "fused_trace": fused.get("trace", ""),
        "unfused_trace": unfused.get("trace", ""),
    }


def print_case_row(row: dict) -> None:
    prefix = (
        f"CASE tp={row['tp']} h={row['hidden']} token={row['token']} "
        f"calls={row.get('graph_calls', 1)} {row['case']} status={row['status']}"
    )
    if row["case"] == "fused":
        detail = (
            f" fused_min={fmt_us(row.get('fused_min_us'))}"
            f" fused_med={fmt_us(row.get('fused_median_us'))}"
        )
    else:
        detail = (
            f" ar_min={fmt_us(row.get('ar_min_us'))}"
            f" rms_min={fmt_us(row.get('rms_min_us'))}"
            f" unfused_sum_min={fmt_us(row.get('unfused_sum_min_us'))}"
        )
    print(prefix + detail + f" trace={row['trace']}", flush=True)


def print_compare_row(row: dict) -> None:
    print(
        "COMPARE "
        f"tp={row['tp']} h={row['hidden']} token={row['token']} "
        f"calls={row.get('graph_calls', 1)} "
        f"fused={fmt_us(row['fused_min_us'])} "
        f"unfused={fmt_us(row['unfused_sum_min_us'])}"
        f"(ar={fmt_us(row['unfused_ar_min_us'])}+rms={fmt_us(row['unfused_rms_min_us'])}) "
        f"save={fmt_us(row['saving_us'])} rms_saved={row['rms_saved_pct']}% "
        f"speedup={row['speedup']}x "
        f"winner={row['winner']}",
        flush=True,
    )


def worker(rank: int, args: argparse.Namespace, port: int) -> None:
    torch.musa.set_device(rank)
    init_distributed_environment(
        backend=args.backend,
        world_size=args.world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    ranks = list(range(args.world_size))
    data_group = dist.new_group(ranks, backend=args.backend)
    check_group = dist.new_group(ranks, backend="gloo")
    device = torch.device(f"musa:{rank}")
    comm = MusaJitCustomAllreduce(data_group, device, max_size=args.max_size)
    try:
        completed: dict[tuple[int, int], dict[str, dict]] = {}
        for hidden in args.hidden_values:
            for rows in args.tokens:
                for case in args.cases:
                    trace_path = args.output_dir / (
                        f"tp{args.world_size}_rank{rank}_h{hidden}_m{rows}"
                        f"_calls{args.graph_calls}_{case}.json"
                    )
                    row = bench_graph_case(
                        comm,
                        rank,
                        rows,
                        hidden,
                        args,
                        device,
                        data_group,
                        check_group,
                        case,
                        trace_path,
                    )
                    if rank == 0:
                        append_row(args.output, row)
                        print_case_row(row)
                        key = (hidden, rows)
                        completed.setdefault(key, {})[case] = row
                        pair = completed[key]
                        if "fused" in pair and "unfused" in pair:
                            compare_row = build_compare_row(
                                pair["fused"], pair["unfused"]
                            )
                            append_row(
                                args.compare_output, compare_row, COMPARE_COLUMNS
                            )
                            print_compare_row(compare_row)
    finally:
        comm.close()
        dist.destroy_process_group(check_group)
        dist.destroy_process_group(data_group)
        destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--hidden-values", default="4096")
    parser.add_argument("--tokens", default="1,4,16,64")
    parser.add_argument("--cases", default="fused,unfused")
    parser.add_argument("--graph-calls", type=int, default=1)
    parser.add_argument("--backend", default="mccl")
    parser.add_argument("--num-tests", type=int, default=12)
    parser.add_argument("--max-size", type=int, default=8 * 1024 * 1024 * 1024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("custom_ar_graph_kineto"),
    )
    args = parser.parse_args()
    args.hidden_values = parse_ints(args.hidden_values)
    args.tokens = parse_ints(args.tokens)
    args.cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    if args.graph_calls < 1:
        raise ValueError("--graph-calls must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output = args.output_dir / f"tp{args.world_size}_summary.tsv"
    args.compare_output = args.output_dir / f"tp{args.world_size}_compare.tsv"

    print(
        f"Writing raw summary to {args.output}; comparison summary to {args.compare_output}",
        flush=True,
    )

    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args, open_port()), nprocs=args.world_size)


if __name__ == "__main__":
    main()
