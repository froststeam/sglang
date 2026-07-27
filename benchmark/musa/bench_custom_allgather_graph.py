#!/usr/bin/env python3
"""Benchmark MUSA custom all-gather eager and graph paths.

Example:
    PYTHONPATH=/data/sglang/python SGLANG_MUSA_USE_JIT_ALL_GATHER=1 \
    MUSA_VISIBLE_DEVICES=0,1 \
    python benchmark/musa/bench_custom_allgather_graph.py \
      --world-size 2 --hidden 2048 --tokens 1 1024 32768
"""

import argparse
import csv
import json
import os
import socket
from contextlib import nullcontext
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_musa  # noqa: F401

from sglang.srt.distributed import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from sglang.srt.distributed.device_communicators.custom_all_gather import (
    MusaJitCustomAllGather,
)


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}


def open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def sync():
    torch.musa.synchronize()


def event_ms(fn, iters, warmup, group):
    dist.barrier(group=group)
    for _ in range(warmup):
        fn()
    sync()
    dist.barrier(group=group)
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    sync()
    dist.barrier(group=group)
    return start.elapsed_time(end) / iters


def algbw_gbs(nbytes, ms):
    return nbytes / (ms / 1e3) / 1e9


def busbw_gbs(nbytes, world_size, ms):
    return nbytes * ((world_size - 1) / world_size) / (ms / 1e3) / 1e9


def split_result_name(name):
    if name.startswith("torch_ag_"):
        impl = "torch"
        mode = name[len("torch_ag_") :]
    elif name.startswith("jit_cag_"):
        impl = "jit"
        mode = name[len("jit_cag_") :]
    else:
        impl = name
        mode = ""
    return impl, mode


def env_value(*names):
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return ""


def output_row(args, tokens, nbytes, name, ms):
    impl, mode = split_result_name(name)
    return {
        "world_size": args.world_size,
        "hidden": args.hidden,
        "dtype": args.dtype,
        "tokens": tokens,
        "bytes": nbytes,
        "impl": impl,
        "mode": mode,
        "name": name,
        "lat_ms": "" if ms is None else f"{ms:.6f}",
        "algBW_GB_s": "" if ms is None else f"{algbw_gbs(nbytes, ms):.6f}",
        "busBW_GB_s": ""
        if ms is None
        else f"{busbw_gbs(nbytes, args.world_size, ms):.6f}",
        "threads": env_value("SGLANG_CUSTOM_AG_THREADS", "SGL_CUSTOM_AG_THREADS"),
        "blocks": env_value("SGLANG_CUSTOM_AG_BLOCKS", "SGL_CUSTOM_AG_BLOCKS"),
        "max_blocks": env_value(
            "SGLANG_CUSTOM_AG_MAX_BLOCKS", "SGL_CUSTOM_AG_MAX_BLOCKS"
        ),
        "atomic_barrier": env_value(
            "SGLANG_CUSTOM_AG_ATOMIC_BARRIER", "SGL_CUSTOM_AG_ATOMIC_BARRIER"
        ),
        "dynamic_blocks": env_value(
            "SGLANG_CUSTOM_AG_DYNAMIC_BLOCKS", "SGL_CUSTOM_AG_DYNAMIC_BLOCKS"
        ),
    }


def append_output_rows(args, rows):
    if not rows:
        return
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    if args.output_jsonl:
        path = Path(args.output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def tensor_ok(actual, expected):
    if actual.dtype.is_floating_point:
        return torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)
    return torch.equal(actual, expected)


def all_ok(ok, group):
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def close_comm(comm):
    if hasattr(comm, "close"):
        comm.close()


def make_ref(inp, world_size):
    ref = torch.empty(
        (inp.shape[0] * world_size,) + inp.shape[1:],
        dtype=inp.dtype,
        device=inp.device,
    )
    dist.all_gather_into_tensor(ref, inp, group=dist.group.WORLD)
    return ref


def make_graph_fn(comm, inp, ref, group):
    graph = torch.musa.MUSAGraph()
    graph_out = torch.empty_like(ref)
    capture_ctx = comm.capture() if hasattr(comm, "capture") else nullcontext()
    with capture_ctx:
        if isinstance(comm, MusaJitCustomAllGather):
            comm.prepare_registered_input(inp)
        with torch.musa.graph(graph):
            comm.custom_all_gather(graph_out, inp)
    sync()
    graph.replay()
    sync()
    if not all_ok(tensor_ok(graph_out, ref), group):
        return None
    return graph.replay


def make_registered_fn(comm, inp, ref, group):
    if not hasattr(comm, "prepare_registered_input") or not hasattr(
        comm, "custom_all_gather_registered"
    ):
        return None
    if not comm.prepare_registered_input(inp):
        return None

    registered_out = torch.empty_like(ref)
    out = comm.custom_all_gather_registered(registered_out, inp)
    sync()
    if out is None or not all_ok(tensor_ok(registered_out, ref), group):
        return None
    return lambda: comm.custom_all_gather_registered(registered_out, inp)


def bench_comm(name, comm_cls, cpu_group, device, inp, ref, args):
    comm = comm_cls(cpu_group, device, max_size=args.max_size)
    try:
        return bench_existing_comm(name, comm, cpu_group, inp, ref, args)
    finally:
        close_comm(comm)


def bench_existing_comm(name, comm, cpu_group, inp, ref, args):
    out = torch.empty_like(ref)
    if comm.disabled or not comm.should_custom_ag(out, inp):
        return [
            (f"{name}_{mode}_SKIP", None)
            for mode in ("eager", "registered", "graph")
            if mode in args.modes
        ]

    comm.custom_all_gather(out, inp)
    if not all_ok(tensor_ok(out, ref), cpu_group):
        return [
            (f"{name}_eager_FAIL", None),
            *([(f"{name}_graph_SKIP", None)] if "graph" in args.modes else []),
        ]

    results = []
    if "eager" in args.modes:
        eager_out = torch.empty_like(ref)
        eager_samples = [
            event_ms(
                lambda: comm.custom_all_gather(eager_out, inp),
                args.iters,
                args.warmup,
                cpu_group,
            )
            for _ in range(args.repeats)
        ]
        if args.repeats == 1:
            results.append((f"{name}_eager", eager_samples[0]))
        else:
            results.append((f"{name}_eager_med", median(eager_samples)))
            results.append((f"{name}_eager_min", min(eager_samples)))
            if args.print_samples:
                results.extend(
                    (f"{name}_eager_s{i}", sample)
                    for i, sample in enumerate(eager_samples)
                )

    if "registered" in args.modes:
        registered_fn = make_registered_fn(comm, inp, ref, cpu_group)
        registered_samples = None
        if registered_fn is not None:
            registered_samples = [
                event_ms(registered_fn, args.iters, args.warmup, cpu_group)
                for _ in range(args.repeats)
            ]
        if registered_samples is None:
            results.append((f"{name}_registered", None))
        elif args.repeats == 1:
            results.append((f"{name}_registered", registered_samples[0]))
        else:
            results.append((f"{name}_registered_med", median(registered_samples)))
            results.append((f"{name}_registered_min", min(registered_samples)))
            if args.print_samples:
                results.extend(
                    (f"{name}_registered_s{i}", sample)
                    for i, sample in enumerate(registered_samples)
                )

    if "graph" in args.modes:
        graph_fn = make_graph_fn(comm, inp, ref, cpu_group)
        graph_samples = None
        if graph_fn is not None:
            graph_samples = [
                event_ms(graph_fn, args.iters, args.warmup, cpu_group)
                for _ in range(args.repeats)
            ]
        if graph_samples is None:
            results.append((f"{name}_graph", None))
        elif args.repeats == 1:
            results.append((f"{name}_graph", graph_samples[0]))
        else:
            results.append((f"{name}_graph_med", median(graph_samples)))
            results.append((f"{name}_graph_min", min(graph_samples)))
            if args.print_samples:
                results.extend(
                    (f"{name}_graph_s{i}", sample)
                    for i, sample in enumerate(graph_samples)
                )
    return results


def bench_torch_allgather(inp, ref, args):
    out = torch.empty_like(ref)
    dist.all_gather_into_tensor(out, inp, group=dist.group.WORLD)
    if not tensor_ok(out, ref):
        return [("torch_ag_eager_FAIL", None)]

    eager_out = torch.empty_like(ref)

    def run():
        dist.all_gather_into_tensor(eager_out, inp, group=dist.group.WORLD)

    eager_samples = [
        event_ms(
            run,
            args.iters,
            args.warmup,
            dist.group.WORLD,
        )
        for _ in range(args.repeats)
    ]
    if args.repeats == 1:
        return [("torch_ag_eager", eager_samples[0])]
    return [
        ("torch_ag_eager_med", median(eager_samples)),
        ("torch_ag_eager_min", min(eager_samples)),
    ]


def worker(rank, args, port):
    torch.musa.set_device(rank)
    init_distributed_environment(
        backend=args.backend,
        world_size=args.world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    cpu_group = dist.new_group(list(range(args.world_size)), backend="gloo")
    device = torch.device(f"musa:{rank}")
    dtype = DTYPE_MAP[args.dtype]

    if rank == 0:
        print(
            f"world_size={args.world_size} hidden={args.hidden} "
            f"dtype={args.dtype} iters={args.iters} warmup={args.warmup}"
        )
        print(
            f"{'tokens':>8} {'bytes':>12} {'name':<18} "
            f"{'lat_ms':>10} {'algBW_GB/s':>12} {'busBW_GB/s':>12}"
        )

    reusable_comms = {}
    if args.reuse_comm and "jit" in args.impls:
        reusable_comms["jit"] = MusaJitCustomAllGather(
            cpu_group, device, max_size=args.max_size
        )

    output_rows = []
    try:
        for tokens in args.tokens:
            numel = tokens * args.hidden
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            inp = torch.full(
                (tokens, args.hidden), rank + 1, dtype=dtype, device=device
            )
            ref = make_ref(inp, args.world_size)

            results = []
            for impl in args.impls:
                if impl == "torch":
                    results.extend(bench_torch_allgather(inp, ref, args))
                elif impl == "jit":
                    if args.reuse_comm:
                        results.extend(
                            bench_existing_comm(
                                "jit_cag",
                                reusable_comms["jit"],
                                cpu_group,
                                inp,
                                ref,
                                args,
                            )
                        )
                    else:
                        results.extend(
                            bench_comm(
                                "jit_cag",
                                MusaJitCustomAllGather,
                                cpu_group,
                                device,
                                inp,
                                ref,
                                args,
                            )
                        )
            if rank == 0:
                for name, ms in results:
                    if ms is None:
                        print(
                            f"{tokens:8d} {nbytes:12d} {name:<18} "
                            f"{'NA':>10} {'NA':>12} {'NA':>12}"
                        )
                    else:
                        print(
                            f"{tokens:8d} {nbytes:12d} {name:<18} "
                            f"{ms:10.4f} {algbw_gbs(nbytes, ms):12.2f} "
                            f"{busbw_gbs(nbytes, args.world_size, ms):12.2f}"
                        )
                    output_rows.append(output_row(args, tokens, nbytes, name, ms))
                print("", flush=True)
    finally:
        if rank == 0:
            append_output_rows(args, output_rows)
        for comm in reusable_comms.values():
            close_comm(comm)

    dist.destroy_process_group(cpu_group)
    destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP.keys()),
        default="bf16",
    )
    parser.add_argument("--backend", default="mccl")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--print-samples",
        action="store_true",
        help="Print each repeat sample in addition to median/min.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Append rank-0 benchmark rows to this CSV file.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Append rank-0 benchmark rows to this JSONL file.",
    )
    parser.add_argument(
        "--output-overwrite",
        action="store_true",
        help="Delete output CSV/JSONL before running.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help=(
            "Override communicator max input bytes. By default the communicator "
            "uses its production threshold; pass 536870912 to characterize large "
            "payloads up to 512 MiB."
        ),
    )
    parser.add_argument(
        "--reuse-comm",
        action="store_true",
        help="Reuse one communicator per implementation across payload sizes.",
    )
    parser.add_argument(
        "--isolate-impls",
        action="store_true",
        help="Run each implementation in a fresh worker group to avoid order effects.",
    )
    parser.add_argument(
        "--impls",
        nargs="+",
        choices=("torch", "jit"),
        default=("torch", "jit"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eager", "registered", "graph"),
        default=("eager", "graph"),
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
        ],
    )
    args = parser.parse_args()
    if args.output_overwrite:
        for path in (args.output_csv, args.output_jsonl):
            if path:
                Path(path).unlink(missing_ok=True)
    mp.set_start_method("spawn", force=True)
    if args.isolate_impls and len(args.impls) > 1:
        impls = tuple(args.impls)
        for impl in impls:
            impl_args = argparse.Namespace(**vars(args))
            impl_args.impls = (impl,)
            mp.spawn(
                worker,
                args=(impl_args, open_port()),
                nprocs=impl_args.world_size,
                join=True,
            )
    else:
        mp.spawn(worker, args=(args, open_port()), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
