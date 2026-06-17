#!/usr/bin/env python3
"""Benchmark MUSA custom all-reduce eager and graph paths.

Example:
    PYTHONPATH=/data/sglang/python MUSA_VISIBLE_DEVICES=4,5 \
    python benchmark/musa/bench_custom_allreduce_graph.py \
      --world-size 2 --hidden 2048 --tokens 1 1024 32768
"""

import argparse
import socket
from contextlib import nullcontext
from statistics import median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_musa  # noqa: F401

from sglang.srt.distributed import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    MusaJitCustomAllreduce,
)


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
    return nbytes * (2 * (world_size - 1) / world_size) / (ms / 1e3) / 1e9


def all_ok(ok, group):
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def close_comm(comm):
    if hasattr(comm, "close"):
        comm.close()


def make_graph_fn(comm, inp, ref, group):
    graph = torch.musa.MUSAGraph()
    capture_ctx = comm.capture() if hasattr(comm, "capture") else nullcontext()
    # Match SGLang's graph_capture ordering: the communicator capture context
    # wraps the actual graph capture, so sgl-kernel CA can register graph inputs
    # in capture().__exit__ after the graph has recorded them.
    with capture_ctx:
        if isinstance(comm, MusaJitCustomAllreduce):
            comm._rank_data_for_input(inp)
        else:
            comm.custom_all_reduce(inp)
        with torch.musa.graph(graph):
            graph_out = comm.custom_all_reduce(inp)
    sync()
    graph.replay()
    sync()
    if not all_ok(torch.allclose(graph_out, ref, rtol=1e-2, atol=1e-2), group):
        return None
    return graph.replay


def bench_comm(name, comm_cls, cpu_group, device, inp, ref, args):
    comm = comm_cls(cpu_group, device, max_size=args.max_size)
    try:
        return bench_existing_comm(name, comm, cpu_group, inp, ref, args)
    finally:
        close_comm(comm)


def bench_existing_comm(name, comm, cpu_group, inp, ref, args):
    if comm.disabled or not comm.should_custom_ar(inp):
        return [
            (f"{name}_{mode}_SKIP", None)
            for mode in ("eager", "graph")
            if mode in args.modes
        ]

    out = comm.custom_all_reduce(inp)
    if not all_ok(torch.allclose(out, ref, rtol=1e-2, atol=1e-2), cpu_group):
        return [
            (f"{name}_eager_FAIL", None),
            *([(f"{name}_graph_SKIP", None)] if "graph" in args.modes else []),
        ]

    results = []
    if "eager" in args.modes:
        eager_samples = [
            event_ms(
                lambda: comm.custom_all_reduce(inp), args.iters, args.warmup, cpu_group
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


def bench_torch_allreduce(inp, ref, args):
    work = inp.clone()
    dist.all_reduce(work, group=dist.group.WORLD)
    if not torch.allclose(work, ref, rtol=1e-2, atol=1e-2):
        return [("torch_ar_eager_FAIL", None)]

    def run():
        work.copy_(inp)
        dist.all_reduce(work, group=dist.group.WORLD)

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
        return [("torch_ar_eager", eager_samples[0])]
    return [
        ("torch_ar_eager_med", median(eager_samples)),
        ("torch_ar_eager_min", min(eager_samples)),
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
    dtype = torch.bfloat16

    if rank == 0:
        print(
            f"world_size={args.world_size} hidden={args.hidden} dtype=bf16 "
            f"iters={args.iters} warmup={args.warmup}"
        )
        print(
            f"{'tokens':>8} {'bytes':>12} {'name':<18} "
            f"{'lat_ms':>10} {'algBW_GB/s':>12} {'busBW_GB/s':>12}"
        )

    reusable_comms = {}
    if args.reuse_comm:
        if "sgl" in args.impls:
            reusable_comms["sgl"] = CustomAllreduce(
                cpu_group, device, max_size=args.max_size
            )
        if "jit" in args.impls:
            reusable_comms["jit"] = MusaJitCustomAllreduce(
                cpu_group, device, max_size=args.max_size
            )

    try:
        for tokens in args.tokens:
            numel = tokens * args.hidden
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            inp = torch.full(
                (tokens, args.hidden), rank + 1, dtype=dtype, device=device
            )
            ref = inp.clone()
            dist.all_reduce(ref, group=dist.group.WORLD)

            results = []
            for impl in args.impls:
                if impl == "torch":
                    results.extend(bench_torch_allreduce(inp, ref, args))
                elif impl == "sgl":
                    if args.reuse_comm:
                        results.extend(
                            bench_existing_comm(
                                "sgl_ca",
                                reusable_comms["sgl"],
                                cpu_group,
                                inp,
                                ref,
                                args,
                            )
                        )
                    else:
                        results.extend(
                            bench_comm(
                                "sgl_ca",
                                CustomAllreduce,
                                cpu_group,
                                device,
                                inp,
                                ref,
                                args,
                            )
                        )
                elif impl == "jit":
                    if args.reuse_comm:
                        results.extend(
                            bench_existing_comm(
                                "jit_ca",
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
                                "jit_ca",
                                MusaJitCustomAllreduce,
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
                print("", flush=True)
    finally:
        for comm in reusable_comms.values():
            close_comm(comm)

    dist.destroy_process_group(cpu_group)
    destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--backend", default="mccl")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--print-samples",
        action="store_true",
        help="Print each repeat sample in addition to median/min.",
    )
    parser.add_argument("--max-size", type=int, default=512 * 1024 * 1024)
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
        choices=("torch", "sgl", "jit"),
        default=("torch", "sgl", "jit"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eager", "graph"),
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
