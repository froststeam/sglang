# SPDX-License-Identifier: Apache-2.0

import argparse
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _measure(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def _worker(rank, world_size, port, sizes_mib, qwen_local_sequence, warmup, iterations):
    torch.musa.set_device(rank)

    from sglang.srt.distributed import (
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from sglang.srt.distributed.device_communicators.custom_all_to_all import (
        MusaJitCustomAllToAll,
    )

    init_distributed_environment(
        backend="mccl",
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
    device = torch.device(f"musa:{rank}")
    custom = MusaJitCustomAllToAll(cpu_group, device)

    try:
        if rank == 0:
            print(
                f"world_size={world_size}, warmup={warmup}, iterations={iterations}",
                flush=True,
            )
        for size_mib in sizes_mib:
            numel = size_mib * 1024 * 1024 // torch.bfloat16.itemsize
            input_ = torch.randn(numel, dtype=torch.bfloat16, device=device)
            mccl_output = torch.empty_like(input_)

            def run_mccl():
                dist.all_to_all_single(mccl_output, input_)

            def run_custom():
                assert custom.custom_all_to_all(input_) is not None

            dist.barrier(group=cpu_group)
            mccl_ms = _measure(run_mccl, warmup, iterations)
            dist.barrier(group=cpu_group)
            custom_ms = _measure(run_custom, warmup, iterations)
            measurements = [None] * world_size if rank == 0 else None
            dist.gather_object(
                (mccl_ms, custom_ms),
                measurements,
                dst=0,
                group=cpu_group,
            )
            if rank == 0:
                mccl_max = max(item[0] for item in measurements)
                custom_max = max(item[1] for item in measurements)
                print(
                    f"{size_mib:>3} MiB: MCCL {mccl_max:.3f} ms, "
                    f"custom {custom_max:.3f} ms, "
                    f"speedup {mccl_max / custom_max:.2f}x",
                    flush=True,
                )

        if qwen_local_sequence:
            shape = (1, qwen_local_sequence, 24, 128)
            query = torch.randn(shape, dtype=torch.bfloat16, device=device)
            key = torch.randn_like(query)
            value = torch.randn_like(query)

            def run_mccl_qkv():
                for tensor in (query, key, value):
                    packed = tensor.permute(2, 0, 1, 3).contiguous()
                    output = torch.empty_like(packed)
                    dist.all_to_all_single(output, packed)
                    (
                        output.view(
                            world_size,
                            24 // world_size,
                            1,
                            qwen_local_sequence,
                            128,
                        )
                        .permute(2, 0, 3, 1, 4)
                        .contiguous()
                    )

            def run_three():
                assert custom.custom_ulysses(query, 2, True) is not None
                assert custom.custom_ulysses(key, 2, True) is not None
                assert custom.custom_ulysses(value, 2, True) is not None

            def run_qkv():
                assert custom.custom_qkv_ulysses(query, key, value) is not None

            sharded = torch.randn(
                (1, qwen_local_sequence * world_size, 24 // world_size, 128),
                dtype=torch.bfloat16,
                device=device,
            )

            def run_mccl_output():
                packed = sharded.permute(1, 0, 2, 3).contiguous()
                output = torch.empty_like(packed)
                dist.all_to_all_single(output, packed)
                (
                    output.view(
                        world_size,
                        qwen_local_sequence,
                        1,
                        24 // world_size,
                        128,
                    )
                    .permute(2, 1, 0, 3, 4)
                    .contiguous()
                )

            def run_custom_output():
                assert custom.custom_ulysses(sharded, 2, False) is not None

            dist.barrier(group=cpu_group)
            mccl_qkv_ms = _measure(run_mccl_qkv, warmup, iterations)
            dist.barrier(group=cpu_group)
            three_ms = _measure(run_three, warmup, iterations)
            dist.barrier(group=cpu_group)
            qkv_ms = _measure(run_qkv, warmup, iterations)
            dist.barrier(group=cpu_group)
            mccl_output_ms = _measure(run_mccl_output, warmup, iterations)
            dist.barrier(group=cpu_group)
            custom_output_ms = _measure(run_custom_output, warmup, iterations)
            measurements = [None] * world_size if rank == 0 else None
            dist.gather_object(
                (
                    mccl_qkv_ms,
                    three_ms,
                    qkv_ms,
                    mccl_output_ms,
                    custom_output_ms,
                ),
                measurements,
                dst=0,
                group=cpu_group,
            )
            if rank == 0:
                mccl_qkv_max = max(item[0] for item in measurements)
                three_max = max(item[1] for item in measurements)
                qkv_max = max(item[2] for item in measurements)
                mccl_output_max = max(item[3] for item in measurements)
                custom_output_max = max(item[4] for item in measurements)
                print(
                    f"QKV S={qwen_local_sequence}: MCCL {mccl_qkv_max:.3f} ms, "
                    f"custom-three {three_max:.3f} ms, "
                    f"custom-fused {qkv_max:.3f} ms, "
                    f"speedup {mccl_qkv_max / qkv_max:.2f}x",
                    flush=True,
                )
                print(
                    f"Output S={qwen_local_sequence}: MCCL {mccl_output_max:.3f} ms, "
                    f"custom {custom_output_max:.3f} ms, "
                    f"speedup {mccl_output_max / custom_output_max:.2f}x",
                    flush=True,
                )
    finally:
        custom.close()
        dist.destroy_process_group(cpu_group)
        destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--qwen-local-sequence", type=int, default=0)
    parser.add_argument(
        "--sizes-mib", type=int, nargs="+", default=[16, 32, 48, 50, 64]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    world_size = args.world_size
    if torch.musa.device_count() < world_size:
        raise RuntimeError(f"This benchmark requires {world_size} MUSA devices")
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        _worker,
        args=(
            world_size,
            _open_port(),
            args.sizes_mib,
            args.qwen_local_sequence,
            args.warmup,
            args.iterations,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
