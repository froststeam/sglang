# SPDX-License-Identifier: Apache-2.0

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch_musa = pytest.importorskip("torch_musa")  # noqa: F401

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="MUSA device is not available.",
)


DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.int64)
SHAPES = ((1, 2048), (3, 2048), (16, 1024))


def open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def tensor_ok(actual, expected):
    if actual.dtype.is_floating_point:
        return torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)
    return torch.equal(actual, expected)


def make_expected(inp, world_size):
    return torch.cat(
        [torch.full_like(inp, src + 1) for src in range(world_size)], dim=0
    )


def check_all_ranks(ok, group):
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    assert bool(flag.item())


def graph_runner(comm, inp, out):
    graph = torch.musa.MUSAGraph()
    with comm.capture():
        comm.prepare_registered_input(inp)
        with torch.musa.graph(graph):
            comm.custom_all_gather(out, inp)
    torch.musa.synchronize()
    return graph.replay


def worker(rank, world_size, port):
    os.environ["SGLANG_MUSA_USE_JIT_ALL_GATHER"] = "1"
    torch.musa.set_device(rank)

    from sglang.srt.distributed import (
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from sglang.srt.distributed.device_communicators.custom_all_gather import (
        MusaJitCustomAllGather,
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
    comm = MusaJitCustomAllGather(cpu_group, device)
    assert not comm.disabled

    try:
        for dtype in DTYPES:
            for shape in SHAPES:
                inp = torch.full(shape, rank + 1, dtype=dtype, device=device)
                expected = make_expected(inp, world_size)
                out = torch.empty_like(expected)

                assert comm.should_custom_ag(out, inp)
                comm.custom_all_gather(out, inp)
                torch.musa.synchronize()
                check_all_ranks(tensor_ok(out, expected), cpu_group)

                assert comm.prepare_registered_input(inp)
                registered_out = torch.empty_like(expected)
                assert comm.custom_all_gather_registered(registered_out, inp) is not None
                torch.musa.synchronize()
                check_all_ranks(tensor_ok(registered_out, expected), cpu_group)

                graph_out = torch.empty_like(expected)
                replay = graph_runner(comm, inp, graph_out)
                replay()
                torch.musa.synchronize()
                check_all_ranks(tensor_ok(graph_out, expected), cpu_group)

        fallback_inp = torch.full((1, 1), rank + 1, dtype=torch.bfloat16, device=device)
        fallback_out = torch.empty(
            (world_size, 1), dtype=fallback_inp.dtype, device=device
        )
        assert not comm.should_custom_ag(fallback_out, fallback_inp)
        dist.all_gather_into_tensor(fallback_out, fallback_inp, group=dist.group.WORLD)
        torch.musa.synchronize()
        check_all_ranks(
            tensor_ok(fallback_out, make_expected(fallback_inp, world_size)),
            cpu_group,
        )

        if world_size >= 4:
            large_inp = torch.full(
                (1025, 2048), rank + 1, dtype=torch.bfloat16, device=device
            )
            large_out = torch.empty(
                (world_size * large_inp.shape[0], large_inp.shape[1]),
                dtype=large_inp.dtype,
                device=device,
            )
            assert comm.should_custom_ag(large_out, large_inp)
            comm.custom_all_gather(large_out, large_inp)
            torch.musa.synchronize()
            check_all_ranks(
                tensor_ok(large_out, make_expected(large_inp, world_size)),
                cpu_group,
            )
    finally:
        comm.close()
        dist.destroy_process_group(cpu_group)
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4])
def test_musa_custom_all_gather(world_size):
    if torch.musa.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} MUSA devices")
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(world_size, open_port()), nprocs=world_size, join=True)
