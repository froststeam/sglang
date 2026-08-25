# SPDX-License-Identifier: Apache-2.0

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


def open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def check_all_ranks(ok: bool, group):
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    assert bool(flag.item())


def worker(rank, world_size, port):
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
    comm = MusaJitCustomAllToAll(cpu_group, device)
    assert not comm.disabled

    try:
        sizes = (2048, 64 * 1024, 2 * 1024 * 1024) * 3 + (32 * 1024 * 1024,)
        for call, numel in enumerate(sizes):
            inp = torch.empty(numel, dtype=torch.bfloat16, device=device)
            chunk = numel // world_size
            for dst in range(world_size):
                inp.narrow(0, dst * chunk, chunk).fill_(rank * 100 + dst + call)
            expected = torch.empty_like(inp)
            dist.all_to_all_single(expected, inp)
            actual = comm.custom_all_to_all(inp)
            assert actual is not None
            torch.musa.synchronize()
            check_all_ranks(torch.equal(actual, expected), cpu_group)

        fallback = torch.ones(2048, dtype=torch.float16, device=device)
        assert comm.custom_all_to_all(fallback) is None

        invalid_output = torch.empty(
            (1, world_size + 1, 24 // world_size, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        assert comm.custom_ulysses(invalid_output, 2, input_layout=False) is None

        batch, local_sequence, global_heads, dim = 1, 8192, 24, 128
        local_heads = global_heads // world_size
        ulysses_input = (
            torch.arange(batch * local_sequence * global_heads * dim, device=device)
            .remainder(1024)
            .to(torch.bfloat16)
            .view(batch, local_sequence, global_heads, dim)
            .add_(rank * 2048)
        )
        packed_input = ulysses_input.permute(2, 0, 1, 3).contiguous()
        packed_output = torch.empty_like(packed_input)
        dist.all_to_all_single(packed_output, packed_input)
        expected_input = (
            packed_output.view(world_size, local_heads, batch, local_sequence, dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
            .view(
                batch,
                local_sequence * world_size,
                local_heads,
                dim,
            )
        )
        actual_input = comm.custom_ulysses(ulysses_input, 2, input_layout=True)
        assert actual_input is not None
        torch.musa.synchronize()
        check_all_ranks(torch.equal(actual_input, expected_input), cpu_group)

        qkv_actual = comm.custom_qkv_ulysses(
            ulysses_input, ulysses_input.add(1), ulysses_input.add(2)
        )
        assert qkv_actual is not None
        torch.musa.synchronize()
        for offset, actual in enumerate(qkv_actual):
            check_all_ranks(torch.equal(actual, expected_input.add(offset)), cpu_group)

        packed_output = expected_input.permute(1, 0, 2, 3).contiguous()
        inverse_output = torch.empty_like(packed_output)
        dist.all_to_all_single(inverse_output, packed_output)
        expected_output = (
            inverse_output.view(world_size, local_sequence, batch, local_heads, dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
            .view(batch, local_sequence, global_heads, dim)
        )
        actual_output = comm.custom_ulysses(expected_input, 2, input_layout=False)
        assert actual_output is not None
        torch.musa.synchronize()
        check_all_ranks(torch.equal(actual_output, expected_output), cpu_group)

        prefix = torch.full(
            (batch, 64, local_heads, dim),
            rank,
            dtype=torch.bfloat16,
            device=device,
        )
        gathered_prefix = [torch.empty_like(prefix) for _ in range(world_size)]
        dist.all_gather(gathered_prefix, prefix)
        expected_combined = torch.cat(
            [torch.cat(gathered_prefix, dim=2), expected_output], dim=1
        )
        actual_combined = comm.custom_ulysses_prefix_output(prefix, expected_input)
        assert actual_combined is not None
        torch.musa.synchronize()
        check_all_ranks(torch.equal(actual_combined, expected_combined), cpu_group)
    finally:
        comm.close()
        dist.destroy_process_group(cpu_group)
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_musa_custom_all_to_all(world_size):
    if torch.musa.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} MUSA devices")
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(world_size, open_port()), nprocs=world_size, join=True)
