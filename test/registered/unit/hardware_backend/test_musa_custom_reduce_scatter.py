import pytest
import torch

from sglang.srt.distributed.device_communicators import custom_reduce_scatter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def test_custom_reduce_scatter_gate_is_independent_from_all_gather(monkeypatch):
    monkeypatch.setattr(custom_reduce_scatter, "_is_musa", True)
    monkeypatch.setenv("SGLANG_MUSA_USE_JIT_ALL_GATHER", "1")
    monkeypatch.delenv("SGLANG_MUSA_USE_JIT_REDUCE_SCATTER", raising=False)

    assert custom_reduce_scatter.dispatch_custom_reduce_scatter() is None

    monkeypatch.setenv("SGLANG_MUSA_USE_JIT_ALL_GATHER", "0")
    monkeypatch.setenv("SGLANG_MUSA_USE_JIT_REDUCE_SCATTER", "1")

    assert (
        custom_reduce_scatter.dispatch_custom_reduce_scatter()
        is custom_reduce_scatter.MusaJitCustomReduceScatter
    )


def make_predicate_only_communicator(rank=1, world_size=4, max_size=1 << 20):
    comm = object.__new__(custom_reduce_scatter.MusaJitCustomReduceScatter)
    comm.disabled = False
    comm.device = torch.device("cpu")
    comm.rank = rank
    comm.world_size = world_size
    comm.max_size = max_size
    return comm


def test_custom_reduce_scatter_predicate_accepts_rank_shard_alias():
    comm = make_predicate_only_communicator()
    inp = torch.empty((8, 16), dtype=torch.float32)
    output = inp.tensor_split(comm.world_size)[comm.rank]

    assert comm.should_custom_rs(output, inp)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda output, inp: (torch.empty((2, 16), dtype=torch.int32), inp),
        lambda output, inp: (output, torch.empty((7, 16), dtype=torch.float32)),
        lambda output, inp: (inp.tensor_split(4)[0], inp),
    ],
)
def test_custom_reduce_scatter_predicate_rejects_unsupported_inputs(mutate):
    comm = make_predicate_only_communicator()
    inp = torch.empty((8, 16), dtype=torch.float32)
    output = inp.tensor_split(comm.world_size)[comm.rank]
    output, inp = mutate(output, inp)

    assert not comm.should_custom_rs(output, inp)


def test_custom_reduce_scatter_predicate_rejects_oversized_input():
    comm = make_predicate_only_communicator(max_size=16)
    inp = torch.empty((8, 16), dtype=torch.float32)
    output = torch.empty((2, 16), dtype=torch.float32)

    assert not comm.should_custom_rs(output, inp)
