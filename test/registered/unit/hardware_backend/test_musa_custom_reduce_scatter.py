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


def test_custom_reduce_scatter_predicate_rejects_unaligned_input():
    comm = make_predicate_only_communicator()
    storage = torch.empty(8 * 16 + 1, dtype=torch.float32)
    inp = storage[1:].view(8, 16)
    output = torch.empty((2, 16), dtype=torch.float32)

    assert inp.is_contiguous()
    assert int(inp.data_ptr()) % 16 != 0
    assert not comm.should_custom_rs(output, inp)


def test_graph_registration_env_keeps_ar_compatibility_and_rs_precedence(
    monkeypatch,
):
    names = (
        "SGLANG_MUSA_CUSTOM_RS_GRAPH_REGISTERED_INPUT",
        "SGL_CUSTOM_RS_GRAPH_REGISTERED_INPUT",
        "SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT",
        "SGL_CUSTOM_AR_GRAPH_REGISTERED_INPUT",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    assert custom_reduce_scatter._use_graph_registered_input()

    monkeypatch.setenv("SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT", "0")
    assert not custom_reduce_scatter._use_graph_registered_input()

    monkeypatch.setenv("SGLANG_MUSA_CUSTOM_RS_GRAPH_REGISTERED_INPUT", "1")
    assert custom_reduce_scatter._use_graph_registered_input()


def make_graph_registration_communicator():
    comm = object.__new__(custom_reduce_scatter.MusaJitCustomReduceScatter)
    comm.disabled = False
    comm.group = object()
    comm.device = torch.device("cpu")
    comm.rank = 0
    comm.world_size = 1
    comm._IS_CAPTURING = False
    comm._graph_registered_input_enabled = True
    comm.requires_graph_capture_registration_recapture = True
    comm._graph_inputs = []
    comm._graph_registered_input_sequence = []
    comm._graph_registered_sequence_signature = ()
    comm._graph_registered_cursor = 0
    comm._graph_registered_miss = False
    comm._graph_rank_data_pools = []
    comm._graph_capture_rank_data_slots = None
    comm._rank_data_cache = {}
    comm._last_input_ptr = None
    comm._last_rank_data = None
    comm._opened_ipc_ptrs = {}
    comm._local_ipc_record_for_input = lambda inp: (
        int(inp.data_ptr()),
        b"local",
        0,
    )
    comm._rank_data_from_ipc_records = lambda ptr, records: torch.tensor(
        [ptr] + [0] * 7, dtype=torch.int64
    )
    comm._graph_object_broadcast_device = lambda: "cpu"
    return comm


def _patch_single_rank_registration(monkeypatch):
    class DeviceModule:
        @staticmethod
        def is_current_stream_capturing():
            return True

        @staticmethod
        def synchronize():
            return None

    monkeypatch.setattr(torch, "get_device_module", lambda: DeviceModule, raising=False)
    monkeypatch.setattr(
        custom_reduce_scatter.dist,
        "get_process_group_ranks",
        lambda group: [0],
    )
    monkeypatch.setattr(
        custom_reduce_scatter.dist,
        "broadcast_object_list",
        lambda holder, **kwargs: None,
    )


def _capture_registered_input(comm, inp):
    comm.prepare_graph_capture()
    comm._IS_CAPTURING = True
    try:
        return comm._rank_data_for_registered_input(inp)
    finally:
        comm._IS_CAPTURING = False


def test_capture_context_drops_stale_registered_sequence():
    comm = make_graph_registration_communicator()
    comm._graph_registered_input_sequence = [(object(), object(), object())]
    comm._graph_registered_sequence_signature = ((object(), object()),)

    with comm.capture():
        assert comm._graph_registered_input_sequence == []
        assert comm._graph_registered_sequence_signature == ()


def test_graph_registration_first_capture_recaptures_then_reuses_stable_slot(
    monkeypatch,
):
    _patch_single_rank_registration(monkeypatch)
    comm = make_graph_registration_communicator()
    inp = torch.empty((2, 8), dtype=torch.float32)

    comm.begin_graph_capture_registration()
    assert _capture_registered_input(comm, inp) is None
    assert comm.register_graph_buffers() == 1

    pool = comm._graph_capture_rank_data_slots
    assert tuple(pool.shape) == (1, 8)
    pool_ptr = int(pool.data_ptr())

    rank_data_slot = _capture_registered_input(comm, inp)
    assert int(rank_data_slot.data_ptr()) == int(pool[0].data_ptr())
    assert comm.register_graph_buffers() == 0
    assert int(comm._graph_capture_rank_data_slots.data_ptr()) == pool_ptr

    comm.end_graph_capture_registration()
    assert comm._graph_capture_rank_data_slots is None
    assert len(comm._graph_rank_data_pools) == 1
    assert int(comm._graph_rank_data_pools[0].data_ptr()) == pool_ptr

    monkeypatch.setattr(custom_reduce_scatter.dist, "is_initialized", lambda: False)
    comm.close()
    assert not comm._graph_rank_data_pools
    assert not comm.requires_graph_capture_registration_recapture


def test_graph_capture_routes_first_attempt_to_staging_then_registered(
    monkeypatch,
):
    _patch_single_rank_registration(monkeypatch)
    comm = make_graph_registration_communicator()
    inp = torch.empty((2, 8), dtype=torch.float32)
    output = torch.empty((1, 8), dtype=torch.float32)
    launches = []
    comm.should_custom_rs = lambda out, input_: True
    comm._launch_unregistered = lambda out, input_: launches.append("staging")
    comm._launch_registered = lambda rank_data, out: launches.append(
        ("registered", int(rank_data.data_ptr()))
    )

    comm.begin_graph_capture_registration()
    comm.prepare_graph_capture()
    comm._IS_CAPTURING = True
    try:
        assert comm.custom_reduce_scatter(output, inp) is output
    finally:
        comm._IS_CAPTURING = False
    assert launches == ["staging"]
    assert comm.register_graph_buffers() == 1

    expected_slot_ptr = int(comm._graph_capture_rank_data_slots[0].data_ptr())
    comm.prepare_graph_capture()
    comm._IS_CAPTURING = True
    try:
        assert comm.custom_reduce_scatter(output, inp) is output
    finally:
        comm._IS_CAPTURING = False
    assert launches == ["staging", ("registered", expected_slot_ptr)]
    assert comm.register_graph_buffers() == 0
    comm.end_graph_capture_registration()


def test_graph_registration_miss_requests_another_recapture(monkeypatch):
    _patch_single_rank_registration(monkeypatch)
    comm = make_graph_registration_communicator()
    first = torch.empty((2, 8), dtype=torch.float32)
    replacement = torch.empty_like(first)

    comm.begin_graph_capture_registration()
    assert _capture_registered_input(comm, first) is None
    assert comm.register_graph_buffers() == 1

    assert _capture_registered_input(comm, replacement) is None
    assert comm._graph_registered_miss
    assert comm.register_graph_buffers() == 1

    assert _capture_registered_input(comm, replacement) is not None
    assert comm.register_graph_buffers() == 0
    comm.end_graph_capture_registration()


def test_graph_registration_keeps_replaced_pool_alive(monkeypatch):
    _patch_single_rank_registration(monkeypatch)
    comm = make_graph_registration_communicator()
    first = torch.empty((2, 8), dtype=torch.float32)
    second = torch.empty_like(first)

    comm.begin_graph_capture_registration()
    assert _capture_registered_input(comm, first) is None
    assert comm.register_graph_buffers() == 1
    old_pool = comm._graph_capture_rank_data_slots

    comm.prepare_graph_capture()
    comm._IS_CAPTURING = True
    try:
        assert comm._rank_data_for_registered_input(first) is not None
        assert comm._rank_data_for_registered_input(second) is None
    finally:
        comm._IS_CAPTURING = False
    assert comm.register_graph_buffers() == 1
    assert tuple(comm._graph_capture_rank_data_slots.shape) == (2, 8)
    assert any(pool is old_pool for pool in comm._graph_rank_data_pools)
    comm.end_graph_capture_registration()


def test_reduce_scatter_launchers_keep_d3_and_add_registered_d3(monkeypatch):
    from sglang.srt.hardware_backend.musa.jit_kernel.csrc import reduce_scatter

    unregistered = object()
    registered = object()

    class Module:
        sgl_musa_custom_rs_launch_unregistered_chunked = unregistered
        sgl_musa_custom_rs_launch_registered_chunked = registered

    monkeypatch.setattr(reduce_scatter, "_custom_rs_module", lambda world_size: Module)

    assert reduce_scatter.launch_d3_func(8) is unregistered
    assert reduce_scatter.launch_registered_d3_func(8) is registered
