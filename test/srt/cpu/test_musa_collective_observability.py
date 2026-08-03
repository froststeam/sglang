import logging
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators import (
    musa_collective_observability as collective_observability,
)
from sglang.srt.distributed.device_communicators.custom_all_gather import (
    MusaJitCustomAllGather,
)
from sglang.srt.distributed.device_communicators.custom_reduce_scatter import (
    MusaJitCustomReduceScatter,
)


class FakeDevice:
    def __init__(self, type_, index=0):
        self.type = type_
        self.index = index

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and (self.type, self.index) == (
            other.type,
            other.index,
        )


class FakeStorage:
    def __init__(self, ptr):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        device,
        dtype=torch.float16,
        ptr=16,
        storage_ptr=None,
        contiguous=True,
        layout=torch.strided,
    ):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self._ptr = ptr
        self._storage = FakeStorage(ptr if storage_ptr is None else storage_ptr)
        self._contiguous = contiguous
        self.layout = layout

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()

    def data_ptr(self):
        return self._ptr

    def is_contiguous(self):
        return self._contiguous

    def untyped_storage(self):
        return self._storage


def _group(cag_comm=None, crs_comm=None):
    group = SimpleNamespace(
        group_name="attention_tp",
        unique_name="attention_tp:test",
        ranks=(0, 1),
        world_size=2,
        cag_comm=cag_comm,
        crs_comm=crs_comm,
        pynccl_comm=None,
        device_group=object(),
    )
    group._should_use_pynccl_tensor_collective = lambda: False
    return group


def _summary_count(phase, op, route, reason):
    return collective_observability.route_summary().get(
        f"{phase}|{op}|{route}|{reason}|attention_tp|0,1|2",
        0,
    )


class TestMusaCollectiveObservability(unittest.TestCase):
    def setUp(self):
        self.enabled = patch.object(
            collective_observability, "enabled", return_value=True
        )
        self.max_events = patch.object(
            collective_observability, "_max_events", return_value=100
        )
        self.enabled.start()
        self.max_events.start()
        collective_observability._event_counts.clear()
        collective_observability._event_count = 0

    def tearDown(self):
        self.max_events.stop()
        self.enabled.stop()
        collective_observability._event_counts.clear()
        collective_observability._event_count = 0

    def test_custom_all_gather_reason(self):
        comm = object.__new__(MusaJitCustomAllGather)
        comm.disabled = False
        comm.device = FakeDevice("musa")
        comm.world_size = 2
        comm.max_size = 1024

        musa_input = FakeTensor((8,), device=FakeDevice("musa"))
        musa_output = FakeTensor((16,), device=FakeDevice("musa"))
        cpu_input = FakeTensor((8,), device=FakeDevice("cpu"))

        self.assertEqual(
            comm.custom_ag_reason(musa_output, cpu_input), "non_musa_tensor"
        )
        self.assertEqual(comm.custom_ag_reason(musa_output, musa_input), "eligible")
        self.assertTrue(comm.should_custom_ag(musa_output, musa_input))

    def test_custom_reduce_scatter_reason(self):
        comm = object.__new__(MusaJitCustomReduceScatter)
        comm.disabled = False
        comm.device = FakeDevice("musa")
        comm.world_size = 2
        comm.rank = 0
        comm.max_size = 1024

        output = FakeTensor((8,), device=FakeDevice("musa"))
        input = FakeTensor((16,), device=FakeDevice("musa"), ptr=32)
        wrong_shape_input = FakeTensor((8,), device=FakeDevice("musa"), ptr=32)

        self.assertEqual(
            comm.custom_rs_reason(output, wrong_shape_input), "shape_mismatch"
        )
        self.assertEqual(comm.custom_rs_reason(output, input), "eligible")
        self.assertTrue(comm.should_custom_rs(output, input))

    def test_all_gather_records_custom_route_only_after_success(self):
        output = torch.empty(16)
        input = torch.ones(8)
        fallback = Mock()

        class CagComm:
            disabled = False

            @staticmethod
            def custom_ag_reason(output, input):
                return "eligible"

            @staticmethod
            def custom_all_gather(output, input):
                return output

        group = _group(cag_comm=CagComm())
        with patch.object(torch.distributed, "all_gather_into_tensor", fallback):
            parallel_state.GroupCoordinator._all_gather_into_tensor(
                group, output, input
            )

        self.assertFalse(fallback.called)
        self.assertEqual(
            _summary_count("unscoped", "all_gather", "jit_all_gather", "eligible"),
            1,
        )

    def test_all_gather_records_fallback_when_custom_launcher_declines(self):
        output = torch.empty(16)
        input = torch.ones(8)
        fallback = Mock()

        class CagComm:
            disabled = False

            @staticmethod
            def custom_ag_reason(output, input):
                return "eligible"

            @staticmethod
            def custom_all_gather(output, input):
                return None

        group = _group(cag_comm=CagComm())
        with patch.object(torch.distributed, "all_gather_into_tensor", fallback):
            parallel_state.GroupCoordinator._all_gather_into_tensor(
                group, output, input
            )

        fallback.assert_called_once_with(output, input, group=group.device_group)
        self.assertEqual(
            _summary_count(
                "unscoped",
                "all_gather",
                "torch_distributed",
                "custom_launcher_declined",
            ),
            1,
        )

    def test_reduce_scatter_records_custom_route_only_after_success(self):
        output = torch.empty(8)
        input = torch.ones(16)
        fallback = Mock()

        class CrsComm:
            disabled = False

            @staticmethod
            def custom_rs_reason(output, input):
                return "eligible"

            @staticmethod
            def custom_reduce_scatter(output, input):
                return output

        group = _group(crs_comm=CrsComm())
        with patch.object(torch.distributed, "reduce_scatter_tensor", fallback):
            parallel_state.GroupCoordinator._reduce_scatter_tensor(group, output, input)

        self.assertFalse(fallback.called)
        self.assertEqual(
            _summary_count(
                "unscoped", "reduce_scatter", "jit_reduce_scatter_d3", "eligible"
            ),
            1,
        )

    def test_reduce_scatter_records_fallback_after_custom_rejection(self):
        output = torch.empty(8)
        input = torch.ones(16)
        fallback = Mock()

        class CrsComm:
            disabled = False

            @staticmethod
            def custom_rs_reason(output, input):
                return "shape_mismatch"

        group = _group(crs_comm=CrsComm())
        with patch.object(torch.distributed, "reduce_scatter_tensor", fallback):
            parallel_state.GroupCoordinator._reduce_scatter_tensor(group, output, input)

        fallback.assert_called_once_with(output, input, group=group.device_group)
        self.assertEqual(
            _summary_count(
                "unscoped", "reduce_scatter", "torch_distributed", "shape_mismatch"
            ),
            1,
        )

    def test_public_reduce_scatter_routes_to_custom_communicator(self):
        output = torch.empty(8)
        input = torch.ones(16)
        group = _group(crs_comm=SimpleNamespace(disabled=False))
        group._should_use_musa_attn_tp_pynccl_collective = lambda: False
        group._should_use_musa_custom_reduce_scatter = lambda: (
            parallel_state.GroupCoordinator._should_use_musa_custom_reduce_scatter(
                group
            )
        )
        group._reduce_scatter_tensor = Mock()

        with (
            patch.object(parallel_state, "_is_musa", True),
            patch.object(parallel_state, "_is_npu", False),
            patch.object(parallel_state, "reg_reduce_scatter_tensor") as registered,
        ):
            parallel_state.GroupCoordinator.reduce_scatter_tensor(group, output, input)

        group._reduce_scatter_tensor.assert_called_once_with(output, input)
        self.assertFalse(registered.called)

    def test_public_all_gather_routes_to_custom_communicator(self):
        output = torch.empty(16)
        input = torch.ones(8)
        group = _group(cag_comm=SimpleNamespace(disabled=False))
        group._should_use_musa_attn_tp_pynccl_collective = lambda: False
        group._should_use_musa_custom_all_gather = lambda: (
            parallel_state.GroupCoordinator._should_use_musa_custom_all_gather(group)
        )
        group._all_gather_into_tensor = Mock()

        with (
            patch.object(parallel_state, "_is_musa", True),
            patch.object(parallel_state, "_is_npu", False),
            patch.object(parallel_state, "_is_xpu", False),
            patch.object(parallel_state, "reg_all_gather_into_tensor") as registered,
        ):
            parallel_state.GroupCoordinator.all_gather_into_tensor(group, output, input)

        group._all_gather_into_tensor.assert_called_once_with(output, input)
        self.assertFalse(registered.called)

    def test_capture_manifest_is_referenced_without_recording_replay_route(self):
        group = _group()
        output = torch.empty(16)
        input = torch.ones(8)

        with self.assertLogs(collective_observability.logger, logging.INFO) as logs:
            with collective_observability.phase_scope("target_verify"):
                with collective_observability.graph_capture_scope(
                    runner="UnitTestRunner", batch_size=1
                ) as capture:
                    collective_observability.record_route(
                        op="all_gather",
                        route="jit_all_gather",
                        reason="eligible",
                        group=group,
                        output=output,
                        input=input,
                    )
            manifest = capture.manifest
            before_replay = collective_observability.route_summary()
            collective_observability.record_graph_replay(
                manifest,
                runner="UnitTestRunner",
                graph_key="1",
                phase="target_verify",
            )

        self.assertEqual(collective_observability.route_summary(), before_replay)
        self.assertTrue(
            any("graph_capture_manifest" in message for message in logs.output)
        )
        self.assertTrue(any("graph_replay" in message for message in logs.output))


if __name__ == "__main__":
    unittest.main()
