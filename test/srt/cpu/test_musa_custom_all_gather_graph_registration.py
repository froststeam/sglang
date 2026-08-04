import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def _torch_stubs_if_needed():
    try:
        import torch

        return {}
    except ModuleNotFoundError:
        torch = ModuleType("torch")
        torch.__path__ = []
        torch.Tensor = object
        torch.device = object
        torch.version = SimpleNamespace(musa=None)
        torch.get_device_module = lambda: None

        distributed = ModuleType("torch.distributed")
        distributed.ProcessGroup = object
        torch.distributed = distributed
        return {"torch": torch, "torch.distributed": distributed}


def _load_custom_all_gather_module():
    cuda_wrapper_name = "sglang.srt.distributed.device_communicators.cuda_wrapper"
    cuda_wrapper = ModuleType(cuda_wrapper_name)
    cuda_wrapper.CudaRTLibrary = object
    cuda_wrapper.cudaIpcMemHandle_t = object

    module_path = (
        Path(__file__).parents[3]
        / "python/sglang/srt/distributed/device_communicators/custom_all_gather.py"
    )
    spec = importlib.util.spec_from_file_location(
        "musa_custom_all_gather_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import_stubs = {cuda_wrapper_name: cuda_wrapper}
    import_stubs.update(_torch_stubs_if_needed())
    with patch.dict(sys.modules, import_stubs):
        spec.loader.exec_module(module)
    return module


custom_all_gather = _load_custom_all_gather_module()
MusaJitCustomAllGather = custom_all_gather.MusaJitCustomAllGather


class FakeTensor:
    def __init__(
        self,
        ptr: int,
        *,
        shape: tuple[int, ...] = (8,),
        dtype: str = "float16",
        element_size: int = 2,
    ):
        self._ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self._element_size = element_size

    def data_ptr(self) -> int:
        return self._ptr

    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def element_size(self) -> int:
        return self._element_size


class FakeRankData:
    def __init__(self, pointers):
        self._pointers = tuple(pointers)

    def tolist(self):
        return list(self._pointers)


class TestMusaCustomAllGatherGraphRegistration(unittest.TestCase):
    @staticmethod
    def _make_comm():
        comm = object.__new__(MusaJitCustomAllGather)
        comm.group = object()
        comm.rank = 0
        comm.world_size = 2
        comm._graph_inputs = []
        comm._graph_registered_input_sequence = []
        comm._graph_registered_sequence_signature = ()
        comm._graph_registered_cursor = 0
        comm._graph_registered_miss = False
        comm._rank_data_cache = {}
        comm._rank_data_refs = {}
        comm._last_input_ptr = None
        comm._last_rank_data = None
        comm._rank_data_for_input = Mock(
            side_effect=AssertionError("graph registration used eager cache API")
        )
        comm._local_ipc_record_for_input = Mock(
            side_effect=lambda inp: (
                inp.data_ptr(),
                f"local:{inp.data_ptr()}".encode(),
                0,
            )
        )
        comm._rank_data_from_ipc_records = Mock(
            side_effect=lambda _ptr, records: FakeRankData(
                [record[0] for record in records] + [0] * 6
            )
        )
        return comm

    def test_registration_preserves_order_and_detects_peer_pointer_change(self):
        comm = self._make_comm()
        inp = FakeTensor(0x1000)
        peer_ptrs = [0x2000, 0x2000]
        local_entries = []

        def broadcast_entries(holder, src, group, device):
            self.assertIs(group, comm.group)
            self.assertEqual(device, "cpu")
            if src == 0:
                local_entries[:] = holder[0]
                return
            holder[0] = [
                (
                    peer_ptrs[index],
                    f"peer:{peer_ptrs[index]}".encode(),
                    0,
                    entry[3],
                )
                for index, entry in enumerate(local_entries)
            ]

        with (
            patch.object(
                custom_all_gather.dist,
                "get_process_group_ranks",
                return_value=[0, 1],
                create=True,
            ),
            patch.object(
                custom_all_gather.dist,
                "broadcast_object_list",
                side_effect=broadcast_entries,
                create=True,
            ),
        ):
            comm._graph_inputs = [inp, inp]
            self.assertEqual(comm.register_graph_buffers(), 1)
            self.assertEqual(len(comm._graph_registered_input_sequence), 2)
            self.assertEqual(comm._local_ipc_record_for_input.call_count, 2)

            comm._graph_inputs = [inp, inp]
            self.assertEqual(comm.register_graph_buffers(), 0)

            peer_ptrs[1] = 0x3000
            comm._graph_inputs = [inp, inp]
            self.assertEqual(comm.register_graph_buffers(), 1)

        comm._rank_data_for_input.assert_not_called()
        self.assertEqual(comm._graph_inputs, [])
        self.assertFalse(comm._graph_registered_miss)

    def test_registration_rejects_rank_entry_count_mismatch(self):
        comm = self._make_comm()
        comm._graph_inputs = [FakeTensor(0x1000)]

        def broadcast_entries(holder, src, group, device):
            self.assertIs(group, comm.group)
            self.assertEqual(device, "cpu")
            if src == 1:
                holder[0] = []

        with (
            patch.object(
                custom_all_gather.dist,
                "get_process_group_ranks",
                return_value=[0, 1],
                create=True,
            ),
            patch.object(
                custom_all_gather.dist,
                "broadcast_object_list",
                side_effect=broadcast_entries,
                create=True,
            ),
            self.assertRaisesRegex(RuntimeError, "rank entry counts"),
        ):
            comm.register_graph_buffers()

        self.assertEqual(comm._graph_inputs, [])

    def test_registration_rejects_rank_signature_mismatch(self):
        comm = self._make_comm()
        inp = FakeTensor(0x1000)
        comm._graph_inputs = [inp]
        local_entries = []

        def broadcast_entries(holder, src, group, device):
            self.assertIs(group, comm.group)
            self.assertEqual(device, "cpu")
            if src == 0:
                local_entries[:] = holder[0]
                return
            peer_entry = list(local_entries[0])
            peer_entry[0] = 0x2000
            peer_entry[3] = ((16,), "float16", 16, 2)
            holder[0] = [tuple(peer_entry)]

        with (
            patch.object(
                custom_all_gather.dist,
                "get_process_group_ranks",
                return_value=[0, 1],
                create=True,
            ),
            patch.object(
                custom_all_gather.dist,
                "broadcast_object_list",
                side_effect=broadcast_entries,
                create=True,
            ),
            self.assertRaisesRegex(RuntimeError, "order mismatch"),
        ):
            comm.register_graph_buffers()

        self.assertEqual(comm._graph_inputs, [])

    def test_graph_launch_miss_requests_recapture_then_uses_registered_path(self):
        comm = self._make_comm()
        inp = FakeTensor(0x1000)
        other_inp = FakeTensor(0x2000)
        output = object()
        rank_data = FakeRankData([0x1000, 0x3000, 0, 0, 0, 0, 0, 0])
        comm._IS_CAPTURING = True
        signature = comm._graph_input_signature(inp)
        comm._graph_registered_input_sequence = [
            (signature, inp.data_ptr(), rank_data)
        ]
        comm.should_custom_ag = Mock(return_value=True)
        comm._launch_registered = Mock()
        comm._launch_unregistered = Mock()

        device_module = SimpleNamespace(is_current_stream_capturing=lambda: True)
        with patch.object(
            custom_all_gather.torch,
            "get_device_module",
            return_value=device_module,
            create=True,
        ):
            self.assertIs(comm.custom_all_gather(output, inp), output)
            comm._launch_registered.assert_called_once_with(rank_data, output, inp)
            comm._launch_unregistered.assert_not_called()

            comm.prepare_graph_capture()
            comm._graph_inputs.clear()
            comm._launch_registered.reset_mock()
            comm._launch_unregistered.reset_mock()
            self.assertIs(comm.custom_all_gather(output, other_inp), output)

        comm._launch_registered.assert_not_called()
        comm._launch_unregistered.assert_called_once_with(output, other_inp)
        self.assertTrue(comm._graph_registered_miss)
        self.assertEqual(comm._graph_inputs, [other_inp])

    def test_registration_lifecycle_clears_capture_only_state(self):
        comm = self._make_comm()
        comm._graph_inputs = [object()]
        comm._graph_registered_input_sequence = [object()]
        comm._graph_registered_sequence_signature = ((object(), object()),)
        comm._graph_registered_cursor = 1
        comm._graph_registered_miss = True

        comm.begin_graph_capture_registration()
        self.assertEqual(comm._graph_inputs, [])
        self.assertEqual(comm._graph_registered_input_sequence, [])
        self.assertEqual(comm._graph_registered_sequence_signature, ())
        self.assertEqual(comm._graph_registered_cursor, 0)
        self.assertFalse(comm._graph_registered_miss)

        comm._graph_inputs.append(object())
        comm._graph_registered_input_sequence.append(object())
        comm._graph_registered_sequence_signature = ((object(), object()),)
        comm._graph_registered_cursor = 1
        comm._graph_registered_miss = True
        comm.end_graph_capture_registration()
        self.assertEqual(comm._graph_inputs, [])
        self.assertEqual(comm._graph_registered_input_sequence, [])
        self.assertEqual(comm._graph_registered_sequence_signature, ())
        self.assertEqual(comm._graph_registered_cursor, 0)
        self.assertFalse(comm._graph_registered_miss)

        comm._graph_registered_cursor = 1
        comm._graph_registered_miss = True
        comm.prepare_graph_capture()
        self.assertEqual(comm._graph_registered_cursor, 0)
        self.assertFalse(comm._graph_registered_miss)
        comm._graph_registered_cursor = 1
        comm.prepare_graph_replay()
        self.assertEqual(comm._graph_registered_cursor, 0)
        self.assertTrue(comm.requires_graph_capture_registration_recapture)

    def test_capture_context_resets_miss_and_registers_on_exit(self):
        comm = self._make_comm()
        comm.disabled = False
        comm._IS_CAPTURING = False
        comm._graph_inputs = [object()]
        comm._graph_registered_input_sequence = [object()]
        comm._graph_registered_sequence_signature = ((object(), object()),)
        comm._graph_registered_cursor = 1
        comm._graph_registered_miss = True
        comm.register_graph_buffers = Mock(return_value=1)

        with comm.capture():
            self.assertTrue(comm._IS_CAPTURING)
            self.assertEqual(comm._graph_inputs, [])
            self.assertEqual(comm._graph_registered_input_sequence, [])
            self.assertEqual(comm._graph_registered_sequence_signature, ())
            self.assertEqual(comm._graph_registered_cursor, 0)
            self.assertFalse(comm._graph_registered_miss)
            comm._graph_inputs.append(object())

        self.assertFalse(comm._IS_CAPTURING)
        self.assertEqual(comm._graph_inputs, [])
        comm.register_graph_buffers.assert_called_once_with()

    def test_close_clears_graph_registration_state(self):
        comm = self._make_comm()
        comm.disabled = False
        comm._opened_ipc_ptrs = {}
        comm._rank_data_cache = {0x1000: object()}
        comm._rank_data_refs = {0x1000: object()}
        comm._graph_inputs = [object()]
        comm._graph_registered_input_sequence = [object()]
        comm._graph_registered_sequence_signature = ((object(), object()),)
        comm._graph_registered_cursor = 1
        comm._graph_registered_miss = True
        comm.buffer_ptrs = [0x2000]
        comm.meta_ptrs = [0x3000]

        with (
            patch.object(
                custom_all_gather.dist,
                "is_initialized",
                return_value=True,
                create=True,
            ),
            patch.object(custom_all_gather, "_free_shared_buffer") as free_buffer,
        ):
            comm.close()

        self.assertTrue(comm.disabled)
        self.assertEqual(comm._graph_inputs, [])
        self.assertEqual(comm._graph_registered_input_sequence, [])
        self.assertEqual(comm._graph_registered_sequence_signature, ())
        self.assertEqual(comm._graph_registered_cursor, 0)
        self.assertFalse(comm._graph_registered_miss)
        self.assertEqual(free_buffer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
