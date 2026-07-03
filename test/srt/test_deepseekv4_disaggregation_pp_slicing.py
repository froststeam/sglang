import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender


def _manager(start_layer, end_layer, ratios):
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(
        prefill_start_layer=start_layer,
        prefill_end_layer=end_layer,
        mla_compression_ratios=ratios,
    )
    return mgr


class TestDeepSeekV4DisaggregationPPSlicing(unittest.TestCase):
    def test_bucketed_kv_layout_slicing(self):
        ratios = [0, 4, 128, 4, 0, 128]
        mgr = _manager(1, 4, ratios)

        src = [10, 11, 12, 13, 14]
        dst = [100, 101, 200, 201, 300, 301]

        sliced_src, sliced_dst, num_layers = mgr.get_mla_kv_ptrs_with_pp(src, dst)

        self.assertEqual(sliced_src, src)
        self.assertEqual(sliced_dst, [100, 101, 200, 201, 300])
        self.assertEqual(num_layers, len(src))

    def test_fast_path_keeps_matching_layout(self):
        ratios = [0, 4, 128, 4]
        mgr = _manager(1, 3, ratios)

        src = [10, 11, 12]
        dst = [100, 101, 102]

        sliced_src, sliced_dst, num_layers = mgr.get_mla_kv_ptrs_with_pp(src, dst)

        self.assertEqual(sliced_src, src)
        self.assertEqual(sliced_dst, dst)
        self.assertEqual(num_layers, len(src))

    def test_deepseek_v4_state_layout_slicing(self):
        ratios = [0, 4, 128, 4, 0, 128]
        mgr = _manager(1, 4, ratios)

        src = [10, 11, 12, 13, 14, 15, 16, 17]
        dst = [100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 300, 301]

        sliced_src, sliced_dst, num_layers = mgr.get_mla_kv_ptrs_with_pp(src, dst)

        self.assertEqual(sliced_src, src)
        self.assertEqual(sliced_dst, [101, 102, 103, 200, 201, 202, 300, 301])
        self.assertEqual(num_layers, len(src))

    def test_invalid_stage_range_fails(self):
        ratios = [0, 4, 128]
        mgr = _manager(1, 4, ratios)

        with self.assertRaises(AssertionError):
            mgr.get_mla_kv_ptrs_with_pp([10, 11], [100, 101, 200, 300])

    def test_record_transfer_indices_counts_component_lists(self):
        sender = CommonKVSender.__new__(CommonKVSender)
        sender._transfer_num_kv_indices = 0
        sender._transfer_num_state_indices = 0

        sender._record_transfer_indices([1, 2, 3], [[4], None, [5, 6]])

        self.assertEqual(sender._transfer_num_kv_indices, 3)
        self.assertEqual(sender._transfer_num_state_indices, 3)


if __name__ == "__main__":
    unittest.main()
