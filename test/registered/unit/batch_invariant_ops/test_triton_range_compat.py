import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import triton.language as tl

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    _get_device_multi_processor_count,
    _enable_triton_range_flatten_compat,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestTritonRangeFlattenCompat(unittest.TestCase):
    def test_installed_triton_accepts_flatten(self):
        iterator = tl.range(0, 8, 1, flatten=True)
        self.assertEqual(getattr(iterator.start, "value", iterator.start), 0)
        self.assertEqual(getattr(iterator.end, "value", iterator.end), 8)
        self.assertEqual(getattr(iterator.step, "value", iterator.step), 1)

    def test_legacy_range_accepts_and_ignores_flatten(self):
        class LegacyRange:
            def __init__(self, start, end=None, step=None):
                self.args = (start, end, step)

        self.assertTrue(_enable_triton_range_flatten_compat(LegacyRange))
        self.assertIn("flatten", inspect.signature(LegacyRange).parameters)
        self.assertEqual(LegacyRange(0, 8, 1, flatten=True).args, (0, 8, 1))

    def test_native_flatten_implementation_is_untouched(self):
        class NativeRange:
            def __init__(self, start, end=None, step=None, flatten=False):
                self.flatten = flatten

        original_init = NativeRange.__init__
        self.assertFalse(_enable_triton_range_flatten_compat(NativeRange))
        self.assertIs(NativeRange.__init__, original_init)
        self.assertTrue(NativeRange(0, 8, 1, flatten=True).flatten)

    def test_non_musa_device_uses_shared_core_count_helper(self):
        xpu_device = SimpleNamespace(type="xpu")

        with patch.object(
            batch_invariant_ops, "get_device_core_count", return_value=64
        ) as get_core_count:
            self.assertEqual(_get_device_multi_processor_count(xpu_device), 64)

        get_core_count.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
