"""Unit tests for the EPLB forward-pass schedule."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

import types
import unittest

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.test.test_utils import CustomTestCase


class TestEPLBManagerSchedule(CustomTestCase):
    def test_rebalance_keeps_periodic_forward_end_schedule(self):
        """A rebalance must not get stuck waiting for an external busy flag."""
        manager = EPLBManager.__new__(EPLBManager)
        manager._rebalance_num_iterations = 3
        rebalance_calls = []

        def rebalance(_self):
            rebalance_calls.append(1)
            yield

        manager.rebalance = types.MethodType(rebalance, manager)
        manager.reset_generator()

        for _ in range(8):
            manager.on_forward_pass_end()

        self.assertEqual(len(rebalance_calls), 2)

    def test_forward_end_has_no_disaggregation_busy_argument(self):
        """EPLB advances after every forward, including prefill extend forwards."""
        manager = EPLBManager.__new__(EPLBManager)
        manager._rebalance_num_iterations = 1
        manager.rebalance = types.MethodType(lambda _self: iter((None,)), manager)
        manager.reset_generator()

        manager.on_forward_pass_end()
        manager.on_forward_pass_end()


if __name__ == "__main__":
    unittest.main()
