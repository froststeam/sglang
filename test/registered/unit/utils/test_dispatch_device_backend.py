from unittest.mock import patch

import pytest

from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def test_musa_uses_privateuse1_dispatch_key():
    with (
        patch.object(common, "is_cuda_alike", return_value=False),
        patch.object(common, "is_musa", return_value=True),
        patch.object(common, "is_xpu", return_value=False),
    ):
        assert common.get_dispatch_device_backend() == "PrivateUse1"


def test_cuda_dispatch_key_is_unchanged():
    with patch.object(common, "is_cuda_alike", return_value=True):
        assert common.get_dispatch_device_backend() == "CUDA"


def test_unsupported_device_raises():
    with (
        patch.object(common, "is_cuda_alike", return_value=False),
        patch.object(common, "is_musa", return_value=False),
        patch.object(common, "is_xpu", return_value=False),
        pytest.raises(RuntimeError, match="CUDA/MUSA/XPU"),
    ):
        common.get_dispatch_device_backend()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
