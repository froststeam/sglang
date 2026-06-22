import unittest

from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=600, suite="stage-a-test-4-gpu-musa-smoke")


class TestMusaServerSmoke4GPU(musa_server_smoke.MusaServerSmokeTest):
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="qwen-moe-tp4",
        model_env="MUSA_SMOKE_QWEN_MOE_MODEL",
        tp_size=4,
        default_gsm8k_min_score=0.94,
        extra_args_env="MUSA_SMOKE_QWEN_MOE_EXTRA_ARGS",
    )


if __name__ == "__main__":
    unittest.main()
