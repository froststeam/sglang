import unittest

from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=300, suite="stage-a-test-1-gpu-musa-smoke")


class TestMusaServerSmoke1GPU(musa_server_smoke.MusaServerSmokeTest):
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="qwen-dense-1gpu",
        model_env="MUSA_SMOKE_QWEN_DENSE_MODEL",
        tp_size=1,
        default_gsm8k_min_score=0.92,
        default_gsm8k_chat_template_kwargs='{"enable_thinking": false}',
        default_gsm8k_reasoning_effort="none",
        extra_args_env="MUSA_SMOKE_QWEN_DENSE_EXTRA_ARGS",
    )


if __name__ == "__main__":
    unittest.main()
