import unittest

from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=420, suite="stage-a-test-2-gpu-musa-smoke")


class TestMusaServerSmoke2GPU(musa_server_smoke.MusaServerSmokeTest):
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="qwen-dense-tp2",
        model_env="MUSA_SMOKE_QWEN_DENSE_TP_MODEL",
        tp_size=2,
        default_gsm8k_min_score=0.94,
        default_gsm8k_chat_template_kwargs='{"enable_thinking": false}',
        default_gsm8k_reasoning_effort="none",
        extra_args_env="MUSA_SMOKE_QWEN_DENSE_TP_EXTRA_ARGS",
    )


if __name__ == "__main__":
    unittest.main()
