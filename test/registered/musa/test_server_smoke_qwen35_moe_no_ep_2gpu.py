import unittest

from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=600, suite="stage-a-test-2-gpu-musa-qwen35-moe-no-ep-smoke")


class TestMusaServerSmokeQwen35MoeNoEP2GPU(musa_server_smoke.MusaServerSmokeTest):
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="qwen35-moe-tp2-no-ep",
        model_env="MUSA_SMOKE_QWEN_MOE_TP_MODEL",
        tp_size=2,
        default_gsm8k_min_score=0.90,
        default_gsm8k_chat_template_kwargs='{"enable_thinking": false}',
        default_gsm8k_reasoning_effort="none",
        extra_args_env="MUSA_SMOKE_QWEN_MOE_TP_EXTRA_ARGS",
    )


if __name__ == "__main__":
    unittest.main()
