import unittest

from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(
    est_time=900,
    suite="stage-b-test-2-gpu-musa-qwen35-35b-a3b-temp06-smoke",
)


class TestMusaServerSmokeQwen3535BA3BTemp062GPU(
    musa_server_smoke.MusaServerSmokeTest
):
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="qwen35-35b-a3b-temp06-tp2",
        model_env="MUSA_SMOKE_QWEN35_35B_A3B_TEMP06_MODEL",
        tp_size=2,
        default_gsm8k_min_score=0.85,
        default_gsm8k_chat_template_kwargs='{"enable_thinking": false}',
        default_gsm8k_reasoning_effort="none",
        extra_args_env="MUSA_SMOKE_QWEN35_35B_A3B_TEMP06_EXTRA_ARGS",
    )


if __name__ == "__main__":
    unittest.main()
