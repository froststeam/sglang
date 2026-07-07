import unittest

from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=1800, suite="stage-a-test-4-gpu-musa-joyai-llm-flash-smoke")


class TestMusaServerSmokeJoyAILLMFlash4GPU(musa_server_smoke.MusaServerSmokeTest):
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="joyai-llm-flash-tp2-dp2",
        model_env="MUSA_SMOKE_JOYAI_LLM_FLASH_MODEL",
        tp_size=2,
        default_gsm8k_min_score=0.90,
        extra_args_env="MUSA_SMOKE_JOYAI_LLM_FLASH_EXTRA_ARGS",
    )


if __name__ == "__main__":
    unittest.main()
