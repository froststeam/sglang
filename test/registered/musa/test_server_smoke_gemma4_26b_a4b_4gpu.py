import unittest

from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(
    est_time=600,
    suite="stage-a-test-4-gpu-musa-gemma4-26b-a4b-it-smoke",
)


class TestMusaServerSmokeGemma426BA4B4GPU(musa_server_smoke.MusaServerSmokeTest):
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="gemma4-26b-a4b-it-tp4",
        model_env="MUSA_SMOKE_GEMMA4_26B_A4B_MODEL",
        tp_size=4,
        default_gsm8k_min_score=0.70,
        default_gsm8k_stop=(
            "Question",
            "Assistant:",
            "<|separator|>",
            "<turn|>",
            "<|tool_response>",
            "<|turn>",
        ),
        default_extra_args=(
            "--ep 4 --moe-runner-backend auto --skip-server-warmup "
            "--max-running-requests 128 --mem-fraction-static 0.70 "
            "--cuda-graph-bs 1 2 4 8 16 32 64 128 "
            "--watchdog-timeout 900"
        ),
        extra_args_env="MUSA_SMOKE_GEMMA4_26B_A4B_EXTRA_ARGS",
    )


if __name__ == "__main__":
    unittest.main()
