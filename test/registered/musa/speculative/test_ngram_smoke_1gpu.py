import unittest

from sglang.test import musa_speculative_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=900, suite="stage-a-test-1-gpu-musa-ngram-smoke")


class TestMusaNgramSmoke1GPU(musa_speculative_smoke.MusaSpeculativeSmokeTest):
    speculative_case = musa_speculative_smoke.MusaSpeculativeSmokeCase(
        name="qwen3-8b-ngram-tp1",
        model_env="MUSA_SPEC_NGRAM_MODEL",
        algorithm="NGRAM",
        tp_size=1,
        default_common_extra_args=(
            "--mem-fraction-static 0.70 --page-size 1 --max-running-requests 1"
        ),
        default_speculative_extra_args=(
            "--speculative-num-draft-tokens 16 "
            "--speculative-ngram-max-bfs-breadth 10"
        ),
        speculative_extra_args_env="MUSA_SPEC_NGRAM_EXTRA_ARGS",
        default_accuracy_threshold=0.92,
        default_min_accept_length=1.8,
        default_min_speedup=1.3,
    )


if __name__ == "__main__":
    unittest.main()
