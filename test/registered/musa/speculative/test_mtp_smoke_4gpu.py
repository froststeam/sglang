import unittest

from sglang.test import musa_speculative_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=1200, suite="stage-a-test-4-gpu-musa-mtp-smoke")


class TestMusaMTPSmoke4GPU(musa_speculative_smoke.MusaSpeculativeSmokeTest):
    speculative_case = musa_speculative_smoke.MusaSpeculativeSmokeCase(
        name="qwen3.5-35b-a3b-fp8-mtp-tp4",
        model_env="MUSA_SPEC_MTP_MODEL",
        algorithm="NEXTN",
        tp_size=4,
        default_common_extra_args="--mem-fraction-static 0.70 --max-running-requests 1",
        default_speculative_extra_args=(
            "--speculative-num-steps 3 --speculative-eagle-topk 1 "
            "--speculative-num-draft-tokens 4"
        ),
        speculative_extra_args_env="MUSA_SPEC_MTP_EXTRA_ARGS",
        default_accuracy_threshold=0.94,
        default_min_accept_length=3.0,
        default_min_speedup=1.5,
    )


if __name__ == "__main__":
    unittest.main()
