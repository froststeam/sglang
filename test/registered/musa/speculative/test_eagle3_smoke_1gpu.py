import unittest

from sglang.test import musa_speculative_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=900, suite="stage-a-test-1-gpu-musa-eagle3-smoke")


class TestMusaEagle3Smoke1GPU(musa_speculative_smoke.MusaSpeculativeSmokeTest):
    speculative_case = musa_speculative_smoke.MusaSpeculativeSmokeCase(
        name="qwen3-8b-eagle3-tp1",
        model_env="MUSA_SPEC_EAGLE3_TARGET_MODEL",
        draft_model_env="MUSA_SPEC_EAGLE3_DRAFT_MODEL",
        algorithm="EAGLE3",
        tp_size=1,
        default_common_extra_args="--mem-fraction-static 0.70",
        default_speculative_extra_args=(
            "--speculative-num-steps 2 --speculative-eagle-topk 1 "
            "--speculative-num-draft-tokens 3"
        ),
        speculative_extra_args_env="MUSA_SPEC_EAGLE3_EXTRA_ARGS",
        default_accuracy_threshold=0.92,
        default_min_accept_length=2.0,
        default_min_speedup=1.3,
    )


if __name__ == "__main__":
    unittest.main()
