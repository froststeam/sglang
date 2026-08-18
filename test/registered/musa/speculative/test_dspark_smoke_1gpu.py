import unittest

from sglang.test import musa_speculative_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=900, suite="stage-a-test-1-gpu-musa-dspark-smoke")


class TestMusaDSparkSmoke1GPU(musa_speculative_smoke.MusaSpeculativeSmokeTest):
    speculative_case = musa_speculative_smoke.MusaSpeculativeSmokeCase(
        name="qwen3-8b-dspark-tp1",
        model_env="MUSA_SPEC_DSPARK_TARGET_MODEL",
        draft_model_env="MUSA_SPEC_DSPARK_DRAFT_MODEL",
        algorithm="DSPARK",
        tp_size=1,
        default_common_extra_args="--mem-fraction-static 0.70",
        default_speculative_extra_args="--speculative-dspark-block-size 7",
        speculative_extra_args_env="MUSA_SPEC_DSPARK_EXTRA_ARGS",
        default_accuracy_threshold=0.92,
        default_min_accept_length=4.0,
        default_min_speedup=2.0,
    )


if __name__ == "__main__":
    unittest.main()
