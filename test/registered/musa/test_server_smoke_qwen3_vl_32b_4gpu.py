import unittest

from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=600, suite="stage-a-test-4-gpu-musa-qwen3-vl-32b-smoke")


class TestMusaServerSmokeQwen3VL32B4GPU(musa_server_smoke.MusaServerSmokeTest):
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="qwen3-vl-32b-tp4",
        model_env="MUSA_SMOKE_QWEN3_VL_32B_MODEL",
        tp_size=4,
        default_eval_name="vlm",
        default_vlm_dataset="mmmu_val",
        default_vlm_metric="mmmu_acc,none",
        default_vlm_min_score=0.55,
        default_gsm8k_min_score=0.94,
        default_extra_args=(
            "--ep 4 --attention-backend fa3 --mm-attention-backend fa3 "
            "--sampling-backend flashinfer --disable-custom-all-reduce "
            "--moe-runner-backend auto --max-running-requests 128 "
            "--max-prefill-tokens 2048 "
            "--cuda-graph-bs 1 2 4 8 16 32 64 128 "
            "--skip-server-warmup --watchdog-timeout 900"
        ),
        extra_args_env="MUSA_SMOKE_QWEN3_VL_32B_EXTRA_ARGS",
    )


if __name__ == "__main__":
    unittest.main()
