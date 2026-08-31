import os
import shlex
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test import musa_server_smoke
from sglang.test.ci.ci_register import register_musa_ci
from sglang.test.kits.cache_hit_kit import run_fixed_prefix_cache_hit_test
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_musa_ci(
    est_time=900, suite="stage-b-test-2-gpu-musa-radix-prefix-cache-smoke"
)


def _split_args(value: str | None) -> list[str]:
    return shlex.split(value) if value else []


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


class TestMusaRadixPrefixCache2GPU(CustomTestCase):
    process = None
    server_log_files = None
    smoke_case = musa_server_smoke.MusaSmokeCase(
        name="qwen35-radix-prefix-cache-tp2",
        model_env="MUSA_RADIX_PREFIX_CACHE_MODEL",
        tp_size=2,
    )

    @classmethod
    def setUpClass(cls):
        cls.model = os.getenv(cls.smoke_case.model_env)
        if not cls.model:
            raise unittest.SkipTest(
                f"{cls.smoke_case.model_env} is not configured for MUSA radix "
                "prefix cache test"
            )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.server_log_files = musa_server_smoke._open_server_log_files(
            cls.smoke_case.name
        )
        musa_server_smoke._wait_for_musa_free_memory(cls.smoke_case)

        server_args = [
            "--device",
            "musa",
            "--tp",
            "2",
            "--trust-remote-code",
        ]
        server_args.extend(
            _split_args(
                os.getenv(
                    "MUSA_RADIX_PREFIX_CACHE_COMMON_EXTRA_ARGS",
                    "--disable-piecewise-cuda-graph --attention-backend fa3 "
                    "--disable-overlap-schedule --chunked-prefill-size 2048",
                )
            )
        )
        server_args.extend(
            _split_args(
                os.getenv(
                    "MUSA_RADIX_PREFIX_CACHE_EXTRA_ARGS",
                    "--linear-attn-prefill-backend flashinfer "
                    "--linear-attn-decode-backend flashinfer "
                    "--moe-runner-backend auto "
                    "--mamba-scheduler-strategy extra_buffer "
                    "--mamba-track-interval 128 "
                    "--max-running-requests 64 "
                    "--mem-fraction-static 0.70 "
                    "--cuda-graph-bs 1 2 4 8 16 32 64",
                )
            )
        )
        musa_server_smoke._write_server_args_artifact(
            cls.smoke_case, cls.model, server_args
        )

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=int(
                    os.getenv(
                        "MUSA_RADIX_PREFIX_CACHE_SERVER_TIMEOUT",
                        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    )
                ),
                other_args=server_args,
                return_stdout_stderr=cls.server_log_files,
                device="musa",
                health_endpoint=os.getenv(
                    "MUSA_RADIX_PREFIX_CACHE_SERVER_HEALTH_ENDPOINT",
                    "/health_generate",
                ),
            )
        except Exception:
            musa_server_smoke._close_server_log_files(cls.server_log_files)
            cls.server_log_files = None
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        musa_server_smoke._close_server_log_files(cls.server_log_files)
        cls.server_log_files = None

    def test_radix_prefix_cache_hit_rate(self):
        response = requests.get(self.base_url + "/health", timeout=30)
        response.raise_for_status()

        result = run_fixed_prefix_cache_hit_test(
            base_url=self.base_url,
            model_path=self.model,
            prefix_len=_env_int("MUSA_RADIX_PREFIX_CACHE_PREFIX_LEN", 1024),
            suffix_len=_env_int("MUSA_RADIX_PREFIX_CACHE_SUFFIX_LEN", 1024),
            num_groups=_env_int("MUSA_RADIX_PREFIX_CACHE_GROUPS", 4),
            prompts_per_group=_env_int("MUSA_RADIX_PREFIX_CACHE_PROMPTS_PER_GROUP", 4),
            output_len=_env_int("MUSA_RADIX_PREFIX_CACHE_OUTPUT_LEN", 32),
            min_hit_rate=_env_float("MUSA_RADIX_PREFIX_CACHE_MIN_HIT_RATE", 0.45),
            max_parallel=_env_int("MUSA_RADIX_PREFIX_CACHE_MAX_PARALLEL", 16),
            seed=_env_int("MUSA_RADIX_PREFIX_CACHE_SEED", 1),
        )
        print(f"MUSA radix prefix cache metrics: {result}")


if __name__ == "__main__":
    unittest.main()
