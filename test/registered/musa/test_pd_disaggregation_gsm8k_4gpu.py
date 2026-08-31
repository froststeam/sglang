import json
import os
import shlex
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_musa_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_musa_ci(
    est_time=1200, suite="stage-b-test-4-gpu-musa-pd-disaggregation-smoke"
)


def _split_args(value: str | None) -> list[str]:
    return shlex.split(value) if value else []


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_optional_str_list(name: str):
    value = os.getenv(name)
    if value is None:
        return None
    parsed = json.loads(value)
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    raise ValueError(f"{name} must be a JSON string or list of strings")


class TestMusaPDDisaggregationGSM8K4GPU(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.getenv("MUSA_PD_QWEN35_MOE_MODEL")
        if not cls.model:
            raise unittest.SkipTest("MUSA_PD_QWEN35_MOE_MODEL is not configured")

        super().setUpClass()
        cls.common_server_args = _split_args(
            os.getenv(
                "MUSA_PD_COMMON_EXTRA_ARGS",
                "--disable-piecewise-cuda-graph --attention-backend fa3 "
                "--disable-overlap-schedule --chunked-prefill-size 2048",
            )
        )
        cls.qwen35_extra_args = _split_args(
            os.getenv(
                "MUSA_PD_QWEN35_MOE_EXTRA_ARGS",
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

        try:
            cls.start_prefill()
            cls.start_decode()
            cls.wait_server_ready(
                cls.prefill_url + "/health", process=cls.process_prefill
            )
            cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
            cls.launch_lb()
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def _role_args(cls, mode: str, base_gpu_id: int) -> list[str]:
        args = [
            "--device",
            "musa",
            "--trust-remote-code",
            "--disaggregation-mode",
            mode,
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "2",
            "--base-gpu-id",
            str(base_gpu_id),
        ]
        args += cls.common_server_args + cls.qwen35_extra_args
        args += cls.transfer_backend + cls.rdma_devices
        return args

    @classmethod
    def start_prefill(cls):
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._role_args("prefill", 0),
        )

    @classmethod
    def start_decode(cls):
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._role_args("decode", 2),
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api=os.getenv("MUSA_PD_GSM8K_API", "chat"),
            num_examples=_env_int("MUSA_PD_GSM8K_NUM_EXAMPLES", 200),
            num_threads=_env_int("MUSA_PD_GSM8K_NUM_THREADS", 16),
            num_shots=_env_int("MUSA_PD_GSM8K_NUM_SHOTS", 5),
            max_tokens=_env_int("MUSA_PD_GSM8K_MAX_TOKENS", 512),
            temperature=_env_float("MUSA_PD_GSM8K_TEMPERATURE", 0.0),
            top_p=_env_float("MUSA_PD_GSM8K_TOP_P", 1.0),
            top_k=None,
            min_p=None,
            stop=_env_optional_str_list("MUSA_PD_GSM8K_STOP"),
            chat_template_kwargs=os.getenv(
                "MUSA_PD_GSM8K_CHAT_TEMPLATE_KWARGS",
                '{"enable_thinking": false}',
            ),
            reasoning_effort=os.getenv("MUSA_PD_GSM8K_REASONING_EFFORT", "none"),
            gsm8k_data_path=os.getenv(
                "MUSA_PD_GSM8K_DATA_PATH", "/data/eval/gsm8k/test.jsonl"
            ),
        )
        metrics = run_eval(args)
        print(f"MUSA PD disaggregation GSM8K metrics: {metrics}")

        min_score = _env_float("MUSA_PD_GSM8K_MIN_SCORE", 0.90)
        self.assertGreaterEqual(metrics["score"], min_score)


if __name__ == "__main__":
    unittest.main()
