import glob
import json
import os
import shutil
import shlex
import tempfile
import time
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import requests

from sglang.srt.environ import temp_set_env
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.kits.mmmu_vlm_kit import _run_lmms_eval_with_retry
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


@dataclass(frozen=True)
class MusaSmokeCase:
    name: str
    model_env: str
    tp_size: int = 1
    default_eval_name: str = "gsm8k"
    default_vlm_dataset: str = "mmmu_val"
    default_vlm_metric: str = "mmmu_acc,none"
    default_vlm_min_score: float = 0.35
    default_extra_args: str = ""
    default_gsm8k_min_score: float = 0.85
    default_gsm8k_chat_template_kwargs: Optional[str] = None
    default_gsm8k_reasoning_effort: Optional[str] = None
    default_gsm8k_stop: Optional[tuple[str, ...]] = None
    extra_args_env: Optional[str] = None


def _split_args(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return shlex.split(value)


def _build_server_args(case: MusaSmokeCase) -> list[str]:
    args = [
        "--device",
        "musa",
        "--tp",
        str(case.tp_size),
        "--trust-remote-code",
    ]
    args.extend(
        _split_args(
            os.getenv(
                "MUSA_SMOKE_COMMON_EXTRA_ARGS",
                "--disable-cuda-graph --disable-piecewise-cuda-graph "
                "--attention-backend fa3 --disable-overlap-schedule "
                "--chunked-prefill-size 2048",
            )
        )
    )
    args.extend(_split_args(case.default_extra_args))
    if case.extra_args_env:
        args.extend(_split_args(os.getenv(case.extra_args_env)))
    return args


def _write_server_args_artifact(case: MusaSmokeCase, model: str, args: list[str]) -> None:
    artifact_dir = os.getenv("MUSA_SMOKE_ARTIFACT_DIR")
    if not artifact_dir:
        return

    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, "server_args.json")
    with open(path, "w") as fout:
        json.dump(
            {
                "case": case.name,
                "model": model,
                "args": args,
            },
            fout,
            indent=2,
        )
        fout.write("\n")


def _run_generate(base_url: str):
    response = requests.post(
        base_url + "/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    assert "text" in result, result
    assert isinstance(result["text"], str), result
    assert result["text"].strip(), result


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_optional_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value is not None else None


def _env_optional_limit(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.lower() in ("all", "full", "none"):
        return None
    return value


def _env_optional_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    return float(value) if value is not None else None


def _env_optional_str_list(
    name: str, default: Optional[list[str]] = None
) -> Optional[list[str]]:
    value = os.getenv(name)
    if value is None:
        return default

    parsed = json.loads(value)
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    raise ValueError(f"{name} must be a JSON string or list of strings")


def _open_server_log_files(case_name: str):
    artifact_dir = os.getenv("MUSA_SMOKE_ARTIFACT_DIR")
    if not artifact_dir:
        return None

    os.makedirs(artifact_dir, exist_ok=True)
    prefix = case_name.replace("/", "_")
    stdout = open(os.path.join(artifact_dir, f"{prefix}_server_stdout.log"), "w")
    stderr = open(os.path.join(artifact_dir, f"{prefix}_server_stderr.log"), "w")
    return stdout, stderr


def _close_server_log_files(log_files):
    if not log_files:
        return
    for log_file in log_files:
        try:
            log_file.close()
        except Exception:
            pass


def _get_visible_musa_device_count() -> int:
    visible = os.getenv("MUSA_VISIBLE_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES")
    if not visible:
        return 0
    return len([part for part in visible.split(",") if part.strip() != ""])


def _wait_for_musa_free_memory(case: MusaSmokeCase) -> None:
    if not os.getenv("SGLANG_IS_IN_CI"):
        return

    try:
        import torch
    except Exception:
        return

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        return

    device_count = torch.musa.device_count()
    if device_count <= 0:
        return

    visible_count = _get_visible_musa_device_count()
    num_devices = min(
        case.tp_size,
        visible_count if visible_count > 0 else device_count,
        device_count,
    )
    if num_devices <= 0:
        return

    min_free_gb = float(os.getenv("MUSA_SMOKE_MIN_FREE_MEMORY_GB", "70"))
    timeout_s = float(os.getenv("MUSA_SMOKE_FREE_MEMORY_TIMEOUT", "300"))
    poll_s = float(os.getenv("MUSA_SMOKE_FREE_MEMORY_POLL_INTERVAL", "5"))
    deadline = time.time() + timeout_s
    last_snapshot = None

    while True:
        free_gb = []
        for device_id in range(num_devices):
            with torch.musa.device(device_id):
                free_bytes, _total_bytes = torch.musa.mem_get_info()
            free_gb.append(free_bytes / (1024**3))

        last_snapshot = ", ".join(
            f"gpu{device_id}={free:.2f}GB" for device_id, free in enumerate(free_gb)
        )
        if all(free >= min_free_gb for free in free_gb):
            print(
                "MUSA smoke: enough free GPU memory before launch: "
                f"{last_snapshot} (threshold={min_free_gb:.2f}GB)",
                flush=True,
            )
            return

        remaining = deadline - time.time()
        if remaining <= 0:
            raise unittest.SkipTest(
                "MUSA smoke skipped due to insufficient free GPU memory before "
                f"launch: {last_snapshot} (threshold={min_free_gb:.2f}GB, "
                f"timeout={timeout_s:.0f}s)"
            )

        print(
            "MUSA smoke: waiting for GPU memory to clear before launch: "
            f"{last_snapshot} (threshold={min_free_gb:.2f}GB, "
            f"remaining={remaining:.0f}s)",
            flush=True,
        )
        time.sleep(poll_s)


def _assert_server_process_alive(process):
    if process is None:
        return
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(
            "SGLang server exited during MUSA smoke test with code "
            f"{return_code}. Check server stdout/stderr artifacts for details."
        )


def _run_gsm8k_eval(
    base_url: str,
    model: str,
    case: MusaSmokeCase,
    server_process=None,
):
    data_path = os.getenv("MUSA_SMOKE_GSM8K_DATA_PATH", "/data/eval/gsm8k/test.jsonl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"MUSA smoke GSM8K data file does not exist: {data_path}"
        )

    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="gsm8k",
        api=os.getenv("MUSA_SMOKE_GSM8K_API", "chat"),
        num_examples=_env_int("MUSA_SMOKE_GSM8K_NUM_EXAMPLES", 16),
        num_threads=_env_int("MUSA_SMOKE_GSM8K_NUM_THREADS", 8),
        num_shots=_env_int("MUSA_SMOKE_GSM8K_NUM_SHOTS", 5),
        max_tokens=_env_int("MUSA_SMOKE_GSM8K_MAX_TOKENS", 1024),
        temperature=_env_float("MUSA_SMOKE_GSM8K_TEMPERATURE", 0.0),
        top_p=_env_float("MUSA_SMOKE_GSM8K_TOP_P", 1.0),
        top_k=_env_optional_int("MUSA_SMOKE_GSM8K_TOP_K"),
        min_p=_env_optional_float("MUSA_SMOKE_GSM8K_MIN_P"),
        stop=_env_optional_str_list(
            "MUSA_SMOKE_GSM8K_STOP",
            list(case.default_gsm8k_stop) if case.default_gsm8k_stop else None,
        ),
        chat_template_kwargs=os.getenv(
            "MUSA_SMOKE_GSM8K_CHAT_TEMPLATE_KWARGS",
            case.default_gsm8k_chat_template_kwargs,
        ),
        reasoning_effort=os.getenv(
            "MUSA_SMOKE_GSM8K_REASONING_EFFORT",
            case.default_gsm8k_reasoning_effort,
        ),
        gsm8k_data_path=data_path,
    )
    print(
        "MUSA smoke GSM8K eval config: "
        + json.dumps(
            {
                "api": args.api,
                "chat_template_kwargs": args.chat_template_kwargs,
                "data_path": data_path,
                "max_tokens": args.max_tokens,
                "model": model,
                "num_examples": args.num_examples,
                "num_shots": args.num_shots,
                "num_threads": args.num_threads,
                "reasoning_effort": args.reasoning_effort,
                "stop": args.stop,
                "temperature": args.temperature,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _assert_server_process_alive(server_process)
    try:
        metrics = run_eval(args)
    except Exception:
        _assert_server_process_alive(server_process)
        raise
    _assert_server_process_alive(server_process)
    score = metrics.get("score")
    min_score = _env_float("MUSA_SMOKE_GSM8K_MIN_SCORE", case.default_gsm8k_min_score)
    assert score is not None, metrics
    assert score >= min_score, (
        f"GSM8K score {score:.4f} is below threshold {min_score:.4f}; "
        f"metrics={metrics}"
    )


def _find_lmms_eval_result_file(output_path: str, dataset: str) -> str:
    result_files = glob.glob(f"{output_path}/**/*.json", recursive=True)
    if not result_files:
        result_files = glob.glob(f"{output_path}/*.json")
    if not result_files:
        raise FileNotFoundError(f"No JSON result files found in {output_path}")

    for result_file in result_files:
        try:
            with open(result_file, "r") as f:
                result = json.load(f)
        except Exception:
            continue
        if isinstance(result, dict) and dataset in result.get("results", {}):
            return result_file

    return result_files[0]


def _safe_model_stem(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _write_vlm_metrics_artifact(model: str, metrics: dict) -> None:
    report_dir = os.environ.get("SGLANG_EVAL_REPORT_DIR") or os.environ.get(
        "MUSA_SMOKE_ARTIFACT_DIR", "/tmp"
    )
    os.makedirs(report_dir, exist_ok=True)
    result_filename = os.path.join(report_dir, f"vlm__{_safe_model_stem(model)}.json")
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"Writing VLM results to {result_filename}")


def _lmms_num_samples(result: dict, dataset: str) -> Optional[int]:
    samples = result.get("n-samples", {}).get(dataset)
    if isinstance(samples, dict):
        value = samples.get("effective") or samples.get("original")
        return int(value) if value is not None else None
    return None


def _iter_response_texts(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_response_texts(item)
    elif isinstance(value, dict):
        for key in ("content", "text", "response"):
            if key in value:
                yield from _iter_response_texts(value[key])


def _lmms_sample_response_texts(sample: dict) -> list[str]:
    for key in ("filtered_resps", "resps"):
        texts = [text for text in _iter_response_texts(sample.get(key)) if text]
        if texts:
            return texts
    return []


def _count_lmms_sample_output_tokens(output_path: str, model: str) -> Optional[int]:
    sample_files = glob.glob(f"{output_path}/**/*_samples_*.jsonl", recursive=True)
    if not sample_files:
        return None

    tokenizer = get_tokenizer(model, trust_remote_code=True)
    total_tokens = 0
    for sample_file in sample_files:
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                for text in _lmms_sample_response_texts(sample):
                    total_tokens += len(
                        tokenizer.encode(text, add_special_tokens=False)
                    )
    return total_tokens


def _archive_lmms_eval_outputs(
    output_path: str,
    model: str,
    dataset: str,
    result_file_path: str,
) -> None:
    artifact_dir = os.getenv("MUSA_SMOKE_ARTIFACT_DIR")
    if not artifact_dir:
        return

    archive_dir = os.path.join(artifact_dir, "lmms_eval")
    os.makedirs(archive_dir, exist_ok=True)

    archive_prefix = f"{_safe_model_stem(model)}__{dataset}"
    result_dst = os.path.join(archive_dir, f"{archive_prefix}__result.json")
    shutil.copy2(result_file_path, result_dst)
    print(f"Archived lmms-eval result to {result_dst}")

    sample_files = glob.glob(f"{output_path}/**/*_samples_{dataset}.jsonl", recursive=True)
    if not sample_files:
        sample_files = glob.glob(f"{output_path}/**/*_samples_*.jsonl", recursive=True)

    for sample_file in sample_files:
        sample_dst = os.path.join(
            archive_dir,
            f"{archive_prefix}__{os.path.basename(sample_file)}",
        )
        shutil.copy2(sample_file, sample_dst)
        print(f"Archived lmms-eval samples to {sample_dst}")


def _run_vlm_eval(
    base_url: str,
    model: str,
    case: MusaSmokeCase,
    server_process=None,
):
    dataset = os.getenv("MUSA_SMOKE_VLM_DATASET", case.default_vlm_dataset)
    metric = os.getenv("MUSA_SMOKE_VLM_METRIC", case.default_vlm_metric)
    raw_limit = os.getenv("MUSA_SMOKE_VLM_LIMIT", "all")
    limit = _env_optional_limit("MUSA_SMOKE_VLM_LIMIT")
    batch_size = _env_int("MUSA_SMOKE_VLM_BATCH_SIZE", 64)
    timeout = _env_int("MUSA_SMOKE_VLM_TIMEOUT", 3600)

    model_args = (
        f'model_version="{model}",'
        f'tp={_env_int("MUSA_SMOKE_VLM_EVAL_TP", 1)}'
    )

    with tempfile.TemporaryDirectory(prefix="musa_smoke_vlm_") as output_path:
        cmd = [
            "python3",
            "-m",
            "lmms_eval",
            "--model",
            "openai_compatible",
            "--model_args",
            model_args,
            "--tasks",
            dataset,
            "--batch_size",
            str(batch_size),
            "--log_samples",
            "--log_samples_suffix",
            "openai_compatible",
            "--output_path",
            output_path,
        ]
        if limit is not None:
            cmd.extend(["--limit", limit])

        print(
            "MUSA smoke VLM eval config: "
            + json.dumps(
                {
                    "batch_size": batch_size,
                    "dataset": dataset,
                    "limit": raw_limit,
                    "metric": metric,
                    "model": model,
                    "model_args": model_args,
                    "timeout": timeout,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        _assert_server_process_alive(server_process)
        tic = time.perf_counter()
        try:
            with temp_set_env(
                OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "sk-123456"),
                OPENAI_API_BASE=f"{base_url}/v1",
            ):
                _run_lmms_eval_with_retry(cmd, timeout=timeout)
        except Exception:
            _assert_server_process_alive(server_process)
            raise
        latency = time.perf_counter() - tic
        _assert_server_process_alive(server_process)

        result_file_path = _find_lmms_eval_result_file(output_path, dataset)
        with open(result_file_path, "r") as f:
            result = json.load(f)
            print(f"VLM eval result: {result}")

        try:
            num_output_tokens = _count_lmms_sample_output_tokens(output_path, model)
        except Exception as exc:
            print(f"Failed to count VLM output tokens from lmms-eval samples: {exc}")
            num_output_tokens = None
        _archive_lmms_eval_outputs(output_path, model, dataset, result_file_path)

    score = result["results"][dataset][metric]
    metrics = {
        "eval_name": "vlm",
        "dataset": dataset,
        "metric": metric,
        "model": model,
        "score": score,
        "batch_size": batch_size,
        "limit": raw_limit,
        "latency": latency,
        "num_examples_actual": _lmms_num_samples(result, dataset),
    }
    if num_output_tokens is not None:
        metrics["num_output_tokens"] = num_output_tokens
        metrics["output_throughput_source"] = "lmms_eval_sample_tokenized_response"
        if num_output_tokens > 0 and latency > 0:
            metrics["output_throughput"] = num_output_tokens / latency
            print(f"VLM output throughput: {metrics['output_throughput']:.3f} token/s")
    _write_vlm_metrics_artifact(model, metrics)

    min_score = _env_float("MUSA_SMOKE_VLM_MIN_SCORE", case.default_vlm_min_score)
    assert score >= min_score, (
        f"{dataset} {metric} {score:.4f} is below threshold {min_score:.4f}; "
        f"result={result}"
    )


def _run_model_eval(
    base_url: str,
    model: str,
    case: MusaSmokeCase,
    server_process=None,
):
    eval_name = os.getenv("MUSA_SMOKE_EVAL_NAME", case.default_eval_name).lower()
    if eval_name == "gsm8k":
        _run_gsm8k_eval(base_url, model, case, server_process=server_process)
    elif eval_name in ("vlm", "mmmu"):
        _run_vlm_eval(base_url, model, case, server_process=server_process)
    else:
        raise ValueError(f"Unsupported MUSA smoke eval: {eval_name}")


class MusaServerSmokeTest(CustomTestCase):
    smoke_case: MusaSmokeCase
    process = None
    server_log_files = None

    @classmethod
    def setUpClass(cls):
        model = os.getenv(cls.smoke_case.model_env)
        if not model:
            raise unittest.SkipTest(
                f"{cls.smoke_case.model_env} is not configured for MUSA smoke test"
            )

        cls.model = model
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.server_log_files = _open_server_log_files(cls.smoke_case.name)
        _wait_for_musa_free_memory(cls.smoke_case)
        server_args = _build_server_args(cls.smoke_case)
        _write_server_args_artifact(cls.smoke_case, cls.model, server_args)
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=int(
                    os.getenv(
                        "MUSA_SMOKE_SERVER_TIMEOUT", DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
                    )
                ),
                other_args=server_args,
                return_stdout_stderr=cls.server_log_files,
                device="musa",
            )
        except Exception:
            _close_server_log_files(cls.server_log_files)
            cls.server_log_files = None
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        _close_server_log_files(cls.server_log_files)
        cls.server_log_files = None

    def test_health(self):
        _assert_server_process_alive(self.process)
        response = requests.get(self.base_url + "/health", timeout=30)
        response.raise_for_status()
        _assert_server_process_alive(self.process)

    def test_generate(self):
        _assert_server_process_alive(self.process)
        _run_generate(self.base_url)
        _assert_server_process_alive(self.process)

    def test_model_eval(self):
        _run_model_eval(
            self.base_url,
            self.model,
            self.smoke_case,
            server_process=self.process,
        )
