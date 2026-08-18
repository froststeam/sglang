import concurrent.futures
import json
import os
import random
import shlex
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import requests
from openai import OpenAI
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test import musa_server_smoke
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


@dataclass(frozen=True)
class MusaSpeculativeSmokeCase:
    name: str
    model_env: str
    algorithm: str
    tp_size: int = 1
    draft_model_env: Optional[str] = None
    default_common_extra_args: str = ""
    common_extra_args_env: Optional[str] = None
    default_speculative_extra_args: str = ""
    speculative_extra_args_env: Optional[str] = None
    default_min_accept_length: float = 1.01
    default_min_speedup: float = 1.10
    default_max_new_tokens: int = 128
    default_accuracy_threshold: float = 0.70


def _split_args(value: Optional[str]) -> list[str]:
    return shlex.split(value) if value else []


def _common_server_args(case: MusaSpeculativeSmokeCase) -> list[str]:
    args = [
        "--device",
        "musa",
        "--tp",
        str(case.tp_size),
        "--trust-remote-code",
    ]
    common_extra_args = os.getenv("MUSA_SPEC_COMMON_EXTRA_ARGS")
    if common_extra_args is None:
        args.extend(_split_args(case.default_common_extra_args))
        common_extra_args = (
            "--disable-piecewise-cuda-graph --attention-backend fa3 "
            "--disable-overlap-schedule --chunked-prefill-size 2048 "
            "--cuda-graph-bs 1 2 4"
        )
    args.extend(_split_args(common_extra_args))
    if case.common_extra_args_env:
        args.extend(_split_args(os.getenv(case.common_extra_args_env)))
    return args


def _speculative_server_args(
    case: MusaSpeculativeSmokeCase, draft_model: Optional[str]
) -> list[str]:
    args = _common_server_args(case)
    args.extend(
        [
            "--speculative-algorithm",
            case.algorithm,
        ]
    )
    if draft_model:
        args.extend(["--speculative-draft-model-path", draft_model])
    args.extend(_split_args(case.default_speculative_extra_args))
    if case.speculative_extra_args_env:
        args.extend(_split_args(os.getenv(case.speculative_extra_args_env)))
    return args


def _open_log_files(name: str, phase: str):
    artifact_dir = os.getenv("MUSA_SMOKE_ARTIFACT_DIR")
    if not artifact_dir:
        return None
    os.makedirs(artifact_dir, exist_ok=True)
    prefix = name.replace("/", "_")
    return (
        open(os.path.join(artifact_dir, f"{prefix}_{phase}_stdout.log"), "w"),
        open(os.path.join(artifact_dir, f"{prefix}_{phase}_stderr.log"), "w"),
    )


def _close_log_files(log_files):
    if log_files:
        for log_file in log_files:
            log_file.close()


def _launch_server(model: str, args: list[str], name: str, phase: str):
    log_files = _open_log_files(name, phase)
    try:
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=int(
                os.getenv("MUSA_SPEC_SERVER_TIMEOUT", DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)
            ),
            other_args=args,
            return_stdout_stderr=log_files,
            device="musa",
            health_endpoint=os.getenv(
                "MUSA_SPEC_SERVER_HEALTH_ENDPOINT", "/health_generate"
            ),
        )
    except Exception:
        _close_log_files(log_files)
        raise
    return process, log_files


def _stop_server(process, log_files):
    if process is not None:
        kill_process_tree(process.pid)
    _close_log_files(log_files)


def _build_decode_prompts(model: str) -> list[str]:
    data_path = os.getenv(
        "MUSA_SPEC_PERF_DATA_PATH",
        os.getenv("MUSA_SPEC_EVAL_GSM8K_DATA_PATH", "/data/eval/gsm8k/test.jsonl"),
    )
    rows = []
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("question")
        if not isinstance(prompt, str):
            turns = row.get("turns")
            prompt = (
                turns[0] if isinstance(turns, list) and turns else row.get("prompt")
            )
        if not isinstance(prompt, str):
            raise ValueError(f"unsupported performance dataset row: {row!r}")
        suffix = (
            "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        if suffix not in prompt:
            prompt = prompt.rstrip() + suffix
        rows.append(prompt)

    rng = random.Random(int(os.getenv("MUSA_SPEC_PERF_SEED", "980406")))
    rng.shuffle(rows)
    rows = rows[: int(os.getenv("MUSA_SPEC_PERF_NUM_EXAMPLES", "64"))]
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = []
    for prompt in rows:
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(rendered)
    return prompts


def _decode_probe(model: str, prompts: list[str]) -> dict:
    bs = int(os.getenv("MUSA_SPEC_PERF_BS", "1"))
    max_new_tokens = int(os.getenv("MUSA_SPEC_PERF_MAX_NEW_TOKENS", "512"))
    timeout = float(os.getenv("MUSA_SPEC_REQUEST_TIMEOUT", "600"))
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, int):
        stop_token_ids = [eos_id]
    elif eos_id is None:
        stop_token_ids = None
    else:
        stop_token_ids = [int(x) for x in eos_id]
    client = OpenAI(base_url=DEFAULT_URL_FOR_TEST + "/v1", api_key="EMPTY")

    def one(item):
        idx, prompt = item
        start = time.perf_counter()
        first = None
        tokens = 0
        text = []
        extra_body = {
            "top_k": -1,
            "sampling_seed": int(os.getenv("MUSA_SPEC_PERF_SEED", "980406")) + idx,
            "skip_special_tokens": True,
        }
        if stop_token_ids is not None:
            extra_body["stop_token_ids"] = stop_token_ids
        stream = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            stream=True,
            stream_options={"include_usage": True},
            timeout=timeout,
            extra_body=extra_body,
        )
        for chunk in stream:
            now = time.perf_counter()
            if chunk.usage is not None:
                tokens = chunk.usage.completion_tokens or 0
            for choice in chunk.choices:
                piece = choice.text or ""
                if piece and first is None:
                    first = now
                text.append(piece)
        end = time.perf_counter()
        return {
            "decode_s": end - (first or end),
            "request_s": end - start,
            "completion_tokens": tokens,
            "text": "".join(text),
        }

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
        results = list(pool.map(one, enumerate(prompts)))
    wall_s = time.perf_counter() - wall_start
    decode_times = [float(row["decode_s"]) for row in results]
    completion_tokens = sum(int(row["completion_tokens"] or 0) for row in results)
    decode_batch_s = sum(
        max(decode_times[i : i + bs]) for i in range(0, len(decode_times), bs)
    )
    return {
        "num_examples": len(results),
        "num_threads": bs,
        "wall_s": wall_s,
        "decode_sum_s": sum(decode_times),
        "decode_batch_s": decode_batch_s,
        "completion_tokens": completion_tokens,
        "decode_batch_tps": (
            completion_tokens / decode_batch_s if decode_batch_s > 0 else 0.0
        ),
        "results": results,
    }


def _server_accept_length() -> Optional[float]:
    response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info", timeout=30)
    response.raise_for_status()
    values = []
    for state in response.json().get("internal_states", []):
        value = state.get("avg_spec_accept_length")
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _write_server_args_artifact(
    case: MusaSpeculativeSmokeCase,
    model: str,
    draft_model: Optional[str],
    baseline_args: list[str],
    speculative_args: list[str],
) -> None:
    artifact_dir = os.getenv("MUSA_SMOKE_ARTIFACT_DIR")
    if not artifact_dir:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "server_args.json"), "w") as fout:
        json.dump(
            {
                "case": case.name,
                "model": model,
                "draft_model": draft_model,
                "baseline_args": baseline_args,
                "speculative_args": speculative_args,
                # Keep the generic collector's topology parser working.
                "args": speculative_args,
            },
            fout,
            indent=2,
        )
        fout.write("\n")


def _write_metrics_artifact(case: MusaSpeculativeSmokeCase, metrics: dict) -> None:
    artifact_dir = os.getenv("MUSA_SMOKE_ARTIFACT_DIR")
    if not artifact_dir:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    metrics_path = os.path.join(artifact_dir, f"speculative__{case.name}.json")
    with open(metrics_path, "w") as fout:
        json.dump(metrics, fout, indent=2, sort_keys=True)
        fout.write("\n")


class MusaSpeculativeSmokeTest(CustomTestCase):
    speculative_case: MusaSpeculativeSmokeCase

    def test_speculative_accuracy(self):
        case = self.speculative_case
        model = os.getenv(case.model_env)
        if not model:
            raise unittest.SkipTest(f"{case.model_env} is not configured")
        draft_model = os.getenv(case.draft_model_env) if case.draft_model_env else None
        if case.draft_model_env and not draft_model:
            raise unittest.SkipTest(f"{case.draft_model_env} is not configured")

        wait_case = musa_server_smoke.MusaSmokeCase(
            name=case.name,
            model_env=case.model_env,
            tp_size=case.tp_size,
        )
        musa_server_smoke._wait_for_musa_free_memory(wait_case)

        # Keep the performance and acceptance workload identical to the DSpark
        # benchmark: prepared GSM8K prompts, Qwen chat template, and decode-only
        # timing after the first streamed token.
        performance_prompts = _build_decode_prompts(model)
        baseline_args = _common_server_args(case)
        speculative_args = _speculative_server_args(case, draft_model)
        _write_server_args_artifact(
            case, model, draft_model, baseline_args, speculative_args
        )

        baseline_process = baseline_logs = None
        speculative_process = speculative_logs = None
        try:
            baseline_process, baseline_logs = _launch_server(
                model, baseline_args, case.name, "baseline"
            )
            baseline_probe = _decode_probe(model, performance_prompts)
        finally:
            _stop_server(baseline_process, baseline_logs)

        # Give the driver a short window to release allocations before relaunching.
        time.sleep(float(os.getenv("MUSA_SPEC_RELAUNCH_DELAY", "5")))
        musa_server_smoke._wait_for_musa_free_memory(wait_case)

        try:
            speculative_process, speculative_logs = _launch_server(
                model, speculative_args, case.name, "speculative"
            )
            speculative_probe = _decode_probe(model, performance_prompts)
            server_accept_length = _server_accept_length()

            # Match the precision contract used by the other model CI tests:
            # speculative decoding is valid when the task accuracy reaches the
            # configured threshold.  Do not compare generated text with a
            # separately sampled baseline request byte-for-byte.
            eval_args = SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                model=model,
                eval_name=os.getenv("MUSA_SPEC_EVAL_NAME", "gsm8k"),
                api=os.getenv("MUSA_SPEC_EVAL_API", "completion"),
                max_tokens=int(os.getenv("MUSA_SPEC_EVAL_MAX_TOKENS", "512")),
                num_examples=int(os.getenv("MUSA_SPEC_EVAL_NUM_EXAMPLES", "200")),
                num_threads=int(os.getenv("MUSA_SPEC_EVAL_NUM_THREADS", "128")),
                num_shots=int(os.getenv("MUSA_SPEC_EVAL_NUM_SHOTS", "5")),
                temperature=float(os.getenv("MUSA_SPEC_EVAL_TEMPERATURE", "0.0")),
                top_p=float(os.getenv("MUSA_SPEC_EVAL_TOP_P", "1.0")),
                chat_template_kwargs=os.getenv("MUSA_SPEC_EVAL_CHAT_TEMPLATE_KWARGS"),
                reasoning_effort=os.getenv("MUSA_SPEC_EVAL_REASONING_EFFORT"),
                gsm8k_data_path=os.getenv("MUSA_SPEC_EVAL_GSM8K_DATA_PATH"),
            )
            accuracy_metrics = run_eval(eval_args)
        finally:
            _stop_server(speculative_process, speculative_logs)

        accept_length = server_accept_length
        min_accept_length = float(
            os.getenv(
                "MUSA_SPEC_MIN_ACCEPT_LENGTH", str(case.default_min_accept_length)
            )
        )
        baseline_tokens = baseline_probe["completion_tokens"]
        speculative_tokens = speculative_probe["completion_tokens"]
        baseline_latency = baseline_probe["decode_batch_s"]
        speculative_latency = speculative_probe["decode_batch_s"]
        baseline_tps = baseline_probe["decode_batch_tps"]
        speculative_tps = speculative_probe["decode_batch_tps"]
        speedup = speculative_tps / baseline_tps if baseline_tps > 0 else None
        min_speedup = float(
            os.getenv("MUSA_SPEC_MIN_SPEEDUP", str(case.default_min_speedup))
        )
        accuracy_threshold = float(
            os.getenv(
                "MUSA_SPEC_ACCURACY_THRESHOLD",
                str(case.default_accuracy_threshold),
            )
        )
        accuracy = float(accuracy_metrics["score"])

        metrics = {
            "eval_name": eval_args.eval_name,
            "dataset": eval_args.eval_name,
            "algorithm": case.algorithm,
            "model": model,
            "draft_model": draft_model,
            "num_examples_actual": accuracy_metrics.get("num_examples_actual"),
            "performance_num_examples": baseline_probe["num_examples"],
            "performance_num_threads": baseline_probe["num_threads"],
            "accuracy": accuracy,
            "score": accuracy,
            "accuracy_threshold": accuracy_threshold,
            "accuracy_eval_name": eval_args.eval_name,
            "baseline_output_tokens": baseline_tokens,
            "speculative_output_tokens": speculative_tokens,
            "baseline_latency": baseline_latency,
            "speculative_latency": speculative_latency,
            "baseline_tps": baseline_tps,
            "speculative_tps": speculative_tps,
            "speedup": speedup,
            "speedup_basis": "decode_batch_tps_ratio",
            "avg_spec_accept_length": accept_length,
            "server_avg_spec_accept_length": server_accept_length,
            "response_accept_length": None,
        }
        _write_metrics_artifact(case, metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)

        self.assertGreaterEqual(
            accuracy,
            accuracy_threshold,
            f"{eval_args.eval_name} accuracy below threshold",
        )
        self.assertIsNotNone(
            accept_length, "No speculative acceptance metric was exposed"
        )
        self.assertGreater(float(accept_length), min_accept_length)
        if min_speedup > 0:
            self.assertIsNotNone(speedup, "Unable to compute speculative speedup")
            self.assertGreater(speedup, min_speedup)
