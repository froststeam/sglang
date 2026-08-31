import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp
import requests

from sglang.bench_serving import RequestFuncOutput
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer, remove_prefix

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


async def async_request_sglang_generate(
    payload,
    url,
    pbar=None,
):
    """Send a streaming request to the server and collect cache metrics.

    Returns a RequestFuncOutput with additional cached_tokens and output_ids attributes.
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {}
        generated_text = ""
        all_output_ids = []
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0

                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st

                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # output_ids and text are always returned together
                            if data.get("output_ids"):
                                all_output_ids = data["output_ids"]
                                generated_text = data.get("text", "")
                                timestamp = time.perf_counter()

                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                    prompt_tokens = (data.get("meta_info") or {}).get(
                                        "prompt_tokens", 0
                                    )
                                    cached_tokens = (data.get("meta_info") or {}).get(
                                        "cached_tokens", 0
                                    )
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.output_ids = all_output_ids
                    output.success = True
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                    output.generated_len = len(output.itl) + 1
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    payload,
    url,
    pbar=None,
):
    """Send a streaming request to an OpenAI-compatible /v1/chat/completions endpoint.

    Returns a RequestFuncOutput with the same dynamic attributes as
    async_request_sglang_generate (except output_ids, which is unavailable).
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        generated_text = ""
        ttft = 0.0
        latency = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0
                    completion_tokens = 0

                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st

                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # Streaming token chunks
                            if data.get("choices"):
                                raw_delta = data["choices"][0].get("delta")
                                text = raw_delta.get("content", "") if raw_delta else ""
                                if text:
                                    generated_text += text
                                    timestamp = time.perf_counter()

                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft
                                    else:
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    most_recent_timestamp = timestamp

                            # Final chunk with usage stats
                            usage = data.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                details = usage.get("prompt_tokens_details", {}) or {}
                                cached_tokens = details.get("cached_tokens", 0)

                    output.generated_text = generated_text
                    output.output_ids = []  # Not available from OpenAI endpoint
                    output.success = True
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                    output.generated_len = (
                        completion_tokens if completion_tokens else len(output.itl) + 1
                    )
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    if pbar:
        pbar.update(1)
    return output


def gen_payload_openai(messages, output_len, model):
    return {
        "model": model,
        "messages": messages,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def gen_payload(input_ids, output_len, lora_path=""):
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "lora_path": lora_path,
        "return_logprob": False,
        "logprob_start_len": -1,
    }


async def _send_round(
    payloads,
    url,
    max_parallel,
):
    """Send a batch of payloads concurrently with concurrency limit."""
    semaphore = asyncio.Semaphore(max_parallel)

    async def _send_one(payload):
        async with semaphore:
            return await async_request_sglang_generate(payload, url)

    tasks = [asyncio.create_task(_send_one(p)) for p in payloads]
    return await asyncio.gather(*tasks)


def _get_page_size(base_url: str) -> int:
    """Query server for page_size used by radix cache."""
    try:
        resp = requests.get(f"{base_url}/server_info", timeout=10)
        resp.raise_for_status()
        info = resp.json()
        return info.get("page_size", 1)
    except Exception:
        return 1


def _safe_model_stem(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _valid_token_ids(tokenizer) -> list[int]:
    vocab = tokenizer.get_vocab()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_ids = sorted(
        int(token_id)
        for token_id in vocab.values()
        if isinstance(token_id, int) and token_id not in special_ids
    )
    if not token_ids:
        raise ValueError("Tokenizer did not expose any non-special token ids.")
    return token_ids


def _build_token_sequence(
    token_ids: list[int],
    length: int,
    offset: int,
    stride: int,
) -> list[int]:
    vocab_size = len(token_ids)
    return [token_ids[(offset + i * stride) % vocab_size] for i in range(length)]


def run_fixed_prefix_cache_hit_test(
    base_url: str,
    model_path: str,
    prefix_len: int = 1024,
    suffix_len: int = 1024,
    num_groups: int = 4,
    prompts_per_group: int = 4,
    output_len: int = 32,
    min_hit_rate: float = 0.45,
    max_parallel: int = 16,
    seed: int = 1,
) -> dict:
    """Run a deterministic shared-prefix workload and verify radix cache hits.

    Each group sends one warmup request and one branch-seed request before
    measured requests reuse exactly `prefix_len` input ids and diverge for
    `suffix_len` ids.
    """
    if prefix_len <= 0 or suffix_len <= 0:
        raise ValueError("prefix_len and suffix_len must be positive.")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")
    if prompts_per_group < 3:
        raise ValueError(
            "prompts_per_group must include warmup, branch-seed, and measured prompts."
        )

    generate_url = f"{base_url}/generate"
    tokenizer = get_tokenizer(model_path)
    token_ids = _valid_token_ids(tokenizer)
    page_size = max(int(_get_page_size(base_url)), 1)

    requests.post(f"{base_url}/flush_cache", timeout=30).raise_for_status()
    time.sleep(1)

    prefixes = []
    suffixes = []
    stride = 997
    for group_idx in range(num_groups):
        group_base = seed * 100003 + group_idx * 10007
        prefixes.append(
            _build_token_sequence(token_ids, prefix_len, group_base, stride)
        )
        suffixes.append(
            [
                _build_token_sequence(
                    token_ids,
                    suffix_len,
                    group_base + (prompt_idx + 1) * 1009,
                    stride + prompt_idx + 1,
                )
                for prompt_idx in range(prompts_per_group)
            ]
        )

    warmup_payloads = [
        gen_payload(prefixes[group_idx] + suffixes[group_idx][0], output_len)
        for group_idx in range(num_groups)
    ]
    warmup_responses = asyncio.run(
        _send_round(warmup_payloads, generate_url, max_parallel)
    )
    for group_idx, response in enumerate(warmup_responses):
        assert response.success, f"Warmup group {group_idx} failed: {response.error}"

    branch_seed_payloads = [
        gen_payload(prefixes[group_idx] + suffixes[group_idx][1], output_len)
        for group_idx in range(num_groups)
    ]
    branch_seed_responses = asyncio.run(
        _send_round(branch_seed_payloads, generate_url, max_parallel)
    )
    for group_idx, response in enumerate(branch_seed_responses):
        assert response.success, (
            f"Branch-seed group {group_idx} failed: {response.error}"
        )

    measured_payloads = []
    for group_idx in range(num_groups):
        for prompt_idx in range(2, prompts_per_group):
            measured_payloads.append(
                gen_payload(
                    prefixes[group_idx] + suffixes[group_idx][prompt_idx], output_len
                )
            )

    measured_responses = asyncio.run(
        _send_round(measured_payloads, generate_url, max_parallel)
    )

    per_request = []
    total_prompt = 0
    total_cached = 0
    total_ttft = 0.0
    expected_cached = (prefix_len // page_size) * page_size
    expected_hit_rate = expected_cached / (prefix_len + suffix_len)

    for request_idx, response in enumerate(measured_responses):
        assert response.success, (
            f"Measured request {request_idx} failed: {response.error}"
        )
        assert response.cached_tokens >= expected_cached, (
            f"Measured request {request_idx}: cached_tokens={response.cached_tokens}, "
            f"expected>={expected_cached}, page_size={page_size}"
        )

        total_prompt += response.prompt_len
        total_cached += response.cached_tokens
        total_ttft += response.ttft
        per_request.append(
            {
                "request_index": request_idx,
                "prompt_tokens": response.prompt_len,
                "cached_tokens": response.cached_tokens,
                "ttft": response.ttft,
            }
        )

    measured_count = len(measured_responses)
    cache_hit_rate = total_cached / total_prompt if total_prompt > 0 else 0.0
    average_ttft = total_ttft / measured_count if measured_count > 0 else 0.0
    assert cache_hit_rate >= min_hit_rate, (
        f"cache_hit_rate={cache_hit_rate:.4f} is below min_hit_rate={min_hit_rate:.4f}"
    )

    result = {
        "eval_name": "radix_prefix_cache",
        "model": model_path,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "prompt_len": prefix_len + suffix_len,
        "configured_prefix_ratio": prefix_len / (prefix_len + suffix_len),
        "cache_hit_rate": cache_hit_rate,
        "min_hit_rate": min_hit_rate,
        "expected_hit_rate": expected_hit_rate,
        "expected_cached_tokens_per_request": expected_cached,
        "page_size": page_size,
        "num_groups": num_groups,
        "prompts_per_group": prompts_per_group,
        "warmup_requests": len(warmup_responses),
        "branch_seed_requests": len(branch_seed_responses),
        "measured_requests": measured_count,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "average_ttft": average_ttft,
        "per_request": per_request,
    }

    report_dir = Path(
        os.environ.get("SGLANG_EVAL_REPORT_DIR")
        or os.environ.get("MUSA_SMOKE_ARTIFACT_DIR", "/tmp")
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    result_filename = (
        report_dir / f"radix_prefix_cache__{_safe_model_stem(model_path)}.json"
    )
    result_filename.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Writing radix prefix cache results to {result_filename}")
    print(
        f"  Radix prefix cache hit_rate={cache_hit_rate:.4f}, "
        f"cached={total_cached}/{total_prompt} tokens, page_size={page_size}"
    )

    return result


def run_multiturn_cache_hit_test(
    base_url: str,
    model_path: str,
    num_clients: int = 8,
    num_rounds: int = 3,
    request_length: int = 256,
    output_length: int = 32,
    miss_tolerance: int = 1,
    sub_question_input_length: int = 0,
    lora_path: str = "",
    dataset_path: str = "",
    max_parallel: int = 64,
    seed: int = 1,
) -> dict:
    """Run a multi-turn workload and verify cache hit rate.

    Sends requests in round-barrier mode: all clients complete round i
    before round i+1 starts, ensuring deterministic cache state.

    The expected cache hit rate is self-computed from the workload structure:
    - Round 0: expected cached_tokens = 0 (cold start after flush)
    - Round r (r >= 1): each client's prefix from round r-1 should be cached,
      minus up to previous round's (prompt_len + decoding output - miss_tolerance) // page * page.

    Returns metrics dict with per-round and overall cache_hit_rate.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    generate_url = f"{base_url}/generate"
    page_size = _get_page_size(base_url)

    # Flush cache for clean state
    requests.post(f"{base_url}/flush_cache")
    time.sleep(1)

    # Resolve sub-question length (0 means same as request_length)
    effective_sub_len = (
        sub_question_input_length if sub_question_input_length != 0 else request_length
    )

    # Sample initial prompts and sub-question prompts as token ids
    tokenizer = get_tokenizer(model_path)

    initial_inputs = sample_random_requests(
        input_len=request_length,
        output_len=output_length,
        num_prompts=num_clients,
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        return_text=False,
    )
    # r.prompt is now List[int] when return_text=False
    initial_token_ids = [list(r.prompt) for r in initial_inputs]

    sub_question_inputs = sample_random_requests(
        input_len=effective_sub_len,
        output_len=output_length,
        num_prompts=num_clients * max(num_rounds - 1, 1),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        return_text=False,
    )
    sub_question_token_ids = [list(r.prompt) for r in sub_question_inputs]

    # Per-round metrics and per-client tracking for expected cache computation
    round_metrics = {
        i: {"prompt_len": [], "cached_tokens": [], "ttft": []}
        for i in range(num_rounds)
    }
    # Track the previous round's prompt_len per client to compute expected cache
    prev_prompt_lens = [0] * num_clients
    # histories now stores List[int] (token ids) for each client
    histories = [list(ids) for ids in initial_token_ids]
    sub_idx = 0

    for round_num in range(num_rounds):
        payloads = [gen_payload(h, output_length, lora_path) for h in histories]
        responses = asyncio.run(_send_round(payloads, generate_url, max_parallel))

        for i, resp in enumerate(responses):
            assert resp.success, f"Round {round_num}, client {i} failed: {resp.error}"

            round_metrics[round_num]["prompt_len"].append(resp.prompt_len)
            round_metrics[round_num]["cached_tokens"].append(resp.cached_tokens)
            round_metrics[round_num]["ttft"].append(resp.ttft)

            # Verify cache hit against expected value
            if round_num == 0:
                # Cold start: no cache expected
                expected_cached = 0
            else:
                # Previous round's prompt + output are in cache.
                # Radix cache aligns to page_size, so the last partial page
                # may not be cached.
                cacheable = prev_prompt_lens[i] + output_length - miss_tolerance
                expected_cached = (cacheable // page_size) * page_size

            msg = (
                f"Round {round_num}, client {i}: "
                f"cached_tokens={resp.cached_tokens}, "
                f"expected>={expected_cached} "
                f"(prev_prompt={prev_prompt_lens[i]}, "
                f"output={output_length}, page_size={page_size})"
            )

            print(msg)

            assert resp.cached_tokens >= expected_cached

            # Record this round's prompt_len for next round's expected calc
            prev_prompt_lens[i] = resp.prompt_len

            # Accumulate history for next round using output_ids (token ids)
            histories[i].extend(resp.output_ids)
            if round_num < num_rounds - 1:
                histories[i].extend(sub_question_token_ids[sub_idx])
                sub_idx += 1

    # Compute per-round and overall cache hit rate
    total_prompt = 0
    total_cached = 0
    result = {"rounds": {}, "overall": {}}

    for r in range(num_rounds):
        rm = round_metrics[r]
        r_prompt = sum(rm["prompt_len"])
        r_cached = sum(rm["cached_tokens"])
        r_hit_rate = r_cached / r_prompt if r_prompt > 0 else 0.0
        r_avg_ttft = sum(rm["ttft"]) / len(rm["ttft"]) if rm["ttft"] else 0.0

        result["rounds"][f"round_{r}"] = {
            "cache_hit_rate": r_hit_rate,
            "average_ttft": r_avg_ttft,
            "total_prompt_tokens": r_prompt,
            "total_cached_tokens": r_cached,
            "request_count": len(rm["ttft"]),
        }

        total_prompt += r_prompt
        total_cached += r_cached

        print(
            f"  Round {r}: cache_hit_rate={r_hit_rate:.4f}, "
            f"avg_ttft={r_avg_ttft:.4f}s, "
            f"cached={r_cached}/{r_prompt} tokens"
        )

    overall_hit_rate = total_cached / total_prompt if total_prompt > 0 else 0.0
    result["overall"] = {
        "cache_hit_rate": overall_hit_rate,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
    }
    print(f"  Overall cache_hit_rate={overall_hit_rate:.4f}")

    return result
