from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
from tqdm import tqdm

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    MusaMoeBucket,
    get_moe_runner_backend,
    get_musa_moe_bucket_policy,
    set_musa_moe_bucket_policy,
)
from sglang.srt.utils import is_musa

logger = logging.getLogger(__name__)

_AUTOTUNE_MIN_TOKENS = 1
_AUTOTUNE_MAX_TOKENS = 65536
_AUTOTUNE_WARMUP = 3
_AUTOTUNE_ITERS = 7
_AUTOTUNE_WIN_MARGIN = 0.98
_AUTOTUNE_BLOCK_M_WIN_MARGIN = 1.0
_AUTOTUNE_MASKED_DEEPGEMM_MIN_THRESHOLD = 128
_DEEPGEMM_BLOCK_M_SMALL = 128
_DEEPGEMM_BLOCK_M_LARGE = 256
_TRITON_BACKEND = "triton"
_DEEPGEMM_BACKEND = "deepgemm"


@dataclass
class _MusaMoeAutotuneTarget:
    layer: torch.nn.Module
    quant_method: Any
    hidden_size: int
    num_experts: int
    num_local_experts: int
    top_k: int
    device: torch.device
    dtype: torch.dtype
    weight_dtype: torch.dtype


@dataclass(frozen=True)
class _MusaMoeAutotunePoint:
    tokens: int
    triton_us: float
    deepgemm_us: float
    winner: str
    block_m: Optional[int] = None


def maybe_autotune_musa_moe_deepgemm_threshold(
    model: torch.nn.Module,
    *,
    rank: int = 0,
) -> None:
    if not is_musa() or not get_moe_runner_backend().is_mixed():
        return

    target = _find_autotune_target(model)
    if target is None:
        logger.info(
            "Skip MUSA MoE DeepGEMM threshold autotune: no mixed MoE layer found."
        )
        return

    policy: tuple[MusaMoeBucket, ...] | None = None
    points: list[_MusaMoeAutotunePoint] = []
    try:
        points = _scan_bucket_points(target, rank=rank)
        policy = _build_bucket_policy(target, points)
    except Exception:
        logger.exception("MUSA MoE bucket autotune failed; fall back to Triton.")
    finally:
        policy = _broadcast_bucket_policy_from_tp_rank0(policy)
        set_musa_moe_bucket_policy(policy)
        if rank == 0:
            if policy is None:
                logger.info(
                    "MUSA MoE bucket autotune did not select a policy; using "
                    "Triton fallback.",
                )
            else:
                logger.info(
                    "MUSA MoE bucket autotune selected policy:\n%s\n%s",
                    _format_bucket_policy(policy),
                    _format_bucket_points(points),
                )
        _synchronize(target.device)
        torch.get_device_module(target.device.type).empty_cache()


def _find_autotune_target(model: torch.nn.Module) -> Optional[_MusaMoeAutotuneTarget]:
    for layer in model.modules():
        quant_method = getattr(layer, "quant_method", None)
        if not getattr(quant_method, "use_musa_contig_deepgemm", False):
            continue
        if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
            continue

        w13_weight = layer.w13_weight
        config = quant_method.moe_runner_config
        return _MusaMoeAutotuneTarget(
            layer=layer,
            quant_method=quant_method,
            hidden_size=int(config.hidden_size),
            num_experts=int(config.num_experts),
            num_local_experts=int(config.num_local_experts),
            top_k=int(config.top_k),
            device=w13_weight.device,
            dtype=torch.bfloat16,
            weight_dtype=w13_weight.dtype,
        )
    return None


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


def _scan_bucket_points(
    target: _MusaMoeAutotuneTarget, *, rank: int
) -> list[_MusaMoeAutotunePoint]:
    min_tokens = _AUTOTUNE_MIN_TOKENS
    max_tokens = max(min_tokens, _AUTOTUNE_MAX_TOKENS)
    candidates = _token_candidates(min_tokens, max_tokens)
    warmup = _AUTOTUNE_WARMUP
    iters = _AUTOTUNE_ITERS

    points: list[_MusaMoeAutotunePoint] = []
    pbar = tqdm(
        total=len(candidates),
        desc="MUSA MoE bucket autotune",
        disable=rank != 0,
        dynamic_ncols=True,
    )
    try:
        for tokens in candidates:
            point = _measure_bucket_point(target, tokens, warmup=warmup, iters=iters)
            points.append(point)
            pbar.update(1)
            pbar.set_postfix(
                tokens=tokens,
                triton=f"{point.triton_us:.1f}us",
                deepgemm=f"{point.deepgemm_us:.1f}us",
                winner=_format_winner(point),
            )
    finally:
        pbar.close()
    return points


def _measure_bucket_point(
    target: _MusaMoeAutotuneTarget,
    num_tokens: int,
    *,
    warmup: int,
    iters: int,
) -> _MusaMoeAutotunePoint:
    triton_us = _measure_one(
        target, num_tokens, backend=_TRITON_BACKEND, warmup=warmup, iters=iters
    )
    if target.num_experts > target.num_local_experts:
        deepgemm_us = _measure_one(
            target,
            num_tokens,
            backend=_DEEPGEMM_BACKEND,
            warmup=warmup,
            iters=iters,
        )
        winner = (
            _DEEPGEMM_BACKEND
            if deepgemm_us <= triton_us * _AUTOTUNE_WIN_MARGIN
            else _TRITON_BACKEND
        )
        return _MusaMoeAutotunePoint(
            tokens=num_tokens,
            triton_us=triton_us,
            deepgemm_us=deepgemm_us,
            winner=winner,
        )

    deepgemm_128_us = _measure_one(
        target,
        num_tokens,
        backend=_DEEPGEMM_BACKEND,
        block_m=_DEEPGEMM_BLOCK_M_SMALL,
        warmup=warmup,
        iters=iters,
    )
    deepgemm_256_us = _measure_one(
        target,
        num_tokens,
        backend=_DEEPGEMM_BACKEND,
        block_m=_DEEPGEMM_BLOCK_M_LARGE,
        warmup=warmup,
        iters=iters,
    )
    if deepgemm_256_us <= deepgemm_128_us * _AUTOTUNE_BLOCK_M_WIN_MARGIN:
        deepgemm_us = deepgemm_256_us
        block_m = _DEEPGEMM_BLOCK_M_LARGE
    else:
        deepgemm_us = deepgemm_128_us
        block_m = _DEEPGEMM_BLOCK_M_SMALL
    winner = (
        _DEEPGEMM_BACKEND
        if deepgemm_us <= triton_us * _AUTOTUNE_WIN_MARGIN
        else _TRITON_BACKEND
    )
    return _MusaMoeAutotunePoint(
        tokens=num_tokens,
        triton_us=triton_us,
        deepgemm_us=deepgemm_us,
        winner=winner,
        block_m=block_m,
    )


def _build_bucket_policy(
    target: _MusaMoeAutotuneTarget,
    points: list[_MusaMoeAutotunePoint],
) -> tuple[MusaMoeBucket, ...]:
    if not points:
        return (
            MusaMoeBucket(max_tokens=_AUTOTUNE_MAX_TOKENS, backend=_TRITON_BACKEND),
        )

    smoothed = _smooth_bucket_points(points)
    if target.num_experts > target.num_local_experts:
        smoothed = _apply_masked_deepgemm_guards(target, smoothed)

    buckets: list[MusaMoeBucket] = []
    current = smoothed[0]
    previous = smoothed[0]
    for next_point in smoothed[1:]:
        if (
            next_point.winner == current.winner
            and next_point.block_m == current.block_m
        ):
            previous = next_point
            continue
        buckets.append(_point_to_bucket(previous))
        current = next_point
        previous = next_point
    buckets.append(_point_to_bucket(previous))
    return tuple(buckets)


def _smooth_bucket_points(
    points: list[_MusaMoeAutotunePoint],
) -> list[_MusaMoeAutotunePoint]:
    if len(points) < 3:
        return points
    smoothed = list(points)
    for index in range(1, len(points) - 1):
        prev_point = smoothed[index - 1]
        point = smoothed[index]
        next_point = points[index + 1]
        prev_key = (prev_point.winner, prev_point.block_m)
        point_key = (point.winner, point.block_m)
        next_key = (next_point.winner, next_point.block_m)
        if prev_key == next_key and point_key != prev_key:
            smoothed[index] = _replace_point_choice(point, prev_point)
    return smoothed


def _apply_masked_deepgemm_guards(
    target: _MusaMoeAutotuneTarget,
    points: list[_MusaMoeAutotunePoint],
) -> list[_MusaMoeAutotunePoint]:
    guarded = []
    for point in points:
        if _is_fp8_dtype(target.weight_dtype) and point.winner == _DEEPGEMM_BACKEND:
            guarded.append(
                _MusaMoeAutotunePoint(
                    tokens=point.tokens,
                    triton_us=point.triton_us,
                    deepgemm_us=point.deepgemm_us,
                    winner=_TRITON_BACKEND,
                    block_m=None,
                )
            )
            continue
        if (
            point.tokens < _AUTOTUNE_MASKED_DEEPGEMM_MIN_THRESHOLD
            and point.winner == _DEEPGEMM_BACKEND
        ):
            guarded.append(
                _MusaMoeAutotunePoint(
                    tokens=point.tokens,
                    triton_us=point.triton_us,
                    deepgemm_us=point.deepgemm_us,
                    winner=_TRITON_BACKEND,
                    block_m=None,
                )
            )
            continue
        guarded.append(point)
    return guarded


def _replace_point_choice(
    point: _MusaMoeAutotunePoint,
    choice: _MusaMoeAutotunePoint,
) -> _MusaMoeAutotunePoint:
    return _MusaMoeAutotunePoint(
        tokens=point.tokens,
        triton_us=point.triton_us,
        deepgemm_us=point.deepgemm_us,
        winner=choice.winner,
        block_m=choice.block_m,
    )


def _point_to_bucket(point: _MusaMoeAutotunePoint) -> MusaMoeBucket:
    return MusaMoeBucket(
        max_tokens=point.tokens,
        backend=point.winner,
        block_m=point.block_m if point.winner == _DEEPGEMM_BACKEND else None,
    )


def _token_candidates(min_tokens: int, max_tokens: int) -> list[int]:
    candidates = {int(min_tokens), int(max_tokens)}
    tokens = 1
    while tokens < min_tokens:
        tokens *= 2
    while tokens <= max_tokens:
        candidates.add(tokens)
        tokens *= 2
    return sorted(candidates)


def _measure_one(
    target: _MusaMoeAutotuneTarget,
    num_tokens: int,
    *,
    backend: str,
    block_m: Optional[int] = None,
    warmup: int,
    iters: int,
) -> float:
    old_policy = get_musa_moe_bucket_policy()
    set_musa_moe_bucket_policy(
        (
            MusaMoeBucket(
                max_tokens=_AUTOTUNE_MAX_TOKENS,
                backend=backend,
                block_m=block_m if backend == _DEEPGEMM_BACKEND else None,
            ),
        )
    )
    try:
        for _ in range(warmup):
            _run_once(target, _make_dispatch_output(target, num_tokens))
        _synchronize(target.device)

        samples = []
        for _ in range(iters):
            dispatch_output = _make_dispatch_output(target, num_tokens)
            _synchronize(target.device)
            begin = time.perf_counter()
            _run_once(target, dispatch_output)
            _synchronize(target.device)
            samples.append((time.perf_counter() - begin) * 1e6)
        return statistics.median(samples)
    finally:
        set_musa_moe_bucket_policy(old_policy)


def _run_once(
    target: _MusaMoeAutotuneTarget,
    dispatch_output: StandardDispatchOutput,
) -> None:
    with torch.inference_mode():
        output = target.quant_method.apply(target.layer, dispatch_output)
        del output


def _make_dispatch_output(
    target: _MusaMoeAutotuneTarget,
    num_tokens: int,
) -> StandardDispatchOutput:
    hidden_states = torch.randn(
        (num_tokens, target.hidden_size),
        device=target.device,
        dtype=target.dtype,
    )
    topk_ids = torch.randint(
        0,
        target.num_local_experts,
        (num_tokens, target.top_k),
        device=target.device,
        dtype=torch.int32,
    )
    topk_weights = torch.rand(
        (num_tokens, target.top_k),
        device=target.device,
        dtype=torch.float32,
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    router_logits = torch.empty(
        (num_tokens, target.num_experts),
        device=target.device,
        dtype=torch.float32,
    )
    return StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(topk_weights, topk_ids, router_logits),
    )


def _synchronize(device: torch.device) -> None:
    torch.get_device_module(device.type).synchronize()


def _format_winner(point: _MusaMoeAutotunePoint) -> str:
    if point.winner == _TRITON_BACKEND:
        return _TRITON_BACKEND
    if point.block_m is None:
        return _DEEPGEMM_BACKEND
    return f"{_DEEPGEMM_BACKEND}:block_m={point.block_m}"


def _format_bucket_policy(policy: tuple[MusaMoeBucket, ...]) -> str:
    lines = []
    min_tokens = 1
    for index, bucket in enumerate(policy):
        backend = bucket.backend
        if bucket.block_m is not None:
            backend = f"{backend} block_m={bucket.block_m}"
        max_tokens = "inf" if index == len(policy) - 1 else str(bucket.max_tokens)
        lines.append(f"  [{min_tokens}, {max_tokens}] -> {backend}")
        min_tokens = bucket.max_tokens + 1
    return "\n".join(lines)


def _format_bucket_points(points: list[_MusaMoeAutotunePoint]) -> str:
    if not points:
        return "MUSA MoE bucket autotune measurements: <empty>"
    lines = ["MUSA MoE bucket autotune measurements:"]
    for point in points:
        ratio = (
            point.deepgemm_us / point.triton_us if point.triton_us > 0 else float("inf")
        )
        lines.append(
            "  tokens=%d triton=%.2fus deepgemm=%.2fus ratio=%.3f winner=%s"
            % (
                point.tokens,
                point.triton_us,
                point.deepgemm_us,
                ratio,
                _format_winner(point),
            )
        )
    return "\n".join(lines)


def _serialize_bucket_policy(
    policy: tuple[MusaMoeBucket, ...] | None,
) -> tuple[tuple[int, str, Optional[int]], ...] | None:
    if policy is None:
        return None
    return tuple(
        (bucket.max_tokens, bucket.backend, bucket.block_m) for bucket in policy
    )


def _deserialize_bucket_policy(
    policy: tuple[tuple[int, str, Optional[int]], ...] | None,
) -> tuple[MusaMoeBucket, ...] | None:
    if policy is None:
        return None
    return tuple(
        MusaMoeBucket(max_tokens=int(max_tokens), backend=backend, block_m=block_m)
        for max_tokens, backend, block_m in policy
    )


def _broadcast_bucket_policy_from_tp_rank0(
    policy: tuple[MusaMoeBucket, ...] | None,
) -> tuple[MusaMoeBucket, ...] | None:
    try:
        result = get_tp_group().broadcast_object(
            _serialize_bucket_policy(policy), src=0
        )
        return _deserialize_bucket_policy(result)
    except Exception:
        logger.debug(
            "Skip MUSA MoE bucket autotune broadcast; TP group is unavailable.",
            exc_info=True,
        )
        return policy
