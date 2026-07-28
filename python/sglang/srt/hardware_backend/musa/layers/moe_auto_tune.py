from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
from tqdm import tqdm

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_tp_group,
)
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
_AUTOTUNE_MASKED_DEEPGEMM_MIN_THRESHOLD = 0
_AUTOTUNE_MASKED_DEEPGEMM_FREE_MEM_FRACTION = float(
    os.getenv("SGLANG_MUSA_MOE_MASKED_DEEPGEMM_FREE_MEM_FRACTION", "0.85")
)
_AUTOTUNE_MASKED_DEEPGEMM_MAX_BUFFER_GIB = float(
    os.getenv("SGLANG_MUSA_MOE_MASKED_DEEPGEMM_MAX_BUFFER_GIB", "0")
)
_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS = int(
    # Keep the M=1..32 decode window consistent with the Triton MoE runner.
    # The autotuner still measures the exact token buckets; this is only the
    # upper bound used when temporarily forcing the fused GEMV candidate.
    os.getenv("SGLANG_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS", "32")
)
_AUTOTUNE_PROFILER_TOPK = int(os.getenv("SGLANG_MUSA_MOE_AUTOTUNE_PROFILER_TOPK", "0"))
_DEEPGEMM_BLOCK_M_SMALL = 128
_DEEPGEMM_BLOCK_M_LARGE = 256
_TRITON_BACKEND = "triton"
_DEEPGEMM_BACKEND = "deepgemm"
_GEMV_BACKEND = "gemv"


@dataclass
class _MusaMoeAutotuneTarget:
    layer: torch.nn.Module
    quant_method: Any
    hidden_size: int
    num_experts: int
    num_local_experts: int
    num_fused_shared_experts: int
    moe_ep_size: int
    moe_ep_rank: int
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
    gemv_us: Optional[float] = None
    deepgemm_128_us: Optional[float] = None
    deepgemm_256_us: Optional[float] = None
    masked_deepgemm_us: Optional[float] = None
    block_m: Optional[int] = None
    use_contiguous_gemm: Optional[bool] = None
    masked_deepgemm_buffer_gib: Optional[float] = None


def maybe_autotune_musa_moe_deepgemm_threshold(
    model: torch.nn.Module,
    *,
    rank: int = 0,
    reuse_only: bool = False,
) -> None:
    if not is_musa() or not get_moe_runner_backend().is_mixed():
        return

    target = _find_autotune_target(model)
    if target is None:
        logger.info(
            "Skip MUSA MoE DeepGEMM threshold autotune: no mixed MoE layer found."
        )
        return

    if reuse_only:
        policy = get_musa_moe_bucket_policy()
        if rank == 0:
            if policy is None:
                logger.info(
                    "MUSA MoE bucket autotune skipped: no existing policy to reuse; "
                    "using Triton fallback."
                )
            else:
                logger.info(
                    "MUSA MoE bucket autotune skipped: reusing existing policy with "
                    "%d buckets.",
                    len(policy),
                )
        return

    policy: tuple[MusaMoeBucket, ...] | None = None
    points: list[_MusaMoeAutotunePoint] = []
    try:
        if rank == 0:
            points = _scan_bucket_points(
                target,
                rank=rank,
            )
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
                    _format_bucket_points(points, target),
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
            num_fused_shared_experts=int(config.num_fused_shared_experts or 0),
            moe_ep_size=_get_moe_ep_size(),
            moe_ep_rank=_get_moe_ep_rank(),
            top_k=int(config.top_k),
            device=w13_weight.device,
            dtype=torch.bfloat16,
            weight_dtype=w13_weight.dtype,
        )
    return None


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


def _uses_expert_parallel(target: _MusaMoeAutotuneTarget) -> bool:
    return target.moe_ep_size > 1 or target.num_experts > target.num_local_experts


def _get_moe_ep_size() -> int:
    try:
        return int(get_moe_expert_parallel_world_size())
    except Exception:
        return 1


def _get_moe_ep_rank() -> int:
    try:
        return int(get_moe_expert_parallel_rank())
    except Exception:
        return 0


def _scan_bucket_points(
    target: _MusaMoeAutotuneTarget,
    *,
    rank: int,
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
            try:
                point = _measure_bucket_point(
                    target, tokens, warmup=warmup, iters=iters
                )
            except torch.OutOfMemoryError:
                torch.get_device_module(target.device.type).empty_cache()
                logger.warning(
                    "MUSA MoE bucket autotune stops at tokens=%s because the "
                    "measurement ran out of memory; using previous points.",
                    tokens,
                )
                break
            points.append(point)
            pbar.update(1)
            postfix = {
                "tokens": tokens,
                "triton": f"{point.triton_us:.1f}us",
            }
            if point.gemv_us is not None:
                postfix["gemv"] = f"{point.gemv_us:.1f}us"
            if point.deepgemm_128_us is not None:
                postfix["dg128"] = f"{point.deepgemm_128_us:.1f}us"
            if point.deepgemm_256_us is not None:
                postfix["dg256"] = f"{point.deepgemm_256_us:.1f}us"
            if point.masked_deepgemm_us is not None:
                postfix["dgmasked"] = f"{point.masked_deepgemm_us:.1f}us"
            postfix["deepgemm"] = f"{point.deepgemm_us:.1f}us"
            postfix["winner"] = _format_winner(point)
            policy_winner = _format_winner(_guard_masked_deepgemm_point(target, point))
            if policy_winner != postfix["winner"]:
                postfix["policy"] = policy_winner
            if point.masked_deepgemm_buffer_gib is not None:
                postfix["masked_buf"] = f"{point.masked_deepgemm_buffer_gib:.1f}GiB"
            pbar.set_postfix(postfix)
            if point.deepgemm_us == float("inf"):
                break
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
        target,
        num_tokens,
        backend=_TRITON_BACKEND,
        warmup=warmup,
        iters=iters,
        gemv_max_tokens_override=0,
    )
    gemv_us: Optional[float] = None
    if 0 < num_tokens <= _MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS:
        gemv_us = _measure_one(
            target,
            num_tokens,
            backend=_GEMV_BACKEND,
            warmup=warmup,
            iters=iters,
            gemv_max_tokens_override=_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS,
        )
    masked_buffer_gib = _estimate_masked_deepgemm_buffer_gib(target, num_tokens)
    if _uses_expert_parallel(target):
        deepgemm_us = _measure_masked_deepgemm_if_safe(
            target,
            num_tokens,
            masked_buffer_gib=masked_buffer_gib,
            warmup=warmup,
            iters=iters,
        )
        if deepgemm_us is None:
            return _MusaMoeAutotunePoint(
                tokens=num_tokens,
                triton_us=triton_us,
                deepgemm_us=float("inf"),
                winner=_TRITON_BACKEND,
                gemv_us=gemv_us,
                block_m=None,
                use_contiguous_gemm=None,
                masked_deepgemm_us=None,
                masked_deepgemm_buffer_gib=masked_buffer_gib,
            )
        winner, winner_us = min(
            _candidate_times(triton_us, deepgemm_us, gemv_us),
            key=lambda item: item[1],
        )
        if winner_us > triton_us * _AUTOTUNE_WIN_MARGIN:
            winner = _TRITON_BACKEND
        return _MusaMoeAutotunePoint(
            tokens=num_tokens,
            triton_us=triton_us,
            deepgemm_us=deepgemm_us,
            winner=winner,
            gemv_us=gemv_us,
            masked_deepgemm_us=deepgemm_us,
            use_contiguous_gemm=False if winner == _DEEPGEMM_BACKEND else None,
            masked_deepgemm_buffer_gib=masked_buffer_gib,
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
    masked_deepgemm_us = _measure_masked_deepgemm_if_safe(
        target,
        num_tokens,
        masked_buffer_gib=masked_buffer_gib,
        warmup=warmup,
        iters=iters,
    )
    deepgemm_candidates = [
        (_DEEPGEMM_BACKEND, deepgemm_us, block_m, True),
    ]
    if masked_deepgemm_us is not None:
        deepgemm_candidates.append((_DEEPGEMM_BACKEND, masked_deepgemm_us, None, False))
    deepgemm_name, deepgemm_us, block_m, use_contiguous_gemm = min(
        deepgemm_candidates,
        key=lambda item: item[1],
    )
    assert deepgemm_name == _DEEPGEMM_BACKEND
    winner, winner_us = min(
        _candidate_times(triton_us, deepgemm_us, gemv_us),
        key=lambda item: item[1],
    )
    if winner_us > triton_us * _AUTOTUNE_WIN_MARGIN:
        winner = _TRITON_BACKEND
    return _MusaMoeAutotunePoint(
        tokens=num_tokens,
        triton_us=triton_us,
        deepgemm_us=deepgemm_us,
        winner=winner,
        gemv_us=gemv_us,
        deepgemm_128_us=deepgemm_128_us,
        deepgemm_256_us=deepgemm_256_us,
        masked_deepgemm_us=masked_deepgemm_us,
        block_m=block_m if winner == _DEEPGEMM_BACKEND else None,
        use_contiguous_gemm=(
            use_contiguous_gemm if winner == _DEEPGEMM_BACKEND else None
        ),
        masked_deepgemm_buffer_gib=masked_buffer_gib,
    )


def _measure_masked_deepgemm_if_safe(
    target: _MusaMoeAutotuneTarget,
    num_tokens: int,
    *,
    masked_buffer_gib: float,
    warmup: int,
    iters: int,
) -> Optional[float]:
    if not _masked_deepgemm_memory_guard_allows(target, masked_buffer_gib):
        logger.debug(
            "Skip MUSA MoE masked DeepGEMM autotune at tokens=%s: "
            "estimated buffer %.2fGiB exceeds memory guard.",
            num_tokens,
            masked_buffer_gib,
        )
        return None
    try:
        return _measure_one(
            target,
            num_tokens,
            backend=_DEEPGEMM_BACKEND,
            use_contiguous_gemm=False,
            warmup=warmup,
            iters=iters,
        )
    except torch.OutOfMemoryError:
        torch.get_device_module(target.device.type).empty_cache()
        logger.warning(
            "MUSA MoE masked DeepGEMM autotune OOM at tokens=%s; "
            "exclude masked DeepGEMM from this bucket.",
            num_tokens,
        )
        return None


def _masked_deepgemm_memory_guard_allows(
    target: _MusaMoeAutotuneTarget,
    masked_buffer_gib: float,
) -> bool:
    if _AUTOTUNE_MASKED_DEEPGEMM_MAX_BUFFER_GIB > 0:
        return masked_buffer_gib <= _AUTOTUNE_MASKED_DEEPGEMM_MAX_BUFFER_GIB
    device_module = torch.get_device_module(target.device.type)
    try:
        free_bytes, _ = device_module.mem_get_info(target.device)
    except Exception:
        try:
            with torch.device(target.device):
                free_bytes, _ = device_module.mem_get_info()
        except Exception:
            return True
    free_gib = free_bytes / (1024**3)
    return masked_buffer_gib <= free_gib * _AUTOTUNE_MASKED_DEEPGEMM_FREE_MEM_FRACTION


def _candidate_times(
    triton_us: float,
    deepgemm_us: float,
    gemv_us: Optional[float],
) -> list[tuple[str, float]]:
    candidates = [(_TRITON_BACKEND, triton_us), (_DEEPGEMM_BACKEND, deepgemm_us)]
    if gemv_us is not None:
        candidates.append((_GEMV_BACKEND, gemv_us))
    return candidates


def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 1
    return torch.empty((), dtype=dtype).element_size()


def _estimate_masked_deepgemm_buffer_gib(
    target: _MusaMoeAutotuneTarget,
    num_tokens: int,
) -> float:
    quant_config = getattr(target.quant_method, "quant_config", None)
    block_shape = getattr(quant_config, "weight_block_size", None)
    block_k = int(block_shape[1]) if block_shape is not None else 128
    scale_block_size = 128
    num_groups = target.num_local_experts
    m_max = (num_tokens // 256 + 1) * 256
    hidden_size = target.hidden_size
    gateup_n = int(target.layer.w13_weight.size(1))
    down_n = int(target.layer.w2_weight.size(1))
    is_fp8 = _is_fp8_dtype(target.weight_dtype)

    hidden_bytes = _dtype_nbytes(torch.float8_e4m3fn if is_fp8 else torch.bfloat16)
    total_bytes = (
        num_tokens * hidden_size * _dtype_nbytes(target.dtype)
        + num_tokens * target.num_experts * _dtype_nbytes(torch.float32)
        + num_tokens * target.top_k * 8
        + num_tokens * target.top_k * _dtype_nbytes(torch.int32)
        + num_groups * _dtype_nbytes(torch.int32)
        + num_groups * m_max * hidden_size * hidden_bytes
        + (
            num_groups
            * m_max
            * ((hidden_size + block_k - 1) // block_k)
            * _dtype_nbytes(torch.float32)
            if is_fp8
            else 0
        )
        + num_groups * m_max * gateup_n * _dtype_nbytes(torch.bfloat16)
        + num_groups * m_max * (gateup_n // 2) * hidden_bytes
        + num_groups
        * m_max
        * ((gateup_n // 2 + scale_block_size - 1) // scale_block_size)
        * _dtype_nbytes(torch.float32)
        + num_groups * m_max * down_n * _dtype_nbytes(torch.bfloat16)
    )
    return total_bytes / (1024**3)


def _build_bucket_policy(
    target: _MusaMoeAutotuneTarget,
    points: list[_MusaMoeAutotunePoint],
) -> tuple[MusaMoeBucket, ...]:
    if not points:
        return (
            MusaMoeBucket(max_tokens=_AUTOTUNE_MAX_TOKENS, backend=_TRITON_BACKEND),
        )

    smoothed = _smooth_bucket_points(points)
    smoothed = _apply_masked_deepgemm_guards(target, smoothed)

    buckets: list[MusaMoeBucket] = []
    current = smoothed[0]
    previous = smoothed[0]
    for next_point in smoothed[1:]:
        if (
            next_point.winner == current.winner
            and next_point.block_m == current.block_m
            and next_point.use_contiguous_gemm == current.use_contiguous_gemm
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
        prev_key = (
            prev_point.winner,
            prev_point.block_m,
            prev_point.use_contiguous_gemm,
        )
        point_key = (point.winner, point.block_m, point.use_contiguous_gemm)
        next_key = (
            next_point.winner,
            next_point.block_m,
            next_point.use_contiguous_gemm,
        )
        if prev_key == next_key and point_key != prev_key:
            smoothed[index] = _replace_point_choice(point, prev_point)
    return smoothed


def _apply_masked_deepgemm_guards(
    target: _MusaMoeAutotuneTarget,
    points: list[_MusaMoeAutotunePoint],
) -> list[_MusaMoeAutotunePoint]:
    guarded = []
    for point in points:
        guarded.append(_guard_masked_deepgemm_point(target, point))
    return guarded


def _guard_masked_deepgemm_point(
    target: _MusaMoeAutotuneTarget,
    point: _MusaMoeAutotunePoint,
) -> _MusaMoeAutotunePoint:
    if (
        point.tokens < _AUTOTUNE_MASKED_DEEPGEMM_MIN_THRESHOLD
        and point.winner == _DEEPGEMM_BACKEND
        and point.use_contiguous_gemm is False
    ):
        return _MusaMoeAutotunePoint(
            tokens=point.tokens,
            triton_us=point.triton_us,
            deepgemm_us=point.deepgemm_us,
            winner=_TRITON_BACKEND,
            gemv_us=point.gemv_us,
            deepgemm_128_us=point.deepgemm_128_us,
            deepgemm_256_us=point.deepgemm_256_us,
            masked_deepgemm_us=point.masked_deepgemm_us,
            block_m=None,
            masked_deepgemm_buffer_gib=point.masked_deepgemm_buffer_gib,
        )
    return point


def _replace_point_choice(
    point: _MusaMoeAutotunePoint,
    choice: _MusaMoeAutotunePoint,
) -> _MusaMoeAutotunePoint:
    return _MusaMoeAutotunePoint(
        tokens=point.tokens,
        triton_us=point.triton_us,
        deepgemm_us=point.deepgemm_us,
        winner=choice.winner,
        gemv_us=point.gemv_us,
        deepgemm_128_us=point.deepgemm_128_us,
        deepgemm_256_us=point.deepgemm_256_us,
        masked_deepgemm_us=point.masked_deepgemm_us,
        block_m=choice.block_m,
        use_contiguous_gemm=choice.use_contiguous_gemm,
        masked_deepgemm_buffer_gib=point.masked_deepgemm_buffer_gib,
    )


def _point_to_bucket(point: _MusaMoeAutotunePoint) -> MusaMoeBucket:
    return MusaMoeBucket(
        max_tokens=point.tokens,
        backend=point.winner,
        block_m=point.block_m if point.winner == _DEEPGEMM_BACKEND else None,
        use_contiguous_gemm=(
            point.use_contiguous_gemm if point.winner == _DEEPGEMM_BACKEND else None
        ),
    )


def _token_candidates(min_tokens: int, max_tokens: int) -> list[int]:
    candidates = {int(min_tokens), int(max_tokens)}
    dense_max_tokens = min(max_tokens, max(0, _MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS))
    if dense_max_tokens >= min_tokens:
        candidates.update(range(int(min_tokens), int(dense_max_tokens) + 1))
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
    use_contiguous_gemm: Optional[bool] = None,
    gemv_max_tokens_override: Optional[int] = None,
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
                use_contiguous_gemm=(
                    use_contiguous_gemm if backend == _DEEPGEMM_BACKEND else None
                ),
            ),
        )
    )
    try:
        gemv_context = (
            _temporary_musa_gemv_max_tokens(gemv_max_tokens_override)
            if gemv_max_tokens_override is not None
            else nullcontext()
        )
        with gemv_context:
            for _ in range(warmup):
                _run_once(target, _make_dispatch_output(target, num_tokens))
            _synchronize(target.device)

            dispatch_outputs = [
                _make_dispatch_output(target, num_tokens) for _ in range(iters)
            ]
            _synchronize(target.device)
            avg_us = _measure_run_profiler_device_us(
                target,
                dispatch_outputs,
                backend=backend,
                num_tokens=num_tokens,
                iters=iters,
            )
            if avg_us <= 0:
                raise RuntimeError(
                    "MUSA MoE bucket autotune profiler returned no GPU kernel time."
                )
            return avg_us
    finally:
        set_musa_moe_bucket_policy(old_policy)


@contextmanager
def _temporary_musa_gemv_max_tokens(max_tokens: int):
    from sglang.srt.layers.moe.moe_runner import triton as triton_runner

    old_value = triton_runner._MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS
    triton_runner._MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS = int(max_tokens)
    try:
        yield
    finally:
        triton_runner._MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS = old_value


def _measure_run_profiler_device_us(
    target: _MusaMoeAutotuneTarget,
    dispatch_outputs: list[StandardDispatchOutput],
    *,
    backend: str,
    num_tokens: int,
    iters: int,
) -> float:
    iters = max(1, iters)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
        with_flops=False,
        with_modules=False,
    ) as prof:
        for dispatch_output in dispatch_outputs:
            _run_once(target, dispatch_output)
        _synchronize(target.device)

    rows = []
    for event in prof.key_averages():
        device_us = _profiler_self_device_time_us(event)
        if device_us > 0:
            rows.append((device_us, event.key))

    total_us = sum(device_us for device_us, _ in rows)
    avg_us = total_us / max(1, iters)
    if _AUTOTUNE_PROFILER_TOPK > 0:
        rows.sort(reverse=True)
        top_rows = ", ".join(
            f"{name}={device_us:.1f}us"
            for device_us, name in rows[:_AUTOTUNE_PROFILER_TOPK]
        )
        logger.info(
            "MUSA MoE bucket autotune profiler: tokens=%s backend=%s "
            "iters=%s gpu_kernel_sum=%.1fus avg=%.1fus%s",
            num_tokens,
            backend,
            iters,
            total_us,
            avg_us,
            f" top=[{top_rows}]" if top_rows else "",
        )
    return avg_us


def _profiler_self_device_time_us(event: Any) -> float:
    return float(
        max(
            getattr(event, "self_device_time_total", 0.0),
            getattr(event, "self_cuda_time_total", 0.0),
            getattr(event, "self_xpu_time_total", 0.0),
        )
    )


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
    topk_ids = _make_synthetic_topk_ids(target, num_tokens)
    # Fused shared experts have fixed unit weight in the model contract.  Do
    # not normalize them together with routed experts: that would benchmark a
    # different workload from the actual dispatcher and understate shared
    # expert cost.  Routed weights are normalized only within the routed set.
    num_shared = max(0, min(target.num_fused_shared_experts, target.top_k))
    routed_top_k = target.top_k - num_shared
    topk_weights = torch.zeros(
        (num_tokens, target.top_k),
        device=target.device,
        dtype=torch.float32,
    )
    if routed_top_k > 0:
        routed_weights = torch.rand(
            (num_tokens, routed_top_k), device=target.device, dtype=torch.float32
        )
        valid_routed = topk_ids[:, :routed_top_k] >= 0
        routed_weights = torch.where(
            valid_routed, routed_weights, torch.zeros_like(routed_weights)
        )
        routed_sum = routed_weights.sum(dim=-1, keepdim=True)
        routed_weights = torch.where(
            routed_sum > 0,
            routed_weights / routed_sum.clamp_min(1e-20),
            routed_weights,
        )
        topk_weights[:, :routed_top_k] = routed_weights
    if num_shared > 0:
        topk_weights[:, routed_top_k:] = (topk_ids[:, routed_top_k:] >= 0).to(
            torch.float32
        )
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


def _make_synthetic_topk_ids(
    target: _MusaMoeAutotuneTarget,
    num_tokens: int,
) -> torch.Tensor:
    num_shared = max(0, min(target.num_fused_shared_experts, target.top_k))
    num_local_routed = target.num_local_experts - num_shared
    num_global_routed = target.num_experts
    routed_top_k = target.top_k - num_shared

    if _uses_expert_parallel(target) and num_local_routed > 0:
        pieces = []
        if routed_top_k > 0:
            # EP receives local experts after global routing; spread synthetic
            # global topk ids across EP shards, then apply the same local-id
            # mapping as StandardDispatcher.
            global_routed_ids = _make_strided_uniform_topk_ids(
                num_tokens,
                num_global_routed,
                routed_top_k,
                target.device,
            )
            pieces.append(_map_ep_global_routed_to_local(target, global_routed_ids))
        if num_shared > 0:
            pieces.append(
                _make_ep_shared_topk_ids(
                    target,
                    num_tokens,
                    num_shared,
                    num_local_routed,
                )
            )
        return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]

    return _make_balanced_topk_ids(
        num_tokens,
        target.num_local_experts,
        target.top_k,
        target.device,
    )


def _map_ep_global_routed_to_local(
    target: _MusaMoeAutotuneTarget,
    global_routed_ids: torch.Tensor,
) -> torch.Tensor:
    num_shared = max(0, min(target.num_fused_shared_experts, target.top_k))
    num_local_routed = target.num_local_experts - num_shared
    local_start = target.moe_ep_rank * num_local_routed
    local_end = local_start + num_local_routed
    is_local = (global_routed_ids >= local_start) & (global_routed_ids < local_end)
    local_ids = global_routed_ids - local_start
    return torch.where(is_local, local_ids, torch.full_like(global_routed_ids, -1))


def _make_ep_shared_topk_ids(
    target: _MusaMoeAutotuneTarget,
    num_tokens: int,
    num_shared: int,
    num_local_routed: int,
) -> torch.Tensor:
    if target.moe_ep_rank != 0:
        return torch.full(
            (num_tokens, num_shared),
            -1,
            device=target.device,
            dtype=torch.int32,
        )
    shared_ids = torch.arange(
        num_local_routed,
        num_local_routed + num_shared,
        device=target.device,
        dtype=torch.int32,
    )
    return shared_ids.expand(num_tokens, -1)


def _make_balanced_topk_ids(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    token_offsets = torch.arange(num_tokens, device=device).unsqueeze(1) * top_k
    topk_offsets = torch.arange(top_k, device=device).unsqueeze(0)
    return ((token_offsets + topk_offsets) % num_experts).to(torch.int32)


def _make_strided_uniform_topk_ids(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    stride = max(1, num_experts // max(1, top_k))
    token_offsets = torch.arange(num_tokens, device=device).unsqueeze(1)
    topk_offsets = torch.arange(top_k, device=device).unsqueeze(0) * stride
    return ((token_offsets + topk_offsets) % num_experts).to(torch.int32)


def _synchronize(device: torch.device) -> None:
    torch.get_device_module(device.type).synchronize()


def _format_winner(point: _MusaMoeAutotunePoint) -> str:
    if point.winner == _GEMV_BACKEND:
        return _GEMV_BACKEND
    if point.winner == _TRITON_BACKEND:
        return _TRITON_BACKEND
    if point.use_contiguous_gemm is False:
        return f"{_DEEPGEMM_BACKEND}:masked"
    if point.block_m is None:
        return _DEEPGEMM_BACKEND
    return f"{_DEEPGEMM_BACKEND}:contig:block_m={point.block_m}"


def _format_bucket_policy(policy: tuple[MusaMoeBucket, ...]) -> str:
    lines = []
    min_tokens = 1
    for index, bucket in enumerate(policy):
        backend = bucket.backend
        if bucket.backend == _DEEPGEMM_BACKEND:
            if bucket.use_contiguous_gemm is False:
                backend = f"{backend} masked"
            elif bucket.block_m is not None:
                backend = f"{backend} contig block_m={bucket.block_m}"
        max_tokens = "inf" if index == len(policy) - 1 else str(bucket.max_tokens)
        lines.append(f"  [{min_tokens}, {max_tokens}] -> {backend}")
        min_tokens = bucket.max_tokens + 1
    return "\n".join(lines)


def _format_bucket_points(
    points: list[_MusaMoeAutotunePoint],
    target: Optional[_MusaMoeAutotuneTarget] = None,
) -> str:
    if not points:
        return "MUSA MoE bucket autotune measurements: <empty>"
    lines = ["MUSA MoE bucket autotune measurements:"]
    if target is not None and _uses_expert_parallel(target):
        lines.append(
            "  ep_rank=%d ep_size=%d local_experts=%d global_experts=%d "
            "fused_shared=%d"
            % (
                target.moe_ep_rank,
                target.moe_ep_size,
                target.num_local_experts,
                target.num_experts,
                target.num_fused_shared_experts,
            )
        )
    for point in points:
        ratio = (
            point.deepgemm_us / point.triton_us if point.triton_us > 0 else float("inf")
        )
        gemv = " gemv=%.2fus" % point.gemv_us if point.gemv_us is not None else ""
        deepgemm_blocks = ""
        if point.deepgemm_128_us is not None:
            deepgemm_blocks += " deepgemm128=%.2fus" % point.deepgemm_128_us
        if point.deepgemm_256_us is not None:
            deepgemm_blocks += " deepgemm256=%.2fus" % point.deepgemm_256_us
        if point.masked_deepgemm_us is not None:
            deepgemm_blocks += " deepgemm_masked=%.2fus" % point.masked_deepgemm_us
        line = "  tokens=%d triton=%.2fus%s%s ratio=%.3f winner=%s" % (
            point.tokens,
            point.triton_us,
            gemv,
            deepgemm_blocks,
            ratio,
            _format_winner(point),
        )
        if target is not None:
            policy_winner = _format_winner(_guard_masked_deepgemm_point(target, point))
            if policy_winner != _format_winner(point):
                line += f" policy={policy_winner}"
        if point.masked_deepgemm_buffer_gib is not None:
            line += " masked_buf=%.2fGiB" % point.masked_deepgemm_buffer_gib
        lines.append(line)
    return "\n".join(lines)


def _serialize_bucket_policy(
    policy: tuple[MusaMoeBucket, ...] | None,
) -> tuple[tuple[int, str, Optional[int], Optional[bool]], ...] | None:
    if policy is None:
        return None
    return tuple(
        (
            bucket.max_tokens,
            bucket.backend,
            bucket.block_m,
            bucket.use_contiguous_gemm,
        )
        for bucket in policy
    )


def _deserialize_bucket_policy(
    policy: (
        tuple[
            tuple[int, str, Optional[int]]
            | tuple[int, str, Optional[int], Optional[bool]],
            ...,
        ]
        | None
    ),
) -> tuple[MusaMoeBucket, ...] | None:
    if policy is None:
        return None
    buckets = []
    for bucket in policy:
        max_tokens, backend, block_m, *rest = bucket
        use_contiguous_gemm = rest[0] if rest else None
        buckets.append(
            MusaMoeBucket(
                max_tokens=int(max_tokens),
                backend=backend,
                block_m=block_m,
                use_contiguous_gemm=use_contiguous_gemm,
            )
        )
    return tuple(buckets)


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
