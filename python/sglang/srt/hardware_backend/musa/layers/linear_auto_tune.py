from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.utils import is_musa

logger = logging.getLogger(__name__)

_GEMM_BACKEND = "gemm"
_GEMV_BACKEND = "gemv"
_AUTOTUNE_MAX_TOKENS = int(os.getenv("SGLANG_MUSA_LINEAR_AUTOTUNE_MAX_TOKENS", "8"))
_AUTOTUNE_WARMUP = int(os.getenv("SGLANG_MUSA_LINEAR_AUTOTUNE_WARMUP", "3"))
_AUTOTUNE_ITERS = int(os.getenv("SGLANG_MUSA_LINEAR_AUTOTUNE_ITERS", "7"))
_WIN_MARGIN = float(os.getenv("SGLANG_MUSA_LINEAR_GEMV_WIN_MARGIN", "0.98"))
_AUTOTUNE_PROFILER_TOPK = int(
    os.getenv("SGLANG_MUSA_LINEAR_AUTOTUNE_PROFILER_TOPK", "0")
)
_AUTOTUNE_LOG_EACH_POINT = (
    os.getenv("SGLANG_MUSA_LINEAR_AUTOTUNE_LOG_EACH_POINT", "0") == "1"
)
_DISABLE_POLICY = False
_LINEAR_POLICY: dict[tuple[str, torch.dtype, torch.dtype, int, int, int], str] = {}
_GEMV_CONFIGS: tuple[tuple[int, int], ...] = ((8, 16), (16, 8), (32, 4), (4, 32))


@dataclass(frozen=True)
class _LinearTarget:
    layer: torch.nn.Module
    quant_method: object
    quant_kind: str
    input_dtype: torch.dtype
    weight_dtype: torch.dtype
    k: int
    n: int
    device: torch.device


@contextmanager
def _disable_linear_policy():
    global _DISABLE_POLICY
    old_value = _DISABLE_POLICY
    _DISABLE_POLICY = True
    try:
        yield
    finally:
        _DISABLE_POLICY = old_value


def _policy_key(
    quant_kind: str,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    tokens: int,
    n: int,
    k: int,
) -> tuple[str, torch.dtype, torch.dtype, int, int, int]:
    return (quant_kind, input_dtype, weight_dtype, int(tokens), int(n), int(k))


def should_use_musa_linear_gemv(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    quant_kind: str,
) -> bool:
    if _DISABLE_POLICY or not is_musa() or not isinstance(x, torch.Tensor):
        return False
    if x.dim() < 2 or x.shape[-1] != layer.weight.shape[-1]:
        return False
    tokens = int(x.numel() // x.shape[-1])
    key = _policy_key(
        quant_kind,
        x.dtype,
        layer.weight.dtype,
        tokens,
        layer.weight.shape[0],
        layer.weight.shape[1],
    )
    return _LINEAR_POLICY.get(key) == _GEMV_BACKEND


def maybe_apply_musa_linear_activation(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    activation: str,
) -> Optional[torch.Tensor]:
    if _DISABLE_POLICY or not is_musa() or not isinstance(x, torch.Tensor):
        return None
    if getattr(layer, "skip_bias_add", False):
        return None
    if activation != "silu":
        return None

    quant_kind = _get_supported_linear_quant_kind(layer)
    if quant_kind is None or not should_use_musa_linear_gemv(
        layer, x, quant_kind=quant_kind
    ):
        return None

    from sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemm import musa_linear_gemv

    weight_scale = layer.weight_scale_inv if quant_kind == "fp8_block" else None
    output = musa_linear_gemv(
        x,
        layer.weight,
        weight_scale,
        bias=layer.bias,
        use_silu=True,
    )
    if getattr(layer, "gather_output", False):
        output = tensor_model_parallel_all_gather(output)
    return output


def maybe_apply_musa_linear_silu(
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> Optional[torch.Tensor]:
    return maybe_apply_musa_linear_activation(layer, x, activation="silu")


def _find_linear_targets(model: torch.nn.Module) -> list[_LinearTarget]:
    targets = []
    seen = set()
    for layer in model.modules():
        quant_method = getattr(layer, "quant_method", None)
        weight = getattr(layer, "weight", None)
        if weight is None or not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() != 2 or not weight.is_contiguous():
            continue

        quant_kind = _get_supported_linear_quant_kind(layer)
        input_dtype = torch.bfloat16

        if quant_kind is None:
            continue

        key = (
            quant_kind,
            input_dtype,
            weight.dtype,
            int(weight.shape[0]),
            int(weight.shape[1]),
        )
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            _LinearTarget(
                layer=layer,
                quant_method=quant_method,
                quant_kind=quant_kind,
                input_dtype=input_dtype,
                weight_dtype=weight.dtype,
                k=int(weight.shape[1]),
                n=int(weight.shape[0]),
                device=weight.device,
            )
        )
    return targets


def maybe_autotune_musa_linear_gemv(
    model: torch.nn.Module, *, rank: int = 0, reuse_only: bool = False
) -> None:
    if not is_musa() or _AUTOTUNE_MAX_TOKENS <= 0:
        return
    targets = _find_linear_targets(model)
    if not targets:
        return

    if reuse_only:
        missing_count = sum(
            _policy_key(
                target.quant_kind,
                target.input_dtype,
                target.weight_dtype,
                tokens,
                target.n,
                target.k,
            )
            not in _LINEAR_POLICY
            for target in targets
            for tokens in range(1, _AUTOTUNE_MAX_TOKENS + 1)
        )
        if rank == 0:
            if missing_count:
                logger.info(
                    "MUSA linear autotune skipped: reusing existing policy for %d targets; "
                    "%d token points are uncovered and will use the default GEMM path.",
                    len(targets),
                    missing_count,
                )
            else:
                logger.info(
                    "MUSA linear autotune skipped: existing policy covers %d targets and %d token points.",
                    len(targets),
                    len(targets) * _AUTOTUNE_MAX_TOKENS,
                )
        return

    new_policy: dict[tuple[str, torch.dtype, torch.dtype, int, int, int], str] = {}
    summaries: list[tuple[_LinearTarget, bool, str]] = []
    pbar = tqdm(
        total=len(targets) * _AUTOTUNE_MAX_TOKENS,
        desc="MUSA linear autotune",
        disable=rank != 0,
        dynamic_ncols=True,
    )
    try:
        with _disable_linear_policy():
            for target in targets:
                can_use_gemv = _can_use_musa_linear_gemv(target)
                target_policy: list[tuple[int, str]] = []
                for tokens in range(1, _AUTOTUNE_MAX_TOKENS + 1):
                    gemm_us = _measure_target(target, tokens, _GEMM_BACKEND)
                    gemv_us = (
                        _measure_target(target, tokens, _GEMV_BACKEND)
                        if can_use_gemv
                        else float("inf")
                    )
                    winner = (
                        _GEMV_BACKEND
                        if gemv_us < gemm_us * _WIN_MARGIN
                        else _GEMM_BACKEND
                    )
                    key = _policy_key(
                        target.quant_kind,
                        target.input_dtype,
                        target.weight_dtype,
                        tokens,
                        target.n,
                        target.k,
                    )
                    new_policy[key] = winner
                    target_policy.append((tokens, winner))
                    if rank == 0 and _AUTOTUNE_LOG_EACH_POINT:
                        logger.info(
                            "MUSA linear autotune: quant=%s tokens=%d n=%d k=%d "
                            "gemm=%.1fus gemv=%.1fus winner=%s",
                            target.quant_kind,
                            tokens,
                            target.n,
                            target.k,
                            gemm_us,
                            gemv_us,
                            winner,
                        )
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "quant": target.quant_kind,
                            "tokens": tokens,
                            "n": target.n,
                            "k": target.k,
                            "gemm": f"{gemm_us:.1f}us",
                            "gemv": (
                                "inf" if gemv_us == float("inf") else f"{gemv_us:.1f}us"
                            ),
                            "winner": winner,
                        }
                    )
                if rank == 0:
                    policy_ranges = _format_policy_ranges(target_policy)
                    summaries.append((target, can_use_gemv, policy_ranges))
                    logger.info(
                        "MUSA linear autotune summary: quant=%s n=%d k=%d gemv_supported=%s policy=%s",
                        target.quant_kind,
                        target.n,
                        target.k,
                        can_use_gemv,
                        policy_ranges,
                    )
    finally:
        pbar.close()

    _LINEAR_POLICY.clear()
    _LINEAR_POLICY.update(new_policy)
    if rank == 0 and summaries:
        logger.info(
            "MUSA linear autotune selected policy:\n%s",
            _format_linear_summaries(summaries),
        )


def _measure_target(target: _LinearTarget, tokens: int, backend: str) -> float:
    x = torch.randn((tokens, target.k), device=target.device, dtype=target.input_dtype)
    run = _make_runner(target, x, backend)
    for _ in range(_AUTOTUNE_WARMUP):
        run()
    _synchronize(target.device)
    avg_us = _measure_run_profiler_device_us(
        target,
        run,
        backend=backend,
        tokens=tokens,
        iters=_AUTOTUNE_ITERS,
    )
    if avg_us <= 0:
        raise RuntimeError("MUSA linear autotune profiler returned no GPU kernel time.")
    return avg_us


def _make_runner(
    target: _LinearTarget,
    x: torch.Tensor,
    backend: str,
) -> Callable[[], torch.Tensor]:
    if backend == _GEMV_BACKEND:
        return lambda: _run_gemv(target, x)
    if target.quant_kind == "bf16":
        return lambda: F.linear(x, target.layer.weight, None)
    return lambda: target.quant_method.w8a8_block_fp8_linear(
        input=x,
        weight=target.layer.weight,
        block_size=target.quant_method.quant_config.weight_block_size,
        weight_scale=target.layer.weight_scale_inv,
        input_scale=None,
        bias=None,
    )


def _run_gemv(target: _LinearTarget, x: torch.Tensor) -> torch.Tensor:
    from sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemm import musa_linear_gemv

    weight_scale = (
        target.layer.weight_scale_inv if target.quant_kind == "fp8_block" else None
    )
    return musa_linear_gemv(x, target.layer.weight, weight_scale)


def _get_supported_linear_quant_kind(layer: torch.nn.Module) -> Optional[str]:
    quant_method = getattr(layer, "quant_method", None)
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None

    method_name = quant_method.__class__.__name__ if quant_method is not None else ""
    if method_name == "UnquantizedLinearMethod" and weight.dtype == torch.bfloat16:
        return "bf16"
    if (
        method_name == "Fp8LinearMethod"
        and bool(getattr(quant_method, "block_quant", False))
        and weight.dtype == torch.float8_e4m3fn
        and getattr(layer, "weight_scale_inv", None) is not None
    ):
        return "fp8_block"
    return None


def _measure_run_profiler_device_us(
    target: _LinearTarget,
    run: Callable[[], torch.Tensor],
    *,
    backend: str,
    tokens: int,
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
        for _ in range(iters):
            run()
        _synchronize(target.device)

    rows = []
    for event in prof.key_averages():
        device_us = _profiler_self_device_time_us(event)
        if device_us > 0:
            rows.append((device_us, event.key))

    total_us = sum(device_us for device_us, _ in rows)
    avg_us = total_us / iters
    if _AUTOTUNE_PROFILER_TOPK > 0:
        rows.sort(reverse=True)
        top_rows = ", ".join(
            f"{name}={device_us:.1f}us"
            for device_us, name in rows[:_AUTOTUNE_PROFILER_TOPK]
        )
        logger.info(
            "MUSA linear autotune profiler: quant=%s tokens=%s n=%s k=%s backend=%s "
            "iters=%s gpu_kernel_sum=%.1fus avg=%.1fus%s",
            target.quant_kind,
            tokens,
            target.n,
            target.k,
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


def _format_policy_ranges(policy: list[tuple[int, str]]) -> str:
    if not policy:
        return "empty"

    ranges = []
    start_token, current_backend = policy[0]
    end_token = start_token
    for token, backend in policy[1:]:
        if backend == current_backend and token == end_token + 1:
            end_token = token
            continue
        ranges.append((start_token, end_token, current_backend))
        start_token = token
        end_token = token
        current_backend = backend
    ranges.append((start_token, end_token, current_backend))

    return ", ".join(
        f"{backend}[{start}]" if start == end else f"{backend}[{start}-{end}]"
        for start, end, backend in ranges
    )


def _format_linear_summaries(
    summaries: list[tuple[_LinearTarget, bool, str]],
) -> str:
    lines = []
    for target, can_use_gemv, policy_ranges in summaries:
        lines.append(
            "  quant=%s n=%d k=%d gemv_supported=%s policy=%s"
            % (
                target.quant_kind,
                target.n,
                target.k,
                can_use_gemv,
                policy_ranges,
            )
        )
    return "\n".join(lines)


def _can_use_musa_linear_gemv(target: _LinearTarget) -> bool:
    if target.n <= 0 or target.k <= 0:
        return False

    element_size = torch.empty((), dtype=target.weight_dtype).element_size()
    if element_size <= 0:
        return False
    vlen = 128 // (element_size * 8)
    if vlen <= 0:
        return False

    scale_k_group_tile = 1
    if target.quant_kind == "fp8_block":
        weight_scale = getattr(target.layer, "weight_scale_inv", None)
        if not isinstance(weight_scale, torch.Tensor):
            return False
        if weight_scale.dim() >= 2 and (
            weight_scale.shape[-2] != 1 or weight_scale.shape[-1] != 1
        ):
            scale_k_len = int(weight_scale.shape[-1])
            if scale_k_len <= 0:
                return False
            scale_k_group_tile = (target.k + scale_k_len - 1) // scale_k_len
            if scale_k_group_tile not in (64, 128):
                return False

    for block_n, block_k in _GEMV_CONFIGS:
        load_size = block_k * vlen
        if target.k % load_size == 0 and load_size % scale_k_group_tile == 0:
            return True

    return False


def _synchronize(device: torch.device) -> None:
    torch.get_device_module(device.type).synchronize()
