from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.utils import is_musa

logger = logging.getLogger(__name__)

_GEMM_BACKEND = "gemm"
_GEMV_BACKEND = "gemv"
_GEMV_BACKEND_PREFIX = "gemv:"
_AUTOTUNE_MAX_TOKENS = int(os.getenv("SGLANG_MUSA_GEMV_AUTOTUNE_MAX_TOKENS", "32"))
_DEFAULT_AUTOTUNE_TOKENS = ",".join(str(tokens) for tokens in range(1, 33))
_AUTOTUNE_TOKENS = tuple(
    sorted(
        {
            int(value)
            for value in os.getenv(
                "SGLANG_MUSA_GEMV_AUTOTUNE_TOKENS", _DEFAULT_AUTOTUNE_TOKENS
            ).split(",")
            if value.strip() and 0 < int(value) <= _AUTOTUNE_MAX_TOKENS
        }
    )
)
_AUTOTUNE_WARMUP = int(os.getenv("SGLANG_MUSA_GEMV_AUTOTUNE_WARMUP", "3"))
_AUTOTUNE_ITERS = int(os.getenv("SGLANG_MUSA_GEMV_AUTOTUNE_ITERS", "7"))
_WIN_MARGIN = float(os.getenv("SGLANG_MUSA_GEMV_WIN_MARGIN", "0.98"))
_AUTOTUNE_PROFILER_TOPK = int(os.getenv("SGLANG_MUSA_GEMV_AUTOTUNE_PROFILER_TOPK", "0"))
_AUTOTUNE_LOG_EACH_POINT = (
    os.getenv("SGLANG_MUSA_GEMV_AUTOTUNE_LOG_EACH_POINT", "0") == "1"
)
_GEMV_CONFIG_ABI = 4
_DISABLE_POLICY = False
_GEMV_POLICY: dict[tuple[str, torch.dtype, torch.dtype, int, int, int], str] = {}
_ACTIVATION_POLICIES: dict[
    str, dict[tuple[str, torch.dtype, torch.dtype, int, int, int], str]
] = {"silu": {}, "swiglu": {}}
_ACTIVATION_LAYERS: WeakKeyDictionary[torch.nn.Module, str] = WeakKeyDictionary()
# Config ids are part of the JIT ABI. Keep this order in sync with configs[] in
# csrc/gemv/gemv.mu.
_GEMV_CONFIGS: tuple[tuple[int, int], ...] = (
    (8, 32),
    (8, 16),
    (16, 4),
    (16, 8),
    (32, 4),
    (4, 16),
    (4, 32),
    (32, 1),
    (128, 1),
)


@dataclass(frozen=True)
class _GemvTarget:
    layer: torch.nn.Module
    quant_method: object
    quant_kind: str
    input_dtype: torch.dtype
    weight_dtype: torch.dtype
    k: int
    n: int
    device: torch.device


@contextmanager
def _disable_gemv_policy():
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


def _resolve_activation(activation: Any) -> Optional[str]:
    if isinstance(activation, SiluAndMul):
        return "swiglu"
    if isinstance(activation, str) and activation in _ACTIVATION_POLICIES:
        return activation
    return None


def _has_musa_gemv_layout(x: torch.Tensor) -> bool:
    """Return whether x has exact row-major strides, including size-1 dims."""
    if x.dim() < 2:
        return False
    expected_stride = 1
    for dim in range(x.dim() - 1, -1, -1):
        if x.stride(dim) != expected_stride:
            return False
        expected_stride *= x.shape[dim]
    return True


def register_musa_gemv_activation(layer: torch.nn.Module, activation: Any) -> None:
    activation_name = _resolve_activation(activation)
    if activation_name is None:
        _ACTIVATION_LAYERS.pop(layer, None)
    else:
        _ACTIVATION_LAYERS[layer] = activation_name


def should_use_musa_gemv(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    quant_kind: str,
) -> bool:
    if _DISABLE_POLICY or not is_musa() or not isinstance(x, torch.Tensor):
        return False
    if x.dim() < 2 or x.shape[-1] != layer.weight.shape[-1]:
        return False
    if not _has_musa_gemv_layout(x):
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
    return _GEMV_POLICY.get(key, "").startswith(_GEMV_BACKEND)


def get_musa_gemv_config(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    quant_kind: str,
) -> int:
    if not should_use_musa_gemv(layer, x, quant_kind=quant_kind):
        return -1
    tokens = int(x.numel() // x.shape[-1])
    backend = _GEMV_POLICY.get(
        _policy_key(
            quant_kind,
            x.dtype,
            layer.weight.dtype,
            tokens,
            layer.weight.shape[0],
            layer.weight.shape[1],
        ),
        _GEMV_BACKEND,
    )
    if not backend.startswith(_GEMV_BACKEND_PREFIX):
        return -1
    return int(backend[len(_GEMV_BACKEND_PREFIX) :])


def maybe_apply_musa_gemv_activation(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    activation: Any,
) -> Optional[torch.Tensor]:
    if _DISABLE_POLICY or not is_musa() or not isinstance(x, torch.Tensor):
        return None
    if x.dim() < 2 or x.shape[-1] != layer.weight.shape[-1]:
        return None
    if not _has_musa_gemv_layout(x):
        return None
    activation_name = _resolve_activation(activation)
    if getattr(layer, "skip_bias_add", False) or activation_name is None:
        return None
    if activation_name == "swiglu" and getattr(layer, "bias", None) is not None:
        return None

    quant_kind = _get_supported_gemv_quant_kind(layer)
    if quant_kind is None:
        return None

    key = _policy_key(
        quant_kind,
        x.dtype,
        layer.weight.dtype,
        int(x.numel() // x.shape[-1]),
        layer.weight.shape[0],
        layer.weight.shape[1],
    )
    backend = _ACTIVATION_POLICIES[activation_name].get(key, "")
    if not backend.startswith(_GEMV_BACKEND):
        return None

    from sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemv import musa_gemv

    weight_scale = layer.weight_scale_inv if quant_kind == "fp8_block" else None
    output = musa_gemv(
        x,
        layer.weight,
        B_scale=weight_scale,
        bias=layer.bias,
        fuse_swiglu=activation_name == "swiglu",
        fuse_silu=activation_name == "silu",
        config_id=(
            int(backend[len(_GEMV_BACKEND_PREFIX) :])
            if backend.startswith(_GEMV_BACKEND_PREFIX)
            else -1
        ),
    )
    if getattr(layer, "gather_output", False):
        output = tensor_model_parallel_all_gather(output)
    return output


def _find_gemv_targets(model: torch.nn.Module) -> list[_GemvTarget]:
    targets = []
    seen = set()
    for layer in model.modules():
        quant_method = getattr(layer, "quant_method", None)
        weight = getattr(layer, "weight", None)
        if weight is None or not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() != 2 or not weight.is_contiguous():
            continue

        quant_kind = _get_supported_gemv_quant_kind(layer)
        input_dtype = torch.bfloat16

        if quant_kind is None:
            continue

        key = (
            quant_kind,
            input_dtype,
            weight.dtype,
            int(weight.shape[0]),
            int(weight.shape[1]),
            _ACTIVATION_LAYERS.get(layer),
        )
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            _GemvTarget(
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


def maybe_autotune_musa_gemv(
    model: torch.nn.Module, *, rank: int = 0, reuse_only: bool = False
) -> None:
    if not is_musa() or not _AUTOTUNE_TOKENS:
        return
    targets = _find_gemv_targets(model)
    if not targets:
        return
    activation_targets = {
        activation: [
            target
            for target in targets
            if _ACTIVATION_LAYERS.get(target.layer) == activation
            and (activation != "swiglu" or target.n % 2 == 0)
        ]
        for activation in ("silu", "swiglu")
    }

    # A persisted policy is an exact-shape cache, not a heuristic override.
    # Entries from another device architecture or kernel ABI are ignored.
    _GEMV_POLICY.update(_load_gemv_policy(targets[0].device))

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
            not in _GEMV_POLICY
            for target in targets
            for tokens in _AUTOTUNE_TOKENS
        )
        missing_count += sum(
            _policy_key(
                target.quant_kind,
                target.input_dtype,
                target.weight_dtype,
                tokens,
                target.n,
                target.k,
            )
            not in _ACTIVATION_POLICIES[activation]
            for activation, activation_targets_for_kind in activation_targets.items()
            for target in activation_targets_for_kind
            for tokens in _AUTOTUNE_TOKENS
        )
        if rank == 0:
            if missing_count:
                logger.info(
                    "MUSA dense GEMV autotune skipped: reusing existing policy for %d targets; "
                    "%d token points are uncovered and will use the default GEMM path.",
                    len(targets),
                    missing_count,
                )
            else:
                logger.info(
                    "MUSA dense GEMV autotune skipped: existing policy covers %d targets and %d token points.",
                    len(targets)
                    + sum(len(value) for value in activation_targets.values()),
                    (
                        len(targets)
                        + sum(len(value) for value in activation_targets.values())
                    )
                    * len(_AUTOTUNE_TOKENS),
                )
        return

    new_policy = dict(_GEMV_POLICY)
    new_activation_policies = {
        activation: dict(_ACTIVATION_POLICIES[activation])
        for activation in activation_targets
    }
    summaries: list[tuple[_GemvTarget, bool, str]] = []
    pbar = tqdm(
        total=(len(targets) + sum(len(value) for value in activation_targets.values()))
        * len(_AUTOTUNE_TOKENS),
        desc="MUSA dense GEMV autotune",
        disable=rank != 0,
        dynamic_ncols=True,
    )
    try:
        with _disable_gemv_policy():
            for target in targets:
                can_use_gemv = _can_use_musa_gemv(target)
                target_policy: list[tuple[int, str]] = []
                for tokens in _AUTOTUNE_TOKENS:
                    key = _policy_key(
                        target.quant_kind,
                        target.input_dtype,
                        target.weight_dtype,
                        tokens,
                        target.n,
                        target.k,
                    )
                    cached_backend = new_policy.get(key)
                    if cached_backend is not None:
                        target_policy.append((tokens, cached_backend))
                        pbar.update(1)
                        continue
                    gemm_us = _measure_target(target, tokens, _GEMM_BACKEND)
                    gemv_results = (
                        [
                            (backend, _measure_target(target, tokens, backend))
                            for backend in _candidate_gemv_backends(target, tokens)
                        ]
                        if can_use_gemv
                        else []
                    )
                    best_gemv_backend, gemv_us = min(
                        gemv_results,
                        key=lambda item: item[1],
                        default=(_GEMV_BACKEND, float("inf")),
                    )
                    winner = (
                        best_gemv_backend
                        if gemv_us < gemm_us * _WIN_MARGIN
                        else _GEMM_BACKEND
                    )
                    new_policy[key] = winner
                    target_policy.append((tokens, winner))
                    if rank == 0 and _AUTOTUNE_LOG_EACH_POINT:
                        logger.info(
                            "MUSA dense GEMV autotune: quant=%s tokens=%d n=%d k=%d "
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
                        "MUSA dense GEMV autotune summary: quant=%s n=%d k=%d gemv_supported=%s policy=%s",
                        target.quant_kind,
                        target.n,
                        target.k,
                        can_use_gemv,
                        policy_ranges,
                    )

            # Fused activation changes the cost model: compare the complete
            # GEMV+activation path against GEMM+activation instead of reusing
            # the plain-linear policy.
            for activation, activation_targets_for_kind in activation_targets.items():
                activation_policy = new_activation_policies[activation]
                target_policy: list[tuple[int, str]] = []
                for target in activation_targets_for_kind:
                    target_policy.clear()
                    for tokens in _AUTOTUNE_TOKENS:
                        key = _policy_key(
                            target.quant_kind,
                            target.input_dtype,
                            target.weight_dtype,
                            tokens,
                            target.n,
                            target.k,
                        )
                        cached_backend = activation_policy.get(key)
                        if cached_backend is not None:
                            target_policy.append((tokens, cached_backend))
                            pbar.update(1)
                            continue
                        gemm_us = _measure_activation_target(
                            target, tokens, _GEMM_BACKEND, activation
                        )
                        gemv_us = _measure_activation_target(
                            target, tokens, _GEMV_BACKEND, activation
                        )
                        winner = (
                            _GEMV_BACKEND
                            if gemv_us < gemm_us * _WIN_MARGIN
                            else _GEMM_BACKEND
                        )
                        activation_policy[key] = winner
                        target_policy.append((tokens, winner))
                        pbar.update(1)
                    if rank == 0:
                        logger.info(
                            "MUSA %s GEMV autotune summary: quant=%s n=%d k=%d policy=%s",
                            activation,
                            target.quant_kind,
                            target.n,
                            target.k,
                            _format_policy_ranges(target_policy),
                        )
    finally:
        pbar.close()

    _GEMV_POLICY.update(new_policy)
    for activation, policy in new_activation_policies.items():
        _ACTIVATION_POLICIES[activation].update(policy)
    if rank == 0:
        _save_gemv_policy(targets[0].device, _GEMV_POLICY)
    if rank == 0 and summaries:
        logger.info(
            "MUSA dense GEMV autotune selected policy:\n%s",
            _format_gemv_summaries(summaries),
        )


def _gemv_config_path() -> Optional[Path]:
    value = os.getenv("SGLANG_MUSA_GEMV_AUTOTUNE_CONFIG")
    if value:
        if value.lower() in {"0", "off", "none", "disabled"}:
            return None
        return Path(value).expanduser()
    cache_home = os.getenv("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "sglang" / "musa" / "gemv_config.json"
    return Path.home() / ".cache" / "sglang" / "musa" / "gemv_config.json"


def _device_capability(device: torch.device) -> tuple[int, int]:
    device_module = torch.get_device_module(device.type)
    major, minor = device_module.get_device_capability(device)
    return int(major), int(minor)


def _cache_dtype_pair(name: object) -> Optional[tuple[str, torch.dtype, torch.dtype]]:
    if name == "bf16":
        return "bf16", torch.bfloat16, torch.bfloat16
    if name == "fp8_block":
        return "fp8_block", torch.bfloat16, torch.float8_e4m3fn
    return None


def _valid_persisted_backend(backend: object) -> bool:
    if backend in (_GEMM_BACKEND, _GEMV_BACKEND):
        return True
    if not isinstance(backend, str) or not backend.startswith(_GEMV_BACKEND_PREFIX):
        return False
    try:
        config_id = int(backend[len(_GEMV_BACKEND_PREFIX) :])
    except ValueError:
        return False
    return 0 <= config_id < len(_GEMV_CONFIGS)


def _load_gemv_policy(
    device: torch.device,
) -> dict[tuple[str, torch.dtype, torch.dtype, int, int, int], str]:
    for activation_policy in _ACTIVATION_POLICIES.values():
        activation_policy.clear()
    path = _gemv_config_path()
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return {}
        if int(payload.get("config_abi", -1)) != _GEMV_CONFIG_ABI:
            return {}
        capability = tuple(int(v) for v in payload.get("device_capability", ()))
        if capability != _device_capability(device):
            return {}
        records = payload.get("records")
        if not isinstance(records, list):
            return {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}

    policy: dict[tuple[str, torch.dtype, torch.dtype, int, int, int], str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        dtype_pair = _cache_dtype_pair(record.get("dtype"))
        backend = record.get("backend")
        if dtype_pair is None or not _valid_persisted_backend(backend):
            continue
        quant_kind, input_dtype, weight_dtype = dtype_pair
        try:
            key = _policy_key(
                quant_kind,
                input_dtype,
                weight_dtype,
                int(record["tokens"]),
                int(record["n"]),
                int(record["k"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        activation = record.get("activation", "linear")
        if activation in _ACTIVATION_POLICIES:
            _ACTIVATION_POLICIES[activation][key] = backend
        elif activation == "linear":
            policy[key] = backend
    return policy


def _save_gemv_policy(
    device: torch.device,
    policy: dict[tuple[str, torch.dtype, torch.dtype, int, int, int], str],
) -> None:
    path = _gemv_config_path()
    if path is None:
        return
    records = [
        {
            "dtype": quant_kind,
            "tokens": tokens,
            "n": n,
            "k": k,
            "backend": backend,
        }
        for (quant_kind, input_dtype, weight_dtype, tokens, n, k), backend in sorted(
            policy.items(),
            key=lambda item: (
                item[0][0],
                str(item[0][1]),
                str(item[0][2]),
                item[0][4],
                item[0][5],
                item[0][3],
            ),
        )
    ]
    for activation, activation_policy in _ACTIVATION_POLICIES.items():
        records.extend(
            {
                "dtype": quant_kind,
                "tokens": tokens,
                "n": n,
                "k": k,
                "backend": backend,
                "activation": activation,
            }
            for (
                quant_kind,
                input_dtype,
                weight_dtype,
                tokens,
                n,
                k,
            ), backend in sorted(
                activation_policy.items(),
                key=lambda item: (
                    item[0][0],
                    str(item[0][1]),
                    str(item[0][2]),
                    item[0][4],
                    item[0][5],
                    item[0][3],
                ),
            )
        )
    payload = {
        "config_abi": _GEMV_CONFIG_ABI,
        "device_capability": list(_device_capability(device)),
        "records": records,
    }
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_path.replace(path)
    except OSError:
        logger.exception("Failed to save MUSA dense GEMV autotune policy to %s", path)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _measure_target(target: _GemvTarget, tokens: int, backend: str) -> float:
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
        raise RuntimeError(
            "MUSA dense GEMV autotune profiler returned no GPU kernel time."
        )
    return avg_us


def _measure_activation_target(
    target: _GemvTarget, tokens: int, backend: str, activation: str
) -> float:
    x = torch.randn((tokens, target.k), device=target.device, dtype=target.input_dtype)
    run = _make_activation_runner(target, x, backend, activation)
    for _ in range(_AUTOTUNE_WARMUP):
        run()
    _synchronize(target.device)
    avg_us = _measure_run_profiler_device_us(
        target,
        run,
        backend=f"{activation}:{backend}",
        tokens=tokens,
        iters=_AUTOTUNE_ITERS,
    )
    if avg_us <= 0:
        raise RuntimeError(
            "MUSA dense activation GEMV autotune profiler returned no GPU kernel time."
        )
    return avg_us


def _make_runner(
    target: _GemvTarget,
    x: torch.Tensor,
    backend: str,
) -> Callable[[], torch.Tensor]:
    if backend.startswith(_GEMV_BACKEND):
        config_id = (
            int(backend[len(_GEMV_BACKEND_PREFIX) :])
            if backend.startswith(_GEMV_BACKEND_PREFIX)
            else -1
        )
        return lambda: _run_gemv(target, x, config_id=config_id)
    if target.quant_kind == "bf16":
        return lambda: F.linear(x, target.layer.weight, target.layer.bias)
    return lambda: target.quant_method.w8a8_block_fp8_linear(
        input=x,
        weight=target.layer.weight,
        block_size=target.quant_method.quant_config.weight_block_size,
        weight_scale=target.layer.weight_scale_inv,
        input_scale=None,
        bias=target.layer.bias,
    )


def _make_activation_runner(
    target: _GemvTarget, x: torch.Tensor, backend: str, activation: str
) -> Callable[[], torch.Tensor]:
    if backend.startswith(_GEMV_BACKEND):
        config_id = (
            int(backend[len(_GEMV_BACKEND_PREFIX) :])
            if backend.startswith(_GEMV_BACKEND_PREFIX)
            else -1
        )
        return lambda: _run_gemv(target, x, config_id=config_id, activation=activation)

    linear = _make_runner(target, x, _GEMM_BACKEND)
    return lambda: _apply_activation(linear(), activation)


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    half = x.shape[-1] // 2
    if activation == "swiglu":
        return F.silu(x[..., :half]) * x[..., half:]
    return F.silu(x)


def _run_gemv(
    target: _GemvTarget,
    x: torch.Tensor,
    *,
    config_id: int = -1,
    activation: Optional[str] = None,
) -> torch.Tensor:
    from sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemv import (
        musa_gemv,
    )

    weight_scale = (
        target.layer.weight_scale_inv if target.quant_kind == "fp8_block" else None
    )
    return musa_gemv(
        x,
        target.layer.weight,
        weight_scale,
        bias=target.layer.bias,
        fuse_swiglu=activation == "swiglu",
        fuse_silu=activation == "silu",
        config_id=config_id,
    )


def _candidate_gemv_backends(target: _GemvTarget, tokens: int) -> list[str]:
    valid = _valid_gemv_config_ids(target)
    if not valid:
        return []

    element_size = torch.empty((), dtype=target.weight_dtype).element_size()
    vlen = 16 // element_size

    def estimated_cost(config_id: int) -> float:
        block_n, block_k = _GEMV_CONFIGS[config_id]
        n_blocks = (target.n + block_n - 1) // block_n
        k_iters = (target.k + block_k * vlen - 1) // (block_k * vlen)
        tail_penalty = 0.75 if target.n % block_n else 0.0
        reduction_penalty = 0.04 * block_k
        token_penalty = 1.0 + 0.08 * max(tokens - 1, 0)
        return n_blocks * (k_iters + reduction_penalty + tail_penalty) * token_penalty

    # Shape ratios alone are not a reliable proxy for occupancy, tail cost and
    # cache behavior. Benchmark every legal launch configuration and let the
    # per-(dtype, tokens, N, K) policy retain only a measured winner.
    ranked = sorted(valid, key=estimated_cost)
    # config_id=-1 contains the shape-specialized families (WMMA, fused
    # SwiGLU and multi-token kernels). Forced generic configs intentionally
    # bypass those paths, so omitting the auto backend made production
    # autotune unable to select the fastest structural implementation.
    return [
        _GEMV_BACKEND,
        *(f"{_GEMV_BACKEND_PREFIX}{config_id}" for config_id in ranked),
    ]


def _valid_gemv_config_ids(target: _GemvTarget) -> list[int]:
    element_size = torch.empty((), dtype=target.weight_dtype).element_size()
    if element_size <= 0:
        return []
    vlen = 16 // element_size
    scale_k_group_tile = _scale_k_group_tile(target)
    if scale_k_group_tile is None:
        return []
    result = []
    for config_id, (block_n, block_k) in enumerate(_GEMV_CONFIGS):
        load_size = block_k * vlen
        # Generic GEMV masks N tails but performs full vector loads along K.
        # Keep startup autotune aligned with the C++ validity guard so an
        # invalid forced config can never enter a persisted policy.
        if load_size % scale_k_group_tile != 0 or target.k % load_size != 0:
            continue
        result.append(config_id)
    return result


def _scale_k_group_tile(target: _GemvTarget) -> Optional[int]:
    if target.quant_kind != "fp8_block":
        return 1
    weight_scale = getattr(target.layer, "weight_scale_inv", None)
    if not isinstance(weight_scale, torch.Tensor):
        return None
    if weight_scale.dim() < 2 or (
        weight_scale.shape[-2] == 1 and weight_scale.shape[-1] == 1
    ):
        return 1
    scale_k_len = int(weight_scale.shape[-1])
    if scale_k_len <= 0:
        return None
    # Block-FP8 metadata uses one scale per 128 K values; the final group may
    # be partial, so K need not be divisible by 128.
    return 128


def _get_supported_gemv_quant_kind(
    layer: torch.nn.Module,
) -> Optional[str]:
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
        # The MUSA GEMV FP8 path is defined for the 128x128 block format.
        # Keep both autotune and persisted-policy reuse out of other block
        # layouts until their scale addressing is implemented and validated.
        block_size = getattr(
            getattr(quant_method, "quant_config", None), "weight_block_size", None
        )
        if tuple(block_size or ()) != (128, 128):
            return None
        return "fp8_block"
    return None


def _measure_run_profiler_device_us(
    target: _GemvTarget,
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
            "MUSA dense GEMV autotune profiler: quant=%s tokens=%s n=%s k=%s backend=%s "
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


def _format_gemv_summaries(
    summaries: list[tuple[_GemvTarget, bool, str]],
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


def _can_use_musa_gemv(target: _GemvTarget) -> bool:
    if target.n <= 0 or target.k <= 0:
        return False

    return bool(_valid_gemv_config_ids(target))


def _synchronize(device: torch.device) -> None:
    torch.get_device_module(device.type).synchronize()
