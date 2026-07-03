import importlib
import logging
import os
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F

from ..kernels.norm_rope_kernels import (
    _tilelang_compress_fused_norm_rope_inplace_kernel,
    _tilelang_compress_fused_norm_rope_prefill_inplace_kernel,
    _tilelang_fused_q_rmsnorm_rope_inplace_kernel,
    _tilelang_fused_norm_rope_inplace_kernel,
    _tilelang_hadamard128_inplace_kernel,
    _tilelang_neox_rope_hadamard_inplace_kernel_fast,
    _tilelang_rmsnorm_self_kernel,
    _tilelang_rmsnorm_self_strided_kernel,
    _tilelang_rope_hadamard_inplace_kernel_fast,
    _tilelang_rope_hadamard_inplace_kernel,
    _tilelang_rope_inplace_flat_kernel,
    _tilelang_rope_inplace_kernel,
    _tilelang_weighted_rmsnorm_base_offset_mudnn_like_kernel,
    _tilelang_weighted_rmsnorm_mudnn_like_kernel,
    _tilelang_weighted_rmsnorm_mudnn_like_blocky_kernel,
    _tilelang_weighted_rmsnorm_kernel,
    _tilelang_weighted_rmsnorm_strided_mudnn_like_kernel,
    _tilelang_weighted_rmsnorm_strided_mudnn_like_blocky_kernel,
    _tilelang_weighted_rmsnorm_strided_kernel,
    _tilelang_weighted_rmsnorm_strided_inplace_kernel,
)
from .compress_ops import _prefill_plan_rows
from .ops_common import (
    _debug_musa_allow_torch_fallback,
    _debug_musa_torch_fallback,
    _has_musa_tensor,
    _musa_graph_capture_enabled,
)

_FREQS_REAL_IMAG_CACHE: dict[tuple[int, str, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
_WEIGHTED_RMSNORM_KERNEL_CACHE: dict[
    tuple[str, bool, int, int, int, int, str, str, str, str, str, str, str], object
] = {}
logger = logging.getLogger(__name__)


_MUDNN_LIKE_RMSNORM_REDUCE_PROFILES = {
    "mudnn",
    "mudnn_like",
    "mudnn-tree",
    "mudnn_welford",
    "mudnn_welford_like",
    "mudnn-welford",
    "mudnn_chunk_mean",
    "chunk_mean",
    # Explicit experiment path: FlashInfer-style sum_sq + tree reduce. This is
    # faster on h512/h1024 but not strict bf16-exact on all inputs, so it must
    # only be selected via env.
    "flashinfer",
    "flashinfer_sum",
    "flashinfer-like",
    "sum",
    "sum_sq",
}


def _weighted_rmsnorm_reduce_profile() -> tuple[str, str]:
    reduce_profile = os.environ.get(
        "SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_REDUCE_PROFILE",
        "mudnn_welford_like",
    ).strip().lower() or "mudnn_welford_like"
    if reduce_profile in {
        "mudnn_welford",
        "mudnn_welford_like",
        "mudnn-welford",
    }:
        return reduce_profile, "welford_mean"
    if reduce_profile in {"mudnn_chunk_mean", "chunk_mean"}:
        return reduce_profile, "chunk_mean"
    return reduce_profile, "sum"


def _trace_rmsnorm_dispatch(kind: str, status: str, tensor: torch.Tensor, reason: str = "") -> None:
    if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_TRACE_DISPATCH") != "1":
        return
    hidden_size = int(tensor.shape[-1]) if tensor.dim() > 0 else 0
    num_rows = int(tensor.numel() // hidden_size) if hidden_size > 0 else 0
    logger.info(
        "RMSNORM_DISPATCH kind=%s status=%s shape=%s rows=%d hidden=%d "
        "stride=%s dtype=%s contiguous=%s reason=%s",
        kind,
        status,
        tuple(tensor.shape),
        num_rows,
        hidden_size,
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.is_contiguous(),
        reason,
    )


def _view_as_real_freqs(freqs_cis: torch.Tensor) -> torch.Tensor:
    if freqs_cis.dtype != torch.complex64:
        return freqs_cis
    key = (
        freqs_cis.data_ptr(),
        str(freqs_cis.device),
        freqs_cis.dtype,
        tuple(freqs_cis.shape),
    )
    cached = _FREQS_REAL_IMAG_CACHE.get(key)
    if cached is not None:
        return cached
    freqs_real_imag = torch.view_as_real(freqs_cis)
    _FREQS_REAL_IMAG_CACHE[key] = freqs_real_imag
    return freqs_real_imag


def _input_dtype_name(dtype: torch.dtype) -> str | None:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return None


def _positions_dtype_name(dtype: torch.dtype) -> str | None:
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.int64:
        return "int64"
    return None


def _rmsnorm_threads(hidden_size: int) -> int:
    return 64 if hidden_size == 512 else 128


def _try_tilelang_rmsnorm_self_musa(q: torch.Tensor, eps: float) -> Optional[torch.Tensor]:
    if q.device.type != "musa":
        _trace_rmsnorm_dispatch("self", "miss", q, "non_musa")
        return None
    if q.dtype != torch.bfloat16 or q.dim() < 2:
        _trace_rmsnorm_dispatch("self", "miss", q, "dtype_or_rank")
        return None
    if q.shape[-1] % 2 != 0:
        _trace_rmsnorm_dispatch("self", "miss", q, "odd_hidden")
        return None

    try:
        if q.is_contiguous():
            q_2d = q.view(-1, q.shape[-1])
            out = torch.empty_like(q_2d)
            kernel = _tilelang_rmsnorm_self_kernel(q.shape[-1], threads=_rmsnorm_threads(q.shape[-1]))
            kernel(q_2d, out, float(eps))
            _trace_rmsnorm_dispatch("self", "hit", q)
            return out.reshape_as(q)
        if q.dim() == 2 and q.stride(-1) == 1:
            out = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=q.dtype)
            kernel = _tilelang_rmsnorm_self_strided_kernel(
                q.shape[-1],
                int(q.stride(0)),
                threads=_rmsnorm_threads(q.shape[-1]),
            )
            kernel(q, out, float(eps))
            _trace_rmsnorm_dispatch("self", "hit", q, "strided")
            return out

        q_2d = q.contiguous().view(-1, q.shape[-1])
        out = torch.empty_like(q_2d)
        kernel = _tilelang_rmsnorm_self_kernel(q.shape[-1], threads=_rmsnorm_threads(q.shape[-1]))
        kernel(q_2d, out, float(eps))
    except Exception as exc:
        _trace_rmsnorm_dispatch("self", "miss", q, f"tilelang_exception:{type(exc).__name__}")
        return None
    _trace_rmsnorm_dispatch("self", "hit", q)
    return out.reshape_as(q)


def rmsnorm_self_musa(q: torch.Tensor, eps: float) -> torch.Tensor:
    tilelang_result = _try_tilelang_rmsnorm_self_musa(q, eps)
    if tilelang_result is not None:
        return tilelang_result
    if q.device.type == "musa":
        message = (
            "DeepSeekV4 MUSA rmsnorm_self has no torch fallback by default; "
            "torch fallback is disabled on MUSA. "
            f"q=device:{q.device},dtype:{q.dtype},shape:{tuple(q.shape)},"
            f"stride:{tuple(q.stride())},contiguous:{q.is_contiguous()},"
            f"storage_offset:{q.storage_offset()},eps={eps}"
        )
        if _musa_graph_capture_enabled():
            raise NotImplementedError(
                "DeepSeekV4 MUSA rmsnorm_self has no torch fallback during graph capture; "
                "TileLang rmsnorm path is required. "
                + message
            )
        if not _debug_musa_allow_torch_fallback():
            raise NotImplementedError(message)
        _debug_musa_torch_fallback(
            "DeepSeekV4 MUSA rmsnorm_self using torch fallback after TileLang miss: "
            + message
        )
    variance = q.float().pow(2).mean(dim=-1, keepdim=True)
    return (q.float() * torch.rsqrt(variance + eps)).to(q.dtype)


def _weighted_rmsnorm_config(
    hidden_size: int,
    num_rows: int,
    compile_profile: str | None,
) -> Optional[tuple[int, str]]:
    if hidden_size not in (512, 1024, 4096):
        return None

    profile = (compile_profile or "auto").strip().lower()
    if profile not in ("", "auto"):
        return 128, profile

    if hidden_size == 512:
        return (64 if num_rows >= 128 else 128), "opt1"
    if hidden_size == 1024:
        return 64, "opt1"
    # h4096 is sensitive to the MUSA LS/opt1/DSA flag combinations; default+t128
    # matches torch numerically and remains the best production choice in the
    # SGLang strided/contiguous benchmark sweep.
    return 128, "default"


def _weighted_rmsnorm_variant_config(hidden_size: int) -> tuple[str, str, str]:
    requested_rsqrt_mode = os.environ.get(
        "SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_RSQRT_MODE",
        "auto",
    ).strip().lower() or "auto"
    if requested_rsqrt_mode in {"auto", "hidden_auto", "hidden-size-aware"}:
        # Match muDNN fast_rsqrtf. This is required for bf16 exactness on the
        # production h512/h1024/h4096 weighted RMSNorm paths.
        resolved_rsqrt_mode = "mudnn_fast_rsqrt"
    else:
        resolved_rsqrt_mode = requested_rsqrt_mode
    mul_order = os.environ.get(
        "SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_MUL_ORDER",
        "x_rrms_weight",
    ).strip().lower() or "x_rrms_weight"
    return requested_rsqrt_mode, resolved_rsqrt_mode, mul_order


def _get_weighted_rmsnorm_kernel(
    *,
    mode: str = "out",
    contiguous: bool,
    hidden_size: int,
    num_rows: int,
    row_stride: int,
    threads: int,
    profile: str,
):
    reduce_profile, variance_mode = _weighted_rmsnorm_reduce_profile()
    requested_rsqrt_mode, rsqrt_mode, mul_order = _weighted_rmsnorm_variant_config(hidden_size)
    rcp_mode = os.environ.get(
        "SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_WELFORD_RCP_MODE",
        "ieee_frcp_newton",
    ).strip().lower() or "ieee_frcp_newton"
    if reduce_profile in _MUDNN_LIKE_RMSNORM_REDUCE_PROFILES and hidden_size == 512:
        # muDNN PH1 RMSNorm uses BLKX=64 for this tune_n; using 128 threads
        # changes both the supported tile shape and the reduction tree.
        threads = 64
    rows_per_cta = 1
    if (
        variance_mode == "welford_mean"
        and os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_BLOCKY_EXPERIMENT") == "1"
    ):
        # Mirror the high-payoff muDNN block-y shapes where several rows share
        # one CTA. This reduces scheduling overhead and improves cache reuse for
        # large-M RMSNorm cases without changing the WELFORD reduction order per row.
        if hidden_size == 4096 and contiguous and num_rows <= 16:
            rows_per_cta = 4
        elif hidden_size == 512 and not contiguous and num_rows >= 1024:
            rows_per_cta = 8
    key = (
        mode,
        contiguous,
        hidden_size,
        row_stride if not contiguous else hidden_size,
        threads,
        rows_per_cta,
        profile,
        reduce_profile,
        requested_rsqrt_mode,
        rsqrt_mode,
        mul_order,
        variance_mode,
        rcp_mode,
    )
    kernel = _WEIGHTED_RMSNORM_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    if reduce_profile in _MUDNN_LIKE_RMSNORM_REDUCE_PROFILES:
        if mode == "inplace":
            raise ValueError("mudnn_like RMSNorm prototype only supports out-of-place tensors")
        if rows_per_cta > 1 and contiguous:
            kernel = _tilelang_weighted_rmsnorm_mudnn_like_blocky_kernel(
                hidden_size,
                threads=threads,
                rows_per_cta=rows_per_cta,
                rsqrt_mode=rsqrt_mode,
                mul_order=mul_order,
                variance_mode=variance_mode,
                rcp_mode=rcp_mode,
            )
        elif rows_per_cta > 1:
            kernel = _tilelang_weighted_rmsnorm_strided_mudnn_like_blocky_kernel(
                hidden_size,
                row_stride=row_stride,
                threads=threads,
                rows_per_cta=rows_per_cta,
                rsqrt_mode=rsqrt_mode,
                mul_order=mul_order,
                variance_mode=variance_mode,
                rcp_mode=rcp_mode,
            )
        elif contiguous:
            kernel = _tilelang_weighted_rmsnorm_mudnn_like_kernel(
                hidden_size,
                threads=threads,
                compile_profile=profile,
                rsqrt_mode=rsqrt_mode,
                mul_order=mul_order,
                variance_mode=variance_mode,
                rcp_mode=rcp_mode,
            )
        else:
            kernel = _tilelang_weighted_rmsnorm_strided_mudnn_like_kernel(
                hidden_size,
                row_stride=row_stride,
                threads=threads,
                compile_profile=profile,
                rsqrt_mode=rsqrt_mode,
                mul_order=mul_order,
                variance_mode=variance_mode,
                rcp_mode=rcp_mode,
            )
    elif mode == "inplace" and not contiguous:
        kernel = _tilelang_weighted_rmsnorm_strided_inplace_kernel(
            hidden_size,
            row_stride=row_stride,
            threads=threads,
            compile_profile=profile,
            reduce_profile=reduce_profile,
            rsqrt_mode=rsqrt_mode,
            mul_order=mul_order,
        )
    elif contiguous:
        kernel = _tilelang_weighted_rmsnorm_kernel(
            hidden_size,
            threads=threads,
            compile_profile=profile,
            reduce_profile=reduce_profile,
            rsqrt_mode=rsqrt_mode,
            mul_order=mul_order,
        )
    else:
        kernel = _tilelang_weighted_rmsnorm_strided_kernel(
            hidden_size,
            row_stride=row_stride,
            threads=threads,
            compile_profile=profile,
            reduce_profile=reduce_profile,
            rsqrt_mode=rsqrt_mode,
            mul_order=mul_order,
        )
    _WEIGHTED_RMSNORM_KERNEL_CACHE[key] = kernel
    return kernel


def _try_base_offset_weighted_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    *,
    hidden_size: int,
    threads: int,
    rsqrt_mode: str,
    mul_order: str,
    rcp_mode: str,
    variance_mode: str,
) -> bool:
    if x.dim() != 2 or x.stride(-1) != 1:
        return False
    if hidden_size not in (512, 1024):
        return False
    x_base = getattr(x, "_base", None)
    if x_base is None:
        return False
    if (
        x_base.device != x.device
        or x_base.dtype != x.dtype
        or x_base.dim() != 2
        or x_base.shape[0] != x.shape[0]
        or x_base.stride(-1) != 1
        or x_base.stride(0) != x.stride(0)
    ):
        return False
    row_stride = int(x.stride(0))
    input_offset = int(x.storage_offset() - x_base.storage_offset())
    if input_offset < 0 or input_offset + hidden_size > int(x_base.shape[1]):
        return False
    kernel = _tilelang_weighted_rmsnorm_base_offset_mudnn_like_kernel(
        hidden_size,
        row_stride=row_stride,
        input_offset=input_offset,
        threads=64 if hidden_size == 512 else threads,
        rsqrt_mode=rsqrt_mode,
        mul_order=mul_order,
        rcp_mode=rcp_mode,
        variance_mode=variance_mode,
    )
    kernel(x_base, weight, out.view(-1, hidden_size), float(eps))
    return True


def _try_tilelang_weighted_rmsnorm_musa_inplace(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    compile_profile: str | None = "auto",
) -> Optional[torch.Tensor]:
    if x.device.type != "musa" or weight.device.type != "musa":
        _trace_rmsnorm_dispatch("weighted_inplace", "miss", x, "non_musa")
        return None
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        _trace_rmsnorm_dispatch("weighted_inplace", "miss", x, "dtype")
        return None
    if x.dim() < 2 or not weight.is_contiguous():
        _trace_rmsnorm_dispatch("weighted_inplace", "miss", x, "rank_or_weight_stride")
        return None
    if x.stride(-1) != 1:
        _trace_rmsnorm_dispatch("weighted_inplace", "miss", x, "last_stride")
        return None
    hidden_size = int(x.shape[-1])
    num_rows = int(x.numel() // hidden_size)
    config = _weighted_rmsnorm_config(hidden_size, num_rows, compile_profile)
    if config is None:
        _trace_rmsnorm_dispatch("weighted_inplace", "miss", x, "unsupported_hidden")
        return None
    threads, profile = config

    try:
        if x.is_contiguous():
            x_2d = x.view(-1, hidden_size)
            kernel = _get_weighted_rmsnorm_kernel(
                mode="out",
                contiguous=True,
                hidden_size=hidden_size,
                num_rows=num_rows,
                row_stride=hidden_size,
                threads=threads,
                profile=profile,
            )
            kernel(x_2d, weight, x_2d, float(eps))
        else:
            if x.dim() != 2:
                _trace_rmsnorm_dispatch("weighted_inplace", "miss", x, "strided_rank")
                return None
            kernel = _get_weighted_rmsnorm_kernel(
                mode="inplace",
                contiguous=False,
                hidden_size=hidden_size,
                num_rows=num_rows,
                row_stride=int(x.stride(0)),
                threads=threads,
                profile=profile,
            )
            kernel(x, weight, float(eps))
    except Exception as exc:
        _trace_rmsnorm_dispatch("weighted_inplace", "miss", x, f"tilelang_exception:{type(exc).__name__}")
        return None
    requested_rsqrt_mode, resolved_rsqrt_mode, mul_order = _weighted_rmsnorm_variant_config(hidden_size)
    _trace_rmsnorm_dispatch(
        "weighted_inplace",
        "hit",
        x,
        (
            f"profile={profile},threads={threads},"
            f"reduce_profile={os.environ.get('SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_REDUCE_PROFILE', 'mudnn_welford_like')},"
            f"rsqrt_mode={requested_rsqrt_mode},resolved_rsqrt_mode={resolved_rsqrt_mode},"
            f"mul_order={mul_order}"
        ),
    )
    return x


def _try_tilelang_weighted_rmsnorm_musa(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    compile_profile: str | None = "auto",
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if x.device.type != "musa" or weight.device.type != "musa":
        _trace_rmsnorm_dispatch("weighted", "miss", x, "non_musa")
        return None
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        _trace_rmsnorm_dispatch("weighted", "miss", x, "dtype")
        return None
    if x.dim() < 2 or not weight.is_contiguous():
        _trace_rmsnorm_dispatch("weighted", "miss", x, "rank_or_weight_stride")
        return None
    if x.stride(-1) != 1:
        _trace_rmsnorm_dispatch("weighted", "miss", x, "last_stride")
        return None
    hidden_size = int(x.shape[-1])
    num_rows = int(x.numel() // hidden_size)
    config = _weighted_rmsnorm_config(hidden_size, num_rows, compile_profile)
    if config is None:
        _trace_rmsnorm_dispatch("weighted", "miss", x, "unsupported_hidden")
        return None
    threads, profile = config

    if out is None:
        out = (
            torch.empty_like(x)
            if x.is_contiguous()
            else torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
        )
    elif (
        out.device != x.device
        or out.dtype != x.dtype
        or out.shape != x.shape
        or not out.is_contiguous()
    ):
        _trace_rmsnorm_dispatch("weighted", "miss", x, "invalid_out")
        return None

    try:
        if x.is_contiguous():
            x_2d = x.view(-1, hidden_size)
            out_2d = out.view(-1, hidden_size)
            kernel = _get_weighted_rmsnorm_kernel(
                mode="out",
                contiguous=True,
                hidden_size=hidden_size,
                num_rows=num_rows,
                row_stride=hidden_size,
                threads=threads,
                profile=profile,
            )
            kernel(x_2d, weight, out_2d, float(eps))
        else:
            if x.dim() != 2:
                _trace_rmsnorm_dispatch("weighted", "miss", x, "strided_rank")
                return None
            requested_rsqrt_mode, rsqrt_mode, mul_order = _weighted_rmsnorm_variant_config(hidden_size)
            rcp_mode = os.environ.get(
                "SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_WELFORD_RCP_MODE",
                "ieee_frcp_newton",
            ).strip().lower() or "ieee_frcp_newton"
            reduce_profile, variance_mode = _weighted_rmsnorm_reduce_profile()
            if _try_base_offset_weighted_rmsnorm(
                x,
                weight,
                out,
                eps,
                hidden_size=hidden_size,
                threads=threads,
                rsqrt_mode=rsqrt_mode,
                mul_order=mul_order,
                rcp_mode=rcp_mode,
                variance_mode=variance_mode,
            ):
                _trace_rmsnorm_dispatch(
                    "weighted",
                    "hit",
                    x,
                    (
                        f"profile={profile},threads={threads},reduce_profile=base_offset_{reduce_profile},"
                        f"variance_mode={variance_mode},"
                        f"rsqrt_mode={requested_rsqrt_mode},resolved_rsqrt_mode={rsqrt_mode},"
                        f"mul_order={mul_order}"
                    ),
                )
                return out
            kernel = _get_weighted_rmsnorm_kernel(
                mode="out",
                contiguous=False,
                hidden_size=hidden_size,
                num_rows=num_rows,
                row_stride=int(x.stride(0)),
                threads=threads,
                profile=profile,
            )
            kernel(x, weight, out.view(-1, hidden_size), float(eps))
    except Exception as exc:
        _trace_rmsnorm_dispatch("weighted", "miss", x, f"tilelang_exception:{type(exc).__name__}")
        return None
    requested_rsqrt_mode, resolved_rsqrt_mode, mul_order = _weighted_rmsnorm_variant_config(hidden_size)
    _trace_rmsnorm_dispatch(
        "weighted",
        "hit",
        x,
        (
            f"profile={profile},threads={threads},"
            f"reduce_profile={os.environ.get('SGLANG_DEEPSEEK_V4_MUSA_RMSNORM_REDUCE_PROFILE', 'mudnn_welford_like')},"
            f"rsqrt_mode={requested_rsqrt_mode},resolved_rsqrt_mode={resolved_rsqrt_mode},"
            f"mul_order={mul_order}"
        ),
    )
    return out


def weighted_rmsnorm_musa_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    *,
    compile_profile: str | None = "auto",
    tag: Optional[str] = None,
) -> torch.Tensor:
    _ = tag
    result = _try_tilelang_weighted_rmsnorm_musa(
        x,
        weight,
        eps,
        compile_profile=compile_profile,
        out=out,
    )
    if result is not None:
        return result
    torch_rms = F.rms_norm(x, (x.shape[-1],), weight, eps)
    out.copy_(torch_rms)
    return out


def weighted_rmsnorm_musa_inplace(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    compile_profile: str | None = "auto",
    tag: Optional[str] = None,
) -> torch.Tensor:
    _ = tag
    result = _try_tilelang_weighted_rmsnorm_musa_inplace(
        x,
        weight,
        eps,
        compile_profile=compile_profile,
    )
    if result is not None:
        return result
    if x.device.type == "musa" or weight.device.type == "musa":
        raise NotImplementedError(
            "DeepSeekV4 MUSA weighted_rmsnorm_inplace has no torch fallback; "
            f"x=device:{x.device},dtype:{x.dtype},shape:{tuple(x.shape)},"
            f"stride:{tuple(x.stride())}, weight=device:{weight.device},"
            f"dtype:{weight.dtype},shape:{tuple(weight.shape)}"
        )
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def weighted_rmsnorm_musa(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    compile_profile: str | None = "auto",
    tag: Optional[str] = None,
) -> torch.Tensor:
    _ = tag
    out = _try_tilelang_weighted_rmsnorm_musa(
        x,
        weight,
        eps,
        compile_profile=compile_profile,
    )
    if out is not None:
        return out
    if x.device.type == "musa" or weight.device.type == "musa":
        raise NotImplementedError(
            "DeepSeekV4 MUSA weighted_rmsnorm has no torch fallback; "
            f"x=device:{x.device},dtype:{x.dtype},shape:{tuple(x.shape)},"
            f"stride:{tuple(x.stride())}, weight=device:{weight.device},"
            f"dtype:{weight.dtype},shape:{tuple(weight.shape)}"
        )
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _rope_tensor_summary(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}: shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
        f"dtype={tensor.dtype} device={tensor.device} storage_offset={tensor.storage_offset()}"
    )

def _tilelang_rope_inplace_musa_result(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool,
    name: str,
) -> tuple[bool, str]:
    ok, reason = _tilelang_rope_inplace_musa_guard(x, freqs_cis, positions, inverse, name)
    if not ok:
        return False, reason

    freqs_real_imag = _view_as_real_freqs(freqs_cis)
    try:
        positions_dtype = "int32" if positions.dtype == torch.int32 else "int64"
        input_dtype = "float32" if x.dtype == torch.float32 else "bfloat16"
        num_heads = x.shape[1]
        rope_flat_mode = os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_ROPE_FLAT", "auto").lower().strip()
        # The flat kernel covers all RoPE pairs; keep the old block mapping only
        # as an explicit opt-out while validating legacy behavior.
        use_flat = rope_flat_mode != "0"
        if use_flat:
            kernel = _tilelang_rope_inplace_flat_kernel(
                input_dtype,
                x.shape[-1],
                num_heads,
                inverse,
                positions_dtype,
                threads=256,
            )
        else:
            kernel = _tilelang_rope_inplace_kernel(
                input_dtype,
                x.shape[-1],
                num_heads,
                inverse,
                positions_dtype,
            )
        x_storage = x.as_strided((x.untyped_storage().nbytes() // x.element_size(),), (1,), storage_offset=0)
        if os.environ.get("SGLANG_DSV4_MUSA_ROPE_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"):
            print(
                "[dsv4_musa_rope_debug] "
                f"name={name} use_flat={use_flat} input_dtype={input_dtype} "
                f"positions_dtype={positions_dtype} num_heads={num_heads} head_dim={x.shape[-1]} "
                f"x={_rope_tensor_summary(name, x)} "
                f"x_storage_numel={x_storage.numel()} "
                f"freqs_real_imag={_rope_tensor_summary('freqs_real_imag', freqs_real_imag)} "
                f"positions={_rope_tensor_summary('positions', positions)} inverse={inverse}",
                flush=True,
            )
        kernel(
            x_storage,
            freqs_real_imag,
            positions,
            x.storage_offset(),
            x.stride(0),
            x.stride(1),
            x.stride(2),
        )
    except Exception as exc:
        context = ", ".join(
            [
                _rope_tensor_summary(name, x),
                _rope_tensor_summary("freqs_cis", freqs_cis),
                _rope_tensor_summary("positions", positions),
                f"inverse={inverse}",
            ]
        )
        return False, f"TileLang launch failed for {name}: {type(exc).__name__}: {exc}; {context}"
    return True, ""

def _tilelang_rope_inplace_musa_guard(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool,
    name: str,
) -> tuple[bool, str]:
    context = ", ".join(
        [
            _rope_tensor_summary(name, x),
            _rope_tensor_summary("freqs_cis", freqs_cis),
            _rope_tensor_summary("positions", positions),
            f"inverse={inverse}",
        ]
    )
    if x.device.type != "musa":
        return False, f"{name} device is {x.device.type}; {context}"
    if x.dtype not in (torch.bfloat16, torch.float32):
        return False, f"{name} dtype is {x.dtype}; {context}"
    if x.dim() != 3:
        return False, f"{name} dim is {x.dim()}; {context}"
    if x.stride(-1) != 1:
        return False, f"{name} last-dim stride is {x.stride(-1)}; {context}"
    if freqs_cis.device != x.device:
        return False, f"freqs_cis device {freqs_cis.device} != {name} device {x.device}; {context}"
    if positions.device != x.device:
        return False, f"positions device {positions.device} != {name} device {x.device}; {context}"
    if freqs_cis.dtype != torch.complex64:
        return False, f"freqs_cis dtype is {freqs_cis.dtype}; {context}"
    if positions.dtype not in (torch.int32, torch.int64):
        return False, f"positions dtype is {positions.dtype}; {context}"
    if positions.dim() != 1:
        return False, f"positions dim is {positions.dim()}; {context}"
    if positions.shape[0] != x.shape[0]:
        return False, f"positions length {positions.shape[0]} != {name} tokens {x.shape[0]}; {context}"
    if x.shape[-1] % 2 != 0:
        return False, f"{name} head_dim is odd; {context}"
    if freqs_cis.shape[-1] != x.shape[-1] // 2:
        return False, f"freqs_cis half dim {freqs_cis.shape[-1]} != {name} half dim {x.shape[-1] // 2}; {context}"
    half_dim = x.shape[-1] // 2
    if half_dim > 32:
        return False, f"{name} half_dim={half_dim} exceeds TileLang rope thread limit (32); {context}"
    return True, ""

def _try_tilelang_rope_inplace_musa(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool,
) -> bool:
    ok, _ = _tilelang_rope_inplace_musa_result(x, freqs_cis, positions, inverse, "x")
    return ok


def _try_tilelang_rope_hadamard_inplace_musa(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    *,
    scale: float = 128.0 ** -0.5,
    heads_per_block: int | None = None,
    pingpong: bool | None = None,
) -> tuple[bool, str]:
    context = ", ".join(
        [
            _rope_tensor_summary("q", q),
            _rope_tensor_summary("freqs_cis", freqs_cis),
            _rope_tensor_summary("positions", positions),
        ]
    )
    if q.device.type != "musa":
        return False, f"q device is {q.device.type}; {context}"
    input_dtype = _input_dtype_name(q.dtype)
    if input_dtype is None:
        return False, f"q dtype is {q.dtype}; {context}"
    positions_dtype = _positions_dtype_name(positions.dtype)
    if positions_dtype is None:
        return False, f"positions dtype is {positions.dtype}; {context}"
    if q.dim() != 3 or q.shape[-1] != 128:
        return False, f"q must be [tokens, heads, 128], got {tuple(q.shape)}; {context}"
    if not q.is_contiguous():
        return False, f"q must be contiguous for temporary fused RoPE+Hadamard path; {context}"
    if freqs_cis.device != q.device or positions.device != q.device:
        return False, f"freqs/positions must be on q device; {context}"
    if freqs_cis.dtype != torch.complex64 or freqs_cis.dim() != 2 or freqs_cis.shape[-1] != 32:
        return False, f"freqs_cis must be complex64 [num_positions, 32]; {context}"
    if positions.dim() != 1 or positions.shape[0] != q.shape[0]:
        return False, f"positions must be 1D with one entry per token; {context}"
    if heads_per_block is None:
        if int(q.shape[0]) >= 256:
            heads_per_block = 8
        elif int(q.shape[0]) >= 128:
            heads_per_block = 4
        else:
            heads_per_block = 1
    if heads_per_block not in (1, 2, 4, 8):
        return False, f"heads_per_block must be one of 1, 2, 4, 8, got {heads_per_block}; {context}"
    if int(q.shape[1]) % heads_per_block != 0:
        return False, f"num_heads={int(q.shape[1])} must be divisible by heads_per_block={heads_per_block}; {context}"
    if pingpong is None:
        pingpong = False
    try:
        if pingpong:
            return False, (
                "fast RoPE+Hadamard path requires pingpong=0; "
                f"{context}"
            )
        if input_dtype == "float32":
            # The production path is bf16. Keep fp32 validation on the
            # conservative 3D kernel because the flat/i32 fast variant is
            # only intended to reduce bf16 prefill integer-address pressure.
            fp32_heads_per_block = min(heads_per_block, 4)
            kernel = _tilelang_rope_hadamard_inplace_kernel(
                input_dtype,
                int(q.shape[1]),
                positions_dtype,
                heads_per_block=fp32_heads_per_block,
                pingpong=False,
                threads=128 * fp32_heads_per_block,
            )
            q_arg = q
        else:
            kernel = _tilelang_rope_hadamard_inplace_kernel_fast(
                input_dtype,
                int(q.shape[1]),
                positions_dtype,
                heads_per_block=heads_per_block,
            )
            q_arg = q.view(-1)
        kernel(q_arg, _view_as_real_freqs(freqs_cis), positions, float(scale))
    except Exception as exc:
        return False, f"TileLang RoPE+Hadamard launch failed: {type(exc).__name__}: {exc}; {context}"
    return True, ""


def _try_tilelang_neox_rope_hadamard_inplace_musa(
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    scale: float = 128.0 ** -0.5,
) -> tuple[bool, str]:
    context = ", ".join(
        [
            _rope_tensor_summary("q", q),
            _rope_tensor_summary("cos_sin_cache", cos_sin_cache),
            _rope_tensor_summary("positions", positions),
        ]
    )
    if q.device.type != "musa":
        return False, f"q device is {q.device.type}; {context}"
    input_dtype = _input_dtype_name(q.dtype)
    if input_dtype is None:
        return False, f"q dtype is {q.dtype}; {context}"
    positions_dtype = _positions_dtype_name(positions.dtype)
    if positions_dtype is None:
        return False, f"positions dtype is {positions.dtype}; {context}"
    if q.dim() == 2:
        if q.shape[-1] != 128:
            return False, f"2D q must be [tokens, 128], got {tuple(q.shape)}; {context}"
        q_view = q.reshape(q.shape[0], 1, 128)
    elif q.dim() == 3:
        if q.shape[-1] != 128:
            return False, f"3D q must be [tokens, heads, 128], got {tuple(q.shape)}; {context}"
        q_view = q
    else:
        return False, f"q must be 2D or 3D, got {tuple(q.shape)}; {context}"
    if not q_view.is_contiguous():
        return False, f"q must be contiguous for Neox RoPE+Hadamard path; {context}"
    if cos_sin_cache.device != q.device or positions.device != q.device:
        return False, f"cos_sin_cache/positions must be on q device; {context}"
    if cos_sin_cache.dtype not in (torch.float32, torch.bfloat16):
        return False, f"cos_sin_cache dtype must be float32/bfloat16; {context}"
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] != 64:
        return False, f"cos_sin_cache must be [num_positions, 64]; {context}"
    if positions.dim() != 1 or positions.shape[0] != q_view.shape[0]:
        return False, f"positions must be 1D with one entry per token; {context}"

    try:
        kernel = _tilelang_neox_rope_hadamard_inplace_kernel_fast(
            input_dtype,
            int(q_view.shape[1]),
            positions_dtype,
        )
        kernel(q_view, cos_sin_cache.float() if cos_sin_cache.dtype != torch.float32 else cos_sin_cache, positions, float(scale))
    except Exception as exc:
        return False, f"TileLang Neox RoPE+Hadamard launch failed: {type(exc).__name__}: {exc}; {context}"
    return True, ""


def _try_tilelang_hadamard128_inplace_musa(
    x: torch.Tensor,
    *,
    scale: float = 128.0 ** -0.5,
    threads: int | None = None,
) -> tuple[bool, str]:
    context = _rope_tensor_summary("x", x)
    if x.device.type != "musa":
        return False, f"x device is {x.device.type}; {context}"
    input_dtype = _input_dtype_name(x.dtype)
    if input_dtype is None:
        return False, f"x dtype is {x.dtype}; {context}"
    if x.dim() < 1 or x.shape[-1] != 128:
        return False, f"x must have last dimension 128, got {tuple(x.shape)}; {context}"
    if not x.is_contiguous():
        return False, f"x must be contiguous for hadamard128 micro-kernel; {context}"
    if threads is None:
        try:
            threads = int(os.environ.get("SGLANG_OPT_DSV4_MUSA_HADAMARD128_THREADS", "16"))
        except ValueError:
            return False, f"SGLANG_OPT_DSV4_MUSA_HADAMARD128_THREADS must be an integer; {context}"
    if threads not in (16, 32):
        return False, f"threads must be one of 16, 32, got {threads}; {context}"

    try:
        x_2d = x.reshape(-1, 128)
        kernel = _tilelang_hadamard128_inplace_kernel(input_dtype, threads=threads)
        kernel(x_2d, float(scale))
    except Exception as exc:
        return False, f"TileLang hadamard128 launch failed: {type(exc).__name__}: {exc}; {context}"
    return True, ""


def hadamard128_inplace_musa(
    x: torch.Tensor,
    *,
    scale: float = 128.0 ** -0.5,
    threads: int | None = None,
) -> torch.Tensor:
    ok, reason = _try_tilelang_hadamard128_inplace_musa(
        x,
        scale=scale,
        threads=threads,
    )
    if ok:
        return x
    if x.device.type == "musa":
        raise NotImplementedError(
            "DeepSeekV4 MUSA hadamard128 micro-kernel unsupported or failed; "
            f"no torch fallback in public MUSA op. Reason: {reason}"
        )
    return x


def fused_rope_hadamard_musa(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    *,
    scale: float = 128.0 ** -0.5,
    heads_per_block: int | None = None,
    pingpong: bool | None = None,
) -> torch.Tensor:
    ok, reason = _try_tilelang_rope_hadamard_inplace_musa(
        q,
        freqs_cis,
        positions,
        scale=scale,
        heads_per_block=heads_per_block,
        pingpong=pingpong,
    )
    if ok:
        return q
    if q.device.type == "musa":
        raise NotImplementedError(
            "DeepSeekV4 MUSA temporary fused RoPE+Hadamard path unsupported or failed; "
            f"no torch fallback in fused op. Reason: {reason}"
        )
    return q


def fused_neox_rope_hadamard_musa(
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    scale: float = 128.0 ** -0.5,
) -> torch.Tensor:
    ok, reason = _try_tilelang_neox_rope_hadamard_inplace_musa(
        q,
        cos_sin_cache,
        positions,
        scale=scale,
    )
    if ok:
        return q
    if q.device.type == "musa":
        raise NotImplementedError(
            "DeepSeekV4 MUSA fused Neox RoPE+Hadamard path unsupported or failed; "
            f"no torch fallback in public MUSA op. Reason: {reason}"
        )
    return q



def fused_rope_musa(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    q_supported, q_reason = _tilelang_rope_inplace_musa_guard(q, freqs_cis, positions, inverse, "q")
    k_supported = k is None
    q_done = False
    k_done = k is None
    k_reason = ""
    if k is not None:
        k_supported, k_reason = _tilelang_rope_inplace_musa_guard(k, freqs_cis, positions, inverse, "k")

    if q_supported and k_supported:
        q_done, q_reason = _tilelang_rope_inplace_musa_result(q, freqs_cis, positions, inverse, "q")
        k_done = k is None
        if k is not None:
            k_done, k_reason = _tilelang_rope_inplace_musa_result(k, freqs_cis, positions, inverse, "k")
        if q_done and k_done:
            return

    if q.device.type == "musa" or (k is not None and k.device.type == "musa"):
        reasons: list[str] = []
        if not q_supported or not q_done:
            reasons.append(q_reason)
        if k is not None and (not k_supported or not k_done):
            reasons.append(k_reason)
        raise NotImplementedError(
            "DeepSeekV4 MUSA fused_rope_musa TileLang path unsupported or failed; "
            f"no torch fallback on MUSA. Reasons: {'; '.join(reason for reason in reasons if reason)}"
        )

    if q_supported:
        q_done, q_reason = _tilelang_rope_inplace_musa_result(q, freqs_cis, positions, inverse, "q")
    if k is not None and k_supported:
        k_done, k_reason = _tilelang_rope_inplace_musa_result(k, freqs_cis, positions, inverse, "k")
    if q_done and k_done:
        return
    if not q_done or (k is not None and not k_done):
        freq_real, freq_imag = _select_rope_freqs_real_imag(freqs_cis, positions, inverse)

        if not q_done:
            _apply_rope_inplace_real_imag(q, freq_real, freq_imag)
        if k is not None and not k_done:
            _apply_rope_inplace_real_imag(k, freq_real, freq_imag)

def _select_rope_freqs_real_imag(
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs_cis.dtype == torch.complex64:
        # MUSA cannot index ComplexFloat tensors; index the real/imag payload instead.
        freqs_real_imag = _view_as_real_freqs(freqs_cis)
    elif freqs_cis.shape[-1] == 2:
        freqs_real_imag = freqs_cis
    else:
        raise TypeError(
            f"DeepSeekV4 MUSA RoPE expects complex64 freqs or real-imag pairs, got {freqs_cis.dtype} {tuple(freqs_cis.shape)}"
        )

    selected = freqs_real_imag.index_select(0, positions.to(device=freqs_cis.device, dtype=torch.long))
    freq_real = selected[..., 0].float()
    freq_imag = selected[..., 1].float()
    if inverse:
        freq_imag = -freq_imag
    return freq_real, freq_imag

def _apply_rope_inplace_real_imag(
    x: torch.Tensor,
    freq_real: torch.Tensor,
    freq_imag: torch.Tensor,
) -> None:
    orig_dtype = x.dtype
    x_float = x.float().reshape(*x.shape[:-1], -1, 2)

    while freq_real.dim() < x_float.dim() - 1:
        freq_real = freq_real.unsqueeze(1)
        freq_imag = freq_imag.unsqueeze(1)

    even = x_float[..., 0]
    odd = x_float[..., 1]
    out_pair = torch.stack(
        (even * freq_real - odd * freq_imag, even * freq_imag + odd * freq_real),
        dim=-1,
    )
    out = out_pair.flatten(-2).to(orig_dtype)
    x.copy_(out)

def _try_tilelang_fused_norm_rope_inplace_musa(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[bool, str | None]:
    debug_guard = os.environ.get("SGLANG_DEBUG_MUSA_FUSED_NORM_ROPE_GUARD") == "1"

    def fail(reason: str) -> tuple[bool, str]:
        message = (
            f"{reason}; "
            f"kv=device:{kv.device},dtype:{kv.dtype},shape:{tuple(kv.shape)},stride:{kv.stride()},contiguous:{kv.is_contiguous()}; "
            f"weight=device:{weight.device},dtype:{weight.dtype},shape:{tuple(weight.shape)},stride:{weight.stride()},contiguous:{weight.is_contiguous()}; "
            f"freq_cis=device:{freq_cis.device},dtype:{freq_cis.dtype},shape:{tuple(freq_cis.shape)},stride:{freq_cis.stride()},contiguous:{freq_cis.is_contiguous()}; "
            f"positions=device:{positions.device},dtype:{positions.dtype},shape:{tuple(positions.shape)},stride:{positions.stride()},contiguous:{positions.is_contiguous()}"
        )
        if debug_guard:
            print(
                "DeepSeekV4 MUSA fused_norm_rope TileLang guard failed: "
                f"{message}",
                flush=True,
            )
        return False, message

    if kv.device.type != "musa" or kv.dtype not in (torch.bfloat16, torch.float32):
        return fail("kv device/dtype")
    if kv.dim() != 2 or not kv.is_contiguous():
        return fail("kv shape/contiguity")
    if weight.device != kv.device or weight.dtype != kv.dtype or weight.shape != (kv.shape[-1],):
        return fail("weight device/dtype/shape")
    if freq_cis.device != kv.device or freq_cis.dtype != torch.complex64:
        return fail("freq_cis device/dtype")
    if positions.device != kv.device or positions.dim() != 1 or positions.shape[0] != kv.shape[0]:
        return fail("positions device/shape")
    if positions.dtype not in (torch.int32, torch.int64):
        return fail("positions dtype")
    rope_complex_dim = freq_cis.shape[-1]
    hidden_size = kv.shape[-1]
    rope_dim = rope_complex_dim * 2
    if hidden_size % 2 != 0 or rope_dim <= 0 or rope_dim > hidden_size:
        return fail(f"rope dimensions hidden_size={hidden_size}, rope_dim={rope_dim}")

    freqs_real_imag = _view_as_real_freqs(freq_cis)
    positions_dtype = "int32" if positions.dtype == torch.int32 else "int64"
    try:
        kernel = _tilelang_fused_norm_rope_inplace_kernel(
            kv.shape[-1],
            rope_dim,
            kv.dtype,
            positions_dtype,
        )
        kernel(
            kv,
            weight,
            freqs_real_imag,
            positions,
            float(eps),
        )
    except Exception as exc:
        return fail(f"kernel exception {type(exc).__name__}: {exc}")
    return True, None


def _try_tilelang_fused_q_rmsnorm_rope_inplace_musa(
    q: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[bool, str | None]:
    debug_guard = os.environ.get("SGLANG_DEBUG_MUSA_FUSED_Q_RMSNORM_ROPE_GUARD") == "1"

    def fail(reason: str) -> tuple[bool, str]:
        message = (
            f"{reason}; "
            f"q=device:{q.device},dtype:{q.dtype},shape:{tuple(q.shape)},stride:{q.stride()},contiguous:{q.is_contiguous()}; "
            f"freq_cis=device:{freq_cis.device},dtype:{freq_cis.dtype},shape:{tuple(freq_cis.shape)},stride:{freq_cis.stride()},contiguous:{freq_cis.is_contiguous()}; "
            f"positions=device:{positions.device},dtype:{positions.dtype},shape:{tuple(positions.shape)},stride:{positions.stride()},contiguous:{positions.is_contiguous()}"
        )
        if debug_guard:
            print(
                "DeepSeekV4 MUSA fused_q_rmsnorm_rope TileLang guard failed: "
                f"{message}",
                flush=True,
            )
        return False, message

    if q.device.type != "musa" or q.dtype != torch.bfloat16:
        return fail("q device/dtype")
    if q.dim() != 3 or q.shape[-1] != 512:
        return fail("q shape")
    if q.stride(-1) != 1 or not q.is_contiguous():
        return fail("q contiguity")
    if freq_cis.device != q.device or freq_cis.dtype != torch.complex64:
        return fail("freq_cis device/dtype")
    if freq_cis.dim() != 2 or freq_cis.shape[-1] != 32:
        return fail("freq_cis shape")
    if positions.device != q.device or positions.dtype not in (torch.int32, torch.int64):
        return fail("positions device/dtype")
    if positions.dim() != 1 or positions.shape[0] != q.shape[0] or not positions.is_contiguous():
        return fail("positions shape/contiguity")

    positions_dtype = "int32" if positions.dtype == torch.int32 else "int64"
    try:
        kernel = _tilelang_fused_q_rmsnorm_rope_inplace_kernel(
            int(q.shape[1]),
            positions_dtype,
        )
        kernel(q, _view_as_real_freqs(freq_cis), positions, float(eps))
    except Exception as exc:
        return fail(f"kernel exception {type(exc).__name__}: {exc}")
    return True, None


def fused_q_rmsnorm_rope_inplace_musa(
    q: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    ok, reason = _try_tilelang_fused_q_rmsnorm_rope_inplace_musa(q, eps, freq_cis, positions)
    if ok:
        return q
    if q.device.type == "musa":
        raise NotImplementedError(
            "DeepSeekV4 MUSA fused_q_rmsnorm_rope_inplace_musa TileLang path unsupported or failed; "
            f"no torch fallback on MUSA. Reason: {reason}"
        )

    q.copy_(rmsnorm_self_musa(q, eps))
    fused_rope_musa(q[..., -64:], None, freq_cis, positions)
    return q


def fused_norm_rope_inplace_musa(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    # The h512/r64 fused path now uses one warp per token and avoids the
    # high-overhead decomposed public path. Keep an opt-out while E2E matures.
    fused_h512_enabled = os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_FUSED_NORM_ROPE_H512", "1") != "0"
    if kv.dim() == 2 and (kv.shape[-1] <= 128 or (fused_h512_enabled and kv.shape[-1] == 512)):
        tilelang_ok, _tilelang_failure = _try_tilelang_fused_norm_rope_inplace_musa(kv, weight, eps, freq_cis, positions)
        if tilelang_ok:
            return

    normalized = rmsnorm_self_musa(kv, eps) * weight
    kv.copy_(normalized)

    if freq_cis.dtype == torch.complex64:
        rope_dim = freq_cis.shape[-1] * 2
    else:
        rope_dim = freq_cis.shape[-1]

    if kv.shape[-1] == rope_dim:
        fused_rope_musa(kv.unsqueeze(1) if kv.dim() == 2 else kv, None, freq_cis, positions)
    else:
        kv_rope = kv[..., -rope_dim:]
        fused_rope_musa(kv_rope.unsqueeze(1) if kv_rope.dim() == 2 else kv_rope, None, freq_cis, positions)

def _try_tilelang_compress_fused_norm_rope_inplace_musa(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_ratio: int,
) -> tuple[bool, str | None]:
    debug_guard = os.environ.get("SGLANG_DEBUG_MUSA_FUSED_NORM_ROPE_GUARD") == "1"

    def fail(reason: str) -> tuple[bool, str]:
        message = (
            f"{reason}; "
            f"kv=device:{kv.device},dtype:{kv.dtype},shape:{tuple(kv.shape)},stride:{kv.stride()},contiguous:{kv.is_contiguous()}; "
            f"weight=device:{weight.device},dtype:{weight.dtype},shape:{tuple(weight.shape)},stride:{weight.stride()},contiguous:{weight.is_contiguous()}; "
            f"freq_cis=device:{freq_cis.device},dtype:{freq_cis.dtype},shape:{tuple(freq_cis.shape)},stride:{freq_cis.stride()},contiguous:{freq_cis.is_contiguous()}; "
            f"seq_lens=device:{seq_lens.device},dtype:{seq_lens.dtype},shape:{tuple(seq_lens.shape)},stride:{seq_lens.stride()},contiguous:{seq_lens.is_contiguous()}; "
            f"compress_ratio={compress_ratio}"
        )
        if debug_guard:
            print(
                "DeepSeekV4 MUSA compress_fused_norm_rope TileLang guard failed: "
                f"{message}",
                flush=True,
            )
        return False, message

    if kv.device.type != "musa" or kv.dtype not in (torch.bfloat16, torch.float32):
        return fail("kv device/dtype")
    if kv.dim() != 2 or not kv.is_contiguous():
        return fail("kv shape/contiguity")
    if weight.device != kv.device or weight.dtype != kv.dtype or weight.shape != (kv.shape[-1],):
        return fail("weight device/dtype/shape")
    if freq_cis.device != kv.device or freq_cis.dtype != torch.complex64:
        return fail("freq_cis device/dtype")
    if seq_lens.device != kv.device or seq_lens.shape != (kv.shape[0],):
        return fail("seq_lens device/shape")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        return fail("seq_lens dtype")
    rope_complex_dim = freq_cis.shape[-1]
    hidden_size = kv.shape[-1]
    rope_dim = rope_complex_dim * 2
    if hidden_size % 2 != 0 or rope_dim <= 0 or rope_dim > hidden_size:
        return fail(f"rope dimensions hidden_size={hidden_size}, rope_dim={rope_dim}")

    freqs_real_imag = _view_as_real_freqs(freq_cis)
    seq_lens_dtype = "int32" if seq_lens.dtype == torch.int32 else "int64"
    try:
        kernel = _tilelang_compress_fused_norm_rope_inplace_kernel(
            kv.shape[-1],
            rope_dim,
            compress_ratio,
            kv.dtype,
            seq_lens_dtype,
        )
        kernel(
            kv,
            weight,
            freqs_real_imag,
            seq_lens,
            float(eps),
        )
    except Exception as exc:
        return fail(f"kernel exception {type(exc).__name__}: {exc}")
    return True, None

def _compress_fused_norm_rope_inplace_reference(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_ratio: int,
) -> None:
    decode_rows = torch.nonzero((seq_lens % compress_ratio) == 0, as_tuple=False).flatten()
    if decode_rows.numel() == 0:
        return

    positions = (seq_lens.index_select(0, decode_rows) - compress_ratio).to(seq_lens.dtype)
    transformed = kv.index_select(0, decode_rows).clone()
    fused_norm_rope_inplace_musa(transformed, weight, eps, freq_cis, positions)
    kv.index_copy_(0, decode_rows, transformed)

def compress_fused_norm_rope_inplace_musa(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_ratio: int,
) -> None:
    if compress_ratio <= 0:
        raise ValueError(f"DeepSeekV4 MUSA compress_ratio must be positive, got {compress_ratio}")
    if seq_lens.shape[0] != kv.shape[0]:
        raise ValueError(
            f"DeepSeekV4 MUSA compress_fused_norm_rope_inplace expected seq_lens shape ({kv.shape[0]},), got {tuple(seq_lens.shape)}"
        )

    tilelang_ok, tilelang_failure = _try_tilelang_compress_fused_norm_rope_inplace_musa(
        kv, weight, eps, freq_cis, seq_lens, compress_ratio
    )
    if tilelang_ok:
        return
    if kv.device.type == "musa":
        if not _debug_musa_allow_torch_fallback():
            raise NotImplementedError(
                "DeepSeekV4 MUSA compress_fused_norm_rope has no torch fallback for supported MUSA input: "
                f"{tilelang_failure}"
            )
        _debug_musa_torch_fallback(
            "DeepSeekV4 MUSA compress_fused_norm_rope using torch fallback after TileLang miss: "
            f"{tilelang_failure}"
        )

    _compress_fused_norm_rope_inplace_reference(kv, weight, eps, freq_cis, seq_lens, compress_ratio)

def compress_fused_norm_rope_prefill_inplace_musa(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    compress_plan: torch.Tensor,
) -> None:
    rows = _prefill_plan_rows(compress_plan)
    if rows.numel() == 0:
        return

    if (
        kv.device.type == "musa"
        and kv.dtype in (torch.bfloat16, torch.float32)
        and kv.dim() == 2
        and kv.is_contiguous()
        and weight.device == kv.device
        and weight.dtype == kv.dtype
        and weight.shape == (kv.shape[-1],)
        and freq_cis.device == kv.device
        and freq_cis.dtype == torch.complex64
    ):
        rope_dim = freq_cis.shape[-1] * 2
        # Direct prefill keeps a different bf16 rounding path from the
        # historical gather + fused_norm_rope + scatter implementation. Limit
        # it to the validated DSV4 production shape; other shapes must preserve
        # old-path bitwise behavior.
        if kv.shape[-1] == 512 and rope_dim == 64:
            freqs_real_imag = _view_as_real_freqs(freq_cis)
            try:
                kernel = _tilelang_compress_fused_norm_rope_prefill_inplace_kernel(
                    kv.shape[-1],
                    rope_dim,
                    kv.dtype,
                    "int32",
                )
                kernel(kv, weight, freqs_real_imag, rows, float(eps))
                return
            except Exception as exc:
                if not _debug_musa_allow_torch_fallback():
                    raise NotImplementedError(
                        "DeepSeekV4 MUSA compress_fused_norm_rope_prefill direct TileLang path failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

    ragged_ids = rows[:, 0].to(device=kv.device, dtype=torch.long)
    positions = rows[:, 2].to(device=kv.device, dtype=torch.int32)
    transformed = kv.index_select(0, ragged_ids).clone()
    fused_norm_rope_inplace_musa(transformed, weight, eps, freq_cis, positions)
    kv.index_copy_(0, ragged_ids, transformed)

__all__ = [
    '_try_tilelang_rmsnorm_self_musa',
    'rmsnorm_self_musa',
    '_try_tilelang_weighted_rmsnorm_musa',
    '_try_tilelang_weighted_rmsnorm_musa_inplace',
    'weighted_rmsnorm_musa_out',
    'weighted_rmsnorm_musa_inplace',
    'weighted_rmsnorm_musa',
    '_view_as_real_freqs',
    '_rmsnorm_threads',
    '_rope_tensor_summary',
    '_tilelang_rope_inplace_musa_result',
    '_tilelang_rope_inplace_musa_guard',
    '_try_tilelang_rope_inplace_musa',
    '_try_tilelang_hadamard128_inplace_musa',
    'hadamard128_inplace_musa',
    '_try_tilelang_rope_hadamard_inplace_musa',
    '_try_tilelang_neox_rope_hadamard_inplace_musa',
    'fused_rope_hadamard_musa',
    'fused_neox_rope_hadamard_musa',
    'fused_rope_musa',
    '_select_rope_freqs_real_imag',
    '_apply_rope_inplace_real_imag',
    '_try_tilelang_fused_norm_rope_inplace_musa',
    'fused_norm_rope_inplace_musa',
    '_try_tilelang_fused_q_rmsnorm_rope_inplace_musa',
    'fused_q_rmsnorm_rope_inplace_musa',
    '_try_tilelang_compress_fused_norm_rope_inplace_musa',
    '_compress_fused_norm_rope_inplace_reference',
    'compress_fused_norm_rope_inplace_musa',
    'compress_fused_norm_rope_prefill_inplace_musa',
]
