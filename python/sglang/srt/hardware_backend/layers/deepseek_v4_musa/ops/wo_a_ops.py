import logging
import os
from typing import Optional

import torch

from ..kernels.wo_a_kernels import (
    WO_A_D,
    WO_A_R,
    _tilelang_wo_a_large_gemm_kernel,
    _tilelang_wo_a_m1_splitk_kernel,
    _tilelang_wo_a_small_gemm_kernel,
    _tilelang_wo_a_small_static_gemm_kernel,
)

logger = logging.getLogger(__name__)

_LOGGED_WO_A_MISSES: set[tuple[str, tuple[int, ...], tuple[int, ...]]] = set()


def _trace_wo_a_dispatch(
    status: str,
    o: torch.Tensor,
    wo_a: torch.Tensor,
    reason: str = "",
) -> None:
    if os.environ.get("SGLANG_DSV4_MUSA_TILELANG_WO_A_TRACE_DISPATCH") == "1":
        logger.info(
            "WO_A_TILELANG_DISPATCH status=%s reason=%s "
            "o_shape=%s o_stride=%s o_dtype=%s o_device=%s "
            "wo_a_shape=%s wo_a_stride=%s wo_a_dtype=%s wo_a_device=%s",
            status,
            reason,
            tuple(o.shape),
            tuple(o.stride()),
            o.dtype,
            o.device,
            tuple(wo_a.shape),
            tuple(wo_a.stride()),
            wo_a.dtype,
            wo_a.device,
        )
    if status != "miss":
        return
    key = (reason, tuple(o.shape), tuple(wo_a.shape))
    if key in _LOGGED_WO_A_MISSES:
        return
    _LOGGED_WO_A_MISSES.add(key)
    logger.warning(
        "TileLang wo_a strided GEMM dispatch miss: reason=%s "
        "o_shape=%s o_stride=%s o_dtype=%s wo_a_shape=%s wo_a_stride=%s wo_a_dtype=%s",
        reason,
        tuple(o.shape),
        tuple(o.stride()),
        o.dtype,
        tuple(wo_a.shape),
        tuple(wo_a.stride()),
        wo_a.dtype,
    )


def _wo_a_dispatch_miss_reason(o: torch.Tensor, wo_a: torch.Tensor) -> Optional[str]:
    if o.device.type != "musa" or wo_a.device.type != "musa":
        return "non_musa"
    if o.dtype != torch.bfloat16 or wo_a.dtype != torch.bfloat16:
        return "dtype_not_bfloat16"
    if o.dim() != 3 or wo_a.dim() != 3:
        return "rank_not_3"
    if tuple(o.shape[1:]) != (1, WO_A_D):
        return "unsupported_o_shape"
    if tuple(wo_a.shape) != (1, WO_A_R, WO_A_D):
        return "unsupported_wo_a_shape"
    if o.stride(-1) != 1 or wo_a.stride(-1) != 1:
        return "last_dim_not_contiguous"
    if o.device != wo_a.device:
        return "device_mismatch"
    num_tokens = int(o.shape[0])
    if num_tokens == 1 or num_tokens == 16:
        return None
    if 2 <= num_tokens <= 15:
        return None
    if 17 <= num_tokens <= 32:
        return None
    if num_tokens >= 2048 and num_tokens % 128 == 0 and o.stride(0) > WO_A_D:
        return None
    return "unsupported_num_tokens"


def try_wo_a_strided_gemm_musa(
    o: torch.Tensor,
    wo_a: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Try TileLang for torch.einsum("tgd,grd->tgr", o, wo_a).

    P0 intentionally supports only the DeepSeek V4 production BF16 shape
    G=1, D=4096, R=1024. Callers should fallback to torch.einsum on None.
    """
    reason = _wo_a_dispatch_miss_reason(o, wo_a)
    if reason is not None:
        _trace_wo_a_dispatch("miss", o, wo_a, reason)
        return None

    num_tokens = int(o.shape[0])
    a_2d = o[:, 0, :]
    b_2d = wo_a[0]
    a_stride_t = int(a_2d.stride(0))
    b_stride_r = int(b_2d.stride(0))

    try:
        if num_tokens == 1:
            kernel = _tilelang_wo_a_m1_splitk_kernel(b_stride_r=b_stride_r)
            branch = "m1_splitk"
            out_1d = kernel(a_2d[0], b_2d)
            _trace_wo_a_dispatch("hit", o, wo_a, branch)
            return out_1d.view(1, 1, WO_A_R)

        out = torch.empty(
            (num_tokens, 1, WO_A_R),
            device=o.device,
            dtype=torch.bfloat16,
        )
        c_2d = out[:, 0, :]
        c_stride_t = int(c_2d.stride(0))
        if 2 <= num_tokens <= 16:
            if a_stride_t == WO_A_D:
                kernel = _tilelang_wo_a_small_static_gemm_kernel(
                    num_tokens_static=num_tokens,
                    block_m=16,
                    a_stride_t=a_stride_t,
                    b_stride_r=b_stride_r,
                    c_stride_t=c_stride_t,
                    block_n=64,
                )
                branch = "small_static_bm16"
            else:
                kernel = _tilelang_wo_a_small_gemm_kernel(
                    block_m=16,
                    a_stride_t=a_stride_t,
                    b_stride_r=b_stride_r,
                    c_stride_t=c_stride_t,
                    block_n=64,
                )
                branch = "small_bm16"
            kernel(a_2d, b_2d, c_2d)
        elif 17 <= num_tokens <= 32:
            if a_stride_t == WO_A_D:
                kernel = _tilelang_wo_a_small_static_gemm_kernel(
                    num_tokens_static=num_tokens,
                    block_m=32,
                    a_stride_t=a_stride_t,
                    b_stride_r=b_stride_r,
                    c_stride_t=c_stride_t,
                    block_n=32,
                )
                branch = "small_static_bm32"
            else:
                kernel = _tilelang_wo_a_small_gemm_kernel(
                    block_m=32,
                    a_stride_t=a_stride_t,
                    b_stride_r=b_stride_r,
                    c_stride_t=c_stride_t,
                    block_n=32,
                )
                branch = "small_bm32"
            kernel(a_2d, b_2d, c_2d)
        elif num_tokens >= 2048:
            kernel = _tilelang_wo_a_large_gemm_kernel(
                a_stride_t,
                b_stride_r=b_stride_r,
                c_stride_t=c_stride_t,
                block_m=256,
                block_n=256,
                block_k=64,
                threads=512,
                num_stages=3,
            )
            branch = "large_gemm_bm256_bn256"
            kernel(a_2d, b_2d, c_2d)
        else:
            kernel = _tilelang_wo_a_large_gemm_kernel(
                a_stride_t,
                b_stride_r=b_stride_r,
                c_stride_t=c_stride_t,
                block_m=128,
                block_n=128,
                block_k=64,
                threads=128,
                num_stages=3,
            )
            branch = "mid_gemm_bm128_bn128"
            kernel(a_2d, b_2d, c_2d)
    except Exception as exc:
        _trace_wo_a_dispatch(
            "miss", o, wo_a, f"tilelang_exception:{type(exc).__name__}"
        )
        return None

    _trace_wo_a_dispatch("hit", o, wo_a, branch)
    return out


def wo_a_strided_gemm_musa(o: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
    out = try_wo_a_strided_gemm_musa(o, wo_a)
    if out is not None:
        return out
    return torch.einsum("tgd,grd->tgr", o, wo_a)


__all__ = [
    "try_wo_a_strided_gemm_musa",
    "wo_a_strided_gemm_musa",
]
