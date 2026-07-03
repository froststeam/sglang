import math

import torch

from ..kernels.mhc_kernels import (
    _tilelang_mhc_prenorm_splitk_deepgemm_ws_like_v0_kernel,
    _tilelang_mhc_prenorm_splitk_x_tme_cast_kernel,
)
from sglang.srt.environ import envs

MHC_PRENORM_BACKEND_AUTO = "auto"
MHC_PRENORM_BACKEND_DEEPGEMM = "deepgemm"
MHC_PRENORM_BACKEND_TILELANG = "tilelang"
MHC_PRENORM_BACKEND_TORCH = "torch"
_MHC_PRENORM_BACKENDS = {
    MHC_PRENORM_BACKEND_AUTO,
    MHC_PRENORM_BACKEND_DEEPGEMM,
    MHC_PRENORM_BACKEND_TILELANG,
    MHC_PRENORM_BACKEND_TORCH,
}
_TILELANG_PRENORM_IMPLS = {
    "auto",
    "deepgemm_ws_like_v0",
    "h200_splitk_x_tme_bk128",
}


def _prefer_tilelang_prenorm_auto(
    num_tokens: int, hc_hidden_size: int, return_partials: bool
) -> bool:
    # Decode MHC pre consumes deterministic split-K partials directly in
    # pre_big_fuse. The H200-style TileLang path is faster than DeepGEMM for
    # the measured small-batch target shape.
    return return_partials and hc_hidden_size == 16384 and num_tokens <= 64


__all__ = [
    "MHC_PRENORM_BACKEND_AUTO",
    "MHC_PRENORM_BACKEND_DEEPGEMM",
    "MHC_PRENORM_BACKEND_TILELANG",
    "MHC_PRENORM_BACKEND_TORCH",
    "mhc_prenorm_gemm_sqrsum",
    "mhc_prenorm_gemm_sqrsum_tilelang",
    "mhc_prenorm_gemm_sqrsum_deepgemm",
    "mhc_prenorm_gemm_sqrsum_torch",
    "select_mhc_prenorm_split_k",
]


def _get_mhc_prenorm_backend() -> str:
    backend = envs.SGLANG_OPT_MHC_PRENORM_BACKEND.get().strip().lower()
    if backend not in _MHC_PRENORM_BACKENDS:
        raise ValueError(
            f"Unsupported SGLANG_OPT_MHC_PRENORM_BACKEND={backend!r}. "
            f"Supported values are {sorted(_MHC_PRENORM_BACKENDS)}."
        )
    return backend


def _get_mhc_prenorm_split_k(*, default: int) -> int:
    split_k = envs.SGLANG_OPT_MHC_PRENORM_SPLIT_K.get()
    if split_k == 0:
        split_k = envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM_SPLIT_K.get()
    if split_k == 0:
        split_k = default
    if split_k < 0:
        raise ValueError(f"MHC prenorm split-K must be >= 0, got {split_k}")
    return split_k


def select_mhc_prenorm_split_k(num_tokens: int, hc_hidden_size: int) -> int:
    """Return the measured default split-K for DeepSeek V4 MHC PreNorm."""
    if hc_hidden_size == 16384:
        if num_tokens <= 64:
            return 64
        if num_tokens <= 128:
            return 16
        if num_tokens <= 256:
            return 8
        if num_tokens <= 1024:
            return 32
        if num_tokens <= 2048:
            return 16
        return 4

    # Conservative fallback for non-target shapes.
    return 16 if num_tokens <= 1024 else 8


def mhc_prenorm_gemm_sqrsum_tilelang(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    *,
    split_k: int,
    impl: str | None = None,
    return_partials: bool = False,
    fn_tf32_rounded: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert residual_flat.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    # Backward-compatible benchmark flag. FP32 operands are lowered to TF32 by
    # TileLang T.gemm; callers should pass raw FP32 weights.
    del fn_tf32_rounded

    num_tokens = residual_flat.shape[0]
    hc_hidden_size = residual_flat.shape[-1]
    mhc_mult3 = fn.shape[0]
    assert fn.shape == (mhc_mult3, hc_hidden_size)
    if split_k <= 0:
        raise ValueError(f"TileLang MHC prenorm split_k must be > 0, got {split_k}")

    impl = (impl or envs.SGLANG_OPT_MHC_PRENORM_TILELANG_IMPL.get()).strip().lower()
    if impl not in _TILELANG_PRENORM_IMPLS:
        raise ValueError(
            "SGLANG_OPT_MHC_PRENORM_TILELANG_IMPL must be one of "
            f"{sorted(_TILELANG_PRENORM_IMPLS)}, got {impl!r}. "
            "Experimental variants live in new_version/sglang."
        )
    if impl == "auto":
        impl = "h200_splitk_x_tme_bk128" if return_partials else "deepgemm_ws_like_v0"

    if hc_hidden_size % split_k != 0:
        raise ValueError(
            f"TileLang MHC prenorm requires K divisible by split_k, "
            f"got K={hc_hidden_size}, split_k={split_k}"
        )

    split_size = hc_hidden_size // split_k
    tilelang_stages = envs.SGLANG_OPT_MHC_PRENORM_TILELANG_STAGES.get()
    if tilelang_stages <= 0:
        tilelang_stages = 2
    if tilelang_stages != 2:
        raise ValueError(
            "SGLANG_OPT_MHC_PRENORM_TILELANG_STAGES must be 0 or 2 "
            f"for production MHC prenorm, got {tilelang_stages}"
        )

    tilelang_token_block = envs.SGLANG_OPT_MHC_PRENORM_TILELANG_TOKEN_BLOCK.get()
    if tilelang_token_block <= 0:
        tilelang_token_block = 64 if impl == "deepgemm_ws_like_v0" else 32

    def _run_stage_reduce(
        factory,
        *,
        hidden_block_value: int,
        partial_width: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_part = torch.empty(
            split_k,
            num_tokens,
            partial_width,
            dtype=torch.float32,
            device=residual_flat.device,
        )
        s_part = torch.empty(
            split_k, num_tokens, dtype=torch.float32, device=residual_flat.device
        )
        kernel_0, kernel_1 = factory(
            mhc_mult3,
            hc_hidden_size,
            split_k=split_k,
            token_block=tilelang_token_block,
            hidden_block=hidden_block_value,
            num_stages=tilelang_stages,
        )
        if not fn.is_contiguous():
            raise ValueError("TileLang MHC prenorm requires contiguous HC weights")
        kernel_0(residual_flat, fn, d_part, s_part)
        if return_partials:
            if partial_width != mhc_mult3:
                return d_part[:, :, :mhc_mult3].contiguous(), s_part
            return d_part, s_part
        d_out = torch.empty(
            num_tokens, mhc_mult3, dtype=torch.float32, device=residual_flat.device
        )
        s_out = torch.empty(
            num_tokens, dtype=torch.float32, device=residual_flat.device
        )
        kernel_1(d_part, s_part, d_out, s_out)
        return d_out, s_out

    if impl == "deepgemm_ws_like_v0":
        if tilelang_token_block not in (32, 64):
            raise ValueError("deepgemm_ws_like_v0 requires token_block=32 or 64")
        return _run_stage_reduce(
            _tilelang_mhc_prenorm_splitk_deepgemm_ws_like_v0_kernel,
            hidden_block_value=32,
        )

    if tilelang_token_block != 32:
        raise ValueError("h200_splitk_x_tme_bk128 requires token_block=32")
    if split_size % 128 != 0:
        raise ValueError(
            "h200_splitk_x_tme_bk128 requires split_size divisible by 128, "
            f"got split_size={split_size}"
        )
    return _run_stage_reduce(
        _tilelang_mhc_prenorm_splitk_x_tme_cast_kernel,
        hidden_block_value=128,
        partial_width=mhc_mult3,
    )


def mhc_prenorm_gemm_sqrsum_deepgemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    *,
    split_k: int,
    return_partials: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
        raise RuntimeError("SGLANG_OPT_DEEPGEMM_HC_PRENORM is disabled")

    try:
        from deep_gemm.interface import tf32_hc_prenorm_gemm
    except ImportError:
        from deep_gemm import tf32_hc_prenorm_gemm

    x_flat = residual_flat.view(residual_flat.shape[0], -1).bfloat16()
    num_tokens, _ = x_flat.shape
    mhc_mult3 = fn.shape[0]
    if split_k < 0:
        raise ValueError(f"DeepGEMM MHC prenorm split_k must be >= 0, got {split_k}")

    if split_k <= 1:
        d_out = torch.empty(
            num_tokens, mhc_mult3, dtype=torch.float32, device=residual_flat.device
        )
        s_out = torch.empty(num_tokens, dtype=torch.float32, device=residual_flat.device)
        tf32_hc_prenorm_gemm(
            x_flat,
            fn.float().contiguous(),
            d_out,
            s_out,
            num_splits=split_k if split_k > 0 else None,
        )
        return d_out, s_out

    d_part = torch.empty(
        split_k, num_tokens, mhc_mult3, dtype=torch.float32, device=residual_flat.device
    )
    s_part = torch.empty(
        split_k, num_tokens, dtype=torch.float32, device=residual_flat.device
    )
    tf32_hc_prenorm_gemm(
        x_flat,
        fn.float().contiguous(),
        d_part,
        s_part,
        num_splits=split_k,
    )
    if return_partials:
        return d_part, s_part
    return d_part.sum(dim=0), s_part.sum(dim=0)


def mhc_prenorm_gemm_sqrsum_torch(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_flat = residual_flat.view(residual_flat.shape[0], -1).float()
    return torch.mm(x_flat, fn.float().t()), x_flat.square().sum(dim=-1)


def mhc_prenorm_gemm_sqrsum(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    *,
    backend: str | None = None,
    split_k: int | None = None,
    return_backend: bool = False,
    return_partials: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]:
    backend = _get_mhc_prenorm_backend() if backend is None else backend.lower()
    if backend not in _MHC_PRENORM_BACKENDS:
        raise ValueError(
            f"Unsupported MHC prenorm backend={backend!r}. "
            f"Supported values are {sorted(_MHC_PRENORM_BACKENDS)}."
        )

    errors: list[str] = []

    def _finish(
        d_out: torch.Tensor, s_out: torch.Tensor, used_backend: str
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]:
        if return_backend:
            return d_out, s_out, used_backend
        return d_out, s_out

    num_tokens = residual_flat.shape[0]
    hc_hidden_size = residual_flat.view(num_tokens, -1).shape[1]

    if backend == MHC_PRENORM_BACKEND_AUTO and _prefer_tilelang_prenorm_auto(
        num_tokens, hc_hidden_size, return_partials
    ):
        tilelang_split_k = (
            split_k
            if split_k is not None
            else _get_mhc_prenorm_split_k(
                default=select_mhc_prenorm_split_k(num_tokens, hc_hidden_size)
            )
        )
        try:
            d_out, s_out = mhc_prenorm_gemm_sqrsum_tilelang(
                residual_flat,
                fn,
                split_k=tilelang_split_k,
                return_partials=return_partials,
            )
            return _finish(d_out, s_out, MHC_PRENORM_BACKEND_TILELANG)
        except Exception as exc:
            errors.append(f"tilelang: {type(exc).__name__}: {exc}")

    if backend in (MHC_PRENORM_BACKEND_AUTO, MHC_PRENORM_BACKEND_DEEPGEMM):
        deepgemm_split_k = (
            split_k
            if split_k is not None
            else _get_mhc_prenorm_split_k(
                default=select_mhc_prenorm_split_k(num_tokens, hc_hidden_size)
            )
        )
        try:
            d_out, s_out = mhc_prenorm_gemm_sqrsum_deepgemm(
                residual_flat,
                fn,
                split_k=deepgemm_split_k,
                return_partials=return_partials,
            )
            return _finish(d_out, s_out, MHC_PRENORM_BACKEND_DEEPGEMM)
        except Exception as exc:
            if backend == MHC_PRENORM_BACKEND_DEEPGEMM:
                raise
            errors.append(f"deepgemm: {type(exc).__name__}: {exc}")

    if backend in (MHC_PRENORM_BACKEND_AUTO, MHC_PRENORM_BACKEND_TILELANG):
        tilelang_split_k = (
            split_k
            if split_k is not None
            else _get_mhc_prenorm_split_k(
                default=select_mhc_prenorm_split_k(num_tokens, hc_hidden_size)
            )
        )
        try:
            d_out, s_out = mhc_prenorm_gemm_sqrsum_tilelang(
                residual_flat,
                fn,
                split_k=tilelang_split_k,
                return_partials=return_partials,
            )
            return _finish(d_out, s_out, MHC_PRENORM_BACKEND_TILELANG)
        except Exception as exc:
            if backend == MHC_PRENORM_BACKEND_TILELANG:
                raise
            errors.append(f"tilelang: {type(exc).__name__}: {exc}")

    if backend in (MHC_PRENORM_BACKEND_AUTO, MHC_PRENORM_BACKEND_TORCH):
        d_out, s_out = mhc_prenorm_gemm_sqrsum_torch(residual_flat, fn)
        return _finish(d_out, s_out, MHC_PRENORM_BACKEND_TORCH)

    raise RuntimeError("No MHC prenorm backend succeeded: " + "; ".join(errors))
