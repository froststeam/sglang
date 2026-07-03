import torch

from ..kernels.mhc_kernels import (
    _tilelang_mhc_post_2d_kernel,
    _tilelang_mhc_post_kernel,
    _tilelang_mhc_pre_big_fuse_decode_split_kernel,
    _tilelang_mhc_pre_big_fuse_kernel,
    _tilelang_mhc_pre_norm_fn_fwd_mul_kernel,
    round_to_tf32,
)
from sglang.srt.environ import envs


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"DeepSeekV4 MUSA MHC {name} must be contiguous.")


def _resolve_big_fuse_config(
    num_tokens: int, n_splits: int = 0
) -> tuple[int, int, str]:
    threads = envs.SGLANG_OPT_MHC_PRE_BIG_FUSE_THREADS.get()
    hidden_block = envs.SGLANG_OPT_MHC_PRE_BIG_FUSE_HIDDEN_BLOCK.get()
    pass_config = envs.SGLANG_OPT_MHC_PRE_BIG_FUSE_PASS_CONFIG.get().strip().lower()

    is_tiny_decode = num_tokens <= 32
    is_decode_like = num_tokens <= 64
    is_mid_prefill = 128 < num_tokens <= 512
    if threads <= 0:
        threads = 128 if is_tiny_decode else 256 if is_decode_like else 128
    if hidden_block <= 0:
        hidden_block = 512 if is_tiny_decode or is_mid_prefill else 1024
    if pass_config == "auto":
        pass_config = (
            "aggressive_index32"
            if (is_decode_like or is_mid_prefill) and n_splits != 1
            else "safe"
        )
    return threads, hidden_block, pass_config


def _resolve_post_config(num_tokens: int) -> tuple[int, int, str, bool, str]:
    threads = envs.SGLANG_OPT_MHC_POST_THREADS.get()
    hidden_block = envs.SGLANG_OPT_MHC_POST_HIDDEN_BLOCK.get()
    pass_config = envs.SGLANG_OPT_MHC_POST_PASS_CONFIG.get().strip().lower()
    layout = envs.SGLANG_OPT_MHC_POST_LAYOUT.get().strip().lower()
    if threads <= 0:
        threads = 256
    if hidden_block <= 0:
        hidden_block = 512
    if pass_config == "auto":
        pass_config = "safe"
    if layout == "auto":
        layout = "2d"
    if layout not in ("1d", "2d"):
        raise ValueError(
            "SGLANG_OPT_MHC_POST_LAYOUT must be one of 'auto', '1d', or '2d', "
            f"got {layout!r}"
        )
    return (
        threads,
        hidden_block,
        pass_config,
        envs.SGLANG_OPT_MHC_POST_DIRECT_STORE.get(),
        layout,
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, mhc_mult, -1).squeeze(-1)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, mhc_mult, mhc_mult)

    assert mhc_mult == 4
    assert x_flat.shape == (num_tokens, hidden_size)
    assert post_layer_mix_flat.shape == (num_tokens, mhc_mult)
    assert comb_res_mix_flat.shape == (num_tokens, mhc_mult, mhc_mult)

    out = torch.empty_like(residual_flat)
    threads, hidden_block, pass_config, direct_store, layout = _resolve_post_config(
        num_tokens
    )
    kernel_factory = (
        _tilelang_mhc_post_2d_kernel if layout == "2d" else _tilelang_mhc_post_kernel
    )
    kernel_factory(
        hidden_size,
        mhc_mult=mhc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
        direct_store=direct_store,
    )(
        comb_res_mix_flat,
        residual_flat,
        post_layer_mix_flat,
        x_flat,
        out,
    )
    return out.view(*outer_shape, mhc_mult, hidden_size)


def _try_prenorm_backend(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    *,
    mhc_mult3: int,
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    from .mhc_prenorm_ops import (
        mhc_prenorm_gemm_sqrsum,
        select_mhc_prenorm_split_k,
    )

    split_k = None
    if (
        envs.SGLANG_OPT_MHC_PRENORM_SPLIT_K.get() == 0
        and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM_SPLIT_K.get() == 0
    ):
        split_k = select_mhc_prenorm_split_k(
            residual_flat.shape[0], residual_flat.shape[1] * residual_flat.shape[2]
        )

    d_out, s_out = mhc_prenorm_gemm_sqrsum(
        residual_flat.view(residual_flat.shape[0], -1),
        fn,
        split_k=split_k,
        return_partials=True,
    )
    assert d_out.shape[-1] == mhc_mult3
    if d_out.ndim == 2:
        d_out = d_out.unsqueeze(0)
    if s_out.ndim == 1:
        s_out = s_out.unsqueeze(0)
    return True, d_out, s_out


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    mhc_hidden_size = mhc_mult * hidden_size

    assert mhc_mult == 4
    assert mhc_mult3 <= 32
    assert fn.shape == (mhc_mult3, mhc_hidden_size)
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    _require_contiguous("residual", residual)
    _require_contiguous("fn", fn)
    _require_contiguous("mhc_scale", mhc_scale)
    _require_contiguous("mhc_base", mhc_base)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    prenorm_ok, gemm_out_mul, gemm_out_sqrsum = _try_prenorm_backend(
        residual_flat,
        fn,
        mhc_mult3=mhc_mult3,
    )
    if not prenorm_ok:
        # This mirrors the current tilekernels implementation: the public
        # n_splits argument is accepted, but the TileLang matmul fallback is
        # single-split. The main optimized path above uses DeepGEMM split-K.
        n_splits = 1
        gemm_out_mul = torch.empty(
            n_splits,
            num_tokens,
            mhc_mult3,
            dtype=torch.float32,
            device=residual.device,
        )
        gemm_out_sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )

        fn_tf32 = round_to_tf32(fn)
        fwd_mul_kernel = _tilelang_mhc_pre_norm_fn_fwd_mul_kernel(
            mhc_mult3, 1, mhc_hidden_size
        )
        fwd_mul_kernel(
            residual_flat.view(-1, mhc_hidden_size),
            fn_tf32,
            gemm_out_mul.view(-1, 1, mhc_mult3),
            gemm_out_sqrsum.view(-1, 1),
        )

    _require_contiguous("gemm_out_mul", gemm_out_mul)
    _require_contiguous("gemm_out_sqrsum", gemm_out_sqrsum)
    _require_contiguous("post_mix", post_mix)
    _require_contiguous("comb_mix", comb_mix)
    _require_contiguous("layer_input", layer_input)

    threads, hidden_block, pass_config = _resolve_big_fuse_config(
        num_tokens, gemm_out_mul.shape[0]
    )
    kernel_factory = (
        _tilelang_mhc_pre_big_fuse_decode_split_kernel
        if num_tokens <= 64
        else _tilelang_mhc_pre_big_fuse_kernel
    )
    kernel_factory(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=gemm_out_mul.shape[0],
        mhc_mult=mhc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    return (
        post_mix.view(*outer_shape, mhc_mult, 1),
        comb_mix.view(*outer_shape, mhc_mult, mhc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )
