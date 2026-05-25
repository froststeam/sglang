from importlib import import_module

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="MUSA device is not available.",
)


def test_gdn_fused_proj_tilelang_matches_triton():
    from sglang.jit_kernel.triton.gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous as triton_gdn,
    )

    tilelang_gdn = import_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.tilelang.fla.gdn_fused_proj"
    ).fused_qkvzba_split_reshape_cat_contiguous

    torch.manual_seed(0)
    m = 16
    num_heads_qk, num_heads_v, head_qk, head_v = 4, 8, 128, 128
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    total_qkvz = qkv_dim + num_heads_v * head_v
    mixed_qkvz = torch.randn(m, total_qkvz, device="musa", dtype=torch.float16)
    mixed_ba = torch.randn(m, num_heads_v * 2, device="musa", dtype=torch.float16)

    out_tilelang = tilelang_gdn(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )
    out_triton = triton_gdn(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )
    torch.musa.synchronize()

    for actual, expected in zip(out_tilelang, out_triton):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_causal_conv1d_tilelang_matches_triton():
    from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.causal_conv1d import (
        causal_conv1d_fwd as tilelang_causal_conv1d_fwd,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as triton_causal_conv1d_fwd,
    )

    torch.manual_seed(0)
    batch, seq_len, dim, width = 2, 8, 512, 4
    dtype = torch.float16
    total = batch * seq_len
    query_start_loc = torch.arange(
        0, total + 1, seq_len, device="musa", dtype=torch.int32
    )
    x = torch.randn(total, dim, device="musa", dtype=dtype).contiguous().t()
    weight = torch.randn(dim, width, device="musa", dtype=dtype)
    bias = torch.randn(dim, device="musa", dtype=dtype)
    state = torch.randn(batch, dim, width - 1, device="musa", dtype=dtype)
    cache_indices = torch.arange(batch, device="musa", dtype=torch.int32)
    has_initial_state = torch.ones(batch, device="musa", dtype=torch.bool)

    out_tilelang = tilelang_causal_conv1d_fwd(
        x,
        weight,
        bias,
        state.clone(),
        query_start_loc,
        [seq_len] * batch,
        cache_indices,
        has_initial_state,
        "silu",
    )
    out_triton = triton_causal_conv1d_fwd(
        x,
        weight,
        bias,
        state.clone(),
        query_start_loc,
        [seq_len] * batch,
        cache_indices,
        has_initial_state,
        "silu",
    )
    torch.musa.synchronize()

    torch.testing.assert_close(out_tilelang, out_triton, rtol=1e-2, atol=1e-2)


def test_rms_norm_gated_tilelang_matches_triton():
    from sglang.srt.layers.attention.fla.layernorm_gated import (
        rms_norm_gated as triton_rms_norm_gated,
    )

    tilelang_rms_norm_gated = import_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.tilelang.fla.layernorm_gated"
    ).rms_norm_gated

    torch.manual_seed(0)
    m, n = 32, 1024
    x = torch.randn(m, n, device="musa", dtype=torch.float16)
    z = torch.randn_like(x)
    weight = torch.randn(n, device="musa", dtype=torch.float16)

    out_tilelang = tilelang_rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="swish",
    )
    out_triton = triton_rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="swish",
    )
    torch.musa.synchronize()

    torch.testing.assert_close(out_tilelang, out_triton, rtol=1e-2, atol=1e-2)
