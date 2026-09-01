from importlib import import_module

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="MUSA device is not available.",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_diffusion_qknorm_rope_matches_reference(dtype):
    from sglang.srt.hardware_backend.musa.jit_kernel.csrc.diffusion import (
        musa_qknorm_rope,
    )

    torch.manual_seed(0)
    tokens, heads, head_dim = 16, 24, 128
    q = torch.randn(tokens, heads, head_dim, device="musa", dtype=dtype)
    k = torch.randn_like(q)
    q_weight = torch.randn(head_dim, device="musa", dtype=dtype)
    k_weight = torch.randn(head_dim, device="musa", dtype=dtype)
    phase = torch.randn(tokens, head_dim // 2, device="musa", dtype=torch.float32)
    rope_cache = torch.cat((phase.cos(), phase.sin()), dim=-1)
    positions = torch.arange(tokens, device="musa", dtype=torch.int64)

    def reference(x, weight):
        value = x.float()
        value *= torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
        value *= weight.float()
        pairs = value.reshape(tokens, heads, head_dim // 2, 2)
        cos = rope_cache[:, : head_dim // 2].unsqueeze(1)
        sin = rope_cache[:, head_dim // 2 :].unsqueeze(1)
        even, odd = pairs[..., 0], pairs[..., 1]
        return torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).reshape_as(x)

    q_expected = reference(q, q_weight)
    k_expected = reference(k, k_weight)
    musa_qknorm_rope(q, k, q_weight, k_weight, rope_cache, positions, 1e-6)
    torch.musa.synchronize()

    torch.testing.assert_close(q.float(), q_expected, rtol=0, atol=0.0625)
    torch.testing.assert_close(k.float(), k_expected, rtol=0, atol=0.0625)


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

    padded_seq_lens_cpu = [seq_len] * batch + [seq_len * 4]
    out_tilelang_padded_lens = tilelang_causal_conv1d_fwd(
        x,
        weight,
        bias,
        state.clone(),
        query_start_loc,
        padded_seq_lens_cpu,
        cache_indices,
        has_initial_state,
        "silu",
    )
    torch.musa.synchronize()

    torch.testing.assert_close(
        out_tilelang_padded_lens, out_triton, rtol=1e-2, atol=1e-2
    )


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


@pytest.mark.parametrize("batch_size", [1, 5, 8])
def test_sigmoid_mul_musa_jit_matches_torch(batch_size):
    from sglang.srt.hardware_backend.musa.jit_kernel import sigmoid_mul

    torch.manual_seed(batch_size)
    gate = torch.randn(
        batch_size, 2048, device="musa", dtype=torch.bfloat16
    )
    value = torch.randn_like(gate)
    actual = sigmoid_mul(gate, value)
    expected = torch.sigmoid(gate.float()) * value.float()
    torch.musa.synchronize()

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=1e-2)
