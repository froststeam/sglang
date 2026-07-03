from __future__ import annotations

import os

import pytest
import torch

from sglang.test.ci.ci_register import register_musa_ci
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.kernels.mhc_kernels import (
    _tilelang_mhc_pre_big_fuse_decode_split_kernel,
    _tilelang_mhc_pre_big_fuse_kernel,
)
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.mhc_ops import (
    _resolve_big_fuse_config,
    mhc_post,
    mhc_pre_big_fuse,
)
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.mhc_prenorm_ops import (
    mhc_prenorm_gemm_sqrsum_tilelang,
)

from ..utils import get_musa_device

register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="DeepSeek V4 MUSA operator test is opt-in outside smoke CI",
)


def _sinkhorn_ref(
    logits: torch.Tensor,
    *,
    eps: float,
    repeats: int,
) -> torch.Tensor:
    cm = logits.float()
    cm = torch.exp(cm - cm.amax(dim=-1, keepdim=True))
    cm = cm / cm.sum(dim=-1, keepdim=True) + eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeats - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + eps)
    return cm


def _mhc_pre_big_fuse_ref(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    *,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, mhc_mult, hidden_size = residual.shape
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mixes = gemm_out_mul.float().sum(dim=0)
    sqrsum = gemm_out_sqrsum.float().sum(dim=0)
    rms = torch.rsqrt(sqrsum / (mhc_mult * hidden_size) + rms_eps)
    mixes = mixes * rms[:, None]
    assert mixes.shape == (num_tokens, mhc_mult3)

    pre = (
        torch.sigmoid(mixes[:, :mhc_mult] * mhc_scale[0] + mhc_base[:mhc_mult])
        + mhc_pre_eps
    )
    post = (
        torch.sigmoid(
            mixes[:, mhc_mult : 2 * mhc_mult] * mhc_scale[1]
            + mhc_base[mhc_mult : 2 * mhc_mult]
        )
        * mhc_post_mult_value
    )
    comb_logits = (
        mixes[:, 2 * mhc_mult :].reshape(num_tokens, mhc_mult, mhc_mult) * mhc_scale[2]
        + mhc_base[2 * mhc_mult :].reshape(mhc_mult, mhc_mult)
    )
    comb = _sinkhorn_ref(comb_logits, eps=mhc_sinkhorn_eps, repeats=sinkhorn_repeat)
    layer_input = (pre[:, :, None] * residual.float()).sum(dim=1).to(residual.dtype)
    return post, comb, layer_input


def _mhc_pre_full_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    *,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, mhc_mult, hidden_size = residual.shape
    residual_flat = residual.view(num_tokens, mhc_mult * hidden_size).float()
    gemm_out_mul = torch.matmul(residual_flat, fn.float().t()).unsqueeze(0)
    gemm_out_sqrsum = (residual_flat * residual_flat).sum(dim=-1).unsqueeze(0)
    return _mhc_pre_big_fuse_ref(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
        rms_eps=rms_eps,
        mhc_pre_eps=mhc_pre_eps,
        mhc_sinkhorn_eps=mhc_sinkhorn_eps,
        mhc_post_mult_value=mhc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


def _assert_repeat_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return

    mismatch = actual != expected
    mismatch_count = int(mismatch.sum().item())
    if actual.dtype.is_floating_point:
        max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
        detail = f", max_abs_diff={max_abs_diff:.6g}"
    else:
        detail = ""
    pytest.fail(
        f"{name} is not repeat-deterministic: "
        f"mismatch_count={mismatch_count}/{actual.numel()}{detail}"
    )


def _assert_row_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    except AssertionError as exc:
        max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
        raise AssertionError(f"{name} max_abs_diff={max_abs_diff:.6g}") from exc


@pytest.mark.parametrize("num_tokens", [1, 32, 65, 513])
def test_mhc_post_is_repeat_deterministic_on_musa(num_tokens: int) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20261125 + num_tokens)

    hidden_size = 4096
    mhc_mult = 4
    x = (torch.randn((num_tokens, hidden_size), device=device) * 0.2).to(
        torch.bfloat16
    )
    residual = (
        torch.randn((num_tokens, mhc_mult, hidden_size), device=device) * 0.2
    ).to(torch.bfloat16)
    post_layer_mix = torch.randn(
        (num_tokens, mhc_mult, 1), device=device, dtype=torch.float32
    )
    comb_res_mix = torch.randn(
        (num_tokens, mhc_mult, mhc_mult), device=device, dtype=torch.float32
    )

    expected = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    torch.musa.synchronize()
    repeat_count = 50
    for repeat_idx in range(repeat_count):
        actual = mhc_post(x, residual, post_layer_mix, comb_res_mix)
        torch.musa.synchronize()
        prefix = f"repeat={repeat_idx}, num_tokens={num_tokens}"
        _assert_repeat_exact(f"mhc_post {prefix}", actual, expected)


@pytest.mark.parametrize("num_tokens", [32, 65, 513])
def test_mhc_post_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20261225 + num_tokens)

    hidden_size = 4096
    mhc_mult = 4
    probe_x = (torch.randn((hidden_size,), device=device) * 0.2).to(torch.bfloat16)
    probe_residual = (
        torch.randn((mhc_mult, hidden_size), device=device) * 0.2
    ).to(torch.bfloat16)
    probe_post_mix = torch.randn((mhc_mult, 1), device=device, dtype=torch.float32)
    probe_comb_mix = torch.randn(
        (mhc_mult, mhc_mult), device=device, dtype=torch.float32
    )

    def run_shape(
        shape_num_tokens: int,
        probe_positions: list[int],
    ) -> torch.Tensor:
        x = (torch.randn((shape_num_tokens, hidden_size), device=device) * 0.2).to(
            torch.bfloat16
        )
        residual = (
            torch.randn((shape_num_tokens, mhc_mult, hidden_size), device=device)
            * 0.2
        ).to(torch.bfloat16)
        post_layer_mix = torch.randn(
            (shape_num_tokens, mhc_mult, 1), device=device, dtype=torch.float32
        )
        comb_res_mix = torch.randn(
            (shape_num_tokens, mhc_mult, mhc_mult),
            device=device,
            dtype=torch.float32,
        )
        for pos in probe_positions:
            x[pos].copy_(probe_x)
            residual[pos].copy_(probe_residual)
            post_layer_mix[pos].copy_(probe_post_mix)
            comb_res_mix[pos].copy_(probe_comb_mix)
        out = mhc_post(x, residual, post_layer_mix, comb_res_mix)
        torch.musa.synchronize()
        assert torch.isfinite(out.float()).all()
        return out

    expected = run_shape(1, [0])[0].detach().clone()
    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    actual = run_shape(num_tokens, probe_positions)
    for pos in probe_positions:
        _assert_repeat_exact(
            f"mhc_post num_tokens={num_tokens}, pos={pos}",
            actual[pos],
            expected,
        )


@pytest.mark.parametrize("n_splits", [1, 32])
@pytest.mark.parametrize("num_tokens", [1, 32, 64, 65, 129, 513])
def test_mhc_pre_big_fuse_kernel_matches_reference_on_musa(
    num_tokens: int,
    n_splits: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20260525 + num_tokens)

    hidden_size = 4096
    mhc_mult = 4
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    rms_eps = 1.0e-6
    mhc_pre_eps = 1.0e-6
    mhc_sinkhorn_eps = 1.0e-6
    mhc_post_mult_value = 2.0
    sinkhorn_repeat = 4

    residual = (
        torch.randn((num_tokens, mhc_mult, hidden_size), device=device) * 0.2
    ).to(torch.bfloat16)
    gemm_out_mul = (
        torch.randn(
            (n_splits, num_tokens, mhc_mult3), device=device, dtype=torch.float32
        )
        * 0.03
    )
    gemm_out_sqrsum = (
        torch.rand((n_splits, num_tokens), device=device, dtype=torch.float32) * 64
        + 512
    )
    mhc_scale = torch.tensor([1.0, 0.7, 0.5], device=device, dtype=torch.float32)
    mhc_base = torch.linspace(
        -0.2, 0.2, steps=mhc_mult3, device=device, dtype=torch.float32
    )
    post_mix = torch.empty((num_tokens, mhc_mult), device=device, dtype=torch.float32)
    comb_mix = torch.empty(
        (num_tokens, mhc_mult * mhc_mult), device=device, dtype=torch.float32
    )
    layer_input = torch.empty(
        (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
    )

    threads, hidden_block, pass_config = _resolve_big_fuse_config(
        num_tokens, n_splits
    )
    kernel_factory = (
        _tilelang_mhc_pre_big_fuse_decode_split_kernel
        if num_tokens <= 64
        else _tilelang_mhc_pre_big_fuse_kernel
    )
    kernel = kernel_factory(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
    )
    kernel(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )

    ref_post, ref_comb, ref_layer_input = _mhc_pre_big_fuse_ref(
        gemm_out_mul.cpu(),
        gemm_out_sqrsum.cpu(),
        mhc_scale.cpu(),
        mhc_base.cpu(),
        residual.cpu(),
        rms_eps=rms_eps,
        mhc_pre_eps=mhc_pre_eps,
        mhc_sinkhorn_eps=mhc_sinkhorn_eps,
        mhc_post_mult_value=mhc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )

    torch.testing.assert_close(post_mix.cpu(), ref_post, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        comb_mix.cpu().reshape(num_tokens, mhc_mult, mhc_mult),
        ref_comb,
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        layer_input.cpu().float(), ref_layer_input.float(), rtol=0, atol=2e-3
    )


@pytest.mark.parametrize("n_splits", [1, 32])
@pytest.mark.parametrize("num_tokens", [32, 64, 65, 129, 513, 9216])
def test_mhc_pre_big_fuse_kernel_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
    n_splits: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20260925 + num_tokens * 100 + n_splits)

    hidden_size = 4096
    mhc_mult = 4
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    rms_eps = 1.0e-6
    mhc_pre_eps = 1.0e-6
    mhc_sinkhorn_eps = 1.0e-6
    mhc_post_mult_value = 2.0
    sinkhorn_repeat = 4

    probe_residual = (
        torch.randn((mhc_mult, hidden_size), device=device) * 0.2
    ).to(torch.bfloat16)
    probe_mul = (
        torch.randn((n_splits, mhc_mult3), device=device, dtype=torch.float32) * 0.03
    )
    probe_sqrsum = (
        torch.rand((n_splits,), device=device, dtype=torch.float32) * 64 + 512
    )
    mhc_scale = torch.tensor([1.0, 0.7, 0.5], device=device, dtype=torch.float32)
    mhc_base = torch.linspace(
        -0.2, 0.2, steps=mhc_mult3, device=device, dtype=torch.float32
    )

    def run_shape(
        shape_num_tokens: int,
        probe_positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = (
            torch.randn((shape_num_tokens, mhc_mult, hidden_size), device=device) * 0.2
        ).to(torch.bfloat16)
        gemm_out_mul = (
            torch.randn(
                (n_splits, shape_num_tokens, mhc_mult3),
                device=device,
                dtype=torch.float32,
            )
            * 0.03
        )
        gemm_out_sqrsum = (
            torch.rand(
                (n_splits, shape_num_tokens), device=device, dtype=torch.float32
            )
            * 64
            + 512
        )
        for pos in probe_positions:
            residual[pos].copy_(probe_residual)
            gemm_out_mul[:, pos, :].copy_(probe_mul)
            gemm_out_sqrsum[:, pos].copy_(probe_sqrsum)

        post_mix = torch.empty(
            (shape_num_tokens, mhc_mult), device=device, dtype=torch.float32
        )
        comb_mix = torch.empty(
            (shape_num_tokens, mhc_mult * mhc_mult),
            device=device,
            dtype=torch.float32,
        )
        layer_input = torch.empty(
            (shape_num_tokens, hidden_size), device=device, dtype=torch.bfloat16
        )

        threads, hidden_block, pass_config = _resolve_big_fuse_config(
            shape_num_tokens, n_splits
        )
        kernel_factory = (
            _tilelang_mhc_pre_big_fuse_decode_split_kernel
            if shape_num_tokens <= 64
            else _tilelang_mhc_pre_big_fuse_kernel
        )
        kernel = kernel_factory(
            hidden_size,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            n_splits=n_splits,
            mhc_mult=mhc_mult,
            threads=threads,
            hidden_block=hidden_block,
            pass_config=pass_config,
        )
        kernel(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
        )
        torch.musa.synchronize()
        assert torch.isfinite(post_mix).all()
        assert torch.isfinite(comb_mix).all()
        assert torch.isfinite(layer_input.float()).all()
        return post_mix, comb_mix, layer_input

    ref_post, ref_comb, ref_layer_input = run_shape(1, [0])
    expected_post = ref_post[0].detach().clone()
    expected_comb = ref_comb[0].detach().clone()
    expected_layer_input = ref_layer_input[0].detach().clone()

    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    post_mix, comb_mix, layer_input = run_shape(num_tokens, probe_positions)
    for pos in probe_positions:
        prefix = f"num_tokens={num_tokens}, n_splits={n_splits}, pos={pos}"
        _assert_row_close(f"post_mix {prefix}", post_mix[pos], expected_post)
        _assert_row_close(f"comb_mix {prefix}", comb_mix[pos], expected_comb)
        _assert_row_close(
            f"layer_input {prefix}",
            layer_input[pos].float(),
            expected_layer_input.float(),
        )


@pytest.mark.parametrize("num_tokens", [64, 65])
def test_mhc_pre_big_fuse_public_dispatch_matches_reference_on_musa(
    num_tokens: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20260625 + num_tokens)

    hidden_size = 4096
    mhc_mult = 4
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    rms_eps = 1.0e-6
    mhc_pre_eps = 1.0e-6
    mhc_sinkhorn_eps = 1.0e-6
    mhc_post_mult_value = 2.0
    sinkhorn_repeat = 4

    residual = (
        torch.randn((num_tokens, mhc_mult, hidden_size), device=device) * 0.2
    ).to(torch.bfloat16)
    fn = torch.randn(
        (mhc_mult3, mhc_mult * hidden_size), device=device, dtype=torch.float32
    ) * 0.01
    mhc_scale = torch.tensor([1.0, 0.7, 0.5], device=device, dtype=torch.float32)
    mhc_base = torch.linspace(
        -0.2, 0.2, steps=mhc_mult3, device=device, dtype=torch.float32
    )

    # Use torch prenorm so this test isolates public dispatch into the two
    # local MHC pre finalization kernels: <=64 decode-split and >64 prefill.
    with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True):
        with envs.SGLANG_OPT_MHC_PRENORM_BACKEND.override("torch"):
            post_mix, comb_mix, layer_input = mhc_pre_big_fuse(
                residual,
                fn,
                mhc_scale,
                mhc_base,
                rms_eps,
                mhc_pre_eps,
                mhc_sinkhorn_eps,
                mhc_post_mult_value,
                sinkhorn_repeat,
                n_splits=1,
            )

    ref_post, ref_comb, ref_layer_input = _mhc_pre_full_ref(
        residual.cpu(),
        fn.cpu(),
        mhc_scale.cpu(),
        mhc_base.cpu(),
        rms_eps=rms_eps,
        mhc_pre_eps=mhc_pre_eps,
        mhc_sinkhorn_eps=mhc_sinkhorn_eps,
        mhc_post_mult_value=mhc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )

    torch.testing.assert_close(
        post_mix.squeeze(-1).cpu(), ref_post, rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(comb_mix.cpu(), ref_comb, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        layer_input.cpu().float(), ref_layer_input.float(), rtol=0, atol=2e-3
    )


@pytest.mark.parametrize("n_splits", [1, 32])
@pytest.mark.parametrize("num_tokens", [32, 64, 65, 129, 513])
def test_mhc_pre_big_fuse_kernel_is_repeat_deterministic_on_musa(
    num_tokens: int,
    n_splits: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20260725 + num_tokens * 100 + n_splits)

    hidden_size = 4096
    mhc_mult = 4
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    rms_eps = 1.0e-6
    mhc_pre_eps = 1.0e-6
    mhc_sinkhorn_eps = 1.0e-6
    mhc_post_mult_value = 2.0
    sinkhorn_repeat = 4

    residual = (
        torch.randn((num_tokens, mhc_mult, hidden_size), device=device) * 0.2
    ).to(torch.bfloat16)
    gemm_out_mul = (
        torch.randn(
            (n_splits, num_tokens, mhc_mult3), device=device, dtype=torch.float32
        )
        * 0.03
    )
    gemm_out_sqrsum = (
        torch.rand((n_splits, num_tokens), device=device, dtype=torch.float32) * 64
        + 512
    )
    mhc_scale = torch.tensor([1.0, 0.7, 0.5], device=device, dtype=torch.float32)
    mhc_base = torch.linspace(
        -0.2, 0.2, steps=mhc_mult3, device=device, dtype=torch.float32
    )

    threads, hidden_block, pass_config = _resolve_big_fuse_config(
        num_tokens, n_splits
    )
    kernel_factory = (
        _tilelang_mhc_pre_big_fuse_decode_split_kernel
        if num_tokens <= 64
        else _tilelang_mhc_pre_big_fuse_kernel
    )
    kernel = kernel_factory(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
    )

    def run_once() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        post_mix = torch.empty(
            (num_tokens, mhc_mult), device=device, dtype=torch.float32
        )
        comb_mix = torch.empty(
            (num_tokens, mhc_mult * mhc_mult), device=device, dtype=torch.float32
        )
        layer_input = torch.empty(
            (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
        )
        post_mix.fill_(float("nan"))
        comb_mix.fill_(float("nan"))
        layer_input.fill_(float("nan"))
        kernel(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
        )
        torch.musa.synchronize()
        assert torch.isfinite(post_mix).all()
        assert torch.isfinite(comb_mix).all()
        assert torch.isfinite(layer_input.float()).all()
        return post_mix, comb_mix, layer_input

    expected_post, expected_comb, expected_layer_input = run_once()
    repeat_count = 100
    for repeat_idx in range(repeat_count):
        post_mix, comb_mix, layer_input = run_once()
        prefix = f"repeat={repeat_idx}, num_tokens={num_tokens}, n_splits={n_splits}"
        _assert_repeat_exact(f"post_mix {prefix}", post_mix, expected_post)
        _assert_repeat_exact(f"comb_mix {prefix}", comb_mix, expected_comb)
        _assert_repeat_exact(
            f"layer_input {prefix}", layer_input, expected_layer_input
        )


def test_mhc_prenorm_x_tme_cast_stage0_handles_9216_tokens_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20260525)

    num_tokens = 9216
    hc_hidden_size = 16384
    mhc_mult3 = 24
    split_k = 32

    residual_flat = (
        torch.randn((num_tokens, hc_hidden_size), device=device) * 0.1
    ).to(torch.bfloat16)
    fn = torch.randn(
        (mhc_mult3, hc_hidden_size), device=device, dtype=torch.float32
    ) * 0.01

    d_part, s_part = mhc_prenorm_gemm_sqrsum_tilelang(
        residual_flat,
        fn,
        split_k=split_k,
        impl="h200_splitk_x_tme_bk128",
        return_partials=True,
    )

    assert d_part.shape == (split_k, num_tokens, mhc_mult3)
    assert s_part.shape == (split_k, num_tokens)
    assert torch.isfinite(d_part).all()
    assert torch.isfinite(s_part).all()

    split_size = hc_hidden_size // split_k
    sample_tokens = torch.tensor([0, 31, 32, num_tokens - 1], device=device)
    for split_id in (0, split_k - 1):
        start = split_id * split_size
        end = start + split_size
        x_ref = residual_flat[sample_tokens, start:end].float().cpu()
        fn_ref = fn[:, start:end].float().cpu()
        d_ref = x_ref @ fn_ref.t()
        s_ref = (x_ref * x_ref).sum(dim=1)

        torch.testing.assert_close(
            d_part[split_id, sample_tokens, :].cpu(),
            d_ref,
            rtol=2e-3,
            atol=2e-3,
        )
        torch.testing.assert_close(
            s_part[split_id, sample_tokens].cpu(),
            s_ref,
            rtol=2e-3,
            atol=2e-3,
        )


@pytest.mark.parametrize("num_tokens", [64, 65, 129, 9216])
def test_mhc_prenorm_x_tme_cast_stage0_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20261025 + num_tokens)

    hc_hidden_size = 16384
    mhc_mult3 = 24
    split_k = 32

    probe_residual = (
        torch.randn((hc_hidden_size,), device=device) * 0.1
    ).to(torch.bfloat16)
    fn = torch.randn(
        (mhc_mult3, hc_hidden_size), device=device, dtype=torch.float32
    ) * 0.01

    def run_shape(
        shape_num_tokens: int,
        probe_positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_flat = (
            torch.randn((shape_num_tokens, hc_hidden_size), device=device) * 0.1
        ).to(torch.bfloat16)
        for pos in probe_positions:
            residual_flat[pos].copy_(probe_residual)

        d_part, s_part = mhc_prenorm_gemm_sqrsum_tilelang(
            residual_flat,
            fn,
            split_k=split_k,
            impl="h200_splitk_x_tme_bk128",
            return_partials=True,
        )
        torch.musa.synchronize()
        assert torch.isfinite(d_part).all()
        assert torch.isfinite(s_part).all()
        return d_part, s_part

    ref_d_part, ref_s_part = run_shape(1, [0])
    expected_d_part = ref_d_part[:, 0, :].detach().clone()
    expected_s_part = ref_s_part[:, 0].detach().clone()

    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    d_part, s_part = run_shape(num_tokens, probe_positions)
    for pos in probe_positions:
        prefix = f"num_tokens={num_tokens}, split_k={split_k}, pos={pos}"
        _assert_repeat_exact(f"d_part {prefix}", d_part[:, pos, :], expected_d_part)
        _assert_repeat_exact(f"s_part {prefix}", s_part[:, pos], expected_s_part)


@pytest.mark.parametrize("num_tokens,split_k", [(64, 32), (65, 32), (9216, 32)])
def test_mhc_prenorm_x_tme_cast_stage0_is_repeat_deterministic_on_musa(
    num_tokens: int,
    split_k: int,
) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")
    device = get_musa_device()
    torch.manual_seed(20260825 + num_tokens)

    hc_hidden_size = 16384
    mhc_mult3 = 24

    residual_flat = (
        torch.randn((num_tokens, hc_hidden_size), device=device) * 0.1
    ).to(torch.bfloat16)
    fn = torch.randn(
        (mhc_mult3, hc_hidden_size), device=device, dtype=torch.float32
    ) * 0.01

    def run_once() -> tuple[torch.Tensor, torch.Tensor]:
        d_part, s_part = mhc_prenorm_gemm_sqrsum_tilelang(
            residual_flat,
            fn,
            split_k=split_k,
            impl="h200_splitk_x_tme_bk128",
            return_partials=True,
        )
        torch.musa.synchronize()
        assert torch.isfinite(d_part).all()
        assert torch.isfinite(s_part).all()
        return d_part, s_part

    expected_d_part, expected_s_part = run_once()
    repeat_count = 20 if num_tokens < 1024 else 5
    for repeat_idx in range(repeat_count):
        d_part, s_part = run_once()
        prefix = (
            f"repeat={repeat_idx}, num_tokens={num_tokens}, split_k={split_k}"
        )
        _assert_repeat_exact(f"d_part {prefix}", d_part, expected_d_part)
        _assert_repeat_exact(f"s_part {prefix}", s_part, expected_s_part)
