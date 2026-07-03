from __future__ import annotations

import os

import pytest
import torch
import triton

from sglang.test.ci.ci_register import register_musa_ci
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.kernels import wo_a_kernels

from ..utils import MUSA_OPS, get_musa_device

register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="DeepSeek V4 MUSA operator test is opt-in outside smoke CI",
)

try_wo_a_strided_gemm_musa = MUSA_OPS.try_wo_a_strided_gemm_musa


WO_A_D = 4096
WO_A_R = 1024


def _make_wo_a_inputs(
    num_tokens: int,
    a_stride_t: int,
    *,
    b_stride_r: int = WO_A_D,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_musa_device()
    torch.manual_seed(20260523 + num_tokens + a_stride_t + b_stride_r)

    if a_stride_t == WO_A_D:
        o = torch.randn((num_tokens, 1, WO_A_D), device=device, dtype=torch.bfloat16)
    else:
        assert a_stride_t % WO_A_D == 0
        storage_groups = a_stride_t // WO_A_D
        base = torch.randn(
            (num_tokens, storage_groups, WO_A_D),
            device=device,
            dtype=torch.bfloat16,
        )
        o = base[:, 0:1, :]
        assert o.stride(0) == a_stride_t

    if b_stride_r == WO_A_D:
        wo_a = torch.randn((1, WO_A_R, WO_A_D), device=device, dtype=torch.bfloat16)
    else:
        assert b_stride_r > WO_A_D
        base = torch.randn((1, WO_A_R, b_stride_r), device=device, dtype=torch.bfloat16)
        wo_a = base[:, :, :WO_A_D]
        assert wo_a.stride(1) == b_stride_r

    assert o.stride(-1) == 1
    assert wo_a.stride(-1) == 1
    return o, wo_a


def _wo_a_ref(o: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
    return torch.mm(o[:, 0, :], wo_a[0].t()).view(o.shape[0], 1, WO_A_R)


def _assert_wo_a_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float() - expected.float()).abs()
    # TileLang and muDNN/torch may not use exactly the same BF16 reduction order.
    assert float(diff.mean().item()) <= 5e-3
    assert float(diff.max().item()) <= 2.0


def _assert_wo_a_repeat_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    context: str,
) -> None:
    if torch.equal(actual, expected):
        return
    diff = (actual.float() - expected.float()).abs()
    pytest.fail(
        f"WO_A output is not repeat-deterministic for {context}: "
        f"mismatch_count={int((actual != expected).sum().item())}/{actual.numel()}, "
        f"max_abs_diff={float(diff.max().item()):.6g}"
    )


@pytest.mark.parametrize(
    "num_tokens,a_stride_t",
    [
        (1, WO_A_D),
        (1, 8 * WO_A_D),
        (2, WO_A_D),
        (8, 8 * WO_A_D),
        (16, WO_A_D),
        (17, WO_A_D),
        (20, 8 * WO_A_D),
        (31, WO_A_D),
        (32, 8 * WO_A_D),
        (2048, 8 * WO_A_D),
    ],
)
def test_wo_a_tilelang_correctness_covers_dynamic_m_and_a_stride(
    num_tokens: int,
    a_stride_t: int,
) -> None:
    o, wo_a = _make_wo_a_inputs(num_tokens, a_stride_t)

    out = try_wo_a_strided_gemm_musa(o, wo_a)

    assert out is not None
    assert tuple(out.shape) == (num_tokens, 1, WO_A_R)
    _assert_wo_a_close(out, _wo_a_ref(o, wo_a))


@pytest.mark.parametrize("num_tokens,a_stride_t", [(20, WO_A_D), (20, 8 * WO_A_D)])
def test_wo_a_tilelang_is_repeat_deterministic(
    num_tokens: int,
    a_stride_t: int,
) -> None:
    o, wo_a = _make_wo_a_inputs(num_tokens, a_stride_t)

    expected = try_wo_a_strided_gemm_musa(o, wo_a)
    assert expected is not None
    torch.musa.synchronize()
    for repeat_idx in range(20):
        actual = try_wo_a_strided_gemm_musa(o, wo_a)
        assert actual is not None
        torch.musa.synchronize()
        _assert_wo_a_repeat_exact(
            actual,
            expected,
            context=(
                f"repeat={repeat_idx}, num_tokens={num_tokens}, "
                f"a_stride_t={a_stride_t}"
            ),
        )


@pytest.mark.parametrize("a_stride_t", [WO_A_D, 8 * WO_A_D])
def test_wo_a_tilelang_probe_row_is_batch_shape_invariant(a_stride_t: int) -> None:
    device = get_musa_device()
    torch.manual_seed(20261525 + a_stride_t)
    num_tokens = 32

    probe = torch.randn((WO_A_D,), device=device, dtype=torch.bfloat16)
    wo_a = torch.randn((1, WO_A_R, WO_A_D), device=device, dtype=torch.bfloat16)

    if a_stride_t == WO_A_D:
        baseline_o = probe.view(1, 1, WO_A_D).contiguous()
        o = torch.randn((num_tokens, 1, WO_A_D), device=device, dtype=torch.bfloat16)
    else:
        storage_groups = a_stride_t // WO_A_D
        baseline_base = torch.empty(
            (1, storage_groups, WO_A_D), device=device, dtype=torch.bfloat16
        )
        baseline_base.normal_()
        baseline_base[0, 0].copy_(probe)
        baseline_o = baseline_base[:, 0:1, :]
        base = torch.randn(
            (num_tokens, storage_groups, WO_A_D),
            device=device,
            dtype=torch.bfloat16,
        )
        o = base[:, 0:1, :]
        assert o.stride(0) == a_stride_t

    expected = try_wo_a_strided_gemm_musa(baseline_o, wo_a)
    assert expected is not None
    torch.musa.synchronize()
    expected_row = expected[0].detach().clone()

    probe_positions = [0, 17, num_tokens - 1]
    for pos in probe_positions:
        o[pos, 0].copy_(probe)

    actual = try_wo_a_strided_gemm_musa(o, wo_a)
    assert actual is not None
    torch.musa.synchronize()
    for pos in probe_positions:
        _assert_wo_a_close(actual[pos : pos + 1], expected_row.view(1, 1, WO_A_R))


def test_wo_a_small_strided_dynamic_m_reuses_same_kernel_for_shape_changes() -> None:
    wo_a_kernels._tilelang_wo_a_small_gemm_kernel.cache_clear()

    o20, wo_a = _make_wo_a_inputs(20, 8 * WO_A_D)
    out20 = try_wo_a_strided_gemm_musa(o20, wo_a)
    assert out20 is not None
    _assert_wo_a_close(out20, _wo_a_ref(o20, wo_a))

    cache_after_m20 = wo_a_kernels._tilelang_wo_a_small_gemm_kernel.cache_info()
    assert cache_after_m20.misses == 1

    o32 = torch.randn(
        (32, 8, WO_A_D), device=o20.device, dtype=torch.bfloat16
    )[:, 0:1, :]
    out32 = try_wo_a_strided_gemm_musa(o32, wo_a)
    assert out32 is not None
    _assert_wo_a_close(out32, _wo_a_ref(o32, wo_a))

    cache_after_m32 = wo_a_kernels._tilelang_wo_a_small_gemm_kernel.cache_info()
    assert cache_after_m32.misses == cache_after_m20.misses
    assert cache_after_m32.hits >= cache_after_m20.hits + 1


def test_wo_a_small_compact_static_m_uses_shape_specialized_kernel() -> None:
    wo_a_kernels._tilelang_wo_a_small_static_gemm_kernel.cache_clear()
    wo_a_kernels._tilelang_wo_a_small_gemm_kernel.cache_clear()

    o20, wo_a = _make_wo_a_inputs(20, WO_A_D)
    out20 = try_wo_a_strided_gemm_musa(o20, wo_a)
    assert out20 is not None
    _assert_wo_a_close(out20, _wo_a_ref(o20, wo_a))

    static_after_m20 = wo_a_kernels._tilelang_wo_a_small_static_gemm_kernel.cache_info()
    dynamic_after_m20 = wo_a_kernels._tilelang_wo_a_small_gemm_kernel.cache_info()
    assert static_after_m20.misses == 1
    assert dynamic_after_m20.misses == 0

    o32, _ = _make_wo_a_inputs(32, WO_A_D)
    out32 = try_wo_a_strided_gemm_musa(o32, wo_a)
    assert out32 is not None
    _assert_wo_a_close(out32, _wo_a_ref(o32, wo_a))

    static_after_m32 = wo_a_kernels._tilelang_wo_a_small_static_gemm_kernel.cache_info()
    dynamic_after_m32 = wo_a_kernels._tilelang_wo_a_small_gemm_kernel.cache_info()
    assert static_after_m32.misses == static_after_m20.misses + 1
    assert dynamic_after_m32.misses == 0


@pytest.mark.parametrize("b_stride_r", [WO_A_D, WO_A_D + 128])
def test_wo_a_tilelang_accepts_strided_b(b_stride_r: int) -> None:
    o, wo_a = _make_wo_a_inputs(20, 8 * WO_A_D, b_stride_r=b_stride_r)

    out = try_wo_a_strided_gemm_musa(o, wo_a)

    assert out is not None
    assert wo_a.stride(1) == b_stride_r
    _assert_wo_a_close(out, _wo_a_ref(o, wo_a))


def test_wo_a_tilelang_rejects_unsupported_middle_m() -> None:
    o, wo_a = _make_wo_a_inputs(128, WO_A_D)

    out = try_wo_a_strided_gemm_musa(o, wo_a)

    assert out is None


def test_wo_a_tilelang_keeps_large_compact_on_fallback() -> None:
    o, wo_a = _make_wo_a_inputs(2048, WO_A_D)

    out = try_wo_a_strided_gemm_musa(o, wo_a)

    assert out is None


@pytest.mark.skipif(
    os.environ.get("SGLANG_DSV4_MUSA_RUN_WO_A_PERF_TESTS") != "1",
    reason="set SGLANG_DSV4_MUSA_RUN_WO_A_PERF_TESTS=1 to run WO_A perf guard",
)
@pytest.mark.parametrize("num_tokens,a_stride_t", [(20, WO_A_D), (20, 8 * WO_A_D)])
def test_wo_a_tilelang_perf_guard(num_tokens: int, a_stride_t: int) -> None:
    o, wo_a = _make_wo_a_inputs(num_tokens, a_stride_t)

    out = try_wo_a_strided_gemm_musa(o, wo_a)
    assert out is not None
    torch.musa.current_stream().synchronize()

    def tilelang_call() -> None:
        try_wo_a_strided_gemm_musa(o, wo_a)

    def matmul_call() -> None:
        _wo_a_ref(o, wo_a)

    def einsum_call() -> None:
        torch.einsum("tgd,grd->tgr", o, wo_a)

    tile_us = triton.testing.do_bench(tilelang_call, warmup=20, rep=100)
    matmul_us = triton.testing.do_bench(matmul_call, warmup=20, rep=100)
    einsum_us = triton.testing.do_bench(einsum_call, warmup=20, rep=100)

    # Keep this as a regression guard, not a fragile tuning assertion.
    # Dynamic-M compact BM32 can trade peak latency for fewer compile variants;
    # require it to beat the original einsum path.  For real strided production
    # input, also require it to stay competitive with the torch.mm fallback.
    assert tile_us <= einsum_us * 0.85, (
        f"TileLang WO_A no longer beats einsum: tile={tile_us:.3f}us "
        f"einsum={einsum_us:.3f}us M={num_tokens} stride={a_stride_t}"
    )
    if a_stride_t > WO_A_D:
        assert tile_us <= matmul_us * 1.15, (
            f"TileLang WO_A strided path too slow: tile={tile_us:.3f}us "
            f"matmul={matmul_us:.3f}us einsum={einsum_us:.3f}us "
            f"M={num_tokens} stride={a_stride_t}"
        )
