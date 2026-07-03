from __future__ import annotations

import types

import pytest
import torch
import triton

from sglang.jit_kernel.deepseek_v4 import silu_and_mul_masked_post_quant
from sglang.test.ci.ci_register import register_musa_ci
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_runner
from sglang.srt.layers.quantization import fp8_kernel

from ..utils import (
    MUSA_OPS,
    assert_sm90_aligned_scale_contract,
    get_musa_device,
    reference_grouped_fp8_quant,
    reference_swiglu,
)

register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="DeepSeek V4 MUSA operator test is opt-in outside smoke CI",
)

silu_and_mul_contig_post_quant_musa = MUSA_OPS.silu_and_mul_contig_post_quant_musa
silu_and_mul_masked_post_quant_musa = MUSA_OPS.silu_and_mul_masked_post_quant_musa


def test_prefill_musa_fp8_quant_helper_batches_groups_per_cta(monkeypatch) -> None:
    calls = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(fp8_kernel, "_is_musa", True)
    monkeypatch.setattr(
        fp8_kernel, "_per_token_group_quant_8bit_multi_group", FakeKernel()
    )

    x = torch.empty((512, 4096), dtype=torch.bfloat16)
    x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_s = torch.empty((512, 32), dtype=torch.float32)

    assert fp8_kernel._try_musa_prefill_per_token_group_quant_8bit(
        x=x,
        x_q=x_q,
        x_s=x_s,
        group_size=128,
        eps=1e-10,
        bit8_min=-448.0,
        bit8_max=448.0,
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )

    total_groups = x.numel() // 128
    groups_per_cta, num_warps = fp8_kernel._musa_prefill_fp8_quant_launch_config(
        x.shape[0], x.shape[-1]
    )
    assert calls[0][0] == (triton.cdiv(total_groups, groups_per_cta),)
    assert calls[0][2]["GROUPS_PER_CTA"] == groups_per_cta
    assert calls[0][2]["num_warps"] == num_warps


def test_prefill_musa_fp8_quant_helper_keeps_decode_on_fallback(monkeypatch) -> None:
    monkeypatch.setattr(fp8_kernel, "_is_musa", True)
    x = torch.empty((16, 4096), dtype=torch.bfloat16)
    x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_s = torch.empty((16, 32), dtype=torch.float32)

    assert not fp8_kernel._try_musa_prefill_per_token_group_quant_8bit(
        x=x,
        x_q=x_q,
        x_s=x_s,
        group_size=128,
        eps=1e-10,
        bit8_min=-448.0,
        bit8_max=448.0,
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )


def test_prefill_musa_fp8_quant_helper_rejects_fused_swiglu(monkeypatch) -> None:
    monkeypatch.setattr(fp8_kernel, "_is_musa", True)
    x = torch.empty((8192, 4096), dtype=torch.bfloat16)
    x_q = torch.empty((8192, 2048), dtype=torch.float8_e4m3fn)
    x_s = torch.empty((8192, 16), dtype=torch.float32)

    assert not fp8_kernel._try_musa_prefill_per_token_group_quant_8bit(
        x=x,
        x_q=x_q,
        x_s=x_s,
        group_size=128,
        eps=1e-10,
        bit8_min=-448.0,
        bit8_max=448.0,
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
        fuse_silu_and_mul=True,
    )


def test_tilekernels_swiglu_quant_debug_prints_miss_reason(monkeypatch, capsys) -> None:
    monkeypatch.setattr(deep_gemm_runner, "_is_musa", True)
    monkeypatch.setenv("SGLANG_DEBUG_TILEKERNELS_SWIGLU_QUANT", "1")
    gateup = torch.empty((16, 4096), dtype=torch.bfloat16)

    with envs.SGLANG_OPT_USE_TILEKERNELS_SWIGLU_QUANT.override(True):
        assert (
            deep_gemm_runner._try_tilekernels_swiglu_quant_musa(gateup, 128, None)
            is None
        )

    assert "rows=16 below_prefill_threshold=1024" in capsys.readouterr().out


def _require_tilekernels_swiglu() -> torch.device:
    device = get_musa_device()
    input = torch.zeros((1, 256), device=device, dtype=torch.bfloat16)
    output = torch.empty((1, 128), device=device, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((1, 1), device=device, dtype=torch.float32)
    if not MUSA_OPS._try_tile_swiglu_per_token_cast_musa(input, output, output_scale, 128, None):
        pytest.skip("tile_kernels.quant swiglu runtime path unavailable")
    return device


def _require_tilekernels_fp8_quant() -> torch.device:
    device = get_musa_device()
    x = torch.zeros((512, 128), device=device, dtype=torch.bfloat16)
    if (
        fp8_kernel._try_tilekernels_per_token_cast_musa(
            x,
            128,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
            fuse_silu_and_mul=False,
            masked_m=None,
        )
        is None
    ):
        pytest.skip("tile_kernels.quant per_token_cast runtime path unavailable")
    return device


def _run_jit_masked_swiglu_quant_compile_case(
    name: str,
    device: torch.device,
    num_experts: int,
    num_tokens: int,
    hidden: int,
    masked_values: list[int],
    swiglu_limit: float | None,
) -> None:
    torch.manual_seed(num_experts + num_tokens + hidden)
    input = (
        torch.randn(
            (num_experts, num_tokens, hidden * 2),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.5
    ).contiguous()
    masked_m = torch.tensor(masked_values, device=device, dtype=torch.int32)

    def call_kernel() -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty(
            (num_experts, num_tokens, hidden),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        output_scale = torch.empty(
            (num_experts, num_tokens, hidden // 128),
            device=device,
            dtype=torch.float32,
        )
        output.view(torch.uint8).zero_()
        output_scale.zero_()
        silu_and_mul_masked_post_quant(
            input,
            output,
            output_scale,
            128,
            masked_m,
            scale_ue8m0=False,
            topk=8,
            transposed=False,
            swiglu_limit=swiglu_limit,
            swizzle=False,
        )
        torch.musa.synchronize()
        return output, output_scale

    expected, expected_scale = call_kernel()
    actual, actual_scale = call_kernel()
    torch.testing.assert_close(
        actual.view(torch.uint8).cpu(),
        expected.view(torch.uint8).cpu(),
        rtol=0,
        atol=0,
        msg=f"{name} output is not repeat deterministic",
    )
    torch.testing.assert_close(
        actual_scale.cpu(),
        expected_scale.cpu(),
        rtol=0,
        atol=0,
        msg=f"{name} scale is not repeat deterministic",
    )
    for expert, valid_rows in enumerate(masked_values):
        if valid_rows < num_tokens:
            torch.testing.assert_close(
                expected[expert, valid_rows:].view(torch.uint8).cpu(),
                torch.zeros(
                    (num_tokens - valid_rows, hidden),
                    dtype=torch.uint8,
                ),
                rtol=0,
                atol=0,
                msg=f"{name} invalid output rows changed",
            )
            torch.testing.assert_close(
                expected_scale[expert, valid_rows:].cpu(),
                torch.zeros((num_tokens - valid_rows, hidden // 128)),
                rtol=0,
                atol=0,
                msg=f"{name} invalid scale rows changed",
            )


def test_original_jit_masked_swiglu_quant_compiles_on_musa(monkeypatch) -> None:
    device = get_musa_device()
    monkeypatch.setenv("SGLANG_DSV4_MUSA_SWIGLU_QUANT_OPT_IN", "0")

    _run_jit_masked_swiglu_quant_compile_case(
        "small_no_limit",
        device,
        num_experts=4,
        num_tokens=8,
        hidden=256,
        masked_values=[0, 1, 5, 8],
        swiglu_limit=None,
    )
    _run_jit_masked_swiglu_quant_compile_case(
        "small_limit",
        device,
        num_experts=4,
        num_tokens=8,
        hidden=256,
        masked_values=[3, 0, 8, 2],
        swiglu_limit=10.0,
    )
    _run_jit_masked_swiglu_quant_compile_case(
        "ep_decode_like",
        device,
        num_experts=256,
        num_tokens=16,
        hidden=7168,
        masked_values=[0, 1, 2, 3, 4, 5, 6, 7] * 32,
        swiglu_limit=10.0,
    )


def test_tilekernels_per_token_fp8_quant_respects_row_threshold() -> None:
    device = _require_tilekernels_fp8_quant()
    small_x = torch.randn((16, 4096), device=device, dtype=torch.bfloat16)
    large_x = torch.randn((512, 4096), device=device, dtype=torch.bfloat16)

    assert (
        fp8_kernel._try_tilekernels_per_token_cast_musa(
            small_x,
            128,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
            fuse_silu_and_mul=False,
            masked_m=None,
        )
        is None
    )

    assert (
        fp8_kernel._try_tilekernels_per_token_cast_musa(
            large_x,
            128,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
            fuse_silu_and_mul=False,
            masked_m=None,
        )
        is not None
    )


def test_tilekernels_swiglu_quant_respects_row_threshold() -> None:
    device = _require_tilekernels_swiglu()
    small_gateup = torch.randn((16, 4096), device=device, dtype=torch.bfloat16)
    large_gateup = torch.randn((1024, 4096), device=device, dtype=torch.bfloat16)

    assert (
        deep_gemm_runner._try_tilekernels_swiglu_quant_musa(
            small_gateup,
            128,
            None,
        )
        is None
    )
    assert (
        deep_gemm_runner._try_tilekernels_swiglu_quant_musa(
            large_gateup,
            128,
            None,
        )
        is not None
    )


def _assert_fp8_real_shape_mostly_exact(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    min_exact_rate: float = 0.99999,
) -> None:
    output_u8 = output.view(torch.uint8).cpu()
    reference_u8 = reference.view(torch.uint8).cpu()
    exact_rate = (output_u8 == reference_u8).float().mean().item()
    assert exact_rate >= min_exact_rate, (
        f"FP8 exact rate {exact_rate:.8f} is below {min_exact_rate:.8f}"
    )

def _assert_repeat_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch_count = int((actual != expected).sum().item())
    if actual.dtype.is_floating_point:
        max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
        detail = f", max_abs_diff={max_abs_diff:.6g}"
    else:
        detail = ""
    pytest.fail(
        f"{name} is not repeat-deterministic: "
        f"mismatch_count={mismatch_count}/{actual.numel()}{detail}"
    )


@pytest.mark.parametrize(
    ("rows", "hidden"),
    [
        (1, 4096),      # decode b1 HC-head quant
        (16, 4096),     # decode b16 HC-head quant
        (2048, 4096),   # b16 i128 prefill HC-head quant
        (8192, 4096),   # b1 i8192 prefill HC-head quant
        (16, 2048),     # small-M MoE down-input quant
        (2048, 2048),   # prefill MoE down-input quant
    ],
)
def test_tilekernels_per_token_fp8_quant_matches_sglang_fallback_real_shapes(
    rows: int,
    hidden: int,
) -> None:
    device = _require_tilekernels_fp8_quant()
    torch.manual_seed(rows + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16)

    with envs.SGLANG_OPT_USE_TILEKERNELS_FP8_QUANT.override(False):
        ref_q, ref_s = fp8_kernel.sglang_per_token_group_quant_fp8(x, 128)
    with envs.SGLANG_OPT_USE_TILEKERNELS_FP8_QUANT.override(True):
        out_q, out_s = fp8_kernel.sglang_per_token_group_quant_fp8(x, 128)

    assert tuple(out_q.shape) == (rows, hidden)
    assert tuple(out_s.shape) == (rows, hidden // 128)
    torch.testing.assert_close(out_s.cpu(), ref_s.cpu(), rtol=0, atol=0)
    if rows * (hidden // 128) >= fp8_kernel._MUSA_PREFILL_FP8_QUANT_MIN_GROUPS:
        _assert_fp8_real_shape_mostly_exact(out_q, ref_q, min_exact_rate=0.998)
    else:
        torch.testing.assert_close(
            out_q.view(torch.uint8).cpu(),
            ref_q.view(torch.uint8).cpu(),
            rtol=0,
            atol=0,
        )

def test_tilekernels_per_token_fp8_quant_is_repeat_deterministic() -> None:
    device = _require_tilekernels_fp8_quant()
    rows = 512
    hidden = 128
    torch.manual_seed(20263225)
    x = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16)

    with envs.SGLANG_OPT_USE_TILEKERNELS_FP8_QUANT.override(True):
        expected_q, expected_s = fp8_kernel.sglang_per_token_group_quant_fp8(x, 128)
        torch.musa.synchronize()
        for repeat_idx in range(20):
            actual_q, actual_s = fp8_kernel.sglang_per_token_group_quant_fp8(x, 128)
            torch.musa.synchronize()
            _assert_repeat_exact(
                f"per_token_fp8_quant output repeat={repeat_idx}",
                actual_q.view(torch.uint8),
                expected_q.view(torch.uint8),
            )
            _assert_repeat_exact(
                f"per_token_fp8_quant scale repeat={repeat_idx}",
                actual_s,
                expected_s,
            )

def test_tilekernels_per_token_fp8_quant_probe_row_is_batch_shape_invariant() -> None:
    device = _require_tilekernels_fp8_quant()
    hidden = 128
    torch.manual_seed(20263325)
    probe = torch.randn((hidden,), device=device, dtype=torch.bfloat16)

    baseline_x = torch.randn((512, hidden), device=device, dtype=torch.bfloat16)
    baseline_x[0].copy_(probe)
    with envs.SGLANG_OPT_USE_TILEKERNELS_FP8_QUANT.override(True):
        baseline_q, baseline_s = fp8_kernel.sglang_per_token_group_quant_fp8(baseline_x, 128)
    torch.musa.synchronize()
    expected_q = baseline_q[0].detach().clone()
    expected_s = baseline_s[0].detach().clone()

    x = torch.randn((1024, hidden), device=device, dtype=torch.bfloat16)
    probe_positions = [0, 17, 513, 1023]
    for pos in probe_positions:
        x[pos].copy_(probe)
    with envs.SGLANG_OPT_USE_TILEKERNELS_FP8_QUANT.override(True):
        actual_q, actual_s = fp8_kernel.sglang_per_token_group_quant_fp8(x, 128)
    torch.musa.synchronize()
    for pos in probe_positions:
        _assert_repeat_exact(
            f"per_token_fp8_quant output pos={pos}",
            actual_q[pos].view(torch.uint8),
            expected_q.view(torch.uint8),
        )
        _assert_repeat_exact(
            f"per_token_fp8_quant scale pos={pos}",
            actual_s[pos],
            expected_s,
        )


@pytest.mark.parametrize(
    ("rows", "hidden2"),
    [
        (1, 4096),      # decode single-token MoE SwiGLU+quant
        (16, 4096),     # decode b16 MoE SwiGLU+quant
        (2048, 4096),   # b16 i128 prefill MoE SwiGLU+quant
        (8192, 4096),   # b1 i8192 prefill MoE SwiGLU+quant
    ],
)
def test_silu_and_mul_contig_post_quant_musa_matches_reference_real_shapes(
    rows: int,
    hidden2: int,
) -> None:
    device = _require_tilekernels_swiglu()
    quant_group_size = 128
    torch.manual_seed(rows + hidden2)
    input = torch.randn((rows, hidden2), device=device, dtype=torch.bfloat16)
    output = torch.empty((rows, hidden2 // 2), device=device, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty(
        (rows, hidden2 // 2 // quant_group_size),
        device=device,
        dtype=torch.float32,
    )

    silu_and_mul_contig_post_quant_musa(
        input,
        output,
        output_scale,
        quant_group_size,
        scale_ue8m0=False,
        transposed=False,
    )

    ref_value = reference_swiglu(input)
    ref_quantized, ref_scale = reference_grouped_fp8_quant(
        ref_value,
        quant_group_size,
    )
    torch.testing.assert_close(output_scale.cpu(), ref_scale.cpu(), rtol=1e-3, atol=1e-6)
    _assert_fp8_real_shape_mostly_exact(output, ref_quantized)

def test_silu_and_mul_contig_post_quant_musa_is_repeat_deterministic() -> None:
    device = _require_tilekernels_swiglu()
    rows = 17
    hidden2 = 256
    quant_group_size = 128
    torch.manual_seed(20263425)
    input = torch.randn((rows, hidden2), device=device, dtype=torch.bfloat16)

    expected = torch.empty((rows, hidden2 // 2), device=device, dtype=torch.float8_e4m3fn)
    expected_scale = torch.empty((rows, 1), device=device, dtype=torch.float32)
    silu_and_mul_contig_post_quant_musa(
        input,
        expected,
        expected_scale,
        quant_group_size,
        scale_ue8m0=False,
        transposed=False,
    )
    torch.musa.synchronize()
    for repeat_idx in range(20):
        actual = torch.empty_like(expected)
        actual_scale = torch.empty_like(expected_scale)
        silu_and_mul_contig_post_quant_musa(
            input,
            actual,
            actual_scale,
            quant_group_size,
            scale_ue8m0=False,
            transposed=False,
        )
        torch.musa.synchronize()
        _assert_repeat_exact(
            f"swiglu_contig_quant output repeat={repeat_idx}",
            actual.view(torch.uint8),
            expected.view(torch.uint8),
        )
        _assert_repeat_exact(
            f"swiglu_contig_quant scale repeat={repeat_idx}",
            actual_scale,
            expected_scale,
        )

def test_silu_and_mul_contig_post_quant_musa_probe_row_is_batch_shape_invariant() -> None:
    device = _require_tilekernels_swiglu()
    hidden2 = 256
    quant_group_size = 128
    torch.manual_seed(20263525)
    probe = torch.randn((hidden2,), device=device, dtype=torch.bfloat16)

    baseline_input = probe.view(1, hidden2).clone()
    baseline = torch.empty((1, hidden2 // 2), device=device, dtype=torch.float8_e4m3fn)
    baseline_scale = torch.empty((1, 1), device=device, dtype=torch.float32)
    silu_and_mul_contig_post_quant_musa(
        baseline_input,
        baseline,
        baseline_scale,
        quant_group_size,
        scale_ue8m0=False,
        transposed=False,
    )
    torch.musa.synchronize()
    expected = baseline[0].detach().clone()
    expected_scale = baseline_scale[0].detach().clone()

    input = torch.randn((65, hidden2), device=device, dtype=torch.bfloat16)
    probe_positions = [0, 17, 64]
    for pos in probe_positions:
        input[pos].copy_(probe)
    actual = torch.empty((65, hidden2 // 2), device=device, dtype=torch.float8_e4m3fn)
    actual_scale = torch.empty((65, 1), device=device, dtype=torch.float32)
    silu_and_mul_contig_post_quant_musa(
        input,
        actual,
        actual_scale,
        quant_group_size,
        scale_ue8m0=False,
        transposed=False,
    )
    torch.musa.synchronize()
    for pos in probe_positions:
        _assert_repeat_exact(
            f"swiglu_contig_quant output pos={pos}",
            actual[pos].view(torch.uint8),
            expected.view(torch.uint8),
        )
        _assert_repeat_exact(
            f"swiglu_contig_quant scale pos={pos}",
            actual_scale[pos],
            expected_scale,
        )


def test_silu_and_mul_contig_post_quant_musa_matches_reference() -> None:
    device = _require_tilekernels_swiglu()
    quant_group_size = 128
    input = torch.linspace(-3.0, 3.0, steps=4 * 256, device=device, dtype=torch.float32).reshape(4, 256).to(torch.bfloat16)
    output = torch.empty((4, 128), device=device, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((4, 1), device=device, dtype=torch.float32)

    silu_and_mul_contig_post_quant_musa(
        input,
        output,
        output_scale,
        quant_group_size,
        scale_ue8m0=False,
        transposed=False,
    )

    ref_value = reference_swiglu(input)
    ref_quantized, ref_scale = reference_grouped_fp8_quant(ref_value, quant_group_size)

    assert tuple(output.shape) == (4, 128)
    assert output.dtype == torch.float8_e4m3fn
    assert_sm90_aligned_scale_contract(output_scale, (4, 1))
    torch.testing.assert_close(output_scale.cpu(), ref_scale.cpu(), rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(output.float().cpu(), ref_quantized.float().cpu(), rtol=0, atol=0)


def test_silu_and_mul_contig_post_quant_musa_respects_swiglu_limit() -> None:
    device = _require_tilekernels_swiglu()
    quant_group_size = 128
    input = torch.linspace(-8.0, 8.0, steps=2 * 256, device=device, dtype=torch.float32).reshape(2, 256).to(torch.bfloat16)
    output = torch.empty((2, 128), device=device, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((2, 1), device=device, dtype=torch.float32)

    silu_and_mul_contig_post_quant_musa(
        input,
        output,
        output_scale,
        quant_group_size,
        scale_ue8m0=False,
        transposed=False,
        swiglu_limit=1.5,
    )

    ref_value = reference_swiglu(input, swiglu_limit=1.5)
    ref_quantized, ref_scale = reference_grouped_fp8_quant(ref_value, quant_group_size)
    torch.testing.assert_close(output_scale.cpu(), ref_scale.cpu(), rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(output.float().cpu(), ref_quantized.float().cpu(), rtol=0, atol=0)


def test_silu_and_mul_masked_post_quant_musa_scalar_mask_matches_contiguous_prefix() -> None:
    device = _require_tilekernels_swiglu()
    quant_group_size = 128
    input = torch.linspace(-2.0, 2.0, steps=4 * 256, device=device, dtype=torch.float32).reshape(4, 256).to(torch.bfloat16)
    output = torch.empty((4, 128), device=device, dtype=torch.float8_e4m3fn)
    output.view(torch.uint8).fill_(0xB8)
    output_scale = torch.full((4, 1), -1.0, device=device, dtype=torch.float32)

    silu_and_mul_masked_post_quant_musa(
        input,
        output,
        output_scale,
        quant_group_size,
        torch.tensor([2], device=device, dtype=torch.int32),
        scale_ue8m0=False,
        transposed=False,
    )

    ref_value = reference_swiglu(input[:2])
    ref_quantized, ref_scale = reference_grouped_fp8_quant(ref_value, quant_group_size)
    torch.testing.assert_close(output[:2].float().cpu(), ref_quantized.float().cpu(), rtol=0, atol=0)
    torch.testing.assert_close(output_scale[:2].cpu(), ref_scale.cpu(), rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(output[2:].view(torch.uint8).cpu(), torch.full((2, 128), 0xB8, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(output_scale[2:].cpu(), torch.full((2, 1), -1.0), rtol=0, atol=0)


def test_silu_and_mul_masked_post_quant_musa_only_updates_valid_expert_rows() -> None:
    device = _require_tilekernels_swiglu()
    quant_group_size = 128
    input = torch.linspace(-2.0, 2.0, steps=3 * 4 * 256, device=device, dtype=torch.float32).reshape(3, 4, 256).to(torch.bfloat16)
    output = torch.empty((3, 4, 128), device=device, dtype=torch.float8_e4m3fn)
    output.view(torch.uint8).fill_(0xB8)
    output_scale = torch.full((3, 4, 1), -1.0, device=device, dtype=torch.float32)
    masked_m = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)

    silu_and_mul_masked_post_quant_musa(
        input,
        output,
        output_scale,
        quant_group_size,
        masked_m,
        scale_ue8m0=False,
        transposed=False,
    )

    assert_sm90_aligned_scale_contract(output_scale, (3, 4, 1))
    for expert, valid_rows in enumerate(masked_m.cpu().tolist()):
        if valid_rows == 0:
            torch.testing.assert_close(output[expert].float().cpu(), torch.full((4, 128), -1.0), rtol=0, atol=0)
            torch.testing.assert_close(output_scale[expert].cpu(), torch.full((4, 1), -1.0), rtol=0, atol=0)
            continue

        ref_value = reference_swiglu(input[expert, :valid_rows])
        ref_quantized, ref_scale = reference_grouped_fp8_quant(ref_value, quant_group_size)
        torch.testing.assert_close(output[expert, :valid_rows].float().cpu(), ref_quantized.float().cpu(), rtol=0, atol=0)
        torch.testing.assert_close(output_scale[expert, :valid_rows].cpu(), ref_scale.cpu(), rtol=1e-3, atol=1e-6)
        if valid_rows < input.shape[1]:
            torch.testing.assert_close(
                output[expert, valid_rows:].float().cpu(),
                torch.full((input.shape[1] - valid_rows, 128), -1.0),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                output_scale[expert, valid_rows:].cpu(),
                torch.full((input.shape[1] - valid_rows, 1), -1.0),
                rtol=0,
                atol=0,
            )

def test_silu_and_mul_masked_post_quant_musa_is_repeat_deterministic() -> None:
    device = _require_tilekernels_swiglu()
    quant_group_size = 128
    torch.manual_seed(20263625)
    input = torch.randn((3, 5, 256), device=device, dtype=torch.bfloat16)
    masked_m = torch.tensor([1, 0, 5], device=device, dtype=torch.int32)

    expected = torch.empty((3, 5, 128), device=device, dtype=torch.float8_e4m3fn)
    expected.view(torch.uint8).fill_(0xB8)
    expected_scale = torch.full((3, 5, 1), -1.0, device=device, dtype=torch.float32)
    silu_and_mul_masked_post_quant_musa(
        input,
        expected,
        expected_scale,
        quant_group_size,
        masked_m,
        scale_ue8m0=False,
        transposed=False,
    )
    torch.musa.synchronize()
    for repeat_idx in range(20):
        actual = torch.empty_like(expected)
        actual.view(torch.uint8).fill_(0xB8)
        actual_scale = torch.full_like(expected_scale, -1.0)
        silu_and_mul_masked_post_quant_musa(
            input,
            actual,
            actual_scale,
            quant_group_size,
            masked_m,
            scale_ue8m0=False,
            transposed=False,
        )
        torch.musa.synchronize()
        _assert_repeat_exact(
            f"swiglu_masked_quant output repeat={repeat_idx}",
            actual.view(torch.uint8),
            expected.view(torch.uint8),
        )
        _assert_repeat_exact(
            f"swiglu_masked_quant scale repeat={repeat_idx}",
            actual_scale,
            expected_scale,
        )


def test_silu_and_mul_contig_post_quant_musa_uses_sm90_float32_scale_contract() -> None:
    device = _require_tilekernels_swiglu()
    input = torch.randn((2, 256), device=device, dtype=torch.bfloat16)
    output = torch.empty((2, 128), device=device, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((2, 1), device=device, dtype=torch.float32)

    silu_and_mul_contig_post_quant_musa(
        input,
        output,
        output_scale,
        128,
        scale_ue8m0=False,
        transposed=False,
    )


def test_silu_and_mul_contig_post_quant_musa_invokes_tilekernels_quant_path(monkeypatch) -> None:
    input = torch.randn((2, 256), dtype=torch.bfloat16)
    output = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((2, 1), dtype=torch.float32)
    calls = []

    def fake_import_module(name: str):
        if name != "tile_kernels.quant":
            raise ImportError(name)

        def fake_swiglu_forward_and_per_token_cast(**kwargs):
            calls.append(kwargs)
            assert kwargs["fmt"] == "e4m3"
            assert kwargs["num_per_channels"] == 128
            assert kwargs["pos_to_expert"] is None
            assert kwargs["use_tma_aligned_col_major_sf"] is False
            assert kwargs["round_sf"] is False
            assert kwargs["use_packed_ue8m0"] is False
            return torch.zeros_like(output), torch.full_like(output_scale, 0.25)

        return types.SimpleNamespace(swiglu_forward_and_per_token_cast=fake_swiglu_forward_and_per_token_cast)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback path was used")

    monkeypatch.setattr(MUSA_OPS.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(MUSA_OPS, "_tile_swiglu_forward", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_quantize_fp8_grouped", fail_fallback)

    silu_and_mul_contig_post_quant_musa(
        input,
        output,
        output_scale,
        128,
        scale_ue8m0=False,
        transposed=False,
    )

    assert len(calls) == 1
    torch.testing.assert_close(output.float(), torch.zeros((2, 128)), rtol=0, atol=0)
    torch.testing.assert_close(output_scale, torch.full((2, 1), 0.25), rtol=0, atol=0)


def test_silu_and_mul_contig_post_quant_musa_tilekernels_failure_fails_closed(monkeypatch) -> None:
    device = get_musa_device()
    quant_group_size = 128
    input = torch.linspace(-3.0, 3.0, steps=4 * 256, device=device, dtype=torch.float32).reshape(4, 256).to(torch.bfloat16)
    output = torch.empty((4, 128), device=device, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((4, 1), device=device, dtype=torch.float32)

    def fake_import_module(name: str):
        raise ImportError(name)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback path was used")

    monkeypatch.setattr(MUSA_OPS.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(MUSA_OPS, "_tile_swiglu_forward", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_quantize_fp8_grouped", fail_fallback)

    try:
        silu_and_mul_contig_post_quant_musa(
            input,
            output,
            output_scale,
            quant_group_size,
            scale_ue8m0=False,
            transposed=False,
        )
    except NotImplementedError as exc:
        assert "tile_kernels.quant.swiglu_forward_and_per_token_cast" in str(exc)
    else:
        raise AssertionError("expected fail-closed NotImplementedError")



def test_silu_and_mul_masked_post_quant_musa_invokes_tilekernels_quant_path(monkeypatch) -> None:
    input = torch.randn((3, 4, 256), dtype=torch.bfloat16)
    output = torch.empty((3, 4, 128), dtype=torch.float8_e4m3fn)
    output.view(torch.uint8).fill_(0xB8)
    output_scale = torch.full((3, 4, 1), -1.0, dtype=torch.float32)
    masked_m = torch.tensor([0, 2, 4], dtype=torch.int32)
    calls = []

    def fake_import_module(name: str):
        if name != "tile_kernels.quant":
            raise ImportError(name)

        def fake_swiglu_forward_and_per_token_cast(**kwargs):
            calls.append(kwargs)
            pos_to_expert = kwargs["pos_to_expert"]
            assert pos_to_expert is not None
            torch.testing.assert_close(pos_to_expert.cpu(), torch.tensor([1, 1, 2, 2, 2, 2], dtype=torch.int32))
            assert kwargs["fmt"] == "e4m3"
            assert kwargs["num_per_channels"] == 128
            assert kwargs["use_tma_aligned_col_major_sf"] is False
            assert kwargs["round_sf"] is False
            assert kwargs["use_packed_ue8m0"] is False
            quantized = torch.zeros((6, 128), dtype=torch.float8_e4m3fn)
            scale = torch.arange(6, dtype=torch.float32).reshape(6, 1) + 0.5
            return quantized, scale

        return types.SimpleNamespace(swiglu_forward_and_per_token_cast=fake_swiglu_forward_and_per_token_cast)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback path was used")

    monkeypatch.setattr(MUSA_OPS.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(MUSA_OPS, "_tile_swiglu_forward", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_quantize_fp8_grouped", fail_fallback)

    silu_and_mul_masked_post_quant_musa(
        input,
        output,
        output_scale,
        128,
        masked_m,
        scale_ue8m0=False,
        transposed=False,
    )

    assert len(calls) == 1
    torch.testing.assert_close(output[0].view(torch.uint8).cpu(), torch.full((4, 128), 0xB8, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(output_scale[1, :2].cpu(), torch.tensor([[0.5], [1.5]], dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(output_scale[2, :4].cpu(), torch.tensor([[2.5], [3.5], [4.5], [5.5]], dtype=torch.float32), rtol=0, atol=0)


def test_silu_and_mul_masked_post_quant_musa_scalar_mask_invokes_tilekernels_quant_path(monkeypatch) -> None:
    input = torch.randn((4, 256), dtype=torch.bfloat16)
    output = torch.empty((4, 128), dtype=torch.float8_e4m3fn)
    output.view(torch.uint8).fill_(0xB8)
    output_scale = torch.full((4, 1), -1.0, dtype=torch.float32)
    masked_m = torch.tensor([2], dtype=torch.int32)
    calls = []

    def fake_import_module(name: str):
        if name != "tile_kernels.quant":
            raise ImportError(name)

        def fake_swiglu_forward_and_per_token_cast(**kwargs):
            calls.append(kwargs)
            torch.testing.assert_close(kwargs["x"], input[:2].contiguous())
            torch.testing.assert_close(kwargs["pos_to_expert"].cpu(), torch.zeros(2, dtype=torch.int32))
            assert kwargs["fmt"] == "e4m3"
            assert kwargs["num_per_channels"] == 128
            assert kwargs["use_tma_aligned_col_major_sf"] is False
            assert kwargs["round_sf"] is False
            assert kwargs["use_packed_ue8m0"] is False
            return torch.zeros((2, 128), dtype=torch.float8_e4m3fn), torch.tensor([[0.5], [1.5]], dtype=torch.float32)

        return types.SimpleNamespace(swiglu_forward_and_per_token_cast=fake_swiglu_forward_and_per_token_cast)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback path was used")

    monkeypatch.setattr(MUSA_OPS.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(MUSA_OPS, "_tile_swiglu_forward", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_quantize_fp8_grouped", fail_fallback)

    silu_and_mul_masked_post_quant_musa(
        input,
        output,
        output_scale,
        128,
        masked_m,
        scale_ue8m0=False,
        transposed=False,
    )

    assert len(calls) == 1
    torch.testing.assert_close(output[:2].float(), torch.zeros((2, 128)), rtol=0, atol=0)
    torch.testing.assert_close(output[2:].view(torch.uint8), torch.full((2, 128), 0xB8, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(output_scale, torch.tensor([[0.5], [1.5], [-1.0], [-1.0]], dtype=torch.float32), rtol=0, atol=0)


def test_silu_and_mul_masked_post_quant_musa_tilekernels_uses_tensorized_valid_row_order(monkeypatch) -> None:
    input = torch.arange(3 * 4 * 256, dtype=torch.float32).reshape(3, 4, 256).to(torch.bfloat16)
    output = torch.empty((3, 4, 128), dtype=torch.float8_e4m3fn)
    output.view(torch.uint8).fill_(0xB8)
    output_scale = torch.full((3, 4, 1), -1.0, dtype=torch.float32)
    masked_m = torch.tensor([1, 3, 2], dtype=torch.int32)
    calls = []

    def fake_import_module(name: str):
        if name != "tile_kernels.quant":
            raise ImportError(name)

        def fake_swiglu_forward_and_per_token_cast(**kwargs):
            calls.append(kwargs)
            packed_input = kwargs["x"]
            expected_input = torch.cat([input[0, :1], input[1, :3], input[2, :2]], dim=0)
            torch.testing.assert_close(packed_input, expected_input)
            torch.testing.assert_close(kwargs["pos_to_expert"].cpu(), torch.tensor([0, 1, 1, 1, 2, 2], dtype=torch.int32))
            quantized = torch.arange(6 * 128, dtype=torch.float32).reshape(6, 128).to(torch.float8_e4m3fn)
            scale = torch.arange(6, dtype=torch.float32).reshape(6, 1) + 10.0
            return quantized, scale

        return types.SimpleNamespace(swiglu_forward_and_per_token_cast=fake_swiglu_forward_and_per_token_cast)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback path was used")

    monkeypatch.setattr(MUSA_OPS.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(MUSA_OPS, "_tile_swiglu_forward", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_quantize_fp8_grouped", fail_fallback)

    silu_and_mul_masked_post_quant_musa(
        input,
        output,
        output_scale,
        128,
        masked_m,
        scale_ue8m0=False,
        transposed=False,
    )

    assert len(calls) == 1
    expected_scale = torch.full((3, 4, 1), -1.0, dtype=torch.float32)
    expected_scale[0, :1] = torch.tensor([[10.0]])
    expected_scale[1, :3] = torch.tensor([[11.0], [12.0], [13.0]])
    expected_scale[2, :2] = torch.tensor([[14.0], [15.0]])
    torch.testing.assert_close(output_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(output[0, 1:].view(torch.uint8), torch.full((3, 128), 0xB8, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(output[1, 3:].view(torch.uint8), torch.full((1, 128), 0xB8, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(output[2, 2:].view(torch.uint8), torch.full((2, 128), 0xB8, dtype=torch.uint8), rtol=0, atol=0)


def test_silu_and_mul_masked_post_quant_musa_rejects_non_sm90_scale_modes_before_fallback(monkeypatch) -> None:
    input = torch.randn((4, 256), dtype=torch.bfloat16)
    output = torch.empty((4, 128), dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((4, 1), dtype=torch.float32)

    class HostileMaskedM:
        def numel(self):
            raise AssertionError("unsupported masked scale modes must fail before reading masked_m")

    def fail_fallback(*args, **kwargs):
        raise AssertionError("unsupported masked scale modes must fail before fallback work")

    monkeypatch.setattr(MUSA_OPS, "_try_tile_swiglu_per_token_cast_musa", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_try_tile_swiglu_expert_post_quant_musa", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_tile_swiglu_forward", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_quantize_fp8_grouped", fail_fallback)

    for kwargs, expected in [
        ({"scale_ue8m0": True}, "UE8M0"),
        ({"transposed": True}, "transposed"),
        ({"swizzle": True}, "swizzled"),
    ]:
        with pytest.raises(NotImplementedError, match=expected):
            silu_and_mul_masked_post_quant_musa(
                input,
                output,
                output_scale,
                128,
                HostileMaskedM(),
                **kwargs,
            )



def test_silu_and_mul_masked_post_quant_musa_scalar_mask_tilekernels_failure_fails_closed(monkeypatch) -> None:
    device = get_musa_device()
    input = torch.linspace(-2.0, 2.0, steps=4 * 256, device=device, dtype=torch.float32).reshape(4, 256).to(torch.bfloat16)
    output = torch.empty((4, 128), device=device, dtype=torch.float8_e4m3fn)
    output.view(torch.uint8).fill_(0xB8)
    output_scale = torch.full((4, 1), -1.0, device=device, dtype=torch.float32)
    masked_m = torch.tensor([2], device=device, dtype=torch.int32)

    def fake_import_module(name: str):
        raise ImportError(name)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback path was used")

    monkeypatch.setattr(MUSA_OPS.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(MUSA_OPS, "_tile_swiglu_forward", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_quantize_fp8_grouped", fail_fallback)

    try:
        silu_and_mul_masked_post_quant_musa(
            input,
            output,
            output_scale,
            128,
            masked_m,
            scale_ue8m0=False,
            transposed=False,
        )
    except NotImplementedError as exc:
        assert "tile_kernels.quant.swiglu_forward_and_per_token_cast" in str(exc)
    else:
        raise AssertionError("expected fail-closed NotImplementedError")


def test_silu_and_mul_masked_post_quant_musa_tilekernels_failure_fails_closed(monkeypatch) -> None:
    device = get_musa_device()
    quant_group_size = 128
    input = torch.linspace(-2.0, 2.0, steps=3 * 4 * 256, device=device, dtype=torch.float32).reshape(3, 4, 256).to(torch.bfloat16)
    output = torch.empty((3, 4, 128), device=device, dtype=torch.float8_e4m3fn)
    output.view(torch.uint8).fill_(0xB8)
    output_scale = torch.full((3, 4, 1), -1.0, device=device, dtype=torch.float32)
    masked_m = torch.tensor([1, 0, 3], device=device, dtype=torch.int32)

    def fake_import_module(name: str):
        raise ImportError(name)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback path was used")

    monkeypatch.setattr(MUSA_OPS.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(MUSA_OPS, "_tile_swiglu_forward", fail_fallback)
    monkeypatch.setattr(MUSA_OPS, "_quantize_fp8_grouped", fail_fallback)

    try:
        silu_and_mul_masked_post_quant_musa(
            input,
            output,
            output_scale,
            quant_group_size,
            masked_m,
            scale_ue8m0=False,
            transposed=False,
        )
    except NotImplementedError as exc:
        assert "tile_kernels.quant.swiglu_forward_and_per_token_cast" in str(exc)
    else:
        raise AssertionError("expected fail-closed NotImplementedError")
