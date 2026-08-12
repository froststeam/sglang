from __future__ import annotations

import inspect
import os
import sys

import pytest
import torch

from sglang.srt.utils import is_musa
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(
    est_time=240,
    suite="stage-a-test-1-gpu-musa-smoke",
    nightly=True,
)

pytestmark = pytest.mark.skipif(not is_musa(), reason="MUSA-only grouped top-k test")


# Keep the same model and generic configurations used by the previous MUSA JIT
# kernel so the implementations are validated against identical topologies.
GROUPED_TOPK_CASES = (
    (64, 1, 1, 6),
    (160, 8, 3, 6),
    (256, 8, 4, 8),
    (256, 1, 1, 8),
)

GENERIC_GROUPED_TOPK_CASES = (
    (48, 3, 2, 5),
    (96, 6, 2, 7),
    (128, 4, 3, 9),
    (128, 32, 3, 9),
    (128, 64, 4, 7),
)

PARALLEL_BOUNDARY_CASES = (
    (512, 16, 4, 32),
    (128, 128, 7, 7),
)

# These cases deliberately exceed one of the parallel-kernel limits and must
# still execute through the general TileLang serial fallback.
SERIAL_FALLBACK_CASES = (
    (513, 3, 2, 7),
    (258, 129, 4, 7),
    (96, 3, 2, 40),
    (96, 3, 3, 40),
)

TOKEN_COUNTS_1_TO_3K = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    3071,
    3072,
)


def _make_unique_logits(
    num_tokens: int, num_experts: int, dtype: torch.dtype
) -> torch.Tensor:
    values = torch.linspace(-8.0, 8.0, num_experts, device="musa")
    expert = torch.arange(num_experts, device="musa", dtype=torch.int64)
    row = torch.arange(num_tokens, device="musa", dtype=torch.int64).unsqueeze(1)
    permutation = (expert.unsqueeze(0) * 131 + row * 17) % num_experts
    return values[permutation].to(dtype)


def _grouped_topk_reference(
    gating_output: torch.Tensor,
    *,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float | None = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the production Torch implementation used as the oracle.

    Keep the reference in ``topk.py`` instead of maintaining a second grouped
    top-k expression in this test.  ``hidden_states`` is only used by the
    production function for its token-count consistency check.
    """
    from sglang.srt.layers.moe.topk import grouped_topk_gpu

    hidden_states = torch.empty(
        (gating_output.shape[0], 1),
        device=gating_output.device,
        dtype=gating_output.dtype,
    )
    return grouped_topk_gpu(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def _as_dense(weights: torch.Tensor, ids: torch.Tensor, width: int) -> torch.Tensor:
    dense = torch.zeros((weights.shape[0], width), device=weights.device)
    dense.scatter_(1, ids.long(), weights)
    return dense


def _assert_matches_reference(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    *,
    num_experts: int,
    num_fused_shared_experts: int = 0,
) -> None:
    actual_weights, actual_ids = actual
    expected_weights, expected_ids = expected
    width = num_experts + num_fused_shared_experts
    torch.testing.assert_close(
        _as_dense(actual_weights, actual_ids, width),
        _as_dense(expected_weights, expected_ids, width),
        rtol=2e-5,
        atol=2e-7,
    )


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", GROUPED_TOPK_CASES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("renormalize", (False, True))
def test_grouped_topk_tilelang_matches_reference_from_1_to_3k_tokens(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    dtype: torch.dtype,
    renormalize: bool,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    for num_tokens in TOKEN_COUNTS_1_TO_3K:
        gating_output = _make_unique_logits(num_tokens, num_experts, dtype)
        actual = grouped_topk_softmax_tilelang(
            gating_output, topk, num_groups, topk_group, renormalize
        )
        expected = _grouped_topk_reference(
            gating_output,
            topk=topk,
            num_expert_group=num_groups,
            topk_group=topk_group,
            renormalize=renormalize,
        )
        _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.skipif(
    os.environ.get("SGLANG_RUN_EXHAUSTIVE_GROUPED_TOPK_MUSA") != "1",
    reason="set SGLANG_RUN_EXHAUSTIVE_GROUPED_TOPK_MUSA=1 for every 1--3072 size",
)
def test_grouped_topk_tilelang_every_token_count_from_1_to_3k() -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    for num_experts, num_groups, topk_group, topk in GROUPED_TOPK_CASES:
        for num_tokens in range(1, 3073):
            gating_output = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
            actual = grouped_topk_softmax_tilelang(
                gating_output, topk, num_groups, topk_group, True
            )
            expected = _grouped_topk_reference(
                gating_output,
                topk=topk,
                num_expert_group=num_groups,
                topk_group=topk_group,
                renormalize=True,
            )
            _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,routed_topk", GROUPED_TOPK_CASES
)
@pytest.mark.parametrize("num_tokens", (1, 257, 3072))
@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("apply_routed_scale", (False, True))
def test_grouped_topk_tilelang_shared_expert_and_routed_scale(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    routed_topk: int,
    num_tokens: int,
    renormalize: bool,
    apply_routed_scale: bool,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    scaling_factor = 2.5
    output_topk = routed_topk + 1
    actual = grouped_topk_softmax_tilelang(
        gating_output,
        output_topk,
        num_groups,
        topk_group,
        renormalize,
        num_fused_shared_experts=1,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scale,
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=output_topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=renormalize,
        num_fused_shared_experts=1,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scale,
    )
    _assert_matches_reference(
        actual,
        expected,
        num_experts=num_experts,
        num_fused_shared_experts=1,
    )


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", GROUPED_TOPK_CASES)
@pytest.mark.parametrize("num_tokens", (1, 257, 3072))
def test_grouped_topk_tilelang_routed_scale_without_shared_expert(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    num_tokens: int,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    scaling_factor = 2.5
    actual = grouped_topk_softmax_tilelang(
        gating_output,
        topk,
        num_groups,
        topk_group,
        True,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=True,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )
    _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", GROUPED_TOPK_CASES)
def test_grouped_topk_tilelang_handles_very_negative_logits(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = _make_unique_logits(17, num_experts, torch.float32) - 20000.0
    actual = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, True
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=True,
    )
    _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk", GENERIC_GROUPED_TOPK_CASES
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("num_tokens", (1, 33, 257))
def test_grouped_topk_tilelang_generic_configs_match_reference(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    dtype: torch.dtype,
    renormalize: bool,
    num_tokens: int,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = _make_unique_logits(num_tokens, num_experts, dtype)
    actual = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, renormalize
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=renormalize,
    )
    _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.parametrize("num_tokens", (1, 33, 257))
@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("apply_routed_scale", (False, True))
def test_grouped_topk_tilelang_generic_config_supports_shared_expert(
    num_tokens: int,
    renormalize: bool,
    apply_routed_scale: bool,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    num_experts, num_groups, topk_group, routed_topk = (96, 6, 2, 7)
    output_topk = routed_topk + 1
    scaling_factor = 2.5
    gating_output = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    actual = grouped_topk_softmax_tilelang(
        gating_output,
        output_topk,
        num_groups,
        topk_group,
        renormalize,
        num_fused_shared_experts=1,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scale,
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=output_topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=renormalize,
        num_fused_shared_experts=1,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scale,
    )
    _assert_matches_reference(
        actual,
        expected,
        num_experts=num_experts,
        num_fused_shared_experts=1,
    )


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk", SERIAL_FALLBACK_CASES
)
@pytest.mark.parametrize("num_tokens", (1, 3))
@pytest.mark.parametrize("renormalize", (False, True))
def test_grouped_topk_tilelang_serial_fallback_matches_reference(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    num_tokens: int,
    renormalize: bool,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    actual = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, renormalize
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=renormalize,
    )
    _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.parametrize("num_tokens", (1, 3))
@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("apply_routed_scale", (False, True))
def test_grouped_topk_tilelang_serial_fallback_supports_shared_expert(
    num_tokens: int,
    renormalize: bool,
    apply_routed_scale: bool,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    num_experts, num_groups, topk_group, routed_topk = (96, 3, 2, 40)
    output_topk = routed_topk + 1
    scaling_factor = 2.5
    gating_output = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    actual = grouped_topk_softmax_tilelang(
        gating_output,
        output_topk,
        num_groups,
        topk_group,
        renormalize,
        num_fused_shared_experts=1,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scale,
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=output_topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=renormalize,
        num_fused_shared_experts=1,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scale,
    )
    _assert_matches_reference(
        actual,
        expected,
        num_experts=num_experts,
        num_fused_shared_experts=1,
    )


def test_grouped_topk_tilelang_serial_fallback_supports_noncontiguous_input() -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    num_experts, num_groups, topk_group, topk = (513, 3, 2, 7)
    storage = _make_unique_logits(6, num_experts * 2, torch.bfloat16)
    gating_output = storage[::2, ::2]
    assert gating_output.shape == (3, num_experts)
    assert not gating_output.is_contiguous()
    actual = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, True
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=True,
    )
    _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk", PARALLEL_BOUNDARY_CASES
)
@pytest.mark.parametrize("num_tokens", (1, 3))
@pytest.mark.parametrize("renormalize", (False, True))
def test_grouped_topk_tilelang_parallel_limits_match_reference(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    num_tokens: int,
    renormalize: bool,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    actual = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, renormalize
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=renormalize,
    )
    _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk",
    GROUPED_TOPK_CASES + GENERIC_GROUPED_TOPK_CASES,
)
def test_grouped_topk_tilelang_ties_are_valid_and_deterministic(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = torch.zeros((17, num_experts), device="musa", dtype=torch.bfloat16)
    actual_weights, actual_ids = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, True
    )
    repeated_weights, repeated_ids = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, True
    )
    torch.testing.assert_close(
        actual_weights, torch.full_like(actual_weights, 1 / topk)
    )
    torch.testing.assert_close(repeated_weights, actual_weights)
    torch.testing.assert_close(repeated_ids, actual_ids)

    scores = torch.softmax(gating_output.float(), dim=-1)
    group_scores = scores.view(17, num_groups, -1).amax(dim=-1)
    group_cutoff = torch.topk(group_scores, k=topk_group, dim=-1).values.amin(
        dim=-1, keepdim=True
    )
    actual_groups = actual_ids.long() // (num_experts // num_groups)
    assert (group_scores.gather(1, actual_groups) >= group_cutoff).all()
    for row_ids, row_groups in zip(actual_ids, actual_groups):
        assert torch.unique(row_ids).numel() == topk
        assert torch.unique(row_groups).numel() <= topk_group


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", GROUPED_TOPK_CASES)
def test_grouped_topk_tilelang_supports_noncontiguous_input(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    storage = _make_unique_logits(66, num_experts * 2, torch.bfloat16)
    gating_output = storage[::2, ::2]
    assert not gating_output.is_contiguous()
    actual = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, True
    )
    expected = _grouped_topk_reference(
        gating_output,
        topk=topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        renormalize=True,
    )
    _assert_matches_reference(actual, expected, num_experts=num_experts)


def test_grouped_topk_tilelang_handles_empty_input() -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    weights, ids = grouped_topk_softmax_tilelang(
        torch.empty((0, 48), device="musa", dtype=torch.bfloat16),
        5,
        3,
        2,
        True,
    )
    assert weights.shape == ids.shape == (0, 5)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32


def test_grouped_topk_uses_musa_tilelang_kernel() -> None:
    from sglang.srt.layers.moe.topk import (
        grouped_topk,
        grouped_topk_gpu,
        grouped_topk_tilelang_musa_impl,
    )

    assert grouped_topk is grouped_topk_tilelang_musa_impl
    assert inspect.signature(grouped_topk_tilelang_musa_impl) == inspect.signature(
        grouped_topk_gpu
    )


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", GROUPED_TOPK_CASES)
@pytest.mark.parametrize("num_tokens", (1, 33, 3072))
def test_grouped_topk_tilelang_dispatch_matches_direct_kernel(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    num_tokens: int,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )
    from sglang.srt.layers.moe.topk import grouped_topk

    gating_output = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    hidden_states = torch.empty((num_tokens, 1), device="musa")
    expected = grouped_topk_softmax_tilelang(
        gating_output, topk, num_groups, topk_group, True
    )
    actual = grouped_topk(
        hidden_states, gating_output, topk, True, num_groups, topk_group
    )
    _assert_matches_reference(actual, expected, num_experts=num_experts)


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk",
    (
        (0, 1, 1, 1),
        (48, 0, 1, 1),
        (48, 5, 1, 1),
        (48, 3, 0, 1),
        (48, 3, 4, 1),
        (48, 3, 2, 33),
    ),
)
def test_grouped_topk_tilelang_rejects_invalid_configs(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
) -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = torch.empty((1, num_experts), device="musa", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        grouped_topk_softmax_tilelang(gating_output, topk, num_groups, topk_group, True)


def test_grouped_topk_tilelang_rejects_unsupported_shared_expert_count() -> None:
    from sglang.srt.hardware_backend.musa.jit_kernel import (
        grouped_topk_softmax_tilelang,
    )

    gating_output = torch.empty((1, 48), device="musa", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        grouped_topk_softmax_tilelang(
            gating_output,
            7,
            3,
            2,
            True,
            num_fused_shared_experts=2,
            routed_scaling_factor=1.0,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
