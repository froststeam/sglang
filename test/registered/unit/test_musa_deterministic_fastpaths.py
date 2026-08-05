import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.musa.layers import gemv_auto_tune
from sglang.srt.layers import sampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


@contextmanager
def _deterministic_mode(enabled):
    old_value = envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
    envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(enabled)
    try:
        yield
    finally:
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(old_value)


def test_deterministic_inference_disables_musa_gemv_paths():
    layer = SimpleNamespace(weight=torch.empty((4, 4)))
    inputs = torch.empty((1, 4))
    key = gemv_auto_tune._policy_key(
        "bf16", torch.float32, torch.float32, 1, 4, 4
    )
    old_deterministic = envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
    old_policy = gemv_auto_tune._GEMV_POLICY.get(key)

    try:
        gemv_auto_tune._GEMV_POLICY[key] = "gemv"
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(True)
        with patch.object(gemv_auto_tune, "is_musa", return_value=True):
            assert not gemv_auto_tune.should_use_musa_gemv(
                layer, inputs, quant_kind="bf16"
            )
            assert (
                gemv_auto_tune.maybe_apply_musa_gemv_activation(
                    layer, inputs, activation="silu"
                )
                is None
            )
            with patch.object(gemv_auto_tune, "_find_gemv_targets") as find_targets:
                gemv_auto_tune.maybe_autotune_musa_gemv(torch.nn.Identity())
                find_targets.assert_not_called()
    finally:
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(old_deterministic)
        if old_policy is None:
            gemv_auto_tune._GEMV_POLICY.pop(key, None)
        else:
            gemv_auto_tune._GEMV_POLICY[key] = old_policy


def test_nondeterministic_inference_keeps_musa_gemv_policy():
    layer = SimpleNamespace(weight=torch.empty((4, 4)))
    inputs = torch.empty((1, 4))
    key = gemv_auto_tune._policy_key(
        "bf16", inputs.dtype, layer.weight.dtype, 1, 4, 4
    )
    old_policy = gemv_auto_tune._GEMV_POLICY.get(key)

    try:
        gemv_auto_tune._GEMV_POLICY[key] = "gemv"
        with (
            _deterministic_mode(False),
            patch.object(gemv_auto_tune, "_DISABLE_POLICY", False),
            patch.object(gemv_auto_tune, "is_musa", return_value=True),
        ):
            assert gemv_auto_tune.should_use_musa_gemv(
                layer, inputs, quant_kind="bf16"
            )
    finally:
        if old_policy is None:
            gemv_auto_tune._GEMV_POLICY.pop(key, None)
        else:
            gemv_auto_tune._GEMV_POLICY[key] = old_policy


def test_nondeterministic_inference_keeps_musa_gemv_activation():
    quant_method_cls = type("UnquantizedLinearMethod", (), {})
    layer = SimpleNamespace(
        weight=torch.empty((4, 4), dtype=torch.bfloat16),
        quant_method=quant_method_cls(),
        bias=None,
        skip_bias_add=False,
        gather_output=False,
    )
    inputs = torch.empty((1, 4), dtype=torch.bfloat16)
    key = gemv_auto_tune._policy_key(
        "bf16", inputs.dtype, layer.weight.dtype, 1, 4, 4
    )
    old_policy = gemv_auto_tune._ACTIVATION_POLICIES["silu"].get(key)
    expected = torch.empty((1, 4), dtype=torch.bfloat16)
    musa_gemv = MagicMock(return_value=expected)
    gemv_module = SimpleNamespace(musa_gemv=musa_gemv)

    try:
        gemv_auto_tune._ACTIVATION_POLICIES["silu"][key] = "gemv"
        with (
            _deterministic_mode(False),
            patch.object(gemv_auto_tune, "_DISABLE_POLICY", False),
            patch.object(gemv_auto_tune, "is_musa", return_value=True),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.hardware_backend.musa.jit_kernel.csrc.gemv": gemv_module
                },
            ),
        ):
            output = gemv_auto_tune.maybe_apply_musa_gemv_activation(
                layer, inputs, activation="silu"
            )

        assert output is expected
        musa_gemv.assert_called_once()
    finally:
        if old_policy is None:
            gemv_auto_tune._ACTIVATION_POLICIES["silu"].pop(key, None)
        else:
            gemv_auto_tune._ACTIVATION_POLICIES["silu"][key] = old_policy


def test_nondeterministic_inference_keeps_musa_gemv_autotune():
    model = torch.nn.Identity()
    with (
        _deterministic_mode(False),
        patch.object(gemv_auto_tune, "is_musa", return_value=True),
        patch.object(gemv_auto_tune, "_AUTOTUNE_TOKENS", (1,)),
        patch.object(gemv_auto_tune, "_find_gemv_targets", return_value=[]) as find,
    ):
        gemv_auto_tune.maybe_autotune_musa_gemv(model)

    find.assert_called_once_with(model)


@pytest.mark.parametrize(
    ("musa", "expected_dtype"),
    [(True, torch.float32), (False, torch.float64)],
)
def test_seeded_sampling_uses_supported_log_dtype(musa, expected_dtype):
    captured = {}

    def fake_multinomial(logprobs, sampling_seed, positions):
        captured["dtype"] = logprobs.dtype
        return torch.zeros((logprobs.shape[0], 1), dtype=torch.long)

    with (
        patch.object(sampler, "is_musa", return_value=musa),
        patch.object(sampler, "multinomial_with_seed", side_effect=fake_multinomial),
    ):
        sampler.top_k_top_p_min_p_sampling_from_probs_torch(
            probs=torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32),
            top_ks=torch.tensor([3]),
            top_ps=torch.tensor([1.0]),
            min_ps=torch.tensor([0.0]),
            need_min_p_sampling=False,
            sampling_seed=torch.tensor([42]),
            positions=torch.tensor([0]),
        )

    assert captured["dtype"] == expected_dtype


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
