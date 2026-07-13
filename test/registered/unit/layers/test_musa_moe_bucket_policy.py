import sys
from dataclasses import dataclass

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    _can_run_musa_moe_gemv_swiglu,
)
from sglang.srt.layers.moe.utils import (
    MusaMoeBucket,
    select_musa_moe_runner,
    set_musa_moe_bucket_policy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


@dataclass
class _Config:
    musa_moe_gemv_enabled: bool | None = None


class _Core:
    def __init__(self, config):
        self.config = config


class _Runner:
    def __init__(self):
        self.config = _Config()
        self.runner_core = _Core(self.config)


def test_musa_moe_bucket_policy_distinguishes_gemv_from_triton():
    triton_runner = _Runner()
    deep_gemm_runner = _Runner()
    set_musa_moe_bucket_policy(
        (
            MusaMoeBucket(max_tokens=5, backend="gemv"),
            MusaMoeBucket(max_tokens=4096, backend="triton"),
        )
    )
    try:
        gemv_runner = select_musa_moe_runner(5, triton_runner, deep_gemm_runner)
        pure_triton_runner = select_musa_moe_runner(11, triton_runner, deep_gemm_runner)

        assert gemv_runner.config.musa_moe_gemv_enabled is True
        assert gemv_runner.runner_core.config.musa_moe_gemv_enabled is True
        assert pure_triton_runner.config.musa_moe_gemv_enabled is False
        assert pure_triton_runner.runner_core.config.musa_moe_gemv_enabled is False
        assert gemv_runner is not pure_triton_runner

        # Variants are cached, so serving does not copy runners on every step.
        assert (
            select_musa_moe_runner(11, triton_runner, deep_gemm_runner)
            is pure_triton_runner
        )
    finally:
        set_musa_moe_bucket_policy(None)


def test_musa_moe_bucket_policy_keeps_legacy_fallback_without_policy():
    triton_runner = _Runner()
    set_musa_moe_bucket_policy(None)

    selected = select_musa_moe_runner(11, triton_runner, _Runner())

    assert selected is triton_runner
    assert selected.config.musa_moe_gemv_enabled is None


def test_musa_moe_gemv_gate_honors_bucket_override(monkeypatch):
    from sglang.srt.layers.moe.moe_runner import triton

    monkeypatch.setattr(triton, "_is_musa", True)
    monkeypatch.setattr(triton, "_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS", 16)

    hidden_states = torch.empty((11, 128), dtype=torch.bfloat16)
    quant_info = TritonMoeQuantInfo(
        w13_weight=torch.empty((2, 256, 128), dtype=torch.bfloat16),
        w2_weight=torch.empty((2, 128, 128), dtype=torch.bfloat16),
    )
    config = MoeRunnerConfig(
        num_experts=2,
        num_local_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=128,
        top_k=1,
    )

    config.musa_moe_gemv_enabled = False
    assert not _can_run_musa_moe_gemv_swiglu(hidden_states, quant_info, config)

    config.musa_moe_gemv_enabled = True
    assert _can_run_musa_moe_gemv_swiglu(hidden_states, quant_info, config)

    # No mixed-policy override keeps the legacy standalone threshold behavior.
    config.musa_moe_gemv_enabled = None
    assert _can_run_musa_moe_gemv_swiglu(hidden_states, quant_info, config)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
