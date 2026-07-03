# DeepSeek V4 MUSA MoE prefill compact DeepGEMM operator tests/benchmarks.

from __future__ import annotations

import os
import time
import types
from dataclasses import dataclass
from typing import Callable, Iterable

import pytest
import torch

from sglang.test.ci.ci_register import register_musa_ci
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend

from ..utils import get_musa_device


register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="opt-in benchmark; set SGLANG_RUN_DEEPSEEK_V4_MUSA_BENCHMARK=1 manually",
)

_BLOCK_SHAPE = [128, 128]
_DEFAULT_TOP_K = 6
_DEFAULT_NUM_EXPERTS = 256
_DEFAULT_NUM_LOCAL_EXPERTS = 16
_DEFAULT_HIDDEN_SIZE = 4096
_DEFAULT_INTERMEDIATE_SIZE = 2048
_SWIGLU_LIMIT = 10.0


@dataclass(frozen=True)
class _MoeBenchCase:
    label: str
    num_tokens: int
    routing: str
    hidden_size: int = _DEFAULT_HIDDEN_SIZE
    intermediate_size: int = _DEFAULT_INTERMEDIATE_SIZE
    top_k: int = _DEFAULT_TOP_K
    num_experts: int = _DEFAULT_NUM_EXPERTS
    num_local_experts: int = _DEFAULT_NUM_LOCAL_EXPERTS


@dataclass(frozen=True)
class _RoutingTensors:
    topk_ids_cpu: torch.Tensor
    topk_weights_cpu: torch.Tensor
    valid_routes: int
    padded_valid_rows: int
    allocated_rows: int
    worst_case_rows: int


@dataclass
class _MoeFixture:
    case: _MoeBenchCase
    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput
    runner_config: MoeRunnerConfig
    triton_quant_info: TritonMoeQuantInfo
    deepgemm_quant_info: DeepGemmMoeQuantInfo
    triton_runner: MoeRunner
    deepgemm_runner: MoeRunner
    valid_routes: int
    padded_valid_rows: int
    allocated_rows: int
    worst_case_rows: int


@dataclass(frozen=True)
class _BenchStats:
    median: float
    min: float
    p95: float


def _experimental_env_field():
    field = getattr(envs, "SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL", None)
    if field is None:
        pytest.skip("SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL is not available in this branch")
    return field


def _ensure_global_server_args() -> None:
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def _require_benchmark_env() -> torch.device:
    if os.environ.get("SGLANG_RUN_DEEPSEEK_V4_MUSA_BENCHMARK") != "1":
        pytest.skip(
            "set SGLANG_RUN_DEEPSEEK_V4_MUSA_BENCHMARK=1 to run MoE MUSA benchmarks"
        )
    _experimental_env_field()
    _ensure_global_server_args()
    pytest.importorskip("tile_kernels")
    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        pytest.skip("DeepGEMM JIT is not enabled on this device/runtime")
    return get_musa_device()


def _sync() -> None:
    getattr(torch, "musa").synchronize()


def _stats(values: list[float]) -> _BenchStats:
    assert values
    values = sorted(values)
    p95_idx = min(len(values) - 1, int(round((len(values) - 1) * 0.95)))
    return _BenchStats(
        median=values[len(values) // 2],
        min=values[0],
        p95=values[p95_idx],
    )


def _bench_device_ms_samples(
    fn: Callable[[], object], *, warmup: int, iters: int
) -> _BenchStats:
    musa = getattr(torch, "musa")
    for _ in range(warmup):
        fn()
    _sync()

    values: list[float] = []
    for _ in range(iters):
        start = musa.Event(enable_timing=True)
        end = musa.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        values.append(start.elapsed_time(end))
    return _stats(values)


def _bench_host_ms_samples(
    fn: Callable[[], object], *, warmup: int, iters: int
) -> _BenchStats:
    for _ in range(warmup):
        fn()
    _sync()

    values: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        values.append((time.perf_counter() - start) * 1.0e3)
    return _stats(values)


def _case_filter(cases: Iterable[_MoeBenchCase]) -> list[_MoeBenchCase]:
    selected = os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_CASES")
    if not selected:
        return list(cases)
    allow = {item.strip() for item in selected.split(",") if item.strip()}
    return [case for case in cases if case.label in allow]



def _static_cap_values(default: str = "128,256") -> list[int]:
    selected = os.environ.get("SGLANG_DSV4_MUSA_MOE_STATIC_CAPS", default)
    caps = [int(item.strip()) for item in selected.split(",") if item.strip()]
    return sorted(set(caps))


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _compact_worst_case_rows(num_routes: int, num_local_experts: int) -> int:
    block_m = deep_gemm_wrapper.DEEPGEMM_BLOCK_M
    return (
        _ceil_div(num_routes + num_local_experts * (block_m - 1), block_m)
        * block_m
    )


def _make_routing(case: _MoeBenchCase) -> _RoutingTensors:
    num_routes = case.num_tokens * case.top_k
    flat = torch.arange(num_routes, dtype=torch.int64)
    token_ids = flat // case.top_k

    if case.routing == "dense_local":
        topk_ids = (flat + token_ids) % case.num_local_experts
    elif case.routing == "ep16_sparse":
        # Simulate EP local filtering after global 256-expert routing.
        global_ids = (flat * 37 + token_ids * 17 + 11) % case.num_experts
        topk_ids = torch.where(
            global_ids < case.num_local_experts,
            global_ids,
            torch.full_like(global_ids, -1),
        )
    elif case.routing == "skewed":
        topk_ids = torch.zeros((case.num_tokens, case.top_k), dtype=torch.int64)
        if case.top_k > 1:
            topk_ids[:, 1] = 0
        for idx in range(2, case.top_k):
            topk_ids[:, idx] = 1 + (torch.arange(case.num_tokens) + idx) % 3
        topk_ids = topk_ids.reshape(-1)
    elif case.routing == "tp8_dense":
        # TP8 + no EP keeps every routed expert local; N is sharded to 256.
        global_ids = (flat * 37 + token_ids * 17 + 11) % case.num_experts
        topk_ids = global_ids
    else:
        raise ValueError(f"unknown routing pattern: {case.routing}")

    topk_ids = topk_ids.reshape(case.num_tokens, case.top_k).to(torch.int32)
    valid = topk_ids >= 0
    route_rank = torch.arange(1, case.top_k + 1, dtype=torch.float32).view(1, -1)
    weights = route_rank.repeat(case.num_tokens, 1)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    weights = weights / denom

    valid_flat = topk_ids[valid].to(torch.int64)
    counts = torch.bincount(valid_flat, minlength=case.num_local_experts)
    block_m = deep_gemm_wrapper.DEEPGEMM_BLOCK_M
    padded_valid_rows = int(
        ((counts + block_m - 1) // block_m * block_m).sum().item()
    )

    return _RoutingTensors(
        topk_ids_cpu=topk_ids.contiguous(),
        topk_weights_cpu=weights.contiguous(),
        valid_routes=int(valid.sum().item()),
        padded_valid_rows=padded_valid_rows,
        allocated_rows=padded_valid_rows,
        worst_case_rows=_compact_worst_case_rows(num_routes, case.num_local_experts),
    )


def _make_fp8_weight(
    shape: tuple[int, int, int], device: torch.device, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    weight = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.02
    return weight.to(torch.float8_e4m3fn).contiguous()


def _make_scale(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    _, n, k = shape
    scale_shape = (shape[0], _ceil_div(n, _BLOCK_SHAPE[0]), _ceil_div(k, _BLOCK_SHAPE[1]))
    return torch.ones(scale_shape, device=device, dtype=torch.float32)


def _make_fixture(case: _MoeBenchCase, device: torch.device) -> _MoeFixture:
    routing = _make_routing(case)
    torch.manual_seed(case.num_tokens + case.hidden_size + case.intermediate_size)
    hidden_states = (
        torch.randn((case.num_tokens, case.hidden_size), device=device, dtype=torch.bfloat16)
        * 0.05
    ).contiguous()

    w13_shape = (
        case.num_local_experts,
        case.intermediate_size * 2,
        case.hidden_size,
    )
    w2_shape = (
        case.num_local_experts,
        case.hidden_size,
        case.intermediate_size,
    )
    w13_weight = _make_fp8_weight(w13_shape, device, seed=case.num_tokens + 13)
    w2_weight = _make_fp8_weight(w2_shape, device, seed=case.num_tokens + 29)
    w13_scale = _make_scale(w13_shape, device)
    w2_scale = _make_scale(w2_shape, device)

    topk_ids = routing.topk_ids_cpu.to(device=device, non_blocking=True)
    topk_weights = routing.topk_weights_cpu.to(device=device, non_blocking=True)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.empty((case.num_tokens, 0), device=device, dtype=torch.float32),
    )

    runner_config = MoeRunnerConfig(
        num_experts=case.num_experts,
        num_local_experts=case.num_local_experts,
        hidden_size=case.hidden_size,
        intermediate_size_per_partition=case.intermediate_size,
        top_k=case.top_k,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        inplace=False,
        no_combine=False,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    triton_quant_info = TritonMoeQuantInfo(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        use_fp8_w8a8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=_BLOCK_SHAPE,
    )
    deepgemm_quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=_BLOCK_SHAPE,
    )

    return _MoeFixture(
        case=case,
        hidden_states=hidden_states,
        topk_output=topk_output,
        runner_config=runner_config,
        triton_quant_info=triton_quant_info,
        deepgemm_quant_info=deepgemm_quant_info,
        triton_runner=MoeRunner(MoeRunnerBackend.TRITON, runner_config),
        deepgemm_runner=MoeRunner(MoeRunnerBackend.DEEP_GEMM, runner_config),
        valid_routes=routing.valid_routes,
        padded_valid_rows=routing.padded_valid_rows,
        allocated_rows=routing.allocated_rows,
        worst_case_rows=routing.worst_case_rows,
    )


def _dispatch_for(fixture: _MoeFixture) -> StandardDispatchOutput:
    # DeepGEMM pre-permute disposes its input tensor object; pass a view so the
    # base tensor remains reusable across benchmark iterations.
    hidden_states = fixture.hidden_states.as_strided(
        fixture.hidden_states.shape, fixture.hidden_states.stride()
    )
    return StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=fixture.topk_output,
    )


def _run_triton(fixture: _MoeFixture) -> torch.Tensor:
    return fixture.triton_runner.run(
        _dispatch_for(fixture), fixture.triton_quant_info
    ).hidden_states


def _run_deepgemm(fixture: _MoeFixture) -> torch.Tensor:
    with _experimental_env_field().override(True):
        return fixture.deepgemm_runner.run(
            _dispatch_for(fixture), fixture.deepgemm_quant_info
        ).hidden_states



def _run_deepgemm_static_cap(
    fixture: _MoeFixture,
    cap_per_expert: int,
    *,
    use_src2dst_count: bool = False,
    post_combine_mode: str = "src2dst",
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.moe_prefill_ops import (
        try_moe_post_combine_tilelang_musa,
        try_moe_post_combine_src2dst_cached_tilelang_musa,
        try_moe_post_combine_src2dst_tilelang_musa,
    )
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_static_cap_preprocess_musa,
        post_reorder_triton_kernel,
        tma_align_input_scale,
    )
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        _try_tilekernels_swiglu_quant_musa,
    )
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    topk_ids = fixture.topk_output.topk_ids
    topk_weights = fixture.topk_output.topk_weights
    hidden_states = fixture.hidden_states.as_strided(
        fixture.hidden_states.shape, fixture.hidden_states.stride()
    )
    quant_info = fixture.deepgemm_quant_info
    runner_config = fixture.runner_config
    block_k = quant_info.block_shape[1]
    scale_block_size = 128
    recipe_a, recipe_b = (
        ((1, 128), (1, 32)) if quant_info.is_fp4_experts else (None, None)
    )

    src2dst, compact_input, compact_scale, m_indices, all_tokens, overflow_flag = (
        moe_ep_deepgemm_static_cap_preprocess_musa(
            topk_ids,
            runner_config.num_local_experts,
            hidden_states,
            runner_config.top_k,
            quant_info.block_shape,
            cap_per_expert,
            use_src2dst_count=use_src2dst_count,
        )
    )

    N = quant_info.w13_weight.size(1)
    K = fixture.hidden_states.shape[1]
    lhs_scale = compact_scale
    if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
        lhs_scale = tma_align_input_scale(lhs_scale)
    gateup_output = torch.empty(
        (all_tokens, N), device=fixture.hidden_states.device, dtype=torch.bfloat16
    )
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
        (compact_input, lhs_scale),
        (quant_info.w13_weight, quant_info.w13_scale),
        gateup_output,
        m_indices,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
    )

    fused_quant = _try_tilekernels_swiglu_quant_musa(
        gateup_output.view(-1, N), scale_block_size, runner_config.swiglu_limit
    )
    if fused_quant is not None:
        down_input_fp8, down_input_scale = fused_quant
    else:
        if runner_config.swiglu_limit is not None:
            half = N // 2
            gateup_output[:, :half].clamp_(max=runner_config.swiglu_limit)
            gateup_output[:, half:].clamp_(
                min=-runner_config.swiglu_limit, max=runner_config.swiglu_limit
            )
        down_input = torch.nn.SwishGLU()(gateup_output.view(-1, N))
        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input, block_k
        )

    rhs_scale = down_input_scale
    if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
        rhs_scale = tma_align_input_scale(rhs_scale)
    down_output = torch.empty(
        (all_tokens, K), device=fixture.hidden_states.device, dtype=torch.bfloat16
    )
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
        (down_input_fp8, rhs_scale),
        (quant_info.w2_weight, quant_info.w2_scale),
        down_output,
        m_indices,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
    )

    output = torch.empty_like(fixture.hidden_states)
    used_post = False
    if post_combine_mode in {"src2dst", "auto"}:
        used_post = try_moe_post_combine_src2dst_tilelang_musa(
            down_output,
            output,
            src2dst,
            topk_weights,
            runner_config.top_k,
            block_h=1024,
        )
    if not used_post and post_combine_mode in {"legacy", "auto"}:
        used_post = try_moe_post_combine_tilelang_musa(
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            runner_config.top_k,
        )
    if not used_post:
        post_reorder_triton_kernel[(fixture.case.num_tokens,)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            runner_config.top_k,
            fixture.case.hidden_size,
            BLOCK_SIZE=512,
        )
    return output, overflow_flag


def _report(
    name: str,
    baseline_device: _BenchStats,
    candidate_device: _BenchStats,
    baseline_host: _BenchStats,
    candidate_host: _BenchStats,
    fixture: _MoeFixture,
) -> None:
    device_speedup = baseline_device.median / candidate_device.median
    host_speedup = baseline_host.median / candidate_host.median
    print(
        f"BENCH {name}: "
        f"baseline_device={baseline_device.median:.4f} ms "
        f"candidate_device={candidate_device.median:.4f} ms "
        f"device_speedup={device_speedup:.3f}x "
        f"baseline_device_min={baseline_device.min:.4f} ms "
        f"candidate_device_min={candidate_device.min:.4f} ms "
        f"baseline_device_p95={baseline_device.p95:.4f} ms "
        f"candidate_device_p95={candidate_device.p95:.4f} ms "
        f"baseline_host={baseline_host.median:.4f} ms "
        f"candidate_host={candidate_host.median:.4f} ms "
        f"host_speedup={host_speedup:.3f}x "
        f"valid_routes={fixture.valid_routes} "
        f"padded_valid_rows={fixture.padded_valid_rows} "
        f"allocated_rows={fixture.allocated_rows} "
        f"worst_case_rows={fixture.worst_case_rows}"
    )


def _fake_forward_batch(mode_name: str):
    return types.SimpleNamespace(forward_mode=types.SimpleNamespace(name=mode_name))


def test_moe_prefill_deepgemm_high_level_dispatch_guard_keeps_decode_off(monkeypatch) -> None:
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    method = object.__new__(fp8_quant.Fp8MoEMethod)
    method.prefill_deepgemm_runner = object()
    method.runner = types.SimpleNamespace(
        runner_backend=types.SimpleNamespace(is_triton=lambda: True)
    )
    layer = types.SimpleNamespace()
    dispatch = types.SimpleNamespace(hidden_states=torch.empty((1024, 128)))

    monkeypatch.setattr(fp8_quant, "_is_musa", True)
    with _experimental_env_field().override(True):
        method._get_forward_batch_from_layer = lambda layer: _fake_forward_batch("EXTEND")
        assert method._should_use_dsv4_musa_prefill_deepgemm(layer, dispatch)

        for mode in ("DECODE", "MIXED"):
            method._get_forward_batch_from_layer = (
                lambda layer, mode=mode: _fake_forward_batch(mode)
            )
            assert not method._should_use_dsv4_musa_prefill_deepgemm(layer, dispatch)

        method._get_forward_batch_from_layer = lambda layer: _fake_forward_batch("EXTEND")
        for batch in (1, 2, 4, 8, 16):
            small_dispatch = types.SimpleNamespace(hidden_states=torch.empty((batch, 128)))
            assert not method._should_use_dsv4_musa_prefill_deepgemm(
                layer, small_dispatch
            )

    with _experimental_env_field().override(False):
        assert not method._should_use_dsv4_musa_prefill_deepgemm(layer, dispatch)


def test_moe_prefill_deepgemm_compact_preprocess_metadata() -> None:
    device = _require_benchmark_env()
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_compact_preprocess_musa,
    )

    case = _MoeBenchCase(
        label="metadata_small",
        num_tokens=8,
        routing="dense_local",
        hidden_size=128,
        intermediate_size=64,
        top_k=3,
        num_experts=8,
        num_local_experts=4,
    )
    topk_ids_cpu = torch.tensor(
        [
            [0, 1, -1],
            [2, 1, 0],
            [3, -1, -1],
            [0, 0, 2],
            [1, 3, 2],
            [-1, -1, -1],
            [2, 3, 3],
            [1, 0, -1],
        ],
        dtype=torch.int32,
    )
    hidden_states = torch.arange(
        case.num_tokens * case.hidden_size,
        device=device,
        dtype=torch.float32,
    ).reshape(case.num_tokens, case.hidden_size).to(torch.bfloat16)
    topk_ids = topk_ids_cpu.to(device=device)

    src2dst, compact_input, compact_scale, m_indices, allocated_rows = (
        moe_ep_deepgemm_compact_preprocess_musa(
            topk_ids,
            case.num_local_experts,
            hidden_states,
            case.top_k,
            _BLOCK_SHAPE,
        )
    )
    _sync()

    block_m = deep_gemm_wrapper.DEEPGEMM_BLOCK_M
    counts = torch.bincount(
        topk_ids_cpu[topk_ids_cpu >= 0].to(torch.int64),
        minlength=case.num_local_experts,
    )
    padded_counts = (counts + block_m - 1) // block_m * block_m
    offsets = torch.cumsum(padded_counts, dim=0) - padded_counts
    src2dst_cpu = src2dst.cpu()
    m_indices_cpu = m_indices.cpu()
    for expert in range(case.num_local_experts):
        start = int(offsets[expert].item())
        count = int(counts[expert].item())
        padded_end = start + int(padded_counts[expert].item())
        if start < padded_end:
            assert torch.all(m_indices_cpu[start:padded_end] == expert)

        route_mask = topk_ids_cpu == expert
        dst = src2dst_cpu[route_mask]
        assert dst.numel() == count
        if count == 0:
            continue
        assert int(dst.min().item()) >= start
        assert int(dst.max().item()) < start + count
        assert torch.unique(dst).numel() == count
        assert torch.all(m_indices_cpu[dst.to(torch.int64)] == expert)

    assert compact_input.shape == (allocated_rows, case.hidden_size)
    assert compact_scale.shape == (allocated_rows, case.hidden_size // _BLOCK_SHAPE[1])
    assert allocated_rows == int(padded_counts.sum().item())
    assert allocated_rows <= _compact_worst_case_rows(
        case.num_tokens * case.top_k, case.num_local_experts
    )


def test_moe_prefill_deepgemm_full_operator_correctness() -> None:
    device = _require_benchmark_env()
    cases = (
        _MoeBenchCase(
            label="correctness_smoke_m1024",
            num_tokens=1024,
            routing="dense_local",
            hidden_size=1024,
            intermediate_size=512,
            top_k=6,
            num_experts=16,
            num_local_experts=4,
        ),
        _MoeBenchCase(
            label="correctness_prefill_m2048_ep16_sparse",
            num_tokens=2048,
            routing="ep16_sparse",
        ),
    )

    for case in cases:
        fixture = _make_fixture(case, device)
        with torch.no_grad():
            baseline = _run_triton(fixture).float()
            candidate = _run_deepgemm(fixture).float()
        _sync()

        assert torch.isfinite(baseline).all().item()
        assert torch.isfinite(candidate).all().item()
        diff = (baseline - candidate).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        print(
            f"CHECK moe_prefill_full/{case.label}: "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
            f"valid_routes={fixture.valid_routes} "
            f"padded_valid_rows={fixture.padded_valid_rows} "
            f"allocated_rows={fixture.allocated_rows} "
        f"worst_case_rows={fixture.worst_case_rows}"
        )
        torch.testing.assert_close(
            candidate.cpu(),
            baseline.cpu(),
            rtol=5e-2,
            atol=5e-1,
        )
        del fixture, baseline, candidate, diff
        _sync()
        getattr(torch, "musa").empty_cache()


def test_moe_prefill_deepgemm_full_operator_benchmark() -> None:
    device = _require_benchmark_env()
    cases = _case_filter(
        (
            _MoeBenchCase("prefill_m2048_ep16_sparse", 2048, "ep16_sparse"),
            _MoeBenchCase("prefill_m4096_ep16_sparse", 4096, "ep16_sparse"),
            _MoeBenchCase("prefill_m8192_ep16_sparse", 8192, "ep16_sparse"),
            _MoeBenchCase(
                "tp8_m4096_e256_n256_dense",
                4096,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m8192_e256_n256_dense",
                8192,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m9216_e256_n256_dense",
                9216,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m16384_e256_n256_dense",
                16384,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase("prefill_m4096_dense_local", 4096, "dense_local"),
            _MoeBenchCase("prefill_m4096_skewed", 4096, "skewed"),
        )
    )
    warmup = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_WARMUP", "10"))
    iters = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_ITERS", "50"))

    for case in cases:
        fixture = _make_fixture(case, device)
        with torch.no_grad():
            baseline = _run_triton(fixture)
            candidate = _run_deepgemm(fixture)
        _sync()
        assert torch.isfinite(baseline).all().item()
        assert torch.isfinite(candidate).all().item()
        del baseline, candidate
        _sync()

        with torch.no_grad():
            baseline_device = _bench_device_ms_samples(
                lambda: _run_triton(fixture), warmup=warmup, iters=iters
            )
            candidate_device = _bench_device_ms_samples(
                lambda: _run_deepgemm(fixture), warmup=warmup, iters=iters
            )
            baseline_host = _bench_host_ms_samples(
                lambda: _run_triton(fixture), warmup=warmup, iters=iters
            )
            candidate_host = _bench_host_ms_samples(
                lambda: _run_deepgemm(fixture), warmup=warmup, iters=iters
            )

        _report(
            f"moe_prefill_full/{case.label}"
            f"[m={case.num_tokens},h={case.hidden_size},inter={case.intermediate_size},"
            f"topk={case.top_k},local_e={case.num_local_experts},routing={case.routing}]",
            baseline_device,
            candidate_device,
            baseline_host,
            candidate_host,
            fixture,
        )
        del fixture
        _sync()
        getattr(torch, "musa").empty_cache()



def test_moe_prefill_deepgemm_static_cap_full_operator_benchmark() -> None:
    device = _require_benchmark_env()
    cases = _case_filter(
        (
            _MoeBenchCase(
                "tp8_m4096_e256_n256_dense",
                4096,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m8192_e256_n256_dense",
                8192,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
        )
    )
    warmup = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_ITERS", "20"))

    for case in cases:
        fixture = _make_fixture(case, device)
        with torch.no_grad():
            triton_device = _bench_device_ms_samples(
                lambda: _run_triton(fixture), warmup=warmup, iters=iters
            )
            exact_device = _bench_device_ms_samples(
                lambda: _run_deepgemm(fixture), warmup=warmup, iters=iters
            )
            exact_output = _run_deepgemm(fixture).float()
            _sync()

            for cap_per_expert in _static_cap_values():
                for use_src2dst_count in (False, True):
                    static_output, overflow_flag = _run_deepgemm_static_cap(
                        fixture,
                        cap_per_expert,
                        use_src2dst_count=use_src2dst_count,
                    )
                    _sync()
                    overflow = int(overflow_flag.item())
                    tag = "src2dst_count" if use_src2dst_count else "rank_prefill"
                    if overflow != 0:
                        print(
                            f"STATIC_CAP_FULL_SKIP {case.label}/{tag}: cap={cap_per_expert} "
                            f"rows={case.num_local_experts * cap_per_expert} overflow={overflow}"
                        )
                        del static_output, overflow_flag
                        continue

                    assert torch.isfinite(static_output).all().item()
                    torch.testing.assert_close(
                        static_output.float().cpu(), exact_output.cpu(), rtol=5e-2, atol=5e-1
                    )
                    static_device = _bench_device_ms_samples(
                        lambda cap=cap_per_expert, use_src=use_src2dst_count: _run_deepgemm_static_cap(
                            fixture, cap, use_src2dst_count=use_src
                        )[0],
                        warmup=warmup,
                        iters=iters,
                    )
                    print(
                        f"STATIC_CAP_FULL {case.label}/{tag}: cap={cap_per_expert} "
                        f"rows={case.num_local_experts * cap_per_expert} "
                        f"triton={triton_device.median:.4f}ms "
                        f"exact_compact={exact_device.median:.4f}ms "
                        f"static_cap={static_device.median:.4f}ms "
                        f"speedup_vs_triton={triton_device.median / static_device.median:.3f}x "
                        f"speedup_vs_exact={exact_device.median / static_device.median:.3f}x"
                    )
                    del static_output, overflow_flag

        del fixture, exact_output
        _sync()
        getattr(torch, "musa").empty_cache()


def test_moe_prefill_deepgemm_static_cap_dispatch_benchmark() -> None:
    device = _require_benchmark_env()
    from sglang.srt.layers.moe.ep_moe.kernels import (
        select_moe_ep_deepgemm_static_cap_tp8_musa,
    )

    cases = _case_filter(
        (
            _MoeBenchCase(
                "tp8_m2048_e256_n256_dense",
                2048,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m4096_e256_n256_dense",
                4096,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m8192_e256_n256_dense",
                8192,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
        )
    )
    warmup = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_ITERS", "20"))

    for case in cases:
        fixture = _make_fixture(case, device)
        cap = select_moe_ep_deepgemm_static_cap_tp8_musa(
            case.num_tokens,
            case.num_local_experts,
            case.top_k,
            chunked_prefill_size=8192,
        )
        if case.num_tokens < 3600:
            assert cap is None
            print(f"STATIC_CAP_DISPATCH_SKIP {case.label}: reason=below_min_tokens")
            del fixture
            continue
        assert cap in {128, 256}

        with torch.no_grad():
            exact_output = _run_deepgemm(fixture).float()
            static_output, overflow_flag = _run_deepgemm_static_cap(
                fixture,
                cap,
                use_src2dst_count=True,
                post_combine_mode="src2dst",
            )
            _sync()
            overflow = int(overflow_flag.item())
            assert overflow == 0
            torch.testing.assert_close(
                static_output.float().cpu(), exact_output.cpu(), rtol=5e-2, atol=5e-1
            )
            exact_stats = _bench_device_ms_samples(
                lambda: _run_deepgemm(fixture), warmup=warmup, iters=iters
            )
            static_stats = _bench_device_ms_samples(
                lambda: _run_deepgemm_static_cap(
                    fixture,
                    cap,
                    use_src2dst_count=True,
                    post_combine_mode="src2dst",
                )[0],
                warmup=warmup,
                iters=iters,
            )
        print(
            f"STATIC_CAP_DISPATCH {case.label}: cap={cap} "
            f"rows={case.num_local_experts * cap} overflow={overflow} "
            f"exact_compact={exact_stats.median:.4f}ms "
            f"static_src2dst={static_stats.median:.4f}ms "
            f"speedup_vs_exact={exact_stats.median / static_stats.median:.3f}x"
        )
        del fixture, exact_output, static_output, overflow_flag
        _sync()
        getattr(torch, "musa").empty_cache()


def test_moe_prefill_deepgemm_static_cap_overflow_guard() -> None:
    device = _require_benchmark_env()
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_static_cap_preprocess_musa,
    )

    case = _MoeBenchCase(
        label="static_cap_overflow_skewed",
        num_tokens=1024,
        routing="skewed",
        intermediate_size=256,
        num_local_experts=256,
    )
    fixture = _make_fixture(case, device)
    result = moe_ep_deepgemm_static_cap_preprocess_musa(
        fixture.topk_output.topk_ids,
        fixture.runner_config.num_local_experts,
        fixture.hidden_states,
        fixture.runner_config.top_k,
        fixture.deepgemm_quant_info.block_shape,
        cap_per_expert=16,
        use_src2dst_count=True,
    )
    src2dst, _, _, _, _, overflow_flag = result
    _sync()
    overflow = int(overflow_flag.item())
    assert overflow > 0
    assert int((src2dst < 0).sum().item()) >= overflow
    print(f"STATIC_CAP_OVERFLOW {case.label}: cap=16 overflow={overflow}")
    del fixture, result
    _sync()
    getattr(torch, "musa").empty_cache()



def _bench_split_device_ms(
    run_once: Callable[[], tuple[dict[str, tuple[object, object]], object]],
    *,
    warmup: int,
    iters: int,
) -> dict[str, _BenchStats]:
    for _ in range(warmup):
        _, keepalive = run_once()
        _sync()
        del keepalive

    samples: dict[str, list[float]] = {}
    for _ in range(iters):
        events, keepalive = run_once()
        _sync()
        for name, (start, end) in events.items():
            samples.setdefault(name, []).append(start.elapsed_time(end))
        del keepalive
    return {name: _stats(values) for name, values in samples.items()}


def _print_split_report(label: str, stats: dict[str, _BenchStats]) -> None:
    total = sum(stat.median for stat in stats.values())
    parts = " ".join(
        f"{name}={stat.median:.4f}ms({stat.median / total * 100.0:.1f}%)"
        for name, stat in stats.items()
    )
    print(f"SPLIT {label}: total_stage_sum={total:.4f}ms {parts}")


def _run_deepgemm_split_once(
    fixture: _MoeFixture,
) -> tuple[dict[str, tuple[object, object]], object]:
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_compact_preprocess_musa,
        post_reorder_triton_kernel,
        tma_align_input_scale,
    )
    from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.moe_prefill_ops import (
        try_moe_post_combine_tilelang_musa,
    )
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        _try_tilekernels_swiglu_quant_musa,
    )
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    musa = getattr(torch, "musa")
    events: dict[str, tuple[object, object]] = {}

    def timed(name: str, fn: Callable[[], object]):
        start = musa.Event(enable_timing=True)
        end = musa.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        events[name] = (start, end)
        return value

    topk_ids = fixture.topk_output.topk_ids
    topk_weights = fixture.topk_output.topk_weights
    hidden_states = fixture.hidden_states.as_strided(
        fixture.hidden_states.shape, fixture.hidden_states.stride()
    )
    quant_info = fixture.deepgemm_quant_info
    runner_config = fixture.runner_config
    block_k = quant_info.block_shape[1]
    scale_block_size = 128
    recipe_a, recipe_b = (
        ((1, 128), (1, 32)) if quant_info.is_fp4_experts else (None, None)
    )

    src2dst, compact_input, compact_scale, m_indices, all_tokens = timed(
        "dg_preprocess",
        lambda: moe_ep_deepgemm_compact_preprocess_musa(
            topk_ids,
            runner_config.num_local_experts,
            hidden_states,
            runner_config.top_k,
            quant_info.block_shape,
        ),
    )

    N = quant_info.w13_weight.size(1)
    K = fixture.hidden_states.shape[1]

    def gemm1():
        lhs_scale = compact_scale
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            lhs_scale = tma_align_input_scale(lhs_scale)
        gateup_output = torch.empty(
            (all_tokens, N),
            device=fixture.hidden_states.device,
            dtype=torch.bfloat16,
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (compact_input, lhs_scale),
            (quant_info.w13_weight, quant_info.w13_scale),
            gateup_output,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return gateup_output

    gateup_output = timed("dg_gemm1", gemm1)

    def act_quant():
        fused_quant = _try_tilekernels_swiglu_quant_musa(
            gateup_output.view(-1, N),
            scale_block_size,
            runner_config.swiglu_limit,
        )
        if fused_quant is not None:
            return fused_quant

        if runner_config.swiglu_limit is not None:
            half = N // 2
            gateup_output[:, :half].clamp_(max=runner_config.swiglu_limit)
            gateup_output[:, half:].clamp_(
                min=-runner_config.swiglu_limit, max=runner_config.swiglu_limit
            )
        down_input = torch.nn.SwishGLU()(gateup_output.view(-1, N))
        return sglang_per_token_group_quant_fp8(down_input, block_k)

    down_input_fp8, down_input_scale = timed("dg_swiglu_quant", act_quant)

    def gemm2():
        rhs_scale = down_input_scale
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            rhs_scale = tma_align_input_scale(rhs_scale)
        down_output = torch.empty(
            (all_tokens, K),
            device=fixture.hidden_states.device,
            dtype=torch.bfloat16,
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, rhs_scale),
            (quant_info.w2_weight, quant_info.w2_scale),
            down_output,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return down_output

    down_output = timed("dg_gemm2", gemm2)

    def post_combine():
        output = torch.empty_like(fixture.hidden_states)
        if not try_moe_post_combine_tilelang_musa(
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            runner_config.top_k,
        ):
            post_reorder_triton_kernel[(fixture.case.num_tokens,)](
                down_output,
                output,
                src2dst,
                topk_ids,
                topk_weights,
                runner_config.top_k,
                fixture.case.hidden_size,
                BLOCK_SIZE=512,
            )
        return output

    output = timed("dg_post_combine", post_combine)
    return events, (
        src2dst,
        compact_input,
        compact_scale,
        m_indices,
        gateup_output,
        down_input_fp8,
        down_input_scale,
        down_output,
        output,
    )


def _run_triton_split_once(
    fixture: _MoeFixture,
) -> tuple[dict[str, tuple[object, object]], object]:
    import triton.language as tl

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
        moe_sum_reduce,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.server_args import get_global_server_args

    musa = getattr(torch, "musa")
    events: dict[str, tuple[object, object]] = {}

    def timed(name: str, fn: Callable[[], object]):
        start = musa.Event(enable_timing=True)
        end = musa.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        events[name] = (start, end)
        return value

    hidden_states = fixture.hidden_states
    topk_ids = fixture.topk_output.topk_ids
    topk_weights = fixture.topk_output.topk_weights
    quant_info = fixture.triton_quant_info
    runner_config = fixture.runner_config
    w1 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    topk = topk_ids.shape[1]
    filter_expert = runner_config.num_experts != runner_config.num_local_experts
    compute_type = tl.bfloat16

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = timed(
        "triton_prepare",
        lambda: _prepare_fused_moe_run(
            hidden_states,
            w1,
            w2,
            topk_ids,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=quant_info.block_shape,
        ),
    )

    num_tokens = hidden_states.shape[0]
    E, N, _ = w1.shape
    padded_tokens = (
        min(num_tokens * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
        if down_moe_use_tma
        else 0
    )
    total_tokens = num_tokens * topk + padded_tokens
    out_hidden_states = torch.empty_like(hidden_states)

    use_fused_moe_sum_all_reduce = (
        get_global_server_args().enable_fused_moe_sum_all_reduce
        and topk > 2
        and not runner_config.no_combine
    )

    def gemm1():
        intermediate_cache1 = torch.empty(
            (total_tokens, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        invoke_fused_moe_kernel(
            hidden_states,
            w1,
            None,
            intermediate_cache1,
            None,
            quant_info.w13_scale,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            runner_config.apply_router_weight_on_input,
            topk,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=quant_info.block_shape,
            c_sorted=down_moe_use_tma,
            filter_expert=filter_expert,
        )
        return intermediate_cache1

    intermediate_cache1 = timed("triton_gemm1", gemm1)

    def activation():
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        assert runner_config.swiglu_limit == 10.0
        if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():
            from sglang.jit_kernel.deepseek_v4 import silu_and_mul_clamp

            silu_and_mul_clamp(
                intermediate_cache1.view(-1, N),
                intermediate_cache2,
                runner_config.swiglu_limit,
            )
        else:
            half = N // 2
            intermediate_cache1[:, :half].clamp_(max=runner_config.swiglu_limit)
            intermediate_cache1[:, half:].clamp_(
                min=-runner_config.swiglu_limit, max=runner_config.swiglu_limit
            )
            intermediate_cache2 = torch.nn.SwishGLU()(intermediate_cache1.view(-1, N))
        return intermediate_cache2

    intermediate_cache2 = timed("triton_swiglu", activation)

    intermediate_cache3 = torch.empty(
        (num_tokens, topk, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    def gemm2():
        out_slice = out_hidden_states if use_fused_moe_sum_all_reduce else None
        if out_slice is not None:
            out_slice.zero_()
        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            None,
            out_slice if out_slice is not None else intermediate_cache3,
            None,
            quant_info.w2_scale,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not runner_config.apply_router_weight_on_input,
            1,
            down_config or config,
            compute_type=compute_type,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=quant_info.block_shape,
            a_use_tma=down_moe_use_tma,
            b_use_tma=down_moe_use_tma,
            filter_expert=filter_expert,
            fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,
            router_topk=topk,
        )
        return out_slice if out_slice is not None else intermediate_cache3

    gemm2_output = timed("triton_gemm2", gemm2)

    def combine():
        if use_fused_moe_sum_all_reduce:
            return out_hidden_states
        moe_sum_reduce(intermediate_cache3, out_hidden_states, 1.0)
        return out_hidden_states

    output = timed("triton_combine", combine)
    return events, (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        intermediate_cache1,
        intermediate_cache2,
        intermediate_cache3,
        gemm2_output,
        output,
    )


def test_moe_prefill_deepgemm_stage_breakdown() -> None:
    device = _require_benchmark_env()
    cases = _case_filter(
        (
            _MoeBenchCase(
                "tp8_m4096_e256_n256_dense",
                4096,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m8192_e256_n256_dense",
                8192,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m9216_e256_n256_dense",
                9216,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m16384_e256_n256_dense",
                16384,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
        )
    )
    warmup = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_ITERS", "20"))

    for case in cases:
        fixture = _make_fixture(case, device)
        with torch.no_grad():
            triton_stats = _bench_split_device_ms(
                lambda: _run_triton_split_once(fixture), warmup=warmup, iters=iters
            )
            deepgemm_stats = _bench_split_device_ms(
                lambda: _run_deepgemm_split_once(fixture), warmup=warmup, iters=iters
            )
        name = (
            f"moe_prefill_split/{case.label}"
            f"[m={case.num_tokens},h={case.hidden_size},inter={case.intermediate_size},"
            f"topk={case.top_k},local_e={case.num_local_experts}]"
        )
        _print_split_report(f"triton/{name}", triton_stats)
        _print_split_report(f"deepgemm/{name}", deepgemm_stats)
        del fixture
        _sync()
        getattr(torch, "musa").empty_cache()



def _bench_preprocess_detail(
    run_once: Callable[[], tuple[dict[str, tuple[object, object]], dict[str, float], object]],
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, _BenchStats], dict[str, _BenchStats]]:
    for _ in range(warmup):
        _, _, keepalive = run_once()
        _sync()
        del keepalive

    event_samples: dict[str, list[float]] = {}
    host_samples: dict[str, list[float]] = {}
    for _ in range(iters):
        events, host_ms, keepalive = run_once()
        _sync()
        for name, (start, end) in events.items():
            event_samples.setdefault(name, []).append(start.elapsed_time(end))
        for name, value in host_ms.items():
            host_samples.setdefault(name, []).append(value)
        del keepalive
    return (
        {name: _stats(values) for name, values in event_samples.items()},
        {name: _stats(values) for name, values in host_samples.items()},
    )


def _print_preprocess_detail_report(
    label: str,
    event_stats: dict[str, _BenchStats],
    host_stats: dict[str, _BenchStats],
) -> None:
    total_event = sum(stat.median for stat in event_stats.values())
    event_parts = " ".join(
        f"{name}={stat.median:.4f}ms({stat.median / total_event * 100.0:.1f}%)"
        for name, stat in event_stats.items()
    )
    host_parts = " ".join(
        f"{name}={stat.median:.4f}ms" for name, stat in host_stats.items()
    )
    print(
        f"PREPROCESS_SPLIT {label}: total_event_sum={total_event:.4f}ms "
        f"{event_parts} host_sync {host_parts}"
    )


def _run_deepgemm_preprocess_detail_once(
    fixture: _MoeFixture,
) -> tuple[dict[str, tuple[object, object]], dict[str, float], object]:
    import triton

    from sglang.srt.layers.moe.ep_moe.kernels import (
        _deepgemm_compact_count_kernel,
        _deepgemm_compact_fixed_bucket_init_kernel,
        _deepgemm_compact_fill_m_indices_fixed_bucket_tp8_kernel,
        _deepgemm_compact_fill_m_indices_kernel,
        _deepgemm_compact_make_src2dst_kernel,
        _deepgemm_compact_prefix_sum_musa,
        _deepgemm_compact_scatter_kernel,
        select_moe_ep_deepgemm_compact_fixed_bucket_rows_musa,
    )
    from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.moe_prefill_ops import (
        try_moe_deepgemm_compact_quant_scatter_tilelang_musa,
        try_moe_deepgemm_compact_src2dst_quant_scatter_tilelang_musa,
    )
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    musa = getattr(torch, "musa")
    events: dict[str, tuple[object, object]] = {}
    host_ms: dict[str, float] = {}

    def timed_event(name: str, fn: Callable[[], object]):
        start = musa.Event(enable_timing=True)
        end = musa.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        events[name] = (start, end)
        return value

    def timed_host(name: str, fn: Callable[[], object]):
        tic = time.perf_counter()
        value = fn()
        host_ms[name] = (time.perf_counter() - tic) * 1.0e3
        return value

    topk_ids = fixture.topk_output.topk_ids
    hidden_states = fixture.hidden_states.as_strided(
        fixture.hidden_states.shape, fixture.hidden_states.stride()
    )
    block_k = fixture.deepgemm_quant_info.block_shape[1]
    num_local_experts = fixture.runner_config.num_local_experts
    top_k = fixture.runner_config.top_k
    num_routes = topk_ids.numel()
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    block_m = deep_gemm_wrapper.DEEPGEMM_BLOCK_M

    def count_stage():
        counts = torch.zeros(num_local_experts, device=topk_ids.device, dtype=torch.int32)
        route_ranks = torch.empty(num_routes, device=topk_ids.device, dtype=torch.int32)
        grid = lambda meta: (triton.cdiv(num_routes, meta["BLOCK_SIZE"]),)
        _deepgemm_compact_count_kernel[grid](
            topk_ids,
            counts,
            route_ranks,
            num_routes,
            num_local_experts,
            BLOCK_SIZE=256,
        )
        return counts, route_ranks

    counts, route_ranks = timed_event("count_kernel", count_stage)

    def prefix_stage():
        padded_counts, offsets = _deepgemm_compact_prefix_sum_musa(
            counts, num_local_experts, block_m
        )
        fixed_bucket = select_moe_ep_deepgemm_compact_fixed_bucket_rows_musa(
            num_tokens,
            num_local_experts,
            top_k,
            block_m,
        )
        if fixed_bucket is None:
            all_tokens = int(padded_counts.sum().item())
            max_m_per_expert = (
                int(padded_counts.max().item()) if num_local_experts > 0 else 0
            )
        else:
            all_tokens, max_m_per_expert = fixed_bucket
        return padded_counts, offsets, all_tokens, max_m_per_expert

    start = musa.Event(enable_timing=True)
    end = musa.Event(enable_timing=True)
    start.record()
    fixed_bucket = select_moe_ep_deepgemm_compact_fixed_bucket_rows_musa(
        num_tokens,
        num_local_experts,
        top_k,
        block_m,
    )
    padded_counts, offsets, all_tokens, max_m_per_expert = timed_host(
        "prefix_fixed_host" if fixed_bucket is not None else "prefix_item_host",
        prefix_stage,
    )
    end.record()
    events["prefix_fixed_device" if fixed_bucket is not None else "prefix_item_device"] = (
        start,
        end,
    )

    scale_size = hidden_size // block_k

    def alloc_stage():
        compact_input = torch.empty(
            (all_tokens, hidden_size),
            device=hidden_states.device,
            dtype=torch.float8_e4m3fn,
        )
        compact_scale = torch.empty(
            (all_tokens, scale_size), device=hidden_states.device, dtype=torch.float32
        )
        m_indices = torch.empty(all_tokens, device=topk_ids.device, dtype=torch.int32)
        return compact_input, compact_scale, m_indices

    compact_input, compact_scale, m_indices = timed_event("alloc", alloc_stage)

    def init_stage():
        if fixed_bucket is not None and all_tokens > 0:
            _deepgemm_compact_fixed_bucket_init_kernel[
                (triton.cdiv(all_tokens, 16),)
            ](
                compact_scale,
                m_indices,
                all_tokens,
                scale_size,
                num_local_experts - 1,
                BLOCK_ROWS=16,
                BLOCK_SCALE=triton.next_power_of_2(scale_size),
            )
        else:
            compact_scale.zero_()
        return compact_scale

    timed_event(
        "init_scale_m_indices_fixed" if fixed_bucket is not None else "zero_scale",
        init_stage,
    )

    def fill_stage():
        if all_tokens > 0:
            if fixed_bucket is None:
                fill_grid = (num_local_experts, triton.cdiv(max_m_per_expert, 256))
                _deepgemm_compact_fill_m_indices_kernel[fill_grid](
                    offsets,
                    padded_counts,
                    m_indices,
                    BLOCK_SIZE=256,
                )
            else:
                _deepgemm_compact_fill_m_indices_fixed_bucket_tp8_kernel[
                    (num_local_experts, triton.cdiv(max_m_per_expert, 1024))
                ](
                    offsets,
                    padded_counts,
                    m_indices,
                    BLOCK_SIZE=1024,
                )
        return m_indices

    timed_event(
        "fill_m_indices_fixed" if fixed_bucket is not None else "fill_m_indices",
        fill_stage,
    )
    src2dst = torch.empty_like(topk_ids, dtype=torch.int32)

    if fixed_bucket is not None:
        timed_event(
            "make_src2dst_fixed",
            lambda: _deepgemm_compact_make_src2dst_kernel[
                (triton.cdiv(num_routes, 256),)
            ](
                topk_ids,
                offsets,
                route_ranks,
                src2dst,
                num_routes,
                num_local_experts,
                BLOCK_SIZE=256,
            ),
        )

    keepalive = [counts, route_ranks, padded_counts, offsets, compact_input, compact_scale, m_indices, src2dst]

    def tilelang_quant_scatter_stage():
        if fixed_bucket is not None:
            return try_moe_deepgemm_compact_src2dst_quant_scatter_tilelang_musa(
                hidden_states,
                compact_input,
                compact_scale,
                src2dst,
                top_k,
                block_k,
            )
        else:
            return try_moe_deepgemm_compact_quant_scatter_tilelang_musa(
                hidden_states,
                compact_input,
                compact_scale,
                topk_ids,
                offsets,
                route_ranks,
                src2dst,
                num_local_experts,
                top_k,
                block_k,
            )

    used_tilelang = timed_event(
        "src2dst_quant_scatter_fixed"
        if fixed_bucket is not None
        else "tilelang_quant_scatter",
        tilelang_quant_scatter_stage,
    )
    if not used_tilelang:
        hidden_states_fp8, scale = timed_event(
            "input_quant", lambda: sglang_per_token_group_quant_fp8(hidden_states, block_k)
        )
        compact_scale.zero_()
        keepalive.extend([hidden_states_fp8, scale])

        def scatter_stage():
            _deepgemm_compact_scatter_kernel[(num_tokens,)](
                hidden_states_fp8,
                scale,
                compact_input,
                compact_scale,
                topk_ids,
                offsets,
                route_ranks,
                src2dst,
                num_local_experts,
                top_k,
                hidden_size,
                scale_size,
                BLOCK_SIZE=1024,
            )
            return src2dst

        timed_event("compact_scatter", scatter_stage)

    return events, host_ms, tuple(keepalive)


def test_moe_prefill_deepgemm_preprocess_detail_benchmark() -> None:
    device = _require_benchmark_env()
    cases = _case_filter(
        (
            _MoeBenchCase(
                "tp8_m4096_e256_n256_dense",
                4096,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m8192_e256_n256_dense",
                8192,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m9216_e256_n256_dense",
                9216,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m16384_e256_n256_dense",
                16384,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
        )
    )
    warmup = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_ITERS", "20"))

    for case in cases:
        fixture = _make_fixture(case, device)
        with torch.no_grad():
            event_stats, host_stats = _bench_preprocess_detail(
                lambda: _run_deepgemm_preprocess_detail_once(fixture),
                warmup=warmup,
                iters=iters,
            )
        name = (
            f"moe_prefill_preprocess/{case.label}"
            f"[m={case.num_tokens},h={case.hidden_size},inter={case.intermediate_size},"
            f"topk={case.top_k},local_e={case.num_local_experts}]"
        )
        _print_preprocess_detail_report(name, event_stats, host_stats)
        del fixture
        _sync()
        getattr(torch, "musa").empty_cache()




def _run_deepgemm_static_cap_preprocess_detail_once(
    fixture: _MoeFixture,
    cap_per_expert: int,
    use_src2dst_count: bool = False,
) -> tuple[dict[str, tuple[object, object]], dict[str, float], object]:
    import triton

    from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.moe_prefill_ops import (
        try_moe_deepgemm_static_cap_quant_scatter_tilelang_musa,
        try_moe_deepgemm_static_cap_src2dst_quant_scatter_tilelang_musa,
    )
    from sglang.srt.layers.moe.ep_moe.kernels import (
        _deepgemm_compact_count_kernel,
        _deepgemm_static_cap_count_src2dst_kernel,
        _get_deepgemm_static_cap_m_indices_musa,
    )

    musa = getattr(torch, "musa")
    events: dict[str, tuple[object, object]] = {}
    host_ms: dict[str, float] = {}

    def timed_event(name: str, fn: Callable[[], object]):
        start = musa.Event(enable_timing=True)
        end = musa.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        events[name] = (start, end)
        return value

    topk_ids = fixture.topk_output.topk_ids
    hidden_states = fixture.hidden_states.as_strided(
        fixture.hidden_states.shape, fixture.hidden_states.stride()
    )
    block_k = fixture.deepgemm_quant_info.block_shape[1]
    num_local_experts = fixture.runner_config.num_local_experts
    top_k = fixture.runner_config.top_k
    num_routes = topk_ids.numel()
    hidden_size = hidden_states.shape[1]
    scale_size = hidden_size // block_k
    all_tokens = int(num_local_experts) * int(cap_per_expert)

    counts = torch.zeros(num_local_experts, device=topk_ids.device, dtype=torch.int32)
    route_ranks = None
    src2dst = torch.empty_like(topk_ids, dtype=torch.int32)
    overflow_flag = torch.zeros((1,), device=topk_ids.device, dtype=torch.int32)

    if use_src2dst_count:
        def count_stage():
            grid = lambda meta: (triton.cdiv(num_routes, meta["BLOCK_SIZE"]),)
            _deepgemm_static_cap_count_src2dst_kernel[grid](
                topk_ids,
                counts,
                src2dst,
                overflow_flag,
                num_routes,
                num_local_experts,
                cap_per_expert,
                BLOCK_SIZE=256,
            )
            return src2dst, overflow_flag

        src2dst, overflow_flag = timed_event("static_count_src2dst", count_stage)
    else:
        route_ranks = torch.empty(num_routes, device=topk_ids.device, dtype=torch.int32)

        def count_stage():
            grid = lambda meta: (triton.cdiv(num_routes, meta["BLOCK_SIZE"]),)
            _deepgemm_compact_count_kernel[grid](
                topk_ids,
                counts,
                route_ranks,
                num_routes,
                num_local_experts,
                BLOCK_SIZE=256,
            )
            return counts, route_ranks

        counts, route_ranks = timed_event("count_kernel", count_stage)

    m_indices = timed_event(
        "static_m_indices",
        lambda: _get_deepgemm_static_cap_m_indices_musa(
            topk_ids.device, num_local_experts, cap_per_expert
        ),
    )

    def alloc_zero_stage():
        compact_input = torch.empty(
            (all_tokens, hidden_size),
            device=hidden_states.device,
            dtype=torch.float8_e4m3fn,
        )
        compact_scale = torch.empty(
            (all_tokens, scale_size), device=hidden_states.device, dtype=torch.float32
        )
        compact_scale.zero_()
        return compact_input, compact_scale, src2dst, overflow_flag

    compact_input, compact_scale, src2dst, overflow_flag = timed_event(
        "alloc_zero", alloc_zero_stage
    )

    def static_quant_scatter_stage():
        if use_src2dst_count:
            ok = try_moe_deepgemm_static_cap_src2dst_quant_scatter_tilelang_musa(
                hidden_states,
                compact_input,
                compact_scale,
                src2dst,
                top_k,
                block_k,
            )
        else:
            ok = try_moe_deepgemm_static_cap_quant_scatter_tilelang_musa(
                hidden_states,
                compact_input,
                compact_scale,
                topk_ids,
                route_ranks,
                src2dst,
                overflow_flag,
                num_local_experts,
                cap_per_expert,
                top_k,
                block_k,
            )
        assert ok
        return ok

    timed_event(
        "static_src2dst_quant_scatter"
        if use_src2dst_count
        else "static_tilelang_quant_scatter",
        static_quant_scatter_stage,
    )
    keepalive = [
        counts,
        route_ranks,
        m_indices,
        compact_input,
        compact_scale,
        src2dst,
        overflow_flag,
    ]
    return events, host_ms, tuple(keepalive)


def test_moe_prefill_deepgemm_static_cap_preprocess_detail_benchmark() -> None:
    device = _require_benchmark_env()
    cases = _case_filter(
        (
            _MoeBenchCase(
                "tp8_m4096_e256_n256_dense",
                4096,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m8192_e256_n256_dense",
                8192,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
        )
    )
    warmup = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_ITERS", "20"))

    for case in cases:
        fixture = _make_fixture(case, device)
        for cap_per_expert in _static_cap_values():
            for use_src2dst_count in (False, True):
                with torch.no_grad():
                    event_stats, host_stats = _bench_preprocess_detail(
                        lambda cap=cap_per_expert, use_src=use_src2dst_count: _run_deepgemm_static_cap_preprocess_detail_once(
                            fixture, cap, use_src2dst_count=use_src
                        ),
                        warmup=warmup,
                        iters=iters,
                    )
                tag = "src2dst_count" if use_src2dst_count else "rank_prefill"
                name = (
                    f"moe_prefill_static_cap_preprocess/{case.label}/{tag}"
                    f"[m={case.num_tokens},h={case.hidden_size},inter={case.intermediate_size},"
                    f"topk={case.top_k},local_e={case.num_local_experts},cap={cap_per_expert},"
                    f"rows={case.num_local_experts * cap_per_expert}]"
                )
                _print_preprocess_detail_report(name, event_stats, host_stats)
        del fixture
        _sync()
        getattr(torch, "musa").empty_cache()


def test_moe_prefill_deepgemm_tilelang_fused_quant_scatter_correctness() -> None:
    device = _require_benchmark_env()
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_compact_preprocess_musa,
    )

    case = _MoeBenchCase(
        label="tilelang_quant_scatter_m1024",
        num_tokens=1024,
        routing="tp8_dense",
        intermediate_size=256,
        num_local_experts=256,
    )
    fixture = _make_fixture(case, device)
    topk_ids = fixture.topk_output.topk_ids
    hidden_states = fixture.hidden_states

    ref = moe_ep_deepgemm_compact_preprocess_musa(
        topk_ids,
        fixture.runner_config.num_local_experts,
        hidden_states,
        fixture.runner_config.top_k,
        fixture.deepgemm_quant_info.block_shape,
        use_tilelang_quant_scatter=False,
    )
    cand = moe_ep_deepgemm_compact_preprocess_musa(
        topk_ids,
        fixture.runner_config.num_local_experts,
        hidden_states,
        fixture.runner_config.top_k,
        fixture.deepgemm_quant_info.block_shape,
        use_tilelang_quant_scatter=True,
    )
    _sync()

    ref_src2dst, ref_input, ref_scale, ref_m_indices, ref_rows = ref
    cand_src2dst, cand_input, cand_scale, cand_m_indices, cand_rows = cand
    assert cand_rows == ref_rows
    torch.testing.assert_close(cand_m_indices.cpu(), ref_m_indices.cpu(), rtol=0, atol=0)

    # The route rank is produced by atomic_add and may differ across two
    # independent preprocess launches. Compare the per-route gathered rows
    # instead of requiring identical compact row ids.
    valid_routes = topk_ids >= 0
    ref_dst = ref_src2dst[valid_routes].to(torch.int64)
    cand_dst = cand_src2dst[valid_routes].to(torch.int64)
    ref_deq = ref_input.float() * ref_scale.repeat_interleave(_BLOCK_SHAPE[1], dim=1)
    cand_deq = cand_input.float() * cand_scale.repeat_interleave(_BLOCK_SHAPE[1], dim=1)
    torch.testing.assert_close(
        cand_deq[cand_dst].cpu(),
        ref_deq[ref_dst].cpu(),
        rtol=5e-2,
        atol=5e-3,
    )
    torch.testing.assert_close(
        cand_m_indices[cand_dst].cpu(),
        topk_ids[valid_routes].cpu(),
        rtol=0,
        atol=0,
    )
    del fixture, ref, cand, ref_deq, cand_deq
    _sync()
    getattr(torch, "musa").empty_cache()


def test_moe_prefill_deepgemm_compact_fixed_bucket_selector(monkeypatch) -> None:
    from sglang.srt.layers.moe.ep_moe import kernels as moe_kernels

    block_m = deep_gemm_wrapper.DEEPGEMM_BLOCK_M
    num_local_experts = 256
    top_k = 6

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(moe_kernels, "_is_musa", True)
        patch_ctx.setattr(
            moe_kernels,
            "_get_deepgemm_compact_fixed_bucket_max_tokens",
            lambda: 16384,
        )
        selector = moe_kernels.select_moe_ep_deepgemm_compact_fixed_bucket_rows_musa

        assert selector(4096, num_local_experts, top_k, block_m) is None
        assert selector(32768, num_local_experts, top_k, block_m) is None
        assert selector(8192, num_local_experts, top_k, block_m) == (
            _compact_worst_case_rows(8192 * top_k, num_local_experts),
            _ceil_div(8192, block_m) * block_m,
        )
        assert selector(9216, num_local_experts, top_k, block_m) == (
            _compact_worst_case_rows(9216 * top_k, num_local_experts),
            _ceil_div(9216, block_m) * block_m,
        )
        assert selector(16384, num_local_experts, top_k, block_m) == (
            _compact_worst_case_rows(16384 * top_k, num_local_experts),
            _ceil_div(16384, block_m) * block_m,
        )
        assert selector(
            32768,
            num_local_experts,
            top_k,
            block_m,
            chunked_prefill_size=32768,
        ) == (
            _compact_worst_case_rows(32768 * top_k, num_local_experts),
            _ceil_div(32768, block_m) * block_m,
        )
        assert (
            selector(
                32769,
                num_local_experts,
                top_k,
                block_m,
                chunked_prefill_size=65536,
            )
            is None
        )
        assert selector(8192, 128, top_k, block_m) is None
        assert selector(8192, num_local_experts, 8, block_m) is None


@pytest.mark.parametrize("num_tokens", [8192, 9216, 16384])
def test_moe_prefill_deepgemm_compact_fixed_bucket_preprocess_correctness(
    num_tokens: int,
    monkeypatch,
) -> None:
    device = _require_benchmark_env()
    from sglang.srt.layers.moe.ep_moe import kernels as moe_kernels

    case = _MoeBenchCase(
        label=f"fixed_bucket_m{num_tokens}",
        num_tokens=num_tokens,
        routing="tp8_dense",
        hidden_size=128,
        intermediate_size=256,
        num_local_experts=256,
    )
    fixture = _make_fixture(case, device)
    topk_ids = fixture.topk_output.topk_ids
    hidden_states = fixture.hidden_states

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(
            moe_kernels,
            "select_moe_ep_deepgemm_compact_fixed_bucket_rows_musa",
            lambda *args, **kwargs: None,
        )
        ref = moe_kernels.moe_ep_deepgemm_compact_preprocess_musa(
            topk_ids,
            fixture.runner_config.num_local_experts,
            hidden_states,
            fixture.runner_config.top_k,
            fixture.deepgemm_quant_info.block_shape,
            use_tilelang_quant_scatter=True,
        )

    cand = moe_kernels.moe_ep_deepgemm_compact_preprocess_musa(
        topk_ids,
        fixture.runner_config.num_local_experts,
        hidden_states,
        fixture.runner_config.top_k,
        fixture.deepgemm_quant_info.block_shape,
        use_tilelang_quant_scatter=True,
    )
    _sync()

    ref_src2dst, ref_input, ref_scale, _, ref_rows = ref
    cand_src2dst, cand_input, cand_scale, cand_m_indices, cand_rows = cand
    expected_bucket = moe_kernels.select_moe_ep_deepgemm_compact_fixed_bucket_rows_musa(
        case.num_tokens,
        fixture.runner_config.num_local_experts,
        fixture.runner_config.top_k,
        deep_gemm_wrapper.DEEPGEMM_BLOCK_M,
    )
    assert expected_bucket is not None
    assert cand_rows == expected_bucket[0]
    assert cand_rows >= ref_rows

    valid_routes = topk_ids >= 0
    ref_dst = ref_src2dst[valid_routes].to(torch.int64)
    cand_dst = cand_src2dst[valid_routes].to(torch.int64)
    ref_deq = ref_input.float() * ref_scale.repeat_interleave(_BLOCK_SHAPE[1], dim=1)
    cand_deq = cand_input.float() * cand_scale.repeat_interleave(_BLOCK_SHAPE[1], dim=1)
    torch.testing.assert_close(
        cand_deq[cand_dst].cpu(),
        ref_deq[ref_dst].cpu(),
        rtol=5e-2,
        atol=5e-3,
    )
    torch.testing.assert_close(
        cand_m_indices[cand_dst].cpu(),
        topk_ids[valid_routes].cpu(),
        rtol=0,
        atol=0,
    )
    if cand_rows > ref_rows:
        assert torch.all(
            cand_m_indices[ref_rows:].cpu()
            == fixture.runner_config.num_local_experts - 1
        ).item()

    del fixture, ref, cand, ref_deq, cand_deq
    _sync()
    getattr(torch, "musa").empty_cache()



def test_moe_prefill_deepgemm_static_cap_quant_scatter_correctness() -> None:
    device = _require_benchmark_env()
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_compact_preprocess_musa,
        moe_ep_deepgemm_static_cap_preprocess_musa,
    )

    case = _MoeBenchCase(
        label="static_cap_quant_scatter_m1024",
        num_tokens=1024,
        routing="tp8_dense",
        intermediate_size=256,
        num_local_experts=256,
    )
    fixture = _make_fixture(case, device)
    topk_ids = fixture.topk_output.topk_ids
    hidden_states = fixture.hidden_states
    cap_per_expert = 128

    ref = moe_ep_deepgemm_compact_preprocess_musa(
        topk_ids,
        fixture.runner_config.num_local_experts,
        hidden_states,
        fixture.runner_config.top_k,
        fixture.deepgemm_quant_info.block_shape,
        use_tilelang_quant_scatter=False,
    )
    ref_src2dst, ref_input, ref_scale, _, _ = ref
    ref_deq = ref_input.float() * ref_scale.repeat_interleave(_BLOCK_SHAPE[1], dim=1)
    valid_routes = topk_ids >= 0
    ref_dst = ref_src2dst[valid_routes].to(torch.int64)

    for use_src2dst_count in (False, True):
        cand = moe_ep_deepgemm_static_cap_preprocess_musa(
            topk_ids,
            fixture.runner_config.num_local_experts,
            hidden_states,
            fixture.runner_config.top_k,
            fixture.deepgemm_quant_info.block_shape,
            cap_per_expert,
            use_src2dst_count=use_src2dst_count,
        )
        _sync()

        cand_src2dst, cand_input, cand_scale, cand_m_indices, cand_rows, overflow_flag = cand
        assert cand_rows == fixture.runner_config.num_local_experts * cap_per_expert
        assert int(overflow_flag.item()) == 0

        cand_dst = cand_src2dst[valid_routes].to(torch.int64)
        # MUSA does not implement advanced indexing directly on fp8 tensors.
        # Dequantize the compact buffers first, then gather the routed rows.
        cand_deq = cand_input.float() * cand_scale.repeat_interleave(_BLOCK_SHAPE[1], dim=1)
        torch.testing.assert_close(
            cand_deq[cand_dst].cpu(), ref_deq[ref_dst].cpu(), rtol=5e-2, atol=5e-3
        )
        torch.testing.assert_close(
            cand_m_indices[cand_dst].cpu(), topk_ids[valid_routes].cpu(), rtol=0, atol=0
        )
        del cand, cand_deq

    del fixture, ref, ref_deq
    _sync()
    getattr(torch, "musa").empty_cache()


def test_moe_prefill_deepgemm_tilelang_post_combine_correctness() -> None:
    device = _require_benchmark_env()
    from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.moe_prefill_ops import (
        try_moe_post_combine_tilelang_musa,
        try_moe_post_combine_src2dst_cached_tilelang_musa,
        try_moe_post_combine_src2dst_tilelang_musa,
    )
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_compact_preprocess_musa,
        post_reorder_triton_kernel,
    )

    case = _MoeBenchCase(
        label="tilelang_post_m1024",
        num_tokens=1024,
        routing="tp8_dense",
        intermediate_size=256,
        num_local_experts=256,
    )
    fixture = _make_fixture(case, device)
    topk_ids = fixture.topk_output.topk_ids
    topk_weights = fixture.topk_output.topk_weights
    src2dst, _, _, _, rows = moe_ep_deepgemm_compact_preprocess_musa(
        topk_ids,
        fixture.runner_config.num_local_experts,
        fixture.hidden_states,
        fixture.runner_config.top_k,
        fixture.deepgemm_quant_info.block_shape,
    )
    torch.manual_seed(123)
    down_output = torch.randn((rows, case.hidden_size), device=device, dtype=torch.bfloat16).contiguous()
    ref = torch.empty_like(fixture.hidden_states)
    cand = torch.empty_like(fixture.hidden_states)
    cand_src2dst = torch.empty_like(fixture.hidden_states)
    cand_cached = torch.empty_like(fixture.hidden_states)
    post_reorder_triton_kernel[(case.num_tokens,)](
        down_output,
        ref,
        src2dst,
        topk_ids,
        topk_weights,
        fixture.runner_config.top_k,
        case.hidden_size,
        BLOCK_SIZE=512,
    )
    assert try_moe_post_combine_tilelang_musa(
        down_output,
        cand,
        src2dst,
        topk_ids,
        topk_weights,
        fixture.runner_config.top_k,
    )
    assert try_moe_post_combine_src2dst_tilelang_musa(
        down_output,
        cand_src2dst,
        src2dst,
        topk_weights,
        fixture.runner_config.top_k,
        block_h=1024,
    )
    assert try_moe_post_combine_src2dst_cached_tilelang_musa(
        down_output,
        cand_cached,
        src2dst,
        topk_weights,
        fixture.runner_config.top_k,
        block_h=1024,
    )
    _sync()
    torch.testing.assert_close(cand.cpu(), ref.cpu(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(cand_src2dst.cpu(), ref.cpu(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(cand_cached.cpu(), ref.cpu(), rtol=5e-2, atol=5e-2)
    del fixture, down_output, ref, cand, cand_src2dst, cand_cached
    _sync()
    getattr(torch, "musa").empty_cache()


def test_moe_prefill_deepgemm_post_combine_tilelang_benchmark() -> None:
    device = _require_benchmark_env()
    from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops.moe_prefill_ops import (
        try_moe_post_combine_tilelang_musa,
        try_moe_post_combine_src2dst_cached_tilelang_musa,
        try_moe_post_combine_src2dst_tilelang_musa,
    )
    from sglang.srt.layers.moe.ep_moe.kernels import (
        moe_ep_deepgemm_compact_preprocess_musa,
        post_reorder_triton_kernel,
    )

    cases = _case_filter(
        (
            _MoeBenchCase(
                "tp8_m4096_e256_n256_dense",
                4096,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
            _MoeBenchCase(
                "tp8_m8192_e256_n256_dense",
                8192,
                "tp8_dense",
                intermediate_size=256,
                num_local_experts=256,
            ),
        )
    )
    warmup = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("SGLANG_DSV4_MUSA_MOE_BENCH_ITERS", "20"))

    for case in cases:
        fixture = _make_fixture(case, device)
        topk_ids = fixture.topk_output.topk_ids
        topk_weights = fixture.topk_output.topk_weights
        src2dst, _, _, _, rows = moe_ep_deepgemm_compact_preprocess_musa(
            topk_ids,
            fixture.runner_config.num_local_experts,
            fixture.hidden_states,
            fixture.runner_config.top_k,
            fixture.deepgemm_quant_info.block_shape,
        )
        down_output = torch.randn((rows, case.hidden_size), device=device, dtype=torch.bfloat16).contiguous()

        def triton_post():
            out = torch.empty_like(fixture.hidden_states)
            post_reorder_triton_kernel[(case.num_tokens,)](
                down_output,
                out,
                src2dst,
                topk_ids,
                topk_weights,
                fixture.runner_config.top_k,
                case.hidden_size,
                BLOCK_SIZE=512,
            )
            return out

        def tilelang_post():
            out = torch.empty_like(fixture.hidden_states)
            ok = try_moe_post_combine_tilelang_musa(
                down_output,
                out,
                src2dst,
                topk_ids,
                topk_weights,
                fixture.runner_config.top_k,
                allow_slow_shape=True,
            )
            assert ok
            return out

        def tilelang_src2dst_post(block_h: int):
            out = torch.empty_like(fixture.hidden_states)
            ok = try_moe_post_combine_src2dst_tilelang_musa(
                down_output,
                out,
                src2dst,
                topk_weights,
                fixture.runner_config.top_k,
                block_h=block_h,
            )
            assert ok
            return out

        def tilelang_cached_post(block_h: int):
            out = torch.empty_like(fixture.hidden_states)
            ok = try_moe_post_combine_src2dst_cached_tilelang_musa(
                down_output,
                out,
                src2dst,
                topk_weights,
                fixture.runner_config.top_k,
                block_h=block_h,
            )
            assert ok
            return out

        with torch.no_grad():
            triton_stats = _bench_device_ms_samples(triton_post, warmup=warmup, iters=iters)
            tilelang_stats = _bench_device_ms_samples(tilelang_post, warmup=warmup, iters=iters)
            src2dst_512_stats = _bench_device_ms_samples(
                lambda: tilelang_src2dst_post(512), warmup=warmup, iters=iters
            )
            src2dst_1024_stats = _bench_device_ms_samples(
                lambda: tilelang_src2dst_post(1024), warmup=warmup, iters=iters
            )
            src2dst_4096_stats = _bench_device_ms_samples(
                lambda: tilelang_src2dst_post(4096), warmup=warmup, iters=iters
            )
            cached_512_stats = _bench_device_ms_samples(
                lambda: tilelang_cached_post(512), warmup=warmup, iters=iters
            )
            cached_1024_stats = _bench_device_ms_samples(
                lambda: tilelang_cached_post(1024), warmup=warmup, iters=iters
            )
            cached_4096_stats = _bench_device_ms_samples(
                lambda: tilelang_cached_post(4096), warmup=warmup, iters=iters
            )
        print(
            f"POST_COMBINE {case.label}: triton={triton_stats.median:.4f}ms "
            f"tilelang={tilelang_stats.median:.4f}ms "
            f"src2dst_b512={src2dst_512_stats.median:.4f}ms "
            f"src2dst_b1024={src2dst_1024_stats.median:.4f}ms "
            f"src2dst_b4096={src2dst_4096_stats.median:.4f}ms "
            f"cached_b512={cached_512_stats.median:.4f}ms "
            f"cached_b1024={cached_1024_stats.median:.4f}ms "
            f"cached_b4096={cached_4096_stats.median:.4f}ms "
            f"legacy_speedup={triton_stats.median / tilelang_stats.median:.3f}x "
            f"src2dst_b512_speedup={triton_stats.median / src2dst_512_stats.median:.3f}x "
            f"src2dst_b1024_speedup={triton_stats.median / src2dst_1024_stats.median:.3f}x "
            f"src2dst_b4096_speedup={triton_stats.median / src2dst_4096_stats.median:.3f}x "
            f"cached_b512_speedup={triton_stats.median / cached_512_stats.median:.3f}x "
            f"cached_b1024_speedup={triton_stats.median / cached_1024_stats.median:.3f}x "
            f"cached_b4096_speedup={triton_stats.median / cached_4096_stats.median:.3f}x"
        )
        del fixture, down_output
        _sync()
        getattr(torch, "musa").empty_cache()
