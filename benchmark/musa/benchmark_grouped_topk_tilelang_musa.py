"""Fair graph benchmark for the MUSA grouped top-k implementations.

Each fixed case warms up both implementations, captures the same number of
operator calls into separate MUSA graphs, replays both graphs repeatedly in
alternating order, and reports arithmetic mean latency per operator call.

Example:

    PYTHONPATH=python python benchmark/musa/benchmark_grouped_topk_tilelang_musa.py \
        --tokens 1,33,257,1024,3072 --dtype bfloat16 \
        --graph-calls 50 --replays 20 --rounds 7 \
        --output-json /tmp/grouped_topk_tilelang_musa.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from sglang.srt.layers.moe.topk import (
    grouped_topk_gpu,
    grouped_topk_tilelang_musa_impl,
)
from sglang.srt.utils import is_musa

DEFAULT_TOKENS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
GroupedTopKOutputs = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class GroupedTopKConfig:
    num_experts: int
    num_expert_group: int
    topk_group: int
    routed_topk: int
    num_fused_shared_experts: int = 0
    renormalize: bool = True
    routed_scaling_factor: Optional[float] = None
    apply_routed_scaling_factor_on_output: bool = False

    def __str__(self) -> str:
        return (
            f"experts={self.num_experts},groups={self.num_expert_group},"
            f"topk_group={self.topk_group},routed_topk={self.routed_topk},"
            f"shared={self.num_fused_shared_experts},"
            f"renormalize={self.renormalize},"
            f"routed_scale={self.routed_scaling_factor},"
            "apply_routed_scale_on_output="
            f"{self.apply_routed_scaling_factor_on_output}"
        )


DEFAULT_CONFIGS = (
    GroupedTopKConfig(
        64, 1, 1, 6, 0, False, 1.0
    ),  # Deepseek v2 lite
    GroupedTopKConfig(160, 8, 3, 6, 0, False, 16.0),  # Deepseek v2
    GroupedTopKConfig(256, 8, 4, 8, 0, True, 2.5),  # Deepseek v3/r1
    GroupedTopKConfig(
        256, 8, 4, 8, 1, True, 2.5
    ),  # Deepseek v3/r1, fused shared expert
    GroupedTopKConfig(256, 1, 1, 8, 0, True, 2.5),  # JoyAI LLM Flash
    GroupedTopKConfig(
        256, 1, 1, 8, 1, True, 2.5
    ),  # JoyAI LLM Flash, fused shared expert
    GroupedTopKConfig(48, 3, 2, 5, 0, False, 1.0),  # generic
    GroupedTopKConfig(96, 6, 2, 7, 0, False, 1.0),  # generic
    GroupedTopKConfig(128, 4, 3, 9, 0, False, 1.0),  # generic
    GroupedTopKConfig(128, 32, 3, 9, 0, False, 1.0),  # generic
    GroupedTopKConfig(128, 64, 4, 7, 0, False, 1.0),  # generic
)


@dataclass(frozen=True)
class Timing:
    mean_us: float
    median_us: float
    min_us: float
    p95_us: float


@dataclass(frozen=True)
class BenchmarkResult:
    num_experts: int
    num_expert_group: int
    topk_group: int
    routed_topk: int
    output_topk: int
    num_fused_shared_experts: int
    tokens: int
    dtype: str
    renormalize: bool
    routed_scaling_factor: Optional[float]
    apply_routed_scaling_factor_on_output: bool
    capture_warmup: int
    replay_warmup: int
    graph_calls: int
    replays: int
    rounds: int
    torch_mean_us: float
    torch_median_us: float
    torch_min_us: float
    torch_p95_us: float
    tilelang_mean_us: float
    tilelang_median_us: float
    tilelang_min_us: float
    tilelang_p95_us: float
    speedup: float
    max_abs_error: float
    mismatched_expert_ids: int


class _GraphRunner:
    """Replay one graph and keep every captured output allocation alive."""

    def __init__(
        self,
        graph: object,
        captured_outputs: tuple[GroupedTopKOutputs, ...],
    ):
        self.graph = graph
        self.captured_outputs = captured_outputs

    @property
    def calls_per_replay(self) -> int:
        return len(self.captured_outputs)

    @property
    def output(self) -> GroupedTopKOutputs:
        return self.captured_outputs[-1]

    def replay(self) -> None:
        self.graph.replay()


def _capture_graph(
    fn: Callable[[], GroupedTopKOutputs],
    *,
    capture_warmup: int,
    graph_calls: int,
) -> _GraphRunner:
    """Warm up ``fn`` and capture ``graph_calls`` identical invocations."""

    for _ in range(capture_warmup):
        fn()
    torch.musa.synchronize()

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        captured_outputs = tuple(fn() for _ in range(graph_calls))
    torch.musa.synchronize()
    return _GraphRunner(graph, captured_outputs)


def _parse_positive_ints(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def _parse_tokens(value: str) -> tuple[int, ...]:
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
        if start < 1 or end < start:
            raise argparse.ArgumentTypeError(
                "token range must satisfy 1 <= start <= end"
            )
        return tuple(range(start, end + 1))
    return _parse_positive_ints(value, name="tokens")


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"{name} must be true or false")


def _parse_optional_float(value: str, *, name: str) -> Optional[float]:
    if value.strip().lower() in {"none", "null"}:
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be a float or none") from error
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"{name} must be finite")
    return parsed


def _validate_config(config: GroupedTopKConfig) -> GroupedTopKConfig:
    if config.num_experts < 1 or config.num_expert_group < 1:
        raise argparse.ArgumentTypeError("expert and group counts must be positive")
    if config.num_experts % config.num_expert_group != 0:
        raise argparse.ArgumentTypeError(
            "NUM_EXPERTS must be divisible by NUM_EXPERT_GROUP"
        )
    if not 1 <= config.topk_group <= config.num_expert_group:
        raise argparse.ArgumentTypeError(
            "TOPK_GROUP must satisfy 1 <= TOPK_GROUP <= NUM_EXPERT_GROUP"
        )
    if config.num_fused_shared_experts not in (0, 1):
        raise argparse.ArgumentTypeError(
            "NUM_FUSED_SHARED_EXPERTS must be zero or one"
        )
    capacity = config.num_experts // config.num_expert_group * config.topk_group
    output_topk = config.routed_topk + config.num_fused_shared_experts
    if config.routed_topk < 1 or output_topk > capacity:
        raise argparse.ArgumentTypeError(
            "ROUTED_TOPK plus NUM_FUSED_SHARED_EXPERTS must not exceed the "
            "selected-group capacity"
        )
    if config.routed_scaling_factor == 0:
        raise argparse.ArgumentTypeError("ROUTED_SCALING_FACTOR must be non-zero")
    if (
        config.num_fused_shared_experts > 0
        or config.apply_routed_scaling_factor_on_output
    ) and config.routed_scaling_factor is None:
        raise argparse.ArgumentTypeError(
            "ROUTED_SCALING_FACTOR is required for a fused shared expert or "
            "output scaling"
        )
    return config


def _parse_config(value: str) -> GroupedTopKConfig:
    fields = tuple(item.strip() for item in value.split(","))
    if len(fields) not in (7, 8):
        raise argparse.ArgumentTypeError(
            "config must contain NUM_EXPERTS,NUM_EXPERT_GROUP,TOPK_GROUP,"
            "ROUTED_TOPK,NUM_FUSED_SHARED_EXPERTS,RENORMALIZE,"
            "ROUTED_SCALING_FACTOR[,APPLY_ROUTED_SCALING_FACTOR_ON_OUTPUT]"
        )
    try:
        integer_fields = tuple(int(item) for item in fields[:5])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "the first five config fields must be integers"
        ) from error
    config = GroupedTopKConfig(
        *integer_fields,
        renormalize=_parse_bool(fields[5], name="RENORMALIZE"),
        routed_scaling_factor=_parse_optional_float(
            fields[6], name="ROUTED_SCALING_FACTOR"
        ),
        apply_routed_scaling_factor_on_output=(
            _parse_bool(fields[7], name="APPLY_ROUTED_SCALING_FACTOR_ON_OUTPUT")
            if len(fields) == 8
            else False
        ),
    )
    return _validate_config(config)


def _validate_default_configs() -> None:
    for config in DEFAULT_CONFIGS:
        try:
            _validate_config(config)
        except argparse.ArgumentTypeError as error:
            raise ValueError(
                f"invalid DEFAULT_CONFIGS entry ({config}): {error}"
            ) from error


def _make_unique_logits(
    num_tokens: int, num_experts: int, dtype: torch.dtype
) -> torch.Tensor:
    values = torch.linspace(-8.0, 8.0, num_experts, device="musa")
    expert = torch.arange(num_experts, device="musa", dtype=torch.int64)
    row = torch.arange(num_tokens, device="musa", dtype=torch.int64).unsqueeze(1)
    permutation = (expert.unsqueeze(0) * 131 + row * 17) % num_experts
    return values[permutation].to(dtype)


def _percentile_95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _time_graph_us(runner: _GraphRunner, *, replays: int) -> float:
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        runner.replay()
    end.record()
    torch.musa.synchronize()
    total_calls = replays * runner.calls_per_replay
    return float(start.elapsed_time(end)) * 1000.0 / total_calls


def _summarize_timing(samples: list[float]) -> Timing:
    return Timing(
        mean_us=statistics.fmean(samples),
        median_us=statistics.median(samples),
        min_us=min(samples),
        p95_us=_percentile_95(samples),
    )


def _bench_graph_pair_us(
    torch_runner: _GraphRunner,
    tilelang_runner: _GraphRunner,
    *,
    replay_warmup: int,
    replays: int,
    rounds: int,
) -> tuple[Timing, Timing]:
    """Benchmark both graphs with identical work and balanced measurement order."""

    assert torch_runner.calls_per_replay == tilelang_runner.calls_per_replay
    for warmup_idx in range(replay_warmup):
        if warmup_idx % 2 == 0:
            torch_runner.replay()
            tilelang_runner.replay()
        else:
            tilelang_runner.replay()
            torch_runner.replay()
    torch.musa.synchronize()

    torch_samples: list[float] = []
    tilelang_samples: list[float] = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            torch_samples.append(_time_graph_us(torch_runner, replays=replays))
            tilelang_samples.append(_time_graph_us(tilelang_runner, replays=replays))
        else:
            tilelang_samples.append(_time_graph_us(tilelang_runner, replays=replays))
            torch_samples.append(_time_graph_us(torch_runner, replays=replays))
    return _summarize_timing(torch_samples), _summarize_timing(tilelang_samples)


def _accuracy(
    actual: GroupedTopKOutputs,
    expected: GroupedTopKOutputs,
    *,
    num_experts: int,
    num_fused_shared_experts: int,
) -> tuple[float, int]:
    actual_weights, actual_ids = actual
    expected_weights, expected_ids = expected
    width = num_experts + num_fused_shared_experts
    actual_dense = torch.zeros((actual_weights.shape[0], width), device="musa")
    expected_dense = torch.zeros_like(actual_dense)
    actual_dense.scatter_(1, actual_ids.long(), actual_weights)
    expected_dense.scatter_(1, expected_ids.long(), expected_weights)
    max_abs_error = float((actual_dense - expected_dense).abs().max().item())
    mismatched_ids = int(
        (
            torch.sort(actual_ids, dim=-1).values
            != torch.sort(expected_ids, dim=-1).values
        )
        .sum()
        .item()
    )
    return max_abs_error, mismatched_ids


def _run_case(
    config: GroupedTopKConfig,
    tokens: int,
    dtype: torch.dtype,
    *,
    capture_warmup: int,
    replay_warmup: int,
    graph_calls: int,
    replays: int,
    rounds: int,
) -> BenchmarkResult:
    hidden_states = torch.empty((tokens, 1), device="musa", dtype=dtype)
    gating_output = _make_unique_logits(tokens, config.num_experts, dtype)
    output_topk = config.routed_topk + config.num_fused_shared_experts

    common_args = (
        hidden_states,
        gating_output,
        output_topk,
        config.renormalize,
        config.num_expert_group,
        config.topk_group,
        config.num_fused_shared_experts,
        config.routed_scaling_factor,
        config.apply_routed_scaling_factor_on_output,
    )

    def torch_impl() -> GroupedTopKOutputs:
        return grouped_topk_gpu(*common_args)

    def tilelang_impl() -> GroupedTopKOutputs:
        return grouped_topk_tilelang_musa_impl(*common_args)

    torch_runner = _capture_graph(
        torch_impl,
        capture_warmup=capture_warmup,
        graph_calls=graph_calls,
    )
    tilelang_runner = _capture_graph(
        tilelang_impl,
        capture_warmup=capture_warmup,
        graph_calls=graph_calls,
    )

    torch_runner.replay()
    tilelang_runner.replay()
    torch.musa.synchronize()
    max_abs_error, mismatched_ids = _accuracy(
        tilelang_runner.output,
        torch_runner.output,
        num_experts=config.num_experts,
        num_fused_shared_experts=config.num_fused_shared_experts,
    )
    if mismatched_ids != 0 or max_abs_error > 2e-5:
        raise AssertionError(
            f"config=({config}) tokens={tokens} "
            f"failed graph accuracy: "
            f"max_abs_error={max_abs_error:.6g}, mismatched_ids={mismatched_ids}"
        )

    torch_timing, tilelang_timing = _bench_graph_pair_us(
        torch_runner,
        tilelang_runner,
        replay_warmup=replay_warmup,
        replays=replays,
        rounds=rounds,
    )
    return BenchmarkResult(
        num_experts=config.num_experts,
        num_expert_group=config.num_expert_group,
        topk_group=config.topk_group,
        routed_topk=config.routed_topk,
        output_topk=output_topk,
        num_fused_shared_experts=config.num_fused_shared_experts,
        tokens=tokens,
        dtype=str(dtype).removeprefix("torch."),
        renormalize=config.renormalize,
        routed_scaling_factor=config.routed_scaling_factor,
        apply_routed_scaling_factor_on_output=(
            config.apply_routed_scaling_factor_on_output
        ),
        capture_warmup=capture_warmup,
        replay_warmup=replay_warmup,
        graph_calls=graph_calls,
        replays=replays,
        rounds=rounds,
        torch_mean_us=torch_timing.mean_us,
        torch_median_us=torch_timing.median_us,
        torch_min_us=torch_timing.min_us,
        torch_p95_us=torch_timing.p95_us,
        tilelang_mean_us=tilelang_timing.mean_us,
        tilelang_median_us=tilelang_timing.median_us,
        tilelang_min_us=tilelang_timing.min_us,
        tilelang_p95_us=tilelang_timing.p95_us,
        speedup=torch_timing.mean_us / tilelang_timing.mean_us,
        max_abs_error=max_abs_error,
        mismatched_expert_ids=mismatched_ids,
    )


def _print_markdown(results: list[BenchmarkResult]) -> None:
    print(
        "| E/G/Gk/Kr | Shared | Renorm | Routed scale | Apply scale | Tokens | "
        "DType | grouped_topk_gpu mean us | TileLang mean us | Speedup | "
        "TileLang min/p95 us | Max error | ID mismatches |"
    )
    print(
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|"
    )
    for result in results:
        topology = (
            f"{result.num_experts}/{result.num_expert_group}/"
            f"{result.topk_group}/{result.routed_topk}"
        )
        print(
            f"| {topology} | {result.num_fused_shared_experts} | "
            f"{result.renormalize} | {result.routed_scaling_factor} | "
            f"{result.apply_routed_scaling_factor_on_output} | {result.tokens} | "
            f"{result.dtype} | {result.torch_mean_us:.3f} | "
            f"{result.tilelang_mean_us:.3f} | {result.speedup:.2f}x | "
            f"{result.tilelang_min_us:.3f}/{result.tilelang_p95_us:.3f} | "
            f"{result.max_abs_error:.3g} | {result.mismatched_expert_ids} |"
        )


def _write_json(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        type=_parse_config,
        metavar=(
            "NUM_EXPERTS,NUM_EXPERT_GROUP,TOPK_GROUP,ROUTED_TOPK,"
            "NUM_FUSED_SHARED_EXPERTS,RENORMALIZE,ROUTED_SCALING_FACTOR"
            "[,APPLY_ROUTED_SCALING_FACTOR_ON_OUTPUT]"
        ),
        help=(
            "complete grouped top-k configuration; repeat for multiple "
            "configurations. Use none for a missing routed scaling factor"
        ),
    )
    parser.add_argument(
        "--tokens",
        type=_parse_tokens,
        default=DEFAULT_TOKENS,
        help="comma-separated counts or an inclusive range such as 1:3072",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--capture-warmup", type=int, default=20)
    parser.add_argument("--replay-warmup", type=int, default=5)
    parser.add_argument("--graph-calls", type=int, default=50)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--min-speedup", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not is_musa() or not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("This benchmark requires an available MUSA device")
    for name in (
        "capture_warmup",
        "replay_warmup",
        "graph_calls",
        "replays",
        "rounds",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    _validate_default_configs()
    dtype = getattr(torch, args.dtype)
    configs = tuple(args.configs) if args.configs else DEFAULT_CONFIGS
    print(
        "Fair MUSA graph comparison: "
        f"graph_calls={args.graph_calls}, replays={args.replays}, "
        f"rounds={args.rounds}, configs={len(configs)}",
        flush=True,
    )

    results: list[BenchmarkResult] = []
    for config in configs:
        for tokens in args.tokens:
            result = _run_case(
                config,
                tokens,
                dtype,
                capture_warmup=args.capture_warmup,
                replay_warmup=args.replay_warmup,
                graph_calls=args.graph_calls,
                replays=args.replays,
                rounds=args.rounds,
            )
            results.append(result)
            print(
                f"completed config=({config}) tokens={tokens}: "
                f"{result.speedup:.2f}x",
                flush=True,
            )
            if args.output_json is not None:
                _write_json(args.output_json, results)

    _print_markdown(results)

    if args.min_speedup > 0:
        failures = [result for result in results if result.speedup < args.min_speedup]
        if failures:
            failed_cases = ", ".join(
                f"({result.num_experts},{result.num_expert_group},"
                f"{result.topk_group},{result.routed_topk})/"
                f"shared{result.num_fused_shared_experts}/"
                f"tokens{result.tokens}={result.speedup:.2f}x"
                for result in failures
            )
            raise AssertionError(
                f"speedup below required {args.min_speedup:.2f}x: {failed_cases}"
            )


if __name__ == "__main__":
    main()
