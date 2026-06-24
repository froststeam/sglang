from __future__ import annotations

import os
from typing import Callable

import torch
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto
from sgl_kernel import topk_softmax as sgl_kernel_topk_softmax

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.moe import (
    topk_softmax as sglang_musa_topk_softmax,
)
from sglang.srt.layers.moe.topk import fused_topk_torch_native
from sglang.srt.utils import get_compiler_backend

compiled_fused_topk_torch_native = torch.compile(
    fused_topk_torch_native,
    dynamic=True,
    backend=get_compiler_backend(),
)

DEFAULT_MS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536"
DEFAULT_SHARED_CASES = "128:8,128:10,256:8,256:10,512:8,512:10"


def bench_one(
    fn: Callable[[], None],
    kernel_names: str | tuple[str, ...],
    *,
    multi: bool = False,
    num_tests: int = 8,
) -> float:
    ret = bench_kineto(
        fn,
        kernel_names=kernel_names,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=multi,
    )
    return sum(ret) if isinstance(ret, tuple) else ret


def parse_ms() -> list[int]:
    return [int(v) for v in os.environ.get("MS", DEFAULT_MS).split(",") if v]


def parse_shared_cases() -> list[tuple[int, int]]:
    cases = []
    for item in os.environ.get("CASES", DEFAULT_SHARED_CASES).split(","):
        if not item:
            continue
        experts, topk = item.split(":")
        cases.append((int(experts), int(topk)))
    return cases


def get_dtype() -> torch.dtype:
    dtype_name = os.environ.get("DTYPE", "bf16").lower()
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_name in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits // 8


def topk_shared_bytes_lower_bound(
    num_tokens: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
) -> int:
    output_topk = topk + 1
    return (
        num_tokens * num_experts * dtype_nbytes(dtype)
        + num_tokens * dtype_nbytes(dtype)
        + num_tokens * output_topk * (4 + 4)
    )


def branch_name(num_tokens: int) -> str:
    onewarp_max_tokens = int(
        os.environ.get("SGLANG_TOPK_SHARED_ONEWARP_MAX_TOKENS", "128")
    )
    if num_tokens <= onewarp_max_tokens:
        return "onewarp"
    if num_tokens <= 8192:
        return "warp4"
    return "halfwarp"


def max_weight_diff(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
) -> float:
    ref, _ = torch.softmax(gating_output.float(), dim=-1).topk(8, dim=-1)
    ref = ref / ref.sum(dim=-1, keepdim=True)
    return (topk_weights - ref).abs().max().item()


def run_shared_matrix() -> None:
    ms = parse_ms()
    cases = parse_shared_cases()
    dtype = get_dtype()
    num_tests = int(os.environ.get("NUM_TESTS", "8"))

    print("device", torch.musa.current_device(), torch.musa.get_device_name())
    print(
        "fused shared-expert topk matrix: "
        f"cases={cases}, dtype={dtype}, num_tests={num_tests}"
    )
    print("Each cell is `time_us / lower-bound GB/s`.")

    results: dict[tuple[int, int], dict[int, tuple[float, float]]] = {}
    for experts, topk in cases:
        case_results = {}
        for m in ms:
            torch.manual_seed(20260622 + experts * 100000 + topk * 1000 + m)
            gating_output = torch.randn((m, experts), device="musa", dtype=dtype)
            shared_gate = torch.randn((m, 1), device="musa", dtype=dtype)
            topk_weights = torch.empty(
                (m, topk + 1), device="musa", dtype=torch.float32
            )
            topk_ids = torch.empty((m, topk + 1), device="musa", dtype=torch.int32)

            def run_csrc() -> None:
                sglang_musa_topk_softmax(
                    topk_weights,
                    topk_ids,
                    gating_output,
                    True,
                    shared_expert_gate_output=shared_gate,
                    num_fused_shared_experts=1,
                )

            run_csrc()
            torch.musa.synchronize()
            seconds = bench_one(run_csrc, "shared1", num_tests=num_tests)
            time_us = seconds * 1e6
            bytes_lb = topk_shared_bytes_lower_bound(m, experts, topk, dtype)
            bandwidth = bytes_lb / time_us / 1000.0
            case_results[m] = (time_us, bandwidth)
        results[(experts, topk)] = case_results

    header = ["M", "branch"]
    for experts, topk in cases:
        header.append(f"E{experts} K{topk}")
    print(
        " | ".join(
            f"{col:>16}" if i >= 2 else f"{col:>8}" for i, col in enumerate(header)
        )
    )
    print("-+-".join("-" * (16 if i >= 2 else 8) for i in range(len(header))))
    for m in ms:
        row = [f"{m:>8}", f"{branch_name(m):>8}"]
        for case in cases:
            time_us, bandwidth = results[case][m]
            row.append(f"{time_us:7.3f} / {bandwidth:6.2f}")
        print(" | ".join(row))


def run_compare() -> None:
    ms = parse_ms()
    experts = int(os.environ.get("EXPERTS", "256"))
    topk = int(os.environ.get("TOPK", "8"))
    num_tests = int(os.environ.get("NUM_TESTS", "8"))
    dtype = get_dtype()

    if topk != 8:
        raise ValueError("This bench currently compares the Qwen path with topk=8.")
    if experts != 256:
        raise ValueError("This bench currently targets the Qwen path with experts=256.")

    print("device", torch.musa.current_device(), torch.musa.get_device_name())
    print(
        "topk softmax compare: "
        f"experts={experts}, topk={topk}, dtype={dtype}, num_tests={num_tests}"
    )
    print(
        "M\tsglang_csrc_us\tsgl_kernel_us\tfused_topk_native_compile_us\t"
        "csrc_vs_sglkernel\tcsrc_vs_compile\tmaxdiff"
    )

    for m in ms:
        torch.manual_seed(20260605 + m)
        gating_output = torch.randn((m, experts), device="musa", dtype=dtype)
        csrc_w = torch.empty((m, topk), device="musa", dtype=torch.float32)
        csrc_ids = torch.empty((m, topk), device="musa", dtype=torch.int32)
        sgl_w = torch.empty_like(csrc_w)
        sgl_ids = torch.empty_like(csrc_ids)
        hidden_states = torch.empty((m, 1), device="musa", dtype=dtype)

        def run_csrc() -> None:
            sglang_musa_topk_softmax(csrc_w, csrc_ids, gating_output, True)

        def run_sgl_kernel() -> None:
            sgl_kernel_topk_softmax(sgl_w, sgl_ids, gating_output, True)

        def run_compile() -> None:
            compiled_fused_topk_torch_native(
                hidden_states,
                gating_output,
                topk,
                True,
                None,
                "softmax",
            )

        # Compile and correctness warmup.
        run_csrc()
        run_sgl_kernel()
        run_compile()
        torch.musa.synchronize()

        csrc_s = bench_one(
            run_csrc,
            "topk",
            num_tests=num_tests,
        )
        sgl_s = bench_one(
            run_sgl_kernel,
            "topk",
            num_tests=num_tests,
        )
        compile_s = bench_one(
            run_compile,
            (
                "TopKSmallNSmallK",
                "triton_red_fused__softmax",
                "triton_red_fused_div_sum",
                "triton_per_fused__softmax",
                "triton_per_fused_copy__div_sum",
                "triton_poi_fused__to_copy_copy",
            ),
            multi=True,
            num_tests=num_tests,
        )
        diff = max_weight_diff(gating_output, csrc_w)
        print(
            f"{m}\t{csrc_s * 1e6:.2f}\t{sgl_s * 1e6:.2f}\t"
            f"{compile_s * 1e6:.2f}\t{sgl_s / csrc_s:.2f}x\t"
            f"{compile_s / csrc_s:.2f}x\t{diff:.3g}"
        )


def main() -> None:
    mode = os.environ.get("MODE", "shared-matrix").lower()
    if mode in ("shared", "shared-matrix", "matrix"):
        run_shared_matrix()
    elif mode == "compare":
        run_compare()
    else:
        raise ValueError(f"Unknown MODE={mode!r}; expected compare or shared-matrix")


if __name__ == "__main__":
    main()
