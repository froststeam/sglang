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


def max_weight_diff(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
) -> float:
    ref, _ = torch.softmax(gating_output.float(), dim=-1).topk(8, dim=-1)
    ref = ref / ref.sum(dim=-1, keepdim=True)
    return (topk_weights - ref).abs().max().item()


def main() -> None:
    ms = [int(v) for v in os.environ.get("MS", "1,2,4,8,16,32,64").split(",")]
    experts = int(os.environ.get("EXPERTS", "256"))
    topk = int(os.environ.get("TOPK", "8"))
    dtype_name = os.environ.get("DTYPE", "bf16").lower()
    num_tests = int(os.environ.get("NUM_TESTS", "8"))
    dtype = (
        torch.float16 if dtype_name in ("fp16", "float16", "half") else torch.bfloat16
    )

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


if __name__ == "__main__":
    main()
