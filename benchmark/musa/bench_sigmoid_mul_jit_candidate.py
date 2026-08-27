#!/usr/bin/env python3
"""Correctness and Kineto benchmark for the MUSA JIT sigmoid-mul kernel."""

import torch
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

from sglang.srt.hardware_backend.musa.jit_kernel import sigmoid_mul


def main() -> None:
    for batch in (1, 2, 4, 5, 8):
        gate = torch.randn((batch, 2048), device="musa", dtype=torch.bfloat16)
        value = torch.randn((batch, 2048), device="musa", dtype=torch.bfloat16)
        reference = torch.sigmoid(gate.float()) * value.float()
        output = sigmoid_mul(gate, value)
        torch.musa.synchronize()
        torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=1e-2)
        latency = bench_kineto(
            lambda: sigmoid_mul(gate, value),
            kernel_names="sigmoid_mul",
            num_tests=200,
            suppress_kineto_output=True,
            trace_path=f"/tmp/sigmoid_mul_jit_b{batch}.trace.json",
        )
        print(
            {
                "batch": batch,
                "latency_us": float(latency) * 1e6,
                "max_abs": float((output.float() - reference).abs().max()),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
