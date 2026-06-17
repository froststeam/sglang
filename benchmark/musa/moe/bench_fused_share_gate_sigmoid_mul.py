from __future__ import annotations

import os

import torch
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit


def _parse_flags(value: str) -> tuple[str, ...]:
    return tuple(flag for flag in value.split() if flag)


def _fused_module():
    return load_musa_jit(
        os.environ.get("JIT_NAME", "sglang_musa_fused_share_gate_sigmoid_mul_bench"),
        ("moe/fused_share_gate_sigmoid_mul.mu",),
        extra_musa_cflags=_parse_flags(os.environ.get("FSG_EXTRA_FLAGS", "")),
    )


def bench(fn, warmup: int = 20, reps: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.musa.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def bench_mate(fn, num_tests: int) -> float:
    ret = bench_kineto(
        fn,
        kernel_names=(
            "fused_share_gate_sigmoid_mul_hdim_bf16_tpr_kernel",
            "fused_share_gate_sigmoid_mul_hdim_kernel",
            "fused_share_gate_sigmoid_mul_kernel",
        ),
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=True,
    )
    return sum(ret) * 1e6


def main() -> None:
    ms = [int(v) for v in os.environ.get("MS", "1,8,32,64,512,4096,16384").split(",")]
    hidden_dim = int(os.environ.get("HIDDEN", "3072"))
    warmup = int(os.environ.get("WARMUP", "20"))
    reps = int(os.environ.get("REPS", "0"))
    small_reps = int(os.environ.get("SMALL_REPS", "200"))
    large_reps = int(os.environ.get("LARGE_REPS", "50"))
    num_tests = int(os.environ.get("NUM_TESTS", "8"))
    timing = os.environ.get("TIMING", "mate").lower()
    dtype = torch.bfloat16
    module = _fused_module()
    print("device", torch.musa.current_device(), torch.musa.get_device_name())
    print(f"fused_share_gate_sigmoid_mul dtype={dtype} hidden={hidden_dim}")
    print(
        "jit_name",
        os.environ.get("JIT_NAME", "sglang_musa_fused_share_gate_sigmoid_mul_bench"),
    )
    print("extra_flags", os.environ.get("FSG_EXTRA_FLAGS", ""))
    print(
        f"timing={timing} num_tests={num_tests} warmup={warmup} "
        f"small_reps={small_reps} large_reps={large_reps} reps_override={reps}"
    )
    print("M\tlatency_us\tBW_lb_TBps\tmaxdiff")
    for m in ms:
        torch.manual_seed(20260605 + m)
        hidden = torch.randn((m, hidden_dim), device="musa", dtype=dtype)
        gate = torch.randn((1, hidden_dim), device="musa", dtype=dtype)
        shared = torch.randn((m, hidden_dim), device="musa", dtype=dtype)
        out = torch.empty_like(shared)

        def run() -> None:
            module.sgl_musa_fused_share_gate_sigmoid_mul(out, hidden, gate, shared)

        run_reps = reps if reps > 0 else (small_reps if m <= 512 else large_reps)
        if timing == "event":
            us = bench(run, warmup=warmup, reps=run_reps)
        elif timing == "mate":
            run()
            torch.musa.synchronize()
            us = bench_mate(run, num_tests=num_tests)
        else:
            raise ValueError(f"Unsupported TIMING={timing!r}; use 'mate' or 'event'.")
        ref = torch.sigmoid(hidden.float() @ gate.float().t()) * shared.float()
        bytes_lb = m * hidden_dim * 2 * 3
        bw = bytes_lb / (us * 1e-6) / 1e12 if us > 0 else 0.0
        print(f"{m}\t{us:.2f}\t{bw:.2f}\t{(out.float() - ref).abs().max().item():.3g}")


if __name__ == "__main__":
    main()
