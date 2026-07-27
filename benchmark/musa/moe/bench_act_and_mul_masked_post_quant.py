import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import torch
from mate.testing.utils import bench_kineto

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sync():
    torch.musa.synchronize()


def _activation(x, activation):
    if activation == "silu":
        return torch.nn.functional.silu(x)
    if activation == "gelu":
        return torch.nn.functional.gelu(x, approximate="none")
    if activation == "gelu_tanh":
        return torch.nn.functional.gelu(x, approximate="tanh")
    raise ValueError(activation)


def _ref(input_tensor, output, output_scale, group_size, masked_m, activation):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    rows = []
    scales = []
    size_n = input_tensor.shape[-1] // 2
    for expert in range(input_tensor.shape[0]):
        m = int(masked_m[expert].item())
        if m == 0:
            continue
        gate = input_tensor[expert, :m, :size_n].float()
        up = input_tensor[expert, :m, size_n:]
        gate = _activation(gate, activation).to(input_tensor.dtype)
        y = (gate * up).float()
        y = y.reshape(m, size_n // group_size, group_size)
        scale = torch.clamp(y.abs().amax(dim=-1), min=1e-10) / fp8_max
        yq = torch.clamp(y / scale.unsqueeze(-1), -fp8_max, fp8_max).to(
            torch.float8_e4m3fn
        )
        rows.append((expert, y.reshape(m, size_n), yq.reshape(m, size_n)))
        scales.append((expert, scale))

    max_abs_deq = 0.0
    max_rel_deq = 0.0
    max_abs_scale = 0.0
    mismatch = 0
    checked = 0
    for expert, y, yq in rows:
        m = y.shape[0]
        got_q = output[expert, :m]
        got_s = output_scale[expert, :m]
        ref_s = dict(scales)[expert]
        got_deq = got_q.float() * got_s.repeat_interleave(group_size, dim=-1)
        ref_deq = yq.float() * ref_s.repeat_interleave(group_size, dim=-1)
        abs_err = (got_deq - ref_deq).abs()
        rel_err = abs_err / torch.clamp(ref_deq.abs(), min=1e-6)
        max_abs_deq = max(max_abs_deq, float(abs_err.max().item()))
        max_rel_deq = max(max_rel_deq, float(rel_err.max().item()))
        max_abs_scale = max(max_abs_scale, float((got_s - ref_s).abs().max().item()))
        mismatch += int((got_q.float() != yq.float()).sum().item())
        checked += got_q.numel()
    return max_abs_deq, max_rel_deq, max_abs_scale, mismatch, checked


def _event_time_ms(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    _sync()
    if hasattr(torch.musa, "Event"):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeat
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / repeat


def _mate_kineto_time_ms(fn, repeat, kernel_name):
    seconds = bench_kineto(
        fn,
        kernel_names=kernel_name,
        num_tests=repeat,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if seconds <= 0:
        raise RuntimeError(
            f"MATE bench_kineto did not capture kernel {kernel_name!r}. "
            "The profiler may be reporting only musaLaunchKernel; rerun with "
            "a MUSA/Torch profiler setup that exposes device kernel symbols, "
            "or use --event only for a non-kineto smoke benchmark."
        )
    return seconds * 1000.0


def _print_markdown(rows):
    print(
        "| act | rows | hidden | experts | latency_us | bandwidth_GB/s | max_abs_deq | max_rel_deq | max_abs_scale | fp8_mismatch | checked |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['activation']} | {row['rows']} | {row['hidden']} | "
            f"{row['experts']} | {row['latency_us']:.3f} | "
            f"{row['bandwidth_gbs']:.3f} | {row['max_abs_deq']:.3g} | "
            f"{row['max_rel_deq']:.3g} | {row['max_abs_scale']:.3g} | "
            f"{row['fp8_mismatch']} | {row['checked']} |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[
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
        ],
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument(
        "--capacity",
        type=int,
        default=0,
        help="Allocated rows per expert; defaults to the valid rows per expert.",
    )
    parser.add_argument(
        "--expected-m",
        type=int,
        default=0,
        help="Host-side expected rows/expert hint; <=128 selects compact decode variants.",
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Require byte-equal valid outputs versus the regular padded-grid kernel.",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--activations", nargs="+", default=["silu", "gelu", "gelu_tanh"]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument(
        "--event",
        action="store_true",
        help="Use simple MUSA event timing instead of MATE bench_kineto.",
    )
    parser.add_argument(
        "--kernel-name",
        default="act_and_mul_masked_post_quant_kernel",
        help="Kernel name passed to MATE bench_kineto.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short rows sweep for quick accuracy/performance checks.",
    )
    args = parser.parse_args()
    if args.quick:
        args.rows = [128, 512, 1024, 8192, 32768]

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("torch.musa is not available")
    torch.musa.set_device(0)
    torch.manual_seed(args.seed)

    root = REPO_ROOT / "python"

    def _pkg(name, relpath):
        mod = sys.modules.setdefault(name, types.ModuleType(name))
        mod.__path__ = [str(root / relpath)]
        return mod

    def _load_module(name, relpath):
        path = root / relpath
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    _pkg("sglang", "sglang")
    _pkg("sglang.jit_kernel", "sglang/jit_kernel")
    _pkg("sglang.srt", "sglang/srt")
    _pkg("sglang.srt.layers", "sglang/srt/layers")
    _pkg("sglang.srt.layers.moe", "sglang/srt/layers/moe")
    _pkg("sglang.srt.layers.moe.ep_moe", "sglang/srt/layers/moe/ep_moe")
    _pkg("sglang.srt.hardware_backend", "sglang/srt/hardware_backend")
    _pkg("sglang.srt.hardware_backend.musa", "sglang/srt/hardware_backend/musa")
    _pkg(
        "sglang.srt.hardware_backend.musa.jit_kernel",
        "sglang/srt/hardware_backend/musa/jit_kernel",
    )
    _pkg(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc",
        "sglang/srt/hardware_backend/musa/jit_kernel/csrc",
    )

    fake_jit_utils = types.ModuleType("sglang.jit_kernel.utils")
    fake_jit_utils.cache_once = lambda fn: fn
    sys.modules["sglang.jit_kernel.utils"] = fake_jit_utils

    fake_custom_op = types.ModuleType("sglang.srt.utils.custom_op")

    def _register_custom_op(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def deco(fn):
            return fn

        return deco

    fake_custom_op.register_custom_op = _register_custom_op

    fake_utils = types.ModuleType("sglang.srt.utils")
    fake_utils.__path__ = [str(root / "sglang/srt/utils")]
    fake_utils.ceil_div = lambda a, b: (a + b - 1) // b
    fake_utils.is_cuda = lambda: False
    fake_utils.is_musa = lambda: True
    fake_utils.custom_op = fake_custom_op
    sys.modules.setdefault(
        "sglang.srt.layers.deep_gemm_wrapper", types.ModuleType("deep_gemm_wrapper")
    )
    sys.modules["sglang.srt.utils"] = fake_utils
    sys.modules["sglang.srt.utils.custom_op"] = fake_custom_op
    _load_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit",
        "sglang/srt/hardware_backend/musa/jit_kernel/csrc/jit.py",
    )
    _load_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc.quant",
        "sglang/srt/hardware_backend/musa/jit_kernel/csrc/quant.py",
    )

    fake_fp8 = types.ModuleType("sglang.srt.layers.quantization.fp8_kernel")
    fake_fp8.sglang_per_token_group_quant_fp8 = None
    sys.modules.setdefault(
        "sglang.srt.layers.quantization", types.ModuleType("quantization")
    )
    sys.modules["sglang.srt.layers.quantization.fp8_kernel"] = fake_fp8
    moe_kernels = _load_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc.moe",
        "sglang/srt/hardware_backend/musa/jit_kernel/csrc/moe.py",
    )
    act_and_mul_masked_post_quant_fwd = moe_kernels.act_and_mul_masked_post_quant_fwd

    results = []
    if not args.markdown:
        print(
            "activation,rows,hidden,experts,latency_us,bandwidth_GBps,"
            "max_abs_deq,max_rel_deq,max_abs_scale,fp8_mismatch,checked"
        )
    for rows in args.rows:
        if rows % args.experts != 0:
            raise ValueError("rows must be divisible by experts")
        valid_m = rows // args.experts
        m = args.capacity or valid_m
        if valid_m > m:
            raise ValueError("valid rows per expert cannot exceed --capacity")
        input_tensor = torch.randn(
            (args.experts, m, args.hidden * 2),
            device="musa",
            dtype=torch.bfloat16,
        )
        masked_m = torch.full(
            (args.experts,), valid_m, device="musa", dtype=torch.int32
        )
        output = torch.empty(
            (args.experts, m, args.hidden), device="musa", dtype=torch.float8_e4m3fn
        )
        output_scale = torch.empty(
            (args.experts, m, args.hidden // args.group_size),
            device="musa",
            dtype=torch.float32,
        )
        valid_bytes = rows * (
            args.hidden * 2 * 2 + args.hidden + (args.hidden // args.group_size) * 4
        )
        for activation in args.activations:

            def run():
                act_and_mul_masked_post_quant_fwd(
                    input_tensor,
                    output,
                    output_scale,
                    args.group_size,
                    masked_m,
                    scale_ue8m0=False,
                    activation=activation,
                    expected_m=args.expected_m,
                )

            ms = (
                _event_time_ms(run, args.warmup, args.repeat)
                if args.event
                else _mate_kineto_time_ms(run, args.repeat, args.kernel_name)
            )
            _sync()
            if args.check_baseline:
                baseline_output = torch.empty_like(output)
                baseline_scale = torch.empty_like(output_scale)
                act_and_mul_masked_post_quant_fwd(
                    input_tensor,
                    baseline_output,
                    baseline_scale,
                    args.group_size,
                    masked_m,
                    scale_ue8m0=False,
                    activation=activation,
                    expected_m=0,
                )
                _sync()
                for expert in range(args.experts):
                    valid = int(masked_m[expert].item())
                    torch.testing.assert_close(
                        output[expert, :valid].float(),
                        baseline_output[expert, :valid].float(),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        output_scale[expert, :valid],
                        baseline_scale[expert, :valid],
                        rtol=0,
                        atol=0,
                    )
            max_abs, max_rel, max_abs_scale, mismatch, checked = _ref(
                input_tensor,
                output,
                output_scale,
                args.group_size,
                masked_m,
                activation,
            )
            bandwidth = valid_bytes / (ms / 1000.0) / 1e9
            result = {
                "activation": activation,
                "rows": rows,
                "hidden": args.hidden,
                "experts": args.experts,
                "latency_us": ms * 1000.0,
                "bandwidth_gbs": bandwidth,
                "max_abs_deq": max_abs,
                "max_rel_deq": max_rel,
                "max_abs_scale": max_abs_scale,
                "fp8_mismatch": mismatch,
                "checked": checked,
            }
            results.append(result)
            if not args.markdown:
                print(
                    f"{activation},{rows},{args.hidden},{args.experts},"
                    f"{result['latency_us']:.3f},{bandwidth:.3f},"
                    f"{max_abs:.6g},{max_rel:.6g},{max_abs_scale:.6g},"
                    f"{mismatch},{checked}",
                    flush=True,
                )
    if args.markdown:
        _print_markdown(results)


if __name__ == "__main__":
    main()
