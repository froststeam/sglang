import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["contig", "triton", "triton_runner"], required=True
    )
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--m-list",
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
    )
    return parser.parse_args()


def sync(torch):
    torch.get_device_module().synchronize()


def bench_one(fn, torch, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync(torch)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(torch)
    return (time.perf_counter() - start) * 1e6 / iters


def main():
    args = parse_args()
    if args.mode in ("contig", "triton_runner"):
        os.environ.setdefault("SGLANG_CI_DISABLE_MOE_FUSED_FUNC", "1")

    import torch

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.server_args import set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(
        SimpleNamespace(
            enable_deterministic_inference=False,
            enable_fused_moe_sum_all_reduce=False,
        )
    )
    torch.set_default_device("musa")
    torch.manual_seed(0)
    torch.get_device_module().manual_seed_all(0)

    hidden_size = 2048
    intermediate_size = 512
    shard_intermediate_size = 2 * intermediate_size
    num_experts = 256
    topk = 8
    block_shape = [128, 128]
    m_list = [int(x) for x in args.m_list.split(",") if x]

    w13_bf16 = torch.randn(
        num_experts, shard_intermediate_size, hidden_size, dtype=torch.bfloat16
    )
    w2_bf16 = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16
    )
    w13 = w13_bf16.to(torch.float8_e4m3fn)
    w2 = w2_bf16.to(torch.float8_e4m3fn)
    del w13_bf16, w2_bf16

    w13_scale = torch.rand(
        num_experts,
        (shard_intermediate_size + block_shape[0] - 1) // block_shape[0],
        (hidden_size + block_shape[1] - 1) // block_shape[1],
        dtype=torch.float32,
    )
    w2_scale = torch.rand(
        num_experts,
        (hidden_size + block_shape[0] - 1) // block_shape[0],
        (intermediate_size + block_shape[1] - 1) // block_shape[1],
        dtype=torch.float32,
    )

    runner_config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        top_k=topk,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        inplace=False,
    )

    if args.mode == "contig":
        runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, runner_config)
        runner.use_contiguous_gemm = True
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            use_fp8=True,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )
        variant = f"contig_block{args.block_m}"
    elif args.mode == "triton_runner":
        runner = MoeRunner(MoeRunnerBackend.TRITON, runner_config)
        quant_info = TritonMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            use_fp8_w8a8=True,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )
        variant = "triton_runner"
    else:
        runner = None
        quant_info = None
        variant = "triton_fused"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{variant}.csv"
    rows = []

    for m in m_list:
        hidden = torch.randn(m, hidden_size, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, num_experts, (m, topk), dtype=torch.int32)
        topk_weights = torch.rand(m, topk, dtype=torch.float32)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        router_logits = torch.empty(m, num_experts, dtype=torch.float32)
        topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)

        def run():
            x = hidden.clone()
            if args.mode == "triton":
                return fused_experts(
                    hidden_states=x,
                    w1=w13,
                    w2=w2,
                    topk_output=topk_output,
                    moe_runner_config=runner_config,
                    use_fp8_w8a8=True,
                    w1_scale=w13_scale,
                    w2_scale=w2_scale,
                    block_shape=block_shape,
                )
            dispatch = StandardDispatchOutput(x, None, topk_output)
            return runner.run(dispatch, quant_info).hidden_states

        try:
            latency_us = bench_one(run, torch, args.warmup, args.iters)
            status = "ok"
            error = ""
        except Exception as exc:
            sync(torch)
            latency_us = float("nan")
            status = "error"
            error = repr(exc)
        row = {
            "variant": variant,
            "mode": args.mode,
            "block_m": args.block_m if args.mode == "contig" else "",
            "m": m,
            "latency_us": latency_us,
            "tokens_per_s": (m * 1e6 / latency_us) if latency_us == latency_us else "",
            "status": status,
            "error": error,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "model": "Qwen3.5-35B-A3B-FP8 synthetic TP1 MoE layer",
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "topk": topk,
        "moe_intermediate_size": intermediate_size,
        "block_shape": block_shape,
        "mode": args.mode,
        "block_m": args.block_m,
        "warmup": args.warmup,
        "iters": args.iters,
        "env_block_m": os.environ.get("SGLANG_DEEP_GEMM_BLOCK_M"),
    }
    (args.out_dir / f"{variant}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
