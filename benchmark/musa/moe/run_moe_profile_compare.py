import argparse
import csv
import json
import math
import os
import subprocess
import sys
import sysconfig
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = SCRIPT_DIR / "moe_profile_compare"
DEFAULT_M_LIST = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536"
DEFAULT_BLOCK_M_LIST = "128,256"
SGLANG_MOE_CONFIG_ENV = "SGLANG_MOE_CONFIG_DIR"
TORCHADA_MOE_CONFIG_RELATIVE_DIRS = (
    # torchada's import side effect currently points SGLANG_MOE_CONFIG_DIR here.
    Path("torchada/triton/autotune/fused_moe"),
    # Keep a fallback for configs produced by some tune_moe.py versions.
    Path("torchada/triton"),
)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    topk: int
    tp_size: int
    ep_size: int = 1
    use_fp8_w8a8: bool = True
    block_shape: tuple[int, int] | None = (128, 128)
    source: str = ""

    @property
    def intermediate_size_per_partition(self) -> int:
        if self.moe_intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"{self.name}: moe_intermediate_size={self.moe_intermediate_size} "
                f"must be divisible by tp_size={self.tp_size}"
            )
        return self.moe_intermediate_size // self.tp_size

    @property
    def file_stem(self) -> str:
        return self.name.lower().replace(".", "_").replace("-", "_")

    @property
    def num_local_experts(self) -> int:
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"{self.name}: num_experts={self.num_experts} must be divisible "
                f"by ep_size={self.ep_size}"
            )
        return self.num_experts // self.ep_size

    @property
    def parallel_suffix(self) -> str:
        suffix = f"tp{self.tp_size}"
        if self.ep_size > 1:
            suffix += f"_ep{self.ep_size}"
        return suffix


BASE_MODEL_PROFILES = {
    "qwen3.5-35b-a3b-fp8": ModelProfile(
        name="Qwen3.5-35B-A3B-FP8",
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
        topk=8,
        tp_size=1,
        source="Qwen/Qwen3.5-35B-A3B-FP8 config",
    ),
    "qwen3.5-35b-a3b-bf16": ModelProfile(
        name="Qwen3.5-35B-A3B-BF16",
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
        topk=8,
        tp_size=1,
        use_fp8_w8a8=False,
        block_shape=None,
        source="Qwen/Qwen3.5-35B-A3B BF16 synthetic profile",
    ),
    "qwen3.5-397b-a17b-fp8": ModelProfile(
        name="Qwen3.5-397B-A17B-FP8",
        hidden_size=4096,
        moe_intermediate_size=1024,
        num_experts=512,
        topk=10,
        tp_size=8,
        source="Qwen/Qwen3.5-397B-A17B-FP8 config",
    ),
    "gemma4-26b-a4b": ModelProfile(
        name="Gemma4-26B-A4B",
        hidden_size=2816,
        moe_intermediate_size=704,
        num_experts=128,
        topk=8,
        tp_size=2,
        use_fp8_w8a8=False,
        block_shape=None,
        source="google/gemma-4-26B-A4B-it config",
    ),
    "qwen3-next-80b-a3b-fp8": ModelProfile(
        name="Qwen3-Next-80B-A3B-FP8",
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=512,
        topk=10,
        tp_size=4,
        source="Qwen/Qwen3-Next-80B-A3B-Instruct config",
    ),
    "qwen3.5-122b-a10b-fp8-bf16": ModelProfile(
        name="Qwen3.5-122B-A10B-FP8-BF16",
        hidden_size=3072,
        moe_intermediate_size=1024,
        num_experts=256,
        topk=8,
        tp_size=8,
        source="Qwen/Qwen3.5-122B-A10B-FP8 config",
    ),
    "qwen3.6-35b-a3b-fp8-bf16": ModelProfile(
        name="Qwen3.6-35B-A3B-FP8-BF16",
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
        topk=8,
        tp_size=1,
        source="Qwen/Qwen3.6-35B-A3B-FP8 config",
    ),
    "joyai-llm-flash": ModelProfile(
        name="JoyAI-LLM-Flash",
        hidden_size=2048,
        moe_intermediate_size=768,
        num_experts=256,
        topk=8,
        tp_size=2,
        use_fp8_w8a8=False,
        block_shape=None,
        source="jdopensource/JoyAI-LLM-Flash-Base config",
    ),
}


MODEL_PROFILES = {
    "qwen3.5-35b-a3b-fp8-tp1": BASE_MODEL_PROFILES["qwen3.5-35b-a3b-fp8"],
}
for tp_size, ep_size in ((4, 1), (4, 4)):
    key = f"qwen3.5-35b-a3b-fp8-tp{tp_size}"
    if ep_size > 1:
        key += f"-ep{ep_size}"
    MODEL_PROFILES[key] = replace(
        BASE_MODEL_PROFILES["qwen3.5-35b-a3b-fp8"],
        tp_size=tp_size,
        ep_size=ep_size,
    )
for tp_size, ep_size in ((1, 1), (8, 1), (8, 8)):
    key = f"qwen3.5-35b-a3b-bf16-tp{tp_size}"
    if ep_size > 1:
        key += f"-ep{ep_size}"
    MODEL_PROFILES[key] = replace(
        BASE_MODEL_PROFILES["qwen3.5-35b-a3b-bf16"],
        tp_size=tp_size,
        ep_size=ep_size,
    )
for base_key, tp_sizes in (
    ("qwen3.5-397b-a17b-fp8", (8,)),
    ("gemma4-26b-a4b", (2, 4, 8)),
    ("qwen3-next-80b-a3b-fp8", (4,)),
    ("qwen3.5-122b-a10b-fp8-bf16", (8,)),
    ("qwen3.6-35b-a3b-fp8-bf16", (1, 2, 4, 8)),
    ("joyai-llm-flash", (2, 4, 8)),
):
    base = BASE_MODEL_PROFILES[base_key]
    for tp_size in tp_sizes:
        MODEL_PROFILES[f"{base_key}-tp{tp_size}"] = replace(base, tp_size=tp_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--ep-size", type=int, default=None)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["contig", "triton", "triton_runner"],
        default=["contig", "triton_runner"],
        help="contig means Contig DeepGEMM for TP-only and Masked DeepGEMM for TP+EP.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--m-list", default=DEFAULT_M_LIST)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--block-ms",
        default=DEFAULT_BLOCK_M_LIST,
        help="Comma-separated TP-only Contig DeepGEMM block_m values to benchmark.",
    )
    parser.add_argument(
        "--block-m",
        type=int,
        default=None,
        help="Benchmark a single DeepGEMM block_m value. Overrides --block-ms.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--_run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_mode", choices=["contig", "triton", "triton_runner"], help=argparse.SUPPRESS
    )
    parser.add_argument("--_model-key", help=argparse.SUPPRESS)
    return parser.parse_args()


def print_model_profiles() -> None:
    for key in sorted(MODEL_PROFILES):
        profile = MODEL_PROFILES[key]
        print(
            json.dumps(
                {
                    "model": key,
                    "name": profile.name,
                    "hidden_size": profile.hidden_size,
                    "moe_intermediate_size": profile.moe_intermediate_size,
                    "intermediate_size_per_partition": profile.intermediate_size_per_partition,
                    "tp_size": profile.tp_size,
                    "ep_size": profile.ep_size,
                    "num_experts": profile.num_experts,
                    "num_local_experts": profile.num_local_experts,
                    "topk": profile.topk,
                    "use_fp8_w8a8": profile.use_fp8_w8a8,
                    "source": profile.source,
                },
                ensure_ascii=False,
            )
        )


def make_model_key(model: str, tp_size: int, ep_size: int) -> str:
    return f"{model}-tp{tp_size}" + (f"-ep{ep_size}" if ep_size > 1 else "")


def resolve_model_key(model: str, tp_size: int | None, ep_size: int | None) -> str:
    if model in MODEL_PROFILES and tp_size is None and ep_size is None:
        return model

    profile = BASE_MODEL_PROFILES.get(model) or MODEL_PROFILES.get(model)
    if profile is None:
        raise ValueError(f"Unknown model '{model}'. Use --list-models first.")

    tp = tp_size if tp_size is not None else profile.tp_size
    ep = ep_size if ep_size is not None else profile.ep_size
    candidates = [
        f"{model}-tp{tp}" + (f"-ep{ep}" if ep > 1 else ""),
        f"{profile.name.lower().replace('.', '_').replace('-', '_')}-tp{tp}"
        + (f"-ep{ep}" if ep > 1 else ""),
    ]
    for candidate in candidates:
        if candidate in MODEL_PROFILES:
            return candidate

    if model in BASE_MODEL_PROFILES:
        candidate = make_model_key(model, tp, ep)
        MODEL_PROFILES[candidate] = replace(profile, tp_size=tp, ep_size=ep)
        return candidate

    raise ValueError(
        f"No profile for model='{model}', tp_size={tp}, ep_size={ep}. "
        "Add it to MODEL_PROFILES or choose one from --list-models."
    )


def resolve_model_keys(
    model: str, tp_size: int | None, ep_size: int | None
) -> list[str]:
    if ep_size is not None or tp_size is None or tp_size <= 1:
        return [resolve_model_key(model, tp_size, ep_size)]

    keys = [
        resolve_model_key(model, tp_size, 1),
        resolve_model_key(model, tp_size, tp_size),
    ]
    return list(dict.fromkeys(keys))


def select_models(args) -> list[str]:
    if args.model is not None and args.models is not None:
        raise ValueError("Use either --model or --models, not both.")
    requested = [args.model] if args.model is not None else args.models
    if requested is None:
        if args.tp_size is not None or args.ep_size is not None:
            raise ValueError("--tp-size/--ep-size require --model or --models.")
        return sorted(MODEL_PROFILES)
    selected = []
    for model in requested:
        selected.extend(resolve_model_keys(model, args.tp_size, args.ep_size))
    return list(dict.fromkeys(selected))


def is_masked_deepgemm(model_key: str, mode: str) -> bool:
    return mode == "contig" and MODEL_PROFILES[model_key].ep_size > 1


def display_block_m(mode: str, block_m: int | None) -> int | str:
    return block_m if mode == "contig" and block_m is not None else ""


def expected_variant(model_key: str, mode: str, block_m: int | None) -> str:
    profile = MODEL_PROFILES[model_key]
    if mode == "contig":
        if is_masked_deepgemm(model_key, mode):
            return f"{profile.file_stem}_{profile.parallel_suffix}_masked_deepgemm"
        return f"{profile.file_stem}_{profile.parallel_suffix}_contig_deepgemm_block{block_m}"
    if mode == "triton_runner":
        return f"{profile.file_stem}_{profile.parallel_suffix}_triton_runner"
    return f"{profile.file_stem}_{profile.parallel_suffix}_triton_fused"


def csv_complete(csv_path: Path, expected_m: list[int]) -> bool:
    if not csv_path.exists():
        return False
    try:
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    return sorted(int(row["m"]) for row in rows if row.get("m")) == expected_m


def parse_block_ms(args) -> list[int]:
    if args.block_m is not None:
        return [args.block_m]
    block_ms = sorted({int(x) for x in args.block_ms.split(",") if x})
    if not block_ms:
        raise ValueError("--block-ms must contain at least one integer")
    return block_ms


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


def run_one_benchmark(args) -> None:
    profile = MODEL_PROFILES[args._model_key]
    masked_deepgemm = is_masked_deepgemm(args._model_key, args._mode)
    if args._mode == "contig" and not masked_deepgemm:
        if args.block_m is None:
            raise ValueError("TP-only contig DeepGEMM mode requires --block-m")
        os.environ["SGLANG_DEEP_GEMM_BLOCK_M"] = str(args.block_m)
    if args._mode == "contig":
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

    hidden_size = profile.hidden_size
    intermediate_size = profile.intermediate_size_per_partition
    shard_intermediate_size = 2 * intermediate_size
    num_experts = profile.num_experts
    num_local_experts = profile.num_local_experts
    topk = profile.topk
    use_fp8_w8a8 = profile.use_fp8_w8a8
    block_shape = list(profile.block_shape) if profile.block_shape is not None else None
    m_list = [int(x) for x in args.m_list.split(",") if x]

    w13 = torch.randn(
        num_local_experts, shard_intermediate_size, hidden_size, dtype=torch.bfloat16
    )
    w2 = torch.randn(
        num_local_experts, hidden_size, intermediate_size, dtype=torch.bfloat16
    )

    if use_fp8_w8a8:
        if block_shape is None:
            raise ValueError(f"{args._model_key}: FP8 profile requires block_shape")
        w13 = w13.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)
        w13_scale = torch.rand(
            num_local_experts,
            (shard_intermediate_size + block_shape[0] - 1) // block_shape[0],
            (hidden_size + block_shape[1] - 1) // block_shape[1],
            dtype=torch.float32,
        )
        w2_scale = torch.rand(
            num_local_experts,
            (hidden_size + block_shape[0] - 1) // block_shape[0],
            (intermediate_size + block_shape[1] - 1) // block_shape[1],
            dtype=torch.float32,
        )
    else:
        w13_scale = None
        w2_scale = None

    runner_config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        top_k=topk,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        inplace=False,
    )

    if args._mode == "contig":
        runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, runner_config)
        runner.use_contiguous_gemm = not masked_deepgemm
        if not masked_deepgemm:
            runner.contiguous_gemm_block_m = args.block_m
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            use_fp8=use_fp8_w8a8,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )
    elif args._mode == "triton_runner":
        runner = MoeRunner(MoeRunnerBackend.TRITON, runner_config)
        quant_info = TritonMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            use_fp8_w8a8=use_fp8_w8a8,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )
    else:
        runner = None
        quant_info = None

    variant = expected_variant(args._model_key, args._mode, args.block_m)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{variant}.csv"
    rows = []

    for m in m_list:
        hidden = torch.randn(m, hidden_size, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, num_local_experts, (m, topk), dtype=torch.int32)
        topk_weights = torch.rand(m, topk, dtype=torch.float32)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        router_logits = torch.empty(m, num_experts, dtype=torch.float32)
        topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)

        def run():
            x = hidden.clone()
            if args._mode == "triton":
                return fused_experts(
                    hidden_states=x,
                    w1=w13,
                    w2=w2,
                    topk_output=topk_output,
                    moe_runner_config=runner_config,
                    use_fp8_w8a8=use_fp8_w8a8,
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
            "model": profile.name,
            "tp_size": profile.tp_size,
            "ep_size": profile.ep_size,
            "num_local_experts": num_local_experts,
            "mode": args._mode,
            "kernel_path": (
                "masked_deepgemm"
                if args._mode == "contig" and profile.ep_size > 1
                else "contig_deepgemm" if args._mode == "contig" else args._mode
            ),
            "block_m": (
                args.block_m if args._mode == "contig" and not masked_deepgemm else ""
            ),
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
        "model": f"{profile.name} synthetic TP{profile.tp_size} MoE layer",
        "model_key": args._model_key,
        "source": profile.source,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "num_local_experts": num_local_experts,
        "topk": topk,
        "moe_intermediate_size": profile.moe_intermediate_size,
        "intermediate_size_per_partition": intermediate_size,
        "tp_size": profile.tp_size,
        "ep_size": profile.ep_size,
        "use_fp8_w8a8": use_fp8_w8a8,
        "block_shape": block_shape,
        "mode": args._mode,
        "block_m": args.block_m if not masked_deepgemm else "",
        "warmup": args.warmup,
        "iters": args.iters,
        "env_block_m": os.environ.get("SGLANG_DEEP_GEMM_BLOCK_M", ""),
    }
    (raw_dir / f"{variant}.json").write_text(json.dumps(meta, indent=2) + "\n")


def read_rows(raw_dir: Path):
    rows = []
    errors = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        meta_path = csv_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                row.update(
                    {
                        "csv": str(csv_path),
                        "model_key": meta.get("model_key", ""),
                        "model_name": meta.get("model", row.get("model", "")),
                        "use_fp8_w8a8": meta.get("use_fp8_w8a8", ""),
                        "ep_size": meta.get("ep_size", row.get("ep_size", 1)),
                    }
                )
                if row.get("status") == "ok":
                    row["m"] = int(row["m"])
                    row["latency_us"] = float(row["latency_us"])
                    row["tokens_per_s"] = float(row["tokens_per_s"])
                    rows.append(row)
                else:
                    errors.append(row)
    return rows, errors


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_torchada_moe_config_dir() -> Path | None:
    candidates = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            candidates.append(Path(path))
    candidates.append(
        Path(
            f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"
        )
    )

    seen = set()
    for site_dir in candidates:
        if site_dir in seen:
            continue
        seen.add(site_dir)
        for relative_dir in TORCHADA_MOE_CONFIG_RELATIVE_DIRS:
            config_dir = site_dir / relative_dir
            if (config_dir / "configs").is_dir():
                return config_dir
    return None


def display_mode(row: dict) -> str:
    mode = row["mode"]
    if mode == "triton_runner":
        return "Triton MoE"
    if mode == "contig" and int(row.get("ep_size") or 1) > 1:
        return "Masked DeepGEMM"
    if mode == "contig":
        return f"Contig DeepGEMM block_m={row.get('block_m')}"
    if mode == "triton":
        return "Triton fused"
    return mode


def compare_mode(row: dict) -> str:
    mode = row["mode"]
    if mode == "triton_runner":
        return "Triton MoE"
    if mode == "contig" and int(row.get("ep_size") or 1) > 1:
        return "Masked DeepGEMM"
    if mode == "contig":
        return "Contig DeepGEMM"
    if mode == "triton":
        return "Triton fused"
    return mode


def keep_fastest(rows_by_mode: dict, row: dict) -> None:
    mode = compare_mode(row)
    old = rows_by_mode.get(mode)
    if old is None or row["latency_us"] < old["latency_us"]:
        rows_by_mode[mode] = row


def build_speedup_rows(rows: list[dict]):
    by_key = defaultdict(dict)
    for row in rows:
        keep_fastest(by_key[(row["model_key"], row["m"])], row)

    speedups = []
    for (model_key, m), modes in sorted(by_key.items()):
        deep_gemm_key = (
            "Masked DeepGEMM" if "Masked DeepGEMM" in modes else "Contig DeepGEMM"
        )
        triton_key = "Triton MoE" if "Triton MoE" in modes else "Triton fused"
        if deep_gemm_key not in modes or triton_key not in modes:
            continue
        deep_gemm = modes[deep_gemm_key]
        triton = modes[triton_key]
        triton_latency = triton["latency_us"]
        speedups.append(
            {
                "model_key": model_key,
                "model": deep_gemm["model_name"],
                "m": m,
                "deep_gemm_mode": deep_gemm_key,
                "deep_gemm_block_m": (
                    ""
                    if deep_gemm_key == "Masked DeepGEMM"
                    else deep_gemm.get("block_m", "")
                ),
                "triton_mode": triton_key,
                "deep_gemm_latency_us": deep_gemm["latency_us"],
                "triton_latency_us": triton_latency,
                "triton_vs_deep_gemm_speedup": (
                    deep_gemm["latency_us"] / triton_latency
                    if triton_latency and not math.isnan(triton_latency)
                    else float("nan")
                ),
            }
        )
    return speedups


def build_threshold_rows(rows: list[dict]):
    by_key = defaultdict(dict)
    for row in rows:
        keep_fastest(by_key[(row["model_key"], row["m"])], row)

    by_model = defaultdict(list)
    for (model_key, m), modes in sorted(by_key.items()):
        deep_gemm_key = (
            "Masked DeepGEMM" if "Masked DeepGEMM" in modes else "Contig DeepGEMM"
        )
        triton_key = "Triton MoE" if "Triton MoE" in modes else "Triton fused"
        if deep_gemm_key not in modes or triton_key not in modes:
            continue
        deep_gemm = modes[deep_gemm_key]
        triton = modes[triton_key]
        by_model[model_key].append(
            {
                "model_key": model_key,
                "model": deep_gemm["model_name"],
                "m": m,
                "triton_mode": triton_key,
                "deep_gemm_mode": deep_gemm_key,
                "deep_gemm_block_m": (
                    ""
                    if deep_gemm_key == "Masked DeepGEMM"
                    else deep_gemm.get("block_m", "")
                ),
                "triton_latency_us": triton["latency_us"],
                "deep_gemm_latency_us": deep_gemm["latency_us"],
                "winner": (
                    triton_key
                    if triton["latency_us"] <= deep_gemm["latency_us"]
                    else deep_gemm_key
                ),
            }
        )

    recommendations = []
    for model_key, model_rows in sorted(by_model.items()):
        model_rows = sorted(model_rows, key=lambda row: row["m"])
        triton_wins = [
            row["m"] for row in model_rows if row["winner"] == row["triton_mode"]
        ]
        deep_gemm_wins = [
            row["m"] for row in model_rows if row["winner"] == row["deep_gemm_mode"]
        ]
        if not triton_wins:
            threshold = 0
            note = "DeepGEMM wins at every measured token count."
            policy = "DeepGEMM for all measured token counts."
        elif not deep_gemm_wins:
            threshold = max(triton_wins)
            note = "Triton wins at every measured token count; retest above this range before using as a final production threshold."
            policy = "Triton for all measured token counts."
        else:
            threshold = 0
            for row in model_rows:
                if row["winner"] != row["triton_mode"]:
                    break
                threshold = row["m"]
            note = "Use Triton at or below this token count, DeepGEMM above it."
            policy = f"Triton <= {threshold}, DeepGEMM > {threshold}."

        recommendations.append(
            {
                "model_key": model_key,
                "model": model_rows[0]["model"],
                "triton_mode": model_rows[0]["triton_mode"],
                "deep_gemm_mode": model_rows[0]["deep_gemm_mode"],
                "recommended_threshold": threshold,
                "recommended_policy": policy,
                "deep_gemm_block_ms": (
                    ""
                    if model_rows[0]["deep_gemm_mode"] == "Masked DeepGEMM"
                    else ",".join(
                        sorted(
                            {
                                str(row["deep_gemm_block_m"])
                                for row in model_rows
                                if row["m"] > threshold
                            }
                        )
                    )
                ),
                "min_measured_m": min(row["m"] for row in model_rows),
                "max_measured_m": max(row["m"] for row in model_rows),
                "note": note,
            }
        )
    return recommendations


def plot_model(
    model_key: str, rows: list[dict], plot_dir: Path, recommendation: dict | None
):
    import matplotlib.pyplot as plt

    by_mode = defaultdict(list)
    for row in rows:
        by_mode[display_mode(row)].append(row)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for mode, mode_rows in sorted(by_mode.items()):
        mode_rows = sorted(mode_rows, key=lambda x: x["m"])
        ax.plot(
            [row["m"] for row in mode_rows],
            [row["latency_us"] for row in mode_rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=mode,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("tokens (M)")
    ax.set_ylabel("latency (us)")
    title = rows[0]["model_name"] or model_key
    dtype = "FP8 W8A8" if rows[0]["use_fp8_w8a8"] is True else "BF16"
    ax.set_title(f"{title} - {dtype}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    if recommendation and recommendation["recommended_threshold"] > 0:
        threshold = recommendation["recommended_threshold"]
        ax.axvline(
            threshold,
            color="tab:red",
            linestyle=":",
            linewidth=1.6,
            label=f"threshold={threshold}",
        )
    ax.legend()
    fig.tight_layout()
    output_path = plot_dir / f"{model_key.replace('.', '_').replace('-', '_')}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_outputs(out_dir: Path) -> dict:
    raw_dir = out_dir / "raw"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows, errors = read_rows(raw_dir)
    if not rows:
        raise RuntimeError(f"No successful CSV rows found under {raw_dir}")

    threshold_rows = build_threshold_rows(rows)
    threshold_by_model = {row["model_key"]: row for row in threshold_rows}

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_key"]].append(row)

    plot_paths = [
        plot_model(model_key, model_rows, plot_dir, threshold_by_model.get(model_key))
        for model_key, model_rows in sorted(by_model.items())
    ]

    write_csv(
        out_dir / "all_results.csv",
        sorted(rows, key=lambda r: (r["model_key"], r["mode"], int(r["m"]))),
        [
            "model_key",
            "model_name",
            "mode",
            "ep_size",
            "block_m",
            "m",
            "latency_us",
            "tokens_per_s",
            "use_fp8_w8a8",
            "csv",
        ],
    )

    speedup_rows = build_speedup_rows(rows)
    write_csv(
        out_dir / "triton_vs_deep_gemm_speedup.csv",
        speedup_rows,
        [
            "model_key",
            "model",
            "m",
            "deep_gemm_mode",
            "deep_gemm_block_m",
            "triton_mode",
            "deep_gemm_latency_us",
            "triton_latency_us",
            "triton_vs_deep_gemm_speedup",
        ],
    )

    if threshold_rows:
        write_csv(
            out_dir / "recommended_thresholds.csv",
            threshold_rows,
            [
                "model_key",
                "model",
                "triton_mode",
                "deep_gemm_mode",
                "recommended_threshold",
                "recommended_policy",
                "deep_gemm_block_ms",
                "min_measured_m",
                "max_measured_m",
                "note",
            ],
        )
        (out_dir / "recommended_thresholds.json").write_text(
            json.dumps(threshold_rows, indent=2, ensure_ascii=False) + "\n"
        )

    if errors:
        write_csv(
            out_dir / "errors.csv",
            errors,
            ["model_key", "model_name", "mode", "m", "status", "error", "csv"],
        )

    return {
        "plots": [str(path) for path in plot_paths],
        "rows": len(rows),
        "errors": len(errors),
        "summary": str(out_dir / "all_results.csv"),
        "speedup": str(out_dir / "triton_vs_deep_gemm_speedup.csv"),
        "thresholds": (
            str(out_dir / "recommended_thresholds.csv") if threshold_rows else ""
        ),
    }


def child_env() -> dict:
    env = os.environ.copy()
    python_path = str(REPO_ROOT / "python")
    existing_python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{python_path}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else python_path
    )
    # torchada sets the same env var from import side effects. The benchmark
    # sets it explicitly so config discovery does not depend on import order.
    if SGLANG_MOE_CONFIG_ENV not in env:
        torchada_config_dir = discover_torchada_moe_config_dir()
        if torchada_config_dir is not None:
            env[SGLANG_MOE_CONFIG_ENV] = str(torchada_config_dir)
    return env


def main():
    args = parse_args()
    if args._run_one:
        run_one_benchmark(args)
        return
    if args.list_models:
        print_model_profiles()
        return

    selected_models = select_models(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    log_dir = args.out_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = child_env()

    expected_m = sorted(int(x) for x in args.m_list.split(",") if x)
    block_ms = parse_block_ms(args)
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": str(args.out_dir),
        "models": selected_models,
        "modes": args.modes,
        "m_list": expected_m,
        "warmup": args.warmup,
        "iters": args.iters,
        "block_ms": block_ms,
        "sglang_moe_config_dir": env.get(SGLANG_MOE_CONFIG_ENV, ""),
        "script": str(Path(__file__).resolve()),
        "runs": [],
    }

    if not args.plot_only:
        for model_key in selected_models:
            for mode in args.modes:
                mode_block_ms = (
                    block_ms
                    if mode == "contig" and not is_masked_deepgemm(model_key, mode)
                    else [None]
                )
                for block_m in mode_block_ms:
                    variant = expected_variant(model_key, mode, block_m)
                    csv_path = raw_dir / f"{variant}.csv"
                    log_path = log_dir / f"{variant}.log"
                    if args.resume and csv_complete(csv_path, expected_m):
                        event = {
                            "model": model_key,
                            "mode": mode,
                            "block_m": display_block_m(mode, block_m),
                            "status": "skip_existing",
                            "csv": str(csv_path),
                        }
                        manifest["runs"].append(event)
                        print(json.dumps(event), flush=True)
                        continue

                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--_run-one",
                        "--_mode",
                        mode,
                        "--_model-key",
                        model_key,
                        "--out-dir",
                        str(args.out_dir),
                        "--m-list",
                        args.m_list,
                        "--warmup",
                        str(args.warmup),
                        "--iters",
                        str(args.iters),
                    ]
                    if mode == "contig" and block_m is not None:
                        cmd.extend(["--block-m", str(block_m)])
                    print(
                        json.dumps(
                            {
                                "event": "start",
                                "model": model_key,
                                "mode": mode,
                                "block_m": display_block_m(mode, block_m),
                                "log": str(log_path),
                            }
                        ),
                        flush=True,
                    )
                    started = time.perf_counter()
                    with log_path.open("w") as log_file:
                        proc = subprocess.run(
                            cmd,
                            cwd=REPO_ROOT,
                            env=env,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                    event = {
                        "model": model_key,
                        "mode": mode,
                        "block_m": display_block_m(mode, block_m),
                        "status": "ok" if proc.returncode == 0 else "failed",
                        "returncode": proc.returncode,
                        "seconds": round(time.perf_counter() - started, 3),
                        "csv": str(csv_path),
                        "log": str(log_path),
                    }
                    manifest["runs"].append(event)
                    print(json.dumps(event), flush=True)
                    (args.out_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2) + "\n"
                    )
                    if proc.returncode != 0 and args.stop_on_error:
                        raise SystemExit(proc.returncode)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.no_plot:
        return

    output = generate_outputs(args.out_dir)
    print(json.dumps(output, indent=2), flush=True)

    threshold_path = args.out_dir / "recommended_thresholds.json"
    if threshold_path.exists():
        for row in json.loads(threshold_path.read_text()):
            if row["model_key"] in selected_models:
                print(
                    json.dumps(
                        {
                            "model": row["model_key"],
                            "recommended_policy": row["recommended_policy"],
                            "triton_mode": row["triton_mode"],
                            "deep_gemm_mode": row["deep_gemm_mode"],
                            "note": row["note"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
