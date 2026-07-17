#!/usr/bin/env python3
"""Shape helpers derived from the ModelScope LLM registry."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MUSA_BENCH_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MUSA_BENCH_DIR / "registry" / "config.json"
DEFAULT_M_VALUES = (1, 16, 256, 4096, 32768)
DEFAULT_LONG_M_VALUES = (1, 64, 1024, 4096, 16384, 65536)


@dataclass(frozen=True)
class AttentionProfile:
    name: str
    family: str
    rank: int
    tp: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rot_dim: int
    has_qk_norm: bool
    mrope_section: tuple[int, int, int] = (0, 0, 0)
    is_interleaved: bool = False
    dtype_name: str = "bf16"
    source_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class MoeProfile:
    key: str
    title: str
    rank: int
    hidden_size: int
    num_experts: int
    topk: int
    intermediate_size: int | None = None
    source_models: tuple[str, ...] = ()


def add_modelscope_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modelscope-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the generated ModelScope config.json.",
    )
    parser.add_argument(
        "--no-modelscope-refresh",
        action="store_true",
        help="Use the local config file without trying to refresh from ModelScope.",
    )


def add_modelscope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modelscope-shapes",
        dest="modelscope_shapes",
        action="store_true",
        help=(
            "Use shape coverage from ModelScope configs. The local config is "
            "refreshed when ModelScope is reachable, otherwise the existing "
            "config file is used."
        ),
    )
    add_modelscope_config_args(parser)


def load_registry(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text())


def ensure_modelscope_config(
    path: Path = DEFAULT_CONFIG,
    *,
    refresh: bool = True,
) -> Path:
    """Refresh the ModelScope registry if possible, otherwise use local config."""
    if refresh:
        script = MUSA_BENCH_DIR / "tools" / "fetch_modelscope_top_llm_config.py"
        if script.exists():
            cmd = [
                sys.executable,
                str(script),
                "--sleep",
                "0",
                "--output",
                str(path),
            ]
            try:
                subprocess.run(cmd, check=True, timeout=600)
                return path
            except Exception as exc:
                print(
                    f"[modelscope] refresh failed ({type(exc).__name__}: {exc}); "
                    f"falling back to local {path}",
                    file=sys.stderr,
                )
    if not path.exists():
        raise FileNotFoundError(
            f"ModelScope config not found: {path}. "
            "Run fetch_modelscope_top_llm_config.py once or allow network refresh."
        )
    return path


def config_for_args(args: argparse.Namespace) -> Path:
    return ensure_modelscope_config(
        args.modelscope_config,
        refresh=not getattr(args, "no_modelscope_refresh", False),
    )


def iter_models(path: Path = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    return list(load_registry(path).get("models", []))


def shape_summary(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    summary = load_registry(path).get("shape_summary")
    return summary if isinstance(summary, dict) else {}


def summary_shapes(path: Path, key: str) -> list[dict[str, Any]]:
    values = shape_summary(path).get(key)
    return (
        [value for value in values if isinstance(value, dict)]
        if isinstance(values, list)
        else []
    )


def source_model_ids(shape: dict[str, Any]) -> tuple[str, ...]:
    sources = shape.get("source_models")
    if not isinstance(sources, list):
        return ()
    result = []
    for source in sources:
        if isinstance(source, dict) and source.get("modelscope_id"):
            result.append(str(source["modelscope_id"]))
    return tuple(result)


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entry_config(entry: dict[str, Any]) -> dict[str, Any]:
    config = entry.get("config")
    return config if isinstance(config, dict) else entry


def _entry_series(entry: dict[str, Any]) -> str:
    model_id = str(entry.get("modelscope_id", ""))
    lowered = model_id.lower()
    rules = (
        ("DeepSeek", ("deepseek",)),
        ("Qwen", ("qwen", "qwq", "qwopus")),
        ("GLM", ("glm", "chatglm")),
        ("Gemma", ("gemma",)),
        ("MiniMax", ("minimax",)),
        ("Llama", ("llama", "meta-llama")),
        ("Kimi", ("kimi", "moonshot")),
        ("Hunyuan", ("hunyuan", "hy-mt", "hy3")),
        ("Mistral", ("mistral", "mixtral")),
        ("GPT-OSS", ("gpt-oss",)),
        ("Baichuan", ("baichuan",)),
        ("Step", ("step", "stepfun")),
        ("LongCat", ("longcat",)),
        ("MiMo", ("mimo",)),
        ("MiniCPM", ("minicpm",)),
    )
    for series, markers in rules:
        if any(marker in lowered for marker in markers):
            return series
    name = model_id.rsplit("/", 1)[-1]
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    return parts[0] if parts else "Other"


def _rank(entry: dict[str, Any]) -> int:
    return int_or_none(entry.get("rank")) or 10**9


def _model_id(entry: dict[str, Any]) -> str:
    return str(entry.get("modelscope_id", "unknown"))


def _dtype_name(entry: dict[str, Any]) -> str:
    dtype = str(_entry_config(entry).get("torch_dtype", "")).lower()
    if "float16" in dtype and "bfloat" not in dtype:
        return "fp16"
    return "bf16"


def _has_qk_norm(entry: dict[str, Any]) -> bool:
    cfg = _entry_config(entry)
    model_id = _model_id(entry).lower()
    model_type = str(cfg.get("model_type", "")).lower()
    if cfg.get("qk_layernorm") is True or cfg.get("use_qk_norm") is True:
        return True
    return any(
        marker in model_id or marker in model_type
        for marker in ("qwen3", "glm", "gemma")
    )


def _rot_dim(entry: dict[str, Any], head_dim: int) -> int:
    cfg = _entry_config(entry)
    attention = cfg.get("attention") if isinstance(cfg.get("attention"), dict) else {}
    for key in (
        "rotary_dim",
        "rope_dim",
        "qk_rope_head_dim",
        "rotary_emb_dim",
        "partial_rotary_factor",
    ):
        value = attention.get(key, cfg.get(key))
        if key == "partial_rotary_factor":
            try:
                return max(1, int(float(value) * head_dim))
            except (TypeError, ValueError):
                continue
        parsed = int_or_none(value)
        if parsed:
            return parsed
    return head_dim


def _mrope_section(entry: dict[str, Any], rot_dim: int) -> tuple[int, int, int]:
    cfg = _entry_config(entry)
    attention = cfg.get("attention") if isinstance(cfg.get("attention"), dict) else {}
    section = (
        attention.get("mrope_section")
        or attention.get("mrope_sections")
        or cfg.get("mrope_section")
        or cfg.get("mrope_sections")
    )
    if isinstance(section, list) and len(section) == 3:
        parsed = tuple(int_or_none(v) or 0 for v in section)
        return parsed  # type: ignore[return-value]
    return (rot_dim // 2, 0, 0)


def _attention_dict(entry: dict[str, Any]) -> dict[str, Any]:
    cfg = _entry_config(entry)
    attention = cfg.get("attention")
    return attention if isinstance(attention, dict) else cfg


def unique_ranked(values: Iterable[int], *, divisible_by: int = 1) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value <= 0 or value in seen:
            continue
        if divisible_by > 1 and value % divisible_by != 0:
            continue
        seen.add(value)
        result.append(value)
    return result


def hidden_values(
    path: Path = DEFAULT_CONFIG,
    *,
    include_hidden: bool = True,
    include_intermediate: bool = False,
    divisible_by: int = 1,
) -> list[int]:
    summary = shape_summary(path)
    if summary:
        values: list[Any] = []
        if include_hidden and isinstance(summary.get("hidden_values"), list):
            values.extend(summary["hidden_values"])
        if include_intermediate and isinstance(
            summary.get("intermediate_values"), list
        ):
            values.extend(summary["intermediate_values"])
        if values:
            return unique_ranked(
                [int_or_none(value) or 0 for value in values],
                divisible_by=divisible_by,
            )

    values: list[int] = []
    for entry in iter_models(path):
        cfg = _entry_config(entry)
        moe = cfg.get("moe") if isinstance(cfg.get("moe"), dict) else {}
        if include_hidden:
            value = int_or_none(cfg.get("hidden_size"))
            if value:
                values.append(value)
        if include_intermediate:
            for key in (
                "intermediate_size",
                "moe_intermediate_size",
                "expert_intermediate_size",
            ):
                value = int_or_none(moe.get(key, cfg.get(key)))
                if value:
                    values.append(value)
    return unique_ranked(values, divisible_by=divisible_by)


def attention_profiles(
    path: Path = DEFAULT_CONFIG,
    *,
    tp_sizes: Iterable[int] = (1, 2, 4, 8),
) -> list[AttentionProfile]:
    profiles: list[AttentionProfile] = []
    seen: set[tuple[int, int, int, int, int, str]] = set()
    summary_attention = summary_shapes(path, "attention_shapes")
    if summary_attention:
        for shape_index, shape in enumerate(summary_attention, start=1):
            q_heads = int_or_none(shape.get("q_heads"))
            kv_heads = int_or_none(shape.get("kv_heads")) or q_heads
            head_dim = int_or_none(shape.get("head_dim"))
            rot_dim = int_or_none(shape.get("rot_dim")) or head_dim
            if not q_heads or not kv_heads or not head_dim or not rot_dim:
                continue
            if rot_dim <= 0 or rot_dim > head_dim or rot_dim % 2 != 0:
                continue
            section = shape.get("mrope_section")
            if isinstance(section, list) and len(section) == 3:
                mrope_section = tuple(int_or_none(v) or 0 for v in section)
            else:
                mrope_section = (rot_dim // 2, 0, 0)
            dtype = str(shape.get("dtype") or "bf16")
            for tp in tp_sizes:
                if q_heads % tp != 0 or kv_heads % tp != 0:
                    continue
                local_q = q_heads // tp
                local_kv = kv_heads // tp
                key = (local_q, local_kv, head_dim, rot_dim, tp, dtype)
                if key in seen:
                    continue
                seen.add(key)
                profiles.append(
                    AttentionProfile(
                        name=f"attn-shape-{shape_index:03d}-tp{tp}",
                        family="shape_summary",
                        rank=shape_index,
                        tp=tp,
                        q_heads=local_q,
                        kv_heads=local_kv,
                        head_dim=head_dim,
                        rot_dim=rot_dim,
                        has_qk_norm=bool(shape.get("has_qk_norm")),
                        mrope_section=mrope_section,  # type: ignore[arg-type]
                        is_interleaved=bool(shape.get("is_interleaved"))
                        or mrope_section != (rot_dim // 2, 0, 0),
                        dtype_name=dtype,
                        source_models=source_model_ids(shape),
                    )
                )
        return profiles

    for entry in iter_models(path):
        attention = _attention_dict(entry)
        hidden = int_or_none(_entry_config(entry).get("hidden_size"))
        q_heads = int_or_none(attention.get("num_attention_heads"))
        kv_heads = int_or_none(attention.get("num_key_value_heads")) or q_heads
        head_dim = int_or_none(attention.get("head_dim"))
        if head_dim is None and hidden and q_heads:
            head_dim = hidden // q_heads
        if not q_heads or not kv_heads or not head_dim:
            continue
        rot_dim = _rot_dim(entry, head_dim)
        if rot_dim <= 0 or rot_dim > head_dim or rot_dim % 2 != 0:
            continue
        mrope_section = _mrope_section(entry, rot_dim)
        interleaved = attention.get("interleaved")
        if interleaved is None:
            interleaved = attention.get("interleave")
        if interleaved is None:
            interleaved = attention.get("rotary_interleaved")
        for tp in tp_sizes:
            if q_heads % tp != 0 or kv_heads % tp != 0:
                continue
            local_q = q_heads // tp
            local_kv = kv_heads // tp
            key = (local_q, local_kv, head_dim, rot_dim, tp, _dtype_name(entry))
            if key in seen:
                continue
            seen.add(key)
            rank = _rank(entry)
            family = _model_id(entry)
            profiles.append(
                AttentionProfile(
                    name=f"ms-r{rank}-{family.rsplit('/', 1)[-1]}-tp{tp}",
                    family=family,
                    rank=rank,
                    tp=tp,
                    q_heads=local_q,
                    kv_heads=local_kv,
                    head_dim=head_dim,
                    rot_dim=rot_dim,
                    has_qk_norm=_has_qk_norm(entry),
                    mrope_section=mrope_section,
                    is_interleaved=bool(interleaved)
                    or mrope_section != (rot_dim // 2, 0, 0),
                    dtype_name=_dtype_name(entry),
                )
            )
    return profiles


def moe_profiles(path: Path = DEFAULT_CONFIG) -> list[MoeProfile]:
    profiles: list[MoeProfile] = []
    seen: set[tuple[int, int, int]] = set()
    summary_moe = summary_shapes(path, "moe_shapes")
    if summary_moe:
        for shape_index, shape in enumerate(summary_moe, start=1):
            hidden = int_or_none(shape.get("hidden_size"))
            experts = int_or_none(shape.get("num_experts"))
            topk = int_or_none(shape.get("topk"))
            intermediate = int_or_none(shape.get("intermediate_size"))
            if not hidden or not experts or not topk:
                continue
            key = (hidden, experts, topk)
            if key in seen:
                continue
            seen.add(key)
            profiles.append(
                MoeProfile(
                    key=f"moe-shape-{shape_index:03d}",
                    title=f"shape_summary / E{experts} / topk{topk} / hidden{hidden}",
                    rank=shape_index,
                    hidden_size=hidden,
                    num_experts=experts,
                    topk=topk,
                    intermediate_size=intermediate,
                    source_models=source_model_ids(shape),
                )
            )
        return profiles

    for entry in iter_models(path):
        cfg = _entry_config(entry)
        moe = cfg.get("moe")
        if not isinstance(moe, dict):
            continue
        hidden = int_or_none(cfg.get("hidden_size"))
        experts = (
            int_or_none(moe.get("num_experts"))
            or int_or_none(moe.get("n_routed_experts"))
            or int_or_none(moe.get("num_local_experts"))
        )
        topk = (
            int_or_none(moe.get("num_experts_per_tok"))
            or int_or_none(moe.get("num_experts_per_token"))
            or int_or_none(moe.get("top_k"))
            or int_or_none(moe.get("topk"))
        )
        intermediate = (
            int_or_none(moe.get("moe_intermediate_size"))
            or int_or_none(moe.get("expert_intermediate_size"))
            or int_or_none(moe.get("intermediate_size"))
            or int_or_none(cfg.get("intermediate_size"))
        )
        if not hidden or not experts or not topk:
            continue
        key = (hidden, experts, topk)
        if key in seen:
            continue
        seen.add(key)
        rank = _rank(entry)
        model_id = _model_id(entry)
        profiles.append(
            MoeProfile(
                key=f"ms-r{rank}-{model_id.rsplit('/', 1)[-1]}",
                title=f"{model_id} / E{experts} / topk{topk} / hidden{hidden}",
                rank=rank,
                hidden_size=hidden,
                num_experts=experts,
                topk=topk,
                intermediate_size=intermediate,
            )
        )
    return profiles


def print_summary(path: Path = DEFAULT_CONFIG) -> None:
    hidden = hidden_values(path, include_hidden=True)
    intermediate_values = hidden_values(
        path, include_hidden=False, include_intermediate=True, divisible_by=128
    )
    attn = attention_profiles(path)
    moe = moe_profiles(path)
    print(f"config={path}")
    print(f"models={len(iter_models(path))}")
    print(f"hidden_values={len(hidden)} {hidden}")
    print(
        f"intermediate_values_div128={len(intermediate_values)} "
        f"{intermediate_values}"
    )
    print(f"attention_profiles={len(attn)}")
    print(f"moe_profiles={len(moe)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print_summary(args.config)


if __name__ == "__main__":
    main()
