#!/usr/bin/env python3

import json
from pathlib import Path
from urllib.parse import unquote

ARTIFACT_ROOT = Path("ci_artifacts/musa-smoke")
SUMMARY_ROOT = Path("ci_artifacts/musa-smoke-summary")


def fmt(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value):
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def display_model_name(model: str) -> str:
    if not model:
        return ""
    return unquote(model.rstrip("/").split("/")[-1])


def model_from_metrics_file(path: Path) -> str:
    stem = path.stem
    for prefix in ("gsm8k__", "vlm__", "speculative__", "radix_prefix_cache__"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    else:
        if "_" in stem:
            stem = stem.split("_", 1)[1]
    if stem.startswith("data_models_"):
        stem = stem[len("data_models_") :]
    return stem


def eval_name(metrics: dict, json_file: Path | None) -> str:
    value = metrics.get("eval_name")
    if value:
        return str(value)
    if json_file is not None and "__" in json_file.name:
        return json_file.name.split("__", 1)[0]
    return ""


def dataset_name(metrics: dict, json_file: Path | None) -> str:
    value = metrics.get("dataset")
    if value:
        return str(value)
    value = eval_name(metrics, json_file)
    return value


def manifest_metrics(manifest: dict) -> dict:
    if manifest.get("radix_prefix_cache") == "1":
        prefix_len = manifest.get("radix_prefix_cache_prefix_len", "")
        suffix_len = manifest.get("radix_prefix_cache_suffix_len", "")
        try:
            configured_prefix_ratio = float(prefix_len) / (
                float(prefix_len) + float(suffix_len)
            )
        except (TypeError, ValueError, ZeroDivisionError):
            configured_prefix_ratio = None
        return {
            "eval_name": "radix_prefix_cache",
            "dataset": "radix_prefix_cache",
            "metric": "cache_hit_rate",
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "configured_prefix_ratio": configured_prefix_ratio,
        }

    eval_value = manifest.get("smoke_eval", "")
    dataset = (
        manifest.get("smoke_vlm_dataset", "") if eval_value == "vlm" else eval_value
    )
    metric = manifest.get("smoke_vlm_metric", "") if eval_value == "vlm" else "score"
    limit = manifest.get("smoke_vlm_limit", "") if eval_value == "vlm" else ""
    return {
        "eval_name": eval_value,
        "dataset": dataset,
        "metric": metric,
        "limit": limit,
    }


def model_name(metrics: dict, manifest: dict, json_file: Path | None) -> str:
    model = metrics.get("model") or manifest.get("smoke_model", "")
    if not model and json_file is not None:
        model = model_from_metrics_file(json_file)
    return display_model_name(model)


def parallel_name(manifest: dict) -> str:
    value = manifest.get("smoke_parallel", "")
    if value:
        return value

    parts = []
    for label, key in (
        ("TP", "smoke_tp"),
        ("EP", "smoke_ep"),
        ("PP", "smoke_pp"),
        ("DP", "smoke_dp"),
    ):
        value = manifest.get(key, "")
        if value and value != "-":
            parts.append(f"{label}{value}")
    return "/".join(parts)


def variant_name(manifest: dict) -> str:
    parts = []
    if manifest.get("pd_disaggregation") == "1":
        parts.append("PD disaggregation")

    temperature = manifest.get("sampling_temperature", "")
    if temperature:
        try:
            temperature_value = float(temperature)
        except ValueError:
            temperature_value = None
        if temperature_value is None or temperature_value != 0.0:
            parts.append(f"temp={temperature}")

    return ", ".join(parts)


def display_model_with_variant(row: dict, metrics: dict, json_file: Path | None) -> str:
    model = row["model"]
    if (
        row.get("variant")
        and dataset_name(metrics, json_file) == "gsm8k"
    ):
        model = f"{model} ({row['variant']})"
    return model


def is_radix_prefix_cache(
    metrics: dict, json_file: Path | None, manifest: dict
) -> bool:
    if manifest.get("radix_prefix_cache") == "1":
        return True
    if eval_name(metrics, json_file) == "radix_prefix_cache":
        return True
    return json_file is not None and json_file.name.startswith("radix_prefix_cache__")


def load_manifest(job_dir: Path) -> dict:
    manifest = job_dir / "manifest.txt"
    data = {}
    if not manifest.exists():
        return data

    for line in manifest.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    if ARTIFACT_ROOT.exists():
        for job_dir in sorted(p for p in ARTIFACT_ROOT.iterdir() if p.is_dir()):
            manifest = load_manifest(job_dir)
            json_files = (
                sorted(job_dir.glob("gsm8k__*.json"))
                + sorted(job_dir.glob("vlm__*.json"))
                + sorted(job_dir.glob("speculative__*.json"))
                + sorted(job_dir.glob("radix_prefix_cache__*.json"))
            )
            if not json_files:
                metrics = manifest_metrics(manifest)
                rows.append(
                    {
                        "model": display_model_name(manifest.get("smoke_model", "")),
                        "parallel": parallel_name(manifest),
                        "suite": manifest.get("musa_run_suite", ""),
                        "variant": variant_name(manifest),
                        "file": "",
                        "metrics": metrics,
                        "missing_metrics": True,
                        "speculative": False,
                        "radix_prefix_cache": is_radix_prefix_cache(
                            metrics, None, manifest
                        ),
                    }
                )
                continue

            for json_file in json_files:
                try:
                    metrics = json.loads(json_file.read_text())
                    parse_error = ""
                except Exception as exc:
                    metrics = {}
                    parse_error = str(exc)

                rows.append(
                    {
                        "model": model_name(metrics, manifest, json_file),
                        "parallel": parallel_name(manifest),
                        "suite": manifest.get("musa_run_suite", ""),
                        "variant": variant_name(manifest),
                        "file": str(json_file),
                        "metrics": metrics,
                        "missing_metrics": False,
                        "parse_error": parse_error,
                        # A speculative job also produces a generic GSM8K
                        # artifact for its accuracy run.  Only the dedicated
                        # speculative artifact belongs in the speculative
                        # summary table.
                        "speculative": json_file.name.startswith("speculative__"),
                        "radix_prefix_cache": is_radix_prefix_cache(
                            metrics, json_file, manifest
                        ),
                    }
                )

    summary_json = SUMMARY_ROOT / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

    # A speculative job emits both a generic GSM8K accuracy artifact and a
    # dedicated speculative artifact.  Suppress the generic artifact from the
    # ordinary table for those jobs; otherwise one CI job appears as two rows
    # and its accuracy throughput is mixed with decode-only speculative TPS.
    speculative_job_dirs = {
        Path(row["file"]).parent
        for row in rows
        if row.get("speculative") and row.get("file")
    }
    ordinary_rows = [
        row
        for row in rows
        if not row.get("speculative")
        and not row.get("radix_prefix_cache")
        and (
            not row.get("file")
            or Path(row["file"]).parent not in speculative_job_dirs
        )
    ]
    speculative_rows = [row for row in rows if row.get("speculative")]
    radix_prefix_cache_rows = [row for row in rows if row.get("radix_prefix_cache")]

    lines = ["# MUSA Smoke Eval Summary", ""]
    if not rows:
        lines.append("No MUSA smoke artifacts were found.")
    else:
        lines.extend(
            [
                "| Model | Parallel | Dataset | Examples | Score | Throughput(tok/s) |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in ordinary_rows:
            metrics = row["metrics"]
            json_file = Path(row["file"]) if row.get("file") else None
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} |".format(
                    display_model_with_variant(row, metrics, json_file),
                    row.get("parallel", ""),
                    dataset_name(metrics, json_file),
                    fmt_int(metrics.get("num_examples_actual")),
                    fmt(metrics.get("score")),
                    fmt(metrics.get("output_throughput")),
                )
            )

        lines.extend(
            [
                "",
                "## Speculative decoding evaluation",
                "",
                "| Model | Algorithm | Parallel | Dataset | Accuracy Score | Accept Length | Speedup |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        if speculative_rows:
            for row in speculative_rows:
                metrics = row["metrics"]
                json_file = Path(row["file"]) if row.get("file") else None
                lines.append(
                    "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                        row["model"],
                        metrics.get("algorithm", ""),
                        row.get("parallel", ""),
                        dataset_name(metrics, json_file),
                        fmt(metrics.get("score")),
                        fmt(metrics.get("avg_spec_accept_length")),
                        fmt(metrics.get("speedup")),
                    )
                )
        else:
            lines.append(
                "| No speculative decoding artifacts found. | | | | | | |"
            )

        lines.extend(
            [
                "",
                "## Radix prefix cache evaluation",
                "",
                "| Model | Parallel | Prefix Ratio | Prefix Len | Suffix Len | "
                "Hit Rate | Cached/Prompt Tokens |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        if radix_prefix_cache_rows:
            for row in radix_prefix_cache_rows:
                metrics = row["metrics"]
                total_cached = metrics.get("total_cached_tokens")
                total_prompt = metrics.get("total_prompt_tokens")
                token_ratio = (
                    f"{fmt_int(total_cached)}/{fmt_int(total_prompt)}"
                    if total_cached is not None and total_prompt is not None
                    else ""
                )
                lines.append(
                    "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                        row["model"],
                        row.get("parallel", ""),
                        fmt(metrics.get("configured_prefix_ratio")),
                        fmt_int(metrics.get("prefix_len")),
                        fmt_int(metrics.get("suffix_len")),
                        fmt(metrics.get("cache_hit_rate")),
                        token_ratio,
                    )
                )
        else:
            lines.append("| No radix prefix cache artifacts found. | | | | | | |")

    summary_md = SUMMARY_ROOT / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n")
    print(summary_md.read_text())


if __name__ == "__main__":
    main()
