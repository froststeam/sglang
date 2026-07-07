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
    for prefix in ("gsm8k__", "vlm__"):
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
    eval_value = manifest.get("smoke_eval", "")
    dataset = manifest.get("smoke_vlm_dataset", "") if eval_value == "vlm" else eval_value
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
            json_files = sorted(job_dir.glob("gsm8k__*.json")) + sorted(
                job_dir.glob("vlm__*.json")
            )
            if not json_files:
                rows.append(
                    {
                        "model": display_model_name(manifest.get("smoke_model", "")),
                        "tp": manifest.get("smoke_tp", ""),
                        "ep": manifest.get("smoke_ep", ""),
                        "suite": manifest.get("musa_run_suite", ""),
                        "file": "",
                        "metrics": manifest_metrics(manifest),
                        "missing_metrics": True,
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
                        "tp": manifest.get("smoke_tp", ""),
                        "ep": manifest.get("smoke_ep", ""),
                        "suite": manifest.get("musa_run_suite", ""),
                        "file": str(json_file),
                        "metrics": metrics,
                        "missing_metrics": False,
                        "parse_error": parse_error,
                    }
                )

    summary_json = SUMMARY_ROOT / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

    lines = ["# MUSA Smoke Eval Summary", ""]
    if not rows:
        lines.append("No MUSA smoke artifacts were found.")
    else:
        lines.append(
            "| Model | TP | EP | Dataset | Examples | Score | Throughput(tok/s) |"
        )
        lines.append(
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |"
        )
        for row in rows:
            metrics = row["metrics"]
            json_file = Path(row["file"]) if row.get("file") else None
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                    row["model"],
                    row.get("tp", ""),
                    row.get("ep", ""),
                    dataset_name(metrics, json_file),
                    fmt_int(metrics.get("num_examples_actual")),
                    fmt(metrics.get("score")),
                    fmt(metrics.get("output_throughput")),
                )
            )

    summary_md = SUMMARY_ROOT / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n")
    print(summary_md.read_text())


if __name__ == "__main__":
    main()
