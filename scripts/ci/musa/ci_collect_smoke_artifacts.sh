#!/usr/bin/env bash

set -euo pipefail

project_dir="${CI_PROJECT_DIR:-$(pwd)}"
job_name="${CI_JOB_NAME:-musa-smoke}"
job_slug="${CI_JOB_NAME_SLUG:-}"

if [[ -z "${job_slug}" ]]; then
  job_slug="$(printf "%s" "${job_name}" | tr -c '[:alnum:]_.-' '-' | sed 's/^-*//; s/-*$//')"
fi

artifact_dir="${MUSA_SMOKE_ARTIFACT_DIR:-${project_dir}/ci_artifacts/musa-smoke/${job_slug}}"
mkdir -p "${artifact_dir}"

smoke_model=""
for model_var in \
  MUSA_SMOKE_QWEN_DENSE_MODEL \
  MUSA_SMOKE_QWEN_DENSE_TP_MODEL \
  MUSA_SMOKE_JOYAI_LLM_FLASH_MODEL \
  MUSA_SMOKE_QWEN_MOE_MODEL \
  MUSA_SMOKE_GEMMA4_26B_A4B_MODEL \
  MUSA_SMOKE_QWEN3_VL_32B_MODEL; do
  if [[ -n "${!model_var:-}" ]]; then
    smoke_model="${!model_var}"
    break
  fi
done

shopt -s nullglob
marker_file="${artifact_dir}/.start"

if [[ -f "${marker_file}" ]]; then
  while IFS= read -r -d "" file; do
    cp -f "${file}" "${artifact_dir}/"
  done < <(
    find /tmp -maxdepth 1 -type f \
      \( -name 'gsm8k__*.html' -o -name 'gsm8k__*.json' \) \
      -newer "${marker_file}" -print0 2>/dev/null
  )
fi

report_files=("${artifact_dir}"/gsm8k__*.html "${artifact_dir}"/gsm8k__*.json)

{
  echo "job_name=${job_name}"
  echo "job_slug=${job_slug}"
  echo "pipeline_id=${CI_PIPELINE_ID:-}"
  echo "job_id=${CI_JOB_ID:-}"
  echo "commit_sha=${CI_COMMIT_SHA:-}"
  echo "musa_run_suite=${MUSA_RUN_SUITE:-}"
  echo "smoke_model=${smoke_model}"
  echo "artifact_dir=${artifact_dir}"
  echo
  echo "files:"
  if ((${#report_files[@]} > 0)); then
    for file in "${report_files[@]}"; do
      basename "${file}"
    done
  else
    echo "none"
  fi
} >"${artifact_dir}/manifest.txt"

python3 - "${artifact_dir}" <<'PY'
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
json_files = sorted(artifact_dir.glob("gsm8k__*.json"))
summary = artifact_dir / "summary.md"

lines = ["# MUSA Smoke Artifacts", ""]
if not json_files:
    lines.append("No GSM8K JSON metrics were found.")
else:
    lines.append(
        "| File | Examples | Requested | Score | Latency(s) | Throughput(tok/s) | Empty | Invalid |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for path in json_files:
        try:
            metrics = json.loads(path.read_text())
        except Exception as exc:
            lines.append(f"| `{path.name}` | | | | | | | parse error: {exc} |")
            continue

        def fmt(key):
            value = metrics.get(key)
            return "" if value is None else f"{float(value):.4f}"

        def fmt_int(key):
            value = metrics.get(key)
            return "" if value is None else str(int(value))

        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                path.name,
                fmt_int("num_examples_actual"),
                fmt_int("num_examples_requested"),
                fmt("score"),
                fmt("latency"),
                fmt("output_throughput"),
                fmt("empty_response"),
                fmt("invalid_answer"),
            )
        )

summary.write_text("\n".join(lines) + "\n")
print(f"MUSA smoke artifacts collected under {artifact_dir}")
PY
