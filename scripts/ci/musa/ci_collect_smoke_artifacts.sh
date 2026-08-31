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
  MUSA_SMOKE_QWEN_MOE_TP_MODEL \
  MUSA_SMOKE_QWEN35_35B_A3B_TEMP06_MODEL \
  MUSA_RADIX_PREFIX_CACHE_MODEL \
  MUSA_SMOKE_QWEN_MOE_MODEL \
  MUSA_SMOKE_GEMMA4_26B_A4B_MODEL \
  MUSA_SMOKE_QWEN3_VL_32B_MODEL \
  MUSA_PD_QWEN35_MOE_MODEL \
  MUSA_SPEC_DSPARK_TARGET_MODEL \
  MUSA_SPEC_EAGLE3_TARGET_MODEL \
  MUSA_SPEC_MTP_MODEL; do
  if [[ -n "${!model_var:-}" ]]; then
    smoke_model="${!model_var}"
    break
  fi
done

parallel_info="$(
  python3 - "${artifact_dir}" <<'PY'
import os
import json
import shlex
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])


PARALLEL_OPTIONS = [
    ("TP", ("--tp", "--tp-size", "--tensor-parallel-size")),
    (
        "EP",
        (
            "--ep",
            "--ep-size",
            "--expert-parallel-size",
            "--expert-model-parallel-size",
            "--moe-ep-size",
        ),
    ),
    ("PP", ("--pp", "--pp-size", "--pipeline-parallel-size")),
    ("DP", ("--dp", "--dp-size", "--data-parallel-size")),
]


def collect_extra_args():
    values = []
    for key, value in os.environ.items():
        if key.startswith("MUSA_SMOKE_") and key.endswith("_EXTRA_ARGS") and value:
            values.append(value)
    args = []
    for value in sorted(values):
        try:
            args.extend(shlex.split(value))
        except ValueError:
            args.extend(value.split())
    return args


def collect_server_args():
    path = artifact_dir / "server_args.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    args = data.get("args", [])
    return args if isinstance(args, list) else []


def option_value(args, names):
    for index, arg in enumerate(args):
        if arg in names and index + 1 < len(args):
            return args[index + 1]
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                return arg[len(prefix) :]
    return ""


def parallel_parts(args):
    parts = []
    for label, names in PARALLEL_OPTIONS:
        value = option_value(args, set(names))
        if value:
            parts.append(f"{label}{value}")
    return parts


args = collect_server_args() or collect_extra_args()
parts = parallel_parts(args)
values = {part[:2]: part[2:] for part in parts}
print(
    values.get("TP", "-"),
    values.get("EP", "-"),
    values.get("PP", "-"),
    values.get("DP", "-"),
    "/".join(parts),
)
PY
)"
read -r smoke_tp smoke_ep smoke_pp smoke_dp smoke_parallel <<<"${parallel_info}"

pd_disaggregation="0"
if [[ "${MUSA_RUN_SUITE:-}" == *pd-disaggregation* ]] || [[ -n "${MUSA_PD_QWEN35_MOE_MODEL:-}" ]]; then
  pd_disaggregation="1"
fi
if [[ -z "${smoke_parallel}" && -n "${MUSA_PD_PARALLEL:-}" ]]; then
  smoke_parallel="${MUSA_PD_PARALLEL}"
fi
if [[ -z "${smoke_parallel}" && -n "${MUSA_RADIX_PREFIX_CACHE_PARALLEL:-}" ]]; then
  smoke_parallel="${MUSA_RADIX_PREFIX_CACHE_PARALLEL}"
fi

sampling_temperature="${MUSA_SMOKE_GSM8K_TEMPERATURE:-${MUSA_PD_GSM8K_TEMPERATURE:-}}"
radix_prefix_cache="0"
if [[ "${MUSA_RUN_SUITE:-}" == *radix-prefix-cache* ]] || [[ -n "${MUSA_RADIX_PREFIX_CACHE_MODEL:-}" ]]; then
  radix_prefix_cache="1"
fi

shopt -s nullglob
marker_file="${artifact_dir}/.start"

if [[ -f "${marker_file}" ]]; then
  while IFS= read -r -d "" file; do
    dst="${artifact_dir}/$(basename "${file}")"
    if [[ ! -e "${dst}" ]]; then
      cp -f "${file}" "${dst}"
    fi
  done < <(
    find /tmp -maxdepth 1 -type f \
      \( -name 'gsm8k__*.html' -o -name 'gsm8k__*.json' -o -name 'vlm__*.html' -o -name 'vlm__*.json' -o -name 'radix_prefix_cache__*.json' \) \
      -newer "${marker_file}" -print0 2>/dev/null
  )

  mudmp_dir="${artifact_dir}/mudmp"
  while IFS= read -r -d "" file; do
    case "${file}" in
      "${project_dir}/"*)
        rel_path="${file#"${project_dir}/"}"
        ;;
      /tmp/*)
        rel_path="tmp/${file#/tmp/}"
        ;;
      *)
        rel_path="$(basename "${file}")"
        ;;
    esac
    mkdir -p "${mudmp_dir}/$(dirname "${rel_path}")"
    cp -f "${file}" "${mudmp_dir}/${rel_path}"
  done < <(
    find "${project_dir}" /tmp -xdev -type f -name '*.mudmp' \
      -newer "${marker_file}" -print0 2>/dev/null
  )
fi

report_files=(
  "${artifact_dir}"/gsm8k__*.html
  "${artifact_dir}"/gsm8k__*.json
  "${artifact_dir}"/vlm__*.html
  "${artifact_dir}"/vlm__*.json
  "${artifact_dir}"/speculative__*.json
  "${artifact_dir}"/radix_prefix_cache__*.json
)
mudmp_files=()
if [[ -d "${artifact_dir}/mudmp" ]]; then
  while IFS= read -r -d "" file; do
    mudmp_files+=("${file}")
  done < <(find "${artifact_dir}/mudmp" -type f -name '*.mudmp' -print0 2>/dev/null)
fi

{
  echo "job_name=${job_name}"
  echo "job_slug=${job_slug}"
  echo "pipeline_id=${CI_PIPELINE_ID:-}"
  echo "job_id=${CI_JOB_ID:-}"
  echo "commit_sha=${CI_COMMIT_SHA:-}"
  echo "musa_run_suite=${MUSA_RUN_SUITE:-}"
  echo "smoke_eval=${MUSA_SMOKE_EVAL_NAME:-gsm8k}"
  echo "speculative_algorithm=${MUSA_SPEC_ALGORITHM:-}"
  echo "smoke_vlm_dataset=${MUSA_SMOKE_VLM_DATASET:-}"
  echo "smoke_vlm_metric=${MUSA_SMOKE_VLM_METRIC:-}"
  echo "smoke_vlm_limit=${MUSA_SMOKE_VLM_LIMIT:-}"
  echo "smoke_model=${smoke_model}"
  echo "pd_disaggregation=${pd_disaggregation}"
  echo "radix_prefix_cache=${radix_prefix_cache}"
  echo "radix_prefix_cache_prefix_len=${MUSA_RADIX_PREFIX_CACHE_PREFIX_LEN:-}"
  echo "radix_prefix_cache_suffix_len=${MUSA_RADIX_PREFIX_CACHE_SUFFIX_LEN:-}"
  echo "sampling_temperature=${sampling_temperature}"
  echo "smoke_tp=${smoke_tp}"
  echo "smoke_ep=${smoke_ep}"
  echo "smoke_pp=${smoke_pp}"
  echo "smoke_dp=${smoke_dp}"
  echo "smoke_parallel=${smoke_parallel}"
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
  echo
  echo "mudmp_files:"
  if ((${#mudmp_files[@]} > 0)); then
    for file in "${mudmp_files[@]}"; do
      printf "%s\n" "${file#"${artifact_dir}/"}"
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
json_files = (
    sorted(artifact_dir.glob("gsm8k__*.json"))
    + sorted(artifact_dir.glob("vlm__*.json"))
    + sorted(artifact_dir.glob("speculative__*.json"))
    + sorted(artifact_dir.glob("radix_prefix_cache__*.json"))
)
summary = artifact_dir / "summary.md"

lines = ["# MUSA Smoke Artifacts", ""]
if not json_files:
    lines.append("No model-eval JSON metrics were found.")
else:
    lines.append(
        "| File | Eval | Dataset | Metric | Examples | Requested/Limit | Score | "
        "Latency(s) | Throughput(tok/s) | Empty | Invalid | Algorithm | Threshold | "
        "Accept Length | Speedup |"
    )
    lines.append(
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | --- | ---: | ---: | ---: |"
    )
    for path in json_files:
        try:
            metrics = json.loads(path.read_text())
        except Exception as exc:
            lines.append(
                f"| `{path.name}` | | | | | | | | | | | | | | parse error: {exc} |"
            )
            continue

        def fmt(key):
            value = metrics.get(key)
            return "" if value is None else f"{float(value):.4f}"

        def fmt_int(key):
            value = metrics.get(key)
            return "" if value is None else str(int(value))

        eval_name = metrics.get("eval_name") or path.name.split("__", 1)[0]
        dataset = metrics.get("dataset") or eval_name
        metric = metrics.get("metric") or "score"
        requested = metrics.get("num_examples_requested", metrics.get("limit", ""))
        requested = "" if requested is None else str(requested)

        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                path.name,
                eval_name,
                dataset,
                metric,
                fmt_int("num_examples_actual"),
                requested,
                fmt("score"),
                fmt("latency"),
                fmt("output_throughput") or fmt("speculative_tps"),
                fmt("empty_response"),
                fmt("invalid_answer"),
                metrics.get("algorithm", ""),
                fmt("accuracy_threshold"),
                fmt("avg_spec_accept_length"),
                fmt("speedup"),
            )
        )

summary.write_text("\n".join(lines) + "\n")
print(f"MUSA smoke artifacts collected under {artifact_dir}")
PY
