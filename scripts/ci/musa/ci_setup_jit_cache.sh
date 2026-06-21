#!/bin/bash

# Source this script before MUSA tests that may JIT-compile kernels.
# It isolates JIT outputs per CI job so parallel pipelines do not mutate a
# shared ~/.cache/sglang_musa_jit directory.

MUSA_CI_ISOLATE_JIT_CACHE="${MUSA_CI_ISOLATE_JIT_CACHE:-1}"
MUSA_CI_JIT_CACHE_ROOT="${MUSA_CI_JIT_CACHE_ROOT:-/data/gitlab-ci/musa-jit-cache}"
MUSA_CI_JIT_CACHE_TTL_DAYS="${MUSA_CI_JIT_CACHE_TTL_DAYS:-3}"
MUSA_CI_CLEAN_OLD_JIT_CACHE="${MUSA_CI_CLEAN_OLD_JIT_CACHE:-1}"

return_or_exit() {
  local code="$1"
  return "${code}" 2>/dev/null || exit "${code}"
}

is_integer() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

sanitize_component() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr -c 'A-Za-z0-9_.-' '_')"
  if [ -z "${value}" ]; then
    value="unknown"
  fi
  printf '%s' "${value}"
}

realpath_or_empty() {
  readlink -f "$1" 2>/dev/null || true
}

process_uses_path() {
  local target_dir="$1"
  local target_real proc pid cmdline cwd fd fd_target maps_file

  target_real="$(realpath_or_empty "${target_dir}")"
  [ -n "${target_real}" ] || return 1

  for proc in /proc/[0-9]*; do
    [ -d "${proc}" ] || continue
    pid="${proc##*/}"
    [ "${pid}" != "$$" ] || continue

    cwd="$(realpath_or_empty "${proc}/cwd")"
    if [[ "${cwd}" == "${target_real}" || "${cwd}" == "${target_real}/"* ]]; then
      return 0
    fi

    cmdline="$(tr '\0' ' ' < "${proc}/cmdline" 2>/dev/null || true)"
    if [[ "${cmdline}" == *"${target_real}"* ]]; then
      return 0
    fi

    for fd in "${proc}"/fd/*; do
      [ -e "${fd}" ] || continue
      fd_target="$(realpath_or_empty "${fd}")"
      if [[ "${fd_target}" == "${target_real}" || "${fd_target}" == "${target_real}/"* ]]; then
        return 0
      fi
    done

    maps_file="${proc}/maps"
    if [ -r "${maps_file}" ] && grep -Fq "${target_real}/" "${maps_file}" 2>/dev/null; then
      return 0
    fi
  done

  return 1
}

cleanup_old_cache_tree() {
  local tree_root="$1"
  local ttl_days="$2"
  local removed=0 skipped=0 candidate

  [ -d "${tree_root}" ] || return 0

  while IFS= read -r -d '' candidate; do
    if process_uses_path "${candidate}"; then
      echo "Skip active MUSA JIT cache: ${candidate}"
      skipped=$((skipped + 1))
      continue
    fi

    echo "Remove expired MUSA JIT cache: ${candidate}"
    rm -rf -- "${candidate}"
    removed=$((removed + 1))
  done < <(
    find "${tree_root}" \
      -mindepth 3 \
      -maxdepth 3 \
      -type d \
      -mtime +"${ttl_days}" \
      -print0
  )

  echo "MUSA JIT cache cleanup finished: root=${tree_root} removed=${removed} skipped=${skipped}"
}

if [ "${MUSA_CI_ISOLATE_JIT_CACHE}" != "1" ]; then
  echo "Skip isolated MUSA JIT cache setup."
  return_or_exit 0
fi

if ! is_integer "${MUSA_CI_JIT_CACHE_TTL_DAYS}"; then
  echo "Invalid MUSA_CI_JIT_CACHE_TTL_DAYS=${MUSA_CI_JIT_CACHE_TTL_DAYS}" >&2
  return_or_exit 2
fi

project_slug="$(sanitize_component "${CI_PROJECT_PATH_SLUG:-sglang}")"
pipeline_id="$(sanitize_component "${CI_PIPELINE_ID:-${CI_COMMIT_SHA:-local}}")"
job_id="$(sanitize_component "${CI_JOB_ID:-${CI_JOB_NAME_SLUG:-local-$$}}")"
cache_key="${project_slug}/${pipeline_id}/${job_id}"

export SGLANG_MUSA_JIT_CACHE_DIR="${MUSA_CI_JIT_CACHE_ROOT}/sglang/${cache_key}"
export TORCH_EXTENSIONS_DIR="${MUSA_CI_JIT_CACHE_ROOT}/torch-extensions/${cache_key}"

mkdir -p "${SGLANG_MUSA_JIT_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

echo "Use isolated MUSA JIT cache:"
echo "  SGLANG_MUSA_JIT_CACHE_DIR=${SGLANG_MUSA_JIT_CACHE_DIR}"
echo "  TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "  ttl days=${MUSA_CI_JIT_CACHE_TTL_DAYS}"

if [ "${MUSA_CI_CLEAN_OLD_JIT_CACHE}" = "1" ]; then
  cleanup_old_cache_tree "${MUSA_CI_JIT_CACHE_ROOT}/sglang" "${MUSA_CI_JIT_CACHE_TTL_DAYS}"
  cleanup_old_cache_tree "${MUSA_CI_JIT_CACHE_ROOT}/torch-extensions" "${MUSA_CI_JIT_CACHE_TTL_DAYS}"
else
  echo "Skip expired MUSA JIT cache cleanup."
fi
