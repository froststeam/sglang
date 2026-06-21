#!/bin/bash
set -euo pipefail

if [ "${MUSA_CI_CLEAN_STALE_JIT_LOCKS:-1}" != "1" ]; then
  echo "Skip stale MUSA JIT lock cleanup."
  exit 0
fi

MIN_AGE_SECONDS="${MUSA_CI_STALE_JIT_LOCK_MIN_AGE_SECONDS:-600}"
CACHE_HOME="${XDG_CACHE_HOME:-${HOME:-/root}/.cache}"
TORCH_EXTENSIONS_ROOT="${TORCH_EXTENSIONS_DIR:-${CACHE_HOME}/torch_extensions}"
SGLANG_MUSA_JIT_ROOT="${SGLANG_MUSA_JIT_CACHE_DIR:-${CACHE_HOME}/sglang_musa_jit}"
NOW="$(date +%s)"
REMOVED=0
SKIPPED=0

is_integer() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

if ! is_integer "${MIN_AGE_SECONDS}"; then
  echo "Invalid MUSA_CI_STALE_JIT_LOCK_MIN_AGE_SECONDS=${MIN_AGE_SECONDS}" >&2
  exit 2
fi

realpath_or_empty() {
  readlink -f "$1" 2>/dev/null || true
}

lock_has_open_fd() {
  local lock_path="$1"
  local lock_real fd target

  lock_real="$(realpath_or_empty "${lock_path}")"
  [ -n "${lock_real}" ] || return 1

  for fd in /proc/[0-9]*/fd/*; do
    [ -e "${fd}" ] || continue
    target="$(realpath_or_empty "${fd}")"
    if [ "${target}" = "${lock_real}" ]; then
      return 0
    fi
  done

  return 1
}

target_has_build_process() {
  local target_dir="$1"
  local target_real proc pid cmdline cwd

  target_real="$(realpath_or_empty "${target_dir}")"
  [ -n "${target_real}" ] || return 1

  for proc in /proc/[0-9]*; do
    [ -d "${proc}" ] || continue
    pid="${proc##*/}"
    [ "${pid}" != "$$" ] || continue

    cmdline="$(tr '\0' ' ' < "${proc}/cmdline" 2>/dev/null || true)"
    [ -n "${cmdline}" ] || continue

    case "${cmdline}" in
      *ninja*|*mcc*|*clang*|*c++*|*g++*)
        if [[ "${cmdline}" == *"${target_real}"* ]]; then
          return 0
        fi

        cwd="$(realpath_or_empty "${proc}/cwd")"
        if [[ "${cwd}" == "${target_real}" || "${cwd}" == "${target_real}/"* ]]; then
          return 0
        fi
        ;;
    esac
  done

  return 1
}

clean_lock() {
  local lock_path="$1"
  local target_dir="$2"
  local mtime age

  [ -e "${lock_path}" ] || return 0

  mtime="$(stat -c %Y "${lock_path}" 2>/dev/null || echo "${NOW}")"
  age=$((NOW - mtime))

  if [ "${age}" -lt "${MIN_AGE_SECONDS}" ]; then
    echo "Skip young JIT lock: ${lock_path} age=${age}s"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  if lock_has_open_fd "${lock_path}"; then
    echo "Skip active JIT lock with open fd: ${lock_path}"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  if target_has_build_process "${target_dir}"; then
    echo "Skip active JIT lock with build process: ${lock_path}"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  echo "Remove stale JIT lock: ${lock_path} age=${age}s"
  rm -f -- "${lock_path}"
  REMOVED=$((REMOVED + 1))
}

echo "Clean stale MUSA JIT locks older than ${MIN_AGE_SECONDS}s"

if [ -d "${TORCH_EXTENSIONS_ROOT}" ]; then
  while IFS= read -r -d '' lock_path; do
    clean_lock "${lock_path}" "$(dirname "${lock_path}")"
  done < <(
    find "${TORCH_EXTENSIONS_ROOT}" \
      -mindepth 3 \
      -maxdepth 3 \
      -type f \
      -name lock \
      -print0
  )
fi

if [ -d "${SGLANG_MUSA_JIT_ROOT}/tmp" ]; then
  while IFS= read -r -d '' lock_path; do
    lock_name="$(basename "${lock_path}")"
    clean_lock "${lock_path}" "${SGLANG_MUSA_JIT_ROOT}/${lock_name%.lock}"
  done < <(
    find "${SGLANG_MUSA_JIT_ROOT}/tmp" \
      -maxdepth 1 \
      -type f \
      -name "*.lock" \
      -print0
  )
fi

echo "Stale MUSA JIT lock cleanup finished: removed=${REMOVED}, skipped=${SKIPPED}"
