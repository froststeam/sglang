#!/bin/bash
set -euo pipefail

PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
PIP_TIMEOUT="${PIP_TIMEOUT:-30}"
PIP_RETRIES="${PIP_RETRIES:-2}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/gitlab-ci/pip-cache}"
MUSA_PYTHON_OVERLAY_ROOT="${MUSA_PYTHON_OVERLAY_ROOT:-/data/gitlab-ci/python-overlays}"
MUSA_PYTHON_OVERLAY_TTL_DAYS="${MUSA_PYTHON_OVERLAY_TTL_DAYS:-7}"
MUSA_PYTHON_OVERLAY_CLEAN="${MUSA_PYTHON_OVERLAY_CLEAN:-1}"
MUSA_PYTHON_OVERLAY_INSTALL_DEPS="${MUSA_PYTHON_OVERLAY_INSTALL_DEPS:-1}"
MUSA_PYTHON_OVERLAY_ENV_FILE="${MUSA_PYTHON_OVERLAY_ENV_FILE:-.musa_python_overlay.env}"

if [ -z "${MUSA_PYTHON_OVERLAY_NAME:-}" ]; then
  echo "MUSA_PYTHON_OVERLAY_NAME is required." >&2
  exit 2
fi

if [ -z "${MUSA_PYTHON_OVERLAY_PACKAGES:-}" ] && \
   [ -z "${MUSA_PYTHON_OVERLAY_REQUIREMENTS:-}" ]; then
  echo "MUSA_PYTHON_OVERLAY_PACKAGES or MUSA_PYTHON_OVERLAY_REQUIREMENTS is required." >&2
  exit 2
fi

case "${MUSA_PYTHON_OVERLAY_NAME}" in
  *[!A-Za-z0-9_.-]*)
    echo "MUSA_PYTHON_OVERLAY_NAME may only contain letters, numbers, '.', '_' and '-'." >&2
    exit 2
    ;;
esac

export PIP_CACHE_DIR
mkdir -p "${PIP_CACHE_DIR}" "${MUSA_PYTHON_OVERLAY_ROOT}"

cleanup_old_overlays() {
  if [ "${MUSA_PYTHON_OVERLAY_CLEAN}" != "1" ]; then
    echo "Skip old Python overlay cleanup."
    return
  fi

  if [ ! -d "${MUSA_PYTHON_OVERLAY_ROOT}" ]; then
    return
  fi

  echo "Clean old Python overlays older than ${MUSA_PYTHON_OVERLAY_TTL_DAYS} days under ${MUSA_PYTHON_OVERLAY_ROOT}"
  find "${MUSA_PYTHON_OVERLAY_ROOT}" \
    -mindepth 3 \
    -maxdepth 3 \
    -type d \
    -mtime +"${MUSA_PYTHON_OVERLAY_TTL_DAYS}" \
    -print \
    -exec rm -rf {} +
}

cleanup_old_overlays || true

project_slug="${CI_PROJECT_PATH_SLUG:-sglang}"
commit_sha="${CI_COMMIT_SHA:-local}"
job_scope="${CI_JOB_ID:-manual-$$}"
overlay_dir="${MUSA_PYTHON_OVERLAY_ROOT}/${project_slug}/${commit_sha}/${job_scope}-${MUSA_PYTHON_OVERLAY_NAME}"

case "${overlay_dir}" in
  "${MUSA_PYTHON_OVERLAY_ROOT}"/*) ;;
  *)
    echo "Refuse to use unsafe overlay dir: ${overlay_dir}" >&2
    exit 2
    ;;
esac

echo "Install Python overlay"
echo "  pip index url: ${PIP_INDEX_URL}"
echo "  pip timeout: ${PIP_TIMEOUT}"
echo "  pip retries: ${PIP_RETRIES}"
echo "  pip cache dir: ${PIP_CACHE_DIR}"
echo "  overlay root: ${MUSA_PYTHON_OVERLAY_ROOT}"
echo "  overlay ttl days: ${MUSA_PYTHON_OVERLAY_TTL_DAYS}"
echo "  overlay dir: ${overlay_dir}"
echo "  overlay name: ${MUSA_PYTHON_OVERLAY_NAME}"
echo "  overlay packages: ${MUSA_PYTHON_OVERLAY_PACKAGES:-}"
echo "  overlay requirements: ${MUSA_PYTHON_OVERLAY_REQUIREMENTS:-}"
echo "  install dependencies: ${MUSA_PYTHON_OVERLAY_INSTALL_DEPS}"

rm -rf "${overlay_dir}"
mkdir -p "${overlay_dir}"

pip_args=(
  python3 -m pip install
  --index-url "${PIP_INDEX_URL}"
  --timeout "${PIP_TIMEOUT}"
  --retries "${PIP_RETRIES}"
  --target "${overlay_dir}"
  --upgrade
)

if [ "${MUSA_PYTHON_OVERLAY_INSTALL_DEPS}" != "1" ]; then
  pip_args+=(--no-deps)
fi

if [ -n "${MUSA_PYTHON_OVERLAY_REQUIREMENTS:-}" ]; then
  pip_args+=(-r "${MUSA_PYTHON_OVERLAY_REQUIREMENTS}")
fi

if [ -n "${MUSA_PYTHON_OVERLAY_PACKAGES:-}" ]; then
  # Intentionally allow shell-style whitespace splitting for CI variable package lists.
  # shellcheck disable=SC2206
  package_args=(${MUSA_PYTHON_OVERLAY_PACKAGES})
  pip_args+=("${package_args[@]}")
fi

"${pip_args[@]}"

printf 'MUSA_PYTHON_OVERLAY_DIR=%q\n' "${overlay_dir}" > "${MUSA_PYTHON_OVERLAY_ENV_FILE}"

echo "Wrote overlay env file: ${MUSA_PYTHON_OVERLAY_ENV_FILE}"

if [ -n "${MUSA_PYTHON_OVERLAY_VERIFY_MODULES:-}" ]; then
  PYTHONPATH="${overlay_dir}:${PYTHONPATH:-}" python3 - <<PY
import importlib

for module_name in "${MUSA_PYTHON_OVERLAY_VERIFY_MODULES}".split():
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "<unknown>")
    path = getattr(module, "__file__", "<unknown>")
    print(f"{module_name} version: {version}")
    print(f"{module_name} path: {path}")
PY
fi
