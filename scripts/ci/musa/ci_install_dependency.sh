#!/bin/bash
set -euo pipefail

OPTIONAL_DEPS=""
SKIP_SGLANG_BUILD=""
SYSTEM_DEPS_ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sglang-build)
      SKIP_SGLANG_BUILD="1"
      shift
      ;;
    --system-deps-only)
      SYSTEM_DEPS_ONLY="1"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS] [OPTIONAL_DEPS]"
      echo "Options:"
      echo "  --skip-sglang-build         Do not build checkout SGLang; use packages shipped with the image"
      echo "  --system-deps-only          Only ensure system build dependencies and SSH known_hosts"
      exit 0
      ;;
    *)
      OPTIONAL_DEPS="$1"
      shift
      ;;
  esac
done

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TIMEOUT="${PIP_TIMEOUT:-30}"
PIP_RETRIES="${PIP_RETRIES:-2}"
PIP_INSTALL=(
  python3 -m pip install
  --index-url "${PIP_INDEX_URL}"
  --timeout "${PIP_TIMEOUT}"
  --retries "${PIP_RETRIES}"
)
REPO_ROOT="${GITHUB_WORKSPACE:-${CI_PROJECT_DIR:-$(pwd)}}"
WHL_DIR="${SGLANG_CI_WHL_DIR:-/data/whl}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/gitlab-ci/pip-cache}"
MTGPU_TARGET="${MTGPU_TARGET:-mp_31}"
MUTLASS_REPO="${MUTLASS_REPO:-git@sh-code.mthreads.com:mcc-qa/third_party/mutlass.git}"
FLASHINFER_REPO="${FLASHINFER_REPO:-git@sh-code.mthreads.com:mcc-qa/third_party/flashinfer.git}"
MUSA_CI_SSH_KNOWN_HOSTS="${MUSA_CI_SSH_KNOWN_HOSTS:-sh-code.mthreads.com}"
# Keep the MUSA CI torch version aligned with the torch_musa wheel in the image.
# Runner-local CPU torch may be newer and ABI-incompatible with torch_musa.
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
SGLANG_CI_INSTALL_DEPS="${SGLANG_CI_INSTALL_DEPS:-1}"
SGLANG_CI_INSTALL_SYSTEM_DEPS="${SGLANG_CI_INSTALL_SYSTEM_DEPS:-1}"
SGLANG_CI_UPGRADE_PIP_TOOLS="${SGLANG_CI_UPGRADE_PIP_TOOLS:-0}"
SGLANG_CI_UPGRADE_TORCHADA="${SGLANG_CI_UPGRADE_TORCHADA:-1}"
SGLANG_CI_INSTALL_LOCK_TIMEOUT_SECONDS="${SGLANG_CI_INSTALL_LOCK_TIMEOUT_SECONDS:-3600}"
SGLANG_CI_TEST_PACKAGES="${SGLANG_CI_TEST_PACKAGES:-pytest tabulate}"
SGLANG_CI_INSTALL_LMMS_EVAL="${SGLANG_CI_INSTALL_LMMS_EVAL:-1}"
LMMS_EVAL_PACKAGE_SPEC="${LMMS_EVAL_PACKAGE_SPEC:-lmms-eval==0.5.0}"
LMMS_EVAL_RUNTIME_DEPS="${LMMS_EVAL_RUNTIME_DEPS:-evaluate>=0.4.0 pytablewriter sacrebleu>=1.5.0 sqlitedict==2.1.0 tenacity==8.3.0 python-dotenv decord qwen-vl-utils>=0.0.14 numexpr zstandard pycocoevalcap nltk hf_transfer yt-dlp ftfy latex2sympy2 openpyxl}"
SGLANG_CI_EDITABLE_INSTALL="${SGLANG_CI_EDITABLE_INSTALL:-0}"
MUSA_CI_CLEAN_PYTHONUSERBASE="${MUSA_CI_CLEAN_PYTHONUSERBASE:-1}"
MUSA_CI_PYTHONUSERBASE_ROOT="${MUSA_CI_PYTHONUSERBASE_ROOT:-/data/gitlab-ci/python-user-base}"
MUSA_CI_PYTHONUSERBASE_TTL_DAYS="${MUSA_CI_PYTHONUSERBASE_TTL_DAYS:-7}"
MUSA_CI_PYTHONUSERBASE_TRASH_ROOT="${MUSA_CI_PYTHONUSERBASE_TRASH_ROOT:-${MUSA_CI_PYTHONUSERBASE_ROOT}/.trash}"
MUSA_CI_CLEAN_STALE_JIT_LOCKS="${MUSA_CI_CLEAN_STALE_JIT_LOCKS:-1}"
MUSA_CI_STALE_JIT_LOCK_MIN_AGE_SECONDS="${MUSA_CI_STALE_JIT_LOCK_MIN_AGE_SECONDS:-600}"
export MUSA_CI_CLEAN_STALE_JIT_LOCKS
export MUSA_CI_STALE_JIT_LOCK_MIN_AGE_SECONDS
MUSA_CI_JOB_SCOPE="${CI_JOB_ID:-local-$$}"
PYTHONUSERBASE="${PYTHONUSERBASE:-${MUSA_CI_PYTHONUSERBASE:-/data/gitlab-ci/python-user-base/${CI_PROJECT_PATH_SLUG:-sglang}/${CI_COMMIT_SHA:-local}/${MUSA_CI_JOB_SCOPE}}}"
export PYTHONUSERBASE
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
export PIP_CACHE_DIR
mkdir -p "${PIP_CACHE_DIR}"
MUSA_CI_INSTALL_DONE="${PYTHONUSERBASE}/.install.done"
MUSA_CI_INSTALL_LOCK="${PYTHONUSERBASE}/.install.lock"
MUSA_CI_INSTALL_LOCK_HELD=""

cleanup_old_python_user_base() {
  if [ "${MUSA_CI_CLEAN_PYTHONUSERBASE}" != "1" ]; then
    echo "Skip old PYTHONUSERBASE cleanup."
    return
  fi

  if [ ! -d "${MUSA_CI_PYTHONUSERBASE_ROOT}" ]; then
    return
  fi

  local cleanup_project_root=""
  local current_cleanup_root=""
  case "${PYTHONUSERBASE}" in
    "${MUSA_CI_PYTHONUSERBASE_ROOT}"/*/* | "${MUSA_CI_PYTHONUSERBASE_ROOT}"/*/*/*)
      read -r cleanup_project_root current_cleanup_root < <(
        python3 - "${MUSA_CI_PYTHONUSERBASE_ROOT}" "${PYTHONUSERBASE}" <<'PY'
import os
import sys

root = os.path.abspath(sys.argv[1])
userbase = os.path.abspath(sys.argv[2])
try:
    rel = os.path.relpath(userbase, root)
except ValueError:
    sys.exit(0)
parts = rel.split(os.sep)
if len(parts) >= 2 and parts[0] != os.pardir:
    print(os.path.join(root, parts[0]), os.path.join(root, parts[0], parts[1]))
PY
      )
      ;;
  esac
  if [ -z "${cleanup_project_root}" ]; then
    cleanup_project_root="${MUSA_CI_PYTHONUSERBASE_ROOT}/${CI_PROJECT_PATH_SLUG:-sglang}"
  fi
  if [ ! -d "${cleanup_project_root}" ]; then
    return
  fi

  echo "Move old PYTHONUSERBASE dirs older than ${MUSA_CI_PYTHONUSERBASE_TTL_DAYS} days under ${cleanup_project_root} to ${MUSA_CI_PYTHONUSERBASE_TRASH_ROOT}"
  find_args=(
    "${cleanup_project_root}"
    -mindepth 1
    -maxdepth 1
    -type d
    -mtime +"${MUSA_CI_PYTHONUSERBASE_TTL_DAYS}"
  )
  if [ -n "${current_cleanup_root}" ]; then
    find_args+=( ! -path "${current_cleanup_root}" )
  fi
  mkdir -p "${MUSA_CI_PYTHONUSERBASE_TRASH_ROOT}"
  while IFS= read -r -d '' stale_dir; do
    local stale_base
    local stale_dest
    stale_base="$(basename "${stale_dir}")"
    stale_dest="${MUSA_CI_PYTHONUSERBASE_TRASH_ROOT}/${CI_PROJECT_PATH_SLUG:-sglang}.${stale_base}.$(date +%s).$$"
    echo "Move stale PYTHONUSERBASE dir ${stale_dir} -> ${stale_dest}"
    mv -- "${stale_dir}" "${stale_dest}" || echo "Failed to move stale PYTHONUSERBASE dir ${stale_dir}" >&2
  done < <(find "${find_args[@]}" -print0)
}

cleanup_old_python_user_base || true

release_install_lock() {
  if [ "${MUSA_CI_INSTALL_LOCK_HELD}" = "1" ]; then
    rm -rf "${MUSA_CI_INSTALL_LOCK}"
  fi
}

acquire_install_lock() {
  mkdir -p "${PYTHONUSERBASE}"

  if [ -f "${MUSA_CI_INSTALL_DONE}" ]; then
    echo "MUSA CI dependency installation already completed: ${MUSA_CI_INSTALL_DONE}"
    exit 0
  fi

  local start_ts
  local now_ts
  local lock_mtime
  start_ts="$(date +%s)"

  while ! mkdir "${MUSA_CI_INSTALL_LOCK}" 2>/dev/null; do
    if [ -f "${MUSA_CI_INSTALL_DONE}" ]; then
      echo "MUSA CI dependency installation completed by another job: ${MUSA_CI_INSTALL_DONE}"
      exit 0
    fi

    now_ts="$(date +%s)"
    lock_mtime="$(python3 - "${MUSA_CI_INSTALL_LOCK}" <<'PY'
import os
import sys

try:
    print(int(os.path.getmtime(sys.argv[1])))
except OSError:
    print(0)
PY
)"
    if [ "${lock_mtime}" != "0" ] && [ $((now_ts - lock_mtime)) -gt "${SGLANG_CI_INSTALL_LOCK_TIMEOUT_SECONDS}" ]; then
      echo "Remove stale MUSA CI dependency install lock: ${MUSA_CI_INSTALL_LOCK}" >&2
      rm -rf "${MUSA_CI_INSTALL_LOCK}"
      continue
    fi
    if [ $((now_ts - start_ts)) -gt "${SGLANG_CI_INSTALL_LOCK_TIMEOUT_SECONDS}" ]; then
      echo "Timed out waiting for MUSA CI dependency install lock: ${MUSA_CI_INSTALL_LOCK}" >&2
      exit 1
    fi

    echo "Wait for MUSA CI dependency install lock: ${MUSA_CI_INSTALL_LOCK}"
    sleep 10
  done

  MUSA_CI_INSTALL_LOCK_HELD="1"
  {
    echo "pid=$$"
    echo "host=$(hostname)"
    echo "job=${CI_JOB_ID:-unknown}"
    echo "started_at=$(date -Iseconds)"
  } > "${MUSA_CI_INSTALL_LOCK}/owner"
  trap release_install_lock EXIT
}

setup_ssh_known_hosts() {
  if [ -z "${MUSA_CI_SSH_KNOWN_HOSTS}" ]; then
    return
  fi

  if ! command -v ssh-keyscan >/dev/null 2>&1; then
    echo "ssh-keyscan is unavailable; skip known_hosts bootstrap." >&2
    return
  fi

  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"
  touch "${HOME}/.ssh/known_hosts"
  chmod 600 "${HOME}/.ssh/known_hosts"

  for host in ${MUSA_CI_SSH_KNOWN_HOSTS}; do
    if [ -z "${host}" ]; then
      continue
    fi
    echo "Add SSH host key for ${host}"
    ssh-keyscan -H "${host}" >> "${HOME}/.ssh/known_hosts" 2>/dev/null || {
      echo "Failed to scan SSH host key for ${host}" >&2
      return 1
    }
  done
}

verify_torch_version() {
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 - <<PY
import torch

expected = "${TORCH_VERSION}"
actual = torch.__version__.split("+")[0]
if actual != expected:
    raise SystemExit(
        f"Expected torch=={expected}, got {torch.__version__} from {torch.__file__}"
    )
print(f"Verified torch {torch.__version__} from {torch.__file__}")
PY
}

python_extra_exists() {
  local extra="$1"
  local pyproject="${REPO_ROOT}/python/pyproject.toml"

  python3 - "${pyproject}" "${extra}" <<'PY'
import re
import sys

pyproject, extra = sys.argv[1], sys.argv[2]
in_optional_deps = False
extra_re = re.compile(rf"^{re.escape(extra)}\s*=")

with open(pyproject, encoding="utf-8") as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if line == "[project.optional-dependencies]":
            in_optional_deps = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_optional_deps = False
        if in_optional_deps and extra_re.match(line):
            sys.exit(0)

sys.exit(1)
PY
}

install_checkout_sglang_extra() {
  local extra="$1"
  local fallback_package="${2:-}"

  if [ "${SGLANG_CI_INSTALL_DEPS}" != "1" ]; then
    echo "Skip checkout SGLang ${extra} extra because dependency installation is disabled."
    return
  fi

  if ! python_extra_exists "${extra}"; then
    if [ -z "${fallback_package}" ]; then
      echo "Skip checkout SGLang ${extra} extra because python[${extra}] is not defined."
      return
    fi
    echo "Install ${fallback_package} because python[${extra}] is not defined."
    "${PIP_INSTALL[@]}" "${fallback_package}" --user
    return
  fi

  echo "Install checkout SGLang ${extra} extra..."
  EXTRA_INSTALL_ARGS=(-v "./python[${extra}]" --user)
  if [ "${SGLANG_CI_EDITABLE_INSTALL}" = "1" ]; then
    EXTRA_INSTALL_ARGS=(-v -e "./python[${extra}]" --user)
  fi
  (cd "${REPO_ROOT}" && "${PIP_INSTALL[@]}" "${EXTRA_INSTALL_ARGS[@]}")
}

install_lmms_eval() {
  if [ "${SGLANG_CI_INSTALL_LMMS_EVAL}" != "1" ]; then
    echo "Skip lmms-eval installation because SGLANG_CI_INSTALL_LMMS_EVAL is disabled."
    return
  fi

  if python3 -c "import lmms_eval.__main__" >/dev/null 2>&1; then
    echo "lmms-eval is already importable; skip installation."
    return
  fi

  if [ -n "${LMMS_EVAL_RUNTIME_DEPS}" ]; then
    echo "Install lmms-eval runtime dependencies for VLM MMMU evaluation..."
    # Install only runtime packages needed by the openai_compatible MMMU path.
    # lmms-eval's metadata also pulls development/logging packages such as
    # wandb, black, isort, and pre-commit; those are unnecessary in this CI case
    # and make the dependency job sensitive to large downloads.
    "${PIP_INSTALL[@]}" ${LMMS_EVAL_RUNTIME_DEPS} --user
  fi

  echo "Install ${LMMS_EVAL_PACKAGE_SPEC} without optional dependency fan-out..."
  "${PIP_INSTALL[@]}" --no-deps "${LMMS_EVAL_PACKAGE_SPEC}" --user
  python3 -c "import lmms_eval.__main__"
}

if [ ! -d "${REPO_ROOT}/python" ] || [ ! -d "${REPO_ROOT}/sgl-kernel" ]; then
  echo "Invalid SGLang checkout: ${REPO_ROOT}" >&2
  exit 2
fi

if [ -d "${REPO_ROOT}/.git" ]; then
  git config --global --add safe.directory "${REPO_ROOT}" || true
fi

echo "Install MUSA CI dependencies"
echo "  repo root: ${REPO_ROOT}"
echo "  wheel dir: ${WHL_DIR}"
echo "  pip index url: ${PIP_INDEX_URL}"
echo "  pip timeout: ${PIP_TIMEOUT}"
echo "  pip retries: ${PIP_RETRIES}"
echo "  pip cache dir: ${PIP_CACHE_DIR}"
echo "  mtgpu target: ${MTGPU_TARGET}"
echo "  mutlass repo: ${MUTLASS_REPO}"
echo "  flashinfer repo: ${FLASHINFER_REPO}"
echo "  ssh known hosts: ${MUSA_CI_SSH_KNOWN_HOSTS}"
echo "  torch version: ${TORCH_VERSION}"
echo "  install sglang dependencies: ${SGLANG_CI_INSTALL_DEPS}"
echo "  install system build dependencies: ${SGLANG_CI_INSTALL_SYSTEM_DEPS}"
echo "  upgrade pip tools: ${SGLANG_CI_UPGRADE_PIP_TOOLS}"
echo "  upgrade torchada: ${SGLANG_CI_UPGRADE_TORCHADA}"
echo "  install lock timeout seconds: ${SGLANG_CI_INSTALL_LOCK_TIMEOUT_SECONDS}"
echo "  ci test packages: ${SGLANG_CI_TEST_PACKAGES}"
echo "  install lmms-eval: ${SGLANG_CI_INSTALL_LMMS_EVAL}"
echo "  lmms-eval package spec: ${LMMS_EVAL_PACKAGE_SPEC}"
echo "  lmms-eval runtime deps: ${LMMS_EVAL_RUNTIME_DEPS}"
echo "  editable sglang install: ${SGLANG_CI_EDITABLE_INSTALL}"
echo "  clean python user base: ${MUSA_CI_CLEAN_PYTHONUSERBASE}"
echo "  python user base root: ${MUSA_CI_PYTHONUSERBASE_ROOT}"
echo "  python user base ttl days: ${MUSA_CI_PYTHONUSERBASE_TTL_DAYS}"
echo "  python user base: ${PYTHONUSERBASE}"
echo "  clean stale jit locks: ${MUSA_CI_CLEAN_STALE_JIT_LOCKS}"
echo "  stale jit lock min age seconds: ${MUSA_CI_STALE_JIT_LOCK_MIN_AGE_SECONDS}"

install_system_build_deps() {
  local python_include
  python_include="$(python3 - <<'PY'
import sysconfig
print(sysconfig.get_paths()["include"])
PY
)"

  if command -v c++ >/dev/null 2>&1 && \
     command -v ninja >/dev/null 2>&1 && \
     command -v zip >/dev/null 2>&1 && \
     command -v unzip >/dev/null 2>&1 && \
     command -v wget >/dev/null 2>&1 && \
     command -v ssh-keyscan >/dev/null 2>&1 && \
     [ -f "${python_include}/Python.h" ]; then
    echo "System build dependencies are already available."
    return
  fi

  if [ "${SGLANG_CI_INSTALL_SYSTEM_DEPS}" != "1" ]; then
    echo "Missing system build dependencies and SGLANG_CI_INSTALL_SYSTEM_DEPS is disabled." >&2
    echo "  required: build-essential python3-dev ninja-build zip unzip wget openssh-client" >&2
    exit 2
  fi

  if [ "$(id -u)" != "0" ] || ! command -v apt-get >/dev/null 2>&1; then
    echo "Missing system build dependencies; run as root in an apt-based image or bake them into the CI image." >&2
    echo "  required: build-essential python3-dev ninja-build zip unzip wget openssh-client" >&2
    exit 2
  fi

  echo "Install system build dependencies..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ninja-build \
    zip \
    unzip \
    wget \
    openssh-client
  apt-get clean
  rm -rf /var/lib/apt/lists/*
}

install_system_build_deps
setup_ssh_known_hosts || true
bash "${REPO_ROOT}/scripts/ci/musa/ci_clean_stale_jit_locks.sh"

if [ -n "${SYSTEM_DEPS_ONLY}" ]; then
  echo "System dependency check finished."
  exit 0
fi

acquire_install_lock

if [ "${SGLANG_CI_UPGRADE_PIP_TOOLS}" = "1" ]; then
  "${PIP_INSTALL[@]}" --upgrade pip setuptools ninja --user
else
  echo "Skip pip/setuptools/ninja upgrade."
fi
if [ "${SGLANG_CI_UPGRADE_TORCHADA}" = "1" ]; then
  torch_constraint_file="$(mktemp)"
  printf "torch==%s\n" "${TORCH_VERSION}" > "${torch_constraint_file}"
  "${PIP_INSTALL[@]}" --upgrade torchada -c "${torch_constraint_file}" --user
  rm -f "${torch_constraint_file}"
fi

if [ -d "${WHL_DIR}" ] && compgen -G "${WHL_DIR}"/*.whl > /dev/null; then
  echo "Uninstall old packages based on wheel METADATA..."
  PKGS=$(
    for whl in "${WHL_DIR}"/*.whl; do
      meta_file=$(zipinfo -1 "$whl" | awk '/\.dist-info\/METADATA$/ {print; exit}')
      [ -n "$meta_file" ] || continue
      unzip -p "$whl" "$meta_file" 2>/dev/null | sed -n 's/^Name: //p' | head -n1
    done | sort -u
  )
  for pkg in $PKGS; do
    echo "Uninstalling $pkg"
    python3 -m pip uninstall -y "$pkg" || true
  done
  echo "Installing wheel files without dependency resolution..."
  "${PIP_INSTALL[@]}" "${WHL_DIR}"/*.whl --user
else
  echo "No wheel files found in ${WHL_DIR}; skip prebuilt wheel install."
fi

if [ -n "${SKIP_SGLANG_BUILD}" ]; then
  echo "Skip checkout SGLang build."
  exit 0
fi

python3 -m pip uninstall -y sgl-kernel sglang-kernel sglang || true

echo "Clear Python cache under checkout..."
find "${REPO_ROOT}" -name "*.pyc" -delete 2>/dev/null || true
find "${REPO_ROOT}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

restore_pyprojects() {
  if [ -f "${REPO_ROOT}/python/pyproject.toml.ci.bak" ]; then
    mv -f "${REPO_ROOT}/python/pyproject.toml.ci.bak" "${REPO_ROOT}/python/pyproject.toml"
  fi
  if [ -f "${REPO_ROOT}/sgl-kernel/pyproject.toml.ci.bak" ]; then
    mv -f "${REPO_ROOT}/sgl-kernel/pyproject.toml.ci.bak" "${REPO_ROOT}/sgl-kernel/pyproject.toml"
  fi
  if [ -f "${REPO_ROOT}/sgl-kernel/setup_musa.py.ci.bak" ]; then
    mv -f "${REPO_ROOT}/sgl-kernel/setup_musa.py.ci.bak" "${REPO_ROOT}/sgl-kernel/setup_musa.py"
  fi
  release_install_lock
}
trap restore_pyprojects EXIT

echo "Install checkout SGLang..."
if [ -f "${REPO_ROOT}/python/pyproject.toml" ]; then
  cp -f "${REPO_ROOT}/python/pyproject.toml" "${REPO_ROOT}/python/pyproject.toml.ci.bak"
fi
cp -f "${REPO_ROOT}/python/pyproject_other.toml" "${REPO_ROOT}/python/pyproject.toml"
sed -i -E "s/torch==[0-9][0-9A-Za-z.+-]*/torch==${TORCH_VERSION}/g" "${REPO_ROOT}/python/pyproject.toml"
sed -i -E "s/\"torch\"/\"torch==${TORCH_VERSION}\"/g" "${REPO_ROOT}/python/pyproject.toml"

EXTRAS="dev_musa"
if [ -n "${OPTIONAL_DEPS}" ]; then
  EXTRAS="${EXTRAS},${OPTIONAL_DEPS}"
fi
SGLANG_INSTALL_ARGS=(-v "./python[${EXTRAS}]" --user)
if [ "${SGLANG_CI_EDITABLE_INSTALL}" = "1" ]; then
  SGLANG_INSTALL_ARGS=(-v -e "./python[${EXTRAS}]" --user)
fi
if [ "${SGLANG_CI_INSTALL_DEPS}" != "1" ]; then
  SGLANG_INSTALL_ARGS+=(--no-deps)
fi
(cd "${REPO_ROOT}" && "${PIP_INSTALL[@]}" "${SGLANG_INSTALL_ARGS[@]}")
install_checkout_sglang_extra fastokens "fastokens>=0.1.1,<0.2.0"
install_lmms_eval

if [ -n "${SGLANG_CI_TEST_PACKAGES}" ]; then
  "${PIP_INSTALL[@]}" ${SGLANG_CI_TEST_PACKAGES} --user
fi

verify_torch_version

echo "Install checkout sgl-kernel for MUSA..."
if [ -f "${REPO_ROOT}/sgl-kernel/pyproject.toml" ]; then
  cp -f "${REPO_ROOT}/sgl-kernel/pyproject.toml" "${REPO_ROOT}/sgl-kernel/pyproject.toml.ci.bak"
fi
cp -f "${REPO_ROOT}/sgl-kernel/setup_musa.py" "${REPO_ROOT}/sgl-kernel/setup_musa.py.ci.bak"
cp -f "${REPO_ROOT}/sgl-kernel/pyproject_musa.toml" "${REPO_ROOT}/sgl-kernel/pyproject.toml"
sed -i "s#https://github.com/MooreThreads/mutlass.git#${MUTLASS_REPO}#g" "${REPO_ROOT}/sgl-kernel/setup_musa.py"
sed -i "s#https://github.com/flashinfer-ai/flashinfer.git#${FLASHINFER_REPO}#g" "${REPO_ROOT}/sgl-kernel/setup_musa.py"
(cd "${REPO_ROOT}/sgl-kernel" && MTGPU_TARGET="${MTGPU_TARGET}" python3 setup_musa.py install --user)

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$HOME/.local/bin" >> "${GITHUB_PATH}"
fi

{
  echo "commit=${CI_COMMIT_SHA:-unknown}"
  echo "job=${CI_JOB_ID:-unknown}"
  echo "finished_at=$(date -Iseconds)"
} > "${MUSA_CI_INSTALL_DONE}"

echo "MUSA CI dependency installation finished."
