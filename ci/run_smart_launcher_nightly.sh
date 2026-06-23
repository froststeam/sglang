#!/usr/bin/env bash
set -euo pipefail

SGLANG_TEST_DIR="${SGLANG_TEST_DIR:-/workspace/sglang_test}"
SGLANG_TEST_REF="${SGLANG_TEST_REF:-origin/master}"
SMART_LAUNCHER_DIR="${SGLANG_TEST_DIR}/smart_launcher"
SMART_LAUNCHER_SUITE="${SMART_LAUNCHER_SUITE:-remote_jingdong_best_baseline}"
SMART_LAUNCHER_REMOTE="${SMART_LAUNCHER_REMOTE:-ci_env}"
TEST_CONTAINER_NODES="${TEST_CONTAINER_NODES:-${SMART_LAUNCHER_NODES:-}}"
SMART_LAUNCHER_CONTAINER_NAME="${SMART_LAUNCHER_CONTAINER_NAME:-sglang-nightly-ci}"
SMART_LAUNCHER_KEEP_CONTAINER_ON_FAILURE="${SMART_LAUNCHER_KEEP_CONTAINER_ON_FAILURE:-1}"
SMART_LAUNCHER_KEEP_CONTAINER_ON_SUCCESS="${SMART_LAUNCHER_KEEP_CONTAINER_ON_SUCCESS:-0}"
SMART_LAUNCHER_SSH_OPTS="${SMART_LAUNCHER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=20}"
SGLANG_NIGHTLY_IMAGE_REPO="${SGLANG_NIGHTLY_IMAGE_REPO:-registry.mthreads.com/mcconline/inference/sglang}"
SGLANG_NIGHTLY_CHANNEL_TAG="${SGLANG_NIGHTLY_CHANNEL_TAG:-${CI_COMMIT_REF_SLUG:-offline-sync-candidate}}"
SGLANG_NIGHTLY_FALLBACK_IMAGE="${SGLANG_NIGHTLY_FALLBACK_IMAGE:-${SGLANG_NIGHTLY_IMAGE_REPO}:offline-sync-candidate}"

ARTIFACT_DIR="${ARTIFACT_DIR:-ci_artifacts/smart_launcher}"
LOG_DIR="${ARTIFACT_DIR}/logs"
MAIN_LOG="${LOG_DIR}/nightly.log"
SCRIPT_START_EPOCH="$(date +%s)"
RUN_ID=""
SELECTED_TEST_IMAGE=""
SUITE_EXIT_CODE=0
SCRIPT_EXIT_CODE=0
CONTAINERS_PREPARED=0

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MAIN_LOG}"
}

die() {
  log "ERROR: $*"
  exit 1
}

ssh_run() {
  local node="$1"
  shift
  local host="${node%%|*}"
  local port=""
  if [[ "${node}" == *"|"* ]]; then
    port="${node#*|}"
  fi
  if [[ -n "${port}" ]]; then
    ssh -n ${SMART_LAUNCHER_SSH_OPTS} -p "${port}" "${host}" "$@"
  else
    ssh -n ${SMART_LAUNCHER_SSH_OPTS} "${host}" "$@"
  fi
}

select_test_image() {
  local manual_image="${TEST_IMAGE:-}"
  local nightly_image="${SGLANG_NIGHTLY_IMAGE:-}"
  local channel_image="${SGLANG_NIGHTLY_CHANNEL_IMAGE:-}"
  if [[ "${manual_image}" == '$'* ]]; then
    manual_image=""
  fi
  if [[ "${nightly_image}" == '$'* ]]; then
    nightly_image=""
  fi
  if [[ "${channel_image}" == '$'* ]]; then
    channel_image=""
  fi

  if [[ -n "${manual_image}" ]]; then
    printf '%s\n' "${manual_image}"
    return
  fi
  if [[ -n "${nightly_image}" ]]; then
    printf '%s\n' "${nightly_image}"
    return
  fi
  if [[ -n "${channel_image}" ]]; then
    printf '%s\n' "${channel_image}"
    return
  fi
  if [[ -n "${SGLANG_NIGHTLY_CHANNEL_TAG:-}" ]]; then
    printf '%s:%s\n' "${SGLANG_NIGHTLY_IMAGE_REPO}" "${SGLANG_NIGHTLY_CHANNEL_TAG}"
    return
  fi
  printf '%s\n' "${SGLANG_NIGHTLY_FALLBACK_IMAGE}"
}

update_sglang_test() {
  [[ -d "${SGLANG_TEST_DIR}/.git" ]] || die "SGLANG_TEST_DIR is not a git checkout: ${SGLANG_TEST_DIR}"
  log "Updating ${SGLANG_TEST_DIR} to ${SGLANG_TEST_REF}"
  git -C "${SGLANG_TEST_DIR}" fetch origin 2>&1 | tee -a "${MAIN_LOG}"
  git -C "${SGLANG_TEST_DIR}" reset --hard "${SGLANG_TEST_REF}" 2>&1 | tee -a "${MAIN_LOG}"
}

resolve_nodes() {
  [[ -n "${TEST_CONTAINER_NODES:-}" ]] || die "TEST_CONTAINER_NODES is required for runner-side container updates"
  tr ', ' '\n\n' <<< "${TEST_CONTAINER_NODES}" | sed '/^$/d'
}

prepare_node_container() {
  local node="$1"
  local container="$2"
  local image="$3"
  local remote_log="${LOG_DIR}/container-${node//[^A-Za-z0-9_.-]/_}.log"

  log "Preparing ${container} on ${node}"
  {
    echo "node=${node}"
    echo "container=${container}"
    echo "image=${image}"
    ssh_run "${node}" "docker pull '${image}'"
    ssh_run "${node}" "docker rm -f '${container}' || true"
    ssh_run "${node}" "docker run -d \
      --name '${container}' \
      --env MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      --env MTHREADS_DRIVER_CAPABILITIES=all \
      --net host \
      --privileged \
      --pid=host \
      --pids-limit=65535 \
      --shm-size 500g \
      -v /mnt/data:/data \
      '${image}' \
      bash -lc 'mkdir -p /var/run/sshd /run/sshd && service ssh start && sleep infinity'"
  } 2>&1 | tee -a "${remote_log}" "${MAIN_LOG}"
}

cleanup_containers() {
  local reason="$1"
  local should_cleanup=0

  if [[ "${CONTAINERS_PREPARED}" != "1" ]]; then
    return
  fi
  if [[ "${reason}" == "success" && "${SMART_LAUNCHER_KEEP_CONTAINER_ON_SUCCESS}" != "1" ]]; then
    should_cleanup=1
  fi
  if [[ "${reason}" != "success" && "${SMART_LAUNCHER_KEEP_CONTAINER_ON_FAILURE}" != "1" ]]; then
    should_cleanup=1
  fi
  if [[ "${should_cleanup}" != "1" ]]; then
    log "Keeping test containers after ${reason}"
    return
  fi

  log "Cleaning test containers after ${reason}"
  while IFS= read -r node; do
    [[ -n "${node}" ]] || continue
    ssh_run "${node}" "docker rm -f '${SMART_LAUNCHER_CONTAINER_NAME}' || true" 2>&1 | tee -a "${MAIN_LOG}" || true
  done < "${ARTIFACT_DIR}/nodes.txt"
}

detect_latest_result_run_id() {
  local summary
  local mtime
  local latest_dir=""
  local latest_mtime=0

  for summary in "${SMART_LAUNCHER_DIR}"/results/r_*/summary.html; do
    [[ -f "${summary}" ]] || continue
    mtime="$(stat -c %Y "${summary}" 2>/dev/null || stat -f %m "${summary}" 2>/dev/null || echo 0)"
    [[ "${mtime}" =~ ^[0-9]+$ ]] || mtime=0
    if (( mtime >= SCRIPT_START_EPOCH && mtime > latest_mtime )); then
      latest_mtime="${mtime}"
      latest_dir="${summary%/summary.html}"
    fi
  done

  if [[ -n "${latest_dir}" ]]; then
    basename "${latest_dir}"
  fi
}

collect_artifacts() {
  mkdir -p "${ARTIFACT_DIR}"

  if [[ -z "${RUN_ID}" ]]; then
    RUN_ID="$(detect_latest_result_run_id)"
    if [[ -n "${RUN_ID}" ]]; then
      log "Detected RUN_ID=${RUN_ID} from results directory"
    fi
  fi

  if [[ -n "${RUN_ID}" ]]; then
    (
      cd "${SMART_LAUNCHER_DIR}"
      PYTHONPATH=. python3 exp.py exp logs pull \
        --remote "${SMART_LAUNCHER_REMOTE}" \
        --run-id "${RUN_ID}"
    ) 2>&1 | tee -a "${MAIN_LOG}" || true
  fi

  if [[ -n "${RUN_ID}" && -d "${SMART_LAUNCHER_DIR}/results/${RUN_ID}" ]]; then
    printf '%s\n' "${RUN_ID}" > "${ARTIFACT_DIR}/run_id.txt"
    rm -rf "${ARTIFACT_DIR}/results"
    mkdir -p "${ARTIFACT_DIR}/results"
    cp -a "${SMART_LAUNCHER_DIR}/results/${RUN_ID}" "${ARTIFACT_DIR}/results/"
    for name in summary.csv summary.html result.json archive_result.json; do
      if [[ -f "${SMART_LAUNCHER_DIR}/results/${RUN_ID}/${name}" ]]; then
        cp -f "${SMART_LAUNCHER_DIR}/results/${RUN_ID}/${name}" "${ARTIFACT_DIR}/${name}"
      fi
    done
  fi

  if [[ -n "${RUN_ID}" && -d "${SMART_LAUNCHER_DIR}/remote_runs/${RUN_ID}" ]]; then
    rm -rf "${ARTIFACT_DIR}/remote_runs"
    mkdir -p "${ARTIFACT_DIR}/remote_runs"
    cp -a "${SMART_LAUNCHER_DIR}/remote_runs/${RUN_ID}" "${ARTIFACT_DIR}/remote_runs/"
  fi
}

cleanup() {
  SCRIPT_EXIT_CODE=$?
  trap - EXIT INT TERM

  if [[ -n "${RUN_ID:-}" ]]; then
    log "Stopping smart_launcher run_id=${RUN_ID}"
    (
      cd "${SMART_LAUNCHER_DIR}"
      PYTHONPATH=. python3 exp.py exp suite stop \
        --remote "${SMART_LAUNCHER_REMOTE}" \
        --run-id "${RUN_ID}"
    ) 2>&1 | tee -a "${MAIN_LOG}" || true
  fi

  collect_artifacts

  if [[ "${SCRIPT_EXIT_CODE}" == "0" ]]; then
    cleanup_containers success
  else
    cleanup_containers failure
  fi

  exit "${SCRIPT_EXIT_CODE}"
}
trap cleanup EXIT INT TERM

main() {
  SELECTED_TEST_IMAGE="$(select_test_image)"
  [[ -n "${SELECTED_TEST_IMAGE}" ]] || die "Unable to resolve TEST_IMAGE"
  log "Selected TEST_IMAGE=${SELECTED_TEST_IMAGE}"
  log "SMART_LAUNCHER_REMOTE=${SMART_LAUNCHER_REMOTE}"
  log "SMART_LAUNCHER_SUITE=${SMART_LAUNCHER_SUITE}"
  log "TEST_CONTAINER_NODES=${TEST_CONTAINER_NODES}"

  update_sglang_test

  mkdir -p "${ARTIFACT_DIR}"
  resolve_nodes | tee "${ARTIFACT_DIR}/nodes.txt" | tee -a "${MAIN_LOG}"
  [[ -s "${ARTIFACT_DIR}/nodes.txt" ]] || die "No nodes resolved from TEST_CONTAINER_NODES"

  while IFS= read -r node; do
    [[ -n "${node}" ]] || continue
    prepare_node_container "${node}" "${SMART_LAUNCHER_CONTAINER_NAME}" "${SELECTED_TEST_IMAGE}"
  done < "${ARTIFACT_DIR}/nodes.txt"
  CONTAINERS_PREPARED=1

  log "Running smart_launcher suite"
  local suite_log="${LOG_DIR}/smart_launcher_suite.log"
  local suite_pid
  set +e
  (
    cd "${SMART_LAUNCHER_DIR}"
    PYTHONPATH=. python3 exp.py exp suite run "${SMART_LAUNCHER_SUITE}" \
      --remote "${SMART_LAUNCHER_REMOTE}"
  ) > >(tee "${suite_log}" | tee -a "${MAIN_LOG}") 2>&1 &
  suite_pid=$!

  while kill -0 "${suite_pid}" 2>/dev/null; do
    if [[ -z "${RUN_ID}" && -f "${suite_log}" ]]; then
      RUN_ID="$(awk '/run_id[[:space:]:]+r_[0-9A-Za-z_:-]+/ { for (i = 1; i <= NF; i++) if ($i ~ /^r_/) v = $i } END { print v }' "${suite_log}")"
      if [[ -n "${RUN_ID}" ]]; then
        log "Detected RUN_ID=${RUN_ID}"
      fi
    fi
    sleep 5
  done
  wait "${suite_pid}"
  SUITE_EXIT_CODE=$?
  set -e

  if [[ -z "${RUN_ID}" ]]; then
    RUN_ID="$(awk '/run_id[[:space:]:]+r_[0-9A-Za-z_:-]+/ { for (i = 1; i <= NF; i++) if ($i ~ /^r_/) v = $i } END { print v }' "${suite_log}")"
  fi
  if [[ -n "${RUN_ID}" ]]; then
    log "Detected RUN_ID=${RUN_ID}"
  else
    log "RUN_ID was not detected from smart_launcher output"
  fi

  if [[ "${SUITE_EXIT_CODE}" != "0" ]]; then
    die "smart_launcher suite failed with exit code ${SUITE_EXIT_CODE}"
  fi
}

main "$@"
