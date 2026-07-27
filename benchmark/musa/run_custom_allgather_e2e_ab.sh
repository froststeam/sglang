#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DRY_RUN="${DRY_RUN:-1}"
GPU_LEASE_ID="${GPU_LEASE_ID:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/home/dist/models/DeepSeek-V2-Lite}"
DATASET_PATH="${DATASET_PATH:-/home/dist/jzxue/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-24586}"
TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-1}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
CASE_PROFILE="${CASE_PROFILE:-default}"
BENCH_REPEATS="${BENCH_REPEATS:-1}"
MODE_ORDER="${MODE_ORDER:-mccl,custom_ag}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmark-results/custom-all-gather-e2e-$(date +%Y%m%d_%H%M%S)}"

SERVER_PID=""
CURRENT_MODE=""

die() {
  echo "ERROR: $*" >&2
  exit 1
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

COMMON_ENV=(
  "PYTHONPATH=$REPO_ROOT/python"
  "LD_LIBRARY_PATH=/usr/local/musa-4.3.5/lib:/usr/lib/x86_64-linux-gnu"
  "MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  "MUSA_LAUNCH_BLOCKING=1"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "LOCAL_WORLD_SIZE=8"
  "GPUS_PER_NODE=8"
  "SGLANG_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS=0"
  "SGLANG_CUSTOM_AG_MAX_SIZE_BYTES=83886080"
  "SGLANG_CUSTOM_AG_THREADS=512"
  "SGLANG_CUSTOM_AG_BLOCKS=48"
)

SERVER_ARGS=(
  -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --served-model-name "$MODEL_PATH"
  --trust-remote-code
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --expert-parallel-size "$EP_SIZE"
  --attention-backend fa3
  --prefill-attention-backend fa3
  --sampling-backend flashinfer
  --mem-fraction-static 0.80
  --context-length "$CONTEXT_LENGTH"
  --max-running-requests 8
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --disable-cuda-graph
  --disable-piecewise-cuda-graph
  --disable-overlap-schedule
  --disable-custom-all-reduce
  --disable-radix-cache
  --skip-server-warmup
)

# name:input_len:output_len:num_prompts:max_concurrency
DEFAULT_CASES=(
  "prefill_8k_1:8192:1:4:1"
  "mixed_4k_32:4096:32:8:4"
  "decode_1k_128:1024:128:4:1"
)

BOUNDARY_CASES=(
  "prefill_128_1:128:1:8:1"
  "prefill_1k_1:1024:1:8:1"
  "prefill_4k_1:4096:1:8:1"
  "prefill_8k_1:8192:1:8:1"
  "prefill_16k_1:16384:1:4:1"
  "mixed_4k_32_c1:4096:32:8:1"
  "mixed_4k_32_c2:4096:32:8:2"
  "mixed_4k_32_c4:4096:32:8:4"
  "mixed_4k_32_c8:4096:32:8:8"
  "decode_1k_128_c1:1024:128:8:1"
  "decode_1k_128_c2:1024:128:8:2"
  "decode_1k_128_c4:1024:128:8:4"
  "decode_1k_128_c8:1024:128:8:8"
)

STABILITY_CASES=(
  "prefill_128_1:128:1:8:1"
  "prefill_1k_1:1024:1:8:1"
  "prefill_16k_1:16384:1:4:1"
  "mixed_4k_32_c2:4096:32:8:2"
  "mixed_4k_32_c4:4096:32:8:4"
  "decode_1k_128_c4:1024:128:8:4"
  "decode_1k_128_c8:1024:128:8:8"
)

PREFILL_BOUNDARY_CASES=(
  "prefill_64_1:64:1:32:1"
  "prefill_128_1:128:1:32:1"
  "prefill_256_1:256:1:32:1"
  "prefill_512_1:512:1:32:1"
  "prefill_768_1:768:1:32:1"
  "prefill_1k_1:1024:1:32:1"
)

PREFILL_LONG_CASES=(
  "prefill_64_1:64:1:256:1"
  "prefill_128_1:128:1:256:1"
  "prefill_256_1:256:1:256:1"
  "prefill_512_1:512:1:256:1"
  "prefill_768_1:768:1:256:1"
  "prefill_1k_1:1024:1:256:1"
  "prefill_16k_1:16384:1:64:1"
)

PREFILL_FINAL_CASES=(
  "prefill_128_1:128:1:256:1"
  "prefill_256_1:256:1:256:1"
  "prefill_768_1:768:1:256:1"
  "prefill_1k_1:1024:1:256:1"
)

case "$CASE_PROFILE" in
  default) CASES=("${DEFAULT_CASES[@]}") ;;
  boundary) CASES=("${BOUNDARY_CASES[@]}") ;;
  stability) CASES=("${STABILITY_CASES[@]}") ;;
  prefill_boundary) CASES=("${PREFILL_BOUNDARY_CASES[@]}") ;;
  prefill_long) CASES=("${PREFILL_LONG_CASES[@]}") ;;
  prefill_final) CASES=("${PREFILL_FINAL_CASES[@]}") ;;
  *) die "CASE_PROFILE must be default, boundary, stability, prefill_boundary, prefill_long, or prefill_final" ;;
esac

validate_inputs() {
  [[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "$TP_SIZE" == "8" && "$EP_SIZE" == "1" ]] || die "This A/B plan requires TP_SIZE=8 and EP_SIZE=1"
  [[ "$BENCH_REPEATS" =~ ^[1-9][0-9]*$ ]] || die "BENCH_REPEATS must be a positive integer"
  [[ "$MODE_ORDER" == "mccl,custom_ag" || "$MODE_ORDER" == "custom_ag,mccl" ]] || \
    die "MODE_ORDER must be mccl,custom_ag or custom_ag,mccl"

  if [[ "$DRY_RUN" == "0" ]]; then
    [[ -n "$GPU_LEASE_ID" ]] || die "GPU_LEASE_ID is required when DRY_RUN=0"
    [[ -d "$MODEL_PATH" ]] || die "Model directory not found: $MODEL_PATH"
    [[ -f "$MODEL_PATH/config.json" ]] || die "Missing model config: $MODEL_PATH/config.json"
    [[ -f "$DATASET_PATH" ]] || die "Dataset not found: $DATASET_PATH"
    command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN"
    command -v curl >/dev/null || die "curl is required"
    mkdir -p "$RESULTS_DIR"
  elif [[ ! -d "$MODEL_PATH" ]]; then
    echo "WARN: model directory is not visible in dry-run environment: $MODEL_PATH"
  fi
}

server_env_for_mode() {
  local mode="$1"
  case "$mode" in
    mccl) echo "SGLANG_MUSA_USE_JIT_ALL_GATHER=0" ;;
    custom_ag) echo "SGLANG_MUSA_USE_JIT_ALL_GATHER=1" ;;
    *) die "Unknown mode: $mode" ;;
  esac
}

start_server() {
  local mode="$1"
  local mode_env
  local log_file="$RESULTS_DIR/${mode}_server.log"
  mode_env="$(server_env_for_mode "$mode")"

  echo "[$mode] start server"
  print_command env "${COMMON_ENV[@]}" "$mode_env" "$PYTHON_BIN" "${SERVER_ARGS[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  stdout/stderr -> $log_file"
    return
  fi

  if curl -fsS --max-time 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    die "Port $PORT already serves a model; stop the existing service first"
  fi

  env "${COMMON_ENV[@]}" "$mode_env" \
    setsid "$PYTHON_BIN" "${SERVER_ARGS[@]}" >"$log_file" 2>&1 &
  SERVER_PID=$!
  CURRENT_MODE="$mode"
}

wait_ready() {
  local mode="$1"
  echo "[$mode] wait for http://$HOST:$PORT/v1/models (timeout=${SERVER_READY_TIMEOUT}s)"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  for _ in $(seq 1 "$SERVER_READY_TIMEOUT"); do
    if curl -fsS --max-time 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
      echo "[$mode] service ready"
      return
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || die "$mode server exited before becoming ready"
    sleep 1
  done
  die "$mode server readiness timeout"
}

run_bench() {
  local mode="$1"
  local name="$2"
  local input_len="$3"
  local output_len="$4"
  local num_prompts="$5"
  local concurrency="$6"
  local repeat="${7:-}"
  local output_file="$RESULTS_DIR/${mode}_${name}.jsonl"
  local -a cmd

  if [[ -n "$repeat" ]]; then
    output_file="$RESULTS_DIR/${mode}_${name}_r${repeat}.jsonl"
  fi

  cmd=(
    "$PYTHON_BIN" -m sglang.bench_serving
    --backend sglang
    --base-url "http://$HOST:$PORT"
    --dataset-name random
    --dataset-path "$DATASET_PATH"
    --model "$MODEL_PATH"
    --tokenizer "$MODEL_PATH"
    --num-prompts "$num_prompts"
    --random-input-len "$input_len"
    --random-output-len "$output_len"
    --random-range-ratio 1.0
    --max-concurrency "$concurrency"
    --ready-check-timeout-sec 0
    --disable-tqdm
    --output-file "$output_file"
  )

  echo "[$mode] benchmark $name"
  print_command env \
    "PYTHONPATH=$REPO_ROOT/python" \
    "LD_LIBRARY_PATH=/usr/local/musa-4.3.5/lib:/usr/lib/x86_64-linux-gnu" \
    "HF_HUB_OFFLINE=1" \
    "TRANSFORMERS_OFFLINE=1" \
    "${cmd[@]}"
  if [[ "$DRY_RUN" == "0" ]]; then
    env \
      "PYTHONPATH=$REPO_ROOT/python" \
      "LD_LIBRARY_PATH=/usr/local/musa-4.3.5/lib:/usr/lib/x86_64-linux-gnu" \
      "HF_HUB_OFFLINE=1" \
      "TRANSFORMERS_OFFLINE=1" \
      "${cmd[@]}"
  fi
}

run_mode() {
  local mode="$1"
  local repeat repeat_suffix spec name input_len output_len num_prompts concurrency

  start_server "$mode"
  wait_ready "$mode"

  run_bench "$mode" warmup 128 1 1 1
  for repeat in $(seq 1 "$BENCH_REPEATS"); do
    repeat_suffix=""
    if (( BENCH_REPEATS > 1 )); then
      repeat_suffix="$repeat"
    fi
    for spec in "${CASES[@]}"; do
      IFS=: read -r name input_len output_len num_prompts concurrency <<<"$spec"
      run_bench "$mode" "$name" "$input_len" "$output_len" "$num_prompts" "$concurrency" "$repeat_suffix"
    done
  done

  echo "[$mode] stop server"
  if [[ "$DRY_RUN" == "0" ]]; then
    cleanup_server
    sleep 5
  fi
}

validate_inputs

echo "Custom All Gather end-to-end A/B"
echo "  dry_run=$DRY_RUN"
echo "  lease_id=${GPU_LEASE_ID:-<required-for-real-run>}"
echo "  model=$MODEL_PATH"
echo "  dataset=$DATASET_PATH"
echo "  results=$RESULTS_DIR"
echo "  tp=$TP_SIZE ep=$EP_SIZE chunked_prefill=$CHUNKED_PREFILL_SIZE"
echo "  case_profile=$CASE_PROFILE repeats=$BENCH_REPEATS mode_order=$MODE_ORDER"

IFS=, read -r -a MODES <<<"$MODE_ORDER"
for mode in "${MODES[@]}"; do
  run_mode "$mode"
done

echo "Done. Compare MCCL and Custom AG JSONL files under: $RESULTS_DIR"
