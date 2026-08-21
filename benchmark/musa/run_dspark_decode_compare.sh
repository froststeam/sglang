#!/usr/bin/env bash
set -euo pipefail

# Baseline vs DSpark decode compare for one or more datasets and batch sizes.
# Run on the GPU host after acquiring a GPU lease.
#
# Dataset specs use the form:
#   label|source_jsonl|prepared_jsonl|num_examples
#
# Examples:
#   LEASE_ID=<lease_id> GPU=<gpu_id> bash benchmark/musa/run_dspark_decode_compare.sh
#
#   LEASE_ID=<lease_id> GPU=<gpu_id> BS_LIST="1 8" \
#   DATASET_SPECS="gsm8k|/ipfs/shiven/dataset/gsm8k/test.jsonl|/ipfs/shiven/dataset/gsm8k/deepspec_gsm8k.jsonl|64;math500|/ipfs/shiven/dataset/math500/test.jsonl|/ipfs/shiven/dataset/math500/deepspec_math500.jsonl|32" \
#   TARGET_MODEL_PATH=/ipfs/models/Qwen3-8B \
#   DRAFT_MODEL_PATH=/ipfs/models/dspark_qwen3_8b_block7 \
#   bash benchmark/musa/run_dspark_decode_compare.sh

LEASE_ID="${LEASE_ID:-manual}"
RUN_TAG="${RUN_TAG:-v2}"
CONTAINER="${CONTAINER:-agent-codex-dspark-compare-${RUN_TAG}-${LEASE_ID#lease-}}"
GPU="${GPU:-0}"
TP_SIZE="${TP_SIZE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
PORT="${PORT:-31408}"
HOST_DIR="${HOST_DIR:-/ipfs/shiven/dataset}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-${MODEL_PATH:-/ipfs/models/Qwen3-8B}}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/ipfs/models/dspark_qwen3_8b_block7}"
BS_LIST="${BS_LIST:-1}"
NUM_EXAMPLES="${NUM_EXAMPLES:-64}"
DATASET_SPECS="${DATASET_SPECS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
READY_SLEEP="${READY_SLEEP:-10}"
SEED="${SEED:-980406}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
SGLANG_REPO="${SGLANG_REPO:-/data/shiven/sglang}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
GEMV_AUTOTUNE_TOKENS="${SGLANG_MUSA_GEMV_AUTOTUNE_TOKENS:-0}"
GEMV_AUTOTUNE_WARMUP="${SGLANG_MUSA_GEMV_AUTOTUNE_WARMUP:-3}"
GEMV_AUTOTUNE_ITERS="${SGLANG_MUSA_GEMV_AUTOTUNE_ITERS:-7}"
GEMV_AUTOTUNE_CONFIG="${SGLANG_MUSA_GEMV_AUTOTUNE_CONFIG:-}"
MOE_GEMV_SWIGLU_MAX_TOKENS="${SGLANG_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS:-}"
MOE_CONFIG_DIR="${SGLANG_MOE_CONFIG_DIR:-}"
# TP>1 always replicates the Markov vocab weights. Setting this to 1 additionally
# replicates the full target lm_head on every rank, trading substantial model
# memory for lower communication latency. Replicas are startup snapshots, so
# restart the workers after online target/draft weight updates.
DSPARK_REPLICATE_VOCAB_WEIGHTS="${SGLANG_DSPARK_REPLICATE_VOCAB_WEIGHTS:-0}"
PROFILE="${PROFILE:-0}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-/data/shiven/profile_runs/${RUN_TAG}}"
PROFILE_STEPS="${PROFILE_STEPS:-8}"
PID_FILE="/tmp/dspark_decode_compare_${RUN_TAG}_server.pid"
CSV_PATH="${CSV_PATH:-$HOST_DIR/dspark_decode_compare_${RUN_TAG}.csv}"
IMAGE="${IMAGE:-}"

if [ -z "$IMAGE" ]; then
  for candidate in \
    registry.mthreads.com/mcconline/inference/sglang:v0.5.12.post1-ph1-4.3.5-torch2.9.0-20260805 \
    sh-harbor.mthreads.com/mcctest/sglang:v0.5.12.post1-ph1-4.3.5-torch2.9.0-20260729 \
    registry.mthreads.com/mcconline/inference/sglang:v0.5.12.post1-ph1-4.3.5-torch2.9.0-20260728; do
    if docker image inspect "$candidate" >/dev/null 2>&1; then
      IMAGE="$candidate"
      break
    fi
  done
fi
[ -n "$IMAGE" ] || { echo "no supported SGLang image found locally; set IMAGE=..." >&2; exit 1; }

run_c() {
  docker exec "$CONTAINER" bash -lc "$1"
}

sanitize_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_'
}

build_dataset() {
  local label="$1"
  local source="$2"
  local target="$3"
  mkdir -p "$(dirname "$target")"

  if [ -s "$source" ]; then
    echo "building dataset $label from $source to $target"
    python3 - "$label" "$source" "$target" <<'PY'
import json
import sys
from pathlib import Path

label = sys.argv[1].strip().lower()
src = Path(sys.argv[2])
dst = Path(sys.argv[3])
reasoning_suffix_datasets = {"gsm8k", "math500", "aime24", "aime25"}
append_suffix = label in reasoning_suffix_datasets
suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
rows = []
for line in src.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    prompt = row.get("question")
    if not isinstance(prompt, str):
        prompt = row.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            prompt = prompt[0]
    if not isinstance(prompt, str):
        prompt = row.get("problem")
    if not isinstance(prompt, str):
        turns = row.get("turns")
        if isinstance(turns, list) and turns and isinstance(turns[0], str):
            prompt = turns[0]
        else:
            raise ValueError(f"unsupported row format: {row!r}")
    prompt = prompt.rstrip()
    if append_suffix and suffix not in prompt:
        prompt += suffix
    rows.append(json.dumps({"turns": [prompt]}, ensure_ascii=False))
dst.write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
  elif [ ! -s "$target" ]; then
    if [ "$label" = "gsm8k" ]; then
      echo "downloading DeepSpec GSM8K dataset to $target"
      curl -L --fail --retry 3 --retry-delay 2 \
        https://raw.githubusercontent.com/deepseek-ai/DeepSpec/main/eval_datasets/gsm8k.jsonl \
        -o "$target"
    else
      echo "dataset $label missing: $source / $target" >&2
      exit 1
    fi
  fi
}

if [ -z "$DATASET_SPECS" ]; then
  DATASET_SPECS="gsm8k|${HOST_DIR}/gsm8k/test.jsonl|${HOST_DIR}/gsm8k/deepspec_gsm8k.jsonl|${NUM_EXAMPLES}"
fi

IFS=';' read -r -a DATASET_SPEC_LIST <<< "$DATASET_SPECS"
declare -a DATASET_LABELS DATASET_SOURCES DATASET_TARGETS DATASET_COUNTS
for spec in "${DATASET_SPEC_LIST[@]}"; do
  [ -n "$spec" ] || continue
  IFS='|' read -r label source target count <<< "$spec"
  label="${label:-gsm8k}"
  source="${source:-}"
  target="${target:-${HOST_DIR}/${label}/deepspec_${label}.jsonl}"
  count="${count:-$NUM_EXAMPLES}"
  DATASET_LABELS+=("$label")
  DATASET_SOURCES+=("$source")
  DATASET_TARGETS+=("$target")
  DATASET_COUNTS+=("$count")
done

mkdir -p "$HOST_DIR"
for i in "${!DATASET_LABELS[@]}"; do
  build_dataset "${DATASET_LABELS[$i]}" "${DATASET_SOURCES[$i]}" "${DATASET_TARGETS[$i]}"
done

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker run -d \
    --name "$CONTAINER" \
    --privileged --ipc=host --network=host \
    -v /ipfs/shiven:/data/shiven \
    -v /mnt/nfs/models:/data/models \
    -v /ipfs:/ipfs:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -e MTHREADS_VISIBLE_DEVICES=all \
    -e MUSA_VISIBLE_DEVICES=all \
    -e MTHREADS_DRIVER_CAPABILITIES=all \
    -e HF_ENDPOINT=https://hf-mirror.com \
    -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:${LD_LIBRARY_PATH:-} \
    "$IMAGE" bash -lc 'tail -f /dev/null' >/dev/null
elif ! docker inspect -f '{{.State.Running}}' "$CONTAINER" | grep -q true; then
  docker start "$CONTAINER" >/dev/null
fi

run_c "source /root/.virtualenvs/sglang-default/bin/activate && python -c 'import openai, requests, transformers' >/dev/null 2>&1 || pip install -q openai requests orjson pybase64"
run_c "source /root/.virtualenvs/sglang-default/bin/activate && python -c 'import openai, requests, transformers' >/dev/null"

echo "remote execution source:"
run_c "git config --global --add safe.directory $SGLANG_REPO >/dev/null 2>&1 || true"
run_c "cd $SGLANG_REPO && git status --short && git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD"

stop_server() {
  run_c "
if test -f $PID_FILE; then
  pid=\$(cat $PID_FILE)
  children=\$(pgrep -P \"\$pid\" 2>/dev/null || true)
  kill \$pid \$children 2>/dev/null || true
  sleep 3
  children=\$(pgrep -P \"\$pid\" 2>/dev/null || true)
  kill -9 \$pid \$children 2>/dev/null || true
fi
rm -f $PID_FILE
"
  sleep 3
}

wait_ready() {
  local log="$1"
  for _ in $(seq 1 180); do
    if run_c "curl -fsS --max-time 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1"; then
      sleep "$READY_SLEEP"
      return
    fi
    sleep 5
  done
  echo "server did not become ready; tailing $log" >&2
  run_c "tail -n 120 /data/shiven/dataset/$log" >&2 || true
  exit 1
}

start_server() {
  local mode="$1" bs="$2" model_path="$3" draft_path="$4"
  local extra=""
  [ "$mode" = dspark ] && extra="--speculative-algorithm dspark --speculative-draft-model-path $draft_path --speculative-dspark-block-size 7"
  [ -n "$ATTENTION_BACKEND" ] && extra="$extra --attention-backend $ATTENTION_BACKEND"
  local log="dspark_decode_${RUN_TAG}_${mode}_bs${bs}.log"

  stop_server
  run_c "
set -euo pipefail
source /root/.virtualenvs/sglang-default/bin/activate
cd $SGLANG_REPO
export PYTHONPATH=$SGLANG_REPO/python:/data/shiven/sglang/python
export MUSA_VISIBLE_DEVICES=$GPU
export MTHREADS_VISIBLE_DEVICES=$GPU
export SGLANG_MUSA_GEMV_AUTOTUNE_TOKENS=$GEMV_AUTOTUNE_TOKENS
export SGLANG_MUSA_GEMV_AUTOTUNE_WARMUP=$GEMV_AUTOTUNE_WARMUP
export SGLANG_MUSA_GEMV_AUTOTUNE_ITERS=$GEMV_AUTOTUNE_ITERS
export SGLANG_MUSA_GEMV_AUTOTUNE_CONFIG='$GEMV_AUTOTUNE_CONFIG'
export SGLANG_DSPARK_REPLICATE_VOCAB_WEIGHTS='$DSPARK_REPLICATE_VOCAB_WEIGHTS'
if [ -n '$MOE_GEMV_SWIGLU_MAX_TOKENS' ]; then
  export SGLANG_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS='$MOE_GEMV_SWIGLU_MAX_TOKENS'
fi
if [ -n '$MOE_CONFIG_DIR' ]; then
  export SGLANG_MOE_CONFIG_DIR='$MOE_CONFIG_DIR'
fi
nohup python3 -m sglang.launch_server \
  --model-path $model_path \
  --tokenizer-path $model_path \
  --trust-remote-code \
  --device musa \
  --host 0.0.0.0 \
  --port $PORT \
  --dtype bfloat16 \
  --mem-fraction-static $MEM_FRACTION_STATIC \
  --max-running-requests $bs \
  --cuda-graph-max-bs $bs \
  --tp-size $TP_SIZE \
  --disable-piecewise-cuda-graph \
  --skip-server-warmup \
  --watchdog-timeout 900 \
  --served-model-name qwen3-8b-${RUN_TAG}-$mode-bs$bs \
  $extra \
  > /data/shiven/dataset/$log 2>&1 < /dev/null &
echo \$! > $PID_FILE
"
  wait_ready "$log"
  if [ "$PROFILE" = 1 ]; then
    run_c "curl -fsS -X POST http://127.0.0.1:$PORT/start_profile -H 'content-type: application/json' -d '{\"output_dir\":\"$PROFILE_OUTPUT_DIR/$mode\",\"num_steps\":$PROFILE_STEPS,\"activities\":[\"CPU\",\"GPU\"],\"profile_by_stage\":true,\"merge_profiles\":true,\"profile_prefix\":\"$RUN_TAG-$mode\"}'"
  fi
}

run_client() {
  local mode="$1" bs="$2" dataset_label="$3" dataset_path="$4" num_examples="$5" model_path="$6"
  local dataset_slug
  dataset_slug="$(sanitize_name "$dataset_label")"
  local out="/data/shiven/dataset/${dataset_slug}/dspark_decode_${RUN_TAG}_${dataset_slug}_${mode}_bs${bs}_${num_examples}.json"
  local model="qwen3-8b-${RUN_TAG}-${dataset_slug}-$mode-bs$bs"

  docker exec -i "$CONTAINER" bash -lc \
    "source /root/.virtualenvs/sglang-default/bin/activate && python3 - '$mode' '$model' '$bs' '$out' '$dataset_path' '$model_path' '$PORT' '$num_examples' '$SEED' '$MAX_NEW_TOKENS' '$TEMPERATURE' '$TOP_P' '$TOP_K' '$REQUEST_TIMEOUT'" <<'PY'
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer

(
    mode,
    model,
    bs,
    out,
    dataset,
    model_path,
    port,
    num_examples,
    seed,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    request_timeout,
) = sys.argv[1:]
bs = int(bs)
num_examples = int(num_examples)
seed = int(seed)
max_new_tokens = int(max_new_tokens)
temperature = float(temperature)
top_p = float(top_p)
top_k = int(top_k)
request_timeout = float(request_timeout)

rows = []
for line in Path(dataset).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"bad dataset row: {row!r}")
    rows.append(turns[0])

rng = random.Random(seed)
rng.shuffle(rows)
rows = rows[: min(num_examples, len(rows))]

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
eos_id = tokenizer.eos_token_id
if isinstance(eos_id, int):
    stop_token_ids = [eos_id]
elif eos_id is None:
    stop_token_ids = None
else:
    stop_token_ids = [int(x) for x in eos_id]

client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")

def render(prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

prompts = [render(prompt) for prompt in rows]

def one(item):
    idx, prompt = item
    t0 = time.perf_counter()
    first = None
    text = []
    tokens = 0
    extra_body = {
        "top_k": top_k,
        "sampling_seed": seed + idx,
        "skip_special_tokens": True,
    }
    if stop_token_ids is not None:
        extra_body["stop_token_ids"] = stop_token_ids
    stream = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stream=True,
        stream_options={"include_usage": True},
        timeout=request_timeout,
        extra_body=extra_body,
    )
    for chunk in stream:
        now = time.perf_counter()
        if chunk.usage is not None:
            tokens = chunk.usage.completion_tokens or 0
        for choice in chunk.choices:
            piece = choice.text or ""
            if piece and first is None:
                first = now
            text.append(piece)
    end = time.perf_counter()
    first = first or end
    return {
        "index": idx,
        "decode_s": end - first,
        "request_s": end - t0,
        "completion_tokens": tokens,
        "text": "".join(text),
    }

wall0 = time.perf_counter()
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
    for result in pool.map(one, list(enumerate(prompts))):
        results.append(result)
        done = len(results)
        if done == 1 or done % 10 == 0 or done == len(prompts):
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "bs": bs,
                        "done": done,
                        "tokens_so_far": sum(int(r["completion_tokens"] or 0) for r in results),
                    }
                ),
                flush=True,
            )
wall1 = time.perf_counter()

decode_times = [float(r["decode_s"]) for r in results]
completion_tokens = [int(r["completion_tokens"] or 0) for r in results]
batch_s = sum(max(decode_times[i : i + bs]) for i in range(0, len(decode_times), bs))
total_tokens = sum(completion_tokens)
summary = {
    "mode": mode,
    "model": model,
    "dataset": dataset,
    "num_examples": len(results),
    "num_threads": bs,
    "seed": seed,
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "wall_s": wall1 - wall0,
    "decode_sum_s": sum(decode_times),
    "decode_batch_s": batch_s,
    "decode_avg_s": sum(decode_times) / len(decode_times) if results else 0.0,
    "completion_tokens": total_tokens,
    "decode_batch_tps": total_tokens / batch_s if batch_s else 0,
    "results": results,
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
PY
}

server_accept_length() {
  if [ "$1" = baseline ]; then
    echo 1.0000
  else
    run_c "curl -fsS http://127.0.0.1:$PORT/server_info" |
      python3 -c 'import json,sys; print("%.4f" % json.load(sys.stdin)["internal_states"][0].get("avg_spec_accept_length", 0.0))'
  fi
}

trap stop_server EXIT
mkdir -p "$(dirname "$CSV_PATH")"
printf "dataset,mode,bs,accept_length,server_accept_length,decode_batch_s,decode_batch_tps,completion_tokens\n" > "$CSV_PATH"

for bs in $BS_LIST; do
  if [ "$RUN_BASELINE" = 1 ]; then
    start_server baseline "$bs" "$TARGET_MODEL_PATH" "$DRAFT_MODEL_PATH"
    for i in "${!DATASET_LABELS[@]}"; do
      dataset_label="${DATASET_LABELS[$i]}"
      dataset_path="${DATASET_TARGETS[$i]}"
      dataset_count="${DATASET_COUNTS[$i]}"
      run_client baseline "$bs" "$dataset_label" "$dataset_path" "$dataset_count" "$TARGET_MODEL_PATH"
      result="${HOST_DIR}/$(sanitize_name "$dataset_label")/dspark_decode_${RUN_TAG}_$(sanitize_name "$dataset_label")_baseline_bs${bs}_${dataset_count}.json"
      server_accept=$(server_accept_length baseline)
      accept="$server_accept"
      batch_s=$(python3 -c "import json; print(json.load(open('$result'))['decode_batch_s'])")
      tps=$(python3 -c "import json; print(json.load(open('$result'))['decode_batch_tps'])")
      tokens=$(python3 -c "import json; print(json.load(open('$result'))['completion_tokens'])")
      printf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$dataset_label" baseline "$bs" "$accept" "$server_accept" "$batch_s" "$tps" "$tokens" >> "$CSV_PATH"
    done
    stop_server
  fi

  for i in "${!DATASET_LABELS[@]}"; do
    dataset_label="${DATASET_LABELS[$i]}"
    dataset_path="${DATASET_TARGETS[$i]}"
    dataset_count="${DATASET_COUNTS[$i]}"
    start_server dspark "$bs" "$TARGET_MODEL_PATH" "$DRAFT_MODEL_PATH"
    run_client dspark "$bs" "$dataset_label" "$dataset_path" "$dataset_count" "$TARGET_MODEL_PATH"
    result="${HOST_DIR}/$(sanitize_name "$dataset_label")/dspark_decode_${RUN_TAG}_$(sanitize_name "$dataset_label")_dspark_bs${bs}_${dataset_count}.json"
    server_accept=$(server_accept_length dspark)
    accept="$server_accept"
    batch_s=$(python3 -c "import json; print(json.load(open('$result'))['decode_batch_s'])")
    tps=$(python3 -c "import json; print(json.load(open('$result'))['decode_batch_tps'])")
    tokens=$(python3 -c "import json; print(json.load(open('$result'))['completion_tokens'])")
    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$dataset_label" dspark "$bs" "$accept" "$server_accept" "$batch_s" "$tps" "$tokens" >> "$CSV_PATH"
    stop_server
  done
done

python3 - "$CSV_PATH" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
print("\ndataset    mode       bs  accept_len  server_accept  decode_batch_s  decode_batch_tps  speedup")
for dataset in sorted({x["dataset"] for x in rows}):
    dataset_rows = [x for x in rows if x["dataset"] == dataset]
    for bs in sorted({x["bs"] for x in dataset_rows}, key=int):
        selected = [x for x in dataset_rows if x["bs"] == bs]
        base = next((x for x in selected if x["mode"] == "baseline"), None)
        base_s = None if base is None else float(base["decode_batch_s"])
        for x in selected:
            speedup = float("nan") if base_s is None else base_s / float(x["decode_batch_s"])
            print(
                f"{dataset:<10} {x['mode']:<10} {bs:<3} "
                f"{float(x['accept_length']):>10.4f} "
                f"{float(x['server_accept_length']):>13.4f} "
                f"{float(x['decode_batch_s']):>14.3f} "
                f"{float(x['decode_batch_tps']):>16.2f} "
                f"{speedup:>8.3f}x"
            )
print(f"\nCSV: {sys.argv[1]}")
PY
