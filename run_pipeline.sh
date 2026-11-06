#!/bin/bash
# Runs the lite-kit runbook on this GPU box. It never touches a wallet, Hugging Face uploads or the chain.
#
#   bash run_pipeline.sh all                      prep + baseline + train v1 + check v1
#   bash run_pipeline.sh prep                     install, relay data, soup, pin rows, 500-row speed test
#   bash run_pipeline.sh baseline                 grade the known models and test the dev set's order
#   bash run_pipeline.sh train v2 --lora 64 --lr 1e-4 --seed 11     train another variant
#   bash run_pipeline.sh check v2                 grade a variant, package it, run the copy check
#
# Finished steps are remembered in data/.done, so rerunning after a crash resumes. Output is in data/logs.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p data/logs data/.done

# The exact versions reviewed line by line. A mismatch stops the run until the new version is read.
INSTALL_SHA=a97d3c1d0ac1a217b8259895f04fae0ce159f8d2b411081b848e5e9f435b54d8
REBUILD_SHA=52ea8621f5225a491446efb60e695ff2e243b63a5a3112976cf2b00922537d25

BIG_DEV=data/cases-big/dev.jsonl
LOG="data/logs/${1:-none}${2:+-$2}.log"
exec > >(tee -a "$LOG") 2>&1

# data/logs/LAST_EXIT records how the run ended, and no monitor outlives the script.
MONITOR=""
on_exit() {
  local code=$?
  [ -z "$MONITOR" ] || kill "$MONITOR" 2>/dev/null || true
  echo "$code ${1:-} $(date -u +%H:%M:%S)" > data/logs/LAST_EXIT
}
trap on_exit EXIT
rm -f data/logs/LAST_EXIT

# step NAME COMMAND...: run once, and skip it on later runs if it succeeded.
step() {
  # Not called "name": bash scoping would hide the caller's $name from the command being run.
  local step_id=$1
  shift
  if [ -e "data/.done/$step_id" ]; then
    echo "== $step_id: already done"
    return 0
  fi
  echo "== $step_id: starting $(date -u +%H:%M:%S)"
  "$@"
  touch "data/.done/$step_id"
  echo "== $step_id: finished $(date -u +%H:%M:%S)"
}

verify() {  # verify FILE SHA
  local got
  got=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$got" != "$2" ]; then
    echo "STOP: $1 changed since it was reviewed (got $got). Read the new version before running it."
    exit 1
  fi
}

do_install() {
  local file=data/zils-install.sh
  curl -fsSL https://zils.ai/install.sh -o "$file"
  verify "$file" "$INSTALL_SHA"
  ZILS_SKIP_SETUP=1 NO_COLOR=1 sh "$file"
}

load_env() {
  # shellcheck disable=SC1091
  . ./env.sh
}

do_gpu_check() {
  "$PY" -c "import torch,zils; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('zils from', zils.__file__)"
}

do_reference() { "$PY" -m zils.d3 reference --model d3-lite-0.8b-v1 --out data/lite-ref; }

do_qwen() {
  hf download "$QWEN_REPO" --revision "$QWEN_REV" --local-dir data/qwen \
    --include "*.safetensors" --include "model.safetensors.index.json"
}

do_relay() {
  mkdir -p data/relay
  hf download zils-ai/zils-relay-answers --repo-type dataset --local-dir data/relay \
    --include answers.jsonl --include rebuild.py
  verify data/relay/rebuild.py "$REBUILD_SHA"
  # It exits 1 if any row could not be rebuilt. A few lost rows are fine, so judge by the output file.
  (cd data/relay && "$RELAY_PY" rebuild.py --records answers.jsonl --out rebuilt.jsonl) || true
  local rows
  rows=$(wc -l < data/relay/rebuilt.jsonl)
  echo "relay rows rebuilt: $rows of $(wc -l < data/relay/answers.jsonl)"
  [ "$rows" -gt 1000 ] || { echo "STOP: relay rebuild produced too few rows"; exit 1; }
}

do_big_dev() {
  # A larger dev set lowers the noise. Splits are routed by hash, so these rows never appear in training.
  "$RELAY_PY" gen_cases.py --out data/cases-big --seeds data/seeds --dev-cap-per-family 400
  wc -l "$BIG_DEV"
}

do_soup() { "$PY" build_soup.py --pool d3-lite --out data/soup-s2 --body "$T_REF" --body "$D_REF"; }

assemble_args() {
  ASSEMBLE=(--cases data/cases/train.jsonl --dev "$BIG_DEV"
            --relay-rebuilt data/relay/rebuilt.jsonl --relay-records data/relay/answers.jsonl)
}

do_pin_pool() { assemble_args; "$PY" assemble.py "${ASSEMBLE[@]}" --write-pin-pool data/pin-pool.jsonl; }
do_pin() { "$PY" -m zils.teacher answer --teacher data/soup-s2 --pool data/pin-pool.jsonl --out data/pin.jsonl; }
do_assemble() { assemble_args; "$PY" assemble.py "${ASSEMBLE[@]}" --pin-answers data/pin.jsonl --out data/train.jsonl; }

do_speedtest() {
  rm -rf data/speedtest
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -l 5 > data/logs/speedtest-gpu.log &
  MONITOR=$!
  local started=$SECONDS
  "$PY" -m zils.teacher train --answers data/train.jsonl --reference data/lite-ref \
    --start data/soup-s2 --out data/speedtest --lora 32 --lr 1e-4 --seed 7 --limit 500
  kill "$MONITOR" || true
  MONITOR=""
  echo "speed test: 500 rows, load and save included, in $((SECONDS - started)) s"
  echo "peak GPU memory: $(sort -n data/logs/speedtest-gpu.log | tail -1)"
  rm -rf data/speedtest
}

prep() {
  step install do_install
  load_env
  step gpu-check do_gpu_check
  step reference do_reference
  step qwen do_qwen
  step relay do_relay
  step big-dev do_big_dev
  step soup do_soup
  step pin-pool do_pin_pool
  step pin do_pin
  step assemble do_assemble
  step speedtest do_speedtest
}

do_baseline_models() {
  "$PY" build_soup.py --out data/soup-house --body "$H1_REF" --body "$H2_REF" --body "$H3_REF"
  "$PY" build_soup.py --out data/cand-T --body "$T_REF"
  "$PY" build_soup.py --out data/cand-D --body "$D_REF"
  "$PY" build_soup.py --out data/tmp-h4 --body "$H4_REF"
  rm -rf data/tmp-h4
}

baseline() {
  load_env
  step baseline-models do_baseline_models
  "$PY" scoreboard.py grade --dev "$BIG_DEV" --cache data/preds \
    house=data/soup-house T=data/cand-T D=data/cand-D S2=data/soup-s2
  "$PY" scoreboard.py report --dev "$BIG_DEV" --cache data/preds --house house --members T D --soup S2
}

train() {
  local name=$1
  shift
  load_env
  [ -s data/train.jsonl ] || { echo "STOP: run prep first"; exit 1; }
  do_train() {
    "$PY" -m zils.teacher train --answers data/train.jsonl --reference data/lite-ref \
      --start data/soup-s2 --out "data/$name" "$@"
  }
  rm -rf "data/$name.partial"
  [ -e "data/.done/train-$name" ] || rm -rf "data/${name:?}"
  step "train-$name" do_train "$@"
}

check() {
  local name=$1
  load_env
  local soup="data/soup-s3-$name" body="data/body-$name" status=0
  [ -e "data/.done/soup-$name" ] || rm -rf "$soup"
  step "soup-$name" "$PY" build_soup.py --out "$soup" --body "$T_REF" --body "$D_REF" --body "data/$name"
  "$PY" scoreboard.py grade --dev "$BIG_DEV" --cache data/preds \
    house=data/soup-house T=data/cand-T D=data/cand-D S2=data/soup-s2 "me-$name=data/$name" "S3-$name=$soup"
  "$PY" scoreboard.py report --dev "$BIG_DEV" --cache data/preds --house house --members T D --soup S2 \
    --mine "me-$name" --soup-with-mine "S3-$name" || status=1
  rm -rf "$body"
  "$PY" package_body.py --candidate "data/$name" --out "$body" --qwen data/qwen \
    --started-from "the uniform average of thermopylae-777/zils-lite-body-2 and dasLOL/zils-lite-skills-v5" \
    --training data/train.manifest.json
  "$PY" own_change.py --qwen data/qwen \
    --house "$(body_dir "$H1_REF")" "$(body_dir "$H2_REF")" "$(body_dir "$H3_REF")" "$(body_dir "$H4_REF")" \
    --start-house "$(body_dir "$H1_REF")" "$(body_dir "$H2_REF")" "$(body_dir "$H3_REF")" \
    --members "$(body_dir "$T_REF")" "$(body_dir "$D_REF")" --mine "$body" || status=1
  [ "$status" -eq 0 ] && echo "RESULT $name: all checks passed" || echo "RESULT $name: at least one check failed"
}

case "${1:-}" in
  all)
    prep
    baseline || echo "baseline: some order checks failed; read them before trusting later scores"
    train v1 --lora 32 --lr 1e-4 --seed 7
    check v1
    ;;
  prep) prep ;;
  baseline) baseline ;;
  train) shift; train "$@" ;;
  check) check "$2" ;;
  *) sed -n 2,9p "$0"; exit 2 ;;
esac
