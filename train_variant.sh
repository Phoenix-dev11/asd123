#!/bin/bash
# bash train_variant.sh NAME DATA_SEED TRAINER_ARGS...   Trains data/NAME from the current soup on its own
# relay draw. Runs beside run_pipeline.sh; the result is written to data/logs/exit-NAME.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
name=$1; data_seed=$2; shift 2
log="data/logs/$name.log"
rm -f "data/logs/exit-$name"
{
  . ./env.sh
  train="data/train-$name.jsonl"
  "$PY" assemble.py --cases data/cases/train.jsonl --dev data/cases-big/dev.jsonl \
    --relay-rebuilt data/relay/rebuilt.jsonl --relay-records data/relay/answers.jsonl \
    --pin-answers data/pin.jsonl --seed "$data_seed" --max-chars "${MAX_CHARS:-6000}" --out "$train" | tail -n 3 || exit 1
  rm -rf "data/$name"
  "$PY" -m zils.teacher train --answers "$train" --reference data/lite-ref \
    --start data/soup-s2 --out "data/$name" "$@"
  code=$?
  echo "$code $(date -u +%H:%M:%S)" > "data/logs/exit-$name"
  exit "$code"
} >> "$log" 2>&1
