#!/bin/bash
# bash select_best.sh v1 v2 v3   Scores each variant and every average of two or more of them on the clean
# dev set, runs packaging and the copy check on each, then prints the ranking. It never commits anything.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
. ./env.sh
DEV=data/cases-clean/dev.jsonl; CACHE=data/preds-clean
mkdir -p data/select data/logs
exec > >(tee -a data/logs/select.log) 2>&1
names=("$@")
candidates=("${names[@]}")
if [ "${#names[@]}" -ge 2 ]; then
  while read -r combo; do
    avg="avg-${combo// /-}"
    if [ ! -d "data/$avg" ]; then
      bodies=(); for part in $combo; do bodies+=(--body "data/$part"); done
      "$PY" build_soup.py --out "data/$avg" "${bodies[@]}" || continue
    fi
    candidates+=("$avg")
  done < <(python3 -I -c "
import itertools,sys
n=sys.argv[1:]
for r in range(2,len(n)+1):
    for c in itertools.combinations(n,r): print(' '.join(c))" "${names[@]}")
fi
specs=()
for c in "${candidates[@]}"; do
  soup="data/select/soup-$c"
  [ -d "$soup" ] || "$PY" build_soup.py --out "$soup" --body "$T_REF" --body "$D_REF" --body "data/$c" || continue
  "$PY" scoreboard.py grade --dev "$DEV" --cache "$CACHE" "me-$c=data/$c" "S3-$c=$soup" || continue
  specs+=("me-$c=S3-$c")
  body="data/select/body-$c"
  rm -rf "$body"
  # The model card must describe this candidate's own data. An average lists its first part's draw.
  first="${c#avg-}"; first="${first%%-*}"
  "$PY" package_body.py --candidate "data/$c" --out "$body" --qwen data/qwen \
    --started-from "the uniform average of thermopylae-777/zils-lite-body-2 and dasLOL/zils-lite-skills-v5" \
    --training "data/train-$first.manifest.json" || { echo "PACKAGE FAILED $c"; continue; }
  echo "=== copy check $c"
  CUDA_VISIBLE_DEVICES="" "$PY" own_change.py --qwen data/qwen \
    --house "$(body_dir "$H1_REF")" "$(body_dir "$H2_REF")" "$(body_dir "$H3_REF")" "$(body_dir "$H4_REF")" \
    --start-house "$(body_dir "$H1_REF")" "$(body_dir "$H2_REF")" "$(body_dir "$H3_REF")" \
    --members "$(body_dir "$T_REF")" "$(body_dir "$D_REF")" --mine "$body" 2>&1 | grep -v Warn | cut -c1-170
done
echo "=== ranking"
"$PY" select_report.py --dev "$DEV" --cache "$CACHE" --soup S2 "${specs[@]}"
echo "SELECT-DONE"
