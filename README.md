# lite-kit: compete in the Zils d3-lite soup pool

Tools for training one d3-lite body, testing it locally and packaging it. They call the Zils
repository's own builder, scorer and validator rather than copying them.

**Status.** Everything except GPU work has been run: the 19 tests, data generation, a real soup build,
packaging through the project's validator, the copy-check scenarios, and the scoreboard on synthetic
predictions. `zils.teacher train`, `zils.teacher answer`, `scoreboard.py grade` and the on-chain commit
have **not** been run, because no GPU or wallet was used. Run Step 2's `--limit 500` test first.

| File | Purpose |
| --- | --- |
| `gen_cases.py` | Generates training and dev cases in the public seed format |
| `assemble.py` | Combines cases, relay answers and pin answers into the trainer's file |
| `build_soup.py` | Builds a soup or a single-body candidate with `zils.soup.build` |
| `scoreboard.py` | Grades models (GPU) and tests the bars (no GPU) |
| `own_change.py` | Measures your own change and runs the copy check under queue scenarios |
| `package_body.py` | Makes the body folder and runs `bodies.validate_body` |
| `soup_watch.py` | Reads `zils.ai/miners`, optionally logging each snapshot |
| `env.sh` | Paths and the current soup's revisions |

`data/cases/` already holds generated `train.jsonl` (33,819 rows) and `dev.jsonl` (2,666 rows).

## Step 0: set up the GPU box

Use a 24 GB RTX 4090 with a driver that supports CUDA 13.0, about 200 GB of disk and 64 GB of RAM.
Keep only a hotkey on this box, never a coldkey.

```bash
curl -fsSL https://zils.ai/install.sh | sh     # asks for your wallet and repo at the end
zils-miner setup --network finney --netuid 127 --model d3-lite-0.8b-v1
# copy this folder to the box, then from inside it:
. ./env.sh
"$PY" -m zils.d3 reference --model d3-lite-0.8b-v1 --out data/lite-ref
hf download $QWEN_REPO --revision $QWEN_REV --local-dir data/qwen \
  --include "*.safetensors" --include "model.safetensors.index.json"
```

## Step 1: rebuild the relay answers

`rebuild.py` is published by Zils in its dataset. It reads pinned public sources and hash-checks every
row. Read it first: it was not run while building this kit.

```bash
mkdir -p data/relay && cd data/relay
hf download zils-ai/zils-relay-answers --repo-type dataset --local-dir . \
  --include answers.jsonl --include rebuild.py
"$PY" rebuild.py --records answers.jsonl --out rebuilt.jsonl
cd -
```

It needs `huggingface_hub` and `pyarrow`; install them if `$PY` lacks them. Without relay answers
`assemble.py` still runs and warns, but that mix lacks the teacher data both winning bodies used.

## Step 2: build the soup and train

```bash
. ./env.sh
"$PY" build_soup.py --pool d3-lite --out data/soup-s2 --body $T_REF --body $D_REF

RELAY="--relay-rebuilt data/relay/rebuilt.jsonl --relay-records data/relay/answers.jsonl"
CASES="--cases data/cases/train.jsonl --dev data/cases/dev.jsonl"

# Pin rows: the soup's own answers keep its behaviour where the new data adds no signal.
"$PY" assemble.py $CASES $RELAY --write-pin-pool data/pin-pool.jsonl
"$PY" -m zils.teacher answer --teacher data/soup-s2 --pool data/pin-pool.jsonl --out data/pin.jsonl
"$PY" assemble.py $CASES $RELAY --pin-answers data/pin.jsonl --out data/train.jsonl

# Measure speed and memory on 500 rows first, and delete the test output.
"$PY" -m zils.teacher train --answers data/train.jsonl --reference data/lite-ref \
  --start data/soup-s2 --out data/speedtest --lora 32 --lr 1e-4 --seed 7 --limit 500
rm -rf data/speedtest

"$PY" -m zils.teacher train --answers data/train.jsonl --reference data/lite-ref \
  --start data/soup-s2 --out data/v1 --lora 32 --lr 1e-4 --seed 7
```

Do not pass `--train-answer-layer`: the chassis supplies the answer layer, so that training is ignored.
To make your change more distinct, add your own labelled cases with `assemble.py --extra FILE`.

## Step 3: test before committing

```bash
. ./env.sh
"$PY" build_soup.py --out data/soup-house --body $H1_REF --body $H2_REF --body $H3_REF
"$PY" build_soup.py --out data/cand-T --body $T_REF
"$PY" build_soup.py --out data/cand-D --body $D_REF
"$PY" build_soup.py --out data/soup-s3 --body $T_REF --body $D_REF --body data/v1

"$PY" scoreboard.py grade --dev data/cases/dev.jsonl --cache data/preds \
  house=data/soup-house T=data/cand-T D=data/cand-D S2=data/soup-s2 me=data/v1 S3=data/soup-s3
"$PY" scoreboard.py report --dev data/cases/dev.jsonl --cache data/preds \
  --house house --members T D --soup S2 --mine me --soup-with-mine S3

"$PY" package_body.py --candidate data/v1 --out data/body-v1 --qwen data/qwen \
  --started-from "the uniform average of thermopylae-777/zils-lite-body-2 and dasLOL/zils-lite-skills-v5" \
  --training data/train.manifest.json

"$PY" own_change.py --qwen data/qwen \
  --house "$(body_dir $H1_REF)" "$(body_dir $H2_REF)" "$(body_dir $H3_REF)" "$(body_dir $H4_REF)" \
  --start-house "$(body_dir $H1_REF)" "$(body_dir $H2_REF)" "$(body_dir $H3_REF)" \
  --members "$(body_dir $T_REF)" "$(body_dir $D_REF)" --mine data/body-v1
```

Download `H4_REF` once with `build_soup.py --out data/tmp --body $H4_REF`. Add likely newcomers with
`--rivals FOLDER ...` after downloading them the same way.

**Commit only if** the scoreboard's checks all pass and `own_change.py` prints `OK to commit`. If its
first four order checks fail, the dev set is not measuring what the exam measures: stop and fix that
first. If your gain is close to the bar, regenerate with a larger `--dev-cap-per-family` (default 120),
because dev noise may be as large as the 1.2-point bar.

## Step 4: submit

A submission is two things: the body on Hugging Face, and one commitment on chain pointing at it.
There is no other route, and no port or validator address is needed. Validators read the commitment,
then download from huggingface.co with no login.

First, from a machine holding the coldkey, register a fresh hotkey on netuid 127 following
`docs/bittensor-registration.md`, then copy only the hotkey file to the GPU box.

```bash
REPO=YOU/REPO; BODY=data/body-v1
zils-miner commit-body --pool d3-lite --path $BODY --check
zils-miner commit-body --pool d3-lite --repo $REPO --path $BODY --write-owner

# Upload privately, so nobody can copy the weights during a slow upload.
hf repo create $REPO --private
hf upload $REPO $BODY .                               # note the commit sha it prints

zils-miner commit-body --pool d3-lite --repo $REPO --revision SHA --path $BODY        # dry run

# Publish and commit straight away, in this order.
hf repo settings $REPO --no-private
zils-miner commit-body --pool d3-lite --repo $REPO --revision SHA --path $BODY --yes
```

`--yes` checks the repo with a logged-out request, so it fails on a private repo (HTTP 401), despite
what `docs/mining.md` suggests. The body hash does not cover the ownership file, so the first valid
commitment of a hash wins: keep the gap between publishing and committing to seconds.

Keep the repo public and do not delete that revision. Each new commitment moves your place in the queue
later and replaces this hotkey's slot, so do not recommit an unchanged body, and never use
`commit-candidate`, which soup pools reject. Lite's deadline is 06:00 UTC.

## Step 5: after committing

```bash
python3 soup_watch.py --log data/log.jsonl --hotkey YOURPREFIX
```

Destroy the GPU instance while you wait. When the soup changes, update `T_REF` and `D_REF`, rebuild, and
retrain with new data for the next attempt. Reusing your hotkey is fine: your earlier bodies do not count
against you in the copy check, and a seated body keeps its seat until it leaves the pool.

## Data rules

Never train on the splits the exam draws from: CMUdict test, PIQA validation, 20 Newsgroups test, CxC
test, the Wikipedia politeness corpus, or the private exam. `gen_cases.py` reads only other splits and a
test enforces that. Seed items are never training rows. Word forms and syllables come from dictionaries
the exam also samples, and only CMUdict has a split to respect, so a few words may coincide with exam
items; no held-out list is public to avoid them.

## Tests

```bash
SEEDS=data/seeds python -m unittest test_kit
```
