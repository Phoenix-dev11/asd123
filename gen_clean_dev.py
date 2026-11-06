#!/usr/bin/env python3
"""Build a dev set that T and D cannot have memorized, for evaluation only. Never train on this file.

gen_cases.py's dev rows come from public train splits, and dasLOL trained on those splits, which inflated
D on dev. These rows come from the held-out splits, rendered with the same templates, plus the public seed
items and fresh computed instances. Rows that also occur in a given training file are removed.

    gen_clean_dev.py --out data/cases-clean/dev.jsonl --base data/cases-big/dev.jsonl \
        --seeds data/seeds --train data/train.jsonl --train data/train-v2.jsonl
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import gen_cases as g

GENERATED = ("kit:count", "kit:coverage", "kit:tools-noul", "kit:tools-choice", "kit:tools-score", "kit:unimorph")


def held_out(rng, cap):
    per = max(1, cap // 2)
    for row in g.balanced(g.parquet_rows("baber/piqa", "piqa_validation.parquet"), lambda r: r["label"], per, rng):
        state = {"goal": row["goal"], "method_a": row["sol1"], "method_b": row["sol2"]}
        q = {"type": "noul", "instructions": "Is method A the better way to achieve the stated goal?"}
        yield g.make("reasoning-known-noul-k6", "reasoning", state, q, "true" if row["label"] == 0 else "false", "piqa:val")

    rows = [json.loads(x) for x in open(g.hf_file("SetFit/20_newsgroups", "test.jsonl"))]
    rows = [r for r in rows if 20 <= len(r["text"].strip()) and len(r["text"]) <= 2500]
    q = {"type": "choice", "instructions": "Which topic category does this post belong to?",
         "criteria": {g.LETTERS[i]: n for i, n in enumerate(g.NEWSGROUPS)}}
    for row in g.balanced(rows, lambda r: r["label_text"], max(1, cap // 20), rng):
        label = g.LETTERS[g.NEWSGROUPS.index(row["label_text"])]
        yield g.make("retrieval-known-choice-k6", "retrieval", {"text": row["text"]}, q, label, "20ng:test")

    rows = list(csv.DictReader(open(g.hf_file("DolphyWind/cxc-sts-sentences", "sts_test.csv"), newline="")))
    q = {"type": "score", "instructions": "Rate the semantic similarity of the two captions.",
         "criteria": [f"Similarity level {i} of5" for i in range(6)]}
    for row in g.balanced(rows, lambda r: round(float(r["agg_score"])), max(1, cap // 6), rng):
        state = {"caption1": row["caption1"], "caption2": row["caption2"]}
        yield g.make("language-known-score-k6", "language", state, q, str(round(float(row["agg_score"]))), "cxc:test")

    rows = g.parquet_rows("jc4p/CMUdict", "data/test-00000-of-00001.parquet")
    rows = [r for r in rows if r["text"].isalpha() and 1 <= g.syllables(r["phonemes"]) <= 5]
    q = {"type": "score", "instructions": "How many syllables does this word have in its dictionary pronunciation?",
         "criteria": ["1 syllable", *(f"{i} syllables" for i in range(2, 6))]}
    for row in g.balanced(rows, lambda r: g.syllables(r["phonemes"]), max(1, cap // 5), rng):
        yield g.make("arts-known-score-k6", "arts", {"word": row["text"]}, q, str(g.syllables(row["phonemes"]) - 1), "cmudict:test")

    q = {"type": "noul", "instructions": "Does the passage entail the hypothesis? Answer true or false."}
    for row in g.balanced(g.parquet_rows("nyu-mll/glue", "rte/validation-00000-of-00001.parquet"), lambda r: r["label"], per, rng):
        text = f"Passage: {row['sentence1']}\nHypothesis: {row['sentence2']}"
        yield g.make("language-known-noul-k2", "language", text, q, "true" if row["label"] == 0 else "false", "rte:val")

    q = {"type": "noul", "instructions": "Does the sentence contain the answer to the question? Answer true or false."}
    for row in g.balanced(g.parquet_rows("nyu-mll/glue", "qnli/validation-00000-of-00001.parquet"), lambda r: r["label"], per, rng):
        text = f'Question: {row["question"]}\nSentence: "{row["sentence"]}"'
        yield g.make("retrieval-known-noul-k2", "retrieval", text, q, "true" if row["label"] == 0 else "false", "qnli:val")

    q = {"type": "choice", "instructions": "Which single emotion does this message primarily express?",
         "criteria": {"A": "sadness", "B": "fear", "C": "anger", "D": "surprise", "E": "love", "F": "joy"}}
    letters = {v: k for k, v in q["criteria"].items()}
    for row in g.balanced(g.parquet_rows("dair-ai/emotion", "split/test-00000-of-00001.parquet"), lambda r: r["label"], max(1, cap // 6), rng):
        text = f'Message: "{row["text"]}"'
        yield g.make("arts-known-choice-k3", "arts", text, q, letters[g.EMOTIONS[row["label"]]], "emotion:test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--train", action="append", default=[])
    p.add_argument("--cap", type=int, default=250)
    args = p.parse_args()
    identity = lambda c: json.dumps([c["state"], c["question"]], sort_keys=True)  # noqa: E731
    text = lambda c: json.dumps(c["state"], sort_keys=True)  # noqa: E731
    trained = set()
    for path in args.train:
        trained |= {text(json.loads(x)) for x in open(path)}
    rng = random.Random(20261107)
    base = [json.loads(x) for x in open(args.base)]
    cases = [c for c in base if c.get("source_id") in GENERATED]
    cases += g.seed_dev_cases(args.seeds)
    cases += list(held_out(rng, args.cap))
    out, seen, dropped = [], set(), Counter()
    for c in cases:
        if identity(c) in seen:
            dropped["duplicate"] += 1
        elif text(c) in trained:
            dropped[f"in training: {c.get('source_id', '')[:14]}"] += 1
        else:
            seen.add(identity(c))
            out.append(c)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        for c in out:
            stream.write(json.dumps(c) + "\n")
    print(len(out), "cases; dropped", dict(dropped))
    print(Counter(c.get("source_id", "").split(":")[0][:18] for c in out).most_common(24))
    return 0


if __name__ == "__main__":
    sys.exit(main())
