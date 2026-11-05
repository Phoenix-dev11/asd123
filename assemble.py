#!/usr/bin/env python3
"""Build the trainer's answers file from generated cases, relay answers and the soup's own answers.

Two stages:

    # 1. Before training: write the pool the current soup will answer.
    assemble.py --cases data/cases/train.jsonl --dev data/cases/dev.jsonl \
        --relay-rebuilt rebuilt.jsonl --relay-records answers.jsonl \
        --write-pin-pool data/pin-pool.jsonl

    # 2. After `python -m zils.teacher answer --teacher SOUP --pool data/pin-pool.jsonl --out data/pin.jsonl`:
    assemble.py --cases data/cases/train.jsonl --dev data/cases/dev.jsonl \
        --relay-rebuilt rebuilt.jsonl --relay-records answers.jsonl \
        --pin-answers data/pin.jsonl --out data/train.jsonl

Output rows are `{"state", "question", "probabilities"}`, the format `zils.teacher train` reads.
Relay and pin rows are optional, so a run without them still works (and reports what is missing).
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from kitpath import add_zils

add_zils()
import zils  # noqa: E402


def read_jsonl(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def identity(state, question):
    return hashlib.sha256(json.dumps([state, question], sort_keys=True).encode()).hexdigest()


def text_of(state):
    return state["text"] if isinstance(state, dict) and set(state) == {"text"} else json.dumps(state)


def probabilities_of(case):
    keys = zils.options(case["question"])
    if case.get("soft"):
        return {k: float(case["soft"][k]) for k in keys}
    return {k: 1.0 if k == case["label"] else 0.0 for k in keys}


def relay_allowed(record):
    """Train-split rows only, and none from sets the exam is known to hold out."""
    index = str(record.get("source_index", ""))
    if ":validation:" in index or ":test:" in index:
        return False, "non-train split"
    name = f"{record.get('source', '')} {record.get('source_dataset', '')}".lower()
    if "vitaminc" in name:
        return False, "held out by the exam"
    return True, ""


def load_relay(rebuilt, records, dev_texts, max_chars):
    meta = {r["i"]: r for r in read_jsonl(records)} if records else {}
    rows, dropped = [], Counter()
    for row in read_jsonl(rebuilt):
        record = meta.get(row.get("i"), {})
        allowed, why = relay_allowed(record)
        if not allowed:
            dropped[why] += 1
        elif "probabilities" not in row:
            dropped["no teacher answer"] += 1
        elif text_of(row["state"]) in dev_texts:
            dropped["text is in dev"] += 1
        elif len(text_of(row["state"])) > max_chars:
            dropped["over the length limit"] += 1
        else:
            rows.append({k: row[k] for k in ("state", "question", "probabilities")})
    return rows, dropped


def pick_relay(rows, count, rng):
    """Relay is about 92% choice, while each exam category weighs all three question types, so keep
    every yes/no and score row first and fill the rest with choice rows."""
    rows = list(rows)
    rng.shuffle(rows)
    rare = [r for r in rows if r["question"]["type"] != "choice"]
    common = [r for r in rows if r["question"]["type"] == "choice"]
    chosen = (rare + common)[:count]
    leftover = (rare + common)[count:]
    return chosen, leftover


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--relay-rebuilt")
    parser.add_argument("--relay-records", help="answers.jsonl, needed to filter relay by split")
    parser.add_argument("--relay-count", type=int, default=40000)
    parser.add_argument("--pin-count", type=int, default=15000)
    parser.add_argument("--pin-answers")
    parser.add_argument("--extra", action="append", default=[], help="more labelled cases, e.g. Outcomes")
    parser.add_argument("--write-pin-pool")
    parser.add_argument("--out")
    parser.add_argument("--max-chars", type=int, default=6000, help="about 1,500 tokens")
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args(argv)
    if bool(args.out) == bool(args.write_pin_pool):
        parser.error("give exactly one of --out and --write-pin-pool")
    rng = random.Random(args.seed)

    dev = list(read_jsonl(args.dev))
    dev_ids = {identity(c["state"], c["question"]) for c in dev}
    dev_texts = {text_of(c["state"]) for c in dev}

    chosen, leftover, dropped = [], [], Counter()
    if args.relay_rebuilt:
        rows, dropped = load_relay(args.relay_rebuilt, args.relay_records, dev_texts, args.max_chars)
        if not args.relay_records:
            print("warning: no --relay-records, so relay rows were not filtered by split", flush=True)
        chosen, leftover = pick_relay(rows, args.relay_count, rng)
    else:
        print("warning: no relay answers; teacher knowledge is missing from this mix", flush=True)

    if args.write_pin_pool:
        pool = [{"state": r["state"], "question": r["question"]} for r in leftover[: args.pin_count]]
        if len(pool) < args.pin_count:
            print(f"warning: pin pool has only {len(pool)} of {args.pin_count} rows", flush=True)
        path = Path(args.write_pin_pool)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            for row in pool:
                stream.write(json.dumps(row) + "\n")
        print(json.dumps({"pin_pool": len(pool), "relay_dropped": dict(dropped)}, indent=1))
        return 0

    train, parts, seen = [], Counter(), set()

    def add(state, question, probabilities, part):
        key = identity(state, question)
        if key in seen or key in dev_ids or text_of(state) in dev_texts:
            parts[f"{part}: dropped (duplicate or in dev)"] += 1
            return
        if len(text_of(state)) > args.max_chars:
            parts[f"{part}: dropped (too long)"] += 1
            return
        keys = zils.options(question)
        if set(probabilities) != set(keys) or abs(sum(probabilities.values()) - 1) > 1e-4:
            raise ValueError(f"{part}: probabilities do not match the question's options")
        seen.add(key)
        parts[part] += 1
        train.append({"state": state, "question": question, "probabilities": probabilities})

    for case in read_jsonl(args.cases):
        add(case["state"], case["question"], probabilities_of(case), "generated and datasets")
    for path in args.extra:
        for case in read_jsonl(path):
            add(case["state"], case["question"], probabilities_of(case), f"extra:{Path(path).name}")
    for row in chosen:
        add(row["state"], row["question"], row["probabilities"], "relay")
    if args.pin_answers:
        for row in read_jsonl(args.pin_answers):
            if "probabilities" in row:  # the teacher keeps over-length rows with an error instead
                add(row["state"], row["question"], row["probabilities"], "pin")
    else:
        print("warning: no pin answers; nothing keeps the soup's behaviour where data adds no signal")

    rng.shuffle(train)  # the trainer holds out the first 1,000 rows of the file
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        for row in train:
            stream.write(json.dumps(row) + "\n")
    manifest = {
        "rows": len(train),
        "parts": dict(sorted(parts.items())),
        "types": dict(Counter(r["question"]["type"] for r in train)),
        "relay_dropped": dict(dropped),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
