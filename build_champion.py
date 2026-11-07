#!/usr/bin/env python3
"""Mix the fixed bank with real Outcomes rows, weighted toward the champion's weak categories.

The bank rows stay, because the fixed exam may still be scored. The Outcomes rows are real
resolved events (Stack Exchange, arXiv, Wikidata, Articles for Creation), not templates.
Stack Exchange score and Wikipedia deletion discussions are left out: the exam's quality
defaults drop them. Quotas are ceilings; a family that collected short is kept as-is.

    build_champion.py --bank data/cases/train.jsonl --dev data/cases/dev.jsonl \
        --outcomes data/outcomes/se.jsonl --outcomes data/outcomes/arxiv.jsonl \
        --outcomes data/outcomes/wiki.jsonl --outcomes data/outcomes/afc.jsonl \
        --out data/cases
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from gen_cases import BANK_TRAIN

# Ceilings for real events. Reasoning and retrieval are the champion's holes
# (15.0 and 32.4 on the 7 Nov slice) and carry most of the crown margin.
REAL_TRAIN = {
    "stackexchange-closed-reasoning": 1400,
    "arxiv-category-math": 900,
    "stackexchange-accepted-reasoning": 3200,
    "wikipedia-afc": 1841,
    "stackexchange-closed-retrieval": 900,
    "wikidata-revert": 1000,
    "arxiv-category-physics": 400,
    "arxiv-category-other": 350,
    "stackexchange-accepted-retrieval": 2800,
    "stackexchange-closed-tools": 400,
    "stackexchange-accepted-tools": 900,
    "arxiv-category-cs": 400,
    "stackexchange-closed-arts": 200,
    "stackexchange-accepted-arts": 600,
    "stackexchange-closed-language": 400,
    "stackexchange-accepted-language": 500,
}
SKIP_PREFIXES = ("stackexchange-score-", "wikipedia-afd")
DEV_PERCENT = 12
DEV_CAP = 120
MAX_CHARS = 6000
SEED = 20261107


def read_jsonl(path):
    rows = []
    with open(path) as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def usable(case):
    state = case.get("state") or {}
    text = state.get("text") if isinstance(state, dict) else None
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
        return False
    if not case.get("question") or "label" not in case:
        return False
    family = case.get("family") or ""
    return not family.startswith(SKIP_PREFIXES)


def holdout(case):
    digest = hashlib.sha256(("champion:" + str(case.get("id"))).encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < DEV_PERCENT


def take(rng, rows, count):
    if len(rows) <= count:
        return list(rows)
    return rng.sample(rows, count)


def mix(rows):
    categories, blocks = Counter(), Counter()
    for case in rows:
        categories[case["category"]] += 1
        blocks[case["block"]] += 1
    total = max(1, len(rows))
    return {
        "rows": len(rows),
        "by_family": dict(sorted(Counter(c["family"] for c in rows).items())),
        "category_share": {k: round(v / total, 3) for k, v in sorted(categories.items())},
        "block_share": {k: round(v / total, 3) for k, v in sorted(blocks.items())},
    }


def both_labels(family, rows):
    """A dev family the local scorer would drop is left out of dev and stays in train."""
    if not rows:
        return rows, []
    kind = rows[0]["question"]["type"]
    labels = [c["label"] for c in rows]
    if kind == "noul":
        if set(labels) >= {"true", "false"}:
            return [], rows
        return rows, []
    if len(set(labels)) >= 2:
        return [], rows
    return rows, []


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--dev", required=True, help="previous dev file; seed rows are kept")
    parser.add_argument("--outcomes", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    bank = {}
    for case in read_jsonl(args.bank):
        if case.get("family") in BANK_TRAIN and usable(case):
            bank.setdefault(case["family"], []).append(case)
    real = {}
    for path in args.outcomes:
        if not Path(path).exists():
            print(f"missing outcomes file, skipped: {path}")
            continue
        for case in read_jsonl(path):
            family = case.get("family")
            if family in REAL_TRAIN and usable(case):
                real.setdefault(family, []).append(case)

    seeds = []
    for case in read_jsonl(args.dev):
        if not str(case.get("id", "")).startswith("kit-") and usable(case):
            seeds.append(case)
    seed_text = {
        (c.get("state") or {}).get("text")
        for c in seeds
        if isinstance((c.get("state") or {}).get("text"), str)
    }

    picked, short = [], {}
    for family, count in {**BANK_TRAIN, **REAL_TRAIN}.items():
        pool = bank.get(family, []) if family in BANK_TRAIN else real.get(family, [])
        chosen = take(rng, pool, count)
        if len(chosen) < count:
            short[family] = {"wanted": count, "have": len(chosen)}
        picked.extend(chosen)

    train, dev = [], []
    dev_counts = Counter()
    for case in picked:
        if (case.get("state") or {}).get("text") in seed_text:
            continue
        if holdout(case) and dev_counts[case["family"]] < DEV_CAP:
            dev.append(case)
            dev_counts[case["family"]] += 1
        else:
            train.append(case)

    by_family = {}
    for case in dev:
        by_family.setdefault(case["family"], []).append(case)
    moved = []
    kept_dev = []
    for family, rows in by_family.items():
        back, stay = both_labels(family, rows)
        moved.extend(back)
        kept_dev.extend(stay)
    dev = kept_dev + seeds
    train.extend(moved)

    seen = set()
    unique = []
    for case in train:
        key = (json.dumps(case["state"], sort_keys=True), json.dumps(case["question"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(case)
    train = unique
    rng.shuffle(train)
    rng.shuffle(dev)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("dev.jsonl", dev)):
        with (out / name).open("w") as stream:
            for case in rows:
                stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    manifest = {
        "seed": args.seed,
        "purpose": "champion chase: bank families kept, real Outcomes rows spent on reasoning and retrieval",
        "score_weights": {
            "reasoning": 0.2585,
            "language": 0.2585,
            "retrieval": 0.2002,
            "tools": 0.1828,
            "arts": 0.1,
            "known": 0.6,
            "new": 0.4,
        },
        "target_category_share": {
            "reasoning": 0.34,
            "retrieval": 0.26,
            "language": 0.18,
            "tools": 0.14,
            "arts": 0.08,
        },
        "train": mix(train),
        "dev": mix(dev),
        "short": short,
        "outcomes": args.outcomes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_category_share": manifest["train"]["category_share"],
        "train_block_share": manifest["train"]["block_share"],
        "short": short,
    }, indent=1))


if __name__ == "__main__":
    main()
