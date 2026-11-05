#!/usr/bin/env python3
"""Local scoreboard: grade models on the dev set with the validators' scoring and test the bars.

GPU step, under the d3 runtime (only missing predictions are computed, so it can resume):

    scoreboard.py grade --dev data/cases/dev.jsonl --cache data/preds \
        house=data/soup-house T=data/body-t D=data/body-d S2=data/soup-s2 me=data/v1 S3=data/soup-s3

No-GPU step:

    scoreboard.py report --dev data/cases/dev.jsonl --cache data/preds \
        --house house --members T D --soup S2 --mine me --soup-with-mine S3

A dev score is not an exam score. The private exam's items and `new` families are unknown, so use
this to rank models and to catch regressions, not to predict a validator's number.
"""

import argparse
import json
import sys
from pathlib import Path

from kitpath import add_zils

add_zils()
import zils  # noqa: E402
from zils import base_metrics, soup_grade  # noqa: E402

SECRET = b"lite-kit-fixed-dev-calibration-split"
MARGIN_SOLO = 1.2  # solo score over the soup: the fine-tune gate and the crown bar
MARGIN_SEAT = 0.8  # soup with the body over soup without it: the paid seat's floor


def load_cases(path):
    cases = []
    with open(path) as stream:
        for line in stream:
            if line.strip():
                case = json.loads(line)
                case.pop("soft", None)  # a training hint, not part of an exam case
                cases.append(case)
    return usable(cases)


def usable(cases):
    """Drop any family the scorer would reject, and say so, rather than fail after GPU time is spent."""
    families = {}
    for case in cases:
        families.setdefault(case["family"], []).append(case)
    kept = []
    for family, rows in families.items():
        try:
            kind = rows[0]["question"]["type"]
            options = [zils.options(c["question"]) for c in rows]
            labels = [c["label"] for c in rows]
            if kind == "noul":
                bad = set(labels) != {"false", "true"}
            else:
                bad = base_metrics._constant_losses(options, labels, kind) <= 1e-12
        except (ValueError, KeyError) as error:
            print(f"dropping dev family {family}: {error}", flush=True)
            continue
        if bad:
            print(f"dropping dev family {family}: its baseline is degenerate", flush=True)
            continue
        kept += rows
    return kept


def grade_all(args):
    cache = soup_grade.PredictionCache(args.cache)
    cases = load_cases(args.dev)
    for spec in args.models:
        name, _, folder = spec.partition("=")
        path = Path(folder)
        if not (path / "model.json").exists():
            raise SystemExit(f"{path}: needs model.json; make it with build_soup.py --body {path}")
        rows = soup_grade.grade(name, cases, cache, soup_grade.production_runner(path))
        print(f"{name}: {len(rows)} rows", flush=True)


def evaluate(cases, rows):
    """Score, temperature and the tempered scored rows, using the validators' own code path."""
    result = soup_grade.evaluate(cases, rows, scoring="calibrated_brier", secret=SECRET)
    held = set(result["calibration_ids"])
    tempered = base_metrics.temper(cases, rows, result["temperature"])
    scored_cases = [c for c in cases if c["id"] not in held]
    wanted = {c["id"] for c in scored_cases}
    return {
        "score": result["score"],
        "temperature": result["temperature"],
        "metrics": result["metrics"],
        "cases": scored_cases,
        "rows": [r for r in tempered if r["id"] in wanted],
    }


def paired(a, b, resamples, seed=553):
    """b minus a, with a paired bootstrap on the items both are scored on."""
    return base_metrics.paired_improvement(
        a["cases"], a["rows"], b["rows"], resamples=resamples, seed=seed, metric="brier"
    )


def report(args):
    cases = load_cases(args.dev)
    cache = soup_grade.PredictionCache(args.cache)
    ids = [c["id"] for c in cases]

    def graded(name):
        have = cache.get(name, ids)
        if len(have) != len(ids):
            raise SystemExit(f"{name}: only {len(have)} of {len(ids)} predictions; run grade first")
        return evaluate(cases, [have[i] for i in ids])

    names = [args.house, *args.members, args.soup]
    if args.mine:
        names.append(args.mine)
    if args.soup_with_mine:
        names.append(args.soup_with_mine)
    results = {name: graded(name) for name in dict.fromkeys(names)}

    print(f"{'model':<14}{'score':>8}{'temperature':>13}")
    for name, result in results.items():
        print(f"{name:<14}{result['score']:>8.2f}{result['temperature']:>13.3f}")

    categories = results[args.soup]["metrics"].get("categories")
    if isinstance(categories, dict):
        print("\nBy category:")
        print(" " * 14 + "".join(f"{c[:10]:>11}" for c in categories))
        for name, result in results.items():
            values = result["metrics"]["categories"]
            cells = (f"{float(values[c]):>11.2f}" if _is_number(values[c]) else f"{'':>11}"
                     for c in categories)  # fmt: skip
            print(f"{name:<14}" + "".join(cells))

    checks = []

    def check(label, passed, detail):
        checks.append(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")

    def gap(low, high):
        return paired(results[low], results[high], args.resamples)

    print("\nDoes the dev set reproduce the known order? If not, fix it before trusting it.")
    for member in args.members:
        g = gap(member, args.soup)
        check(f"{args.soup} above {member}", g["improvement"] > 0, _fmt(g))
        g = gap(args.house, member)
        check(f"{member} above {args.house}", g["improvement"] > 0, _fmt(g))

    if args.mine:
        print("\nDoes your body clear the bars?")
        g = gap(args.soup, args.mine)
        check(f"solo at least {MARGIN_SOLO} above {args.soup}", g["improvement"] >= MARGIN_SOLO, _fmt(g))
        check("that gain is above noise", g["lower_95"] > 0, f"lower 95% {g['lower_95']:+.2f}")
    if args.soup_with_mine:
        g = gap(args.soup, args.soup_with_mine)
        check(f"soup with you at least {MARGIN_SEAT} above soup without you",
              g["improvement"] >= MARGIN_SEAT, _fmt(g))  # fmt: skip
    print(f"\n{sum(checks)} of {len(checks)} checks passed. Only 'known' and stand-in 'new' dev "
          "families are covered, so a pass is necessary, not sufficient.")  # fmt: skip
    return 0 if all(checks) else 1


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(gap):
    return f"{gap['improvement']:+.2f} [{gap['lower_95']:+.2f}, {gap['upper_95']:+.2f}]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    grade = commands.add_parser("grade")
    grade.add_argument("--dev", required=True)
    grade.add_argument("--cache", required=True)
    grade.add_argument("models", nargs="+", help="name=checkpoint_folder")
    rep = commands.add_parser("report")
    rep.add_argument("--dev", required=True)
    rep.add_argument("--cache", required=True)
    rep.add_argument("--house", required=True)
    rep.add_argument("--members", nargs="+", required=True)
    rep.add_argument("--soup", required=True)
    rep.add_argument("--mine")
    rep.add_argument("--soup-with-mine")
    rep.add_argument("--resamples", type=int, default=500)
    args = parser.parse_args(argv)
    if args.command == "grade":
        grade_all(args)
        return 0
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
