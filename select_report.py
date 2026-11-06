#!/usr/bin/env python3
"""Rank graded candidates by what they add to the soup on the clean dev set, and how broadly.

    select_report.py --dev data/cases-clean/dev.jsonl --cache data/preds-clean --soup S2 \
        NAME=SOUP_WITH_IT_NAME ...
"""
import argparse
from collections import defaultdict

import scoreboard as sb
from zils import soup_grade


def group(case):
    source = case.get("source_id", "")
    return "glaive" if source.startswith("glaive") else source.split(":")[0][:22]


def brier(case, row):
    p = row["probabilities"]
    return sum((p[k] - (1.0 if k == case["label"] else 0.0)) ** 2 for k in p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--soup", required=True)
    parser.add_argument("--min-items", type=int, default=20)
    parser.add_argument("candidates", nargs="+", help="NAME=NAME_OF_SOUP_WITH_IT")
    args = parser.parse_args()
    cases = sb.load_cases(args.dev)
    ids = [c["id"] for c in cases]
    cache = soup_grade.PredictionCache(args.cache)

    def graded(name):
        have = cache.get(name, ids)
        if len(have) != len(ids):
            raise SystemExit(f"{name}: only {len(have)} of {len(ids)} predictions")
        return sb.evaluate(cases, [have[i] for i in ids])

    soup = graded(args.soup)
    groups = defaultdict(list)
    for case in soup["cases"]:
        groups[group(case)].append(case)
    groups = {k: v for k, v in groups.items() if len(v) >= args.min_items}

    def per_group(result):
        rows = {r["id"]: r for r in result["rows"]}
        return {k: sum(brier(c, rows[c["id"]]) for c in v) / len(v) for k, v in groups.items()}

    base = per_group(soup)
    print(f"soup {args.soup} {soup['score']:.2f}; {len(groups)} data sources with {args.min_items}+ items")
    print(f"{'candidate':<14}{'solo':>7}{'gain':>7}{'low95':>7}{'in soup':>9}{'gain':>7}{'low95':>7}{'better':>8}{'worse':>7}  seat crown")
    table = []
    for spec in args.candidates:
        name, _, soup_name = spec.partition("=")
        solo, with_it = graded(name), graded(soup_name)
        a = sb.paired(soup, solo, 500)
        b = sb.paired(soup, with_it, 500)
        groups_now = per_group(with_it)
        better = sum(groups_now[k] < base[k] - 0.002 for k in groups)
        worse = sum(groups_now[k] > base[k] + 0.002 for k in groups)
        seat = b["improvement"] >= sb.MARGIN_SEAT and b["lower_95"] > 0
        crown = a["improvement"] >= sb.MARGIN_SOLO and a["lower_95"] > 0
        table.append((b["improvement"], f"{name:<14}{solo['score']:>7.2f}{a['improvement']:>+7.2f}{a['lower_95']:>+7.2f}"
                      f"{with_it['score']:>9.2f}{b['improvement']:>+7.2f}{b['lower_95']:>+7.2f}{better:>8}{worse:>7}  "
                      f"{'yes ' if seat else 'no  '} {'yes' if crown else 'no'}"))
    for _, line in sorted(table, reverse=True):
        print(line)
    print("Local scores compare candidates; they do not predict the live exam's numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
