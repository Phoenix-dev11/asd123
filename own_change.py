#!/usr/bin/env python3
"""Check a body's own change: size, independence, and the project's copy check under queue scenarios.

    own_change.py --qwen QWEN --house H1 H2 H3 H4 --start-house H1 H2 H3 \
        --members T D --mine MY_BODY [--rivals X Y]

CPU only; reads safetensors one tensor at a time. The checks call `zils.soup_copies`, using exact
inner products in place of the validators' private random sketches, which approximate them.

Why scenarios: a queued body is admitted against whatever the soup is by then, so it should survive
the soup changing. Rivals are bodies that may be seated before yours is admitted.
"""

import argparse
import glob
import json
import sys
from itertools import combinations_with_replacement

import numpy as np

from kitpath import add_zils

add_zils()
from zils import soup_copies as copies  # noqa: E402

MIN_SIZE = 9.0  # about thermopylae's own change; below it the simulated copy check can refuse a soup-start body
SCOUT_LINE = 0.05


def tensor_index(folder):
    from safetensors import safe_open

    index = {}
    for path in sorted(glob.glob(f"{folder}/**/*.safetensors", recursive=True)):
        with safe_open(path, "pt") as handle:
            for key in handle.keys():
                index[key[6:] if key.startswith("model.") else key] = (path, key)
    return index


def read(index, name):
    import torch
    from safetensors import safe_open

    path, key = index[name]
    with safe_open(path, "pt") as handle:
        return handle.get_tensor(key).to(torch.float32).flatten()


def gram_matrix(qwen, folders):
    """Inner products of each body's change from the parent, language layers only."""
    parent = tensor_index(qwen)
    names = sorted(n for n in parent if n.startswith("language_model."))
    indexes = [tensor_index(f) for f in folders]
    for folder, index in zip(folders, indexes, strict=True):
        if missing := [n for n in names if n not in index]:
            raise SystemExit(f"{folder}: lacks {missing[0]}")
    size = len(folders)
    gram, parent_norm = np.zeros((size, size)), 0.0
    for name in names:
        base = read(parent, name)
        parent_norm += float(base.dot(base))
        deltas = [read(index, name) - base for index in indexes]
        for i, j in combinations_with_replacement(range(size), 2):
            value = float(deltas[i].dot(deltas[j]))
            gram[i, j] += value
            if i != j:
                gram[j, i] += value
    return gram, parent_norm**0.5


def embed(gram):
    """Vectors whose inner products equal `gram`."""
    values, vectors = np.linalg.eigh(gram)
    return vectors * np.sqrt(np.clip(values, 0, None))


def analyse(gram, parent_norm, labels, *, house, start_house, members, mine, rivals):
    index = {name: i for i, name in enumerate(labels)}
    vectors = {name: row for name, row in zip(labels, embed(gram), strict=True)}

    def combo(weights):
        out = np.zeros(len(labels))
        for name, weight in weights.items():
            out[index[name]] += weight
        return out

    def dot(a, b):
        return float(a @ gram @ b)

    def size(a):
        return dot(a, a) ** 0.5

    me = combo({mine: 1})
    soup = combo({m: 1 / len(members) for m in members})
    start = combo({h: 1 / len(start_house) for h in start_house})
    from_soup = me - soup
    from_start = me - start
    report = {
        "relative_change_from_parent": size(me) / parent_norm,
        "own_change_from_soup": size(from_soup),
        "own_change_from_house_average": size(from_start),
        "cosine_with_members_own_change": {},
    }
    own = from_soup if size(from_soup) <= size(from_start) else from_start
    report["measured_from"] = "soup" if own is from_soup else "house average"
    report["own_change"] = size(own)
    for member in members:
        theirs = combo({member: 1}) - start
        report["cosine_with_members_own_change"][member] = dot(own, theirs) / (
            size(own) * size(theirs)
        )

    house_vectors = [vectors[h] for h in house]
    release = vectors[house[0]]
    earlier = [vectors[m] for m in [*members, *rivals]]

    def soup_of(names):
        return sum(vectors[n] for n in names) / len(names)

    scenarios = {"soup unchanged": list(members)}
    for rival in rivals:
        scenarios[f"{rival} joins"] = [*members, rival]
        scenarios[f"{rival} replaces {members[-1]}"] = [*members[:-1], rival]
    scenarios[f"{members[0]} alone"] = [members[0]]
    target = vectors[mine]
    report["scenarios"] = {}
    for name, seated in scenarios.items():
        current = soup_of(seated)
        reason = copies.is_copy(
            target, earlier, soup=current, release=release, house=house_vectors, explained=earlier
        )
        residual = copies.base_residual(target, house=house_vectors, soup=current, release=release)
        named = {f"house{i}": v for i, v in enumerate(house_vectors)}
        named["soup"] = current
        named["release"] = release
        explained = sum(copies.base_shares(target, named).values())
        report["scenarios"][name] = {
            "copy_check": reason or "accepted",
            "residual": residual,
            "explained": explained,
            "gate_applies": explained >= 0.5,
        }
    return report


def verdict(report):
    problems = []
    if report["relative_change_from_parent"] > 0.15:
        problems.append("over the 0.15 change cap: the hash would be rejected for good")
    if report["own_change"] < MIN_SIZE:
        problems.append(
            f"own change {report['own_change']:.1f} is under {MIN_SIZE}: a changed soup can refuse it"
        )
    for name, row in report["scenarios"].items():
        if row["copy_check"] != "accepted":
            problems.append(f"copy check refuses it if {name}: {row['copy_check']}")
        if row["residual"] < SCOUT_LINE:
            problems.append(f"residual {row['residual']:.3f} if {name}: paid at the 0.25 scout rate")
    for member, value in report["cosine_with_members_own_change"].items():
        if abs(value) >= 0.5:
            problems.append(f"own change overlaps {member} (cosine {value:.2f}): little new to add")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--house", nargs="+", required=True, help="every house body; the first is the release")
    parser.add_argument("--start-house", nargs="+", required=True, help="the house bodies the soup began from")
    parser.add_argument("--members", nargs="+", required=True, help="the soup's current members")
    parser.add_argument("--mine", required=True)
    parser.add_argument("--rivals", nargs="*", default=[])
    parser.add_argument("--json", help="also write the report here")
    args = parser.parse_args(argv)
    folders = list(dict.fromkeys([*args.house, *args.start_house, *args.members, *args.rivals, args.mine]))
    gram, parent_norm = gram_matrix(args.qwen, folders)
    report = analyse(
        gram, parent_norm, folders, house=args.house, start_house=args.start_house,
        members=args.members, mine=args.mine, rivals=args.rivals,
    )  # fmt: skip
    report["problems"] = verdict(report)
    print(json.dumps(report, indent=1))
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(report, stream, indent=1)
    print("\nOK to commit on this check." if not report["problems"] else "\nDo not commit yet:")
    for problem in report["problems"]:
        print(" -", problem)
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
