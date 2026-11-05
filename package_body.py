#!/usr/bin/env python3
"""Turn a trained candidate folder into a soup body folder and run the project's own body checks.

    package_body.py --candidate data/v1 --out data/body-v1 --qwen QWEN_FOLDER \
        --started-from "the uniform average of thermopylae-777/zils-lite-body-2 and dasLOL/zils-lite-skills-v5" \
        --training data/train.manifest.json

A candidate also holds the answer layer, runtime code and metadata. A body must not: the chassis
supplies those, and any tensor outside the chassis layout rejects the body, so only the weight
shards and their index are copied.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from kitpath import POOL_MODELS, add_zils

add_zils()

CARD = """---
license: {license}
base_model: {parent}
tags:
- zils
- decision
- soup-body
- bittensor
---

# {title}

A body for the Zils `{pool}` soup pool (Bittensor subnet 127), evaluated on the pool's pinned chassis,
which supplies the answer layer, prompt and runtime.

Started from {started_from}. Fine-tuned with a merged LoRA using `zils.teacher train`.

## Training data

{training}

Dataset rows come from train splits, with one exception noted by source: Crisscrossed Captions
validation. No validator exam data and no customer data were used. Public seed items were excluded
from training. Check each source's licence for reuse; this repository contains weights only.
"""


def training_summary(path):
    if not path:
        return "Not recorded."
    manifest = json.loads(Path(path).read_text())
    lines = [f"{manifest['rows']:,} rows in total:"]
    lines += [f"- {name}: {count:,}" for name, count in manifest["parts"].items()]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", choices=sorted(POOL_MODELS), default="d3-lite")
    parser.add_argument("--qwen", help="the pool's Qwen parent folder, for the 0.15 change check")
    parser.add_argument("--started-from", required=True)
    parser.add_argument("--training", help="the manifest assemble.py wrote")
    parser.add_argument("--title", default="Zils soup body")
    parser.add_argument("--license", default="apache-2.0")
    args = parser.parse_args(argv)
    from zils import bodies, d3

    model = POOL_MODELS[args.pool]
    chassis = d3.pinned_base_path(download=True, model=model)
    candidate, out = Path(args.candidate), Path(args.out)
    layout = d3.weight_map(candidate)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.mkdir(parents=True)
    for shard in sorted(set(layout.values())):
        shutil.copyfile(candidate / shard, out / shard)
    index = candidate / "model.safetensors.index.json"
    if index.exists():
        shutil.copyfile(index, out / index.name)
    parent = bodies.parent(args.pool)[0]
    (out / "README.md").write_text(
        CARD.format(
            license=args.license, parent=parent, title=args.title, pool=args.pool,
            started_from=args.started_from, training=training_summary(args.training),
        )
    )  # fmt: skip
    digest = bodies.validate_body(
        out,
        chassis,
        hotkey=None,  # the owner file comes later, from `zils-miner commit-body --write-owner`
        base=args.qwen,
        trusted=True,
        max_bytes=bodies.max_bytes(args.pool),
    )
    print(f"body {digest} passes the local checks for {args.pool}: {out}")
    print("Files:", ", ".join(sorted(p.name for p in out.iterdir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
