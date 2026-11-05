#!/usr/bin/env python3
"""Build a uniform soup with the project's own builder, the way a validator builds one.

    build_soup.py --pool d3-lite --out data/soup-s2 \
        --body thermopylae-777/zils-lite-body-2@618946d00aeafc81f701c3789f91c83d9fcb6fe0 \
        --body dasLOL/zils-lite-skills-v5@2c9920558bcd5034cfff05ad1642acbdb7939208

A --body is a local folder or owner/repo@revision (safetensors only are downloaded, so no repo code
reaches this machine). One body makes a single-body candidate, which is how a body gets graded.
"""

import argparse
import re
import sys
from pathlib import Path

from kitpath import POOL_MODELS, add_zils

add_zils()


def resolve(body, store):
    if Path(body).expanduser().is_dir():
        return Path(body).expanduser().resolve()
    match = re.fullmatch(r"([\w.-]+/[\w.-]+)@([a-f0-9]{7,40})", body)
    if not match:
        raise SystemExit(f"--body {body}: not a folder or owner/repo@revision")
    from huggingface_hub import snapshot_download

    repo, revision = match.groups()
    return Path(
        snapshot_download(
            repo,
            revision=revision,
            local_dir=Path(store) / repo.replace("/", "__") / revision[:12],
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", choices=sorted(POOL_MODELS), default="d3-lite")
    parser.add_argument("--out", required=True)
    parser.add_argument("--body", action="append", required=True)
    parser.add_argument("--store", default="data/bodies")
    args = parser.parse_args(argv)
    from zils import d3, soup

    model = POOL_MODELS[args.pool]
    chassis = d3.pinned_base_path(download=True, model=model)
    bodies = [(resolve(b, args.store), "strip-model") for b in args.body]
    out = soup.build(args.out, chassis, bodies, model=model)
    print(f"built {out} from {len(bodies)} bodies on {chassis}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
