#!/usr/bin/env python3
"""Read https://zils.ai/miners: each tier's soup, its members and every miner's status.

    soup_watch.py                      print the current state
    soup_watch.py --log data/log.jsonl also append a timestamped snapshot
    soup_watch.py --hotkey 5GBdhv      also show one miner's rows (matches the displayed prefix)

The page is a website, not an API: if its layout changes this stops with an error rather than
reporting something wrong. Use the output with build_soup.py --body to rebuild on the latest soup.
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from collections import Counter

URL = "https://zils.ai/miners"
TIERS = ("z3-lite", "z3-mini", "z3-flash", "z3")
POOLS = {"z3-lite": "d3-lite", "z3-mini": "d3-mini", "z3-flash": "d3-flash", "z3": "d3"}


def fetch(url=URL):
    request = urllib.request.Request(url, headers={"User-Agent": "lite-kit"})
    return urllib.request.urlopen(request, timeout=30).read().decode("utf8", "replace")


def lines_of(page):
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", body))
    return [line.strip() for line in text.split("\n") if line.strip()]


def parse(page):
    lines = lines_of(page)
    try:
        start = max(i for i, line in enumerate(lines) if line == "Tonight’s soup")
        table = lines.index("Serving", start)
    except ValueError as error:
        raise SystemExit(f"{URL}: page layout changed ({error})") from error

    tiers = {}
    positions = []
    for index in range(start, table):
        if lines[index] in TIERS and re.fullmatch(r"[\d.]+B", lines[index + 1] or ""):
            positions.append((index, lines[index]))
    for (index, tier), (end, _) in zip(positions, [*positions[1:], (table, None)], strict=True):
        block = lines[index:end]
        record = {"pool": POOLS[tier], "members": []}
        record["open"] = not any(line == "Opens later" for line in block)
        for at, line in enumerate(block):
            if re.fullmatch(r"\d+\.\d", line) and "score" not in record:
                record["score"] = float(line)
            if line == "Round" and at + 1 < len(block):
                record["round"] = block[at + 1]
            if line == "by" and at >= 1 and at + 3 < len(block):
                value = block[at + 3]
                if re.fullmatch(r"[+\-−]\d+\.\d+", value):
                    record["members"].append(
                        {
                            "name": block[at - 1],
                            "by": block[at + 1],
                            "status": block[at + 2],
                            "contribution": float(value.replace("−", "-")),
                        }
                    )
            if line in ("body", "bodies") and at >= 1 and block[at - 1].isdigit():
                record["waiting"] = int(block[at - 1])
            if line.startswith("Seats decided on") or line.startswith(". Seats decided on"):
                record["decided_on"] = line.lstrip(". ")
        tiers[POOLS[tier]] = record

    miners, at = [], table + 1
    while at + 6 < len(lines) and lines[at].isdigit():
        uid, hotkey, tier, repo, revision, status, serving = lines[at : at + 7]
        miners.append(
            {
                "uid": int(uid),
                "hotkey": hotkey,
                "pool": POOLS.get(tier, tier),
                "repo": repo,
                "revision": revision,
                "status": status,
                "serving": serving,
            }
        )
        at += 7
    block = None
    if lines[at : at + 1] == ["Block"]:
        block = int(lines[at + 1].replace(",", ""))
    if not tiers or not miners:
        raise SystemExit(f"{URL}: page layout changed (no tiers or no miners found)")
    return {"block": block, "tiers": tiers, "miners": miners}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log")
    parser.add_argument("--hotkey", help="prefix of a displayed hotkey")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    state = parse(fetch())
    state["fetched_at"] = int(time.time())
    if args.log:
        with open(args.log, "a") as stream:
            stream.write(json.dumps(state) + "\n")
    if args.json:
        print(json.dumps(state, indent=1))
        return 0
    print(f"block {state['block']}")
    for pool, tier in state["tiers"].items():
        print(f"\n{pool}: score {tier.get('score')}  round {tier.get('round')}  "
              f"waiting {tier.get('waiting')}  open {tier['open']}")  # fmt: skip
        for member in tier["members"]:
            print(f"  {member['contribution']:+.1f}  {member['name']}  ({member['status']})")
        by_status = Counter(m["status"] for m in state["miners"] if m["pool"] == pool)
        for status, count in by_status.most_common():
            print(f"      {count:3d} {status}")
    if args.hotkey:
        print()
        for miner in state["miners"]:
            if miner["hotkey"].startswith(args.hotkey):
                print(miner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
