"""Rebuild the Zils Relay teacher answers: re-create each row's state and question from its public
source and join the teacher probabilities.

    pip install huggingface_hub pyarrow      # `pip install datasets` brings both
    python rebuild.py --records answers.jsonl --out rebuilt.jsonl

Writes one JSON line per row, ``{"i", "state", "question", "probabilities"}``, in record order:
the input format of ``python -m zils.teacher train --answers``. Every rebuilt row is checked against
the record's ``content_sha256``; rows that do not match, or whose source could not be read, are left
out and listed on stderr.

Sources:
- tasksource/tasksource-jev-typed-decisions at a pinned revision (most rows), MMLU, ARC, WinoGrande
  and HellaSwag: downloaded from the Hugging Face Hub.
- iSarcasmEval, BANKING77, CLINC150 and BFCL: single files downloaded from GitHub at pinned commits,
  sha256-checked.
"""

import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

TASKSOURCE = ("tasksource/tasksource-jev-typed-decisions", "8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8")
MAX_CHARS = 6000
GH = "https://raw.githubusercontent.com/"
FILES = {  # name -> (url, sha256)
    "isarcasm": (GH + "iabufarha/iSarcasmEval/dfc708b53bde1bb571abfb5692f63231c2232195/train/train.En.csv",
                 "fde737b69caffbcf81ebe79858ad16ac21881573c308343cd5219c4f7a800ae7"),
    "banking77-train": (GH + "PolyAI-LDN/task-specific-datasets/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data/train.csv",
                        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "banking77-categories": (GH + "PolyAI-LDN/task-specific-datasets/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data/categories.json",
                             "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63"),
    "clinc": (GH + "clinc/oos-eval/48a0e1cff8f43dd4d0836ecb4ed5df08733e3d2e/data/data_full.json",
              "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"),
}
BFCL = GH + "gorilla-llm/gorilla/916260dfc116bf06793a1af79b4ec8195b0453b6/berkeley-function-call-leaderboard/data/"
for _cat, _data, _gold in (("multiple", "8fb6b043cc80a1a0d5b5a578d5dd0593ed454d922d47034dcfdadfc4e36f7e5d",
                            "a17c86584546eb92899217f55751cecd95dacd2077f30366e19d3a6fa93d5fdd"),
                           ("parallel_multiple", "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
                            "39ed9845f6b748ddc533d6b56353071862a58321e36ae647a22a5fc0bc7c6df7")):
    FILES[f"bfcl-{_cat}"] = (BFCL + f"BFCL_v3_{_cat}.json", _data)
    FILES[f"bfcl-gold-{_cat}"] = (BFCL + f"possible_answer/BFCL_v3_{_cat}.json", _gold)

# v1 families: (source_dataset, source_revision, license, loader).
V1 = {
    "MMLU": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "mit", "hf"),
    "ARC-Challenge": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "cc-by-sa-4.0", "hf"),
    "ARC-Easy": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "cc-by-sa-4.0", "hf"),
    "WinoGrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "apache-2.0", "hf"),
    "HellaSwag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "mit", "hf"),
    "iSarcasmEval-A-En": ("github:iabufarha/iSarcasmEval", "dfc708b53bde1bb571abfb5692f63231c2232195", "mit", "github"),
    "iSarcasmEval-C-En": ("github:iabufarha/iSarcasmEval", "dfc708b53bde1bb571abfb5692f63231c2232195", "mit", "github"),
    "BANKING77": ("github:PolyAI-LDN/task-specific-datasets", "9d081458ff52e53cf7e848f414e6e9344e4e6696", "cc-by-4.0", "github"),
    "CLINC150+OOS": ("github:clinc/oos-eval", "48a0e1cff8f43dd4d0836ecb4ed5df08733e3d2e", "cc-by-3.0", "github"),
    "BFCL-tool-selection": ("github:gorilla-llm/gorilla", "916260dfc116bf06793a1af79b4ec8195b0453b6", "apache-2.0", "github"),
}


def content_sha256(state, question):
    return hashlib.sha256(json.dumps([state, question], sort_keys=True).encode()).hexdigest()


def fetch(name, cache):
    url, digest = FILES[name]
    path = cache / digest
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url).read())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{url}: sha256 mismatch")
    return data


def hf_rows(repo, filename, revision):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(repo, filename, repo_type="dataset", revision=revision)).to_pylist()


def choice(instructions, options):
    keys = [chr(65 + i) if len(options) <= 26 else f"option_{i}" for i in range(len(options))]
    return {}, {"type": "choice", "instructions": instructions, "criteria": dict(zip(keys, options))}


# --- v1 loaders: each yields (source_index, state, question) exactly as dev5's build_train.py ----------


def mmlu():
    repo, rev = V1["MMLU"][:2]
    for split in ("validation", "dev"):
        for i, r in enumerate(hf_rows(repo, f"all/{split}-00000-of-00001.parquet", rev)):
            yield f"MMLU:{split}:{i}", *choice(r["question"], r["choices"])


def arc(family):
    repo, rev = V1[family][:2]
    for split in ("train", "validation"):
        for r in hf_rows(repo, f"{family}/{split}-00000-of-00001.parquet", rev):
            yield f"{family}:{split}:{r['id']}", *choice(r["question"], r["choices"]["text"])


def winogrande():
    repo, rev = V1["WinoGrande"][:2]
    for i, r in enumerate(hf_rows(repo, "winogrande_xl/train-00000-of-00001.parquet", rev)):
        yield f"WinoGrande:train:{i}", *choice("Which option correctly fills the blank?\n" + r["sentence"],
                                              [r["option1"], r["option2"]])


def hellaswag():
    repo, rev = V1["HellaSwag"][:2]
    clean = lambda s: " ".join(str(s).split())  # noqa: E731
    for i, r in enumerate(hf_rows(repo, "data/train-00000-of-00001.parquet", rev)):
        yield f"HellaSwag:train:{i}:{r['ind']}", *choice("Which continuation is most plausible?\n" + clean(r["ctx"]),
                                                         [clean(x) for x in r["endings"]])


def isarcasm(cache):
    text = fetch("isarcasm", cache).decode("utf-8-sig")
    rows = [r for r in csv.DictReader(io.StringIO(text, newline="")) if r["tweet"] and r["sarcastic"] in {"0", "1"}]
    for r in rows:
        yield f"iSarcasmEval-A-En:train:{r['']}", r["tweet"], {
            "type": "choice", "instructions": "Is this text intended to be sarcastic?", "criteria": {"no": "No", "yes": "Yes"}}
    for r in rows:
        if r["sarcastic"] != "1" or not r["rephrase"].strip():
            continue
        flip = int(hashlib.sha256(r[""].encode()).hexdigest(), 16) % 2
        texts = [r["rephrase"], r["tweet"]] if flip else [r["tweet"], r["rephrase"]]
        yield f"iSarcasmEval-C-En:train:{r['']}", "", {
            "type": "choice", "instructions": "Which of these two texts is the sarcastic one?",
            "criteria": {"text_0": texts[0], "text_1": texts[1]}}


def banking77(cache):
    labels = json.loads(fetch("banking77-categories", cache))
    text = fetch("banking77-train", cache).decode("utf-8")
    for i, r in enumerate(csv.DictReader(io.StringIO(text, newline=""))):
        yield f"BANKING77:train:{i}", *choice("Classify the banking intent of this user request:\n" + r["text"], labels)


def clinc(cache):
    obj = json.loads(fetch("clinc", cache))
    labels = sorted({pair[1] for pair in obj["train"]} | {"oos"})
    desc = ["out of scope: none of the listed intents" if x == "oos" else x.replace("_", " ") for x in labels]
    for split in ("val", "oos_val"):
        for i, _ in enumerate(obj[split]):
            yield f"CLINC150+OOS:{split}:{i}", *choice(
                "Classify the intent of this user request, or choose out of scope if none applies:\n" + obj[split][i][0], desc)


def bfcl(cache):
    instructions = ("Given the complete user conversation and the published function schemas in state, mark each "
                    "candidate tool yes if it should be invoked to answer the request, or no otherwise. "
                    "This evaluates tool-name selection only; do not produce arguments or call ordering.\n\n")
    for category in ("multiple", "parallel_multiple"):
        for item in map(json.loads, fetch(f"bfcl-{category}", cache).decode().splitlines()):
            state = {"conversation": item["question"], "functions": item["function"], "task": "BFCL tool-name selection"}
            for f in item.get("function") or []:
                if isinstance(f, dict) and f.get("name"):
                    yield f"BFCL-tool-selection:{category}:{item['id']}:{f['name']}", state, {
                        "type": "choice", "instructions": instructions + f"\n\nCandidate tool: {f['name']}",
                        "criteria": {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}}


def v1_rows(family, args, cache):
    return {"MMLU": mmlu, "ARC-Challenge": lambda: arc("ARC-Challenge"), "ARC-Easy": lambda: arc("ARC-Easy"),
            "WinoGrande": winogrande, "HellaSwag": hellaswag, "iSarcasmEval-A-En": lambda: isarcasm(cache),
            "iSarcasmEval-C-En": lambda: isarcasm(cache), "BANKING77": lambda: banking77(cache),
            "CLINC150+OOS": lambda: clinc(cache), "BFCL-tool-selection": lambda: bfcl(cache)}[family]()


# --- tasksource rows: build_pool_v2.convert, minus the label ------------------------------------------


def convert(r):
    options = r["options"] if isinstance(r["options"], list) else ast.literal_eval(r["options"] or "[]")
    target = r["target"] if isinstance(r["target"], list) else ast.literal_eval(r["target"] or "[]")
    state = {"text": r["state"]}
    if r["kind"] == "noul" and len(target) == 1:
        q = {"type": "noul", "instructions": r["question"]}
    elif r["kind"] == "choice" and 2 <= len(options) <= 26 and len(target) == len(options):
        q = {"type": "choice", "instructions": r["question"], "criteria": dict(zip([chr(65 + i) for i in range(len(options))], options))}
    elif r["kind"] == "score" and len(options) == 5 and len(target) == 5:
        q = {"type": "score", "instructions": r["question"], "criteria": list(options)}
    else:
        return None
    return state, q


def tasksource(records, args):
    import pyarrow.parquet as pq

    by_file = collections.defaultdict(list)
    for rec in records:
        by_file[rec["source_file"]].append(rec)
    out = {}
    for name, recs in sorted(by_file.items()):
        if args.tasksource_dir:
            path = Path(args.tasksource_dir) / name
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(TASKSOURCE[0], name, repo_type="dataset", revision=TASKSOURCE[1])
        table = pq.read_table(path)
        rows = table.take([rec["source_row"] for rec in recs]).to_pylist()
        for rec, r in zip(recs, rows):
            if r["id"] == rec["source_id"] and (made := convert(r)):
                out[rec["i"]] = made
        print(f"{name}: {len(recs)} rows", file=sys.stderr, flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--records", required=True, help="answers.jsonl from this dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=".rebuild-cache", help="where GitHub files are kept (default .rebuild-cache)")
    p.add_argument("--tasksource-dir", help="folder already holding data/train-*.parquet at the pinned revision")
    args = p.parse_args()
    cache = Path(args.cache)
    records = [json.loads(line) for line in open(args.records)]
    made = tasksource([r for r in records if r["source_dataset"] == TASKSOURCE[0]], args)

    wanted = {r["source_index"]: r["i"] for r in records if r["source_dataset"] != TASKSOURCE[0]}
    for family in sorted({r["source"] for r in records if r["source_dataset"] != TASKSOURCE[0]}):
        for sid, state, q in v1_rows(family, args, cache):
            if (i := wanted.get(sid)) is not None:
                made[i] = (state, q)
        print(f"{family}: done", file=sys.stderr, flush=True)

    bad, written = collections.Counter(), 0
    with open(args.out, "w") as f:
        for r in records:
            if r["i"] not in made:
                bad[f"{r['source']}: source row not available"] += 1
                continue
            state, q = made[r["i"]]
            if content_sha256(state, q) != r["content_sha256"]:
                bad[f"{r['source']}: rebuilt text differs from what the teacher saw"] += 1
                continue
            f.write(json.dumps({"i": r["i"], "state": state, "question": q, "probabilities": dict(zip(r["options"], r["probabilities"]))}) + "\n")
            written += 1
    print(json.dumps({"records": len(records), "rebuilt": written, "missing": dict(bad)}, indent=1), file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
