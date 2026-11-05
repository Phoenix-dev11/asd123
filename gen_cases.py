#!/usr/bin/env python3
"""Generate d3-lite training and dev cases in the public seed format.

Writes <out>/train.jsonl and <out>/dev.jsonl plus <out>/manifest.json. A generated instance or a
source row goes to exactly one split, routed by a hash of its key, so dev never overlaps training.

Rules this script enforces:
- Dataset rows come only from splits the exam-authoring code does not draw from: CMUdict, PIQA, 20
  Newsgroups, GLUE, Social IQa, emotion and Sarcasm train splits, CxC validation, and the Stack
  Exchange politeness corpus (the exam uses the Wikipedia one).
- Seed items themselves are never training rows; they only enter dev (`--seeds`).
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

POLICY = (
    "All listed arguments are required. No defaults, optional or extra arguments are allowed. "
    "A field is available only if exactly one distinct non-null value of its required type is "
    "explicitly observed. Conflicting valid values require clarification. No call without exactly "
    "one matching tool and all required fields."
)
TOOLS = [
    {
        "name": "send_parcel",
        "intent": "ship a parcel",
        "required": {
            "recipient": "string",
            "address": "string",
            "parcel_code": "string",
            "mass_kg": "integer",
        },
    },
    {
        "name": "book_journey",
        "intent": "book a journey",
        "required": {
            "origin": "string",
            "destination": "string",
            "day": "string",
            "travelers": "integer",
        },
    },
    {
        "name": "convert_currency",
        "intent": "convert money",
        "required": {
            "from_currency": "string",
            "to_currency": "string",
            "as_of": "string",
            "amount": "integer",
        },
    },
]
UNKNOWN_INTENTS = ["translate a document", "reset a password", "order a meal", "file a claim"]
ATTRIBUTES = {
    "city": ["Accra", "Lyon", "Osaka", "Quito", "Tunis", "Perth", "Riga", "Hanoi", "Bergen", "Cusco"],
    "vehicle": ["bus", "tram", "ferry", "van", "scooter", "bicycle", "truck", "taxi"],
    "parcel": ["green crate", "red box", "blue tube", "small sack", "flat carton", "tin case"],
}
NEWSGROUPS = (
    "alt.atheism comp.graphics comp.os.ms-windows.misc comp.sys.ibm.pc.hardware comp.sys.mac.hardware "
    "comp.windows.x misc.forsale rec.autos rec.motorcycles rec.sport.baseball rec.sport.hockey sci.crypt "
    "sci.electronics sci.med sci.space soc.religion.christian talk.politics.guns talk.politics.mideast "
    "talk.politics.misc talk.religion.misc"
).split()
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
FORM_FEATURES = {
    "simple past": "V;PST",
    "present participle": "V;V.PTCP;PRS",
    "third-person singular present": "V;PRS;3;SG",
    "past participle": "V;V.PTCP;PST",
    "base form": "V;NFIN",
}
UNIMORPH = (
    "https://raw.githubusercontent.com/unimorph/eng/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b/eng"
)
EMOTIONS = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
SARCASM_REV = "cd9dba22bb10fa4a90721e0bb5004ecbf16a2e75"


# ---- Label logic, checked against the real seed items in test_kit.py ----------------------------


def count_answer(state):
    """Items remaining in the exact-count family; empty crates are a distractor."""
    return (
        state["starting_items"]
        + state["shipments"] * state["items_per_shipment"]
        - state["items_removed"]
    )


def covered(pair):
    """A pair counts when the sentence gives the asked attribute for the asked person."""
    asked = re.fullmatch(r"What (\w+) value is recorded for ([\w-]+)\?", pair["question"])
    entry = re.fullmatch(r"The only entry states: ([\w-]+); (\w+) = .+\.", pair["sentence"])
    return asked[1] == entry[2] and asked[2] == entry[1]


def _is_type(value, kind):
    if value is None or isinstance(value, bool):
        return False
    return isinstance(value, str) if kind == "string" else isinstance(value, int)


def available(observations, name, kind):
    """Exactly one distinct non-null value of the required type was observed."""
    values = {v for v in observations.get(name, []) if _is_type(v, kind)}
    return len(values) == 1


def matching_tool(state):
    tools = [t for t in state["tools"] if t["intent"] == state["requested_intent"]]
    return tools[0] if len(tools) == 1 else None


def available_count(state):
    tool = matching_tool(state)
    return sum(available(state["observations"], n, k) for n, k in tool["required"].items())


def can_call(state):
    tool = matching_tool(state)
    return tool is not None and available_count(state) == len(tool["required"])


def syllables(phonemes):
    return sum(re.sub(r"\d", "", p) in VOWELS for p in phonemes.split())


# ---- Case building -------------------------------------------------------------------------------


def make(family, category, state, question, label, source, soft=None):
    text = state if isinstance(state, str) else json.dumps(state)
    body = json.dumps([family, text, question, label], sort_keys=True)
    case = {
        "id": "kit-" + hashlib.sha256(body.encode()).hexdigest()[:24],
        "family": family,
        "category": category,
        "block": "known",
        "source_id": source,
        "state": {"text": text},
        "question": question,
        "label": label,
    }
    if soft is not None:
        case["soft"] = soft
    return case


def gen_count(rng):
    label = rng.randrange(10)
    shipments, per = rng.randint(2, 9), rng.randint(2, 9)
    start = rng.randint(20, 110)
    state = {
        "starting_items": start,
        "shipments": shipments,
        "items_per_shipment": per,
        "items_removed": start + shipments * per - label,
        "unrelated_empty_crates": rng.randint(1, 20),
    }
    question = {
        "type": "score",
        "instructions": "A store starts with the stated items, receives all shipments, then removes"
        " the stated number of items. Empty crates contain no items. How many items remain?",
        "criteria": [str(i) for i in range(10)],
    }
    return make("reasoning-known-score-k6", "reasoning", state, question, str(label), "kit:count")


def gen_coverage(rng):
    label = rng.randrange(3)
    good = rng.sample([0, 1], label)
    pairs = []
    for i in range(2):
        visitor = f"Visitor-{rng.randint(10000, 99999)}"
        attribute, other = rng.sample(sorted(ATTRIBUTES), 2)
        if i in good:
            value = rng.choice(ATTRIBUTES[attribute])
            sentence = f"The only entry states: {visitor}; {attribute} = {value}."
        elif rng.random() < 0.15:  # the right attribute for a different person is also unavailable
            stranger = f"Visitor-{rng.randint(10000, 99999)}"
            value = rng.choice(ATTRIBUTES[attribute])
            sentence = f"The only entry states: {stranger}; {attribute} = {value}."
        else:
            value = rng.choice(ATTRIBUTES[other])
            sentence = f"The only entry states: {visitor}; {other} = {value}."
        pairs.append(
            {"question": f"What {attribute} value is recorded for {visitor}?", "sentence": sentence}
        )
    question = {
        "type": "score",
        "instructions": "For how many pairs does the sentence supply the requested attribute for its"
        " named person? Use only the supplied entries; an unmentioned attribute is unavailable.",
        "criteria": ["0 pairs", "1 pair", "2 pairs"],
    }
    return make(
        "retrieval-known-score-k6", "retrieval", {"pairs": pairs}, question, str(label), "kit:coverage"
    )


def _value(rng, name, kind):
    return rng.randint(1, 99) if kind == "integer" else f"{name}-{rng.randint(10000, 99999)}"


def _wrong(rng, name, kind):
    return f"{name}-{rng.randint(10000, 99999)}" if kind == "integer" else rng.randint(1, 99)


def _observe(rng, name, kind, usable):
    """An observation list that is, or is not, one unambiguous valid value."""
    value = _value(rng, name, kind)
    other = value
    while other == value:  # a repeated value is one distinct value, so it would not conflict
        other = _value(rng, name, kind)
    if usable:
        return rng.choices(
            [[value], [value, value], [value, None], [value, _wrong(rng, name, kind)]],
            weights=[6, 1, 1, 1],
        )[0]
    return rng.choice(
        [
            None,  # not observed at all
            None,
            [None],
            [_wrong(rng, name, kind)],
            [value, other],  # conflicting valid values
        ]
    )


def _tool_state(rng, tool, usable_names, intent=None):
    observations = {}
    for name, kind in tool["required"].items():
        values = _observe(rng, name, kind, name in usable_names)
        if values is not None:
            observations[name] = values
    return {
        "tools": TOOLS,
        "requested_intent": intent or tool["intent"],
        "observations": observations,
        "policy": POLICY,
    }


def gen_tools_score(rng):
    tool = rng.choice(TOOLS)
    label = rng.randrange(5)
    names = rng.sample(sorted(tool["required"]), label)
    state = _tool_state(rng, tool, names)
    question = {
        "type": "score",
        "instructions": "How many of the intended tool's four required arguments have one valid,"
        " unambiguous observed value?",
        "criteria": [str(i) for i in range(5)],
    }
    return make(
        "tools-known-score-k6", "tools", state, question, str(available_count(state)), "kit:tools-score"
    )


def gen_tools_noul(rng):
    tool = rng.choice(TOOLS)
    if rng.random() < 0.5:
        state = _tool_state(rng, tool, sorted(tool["required"]))
    elif rng.random() < 0.2:  # no tool matches the request, however complete the observations
        state = _tool_state(rng, tool, sorted(tool["required"]), rng.choice(UNKNOWN_INTENTS))
    else:
        keep = rng.randrange(len(tool["required"]))
        state = _tool_state(rng, tool, rng.sample(sorted(tool["required"]), keep))
    question = {
        "type": "noul",
        "instructions": "Can the assistant make a valid function call now without clarification?",
    }
    label = "true" if can_call(state) else "false"
    return make("tools-known-noul-k6", "tools", state, question, label, "kit:tools-noul")


def _call(name, arguments):
    return json.dumps({"arguments": arguments, "name": name}, sort_keys=True)


def gen_tools_choice(rng):
    tool = rng.choice(TOOLS)
    state = _tool_state(rng, tool, sorted(tool["required"]))
    arguments = {}
    for name, kind in tool["required"].items():
        arguments[name] = next(v for v in state["observations"][name] if _is_type(v, kind))
    strings = [n for n, k in tool["required"].items() if k == "string"]
    other = rng.choice([t for t in TOOLS if t is not tool])
    dropped = rng.choice(sorted(arguments))
    changed = rng.choice(strings)
    options = [
        (_call(tool["name"], arguments), True),
        (_call(tool["name"], {k: v for k, v in arguments.items() if k != dropped}), False),
        (_call(tool["name"], {**arguments, changed: "different-value"}), False),
        (_call(other["name"], arguments), False),
    ]
    rng.shuffle(options)
    question = {
        "type": "choice",
        "instructions": "Which exact function call satisfies the request and the stated policy?",
        "criteria": {LETTERS[i]: text for i, (text, _) in enumerate(options)},
    }
    label = next(LETTERS[i] for i, (_, good) in enumerate(options) if good)
    return make("tools-known-choice-k6", "tools", state, question, label, "kit:tools-choice")


def word_form_case(rng, lemma, forms):
    """One language-known-choice-s1 case, or None when the lemma cannot make an unambiguous one."""
    wanted = rng.choice(sorted(FORM_FEATURES))
    feature = FORM_FEATURES[wanted]
    answer = forms.get(feature)
    base = forms.get(FORM_FEATURES["base form"])
    if not answer or not base or (wanted != "base form" and answer == base):
        return None
    distinct = list(dict.fromkeys(forms[f] for f in FORM_FEATURES.values() if f in forms))
    if answer not in distinct or len(distinct) < 4:
        return None
    options = [answer, *rng.sample([f for f in distinct if f != answer], 3)]
    rng.shuffle(options)
    question = {
        "type": "choice",
        "instructions": f"Which word is the {wanted} form of the lemma in standard English?"
        if wanted != "base form"
        else "Which word is the base form of the lemma in standard English?",
        "criteria": {LETTERS[i]: form for i, form in enumerate(options)},
    }
    return make(
        "language-known-choice-s1",
        "language",
        {"lemma": lemma},
        question,
        LETTERS[options.index(answer)],
        "kit:unimorph",
    )


# ---- Source readers ------------------------------------------------------------------------------


def hf_file(repo, path, revision=None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset", revision=revision)


def parquet_rows(repo, path, revision=None):
    import pyarrow.parquet as pq

    return pq.read_table(hf_file(repo, path, revision)).to_pylist()


def balanced(rows, key, per_label, rng):
    """At most `per_label` rows per label, in random order."""
    rows = list(rows)
    rng.shuffle(rows)
    seen, out = Counter(), []
    for row in rows:
        if seen[key(row)] < per_label:
            seen[key(row)] += 1
            out.append(row)
    return out


def source_cases(rng, limit, politeness_zip=None):
    """(group key, case) for every dataset family. The key keeps a word or row in a single split."""
    per = limit // 2
    for row in balanced(
        parquet_rows("baber/piqa", "piqa_train.parquet"), lambda r: r["label"], per, rng
    ):
        state = {"goal": row["goal"], "method_a": row["sol1"], "method_b": row["sol2"]}
        question = {
            "type": "noul",
            "instructions": "Is method A the better way to achieve the stated goal?",
        }
        label = "true" if row["label"] == 0 else "false"
        yield (
            row["goal"],
            make("reasoning-known-noul-k6", "reasoning", state, question, label, "piqa:train"),
        )

    rows = [json.loads(x) for x in open(hf_file("SetFit/20_newsgroups", "train.jsonl"))]
    rows = [r for r in rows if 20 <= len(r["text"].strip()) and len(r["text"]) <= 2500]
    criteria = {LETTERS[i]: name for i, name in enumerate(NEWSGROUPS)}
    question = {
        "type": "choice",
        "instructions": "Which topic category does this post belong to?",
        "criteria": criteria,
    }
    for row in balanced(rows, lambda r: r["label_text"], max(1, limit // 20), rng):
        label = LETTERS[NEWSGROUPS.index(row["label_text"])]
        yield (
            row["text"],
            make(
                "retrieval-known-choice-k6", "retrieval", {"text": row["text"]}, question, label,
                "20ng:train",
            ),
        )  # fmt: skip

    # CxC validation. The exam draws from CxC test, so test is never read here.
    path = hf_file("DolphyWind/cxc-sts-sentences", "sts_val.csv")
    rows = list(csv.DictReader(open(path, newline="")))
    question = {
        "type": "score",
        "instructions": "Rate the semantic similarity of the two captions.",
        "criteria": [f"Similarity level {i} of5" for i in range(6)],
    }
    for row in balanced(rows, lambda r: round(float(r["agg_score"])), max(1, limit // 6), rng):
        score = float(row["agg_score"])
        state = {"caption1": row["caption1"], "caption2": row["caption2"]}
        yield (
            row["caption1"] + row["caption2"],
            make(
                "language-known-score-k6", "language", state, question, str(round(score)),
                "cxc:val", soft=ordinal_soft(score, 6),
            ),
        )  # fmt: skip

    rows = parquet_rows("jc4p/CMUdict", "data/train-00000-of-00001.parquet")
    rows = [r for r in rows if r["text"].isalpha() and 1 <= syllables(r["phonemes"]) <= 5]
    question = {
        "type": "score",
        "instructions": "How many syllables does this word have in its dictionary pronunciation?",
        "criteria": ["1 syllable", *(f"{i} syllables" for i in range(2, 6))],
    }
    for row in balanced(rows, lambda r: syllables(r["phonemes"]), max(1, limit // 5), rng):
        label = str(syllables(row["phonemes"]) - 1)
        yield (
            row["text"],
            make("arts-known-score-k6", "arts", {"word": row["text"]}, question, label, "cmudict:train"),
        )

    rows = parquet_rows("nyu-mll/glue", "rte/train-00000-of-00001.parquet")
    question = {
        "type": "noul",
        "instructions": "Does the passage entail the hypothesis? Answer true or false.",
    }
    for row in balanced(rows, lambda r: r["label"], per, rng):
        text = f"Passage: {row['sentence1']}\nHypothesis: {row['sentence2']}"
        label = "true" if row["label"] == 0 else "false"
        yield text, make("language-known-noul-k2", "language", text, question, label, "rte:train")

    rows = parquet_rows("nyu-mll/glue", "qnli/train-00000-of-00001.parquet")
    question = {
        "type": "noul",
        "instructions": "Does the sentence contain the answer to the question? Answer true or false.",
    }
    for row in balanced(rows, lambda r: r["label"], per, rng):
        text = f'Question: {row["question"]}\nSentence: "{row["sentence"]}"'
        label = "true" if row["label"] == 0 else "false"
        yield text, make("retrieval-known-noul-k2", "retrieval", text, question, label, "qnli:train")

    rows = parquet_rows("raquiba/Sarcasm_News_Headline", "default/train/0000.parquet", SARCASM_REV)
    question = {
        "type": "noul",
        "instructions": "Is this news headline sarcastic, meaning it uses mock-serious, ironic or"
        " absurd framing for satire rather than reporting a real event? Answer true or false.",
    }
    for row in balanced(rows, lambda r: r["is_sarcastic"], per, rng):
        text = f'Headline: "{row["headline"]}"'
        label = "true" if row["is_sarcastic"] == 1 else "false"
        yield text, make("arts-known-noul-k5", "arts", text, question, label, "sarcasm:train")

    rows = parquet_rows("lighteval/siqa", "data/train-00000-of-00001-da17dfc1120ed1ee.parquet")
    for row in rows[: limit * 2]:
        question = {
            "type": "choice",
            "instructions": row["question"],
            "criteria": {"A": row["answerA"], "B": row["answerB"], "C": row["answerC"]},
        }
        label = "ABC"[int(row["label"]) - 1]
        yield (
            row["context"] + row["question"],
            make("reasoning-known-choice-k3", "reasoning", row["context"], question, label, "siqa:train"),
        )

    rows = parquet_rows("dair-ai/emotion", "split/train-00000-of-00001.parquet")
    question = {
        "type": "choice",
        "instructions": "Which single emotion does this message primarily express?",
        "criteria": {
            "A": "sadness", "B": "fear", "C": "anger", "D": "surprise", "E": "love", "F": "joy",
        },
    }  # fmt: skip
    letters = {v: k for k, v in question["criteria"].items()}
    for row in balanced(rows, lambda r: r["label"], max(1, limit // 6), rng):
        text = f'Message: "{row["text"]}"'
        yield (
            text,
            make("arts-known-choice-k3", "arts", text, question, letters[EMOTIONS[row["label"]]],
                 "emotion:train"),
        )  # fmt: skip

    if politeness_zip:
        import zipfile

        archive = zipfile.ZipFile(politeness_zip)
        name = next(n for n in archive.namelist() if n.endswith("utterances.jsonl"))
        rows = [json.loads(x) for x in archive.read(name).splitlines()]
        rows = [r for r in rows if r.get("meta", {}).get("Binary") in (-1, 0, 1)]
        question = {
            "type": "choice",
            "instructions": "How polite is this request?",
            "criteria": {"A": "impolite", "B": "neutral", "C": "polite"},
        }
        for row in balanced(rows, lambda r: r["meta"]["Binary"], max(1, limit // 3), rng):
            label = "ABC"[row["meta"]["Binary"] + 1]
            yield (
                row["text"],
                make("arts-known-choice-k6", "arts", {"text": row["text"]}, question, label,
                     "se-politeness"),
            )  # fmt: skip


def ordinal_soft(score, levels, width=0.5):
    """A distribution over ordered levels centred on a human mean score (helps ranked scoring)."""
    import math

    weights = [math.exp(-(((i - score) / width) ** 2) / 2) for i in range(levels)]
    total = sum(weights)
    return {str(i): w / total for i, w in enumerate(weights)}


def word_form_cases(rng, limit):
    with urllib.request.urlopen(UNIMORPH, timeout=120) as stream:
        lines = stream.read().decode().splitlines()
    lemmas, ambiguous = {}, set()
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 3 or not parts[0].isalpha() or not parts[1].isalpha():
            continue
        lemma, form, feature = parts
        if feature not in FORM_FEATURES.values():
            continue
        forms = lemmas.setdefault(lemma, {FORM_FEATURES["base form"]: lemma})
        if feature in forms and forms[feature] != form:
            ambiguous.add(lemma)
        forms[feature] = form
    names = sorted(set(lemmas) - ambiguous)
    rng.shuffle(names)
    produced = 0
    for lemma in names:
        case = word_form_case(rng, lemma, lemmas[lemma])
        if case:
            yield lemma, case
            produced += 1
            if produced >= limit:
                return


# ---- Seeds in dev --------------------------------------------------------------------------------


def letter_keys(case):
    """Older seeds name each option differently (`wall_cabinet`), so a constant answer by key looks
    perfect and the scorer rejects the family. Same options, same order, letter keys, as newer ones."""
    question = case["question"]
    if question["type"] != "choice" or all(len(k) == 1 for k in question["criteria"]):
        return case
    keys = list(question["criteria"])
    letters = dict(zip(keys, LETTERS, strict=False))
    criteria = {letters[k]: question["criteria"][k] for k in keys}
    return {**case, "question": {**question, "criteria": criteria}, "label": letters[case["label"]]}


def seed_dev_cases(root):
    """Seed items as dev rows. The private exam's `new` block cannot be simulated with its own
    families, so older seed families stand in for it; only the families the exam now uses (k6, s1,
    k3 and k5 where they remain) count as `known`."""
    live = {
        "reasoning-known-choice-k3", "language-known-noul-k2", "retrieval-known-noul-k2",
        "arts-known-noul-k5", "language-known-choice-s1",
    }  # fmt: skip
    out = []
    for path in sorted(Path(root).rglob("cases.jsonl")):
        for row in map(json.loads, path.open()):
            row = letter_keys(row)
            row.pop("soft", None)
            family = row["family"]
            if family in live or family.endswith("-k6"):
                out.append(row)
            else:
                out.append({**row, "family": f"devnew-{family}", "block": "new"})
    seen, unique = set(), []
    for row in out:
        if row["id"] not in seen:
            seen.add(row["id"])
            unique.append(row)
    return unique


# ---- Main ----------------------------------------------------------------------------------------

COMPUTED = {
    "count": gen_count,
    "coverage": gen_coverage,
    "tools-score": gen_tools_score,
    "tools-noul": gen_tools_noul,
    "tools-choice": gen_tools_choice,
}


def is_dev(key, percent):
    digest = hashlib.sha256(("split:" + key).encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < percent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--computed-per-family", type=int, default=3400)
    parser.add_argument("--dataset-per-family", type=int, default=1500)
    parser.add_argument("--dev-percent", type=int, default=8)
    parser.add_argument("--dev-cap-per-family", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--seeds", help="folder holding the public base-exam-seed cases.jsonl files")
    parser.add_argument("--politeness-zip", help="ConvoKit stack-exchange-politeness-corpus zip")
    parser.add_argument("--no-datasets", action="store_true", help="computed families only (offline)")
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = seed_dev_cases(args.seeds) if args.seeds else []
    # A seed's question text never becomes a training row, even when a generator reproduces it.
    seed_keys = {json.dumps([c["state"], c["question"]], sort_keys=True) for c in seeds}

    train, dev, seen, dev_count, skipped = [], [], set(), Counter(), Counter()

    def route(key, case):
        identity = json.dumps([case["state"], case["question"]], sort_keys=True)
        if identity in seen:
            return
        seen.add(identity)
        if identity in seed_keys:
            skipped["equals a public seed"] += 1
            return
        if is_dev(key, args.dev_percent):
            if dev_count[case["family"]] < args.dev_cap_per_family:
                dev_count[case["family"]] += 1
                dev.append(case)
            return
        train.append(case)

    for name, generate in COMPUTED.items():
        for _ in range(args.computed_per_family):
            case = generate(rng)
            route(f"{name}:{case['state']['text']}:{case['label']}", case)
    if not args.no_datasets:
        for key, case in word_form_cases(rng, args.computed_per_family):
            route(f"lemma:{key}", case)
        for key, case in source_cases(rng, args.dataset_per_family, args.politeness_zip):
            route(f"{case['family']}:{key}", case)
    dev += seeds
    with (out / "train.jsonl").open("w") as stream:
        for case in train:
            stream.write(json.dumps(case) + "\n")
    with (out / "dev.jsonl").open("w") as stream:
        for case in dev:
            stream.write(json.dumps(case) + "\n")
    manifest = {
        "seed": args.seed,
        "train_by_family": dict(sorted(Counter(c["family"] for c in train).items())),
        "dev_by_family": dict(sorted(Counter(c["family"] for c in dev).items())),
        "skipped": dict(skipped),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
