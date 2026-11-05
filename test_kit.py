#!/usr/bin/env python3
"""Offline checks for the kit: label logic against the real public seeds, format, and split safety.

    SEEDS=/path/to/seeds python -m unittest test_kit -v
"""

import json
import os
import random
import re
import tempfile
import unittest
from pathlib import Path

import gen_cases as g

SEEDS = Path(os.environ.get("SEEDS", "data/seeds"))


def seed_rows(family):
    rows = []
    for path in SEEDS.rglob("cases.jsonl"):
        rows += [r for r in map(json.loads, path.open()) if r["family"] == family]
    return rows


class LabelsMatchRealSeeds(unittest.TestCase):
    def need(self, family):
        rows = seed_rows(family)
        if not rows:
            self.skipTest(f"no seeds for {family} under {SEEDS}")
        return rows

    def test_count(self):
        for r in self.need("reasoning-known-score-k6"):
            self.assertEqual(str(g.count_answer(json.loads(r["state"]["text"]))), r["label"])

    def test_coverage(self):
        for r in self.need("retrieval-known-score-k6"):
            pairs = json.loads(r["state"]["text"])["pairs"]
            self.assertEqual(str(sum(g.covered(p) for p in pairs)), r["label"])

    def test_tools_score(self):
        for r in self.need("tools-known-score-k6"):
            self.assertEqual(str(g.available_count(json.loads(r["state"]["text"]))), r["label"])

    def test_tools_noul(self):
        for r in self.need("tools-known-noul-k6"):
            label = "true" if g.can_call(json.loads(r["state"]["text"])) else "false"
            self.assertEqual(label, r["label"])

    def test_tools_choice(self):
        for r in self.need("tools-known-choice-k6"):
            state = json.loads(r["state"]["text"])
            tool = g.matching_tool(state)
            good = []
            for key, text in r["question"]["criteria"].items():
                call = json.loads(text)
                expected = {k: v[0] for k, v in state["observations"].items()}
                if call["name"] == tool["name"] and call["arguments"] == expected:
                    good.append(key)
            self.assertEqual(good, [r["label"]])

    def test_constants_match_seed_text(self):
        state = json.loads(self.need("tools-known-noul-k6")[0]["state"]["text"])
        self.assertEqual(state["policy"], g.POLICY)
        self.assertEqual(state["tools"], g.TOOLS)
        criteria = self.need("retrieval-known-choice-k6")[0]["question"]["criteria"]
        self.assertEqual(list(criteria.values()), g.NEWSGROUPS)

    def test_question_format_matches_seed(self):
        rng = random.Random(1)
        pairs = {
            "reasoning-known-score-k6": g.gen_count,
            "retrieval-known-score-k6": g.gen_coverage,
            "tools-known-score-k6": g.gen_tools_score,
            "tools-known-noul-k6": g.gen_tools_noul,
            "tools-known-choice-k6": g.gen_tools_choice,
        }
        for family, generate in pairs.items():
            seed = self.need(family)[0]
            made = generate(rng)
            self.assertEqual(made["family"], seed["family"])
            self.assertEqual(made["category"], seed["category"])
            self.assertEqual(made["question"]["type"], seed["question"]["type"])
            self.assertEqual(made["question"]["instructions"], seed["question"]["instructions"])
            if family != "tools-known-choice-k6":
                self.assertEqual(
                    made["question"].get("criteria"), seed["question"].get("criteria")
                )
            self.assertEqual(sorted(json.loads(made["state"]["text"])),
                             sorted(json.loads(seed["state"]["text"])))  # fmt: skip


class GeneratorsAreConsistent(unittest.TestCase):
    def test_labels_recompute(self):
        rng = random.Random(7)
        for _ in range(3000):
            c = g.gen_count(rng)
            state = json.loads(c["state"]["text"])
            self.assertEqual(str(g.count_answer(state)), c["label"])
            self.assertGreater(state["items_removed"], 0)
            c = g.gen_coverage(rng)
            pairs = json.loads(c["state"]["text"])["pairs"]
            self.assertEqual(str(sum(g.covered(p) for p in pairs)), c["label"])
            c = g.gen_tools_score(rng)
            self.assertEqual(str(g.available_count(json.loads(c["state"]["text"]))), c["label"])
            c = g.gen_tools_noul(rng)
            expected = "true" if g.can_call(json.loads(c["state"]["text"])) else "false"
            self.assertEqual(expected, c["label"])

    def test_every_label_appears(self):
        rng = random.Random(3)
        for generate, wanted in (
            (g.gen_count, {str(i) for i in range(10)}),
            (g.gen_coverage, {"0", "1", "2"}),
            (g.gen_tools_score, {str(i) for i in range(5)}),
            (g.gen_tools_noul, {"true", "false"}),
            (g.gen_tools_choice, set("ABCD")),
        ):
            self.assertEqual({generate(rng)["label"] for _ in range(500)}, wanted)

    def test_noul_is_roughly_balanced(self):
        rng = random.Random(5)
        rows = [g.gen_tools_noul(rng)["label"] for _ in range(4000)]
        self.assertTrue(0.4 < rows.count("true") / len(rows) < 0.6)

    def test_choice_has_one_correct_call(self):
        rng = random.Random(9)
        for _ in range(500):
            c = g.gen_tools_choice(rng)
            self.assertEqual(len(set(c["question"]["criteria"].values())), 4)
            state = json.loads(c["state"]["text"])
            tool = g.matching_tool(state)
            call = json.loads(c["question"]["criteria"][c["label"]])
            self.assertEqual(call["name"], tool["name"])
            self.assertEqual(set(call["arguments"]), set(tool["required"]))
            self.assertNotIn("different-value", call["arguments"].values())

    def test_word_forms(self):
        forms = {
            "V;NFIN": "walk", "V;PST": "walked", "V;V.PTCP;PST": "walked",
            "V;V.PTCP;PRS": "walking", "V;PRS;3;SG": "walks",
        }  # fmt: skip
        rng = random.Random(1)
        made = [g.word_form_case(rng, "walk", forms) for _ in range(50)]
        for c in made:
            self.assertEqual(len(c["question"]["criteria"]), 4)
            self.assertRegex(c["question"]["instructions"], r"^Which word is the .* form of the lemma")
        # "put" has the same past and base form, so asking for its past cannot be unambiguous.
        same = {"V;NFIN": "put", "V;PST": "put", "V;V.PTCP;PST": "put",
                "V;V.PTCP;PRS": "putting", "V;PRS;3;SG": "puts"}  # fmt: skip
        self.assertTrue(all(g.word_form_case(rng, "put", same) is None for _ in range(20)))

    def test_syllables(self):
        self.assertEqual(g.syllables("S P AY S L AE N D"), 2)
        self.assertEqual(g.syllables("HH AE1 S W EH0 L"), 2)

    def test_ordinal_soft(self):
        soft = g.ordinal_soft(1.24, 6)
        self.assertAlmostEqual(sum(soft.values()), 1)
        self.assertEqual(max(soft, key=soft.get), "1")


class DevSetIsScorable(unittest.TestCase):
    def test_seed_rows_keep_meaning_when_rekeyed(self):
        case = {
            "question": {
                "type": "choice", "instructions": "x",
                "criteria": {"wall_cabinet": "a cabinet", "gym": "a gym"},
            },
            "label": "gym",
        }  # fmt: skip
        made = g.letter_keys(case)
        self.assertEqual(made["question"]["criteria"], {"A": "a cabinet", "B": "a gym"})
        self.assertEqual(made["question"]["criteria"][made["label"]], "a gym")

    def test_every_seed_family_survives_the_scorer(self):
        if not list(SEEDS.rglob("cases.jsonl")):
            self.skipTest("no seeds")
        import scoreboard

        rows = g.seed_dev_cases(SEEDS)
        kept = scoreboard.usable(rows)
        self.assertEqual({c["family"] for c in kept}, {c["family"] for c in rows})


class AssemblyWithRelay(unittest.TestCase):
    """Synthetic relay rows in the documented format, so filtering and selection are exercised."""

    def write(self, path, rows):
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_filters_pool_and_final_file(self):
        import assemble

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            noul = {"type": "noul", "instructions": "yes?"}
            choice = {"type": "choice", "instructions": "which?", "criteria": {"A": "a", "B": "b"}}
            dev = {"id": "d", "family": "f", "category": "arts", "block": "known", "label": "A",
                   "state": {"text": "in dev"}, "question": choice}  # fmt: skip
            case = {"id": "c", "family": "f", "category": "arts", "block": "known", "label": "true",
                    "state": {"text": "generated"}, "question": noul}  # fmt: skip
            self.write(root / "dev.jsonl", [dev])
            self.write(root / "cases.jsonl", [case])
            rebuilt, records = [], []
            for i in range(100):
                rebuilt.append({"i": i, "state": {"text": f"row {i}"}, "question": choice,
                                "probabilities": {"A": 0.7, "B": 0.3}})  # fmt: skip
                records.append({"i": i, "source": "ok", "source_index": f"x:train:{i}"})
            rebuilt.append({"i": 100, "state": {"text": "val"}, "question": choice,
                            "probabilities": {"A": 0.5, "B": 0.5}})  # fmt: skip
            records.append({"i": 100, "source": "MMLU", "source_index": "MMLU:validation:12"})
            rebuilt.append({"i": 101, "state": {"text": "held"}, "question": choice,
                            "probabilities": {"A": 0.5, "B": 0.5}})  # fmt: skip
            records.append({"i": 101, "source": "VitaminC", "source_index": "x:train:1"})
            rebuilt.append({"i": 102, "state": {"text": "in dev"}, "question": choice,
                            "probabilities": {"A": 0.5, "B": 0.5}})  # fmt: skip
            records.append({"i": 102, "source": "ok", "source_index": "x:train:2"})
            for i in range(103, 113):  # rare question types are kept first
                rebuilt.append({"i": i, "state": {"text": f"yn {i}"}, "question": noul,
                                "probabilities": {"false": 0.2, "true": 0.8}})  # fmt: skip
                records.append({"i": i, "source": "ok", "source_index": f"x:train:{i}"})
            self.write(root / "rebuilt.jsonl", rebuilt)
            self.write(root / "records.jsonl", records)
            common = ["--cases", str(root / "cases.jsonl"), "--dev", str(root / "dev.jsonl"),
                      "--relay-rebuilt", str(root / "rebuilt.jsonl"),
                      "--relay-records", str(root / "records.jsonl"), "--relay-count", "30"]  # fmt: skip
            assemble.main([*common, "--write-pin-pool", str(root / "pool.jsonl"), "--pin-count", "20"])
            pool = [json.loads(x) for x in (root / "pool.jsonl").open()]
            self.assertEqual(len(pool), 20)
            self.assertTrue(all("probabilities" not in r for r in pool))
            pin = [{**r, "probabilities": {"A": 0.6, "B": 0.4}} for r in pool]
            pin.append({"state": {"text": "too long"}, "question": choice, "error": "over limit"})
            self.write(root / "pin.jsonl", pin)
            assemble.main([*common, "--pin-answers", str(root / "pin.jsonl"),
                           "--out", str(root / "train.jsonl")])  # fmt: skip
            rows = [json.loads(x) for x in (root / "train.jsonl").open()]
            texts = {r["state"]["text"] for r in rows}
            self.assertEqual(len(rows), 1 + 30 + 20)
            self.assertFalse(texts & {"val", "held", "in dev", "too long"})
            self.assertTrue({f"yn {i}" for i in range(103, 113)} <= texts)
            self.assertIn("generated", texts)
            manifest = json.loads((root / "train.manifest.json").read_text())
            self.assertEqual(manifest["parts"]["relay"], 30)
            self.assertEqual(manifest["parts"]["pin"], 20)
            self.assertEqual(manifest["relay_dropped"]["non-train split"], 1)
            self.assertEqual(manifest["relay_dropped"]["held out by the exam"], 1)


class SplitsAreSafe(unittest.TestCase):
    def test_no_overlap_and_seeds_never_train(self):
        with tempfile.TemporaryDirectory() as folder:
            extra = Path(folder) / "seedbank"
            extra.mkdir()
            rng = random.Random(20261105)
            # Plant one case the generator will reproduce: it must not reach training.
            planted = g.gen_count(rng)
            (extra / "cases.jsonl").write_text(json.dumps(planted) + "\n")
            g.main(["--out", folder, "--no-datasets", "--computed-per-family", "400",
                    "--seeds", str(extra), "--dev-cap-per-family", "1000"])  # fmt: skip
            read = lambda n: [json.loads(x) for x in (Path(folder) / n).open()]  # noqa: E731
            train, dev = read("train.jsonl"), read("dev.jsonl")
            identity = lambda c: json.dumps([c["state"], c["question"]], sort_keys=True)  # noqa: E731
            self.assertFalse({identity(c) for c in train} & {identity(c) for c in dev})
            self.assertNotIn(identity(planted), {identity(c) for c in train})
            self.assertEqual(len({c["id"] for c in train}), len(train))

    def test_no_dataset_row_reads_a_held_out_split(self):
        source = Path(g.__file__).read_text()
        for banned in ("piqa_validation", "piqa_test", "test.jsonl", "sts_test",
                       "default/test", "split/test", "split/validation", "rte/validation",
                       "qnli/validation", "data/test-", "data/validation-"):  # fmt: skip
            self.assertNotIn(banned, re.sub(r'"""[\s\S]*?"""', "", source))


if __name__ == "__main__":
    unittest.main()
