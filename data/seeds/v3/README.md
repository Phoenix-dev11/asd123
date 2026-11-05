---
license: other
language: [en]
pretty_name: Zils base-exam public seed v3
---
# Zils base-exam public seed v3

Public development sample for the Zils decision-quality base election (`zils.base_exam seed` bundle,
`manifest.json`). 120 decision items across 6 families (reasoning and arts x choice / yes-no / score),
with labels. Publishing these families makes them "known" families for later private election draws; no
private election items are in this dataset.

The six `-k3` families replace the earlier `-k2` reasoning and arts families, whose held-out style sources
had too few unused rows left. The other nine known families keep their `-k2` seeds
([seed v2](https://huggingface.co/datasets/zils-ai/base-exam-seed-v2)).

| Family | Question | Style source (held-out split) |
| --- | --- | --- |
| `reasoning-known-choice-k3` | What a person most likely wants, feels or does next | Social IQa validation |
| `reasoning-known-noul-k3` | Does the stated method achieve the goal? | PIQA validation |
| `reasoning-known-score-k3` | Arithmetic word problems with distracting numbers | ASDiv validation |
| `arts-known-choice-k3` | Which emotion a short message expresses | dair-ai/emotion test |
| `arts-known-noul-k3` | Is this short post ironic? | TweetEval irony validation |
| `arts-known-score-k3` | How positive a short post is, by a stated rule | TweetEval sentiment test |

Each item was newly written by Claude Opus 5.5, using one row from the held-out split of a public dataset
only as a style example (`source_id` names that row), then kept only if an independent blind answer matched.
None of these sources is used by the Decision Index or S1MB. No benchmark test-set rows and no customer
data were used. Check each source's licence before reuse; this card does not relicense them.

Format: `cases.jsonl` (one case per line: `family`, `category`, `block`, `source_id`, `state.text`,
`question` with `type`/`instructions`/`criteria`, `label`), `registry.json` (family type and category).
