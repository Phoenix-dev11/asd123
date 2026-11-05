---
license: other
language: [en]
pretty_name: Zils base-exam public seed v4
---
# Zils base-exam public seed v4

Public development sample for the Zils decision-quality base election (`zils.base_exam seed` bundle,
`manifest.json`). 40 decision items across 2 families, with labels. Publishing these families makes them
"known" families for later private election draws; no private election items are in this dataset.

The two `-k5` families replace `language-known-score-k2` and `arts-known-noul-k3`, whose held-out style
sources had too few unused rows left. The other known families keep their seeds
([seed v2](https://huggingface.co/datasets/zils-ai/base-exam-seed-v2) for `-k2`,
[seed v3](https://huggingface.co/datasets/zils-ai/base-exam-seed-v3) for `-k3`).

| Family | Question | Style source (held-out split) |
| --- | --- | --- |
| `language-known-score-k5` | How similar are two short image captions? | Crisscrossed Captions STS test (Google) |
| `arts-known-noul-k5` | Is this news headline sarcastic? | News Headlines Dataset for Sarcasm Detection (Misra) |

Each item was newly written by Claude Opus 5.5, using one row from a public dataset only as a style
example (`source_id` names that row), then kept only if an independent blind answer matched. Neither
source is used by the Decision Index or S1MB. No benchmark test-set rows and no customer data were used.
Check each source's licence before reuse (Crisscrossed Captions: Google, free for any purpose; Sarcasm
Headlines: CC BY 4.0, Rishabh Misra); this card does not relicense them.

Format: `cases.jsonl` (one case per line: `family`, `category`, `block`, `source_id`, `state.text`,
`question` with `type`/`instructions`/`criteria`, `label`), `registry.json` (family type and category).
