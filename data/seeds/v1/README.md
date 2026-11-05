---
license: other
language: [en]
pretty_name: Zils base-exam public seed v1
---
# Zils base-exam public seed v1

Public development sample for the Zils decision-quality base election (`zils.base_exam seed` bundle,
`manifest.json`). 295 decision items across 15 families (5 areas x choice / yes-no / score), with labels.
Publishing these families makes them "known" families for later private election draws; no private
election items are in this dataset.

Each item was newly written by Claude Opus 5.5, using one row from the training split of a public dataset
only as a style example (`source_id` names that row), then kept only if an independent blind answer matched.
No benchmark test-set rows and no customer data were used.

Style sources (training splits): MMLU auxiliary_train, BoolQ, GSM8K, ANLI, GLUE RTE and STS-B, BANKING77,
CLINC150, Amazon ESCI, NVIDIA When2Call, poem_sentiment, iSarcasmEval. Check each source's licence before
reuse; this card does not relicense them.

Format: `cases.jsonl` (one case per line: `family`, `category`, `block`, `source_id`, `state.text`,
`question` with `type`/`instructions`/`criteria`, `label`), `registry.json` (family type and category).
