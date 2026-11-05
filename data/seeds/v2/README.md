---
license: other
language: [en]
pretty_name: Zils base-exam public seed v2
---
# Zils base-exam public seed v2

Public development sample for the Zils decision-quality base election (`zils.base_exam seed` bundle,
`manifest.json`). 299 decision items across 15 families (5 areas x choice / yes-no / score), with labels.
Publishing these families makes them "known" families for later private election draws; no private
election items are in this dataset.

Each item was newly written by Claude Opus 5.5, using one row from the held-out split of a public dataset
only as a style example (`source_id` names that row), then kept only if an independent blind answer matched.
No benchmark test-set rows and no customer data were used.

Style sources (held-out validation/test splits, none used by Decision Index or S1MB): CommonsenseQA,
SVAMP, GLUE MNLI/RTE/STS-B/QNLI, 20 Newsgroups, TweetEval emotion and irony, Glaive function calling. Check each source's licence before
reuse; this card does not relicense them.

Format: `cases.jsonl` (one case per line: `family`, `category`, `block`, `source_id`, `state.text`,
`question` with `type`/`instructions`/`criteria`, `label`), `registry.json` (family type and category).
