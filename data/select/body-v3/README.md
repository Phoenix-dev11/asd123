---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
- zils
- decision
- soup-body
- bittensor
---

# Zils soup body

A body for the Zils `d3-lite` soup pool (Bittensor subnet 127), evaluated on the pool's pinned chassis,
which supplies the answer layer, prompt and runtime.

Started from the uniform average of thermopylae-777/zils-lite-body-2 and dasLOL/zils-lite-skills-v5. Fine-tuned with a merged LoRA using `zils.teacher train`.

## Training data

86,868 rows in total:
- generated and datasets: 33,817
- generated and datasets: dropped (duplicate or in dev): 2
- pin: 13,051
- pin: dropped (duplicate or in dev): 1,949
- relay: 40,000

Dataset rows come from train splits, with one exception noted by source: Crisscrossed Captions
validation. No validator exam data and no customer data were used. Public seed items were excluded
from training. Check each source's licence for reuse; this repository contains weights only.
