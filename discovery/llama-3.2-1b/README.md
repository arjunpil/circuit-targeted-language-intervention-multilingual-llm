# Llama-3.2-1B EN→ES Circuit Discovery Pilot

This directory contains an EN→ES circuit-discovery pilot for
`meta-llama/Llama-3.2-1B`.

## Language metric

The discovery objective uses model-specific English and Spanish token sets
derived from FLORES-200 dev with the Llama tokenizer.

The metric uses 200 tokens per language. Candidate tokens must occur at least
five times across the English and Spanish corpora, and are ranked by:

`log((Spanish count + 1) / (English count + 1))`

On a 40-example held-out FLORES devtest check (20 English and 20 Spanish
examples), the metric achieved 100% sign accuracy. Mean metric values were
-5.3992 for English and +6.5965 for Spanish.

## Head discovery

Attention-head outputs were screened on 30 aligned FLORES examples using
first-order attribution and exact activation patching.

Across the top 30 attribution candidates:

- Spearman rho between attribution and exact effects: 0.9466
- p-value: 2.70e-15

Using the same selection rule as the Qwen pilot (mean exact effect >= 0.10
and attribution/exact sign agreement >= 0.75), the frozen candidate head set
contains 12 heads:

- L8H25
- L9H13
- L12H7
- L12H9
- L12H17
- L13H4
- L14H4
- L14H5
- L14H17
- L14H18
- L14H23
- L15H24

The intervention-compatible set is stored in
`results/llama32_en_es_heads.json`.

## Preliminary held-out check

The frozen 12-head set was evaluated on 10 disjoint FLORES examples
(indices 100-109) against 20 layer-matched random head sets.

Spanish activations patched into English prompts:

- selected mean effect: +4.4707
- random-control 95th percentile: +0.1257
- percentile versus controls: 100%

English activations patched into Spanish prompts:

- selected mean effect: +4.7622
- random-control 95th percentile: +0.3705
- percentile versus controls: 100%

This is a preliminary held-out pilot rather than the final validation.
A larger disjoint holdout with more matched controls is still required.

## Reproducing the metric

From the repository root:

```bash
python discovery/llama-3.2-1b/build_language_metric.py
```

Access to the gated `meta-llama/Llama-3.2-1B` checkpoint is required.

## Reproducing head discovery

From the repository root:

```bash
python discovery/llama-3.2-1b/run_head_discovery.py \
  --start 0 \
  --n-examples 30 \
  --top-k 30 \
  --out results/llama32_discovery/head_discovery.json
```

Generated outputs under the repository-level `results/` directory are ignored
by Git. The frozen pilot artifacts are stored under this directory's
`results/` folder.

## Scope

These results show that the EN→ES head-discovery signal replicates on a second
model family at the head level. The held-out check is preliminary. Full
held-out validation, source-to-destination path validation, and downstream
intervention evaluation remain to be completed before making broader claims.
