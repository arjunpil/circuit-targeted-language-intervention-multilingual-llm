# Llama-3.2-1B EN→ES Circuit Discovery and Validation

This directory contains the EN→ES circuit-discovery and held-out
validation results for `meta-llama/Llama-3.2-1B`.

## Language metric

The discovery objective uses model-specific English and Spanish token sets
derived from FLORES-200 dev with the Llama tokenizer.

The metric uses 200 tokens per language. Candidate tokens must occur at least
five times across the English and Spanish corpora and are ranked by:

`log((Spanish count + 1) / (English count + 1))`

On a 40-example held-out FLORES devtest check (20 English and 20 Spanish
examples), the metric achieved 100% sign accuracy. Mean metric values were
-5.3992 for English and +6.5965 for Spanish.

## Head discovery

Attention-head outputs were screened on FLORES indices 0-29 using
first-order attribution and exact activation patching.

Across the top 30 attribution candidates:

- Spearman rho between mean attribution and exact effects: 0.9466
- p-value: 2.70e-15

Using the same selection rule as the Qwen experiment (mean exact effect
>= 0.10 and attribution/exact sign agreement >= 0.75), the frozen set
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

The frozen set is stored in `results/llama32_en_es_heads.json`.

## Full held-out head validation

The frozen 12-head set was evaluated on 50 fresh FLORES pairs
(indices 110-159) against 100 unique layer-matched random head sets.
The control sets preserve the number of selected heads in each layer while
excluding the selected heads themselves.

Spanish activations patched into English prompts (sufficiency):

- selected mean effect: +3.8811
- bootstrap 95% CI: [+3.3913, +4.3893]
- random-control 95th percentile: +0.0780
- percentile versus controls: 100%

English activations patched into Spanish prompts (reverse patch):

- selected mean effect: +4.8071
- bootstrap 95% CI: [+4.1161, +5.5450]
- random-control 95th percentile: +0.2891
- percentile versus controls: 100%

Both predefined validation tests passed.

The earlier 10-example pilot on indices 100-109 is retained in
`results/llama32_en_es_head_holdout_pilot.json`.

## Source-to-destination path discovery

All earlier-layer to later-layer pairs among the frozen heads were tested
on FLORES indices 200-229. This produced 53 candidate paths.

- per-example attribution/exact Spearman rho: 0.9998
- path-mean attribution/exact Spearman rho: 0.9981

The five paths with the largest absolute mean exact effects were frozen
before evaluating the path holdout:

- L13H4 -> L14H5: +0.050766
- L12H7 -> L14H18: +0.024209
- L12H7 -> L15H24: -0.020360
- L12H7 -> L14H17: +0.019881
- L12H7 -> L13H4: -0.017523

The frozen set is stored in `results/llama32_en_es_frozen_paths.json`.

## Held-out path validation

The five frozen paths were evaluated on 40 fresh FLORES pairs
(indices 300-339), with 50 matched random controls per path. Controls
preserve the source and destination layers while replacing both head
identities.

| Path | Discovery | Holdout | Aligned 95% CI | Random p95 |
| --- | ---: | ---: | ---: | ---: |
| L13H4 -> L14H5 | +0.050766 | +0.056266 | [+0.042816, +0.070874] | +0.001304 |
| L12H7 -> L14H18 | +0.024209 | +0.017818 | [+0.010805, +0.025552] | +0.000908 |
| L12H7 -> L15H24 | -0.020360 | -0.023136 | [+0.016162, +0.030170] | +0.002574 |
| L12H7 -> L14H17 | +0.019881 | +0.019737 | [+0.009743, +0.032442] | +0.001402 |
| L12H7 -> L13H4 | -0.017523 | -0.011952 | [+0.005062, +0.020690] | +0.000776 |

All five paths replicated in their discovery direction, their aligned
bootstrap confidence intervals excluded zero, and each path exceeded all
50 matched controls.

For paths with negative discovery effects, the confidence interval is
reported after multiplying the holdout effects by the frozen discovery
sign. The raw holdout effect remains negative.

## Reproducing the experiments

Access to the gated `meta-llama/Llama-3.2-1B` checkpoint is required.

Build the language metric:

```bash
python discovery/llama-3.2-1b/build_language_metric.py
```

Run head discovery:

```bash
python discovery/llama-3.2-1b/run_head_discovery.py \
  --start 0 \
  --n-examples 30 \
  --top-k 30 \
  --out results/llama32_discovery/head_discovery.json
```

Run the full held-out head validation:

```bash
python discovery/llama-3.2-1b/run_head_validation.py \
  --start 110 \
  --n-examples 50 \
  --n-random-controls 100 \
  --seed 42 \
  --out results/llama32_full_validation/head_validation.json
```

Run path discovery:

```bash
python discovery/llama-3.2-1b/run_path_validation.py \
  --start 200 \
  --n-examples 30 \
  --out results/llama32_path_validation/path_discovery.json
```

Run the frozen-path holdout:

```bash
python discovery/llama-3.2-1b/run_path_holdout.py \
  --start 300 \
  --n-examples 40 \
  --n-random-controls 50 \
  --seed 42 \
  --out results/llama32_path_validation/path_holdout.json
```

Generated outputs under the repository-level `results/` directory are
ignored by Git. Validated experiment artifacts are committed under this
directory's `results/` folder.

## Scope

These experiments establish reproducible EN→ES head-level and
source-to-destination path-level causal effects in Llama-3.2-1B under the
language-metric activation-patching setup used here. They do not by
themselves establish that the same circuit explains broader
content-triggered language confusion or generalizes across languages and
model scales.

## Other EN<->target language pairs

`common.py`, `build_language_metric.py`, `run_head_discovery.py`, and
`run_head_validation.py` take a `--lang` argument instead of being fixed to
Spanish. `LANG_CODES` in `common.py` carries the FLORES-200 code for each
supported target (`es`, `ru`, `zh`, `hi`, `ko`); `--lang es` reproduces every
command and result above exactly, since the default metric, heads, and output
paths are unchanged for that case.

This has not yet been run for Llama-3.2-1B: the checkpoint is gated and the
environment this generalization was written in had no Hugging Face token
with access to it. `discovery/qwen2.5-1.5b/README.md` has the EN->RU/ZH/HI
results for Qwen2.5-1.5B under the identical pipeline -- RU and ZH both
replicate on held-out data with wide margins, HI's single-head discovery
result does not. Whether Llama's heads behave the same way (and whether the
same EN->ES heads reappear the way they do for Qwen) is open.

To discover and validate a circuit for another language (requires access to
the gated `meta-llama/Llama-3.2-1B` checkpoint):

```bash
python discovery/llama-3.2-1b/build_language_metric.py --lang ru
python discovery/llama-3.2-1b/run_head_discovery.py --lang ru \
  --out results/discovery_repro/llama32_head_discovery_ru.json
python discovery/freeze_heads.py \
  --discovery results/discovery_repro/llama32_head_discovery_ru.json \
  --out discovery/llama-3.2-1b/results/llama32_en_ru_heads.json
python discovery/llama-3.2-1b/run_head_validation.py --lang ru \
  --out discovery/llama-3.2-1b/results/llama32_en_ru_head_validation_holdout.json
```

`freeze_heads.py` turns a `run_head_discovery.py` dump into the compact
heads.json format (`llama32_en_<lang>_heads.json`) that `run_head_validation.py`
and the `intervention/` harness consume. That filename matches the
`heads_json: discovery/{model_dir}/results/{model}_en_{lang}_heads.json`
template `experiments/configs/language_grid.yaml` already uses, so each
validated language slots directly into the existing `qwen25`/`llama32` x
`es`/`ru`/`zh`/`hi` experiment grid without further config changes.

Comparing the frozen head sets across target languages (same model, same
selection rule) is how this repository tests whether EN<->{es,ru,zh,hi}
rely on a shared language-suppression subcircuit or on mostly disjoint
per-language heads, and whether that overlap (or lack of it) matches the
Qwen2.5-1.5B result in `discovery/qwen2.5-1.5b/README.md`.
