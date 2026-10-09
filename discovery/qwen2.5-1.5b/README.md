# Qwen2.5-1.5B EN→ES Circuit Discovery Pilot

This directory contains the first circuit-discovery and validation pilot for
`Qwen/Qwen2.5-1.5B`.

## Language metric

The discovery objective uses an English-vs-Spanish token-set log-probability
metric derived from FLORES-200. The metric achieved 97.5% sign accuracy on
held-out English and Spanish prefixes.

## Head discovery

Attention-head outputs were screened using first-order attribution and then
validated with exact activation patching.

Across the top 30 head-output candidates:

- Spearman rho between attribution and exact effects: 0.8412
- p-value: 5.87e-09

The resulting five-head pilot circuit is:

- L16H9
- L17H7
- L22H6
- L25H10
- L27H6

The intervention-compatible head set is stored in
`results/qwen25_en_es_heads.json`.

## Held-out head-level validation

The five heads were frozen after discovery and evaluated on 50 disjoint
FLORES examples against 100 size- and layer-matched random head sets.

Spanish activations patched into English prompts:

- mean effect: +1.2216
- 95% bootstrap CI: [+1.0542, +1.3983]
- random-control 95th percentile: +0.0989

English activations patched into Spanish prompts:

- mean effect: +0.8613
- 95% bootstrap CI: [+0.4062, +1.3958]
- random-control 95th percentile: +0.1101

Both directions exceeded the 95th percentile of the matched random controls.

## Base-to-Instruct head-set transfer

The frozen five-head set was also evaluated without rediscovery on
`Qwen/Qwen2.5-1.5B-Instruct`. To isolate the model-weight change from prompt
formatting, this comparison uses the same raw FLORES-200 prefixes as the base
model rather than applying the instruction model's chat template.

On 50 FLORES examples with 100 layer-matched random controls:

Spanish activations patched into English prompts:

- mean effect: +1.4457
- 95% bootstrap CI: [+1.2532, +1.6464]
- random-control 95th percentile: +0.1105
- percentile versus controls: 100%

English activations patched into Spanish prompts:

- mean effect: +0.9632
- 95% bootstrap CI: [+0.4862, +1.5030]
- random-control 95th percentile: +0.1157
- percentile versus controls: 100%

Both directions pass the same held-out validation gates on the instruction
model. A code-matched rerun of `Qwen/Qwen2.5-1.5B` with this validation runner
reproduces the selected-head effects at +1.2216 and +0.8613.

Validation summaries are stored in:

- `results/qwen25_base_en_es_head_validation_code_matched.json`
- `results/qwen25_instruct_en_es_head_validation_holdout.json`

The validation runner can reproduce the full per-example and matched-control
outputs when a detailed output path is supplied with `--out`.

This establishes transfer of the frozen head set under direct activation
patching. It does not by itself establish that a particular steering direction
transfers between the base and instruction-tuned models.

## Source-to-destination path validation

Source-head interventions were propagated through the model and the resulting
change in a selected downstream head was isolated with destination-head
patching.

Five candidate paths were frozen after discovery and evaluated on a disjoint
40-example holdout set with source-layer- and destination-layer-matched random
head-pair controls.

Three paths passed all held-out gates:

- L16H9 → L25H10
- L17H7 → L25H10
- L22H6 → L25H10

Their held-out mean effects were +0.0072, +0.0129, and +0.0078 respectively,
with bootstrap confidence intervals excluding zero and effects above the
95th percentile of their matched random controls.

## L25H10 convergence motif

The three validated source paths converge on L25H10. The frozen convergence
motif was evaluated on another disjoint 40-example holdout set.

Combined motif effect:

- mean effect: +0.0146
- 95% bootstrap CI: [+0.0061, +0.0240]

Controls:

- fixed L25H10 destination with randomized source heads:
  95th percentile = +0.0017
- fixed source heads with randomized layer-25 destination heads:
  95th percentile = +0.0047

The selected motif exceeded both control distributions.

The compact circuit description is stored in
`results/qwen25_en_es_circuit.json`.

## Scope

These results establish a held-out EN→ES pilot circuit in Qwen2.5-1.5B and
show that the frozen head set remains causally effective under direct
activation patching in Qwen2.5-1.5B-Instruct. Replication across additional
language pairs and model scales is required before making broader claims about
multilingual language-identity circuitry.

## Other EN<->target language pairs

`common.py`, `build_language_metric.py`, `run_head_discovery.py`, and
`run_head_validation.py` take a `--lang` argument instead of being fixed to
Spanish. `LANG_CODES` in `common.py` carries the FLORES-200 code for each
supported target (`es`, `ru`, `zh`, `hi`, `ko`); `--lang es` reproduces every
command and result above exactly, since the default metric, heads, and output
paths are unchanged for that case.

The identical discovery (30 examples, same selection rule: mean exact effect
>= 0.10, sign agreement >= 0.75) and held-out validation (50 fresh FLORES
pairs, indices 100-149, against layer-matched random controls) used for
EN->ES above were run for EN->RU, EN->ZH, and EN->HI:

| pair | heads | discovery rho | sufficiency mean | vs random p95 | necessity mean | vs random p95 | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| en->es | 5  | 0.8412 | +1.2216 | +0.0989 | +0.8613 | +0.1101 | yes |
| en->ru | 13 | 0.9497 | +5.2045 | +0.3717 | +3.6745 | +0.3448 | yes |
| en->zh | 14 | 0.7366 | +6.6269 | +2.0458 | +3.7011 | +0.8665 | yes |
| en->hi | 1  | 0.7006 | +0.0047 | +0.0384 | +0.1475 | +0.1685 | **no** |

RU and ZH both replicate on held-out data with wide margins over their random
controls. HI's 30-example screen selected a single head (L25H10), which does
not hold up out of sample: its sufficiency bootstrap CI straddles zero and
sits below the random-control 95th percentile, and necessity falls short of
its control bar too. Read this as the EN->HI screen not finding a circuit at
this example count and threshold, not as evidence EN->HI has no circuit; a
rerun with more discovery examples or a lower effect threshold would be
needed before concluding either way. With only one selected head, there are
only `C(11, 1) = 11` layer-matched random head sets available, so
`run_head_validation.py` clamps `--n-random-controls` down to that
combinatorial ceiling (a bug this run exposed and fixed: the script used to
loop forever trying to draw more unique controls than exist) and the
resulting p95 is correspondingly noisy next to the 100-control estimates for
RU, ZH and ES.

Comparing the frozen head sets themselves (same model, same selection rule)
across targets:

| head | es | ru | zh | hi |
| --- | --- | --- | --- | --- |
| L16H9  | selected | selected | | |
| L17H7  | selected | selected | selected | |
| L22H6  | selected | selected | | |
| L25H10 | selected | selected | selected | selected (discovery only; fails holdout) |
| L27H6  | selected | selected | | |

ES's five heads are a strict subset of RU's thirteen. ZH keeps two of the
five (L17H7, L25H10). L25H10 is the only head selected in every language
screened, and it is the same head `## L25H10 convergence motif` above
identifies as where the EN->ES source paths converge -- consistent with a
shared core that a validated EN->HI circuit may or may not extend to. Full
per-example results are in `results/qwen25_en_{ru,zh,hi}_heads.json` and
`results/qwen25_en_{ru,zh,hi}_head_validation_holdout.json`.

To discover and validate a circuit for another language:

```bash
python discovery/qwen2.5-1.5b/build_language_metric.py --lang ru
python discovery/qwen2.5-1.5b/run_head_discovery.py --lang ru \
  --out results/discovery_repro/qwen25_head_discovery_ru.json
python discovery/freeze_heads.py \
  --discovery results/discovery_repro/qwen25_head_discovery_ru.json \
  --out discovery/qwen2.5-1.5b/results/qwen25_en_ru_heads.json
python discovery/qwen2.5-1.5b/run_head_validation.py --lang ru \
  --out discovery/qwen2.5-1.5b/results/qwen25_en_ru_head_validation_holdout.json
```

`freeze_heads.py` turns a `run_head_discovery.py` dump into the compact
heads.json format (`qwen25_en_<lang>_heads.json`) that `run_head_validation.py`
and the `intervention/` harness consume. That filename matches the
`heads_json: discovery/{model_dir}/results/{model}_en_{lang}_heads.json`
template `experiments/configs/language_grid.yaml` already uses, so each
validated language slots directly into the existing `qwen25`/`llama32` x
`es`/`ru`/`zh`/`hi` experiment grid without further config changes.

## Reproducing the discovery pipeline

The discovery scripts can be run from the repository root.
FLORES-200 is downloaded and extracted under `discovery/data/` when it is
not already available. The data directory is ignored by Git.

The EN→ES objective uses the fixed English and Spanish token sets stored in
`results/en_es_language_metric.json` so repeated runs use the same metric.

### Head discovery

Run the 30-example attention-head discovery experiment with:

```bash
python discovery/qwen2.5-1.5b/run_head_discovery.py \
  --start 0 \
  --n-examples 30 \
  --top-k 30 \
  --out results/discovery_repro/head_discovery_full.json
```

The script screens attention-head outputs with first-order attribution and
then evaluates the top 30 candidates with exact activation patching.

For Qwen2.5-1.5B on the EN→ES experiment, this produces:

- Spearman rho = 0.8411568409 between mean attribution and exact effects
- L16H9
- L17H7
- L22H6
- L25H10
- L27H6

Heads are selected when their mean exact effect is at least +0.10 and their
attribution/exact sign agreement is at least 0.75.

### Head-set validation and Base-to-Instruct transfer

Run the frozen five-head validation on the base model with:

```bash
python discovery/qwen2.5-1.5b/run_head_validation.py \
  --model Qwen/Qwen2.5-1.5B \
  --start 100 \
  --n-examples 50 \
  --n-random-controls 100 \
  --seed 42
```

Evaluate the same frozen heads on the instruction-tuned model with:

```bash
python discovery/qwen2.5-1.5b/run_head_validation.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --start 100 \
  --n-examples 50 \
  --n-random-controls 100 \
  --seed 42
```

Both commands use raw FLORES prefixes without a chat template so that the
Base-to-Instruct comparison changes the model while keeping the activation
patching protocol fixed.

### Source-to-destination path validation

Run the source-to-destination path analysis with:

```bash
python discovery/qwen2.5-1.5b/run_path_validation.py \
  --start 200 \
  --n-examples 30 \
  --out results/discovery_repro/path_validation_full.json
```

For every earlier-to-later pair among the selected heads, the script patches
the Spanish source-head activation into the English run, lets the intervention
propagate through the model, captures the resulting destination-head output,
and then patches that destination head into an otherwise unchanged English run.

The EN→ES experiment gives a Spearman correlation of 0.9998406649 across
example-path attribution and exact effects, and 1.0 across path means.

The largest path effects are:

- L17H7 → L25H10: +0.010844
- L22H6 → L25H10: +0.009696
- L22H6 → L27H6: +0.007898
- L16H9 → L25H10: +0.007166
- L16H9 → L27H6: -0.004845

Additional held-out validation results and matched-control comparisons are
stored under `discovery/qwen2.5-1.5b/results/`.

### Generated outputs

The commands above write generated outputs under the repository-level
`results/` directory. That directory is ignored by Git. The validated
experiment artifacts under `discovery/qwen2.5-1.5b/results/` remain unchanged.
