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

These results establish a held-out EN→ES pilot circuit in Qwen2.5-1.5B.
Replication across additional language pairs and downstream intervention
evaluation are required before making broader claims about multilingual
language-identity circuitry.

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
