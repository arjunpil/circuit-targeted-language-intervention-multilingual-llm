# Step 2: intervention and eval harness

How much each intervention reduces code-switching, and what it costs. Works with Qwen2.5-1.5B and Llama-3.2-1B.

## Interventions (`hooks.py`)
- `ResidualSteer`: steering from Goncharov et al. 2025 (arXiv 2510.13849), same as their code: `h + c ((h - mu) . v) v` on the outputs of the last two decoder layers.
- `GatedResidualSteer`: their generation-eval mode (`maintain_direction=True`, `n_source_tokens=20`). The prompt pass is steered everywhere and fixes a reference sign from the mean projection of its first 20 tokens; generated tokens are steered only when their projection has that sign. One prompt at a time.
- `HeadSteer`: the same transform through selected heads' `W_O` only, `c_h + c (c_h . v) v`. `c = -1` removes `v` from those heads.
- `HeadDirSteer`: the same kind of edit along each head's own axis instead of `v`, `z_h + c ((z_h - m_h) . u_h) u_h` at the `o_proj` input. `u_h` is the normalized difference between the head's mean output on `--lang` and on English FLORES text, `m_h` their midpoint. `c = -2` reflects a head across the midpoint. Used for the selected heads and the controls with `--head-dir own`.
- `HeadScale`: scales selected heads.

`v` is PC1 of a per-layer PCA over FLORES-200 dev tokens of both languages. Their code steers layer `l` with the PCA of `hidden_states[l]`, i.e. the previous layer's output. `residual` and `resid_gated` keep that, since it reproduces their numbers. `resid_own` and the head interventions use each layer's own output. The last layer is read before the final norm, because HF returns it normed.

Which head direction works depends on the model. On Llama-3.2-1B-Instruct the circuit heads' language difference, mapped through `W_O`, lines up with `v` (mean |cos| 0.24, random heads 0.07), and `pc1` reaches the same effect at a lower cost than `own`. On Qwen2.5-1.5B-Instruct it is nearly orthogonal to `v` (0.06 vs 0.04), so `pc1` barely moves generation while `own` does.

Paper numbers used as defaults and for comparison are in `reference.py`.

## Eval (`run_pilot.py`)
- Effect: TED prompts from their repo that start in English and switch to `--lang` halfway. CSI is the share of non-English tokens in the continuation. `csi` is their version (fastText `lid.176`, 5 subword tokens, any label). `csi_pair` uses 5 words and only the two languages, and is the one to read.
- Side effects: CSI on the all-English version of the same prompts, FLORES-200 devtest perplexity in both languages, repeated 4-grams.
- Controls: `random*` takes the same number of heads from the same layers (10 distinct draws), `nearby*` from the layer above or below (5 draws). `summary.json` lists where the selected heads fall among them.

Without `--heads-json` the head set is a stand-in, ranked by how differently each head writes along `v` for the two languages. Since it is picked with the same `v` it is steered with, its edge over the controls is partly built in. Use a step 1 circuit for real runs: `{"heads": {"27": [0, 3]}}`.

## LCB (`lcb.py`)
Language Confusion Benchmark (Marchisio et al. 2024, arXiv 2406.20052) on an instruct model with the chat template. Monolingual prompts are in `--lang` and expect a reply in it; crosslingual prompts are English instructions asking for `--lang`. LPR, WPR and line accuracy follow their `compute_metrics.py` and match it on their released completions. `en` is the share of lines identified as English. Generation is greedy, 100 new tokens (they sample with p=0.75, T=0.3).

Head conditions use `--head-coefs` with the same controls as `run_pilot.py`, and `summary.json` compares each condition's LPR change from `none`. `--resid-coefs` adds the steering-paper baseline on `--steer-layers` (default the last two). Every condition also gets FLORES devtest perplexity in English and `--lang`, and `rep4` on the replies, since LPR only checks the language.

With the Llama-3.2-1B en-es circuit on Llama-3.2-1B-Instruct, negative head coefficients turn replies to English in both settings and positive ones reduce crosslingual failures.

## Reproduction checks
- `kl_check.py`: their next-token KL (Tables 3 and 7). Qwen en-es reproduces (7.50 → 4.73 at c = −2.75, paper 7.25 → 4.90 at −2.8). Llama matches unsteered (6.50 vs 6.52) but its best coefficient here is −1.25, not their −3.9.
- `gen_check.py`: their generation eval (Table 9), first TED samples, 100 new tokens. Llama en-es reproduces (CSI 0.62 → 0.22, paper 0.62 → 0.23) and is stable across three PCA calibration samples (`--seed`). On Qwen the same protocol depends on the calibration sample: 0.67 → 0.99, 0.70 → 0.99 and 0.70 → 0.33 for seeds 0, 1, 2.

## Caveats
- TED prompts end in the switched language, so this measures going back to English, not ignoring an instructed language (LCB would).
- When the selected heads fill most of a layer, same-layer draws overlap; that is what `nearby*` is for.
- An empty continuation counts as CSI 0, as in their metric code.

## Run
```
pip install torch transformers scikit-learn fasttext-wheel
python -m intervention.kl_check --model Qwen/Qwen2.5-1.5B --lang es
python -m intervention.gen_check --model meta-llama/Llama-3.2-1B --lang es
python -m intervention.run_pilot --model meta-llama/Llama-3.2-1B --lang es \
    --heads-json discovery/llama-3.2-1b/results/llama32_en_es_heads.json --head-coefs=-1,-3,-5,-10
python -m intervention.lcb --model meta-llama/Llama-3.2-1B-Instruct --lang es \
    --heads-json discovery/llama-3.2-1b/results/llama32_en_es_heads.json
```
Data downloads into `CTLI_DATA` (default `.data/`) on first use. Results go to `results/`, written per condition to `rows.jsonl` and `samples.jsonl` and summarized at the end in `summary.json`.

`lid.py` and `metrics.py` are adapted from github.com/fxlrnrpt/language-steering-in-latent-space (MIT). The LCB metrics in `lcb.py` follow github.com/for-ai/language-confusion (Apache-2.0).
