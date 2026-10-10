# circuit-targeted-language-intervention-multilingual-llm

This repository identifies the attention heads that causally control which
language a multilingual LLM continues in, and intervenes on them.

Step 1 (`discovery/`) finds a small set of attention heads whose output
controls a model's EN<->target language next-token distribution, validated
with activation patching on held-out data against layer-matched random
controls. Step 2 (`intervention/`) steers and scales those heads and
compares the effect to a residual-steering baseline from the literature, on
code-switch continuation (TED) and the Language Confusion Benchmark (LCB).

## Models and language pairs

| model | en->es | en->fr | en->ru | en->zh | en->hi |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | validated | validated | validated | validated | discovery found no held-out-robust circuit |
| Llama-3.2-1B | validated | sufficient, fails held-out necessity | validated | validated | validated |

"Validated" means the frozen head set passed held-out sufficiency and
necessity against 100 layer-matched random control sets. Full numbers,
discovery methodology, and reproduction commands are in
`discovery/qwen2.5-1.5b/README.md` and `discovery/llama-3.2-1b/README.md`.

## Cross-language and cross-model circuit comparison

Comparing the frozen Qwen2.5-1.5B head sets across targets (same model, same
selection rule):

| head | es | fr | ru | zh | hi |
| --- | --- | --- | --- | --- | --- |
| L16H9  | selected | | selected | | |
| L17H7  | selected | | selected | selected | |
| L22H6  | selected | | selected | | |
| L25H7  | | selected | selected | selected | |
| L25H10 | selected | selected | selected | selected | selected (discovery only; fails holdout) |
| L27H6  | selected | | selected | | |

ES's five heads are a strict subset of RU's thirteen; ZH keeps two of the
five; FR keeps one (L25H10) but shares L25H7 with RU and ZH instead. L25H10
is selected at discovery time in every language tried and is the same head
`discovery/qwen2.5-1.5b/README.md` identifies as where the EN->ES source
paths converge, though only es/fr/ru/zh validate it as causally sufficient
and necessary on held-out data -- HI's single-head discovery result does
not replicate (see that README for the numbers).

A minimality ablation splitting RU's and ZH's sets into the subset
overlapping ES's five heads versus the rest shows the shared core's
*causal weight*, not just its presence, shrinking with distance from
Spanish: RU's overlap and extra heads carry roughly equal shares of the
effect (41% and 43% of full sufficiency), while ZH's overlap is a minority
contributor (19%) next to its own extra heads (67%). Full numbers in
`discovery/qwen2.5-1.5b/README.md`.

### Qwen and Llama disagree on how shared the circuit is

Llama-3.2-1B's EN->FR/RU/ZH/HI circuits were discovered and validated the
same way (`discovery/llama-3.2-1b/README.md`). Normalizing by each target
language's own circuit size -- what fraction of a target language's heads
are also in that model's EN->ES circuit -- gives a clean, if informal,
head-count-based measure of how shared each model's multilingual circuitry
is:

| pair | Qwen2.5-1.5B ES-overlap | Llama-3.2-1B ES-overlap |
| --- | ---: | ---: |
| en->fr | 25% (1 of 4) | 75% (3 of 4) |
| en->ru | 38% (5 of 13) | 86% (6 of 7) |
| en->zh | 14% (2 of 14) | 78% (7 of 9) |

Llama's non-ES-pair circuits are 75-86% the same heads as its EN->ES
circuit, with a handful of extras per language. Qwen's are 14-38% --
mostly new, language-specific heads, with only a small shared core (and,
per the minimality ablation above, that core's *causal* share is even
smaller than its head-count share for ZH). Both models have *some* shared
core (L25H10 for Qwen; L14H17 and L14H18, selected in all five of Llama's
pairs, for Llama) and some language-specific recruitment, but at very
different ratios. Llama also validates (sufficiency and necessity) for
RU/ZH/HI, where Qwen's own HI circuit does not -- and Llama's FR circuit
passes sufficiency but fails necessity, the opposite failure pattern from
any of Qwen's results. Any claim this project makes about "shared vs.
disjoint multilingual circuitry" should be read as model-dependent, not as
a single universal answer -- these two 1-1.5B models disagree with each
other on how shared their own multilingual circuits are.

**The two models' EN->ES circuits sit at the same relative depth** despite
having different total depths:

| model | layers used | total layers | depth fraction |
| --- | --- | --- | --- |
| Llama-3.2-1B | 8, 9, 12, 13, 14, 15 | 16 | 0.53 - 1.00 |
| Qwen2.5-1.5B | 16, 17, 22, 25, 27 | 28 | 0.59 - 1.00 |

Both circuits start a little past the midpoint and run to the final layer.
This lines up with the phase structure Wendler et al. (2024, "Do Llamas Work
in English? On Latent Language in Multilingual Transformers," arXiv
2402.10588) report from the logit lens on Llama-2: an English-centric
*concept* representation in middle layers, moving to an input-language-
specific representation in late layers. A circuit that selects the output
language sitting in that late region is consistent with their account,
though we have not run the logit-lens analysis ourselves to check the
transition point directly.

## Independent cross-validation

[MITra](https://github.com/Blyzi/mitra) (Blyzi, code for the ICML 2026 paper
*Translation Heads: Disentangling meaning from language in LLM-based
machine translation*, arXiv 2602.04613) independently identifies a model's
single most consistent "language head" -- the head whose output most often
separates the correct target language from a random fake one, ranked by how
often it comes out on top across language pairs -- using ICL-based
activation patching, a different method from this repo's gradient-
attribution screen plus exact patching.

Running MITra's method (one example, en->fr) on three models this project's
own circuits were already computed for (via a different family, Qwen3
instead of Qwen2.5, for the first row) gives a mixed, genuinely informative
result:

| model | MITra's top head | this project's head | match |
| --- | --- | --- | --- |
| Qwen3-1.7B-Base | L18H12 (15.31, vs. 2.28 runner-up) | L18H12 | exact, by a 6.7x margin |
| Llama-3.2-1B | L12H7 (5.47), with L8H25 2nd (3.87) | L8H25 | top-3 (L12H7, L8H25, L13H4) are all in our 12-head circuit, but not the single top head |
| Gemma-3-1B | L15H2 (8.57, vs. 2.21 runner-up) | L11H3 | no -- L11H3 ranks 85th of 104 heads, slightly negative value |

Qwen3 is a clean independent replication by a different method and a
different model generation. Llama is a real but partial corroboration: the
right small neighborhood of heads, not the same single top head, with a
much tighter margin between candidates than Qwen3's. Gemma is a genuine
miss. These are each a single FLORES example (MITra's own method patches
every head for every example, which costs ~2-2.5 hours per example per
model on CPU here), so treat the Llama and Gemma results as a first pass,
not a final word -- but report them as they came out rather than only
keeping the Qwen3 match.

A closer independent analog in the literature: Zhu et al., *Focusing on
Language: Revealing and Exploiting Language Attention Heads in Multilingual
LLMs* (arXiv 2511.07498), find both language-specific heads and
*language-general* heads (ablating them raises perplexity across every
tested language) in Aya-23-8B, Llama-3.2-3B, and Mistral-7B, using a soft-
mask importance score rather than activation patching. Their language-
general heads are the same kind of finding as this project's cross-language
results, obtained independently -- and this project's own Llama-3.2-1B
result (the same model family they study, L14H17/L14H18 shared across all
five validated language pairs, 71-86% ES-circuit overlap in every other
pair) matches their language-general-head framing more closely than Qwen's
weaker, more language-specific pattern does.

## Related work this project's claims should be read against

- Goncharov et al. (2025), arXiv 2510.13849 -- the residual-steering
  baseline `intervention/kl_check.py` and `intervention/gen_check.py`
  reproduce. Their four language pairs are **es, fr, zh, hi** (not es, ru,
  zh, hi); EN->FR is now run and validated here too, so es/fr/zh/hi are a
  baseline-matched comparison and EN->RU is the extra data point on the
  shared-vs-disjoint-circuit question (see `discovery/qwen2.5-1.5b/README.md`).
- Marchisio et al. (2024), arXiv 2406.20052, the Language Confusion
  Benchmark `intervention/lcb.py` reproduces metrics from. They find
  crosslingual failure (an English instruction asking for another language)
  is the harder, more diagnostic case than monolingual failure -- LCB
  results on the new languages should be read primarily on the crosslingual
  numbers, not monolingual LPR alone.
- Tang et al. (2024) and related 2024-2025 work on language-specific MLP
  neurons (identified via per-language activation-probability entropy)
  report language-specific units concentrated in early *and* late layers,
  with language-agnostic processing in between. This project's attention-
  head circuits are a different granularity and circuit type (attention
  heads feeding `o_proj`, not MLP neurons) and sit in the late region only
  -- a complementary, not competing, locus of language control.

## Repository layout

- `discovery/` -- per-model circuit discovery and held-out validation
  (`qwen2.5-1.5b/`, `llama-3.2-1b/`), plus `freeze_heads.py` and the shared
  FLORES-200 download logic. `qwen2.5-1.5b/run_minimality_ablation.py` is
  the subset-ablation analysis referenced above.
- `intervention/` -- the steering/scaling/patching harness and TED/LCB
  evals; see `intervention/README.md`.
- `attribution/` -- lower-level attribution-vs-exact-patching sanity checks
  that this project's causal claims rest on.
- `experiments/` -- a config-driven runner (`experiments/run.py`) over a
  model x language matrix; `experiments/configs/language_grid.yaml` already
  resolves every validated head set in `discovery/*/results/` to a real
  circuit run.
- `tests/` -- `pytest tests/`.
