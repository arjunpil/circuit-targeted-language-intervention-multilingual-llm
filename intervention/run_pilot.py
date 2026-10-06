import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervention import data
from intervention.directions import collect_hidden_states, fit_pca, head_output_means, pc1_accuracy
from intervention.generation import generate, perplexity
from intervention.heads import load_heads, matched_nearby, matched_randoms, rank_heads, save_heads, top_heads
from intervention.hooks import GatedResidualSteer, HeadDirSteer, HeadSteer, ResidualSteer, applied
from intervention.lid import load_lid, token_labels
from intervention.metrics import summarize
from intervention.reference import coef as paper_coef, last_two


def floats(s):
    return [float(x) for x in s.split(",") if x]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--lang", default="es", choices=["es", "ru", "zh", "hi"])
    p.add_argument("--dtype", default="float32")
    p.add_argument("--n-fit", type=int, default=200)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--n-ppl", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steer-layers", default=None, help="default: last two layers, as in the paper")
    p.add_argument("--resid-coefs", default=None, help="baseline, their layer indexing; default: paper coef,-1,1")
    p.add_argument("--resid-gated-coefs", default=None, help="baseline as in their generation eval; default: paper coef")
    p.add_argument("--resid-own-coefs", default="-1", help="baseline, each layer's own output")
    p.add_argument("--head-coefs", default="-1,-3,-10")
    p.add_argument("--head-dir", default="pc1", choices=["pc1", "own"],
                   help="pc1: shared residual PC1 through W_O; own: each head's mean output difference")
    p.add_argument("--heads-json", default=None)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--head-min-layer", type=int, default=0)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--n-nearby", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/step2_pilot")
    return p.parse_args()


def main():
    a = parse()
    t0 = time.time()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out) / f"{a.model.split('/')[-1]}_{a.lang}"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 attn_implementation="sdpa").cuda().eval()
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    lid = load_lid(data.lid_path())
    tgt = "en"
    steer_layers = [int(x) for x in a.steer_layers.split(",")] if a.steer_layers else last_two(model)
    c0 = paper_coef(a.model, a.lang)
    resid_coefs = floats(a.resid_coefs) if a.resid_coefs is not None else [c for c in (c0, -1.0, 1.0) if c is not None]
    gated_coefs = floats(a.resid_gated_coefs) if a.resid_gated_coefs is not None else [c for c in (c0,) if c is not None]
    heads = load_heads(a.heads_json) if a.heads_json else None
    head_layers = list(heads) if heads else list(range(a.head_min_layer, L))

    fit_idx = rng.sample(range(len(data.flores("en"))), a.n_fit)
    fit = {lang: [data.flores(lang)[i] for i in fit_idx] for lang in (a.lang, tgt)}
    near = {m for l in head_layers for m in (l - 1, l, l + 1) if 0 <= m < L}
    indices = {m + 1 for m in near} | {i for l in steer_layers for i in (l, l + 1)}
    hs = {lang: collect_hidden_states(model, tok, fit[lang], indices, batch_size=a.batch_size) for lang in fit}
    stats = fit_pca(hs)
    acc = pc1_accuracy(stats, hs, a.lang, tgt)
    del hs
    dirs = {l: stats[l + 1]["pc1"] for l in range(L) if l + 1 in stats}
    means = {l: stats[l + 1]["mean"] for l in range(L) if l + 1 in stats}
    ref_dirs = {l: stats[l]["pc1"] for l in steer_layers}
    ref_means = {l: stats[l]["mean"] for l in steer_layers}

    if heads:
        head_source = a.heads_json
        (out / "head_scores.json").unlink(missing_ok=True)
    else:
        scores = rank_heads(model, tok, fit, dirs, a.lang, tgt, layers=head_layers, batch_size=a.batch_size)
        heads, head_source = top_heads(scores, a.top_k), "stand-in: top-k by alignment with the language direction"
        with open(out / "head_scores.json", "w") as f:
            json.dump([{"layer": l, "head": h, "d": d} for l, h, d in scores], f)
    save_heads(heads, out / "heads.json", source=head_source)
    randoms = matched_randoms(heads, H, a.n_random, a.seed)
    nearby = [matched_nearby(heads, H, L, seed=a.seed + 1000 + i) for i in range(a.n_nearby)]
    if a.head_dir == "own":
        used = sorted({l for sel in (heads, *randoms, *nearby) for l in sel})
        mu = {lang: head_output_means(model, tok, fit[lang], used, batch_size=a.batch_size) for lang in fit}

    def head_iv(sel, c):
        return HeadDirSteer(sel, c, mu[a.lang], mu[tgt]) if a.head_dir == "own" else HeadSteer(sel, c, dirs)

    ted_idx = rng.sample(range(len(data.ted_code_switch("en"))), a.n_eval)
    prompts = {"cs": [data.ted_code_switch(a.lang)[i] for i in ted_idx],
               "en": [data.ted_code_switch("en")[i] for i in ted_idx]}
    ppl_idx = rng.sample(range(len(data.flores("en", "devtest"))), a.n_ppl)
    ppl_sets = {lang: [data.flores(lang, "devtest")[i] for i in ppl_idx] for lang in (tgt, a.lang)}

    def cond(name, coef, ivs, per_prompt=None):
        return {"name": name, "coef": coef, "ivs": ivs, "per_prompt": per_prompt}

    conditions = [cond("none", 0.0, [])]
    conditions += [cond("residual", c, [ResidualSteer(steer_layers, c, ref_means, ref_dirs)]) for c in resid_coefs]
    conditions += [cond("resid_gated", c, [ResidualSteer(steer_layers, c, ref_means, ref_dirs)],
                        lambda c=c: GatedResidualSteer(steer_layers, c, ref_means, ref_dirs)) for c in gated_coefs]
    conditions += [cond("resid_own", c, [ResidualSteer(steer_layers, c, means, dirs)]) for c in floats(a.resid_own_coefs)]
    for name, sets in (("heads", [heads]), ("random", randoms), ("nearby", nearby)):
        conditions += [cond(name if name == "heads" else f"{name}{i}", c, [head_iv(hs, c)])
                       for i, hs in enumerate(sets) for c in floats(a.head_coefs)]

    def gen(cnd, ps):
        if cnd["per_prompt"] is None:
            with applied(model, *cnd["ivs"]):
                return generate(model, tok, ps, a.max_new_tokens, a.batch_size)
        ids = []
        for prompt in ps:
            with applied(model, cnd["per_prompt"]()):
                ids += generate(model, tok, [prompt], a.max_new_tokens, 1)
        return ids

    pair = (tgt, a.lang)
    rows = []
    open(out / "samples.jsonl", "w").close()
    for cnd in conditions:
        name, coef = cnd["name"], cnd["coef"]
        res, samples = {}, []
        for split, ps in prompts.items():
            per = []
            for i, g in zip(ted_idx, gen(cnd, ps)):
                m = summarize(token_labels(g, tok, lid), g, tgt)
                m["csi_pair"] = summarize(token_labels(g, tok, lid, allowed=pair, by_word=True), g, tgt)["csi"]
                per.append(m)
                samples.append({"condition": name, "coef": coef, "split": split, "ted_idx": i,
                                "text": tok.decode(g, skip_special_tokens=True), **m})
            for k in ("csi", "csi_pair", "m_index", "i_index", "rep4"):
                res[f"{split}_{k}"] = sum(m[k] for m in per) / len(per)
        with applied(model, *cnd["ivs"]):
            for lang, texts in ppl_sets.items():
                res[f"ppl_{lang}"] = perplexity(model, tok, texts, a.batch_size)
        rows.append({"condition": name, "coef": coef, **res})
        with open(out / "rows.jsonl", "a" if len(rows) > 1 else "w") as f:
            f.write(json.dumps(rows[-1]) + "\n")
        with open(out / "samples.jsonl", "a") as f:
            f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
        print(f"{name:11s} c={coef:6.2f}  cs_csi={res['cs_csi']:.3f}/{res['cs_csi_pair']:.3f}  "
              f"en_csi={res['en_csi']:.3f}  rep4={res['cs_rep4']:.3f}  ppl_en={res['ppl_en']:.2f}  "
              f"ppl_{a.lang}={res['ppl_' + a.lang]:.2f}", flush=True)

    controls = {}
    for coef in floats(a.head_coefs):
        target = next(r["cs_csi_pair"] for r in rows if r["condition"] == "heads" and r["coef"] == coef)
        for kind in ("random", "nearby"):
            vals = [r["cs_csi_pair"] for r in rows if r["condition"].startswith(kind) and r["coef"] == coef]
            if vals:
                controls.setdefault(str(coef), {})[kind] = {
                    "heads": target, "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals),
                    "share_at_or_below_heads": sum(v <= target for v in vals) / len(vals), "n": len(vals)}

    summary = {
        "args": vars(a),
        "steer_layers": steer_layers,
        "resid_coefs": resid_coefs,
        "resid_gated_coefs": gated_coefs,
        "heads": {str(l): hs for l, hs in heads.items()},
        "head_source": head_source,
        "random_heads": [{str(l): hs for l, hs in r.items()} for r in randoms],
        "nearby_heads": [{str(l): hs for l, hs in r.items()} for r in nearby],
        "controls": controls,
        "pc1_accuracy": {str(k): v for k, v in acc.items()},
        "pc1_explained_variance": {str(k): stats[k]["explained_variance_ratio"][0] for k in stats},
        "rows": rows,
        "minutes": (time.time() - t0) / 60,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved to {out} ({summary['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
