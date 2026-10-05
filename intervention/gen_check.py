import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervention import data
from intervention.directions import collect_hidden_states, fit_pca
from intervention.generation import generate
from intervention.hooks import GatedResidualSteer, ResidualSteer, applied
from intervention.lid import load_lid, token_labels
from intervention.metrics import summarize
from intervention.reference import GEN, coef as paper_coef, last_two

def boot_ci(d, n=4000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--lang", default="es", choices=["es", "ru", "zh", "hi"])
    p.add_argument("--coef", type=float, default=None)
    p.add_argument("--layers", default=None)
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n-fit", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/gen_check")
    a = p.parse_args()

    name = a.model.split("/")[-1]
    paper = GEN.get(name, {}).get(a.lang)
    coef = a.coef if a.coef is not None else paper_coef(name, a.lang)
    if coef is None:
        raise SystemExit(f"no paper coefficient for {name} {a.lang}, pass --coef")

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else last_two(model)
    lid = load_lid(data.lid_path())

    rng = random.Random(a.seed)
    fit_idx = rng.sample(range(len(data.flores("en"))), a.n_fit)
    fit = {lang: [data.flores(lang)[i] for i in fit_idx] for lang in (a.lang, "en")}
    stats = fit_pca({lang: collect_hidden_states(model, tok, fit[lang], layers) for lang in fit})
    means = {l: stats[l]["mean"] for l in layers}
    dirs = {l: stats[l]["pc1"] for l in layers}

    en, cs = list(data.ted_code_switch("en")[:a.n]), list(data.ted_code_switch(a.lang)[:a.n])
    gens = {"english": generate(model, tok, en, a.max_new_tokens, a.batch_size),
            "unsteered": generate(model, tok, cs, a.max_new_tokens, a.batch_size)}
    if a.no_gate:
        with applied(model, ResidualSteer(layers, coef, means, dirs)):
            gens["steered"] = generate(model, tok, cs, a.max_new_tokens, a.batch_size)
    else:
        gens["steered"] = []
        for prompt in cs:
            with applied(model, GatedResidualSteer(layers, coef, means, dirs)):
                gens["steered"] += generate(model, tok, [prompt], a.max_new_tokens, 1)

    pair = ("en", a.lang)
    per = {k: [] for k in gens}
    for k, ids in gens.items():
        for g in ids:
            m = summarize(token_labels(g, tok, lid), g, "en")
            m["csi_pair"] = summarize(token_labels(g, tok, lid, allowed=pair, by_word=True), g, "en")["csi"]
            m["text"] = tok.decode(g, skip_special_tokens=True)
            per[k].append(m)

    report = {"model": a.model, "lang": a.lang, "coef": coef, "layers": layers, "gated": not a.no_gate, "n": a.n,
              "paper": dict(zip(("csi_unsteered", "csi_steered"), paper)) if paper else None, "conditions": {}}
    for k, ms in per.items():
        report["conditions"][k] = {key: sum(m[key] for m in ms) / len(ms) for key in ("csi", "csi_pair", "rep4", "i_index")}
    for key in ("csi", "csi_pair"):
        d = [s[key] - u[key] for s, u in zip(per["steered"], per["unsteered"])]
        report[f"steered_minus_unsteered_{key}"] = boot_ci(d)

    c = report["conditions"]
    tag = "gated" if not a.no_gate else "ungated"
    print(f"{name} en-{a.lang} c={coef:+.2f} {tag}, n={a.n}")
    for k in ("english", "unsteered", "steered"):
        print(f"  {k:10s} csi {c[k]['csi']:.3f}  csi_pair {c[k]['csi_pair']:.3f}  rep4 {c[k]['rep4']:.3f}")
    if paper:
        print(f"  paper      english 0.02, unsteered {paper[0]:.2f}, steered {paper[1]:.2f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{a.lang}_{tag}_c{coef:+.2f}"
    with open(out / f"{stem}.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out / f"{stem}_samples.jsonl", "w") as f:
        for k, ms in per.items():
            f.writelines(json.dumps({"condition": k, "idx": i, **m}, ensure_ascii=False) + "\n" for i, m in enumerate(ms))


if __name__ == "__main__":
    main()
