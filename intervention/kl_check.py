import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervention import data
from intervention.directions import collect_hidden_states, fit_pca
from intervention.hooks import ResidualSteer, applied
from intervention.reference import KL, last_two


@torch.no_grad()
def next_token_probs(model, tok, prompts, batch_size=50):
    tok.padding_side = "left"
    out = []
    for s in range(0, len(prompts), batch_size):
        batch = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True).to(model.device)
        out.append(torch.softmax(model(**batch, logits_to_keep=1).logits[:, -1].double(), -1).cpu())
    return torch.cat(out)


def dist_metrics(p, q, eps=1e-10):
    m = (p + q) / 2
    kl = (p * torch.log((p + eps) / (q + eps))).sum(-1)
    js = 0.5 * (p * torch.log((p + eps) / (m + eps))).sum(-1) + 0.5 * (q * torch.log((q + eps) / (m + eps))).sum(-1)
    cos = torch.nn.functional.cosine_similarity(p, q, dim=-1)
    top = [len(set(a.tolist()) & set(b.tolist())) / 50 for a, b in zip(p.topk(50).indices, q.topk(50).indices)]
    return {"kl": kl.mean().item(), "js": js.mean().item(), "cos": cos.mean().item(), "top50": sum(top) / len(top)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--lang", default="es", choices=["es", "ru", "zh", "hi"])
    p.add_argument("--n-fit", type=int, default=200)
    p.add_argument("--n-grid", type=int, default=100)
    p.add_argument("--layers", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/kl_check")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else last_two(model)
    name = a.model.split("/")[-1]

    rng = random.Random(a.seed)
    fit_idx = rng.sample(range(len(data.flores("en"))), a.n_fit)
    fit = {lang: [data.flores(lang)[i] for i in fit_idx] for lang in (a.lang, "en")}
    indices = {i for l in layers for i in (l, l + 1)}
    stats = fit_pca({lang: collect_hidden_states(model, tok, fit[lang], indices) for lang in fit})
    indexing = {"reference": {l: l for l in layers}, "own_output": {l: l + 1 for l in layers}}

    en, cs = list(data.ted_code_switch("en")[:a.n_grid]), list(data.ted_code_switch(a.lang)[:a.n_grid])
    p_en = next_token_probs(model, tok, en)
    base = dist_metrics(p_en, next_token_probs(model, tok, cs))
    paper = KL.get(name, {}).get(a.lang)
    report = {"paper": dict(zip(("kl_unsteered", "kl_steered", "coef"), paper)) if paper else None,
              "unsteered": base, "grid": {}}

    for kind, idx in indexing.items():
        means = {l: stats[i]["mean"] for l, i in idx.items()}
        dirs = {l: stats[i]["pc1"] for l, i in idx.items()}
        rows = []
        for c in np.round(np.linspace(-5, 5, 41), 2).tolist():
            with applied(model, ResidualSteer(layers, c, means, dirs)):
                rows.append({"coef": c, **dist_metrics(p_en, next_token_probs(model, tok, cs))})
        best = min(rows, key=lambda r: r["kl"])
        report["grid"][kind] = {"rows": rows, "best": best}
        line = f"{kind:10s} KL {base['kl']:.2f} -> {best['kl']:.2f} at c={best['coef']:+.2f} ({1 - best['kl'] / base['kl']:.0%} lower)"
        if paper:
            near = min(rows, key=lambda r: abs(r["coef"] - paper[2]))
            line += f"; {near['kl']:.2f} at c={near['coef']:+.2f}"
        print(line, flush=True)
    if paper:
        print(f"{'paper':10s} KL {paper[0]:.2f} -> {paper[1]:.2f} at c={paper[2]:+.2f} ({1 - paper[1] / paper[0]:.0%} lower)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_{a.lang}.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
