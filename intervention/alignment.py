import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervention import data
from intervention.directions import collect_hidden_states, fit_pca, head_output_means
from intervention.heads import load_heads, matched_randoms
from intervention.hooks import head_dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--heads-json", required=True)
    p.add_argument("--lang", default="es", choices=["es", "ru", "zh", "hi", "ko"])
    p.add_argument("--n-fit", type=int, default=200)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa")
    model = model.to(a.device).eval()
    H, DH = model.config.num_attention_heads, head_dim(model.config)
    heads = load_heads(a.heads_json)
    randoms = matched_randoms(heads, H, a.n_random, a.seed)
    layers = sorted({l for sel in (heads, *randoms) for l in sel})

    rng = random.Random(a.seed)
    idx = rng.sample(range(len(data.flores("en"))), a.n_fit)
    fit = {lang: [data.flores(lang)[i] for i in idx] for lang in (a.lang, "en")}
    stats = fit_pca({lang: collect_hidden_states(model, tok, fit[lang], {l + 1 for l in layers}) for lang in fit})
    mu = {lang: head_output_means(model, tok, fit[lang], layers) for lang in fit}

    def cos(l, h):
        W = model.model.layers[l].self_attn.o_proj.weight[:, h * DH:(h + 1) * DH].detach().double().cpu()
        d = W @ (mu[a.lang][l][h] - mu["en"][l][h]).double()
        return float(d @ stats[l + 1]["pc1"].double() / d.norm())

    selected = {(l, h): cos(l, h) for l, hs in heads.items() for h in hs}
    control = [cos(l, h) for r in randoms for l, hs in r.items() for h in hs]
    for (l, h), c in selected.items():
        print(f"L{l}H{h}  {c:+.3f}")
    print(f"mean |cos|: selected {sum(map(abs, selected.values())) / len(selected):.3f}, "
          f"random {sum(map(abs, control)) / len(control):.3f}")


if __name__ == "__main__":
    main()
