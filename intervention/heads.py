import json
import random

import torch

from intervention.hooks import head_dim


@torch.no_grad()
def rank_heads(model, tok, texts_by_lang, dirs, src, tgt, layers=None, batch_size=16):
    H, DH = model.config.num_attention_heads, head_dim(model.config)
    layers = list(layers if layers is not None else range(model.config.num_hidden_layers))
    o_projs = {l: model.model.layers[l].self_attn.o_proj for l in layers}
    u = {l: o.weight.view(o.out_features, H, DH).permute(1, 2, 0) @ dirs[l].to(o.weight) for l, o in o_projs.items()}
    store = {}
    handles = [o.register_forward_pre_hook(lambda m, i, l=l: store.__setitem__(l, i[0])) for l, o in o_projs.items()]
    proj = {lang: {l: [] for l in layers} for lang in (src, tgt)}
    tok.padding_side = "right"
    try:
        for lang in (src, tgt):
            texts = texts_by_lang[lang]
            for s in range(0, len(texts), batch_size):
                batch = tok(texts[s:s + batch_size], return_tensors="pt", padding=True).to(model.device)
                model.model(**batch)
                keep = batch["attention_mask"].bool()
                for l in layers:
                    z = store[l].view(*store[l].shape[:2], H, DH)
                    proj[lang][l].append((z * u[l]).sum(-1)[keep].float().cpu())
    finally:
        for h in handles:
            h.remove()

    scores = []
    for l in layers:
        a, b = torch.cat(proj[src][l]), torch.cat(proj[tgt][l])
        d = (a.mean(0) - b.mean(0)) / (((a.var(0) + b.var(0)) / 2).sqrt() + 1e-8)
        scores += [(l, h, float(d[h])) for h in range(H)]
    return sorted(scores, key=lambda x: -abs(x[2]))


def top_heads(scores, k):
    out = {}
    for l, h, _ in scores[:k]:
        out.setdefault(l, []).append(h)
    return {l: sorted(hs) for l, hs in sorted(out.items())}


def matched_random(heads, n_heads, seed):
    rng = random.Random(seed)
    out = {}
    for l, hs in heads.items():
        pool = [h for h in range(n_heads) if h not in hs]
        if len(pool) < len(hs):
            raise ValueError(f"layer {l}: not enough unselected heads")
        out[l] = sorted(rng.sample(pool, len(hs)))
    return out


def matched_nearby(heads, n_heads, n_layers, seed):
    rng = random.Random(seed)
    out = {}
    for l, hs in heads.items():
        m = rng.choice([m for m in (l - 1, l + 1) if 0 <= m < n_layers])
        pool = [h for h in range(n_heads) if h not in heads.get(m, []) and h not in out.get(m, [])]
        out.setdefault(m, []).extend(rng.sample(pool, len(hs)))
    return {l: sorted(hs) for l, hs in sorted(out.items())}


def save_heads(heads, path, **meta):
    with open(path, "w") as f:
        json.dump({"heads": {str(l): hs for l, hs in heads.items()}, **meta}, f, indent=2)


def load_heads(path):
    with open(path) as f:
        return {int(l): hs for l, hs in json.load(f)["heads"].items()}
