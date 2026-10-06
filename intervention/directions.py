import numpy as np
import torch
from sklearn.decomposition import PCA

from intervention.hooks import head_dim


def _batch(tok, texts, skip_first, device):
    batch = tok(texts, return_tensors="pt", padding=True, return_special_tokens_mask=True).to(device)
    keep = batch["attention_mask"].bool() & ~batch.pop("special_tokens_mask").bool()
    if skip_first:
        keep[torch.arange(keep.shape[0]), keep.float().argmax(1)] = False
    return batch, keep


@torch.no_grad()
def collect_hidden_states(model, tok, texts, indices, skip_first=True, batch_size=16):
    indices = sorted(set(indices))
    L = model.config.num_hidden_layers
    out = {i: [] for i in indices}
    last = {}
    handle = model.model.layers[-1].register_forward_hook(
        lambda m, i, o: last.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    tok.padding_side = "right"
    try:
        for s in range(0, len(texts), batch_size):
            batch, keep = _batch(tok, texts[s:s + batch_size], skip_first, model.device)
            hs = list(model.model(**batch, output_hidden_states=True).hidden_states)
            hs[L] = last["h"]
            for i in indices:
                out[i].append(hs[i][keep].float().cpu())
    finally:
        handle.remove()
    return {i: torch.cat(v) for i, v in out.items()}


@torch.no_grad()
def head_output_means(model, tok, texts, layers, skip_first=True, batch_size=16):
    H, DH = model.config.num_attention_heads, head_dim(model.config)
    store, sums, n = {}, {l: torch.zeros(H, DH, dtype=torch.float64) for l in layers}, 0
    handles = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
        lambda m, i, l=l: store.__setitem__(l, i[0])) for l in layers]
    tok.padding_side = "right"
    try:
        for s in range(0, len(texts), batch_size):
            batch, keep = _batch(tok, texts[s:s + batch_size], skip_first, model.device)
            model.model(**batch)
            for l in layers:
                sums[l] += store[l].view(*keep.shape, H, DH)[keep].double().sum(0).cpu()
            n += int(keep.sum())
    finally:
        for h in handles:
            h.remove()
    return {l: (v / n).float() for l, v in sums.items()}


def fit_pca(hs_by_lang, n_components=10):
    langs = list(hs_by_lang)
    stats = {}
    for i in hs_by_lang[langs[0]]:
        pca = PCA(n_components=n_components).fit(torch.cat([hs_by_lang[lang][i] for lang in langs]).numpy())
        mean = torch.tensor(pca.mean_, dtype=torch.float32)
        pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
        stats[i] = {
            "mean": mean,
            "pc1": pc1,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "lang_pc1_mean": {lang: float(((hs_by_lang[lang][i] - mean) @ pc1).mean()) for lang in langs},
        }
    return stats


def pc1_accuracy(stats, hs_by_lang, src, tgt):
    acc = {}
    for i, st in stats.items():
        ps = ((hs_by_lang[src][i] - st["mean"]) @ st["pc1"]).numpy()
        pt = ((hs_by_lang[tgt][i] - st["mean"]) @ st["pc1"]).numpy()
        mid, sign = (ps.mean() + pt.mean()) / 2, np.sign(ps.mean() - pt.mean())
        acc[i] = float((np.sum(sign * (ps - mid) > 0) + np.sum(sign * (pt - mid) < 0)) / (len(ps) + len(pt)))
    return acc
