# Sliding-window token LID adapted from github.com/fxlrnrpt/language-steering-in-latent-space (MIT)
import sys
from functools import lru_cache

import numpy as np

import fasttext.FastText as _ft

_ft.np = type(sys)("_np_compat")
_ft.np.__dict__.update(np.__dict__)
_ft.np.array = lambda *a, **kw: np.asarray(*a, **{k: v for k, v in kw.items() if k != "copy"})

import fasttext  # noqa: E402


def load_lid(path):
    fasttext.FastText.eprint = lambda *a, **k: None
    return fasttext.load_model(str(path))


@lru_cache(maxsize=200_000)
def _classify(model, text, allowed):
    if allowed is None:
        return model.predict(text, k=1)[0][0][9:]
    labels, probs = model.predict(text, k=-1)
    p = {l[9:]: q for l, q in zip(labels, probs)}
    return max(allowed, key=lambda l: p.get(l, 0.0))


def classify(text, model, allowed=None):
    text = text.replace("\n", " ").strip()
    if not text:
        return "unk"
    return _classify(model, text, tuple(sorted(allowed)) if allowed else None)


def classify_tokens(tokens, model, window=5, allowed=None):
    half = window // 2
    raw = []
    for i in range(len(tokens)):
        w = "".join(tokens[max(0, i - half):i + half + 1]).strip()
        raw.append(classify(w, model, allowed) if any(c.isalpha() for c in w) else None)
    labels = list(raw)
    for i, lab in enumerate(labels):
        if lab is None:
            left = next((labels[j] for j in range(i - 1, -1, -1) if labels[j] is not None), None)
            right = next((raw[j] for j in range(i + 1, len(raw)) if raw[j] is not None), None)
            labels[i] = left or right or "unk"
    return labels


def _cjk(text):
    return any("぀" <= c <= "ヿ" or "㐀" <= c <= "䶿" or "一" <= c <= "鿿" for c in text)


def token_labels(token_ids, tok, model, window=5, allowed=None, by_word=False):
    tokens = [tok.decode([t]) for t in token_ids]
    if not by_word:
        return classify_tokens(tokens, model, window, allowed)
    groups, words, owner = [], [], []
    for t, s in zip(token_ids, tokens):
        prev = words[-1] if words else ""
        whole = "�" not in prev
        if not groups or s[:1].isspace() or (whole and (_cjk(prev) or not any(c.isalnum() for c in prev))):
            groups.append([t])
            words.append(s)
        else:
            groups[-1].append(t)
            words[-1] = tok.decode(groups[-1])
        owner.append(len(groups) - 1)
    labels = classify_tokens(words, model, window, allowed)
    return [labels[w] for w in owner]
