# LPR and WPR follow compute_metrics.py from github.com/for-ai/language-confusion (Apache-2.0)
import argparse
import json
import random
import string
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervention import data
from intervention.directions import collect_hidden_states, fit_pca, head_output_means
from intervention.generation import generate, perplexity
from intervention.heads import load_heads, matched_nearby, matched_randoms
from intervention.hooks import HeadDirSteer, HeadSteer, ResidualSteer, applied
from intervention.lid import load_lid
from intervention.metrics import repeated_ngram_rate
from intervention.reference import last_two

TASKS = ("monolingual", "crosslingual")
NON_LATIN = {"ar", "hi", "ja", "ko", "ru", "zh"}
_PUNCT = str.maketrans("", "", string.punctuation)


def score(text, lang, lid, en_words):
    text = text.split("\nQ:")[0].strip().translate(_PUNCT).replace("—", " ").replace("،", "")
    lines = [line for line in text.split("\n") if len(line.split()) >= 5]
    if not lines:
        return {"skipped": True, "labels": []}
    labels = []
    for line in lines:
        (label,), (p,) = lid.predict(line)
        labels.append(label[9:] if p > 0.3 else "unknown")
    passed = all(l == lang for l in labels)
    return {
        "skipped": False,
        "labels": labels,
        "line_pass": passed,
        "word_error": passed and lang in NON_LATIN and any(t in en_words for line in lines for t in line.split()),
        "acc": sum(l == lang for l in labels) / len(labels),
        "en": sum(l == "en" for l in labels) / len(labels),
    }


def chat(tok, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, date_string="26 Jul 2024")
    return text.removeprefix(tok.bos_token or "")


def aggregate(scores, lang):
    kept = [s for s in scores if not s["skipped"]]
    m = {
        "rep4": sum(s["rep4"] for s in scores) / len(scores),
        "lpr": 1 - sum(not s["line_pass"] for s in kept) / max(1, len(kept)),
        "acc": sum(s["acc"] for s in kept) / len(kept) if kept else 1.0,
        "en": sum(s["en"] for s in kept) / len(kept) if kept else 0.0,
        "skipped": len(scores) - len(kept),
    }
    if lang in NON_LATIN:
        passed = [s for s in kept if s["line_pass"]]
        m["wpr"] = 1 - sum(s["word_error"] for s in passed) / max(1, len(passed))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--lang", default="es", choices=["es", "ru", "hi", "ko"])
    p.add_argument("--heads-json", required=True)
    p.add_argument("--head-coefs", default="-3,-1,1,3")
    p.add_argument("--head-dir", default="pc1", choices=["pc1", "own", "pull"], help="pull: toward --lang")
    p.add_argument("--resid-coefs", default="", help="baseline from the steering paper, their layer indexing")
    p.add_argument("--steer-layers", default=None, help="default: last two layers")
    p.add_argument("--n-random", type=int, default=6)
    p.add_argument("--n-nearby", type=int, default=3)
    p.add_argument("--n-fit", type=int, default=200)
    p.add_argument("--n-per-source", type=int, default=100)
    p.add_argument("--n-ppl", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/lcb")
    a = p.parse_args()
    t0 = time.time()
    out = Path(a.out) / f"{a.model.split('/')[-1]}_{a.lang}"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    lid, en_words = load_lid(data.lid_path()), data.lcb_en_words()

    heads = load_heads(a.heads_json)
    randoms = matched_randoms(heads, H, a.n_random, a.seed)
    nearby = [matched_nearby(heads, H, L, seed=a.seed + 1000 + i) for i in range(a.n_nearby)]
    layers = {l for sel in (heads, *randoms, *nearby) for l in sel}
    head_coefs = [float(x) for x in a.head_coefs.split(",") if x]
    resid_coefs = [float(x) for x in a.resid_coefs.split(",") if x]
    steer_layers = [int(x) for x in a.steer_layers.split(",")] if a.steer_layers else last_two(model)

    rng = random.Random(a.seed)
    fit_idx = rng.sample(range(len(data.flores("en"))), a.n_fit)
    fit = {lang: [data.flores(lang)[i] for i in fit_idx] for lang in (a.lang, "en")}
    indices = {l + 1 for l in layers} | (set(steer_layers) if resid_coefs else set())
    hs = {lang: collect_hidden_states(model, tok, fit[lang], indices, batch_size=a.batch_size) for lang in fit}
    stats = fit_pca(hs)
    del hs
    dirs = {l: stats[l + 1]["pc1"] for l in layers}
    ref_means = {l: stats[l]["mean"] for l in steer_layers if l in stats}
    ref_dirs = {l: stats[l]["pc1"] for l in steer_layers if l in stats}
    if a.head_dir != "pc1":
        mu = {lang: head_output_means(model, tok, fit[lang], sorted(layers), batch_size=a.batch_size) for lang in fit}
    ppl_idx = rng.sample(range(len(data.flores("en", "devtest"))), a.n_ppl)
    ppl_sets = {lang: [data.flores(lang, "devtest")[i] for i in ppl_idx] for lang in ("en", a.lang)}

    items = []
    for task in TASKS:
        seen = {}
        for source, prompt in data.lcb(a.lang, task):
            seen[source] = seen.get(source, 0) + 1
            if seen[source] <= a.n_per_source:
                items.append({"task": task, "source": source, "idx": seen[source] - 1, "prompt": prompt})
    chats = [chat(tok, it["prompt"]) for it in items]
    order = sorted(range(len(chats)), key=lambda i: len(chats[i]))

    def run(ivs):
        with applied(model, *ivs):
            ids = generate(model, tok, [chats[i] for i in order], a.max_new_tokens, a.batch_size)
        texts, reps = [None] * len(chats), [None] * len(chats)
        for i, g in zip(order, ids):
            texts[i], reps[i] = tok.decode(g, skip_special_tokens=True), repeated_ngram_rate(g)
        return texts, reps

    def head_iv(sel, c):
        if a.head_dir == "pc1":
            return HeadSteer(sel, c, dirs)
        return HeadDirSteer(sel, c, mu[a.lang], mu["en"], pull=a.head_dir == "pull")

    conditions = [("none", 0.0, [])]
    conditions += [("heads", c, [head_iv(heads, c)]) for c in head_coefs]
    conditions += [("residual", c, [ResidualSteer(steer_layers, c, ref_means, ref_dirs)]) for c in resid_coefs]
    for kind, sets in (("random", randoms), ("nearby", nearby)):
        conditions += [(f"{kind}{i}", c, [head_iv(sel, c)]) for i, sel in enumerate(sets) for c in head_coefs]

    rows = []
    open(out / "samples.jsonl", "w").close()
    for name, coef, ivs in conditions:
        texts, reps = run(ivs)
        scores = [{**score(t, a.lang, lid, en_words), "rep4": r} for t, r in zip(texts, reps)]
        row = {"condition": name, "coef": coef}
        for task in TASKS:
            per_source = {}
            for it, s in zip(items, scores):
                if it["task"] == task:
                    per_source.setdefault(it["source"], []).append(s)
            ms = [aggregate(v, a.lang) for v in per_source.values()]
            for k in ms[0]:
                row[f"{task}_{k}"] = sum(m[k] for m in ms) / len(ms)
        with applied(model, *ivs):
            for lang, sents in ppl_sets.items():
                row[f"ppl_{lang}"] = perplexity(model, tok, sents, a.batch_size)
        rows.append(row)
        with open(out / "rows.jsonl", "a" if len(rows) > 1 else "w") as f:
            f.write(json.dumps(row) + "\n")
        with open(out / "samples.jsonl", "a") as f:
            f.writelines(json.dumps({"condition": name, "coef": coef, **{k: it[k] for k in ("task", "source", "idx")},
                                     "text": t, **s}, ensure_ascii=False) + "\n" for it, t, s in zip(items, texts, scores))
        print(f"{name:9s} c={coef:5.1f}  mono lpr {row['monolingual_lpr']:.3f} en {row['monolingual_en']:.3f}  "
              f"cross lpr {row['crosslingual_lpr']:.3f} en {row['crosslingual_en']:.3f}  "
              f"rep4 {row['monolingual_rep4']:.3f}/{row['crosslingual_rep4']:.3f}  "
              f"ppl_en {row['ppl_en']:.2f}  ppl_{a.lang} {row['ppl_' + a.lang]:.2f}", flush=True)

    base = rows[0]
    controls = {}
    for coef in head_coefs:
        for task in TASKS:
            key = f"{task}_lpr"
            target = next(r[key] for r in rows if r["condition"] == "heads" and r["coef"] == coef) - base[key]
            deltas = [r[key] - base[key] for r in rows if r["condition"][:6] in ("random", "nearby") and r["coef"] == coef]
            if deltas:
                controls.setdefault(str(coef), {})[task] = {
                    "heads_delta": target, "control_mean": sum(deltas) / len(deltas),
                    "control_min": min(deltas), "control_max": max(deltas),
                    "share_as_large": sum(abs(d) >= abs(target) for d in deltas) / len(deltas), "n": len(deltas)}

    summary = {
        "args": vars(a),
        "steer_layers": steer_layers,
        "heads": {str(l): v for l, v in heads.items()},
        "random_heads": [{str(l): v for l, v in r.items()} for r in randoms],
        "nearby_heads": [{str(l): v for l, v in r.items()} for r in nearby],
        "n_prompts": {task: sum(it["task"] == task for it in items) for task in TASKS},
        "controls": controls,
        "rows": rows,
        "minutes": (time.time() - t0) / 60,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved to {out} ({summary['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
