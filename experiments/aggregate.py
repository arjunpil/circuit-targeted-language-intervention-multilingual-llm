import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONTROL_PREFIXES = ("random", "nearby")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_samples(path):
    out = defaultdict(dict)
    for r in load_jsonl(path):
        out[(r["condition"], float(r["coef"]), r["split"])][r["ted_idx"]] = r
    return out


def boot(d, n_boot, seed):
    d = np.asarray(d, float)
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    m = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def vec(samples, key, idx, metric):
    return np.array([samples[key][i][metric] for i in idx], float)


def analyse_run(result_dir, metric, n_boot, seed=0):
    result_dir = Path(result_dir)
    samples = load_samples(result_dir / "samples.jsonl")
    rows = load_jsonl(result_dir / "rows.jsonl")
    base_key = ("none", 0.0, "cs")
    idx = sorted(samples[base_key])
    none_cs = vec(samples, base_key, idx, metric)
    none_en = vec(samples, ("none", 0.0, "en"), idx, metric)
    base_row = next(r for r in rows if r["condition"] == "none")
    ppl_keys = [k for k in base_row if k.startswith("ppl_")]
    entries = []
    keys = sorted(k for k in samples if k[2] == "cs" and k[0] != "none" and not k[0].startswith(CONTROL_PREFIXES))
    for cond, coef, _ in keys:
        cs = vec(samples, (cond, coef, "cs"), idx, metric)
        en = vec(samples, (cond, coef, "en"), idx, metric)
        row = next(r for r in rows if r["condition"] == cond and float(r["coef"]) == coef)
        entry = {
            "condition": cond,
            "coef": coef,
            "cs_mean": float(cs.mean()),
            "cs_delta": boot(cs - none_cs, n_boot, seed),
            "en_delta": boot(en - none_en, n_boot, seed),
            "ppl_ratio": {k: row[k] / base_row[k] for k in ppl_keys},
            "controls": {},
        }
        if cond == "heads":
            for kind in CONTROL_PREFIXES:
                ctrl = np.array([vec(samples, (c, k, "cs"), idx, metric) for c, k, s in samples
                                 if c.startswith(kind) and k == coef and s == "cs"])
                if len(ctrl):
                    ctrl_means = ctrl.mean(1)
                    entry["controls"][kind] = {
                        "n_sets": int(len(ctrl)),
                        "mean": float(ctrl_means.mean()),
                        "min": float(ctrl_means.min()),
                        "max": float(ctrl_means.max()),
                        "heads_minus_control_mean": boot(cs - ctrl.mean(0), n_boot, seed),
                        "p_one_sided": float((1 + (ctrl_means <= cs.mean()).sum()) / (1 + len(ctrl))),
                    }
        entries.append(entry)
    return {"n_prompts": len(idx), "none_cs": float(none_cs.mean()), "none_en": float(none_en.mean()),
            "none_ppl": {k: base_row[k] for k in ppl_keys}, "entries": entries}


def collect(root):
    runs, skipped = [], []
    for mpath in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(mpath.read_text())
        result_dir = mpath.parent / manifest.get("result_subdir", "")
        if manifest.get("status") != "done" or not (result_dir / "summary.json").exists():
            skipped.append((manifest.get("run_id", mpath.parent.name), manifest.get("status", "unknown")))
            continue
        runs.append((manifest, result_dir))
    return runs, skipped


def flat_run_row(manifest, analysis):
    p = manifest["params"]
    return {
        "run_id": manifest["run_id"], "model": p["model_name"], "lang": p["lang"], "seed": p.get("seed", 0),
        "head_source": manifest["head_source"], "n_prompts": analysis["n_prompts"],
        "none_cs": analysis["none_cs"], "none_en": analysis["none_en"],
        "git_commit": manifest.get("git_commit"), "code_hash": manifest.get("code_hash"), "seconds": manifest.get("seconds"),
        "torch": manifest.get("torch"), "transformers": manifest.get("transformers"),
    }


def flat_condition_rows(manifest, analysis):
    p = manifest["params"]
    rows = []
    for e in analysis["entries"]:
        row = {
            "run_id": manifest["run_id"], "model": p["model_name"], "lang": p["lang"], "seed": p.get("seed", 0),
            "head_source": manifest["head_source"], "condition": e["condition"], "coef": e["coef"],
            "cs_mean": e["cs_mean"], "cs_delta": e["cs_delta"]["mean"], "cs_delta_lo": e["cs_delta"]["lo"],
            "cs_delta_hi": e["cs_delta"]["hi"], "en_delta": e["en_delta"]["mean"],
            "en_delta_lo": e["en_delta"]["lo"], "en_delta_hi": e["en_delta"]["hi"],
        }
        for k, v in e["ppl_ratio"].items():
            row[k + "_ratio"] = v
        for kind in CONTROL_PREFIXES:
            c = e["controls"].get(kind)
            row[f"{kind}_n"] = c["n_sets"] if c else ""
            row[f"{kind}_mean"] = c["mean"] if c else ""
            row[f"{kind}_heads_minus_mean"] = c["heads_minus_control_mean"]["mean"] if c else ""
            row[f"{kind}_heads_minus_mean_lo"] = c["heads_minus_control_mean"]["lo"] if c else ""
            row[f"{kind}_heads_minus_mean_hi"] = c["heads_minus_control_mean"]["hi"] if c else ""
            row[f"{kind}_p"] = c["p_one_sided"] if c else ""
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(x, digits=3):
    return "" if x == "" or x is None else f"{x:+.{digits}f}"


def report(metric, run_rows, cond_rows, skipped):
    lines = [f"metric: {metric} (share of non-target-language tokens, lower is better)", ""]
    if any(r["head_source"] != "circuit" for r in run_rows):
        lines += ["Runs with head_source other than circuit use stand-in heads chosen with the same direction that is "
                  "steered, so their margin over the controls is partly built in and is not evidence for a circuit.", ""]
    hashes = {r.get("code_hash") for r in run_rows}
    if len(hashes) > 1:
        lines += ["These runs were produced by different code versions (see code_hash in runs.csv); rerun the older "
                  "ones before comparing them.", ""]
    lines += ["| run | heads | prompts | unsteered | coef | cs delta [95% CI] | en delta | ppl ratios | "
              "vs random mean (p) | vs nearby mean (p) |", "|---|---|---|---|---|---|---|---|---|---|"]
    by_run = {r["run_id"]: r for r in run_rows}
    for c in sorted(cond_rows, key=lambda r: (r["run_id"], r["condition"], r["coef"])):
        run = by_run[c["run_id"]]
        ppl = ", ".join(f"{k[4:-6]} x{c[k]:.2f}" for k in c if k.startswith("ppl_") and k.endswith("_ratio"))
        cells = [c["run_id"], c["head_source"], str(run["n_prompts"]), f"{run['none_cs']:.3f}",
                 f"{c['condition']} {c['coef']:g}",
                 f"{fmt(c['cs_delta'])} [{fmt(c['cs_delta_lo'])}, {fmt(c['cs_delta_hi'])}]", fmt(c["en_delta"]), ppl]
        for kind in CONTROL_PREFIXES:
            if c.get(f"{kind}_n") not in ("", None):
                cells.append(f"{fmt(c[f'{kind}_heads_minus_mean'])} (p={c[f'{kind}_p']:.3f})")
            else:
                cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
    if skipped:
        lines += ["", "Not aggregated: " + ", ".join(f"{r} ({s})" for r, s in skipped)]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--metric", default="csi_pair")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    runs, skipped = collect(a.root)
    out = Path(a.out or Path(a.root) / "aggregate")
    out.mkdir(parents=True, exist_ok=True)
    run_rows, cond_rows = [], []
    for manifest, result_dir in runs:
        analysis = analyse_run(result_dir, a.metric, a.n_boot)
        run_rows.append(flat_run_row(manifest, analysis))
        cond_rows += flat_condition_rows(manifest, analysis)
    if run_rows:
        write_csv(out / "runs.csv", run_rows)
        write_csv(out / "conditions.csv", cond_rows)
    text = report(a.metric, run_rows, cond_rows, skipped)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
