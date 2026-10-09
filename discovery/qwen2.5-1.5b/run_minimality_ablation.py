import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import (
    ROOT,
    build_position_matched_examples,
    default_metric_json,
    load_metric_tokens,
    load_model_and_tokenizer,
    model_inputs,
)
from run_head_validation import (
    bootstrap_ci,
    forward_capture_final_heads,
    patched_metrics_batch,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Minimality ablation: compare the full frozen head set's "
            "sufficiency/necessity effect against subsets of it on the "
            "same held-out examples."
        )
    )

    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--lang", required=True)
    p.add_argument("--heads-json", default=None)
    p.add_argument("--metric-json", default=None)
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--n-examples", type=int, default=50)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--min-len", type=int, default=6)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reference-heads-json",
        default=None,
        help=(
            "Another language's frozen heads.json to split this "
            "language's set into an overlap subset and an extra subset."
        ),
    )
    p.add_argument("--out", required=True)

    return p.parse_args()


def load_heads(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        int(layer): sorted(int(h) for h in heads)
        for layer, heads in data["heads"].items()
    }


def flatten(heads):
    return {(layer, h) for layer, hs in heads.items() for h in hs}


def unflatten(pairs):
    heads = {}

    for layer, h in sorted(pairs):
        heads.setdefault(layer, []).append(h)

    return heads


def evaluate_head_set(
    model,
    device,
    examples,
    english_ids,
    target_ids,
    heads,
    bootstrap_samples,
    seed,
):
    layers = sorted(heads)

    sufficiency = []
    necessity = []

    for example in examples:
        en_inputs = model_inputs(example["en_ids"], device)
        target_inputs = model_inputs(example["target_ids"], device)

        base_en, en_z = forward_capture_final_heads(
            model, en_inputs, layers, english_ids, target_ids
        )
        base_target, target_z = forward_capture_final_heads(
            model, target_inputs, layers, english_ids, target_ids
        )

        patched = patched_metrics_batch(
            model=model,
            target_inputs=en_inputs,
            source_z=target_z,
            head_sets=[heads],
            english_ids=english_ids,
            target_ids=target_ids,
        )[0]
        sufficiency.append(float(patched - base_en))

        patched_reverse = patched_metrics_batch(
            model=model,
            target_inputs=target_inputs,
            source_z=en_z,
            head_sets=[heads],
            english_ids=english_ids,
            target_ids=target_ids,
        )[0]
        necessity.append(float(base_target - patched_reverse))

    suff = np.asarray(sufficiency, dtype=np.float64)
    nec = np.asarray(necessity, dtype=np.float64)

    return {
        "n_heads": sum(len(h) for h in heads.values()),
        "heads": unflatten(flatten(heads)),
        "sufficiency_mean": float(suff.mean()),
        "sufficiency_ci": bootstrap_ci(suff, bootstrap_samples, seed + 1),
        "necessity_mean": float(nec.mean()),
        "necessity_ci": bootstrap_ci(nec, bootstrap_samples, seed + 2),
        "per_example_sufficiency": sufficiency,
        "per_example_necessity": necessity,
    }


def main():
    args = parse_args()

    heads_json = (
        ROOT / args.heads_json
        if args.heads_json is not None
        else ROOT / f"discovery/qwen2.5-1.5b/results/qwen25_en_{args.lang}_heads.json"
    )
    metric_json = (
        ROOT / args.metric_json
        if args.metric_json is not None
        else default_metric_json(args.lang)
    )

    full_heads = load_heads(heads_json)

    subsets = {"full": full_heads}

    if args.reference_heads_json is not None:
        reference_heads = load_heads(ROOT / args.reference_heads_json)
        full_flat = flatten(full_heads)
        ref_flat = flatten(reference_heads)

        overlap = full_flat & ref_flat
        extra = full_flat - ref_flat

        if overlap:
            subsets["overlap"] = unflatten(overlap)
        if extra:
            subsets["extra"] = unflatten(extra)

    metric_data = load_metric_tokens(metric_json, lang=args.lang)
    english_ids = metric_data["english_token_ids"]
    target_ids = metric_data["target_token_ids"]

    model, tok = load_model_and_tokenizer(args.model)
    device = next(model.parameters()).device

    examples = build_position_matched_examples(
        tok,
        start=args.start,
        n_examples=args.n_examples,
        max_len=args.max_len,
        min_len=args.min_len,
        lang=args.lang,
    )

    print()
    print("=" * 72)
    print("MINIMALITY ABLATION")
    print("=" * 72)
    print("model:", args.model, "lang:", args.lang)
    print("subsets:", {k: sum(len(h) for h in v.values()) for k, v in subsets.items()})
    print("holdout:", args.start, "-", args.start + len(examples) - 1)
    print()

    results = {}

    for name, heads in subsets.items():
        print(f"[{name}] {sum(len(h) for h in heads.values())} heads")

        results[name] = evaluate_head_set(
            model=model,
            device=device,
            examples=examples,
            english_ids=english_ids,
            target_ids=target_ids,
            heads=heads,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )

        print(
            f"  sufficiency mean={results[name]['sufficiency_mean']:+.4f} "
            f"ci={results[name]['sufficiency_ci']}"
        )
        print(
            f"  necessity   mean={results[name]['necessity_mean']:+.4f} "
            f"ci={results[name]['necessity_ci']}"
        )

    if "overlap" in results and "full" in results:
        for metric in ("sufficiency_mean", "necessity_mean"):
            full_val = results["full"][metric]
            overlap_val = results["overlap"][metric]

            retained = (
                overlap_val / full_val if full_val != 0 else float("nan")
            )

            print(
                f"overlap retains {retained * 100:.1f}% of full-set "
                f"{metric}"
            )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "language_pair": f"en->{args.lang}",
                "heads_json": str(heads_json.relative_to(ROOT)),
                "reference_heads_json": args.reference_heads_json,
                "holdout_indices": [
                    examples[0]["index"],
                    examples[-1]["index"],
                ],
                "n_holdout_pairs": len(examples),
                "subsets": results,
            },
            f,
            indent=2,
        )

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
