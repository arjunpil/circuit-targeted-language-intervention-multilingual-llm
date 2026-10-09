import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import (
    ROOT,
    build_position_matched_examples,
    default_heads_json,
    default_metric_json,
    head_dim,
    language_metric,
    load_metric_tokens,
    load_model_and_tokenizer,
    model_inputs,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Validate a frozen Llama-3.2-1B EN<->target head set "
            "against layer-matched random controls."
        )
    )

    p.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
    )

    p.add_argument(
        "--lang",
        default="es",
        help="Target language code (es, ru, zh, hi, ko, ...).",
    )

    p.add_argument(
        "--heads-json",
        default=None,
        help=(
            "Defaults to "
            "discovery/llama-3.2-1b/results/llama32_en_<lang>_heads.json."
        ),
    )

    p.add_argument(
        "--metric-json",
        default=None,
        help=(
            "Defaults to "
            "discovery/llama-3.2-1b/results/en_<lang>_language_metric.json."
        ),
    )

    p.add_argument(
        "--start",
        type=int,
        default=100,
    )

    p.add_argument(
        "--n-examples",
        type=int,
        default=50,
    )

    p.add_argument(
        "--n-random-controls",
        type=int,
        default=100,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--max-len",
        type=int,
        default=32,
    )

    p.add_argument(
        "--min-len",
        type=int,
        default=6,
    )

    p.add_argument(
        "--control-batch-size",
        type=int,
        default=20,
    )

    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
    )

    p.add_argument(
        "--out",
        default=None,
        help=(
            "Defaults to "
            "results/llama32_head_validation/head_validation_<lang>.json "
            "(head_validation.json for --lang es)."
        ),
    )

    return p.parse_args()


def default_out_path(lang):
    base = ROOT / "results/llama32_head_validation"

    if lang == "es":
        return base / "head_validation.json"

    return base / f"head_validation_{lang}.json"


def load_frozen_heads(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    heads = {
        int(layer): sorted(
            int(head)
            for head in values
        )
        for layer, values
        in data["heads"].items()
    }

    return data, heads


def canonical_head_set(heads):
    return tuple(
        (layer, head)
        for layer in sorted(heads)
        for head in sorted(heads[layer])
    )


def max_unique_controls(selected, n_heads):
    total = 1

    for chosen in selected.values():
        available = n_heads - len(chosen)
        total *= math.comb(available, len(chosen))

    return total


def make_random_controls(
    selected,
    n_heads,
    n_controls,
    seed,
):
    rng = np.random.default_rng(seed)

    feasible = max_unique_controls(selected, n_heads)

    if n_controls > feasible:
        print(
            f"Warning: only {feasible} unique matched control set(s) "
            f"exist for this head set; requested {n_controls}. "
            f"Using all {feasible}."
        )
        n_controls = feasible

    controls = []
    seen = set()

    attempts = 0
    max_attempts = max(n_controls, 1) * 1000

    while len(controls) < n_controls:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Could not generate enough unique controls."
            )

        control = {}

        for layer, chosen in selected.items():
            available = [
                h
                for h in range(n_heads)
                if h not in chosen
            ]

            sampled = rng.choice(
                available,
                size=len(chosen),
                replace=False,
            )

            control[layer] = sorted(
                int(h)
                for h in sampled
            )

        key = canonical_head_set(control)

        if key in seen:
            continue

        seen.add(key)
        controls.append(control)

    return controls


@torch.no_grad()
def forward_capture_final_heads(
    model,
    inputs,
    layers,
    english_ids,
    target_ids,
):
    store = {}
    handles = []

    for layer in layers:
        o_proj = (
            model.model.layers[layer]
            .self_attn
            .o_proj
        )

        def hook(
            module,
            hook_inputs,
            layer=layer,
        ):
            store[layer] = (
                hook_inputs[0][0, -1, :]
                .detach()
                .clone()
            )

        handles.append(
            o_proj.register_forward_pre_hook(
                hook
            )
        )

    try:
        outputs = model(
            **inputs,
            use_cache=False,
        )

        metric = language_metric(
            outputs.logits[:, -1, :],
            english_ids,
            target_ids,
        )[0]

    finally:
        for handle in handles:
            handle.remove()

    return float(metric), store


@torch.no_grad()
def patched_metrics_batch(
    model,
    target_inputs,
    source_z,
    head_sets,
    english_ids,
    target_ids,
):
    batch_size = len(head_sets)

    if batch_size == 0:
        return []

    device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.repeat(
            batch_size,
            *([1] * (value.ndim - 1)),
        ).to(device)
        for key, value in target_inputs.items()
    }

    dh = head_dim(model)

    layers = sorted(
        {
            layer
            for head_set in head_sets
            for layer in head_set
        }
    )

    handles = []

    for layer in layers:
        o_proj = (
            model.model.layers[layer]
            .self_attn
            .o_proj
        )

        source = source_z[layer]

        def hook(
            module,
            hook_inputs,
            layer=layer,
            source=source,
        ):
            z = hook_inputs[0].clone()

            source_vector = source.to(
                device=z.device,
                dtype=z.dtype,
            )

            for row, head_set in enumerate(
                head_sets
            ):
                for head in head_set.get(
                    layer,
                    [],
                ):
                    start = head * dh
                    end = start + dh

                    z[
                        row,
                        -1,
                        start:end,
                    ] = source_vector[
                        start:end
                    ]

            return (z,) + tuple(
                hook_inputs[1:]
            )

        handles.append(
            o_proj.register_forward_pre_hook(
                hook
            )
        )

    try:
        outputs = model(
            **inputs,
            use_cache=False,
        )

        metric = language_metric(
            outputs.logits[:, -1, :],
            english_ids,
            target_ids,
        )

        values = (
            metric.detach()
            .float()
            .cpu()
            .tolist()
        )

    finally:
        for handle in handles:
            handle.remove()

    return [
        float(value)
        for value in values
    ]


def bootstrap_ci(
    values,
    n_samples,
    seed,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)

    means = np.empty(
        n_samples,
        dtype=np.float64,
    )

    for i in range(n_samples):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        means[i] = sample.mean()

    low, high = np.percentile(
        means,
        [2.5, 97.5],
    )

    return [
        float(low),
        float(high),
    ]


def summarize(
    selected_values,
    control_means,
    bootstrap_samples,
    seed,
):
    selected = np.asarray(
        selected_values,
        dtype=np.float64,
    )

    controls = np.asarray(
        control_means,
        dtype=np.float64,
    )

    selected_mean = float(
        selected.mean()
    )

    ci = bootstrap_ci(
        selected,
        bootstrap_samples,
        seed,
    )

    p95 = float(
        np.percentile(
            controls,
            95,
        )
    )

    percentile = float(
        100.0
        * np.mean(
            controls <= selected_mean
        )
    )

    return {
        "selected_mean":
            selected_mean,
        "bootstrap_95_ci":
            ci,
        "random_95th_percentile":
            p95,
        "percentile_vs_random":
            percentile,
        "pass":
            bool(
                ci[0] > 0
                and selected_mean > p95
            ),
    }


def json_heads(heads):
    return {
        str(layer): heads[layer]
        for layer in sorted(heads)
    }


def main():
    args = parse_args()

    heads_json = (
        ROOT / args.heads_json
        if args.heads_json is not None
        else default_heads_json(args.lang, "llama32")
    )

    metric_json = (
        ROOT / args.metric_json
        if args.metric_json is not None
        else default_metric_json(args.lang)
    )

    frozen_data, selected = (
        load_frozen_heads(heads_json)
    )

    metric_data = load_metric_tokens(
        metric_json,
        lang=args.lang,
    )

    english_ids = metric_data[
        "english_token_ids"
    ]

    target_ids = metric_data[
        "target_token_ids"
    ]

    model, tok = load_model_and_tokenizer(
        args.model
    )

    device = next(
        model.parameters()
    ).device

    n_heads = (
        model.config.num_attention_heads
    )

    selected_count = sum(
        len(heads)
        for heads in selected.values()
    )

    print()
    print("=" * 72)
    print("FROZEN HEAD-SET VALIDATION")
    print("=" * 72)
    print("model:", args.model)
    print("language pair: en ->", args.lang)
    print(
        "selected heads:",
        selected_count,
    )
    print(
        "holdout requested:",
        args.start,
        "-",
        args.start + args.n_examples - 1,
    )
    print(
        "random controls:",
        args.n_random_controls,
    )
    print("seed:", args.seed)
    print()

    controls = make_random_controls(
        selected=selected,
        n_heads=n_heads,
        n_controls=args.n_random_controls,
        seed=args.seed,
    )

    examples = (
        build_position_matched_examples(
            tok,
            start=args.start,
            n_examples=args.n_examples,
            max_len=args.max_len,
            min_len=args.min_len,
            lang=args.lang,
        )
    )

    if len(examples) != args.n_examples:
        raise RuntimeError(
            "Expected exactly "
            f"{args.n_examples} usable examples, "
            f"found {len(examples)}."
        )

    layers = sorted(selected)

    selected_sufficiency = []
    selected_necessity = []

    random_sufficiency = [
        []
        for _ in controls
    ]

    random_necessity = [
        []
        for _ in controls
    ]

    example_rows = []

    for number, example in enumerate(
        examples,
        start=1,
    ):
        print(
            f"[{number:02d}/{len(examples)}] "
            f"FLORES {example['index']}"
        )

        en_inputs = model_inputs(
            example["en_ids"],
            device,
        )

        target_inputs = model_inputs(
            example["target_ids"],
            device,
        )

        base_en, en_z = (
            forward_capture_final_heads(
                model,
                en_inputs,
                layers,
                english_ids,
                target_ids,
            )
        )

        base_target, target_z = (
            forward_capture_final_heads(
                model,
                target_inputs,
                layers,
                english_ids,
                target_ids,
            )
        )

        patched = patched_metrics_batch(
            model=model,
            target_inputs=en_inputs,
            source_z=target_z,
            head_sets=[selected],
            english_ids=english_ids,
            target_ids=target_ids,
        )[0]

        suff = float(
            patched - base_en
        )

        selected_sufficiency.append(
            suff
        )

        patched_reverse = (
            patched_metrics_batch(
                model=model,
                target_inputs=target_inputs,
                source_z=en_z,
                head_sets=[selected],
                english_ids=english_ids,
                target_ids=target_ids,
            )[0]
        )

        necessity = float(
            base_target - patched_reverse
        )

        selected_necessity.append(
            necessity
        )

        for batch_start in range(
            0,
            len(controls),
            args.control_batch_size,
        ):
            batch_stop = min(
                batch_start
                + args.control_batch_size,
                len(controls),
            )

            batch = controls[
                batch_start:batch_stop
            ]

            suff_values = (
                patched_metrics_batch(
                    model=model,
                    target_inputs=en_inputs,
                    source_z=target_z,
                    head_sets=batch,
                    english_ids=english_ids,
                    target_ids=target_ids,
                )
            )

            nec_values = (
                patched_metrics_batch(
                    model=model,
                    target_inputs=target_inputs,
                    source_z=en_z,
                    head_sets=batch,
                    english_ids=english_ids,
                    target_ids=target_ids,
                )
            )

            for local_i, value in enumerate(
                suff_values
            ):
                control_i = (
                    batch_start + local_i
                )

                random_sufficiency[
                    control_i
                ].append(
                    float(value - base_en)
                )

            for local_i, value in enumerate(
                nec_values
            ):
                control_i = (
                    batch_start + local_i
                )

                random_necessity[
                    control_i
                ].append(
                    float(base_target - value)
                )

        example_rows.append(
            {
                "flores_index":
                    example["index"],
                "length":
                    example["length"],
                "baseline_en_metric":
                    base_en,
                "baseline_target_metric":
                    base_target,
                "sufficiency_effect":
                    suff,
                "necessity_effect":
                    necessity,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    suff_control_means = [
        float(
            np.mean(values)
        )
        for values in random_sufficiency
    ]

    nec_control_means = [
        float(
            np.mean(values)
        )
        for values in random_necessity
    ]

    suff_summary = summarize(
        selected_sufficiency,
        suff_control_means,
        args.bootstrap_samples,
        args.seed + 1,
    )

    nec_summary = summarize(
        selected_necessity,
        nec_control_means,
        args.bootstrap_samples,
        args.seed + 2,
    )

    suff_summary[
        "per_example"
    ] = selected_sufficiency

    suff_summary[
        "random_control_means"
    ] = suff_control_means

    nec_summary[
        "per_example"
    ] = selected_necessity

    nec_summary[
        "random_control_means"
    ] = nec_control_means

    result = {
        "model":
            args.model,
        "frozen_head_source_model":
            frozen_data.get("model"),
        "input_protocol": (
            "raw FLORES-200 devtest prefixes; "
            "no chat template"
        ),
        "language_pair":
            f"en->{args.lang}",
        "activation_site":
            "attention head output before o_proj",
        "selected_heads":
            json_heads(selected),
        "discovery_indices":
            frozen_data.get(
                "discovery_indices"
            ),
        "holdout_indices": [
            examples[0]["index"],
            examples[-1]["index"],
        ],
        "n_holdout_pairs":
            len(examples),
        "n_random_controls":
            len(controls),
        "control_matching": (
            "same selected layers and same "
            "number of heads per layer; "
            "selected heads excluded"
        ),
        "random_seed":
            args.seed,
        "control_batch_size":
            args.control_batch_size,
        "bootstrap_samples":
            args.bootstrap_samples,
        "sufficiency":
            suff_summary,
        "necessity":
            nec_summary,
        "random_control_sets": [
            json_heads(control)
            for control in controls
        ],
        "random_sufficiency_per_example":
            random_sufficiency,
        "random_necessity_per_example":
            random_necessity,
        "example_rows":
            example_rows,
    }

    out = (
        ROOT / args.out
        if args.out is not None
        else default_out_path(args.lang)
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)

    print()
    print("Sufficiency")
    print(
        "  selected mean:",
        f"{suff_summary['selected_mean']:+.6f}",
    )
    print(
        "  bootstrap 95% CI:",
        suff_summary["bootstrap_95_ci"],
    )
    print(
        "  random p95:",
        f"{suff_summary['random_95th_percentile']:+.6f}",
    )
    print(
        "  percentile:",
        suff_summary["percentile_vs_random"],
    )
    print(
        "  pass:",
        suff_summary["pass"],
    )

    print()
    print("Necessity")
    print(
        "  selected mean:",
        f"{nec_summary['selected_mean']:+.6f}",
    )
    print(
        "  bootstrap 95% CI:",
        nec_summary["bootstrap_95_ci"],
    )
    print(
        "  random p95:",
        f"{nec_summary['random_95th_percentile']:+.6f}",
    )
    print(
        "  percentile:",
        nec_summary["percentile_vs_random"],
    )
    print(
        "  pass:",
        nec_summary["pass"],
    )

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
