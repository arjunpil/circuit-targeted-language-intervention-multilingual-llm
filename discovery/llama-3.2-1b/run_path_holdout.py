import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import (
    ROOT,
    build_position_matched_examples,
    capture_head_outputs,
    language_metric,
    load_metric_tokens,
    load_model_and_tokenizer,
    model_inputs,
)

from run_path_validation import (
    patch_destination_only,
    patch_source_and_capture_destination,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Held-out validation of frozen Llama "
            "source-to-destination head paths."
        )
    )

    p.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
    )

    p.add_argument(
        "--frozen-paths",
        default=(
            "discovery/llama-3.2-1b/results/"
            "llama32_en_es_frozen_paths.json"
        ),
    )

    p.add_argument(
        "--metric-json",
        default=(
            "discovery/llama-3.2-1b/results/"
            "en_es_language_metric.json"
        ),
    )

    p.add_argument(
        "--start",
        type=int,
        default=300,
    )

    p.add_argument(
        "--n-examples",
        type=int,
        default=40,
    )

    p.add_argument(
        "--n-random-controls",
        type=int,
        default=50,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
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
        "--out",
        default=(
            "results/llama32_path_validation/"
            "path_holdout.json"
        ),
    )

    return p.parse_args()


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


def make_controls(
    path,
    n_heads,
    n_controls,
    rng,
):
    source_candidates = [
        h
        for h in range(n_heads)
        if h != path["src_head"]
    ]

    destination_candidates = [
        h
        for h in range(n_heads)
        if h != path["dst_head"]
    ]

    candidates = [
        (src, dst)
        for src in source_candidates
        for dst in destination_candidates
    ]

    if n_controls > len(candidates):
        raise ValueError(
            "Requested more unique controls "
            "than are available."
        )

    chosen = rng.choice(
        len(candidates),
        size=n_controls,
        replace=False,
    )

    return [
        {
            "source_head":
                int(candidates[i][0]),
            "destination_head":
                int(candidates[i][1]),
        }
        for i in chosen
    ]


@torch.no_grad()
def baseline_metric(
    model,
    inputs,
    english_ids,
    spanish_ids,
):
    logits = model(
        **inputs,
        use_cache=False,
    ).logits

    value = language_metric(
        logits[:, -1, :],
        english_ids,
        spanish_ids,
    )[0]

    return float(value)


def main():
    args = parse_args()

    frozen = json.loads(
        (ROOT / args.frozen_paths)
        .read_text(encoding="utf-8")
    )

    paths = frozen["paths"]

    metric = load_metric_tokens(
        ROOT / args.metric_json
    )

    english_ids = metric[
        "english_token_ids"
    ]

    spanish_ids = metric[
        "spanish_token_ids"
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

    examples = (
        build_position_matched_examples(
            tok,
            start=args.start,
            n_examples=args.n_examples,
            max_len=args.max_len,
            min_len=args.min_len,
        )
    )

    if len(examples) != args.n_examples:
        raise RuntimeError(
            f"Expected {args.n_examples} examples, "
            f"found {len(examples)}."
        )

    rng = np.random.default_rng(
        args.seed
    )

    controls = [
        make_controls(
            path,
            n_heads,
            args.n_random_controls,
            rng,
        )
        for path in paths
    ]

    selected_effects = [
        []
        for _ in paths
    ]

    control_effects = [
        [
            []
            for _ in path_controls
        ]
        for path_controls in controls
    ]

    source_layers = sorted(
        {
            path["src_layer"]
            for path in paths
        }
    )

    print()
    print("=" * 72)
    print("FROZEN PATH HOLDOUT")
    print("=" * 72)
    print(
        "examples:",
        examples[0]["index"],
        "-",
        examples[-1]["index"],
    )
    print("paths:", len(paths))
    print(
        "controls per path:",
        args.n_random_controls,
    )
    print()

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

        es_inputs = model_inputs(
            example["es_ids"],
            device,
        )

        clean_sources = (
            capture_head_outputs(
                model,
                es_inputs,
                source_layers,
            )
        )

        baseline = baseline_metric(
            model,
            en_inputs,
            english_ids,
            spanish_ids,
        )

        propagation_cache = {}

        def path_effect(
            src_layer,
            src_head,
            dst_layer,
            dst_head,
        ):
            key = (
                src_layer,
                src_head,
                dst_layer,
            )

            if key not in propagation_cache:
                (
                    _,
                    destination_after_source,
                ) = (
                    patch_source_and_capture_destination(
                        model=model,
                        inputs=en_inputs,
                        source_layer=src_layer,
                        source_head=src_head,
                        clean_source_z=
                            clean_sources[src_layer],
                        destination_layer=dst_layer,
                        english_ids=english_ids,
                        spanish_ids=spanish_ids,
                    )
                )

                propagation_cache[key] = (
                    destination_after_source
                )

            destination_after_source = (
                propagation_cache[key]
            )

            patched = patch_destination_only(
                model=model,
                inputs=en_inputs,
                destination_layer=dst_layer,
                destination_head=dst_head,
                replacement_z=
                    destination_after_source,
                english_ids=english_ids,
                spanish_ids=spanish_ids,
            )

            return float(
                patched - baseline
            )

        for path_i, path in enumerate(
            paths
        ):
            value = path_effect(
                path["src_layer"],
                path["src_head"],
                path["dst_layer"],
                path["dst_head"],
            )

            selected_effects[
                path_i
            ].append(value)

            for control_i, control in enumerate(
                controls[path_i]
            ):
                value = path_effect(
                    path["src_layer"],
                    control["source_head"],
                    path["dst_layer"],
                    control[
                        "destination_head"
                    ],
                )

                control_effects[
                    path_i
                ][control_i].append(
                    value
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validation = []

    for path_i, path in enumerate(
        paths
    ):
        raw = np.asarray(
            selected_effects[path_i],
            dtype=np.float64,
        )

        sign = int(
            path["discovery_sign"]
        )

        aligned = raw * sign

        random_raw_means = np.asarray(
            [
                np.mean(values)
                for values
                in control_effects[path_i]
            ],
            dtype=np.float64,
        )

        random_aligned = (
            random_raw_means * sign
        )

        raw_mean = float(
            raw.mean()
        )

        aligned_mean = float(
            aligned.mean()
        )

        ci = bootstrap_ci(
            aligned,
            args.bootstrap_samples,
            args.seed + 100 + path_i,
        )

        p95 = float(
            np.percentile(
                random_aligned,
                95,
            )
        )

        percentile = float(
            100.0
            * np.mean(
                random_aligned
                <= aligned_mean
            )
        )

        direction_replicated = bool(
            raw_mean * sign > 0
        )

        random_gate = bool(
            aligned_mean > p95
        )

        ci_excludes_zero = bool(
            ci[0] > 0
        )

        result = {
            **path,
            "n_holdout":
                len(raw),
            "holdout_raw_mean":
                raw_mean,
            "holdout_aligned_mean":
                aligned_mean,
            "aligned_bootstrap_95_ci":
                ci,
            "random_aligned_95th_percentile":
                p95,
            "percentile_vs_random":
                percentile,
            "direction_replicated":
                direction_replicated,
            "random_gate":
                random_gate,
            "ci_excludes_zero":
                ci_excludes_zero,
            "pass":
                bool(
                    direction_replicated
                    and random_gate
                    and ci_excludes_zero
                ),
            "per_example_raw_effect":
                raw.tolist(),
            "random_controls": [
                {
                    **control,
                    "raw_mean_effect":
                        float(
                            random_raw_means[i]
                        ),
                    "aligned_mean_effect":
                        float(
                            random_aligned[i]
                        ),
                }
                for i, control
                in enumerate(
                    controls[path_i]
                )
            ],
        }

        validation.append(
            result
        )

    output = {
        "model":
            args.model,
        "language_pair":
            "en->es",
        "selection_rule":
            frozen["selection_rule"],
        "discovery_example_indices":
            frozen[
                "discovery_example_indices"
            ],
        "holdout_example_indices": [
            examples[0]["index"],
            examples[-1]["index"],
        ],
        "n_holdout_examples":
            len(examples),
        "n_random_controls_per_path":
            args.n_random_controls,
        "control_matching": (
            "same source layer and destination "
            "layer; both head identities replaced"
        ),
        "random_seed":
            args.seed,
        "validation":
            validation,
    }

    out = ROOT / args.out

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("PATH HOLDOUT SUMMARY")
    print("=" * 72)

    for row in validation:
        print(
            f"L{row['src_layer']}H{row['src_head']}"
            " -> "
            f"L{row['dst_layer']}H{row['dst_head']} | "
            f"discovery={row['discovery_effect']:+.6f} | "
            f"holdout={row['holdout_raw_mean']:+.6f} | "
            f"CI={row['aligned_bootstrap_95_ci']} | "
            f"random p95="
            f"{row['random_aligned_95th_percentile']:+.6f} | "
            f"pct={row['percentile_vs_random']:.1f} | "
            f"pass={row['pass']}"
        )

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
