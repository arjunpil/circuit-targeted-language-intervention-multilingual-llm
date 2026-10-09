import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from common import (
    ROOT,
    build_position_matched_examples,
    capture_head_outputs,
    default_metric_json,
    exact_head_patch_metric,
    forward_capture_head_outputs,
    head_dim,
    language_metric,
    load_metric_tokens,
    load_model_and_tokenizer,
    model_inputs,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run Qwen2.5 EN<->target attention-head discovery "
            "and exact activation patching."
        )
    )

    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B",
    )

    p.add_argument(
        "--lang",
        default="es",
        help="Target language code (es, ru, zh, hi, ko, ...).",
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
    )

    p.add_argument(
        "--n-examples",
        type=int,
        default=30,
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=30,
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
        "--metric-json",
        default=None,
        help=(
            "Defaults to "
            "discovery/qwen2.5-1.5b/results/en_<lang>_language_metric.json."
        ),
    )

    p.add_argument(
        "--out",
        default=None,
        help=(
            "Defaults to "
            "results/discovery_repro/head_discovery_<lang>.json "
            "(head_discovery.json for --lang es)."
        ),
    )

    return p.parse_args()


def default_out_path(lang):
    if lang == "es":
        return ROOT / "results/discovery_repro/head_discovery.json"

    return ROOT / f"results/discovery_repro/head_discovery_{lang}.json"


def main():
    args = parse_args()

    metric_json = (
        ROOT / args.metric_json
        if args.metric_json is not None
        else default_metric_json(args.lang)
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

    H = model.config.num_attention_heads
    DH = head_dim(model)
    LAYERS = list(
        range(
            model.config.num_hidden_layers
        )
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

    if not examples:
        raise RuntimeError(
            "No usable position-matched examples."
        )

    print()
    print("=" * 72)
    print("HEAD ATTRIBUTION SCREEN")
    print("=" * 72)
    print("language pair: en ->", args.lang)
    print("examples:", len(examples))
    print(
        "indices:",
        examples[0]["index"],
        "-",
        examples[-1]["index"],
    )
    print("heads:", H)
    print("head dim:", DH)
    print()

    attribution_rows = []
    clean_cache = {}

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

        # Clean target-language head activations.
        clean_z = capture_head_outputs(
            model,
            target_inputs,
            LAYERS,
        )

        # Keep only the final-position vectors in CPU cache.
        clean_cache[
            example["index"]
        ] = {
            layer: (
                clean_z[layer][
                    0,
                    -1,
                    :,
                ]
                .detach()
                .cpu()
            )
            for layer in LAYERS
        }

        model.zero_grad(
            set_to_none=True
        )

        outputs, corrupted_z = (
            forward_capture_head_outputs(
                model,
                en_inputs,
                LAYERS,
            )
        )

        metric = language_metric(
            outputs.logits[
                :,
                -1,
                :,
            ],
            english_ids,
            target_ids,
        )[0]

        z_tensors = [
            corrupted_z[layer]
            for layer in LAYERS
        ]

        grads = torch.autograd.grad(
            metric,
            z_tensors,
            retain_graph=False,
            create_graph=False,
        )

        for layer, grad in zip(
            LAYERS,
            grads,
        ):
            clean_final = (
                clean_cache[
                    example["index"]
                ][layer]
                .to(
                    device=grad.device,
                    dtype=grad.dtype,
                )
            )

            corrupt_final = (
                corrupted_z[layer][
                    0,
                    -1,
                    :,
                ]
                .detach()
            )

            delta = (
                clean_final
                - corrupt_final
            )

            grad_final = grad[
                0,
                -1,
                :,
            ]

            for head in range(H):
                start = head * DH
                end = start + DH

                effect = float(
                    torch.dot(
                        grad_final[
                            start:end
                        ].float(),
                        delta[
                            start:end
                        ].float(),
                    )
                )

                attribution_rows.append(
                    {
                        "flores_index":
                            example["index"],
                        "layer":
                            layer,
                        "head":
                            head,
                        "attribution":
                            effect,
                    }
                )

        del (
            outputs,
            corrupted_z,
            z_tensors,
            grads,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    grouped = defaultdict(list)

    for row in attribution_rows:
        grouped[
            (
                row["layer"],
                row["head"],
            )
        ].append(
            row["attribution"]
        )

    attribution_summary = []

    for (
        layer,
        head,
    ), values in grouped.items():

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        attribution_summary.append(
            {
                "layer": layer,
                "head": head,
                "mean_attribution":
                    float(values.mean()),
                "mean_abs_attribution":
                    float(
                        np.abs(values).mean()
                    ),
                "std_attribution":
                    float(values.std()),
                "n":
                    int(len(values)),
            }
        )

    attribution_summary.sort(
        key=lambda row: abs(
            row["mean_attribution"]
        ),
        reverse=True,
    )

    candidates = attribution_summary[
        : args.top_k
    ]

    print()
    print("TOP ATTRIBUTION CANDIDATES")

    for row in candidates:
        print(
            f"L{row['layer']:02d}"
            f"H{row['head']:02d} "
            f"{row['mean_attribution']:+.6f}"
        )


    # --------------------------------------------------------
    # Exact patching of frozen top-k attribution candidates
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("EXACT HEAD PATCHING")
    print("=" * 72)

    exact_rows = []

    # Fast lookup of per-example attribution for sign agreement.
    attr_lookup = {
        (
            row["flores_index"],
            row["layer"],
            row["head"],
        ): row["attribution"]
        for row in attribution_rows
    }

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

        # Rebuild the cached final-position clean vectors in the
        # shape expected by exact_head_patch_metric.
        clean_z = {}

        for layer in LAYERS:
            vector = clean_cache[
                example["index"]
            ][layer].to(device)

            clean_z[layer] = (
                vector
                .reshape(1, 1, -1)
            )

        with torch.no_grad():
            baseline_logits = model(
                **en_inputs,
                use_cache=False,
            ).logits

            baseline = float(
                language_metric(
                    baseline_logits[
                        :,
                        -1,
                        :,
                    ],
                    english_ids,
                    target_ids,
                )[0]
            )

        for candidate in candidates:
            layer = candidate["layer"]
            head = candidate["head"]

            patched = (
                exact_head_patch_metric(
                    model=model,
                    inputs=en_inputs,
                    layer=layer,
                    head=head,
                    clean_z=clean_z,
                    english_token_ids=
                        english_ids,
                    target_token_ids=
                        target_ids,
                )
            )

            exact_rows.append(
                {
                    "flores_index":
                        example["index"],
                    "layer":
                        layer,
                    "head":
                        head,
                    "attribution":
                        attr_lookup[
                            (
                                example["index"],
                                layer,
                                head,
                            )
                        ],
                    "exact_effect":
                        patched - baseline,
                }
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    exact_grouped = defaultdict(list)

    for row in exact_rows:
        exact_grouped[
            (
                row["layer"],
                row["head"],
            )
        ].append(row)

    exact_summary = []

    for candidate in candidates:
        layer = candidate["layer"]
        head = candidate["head"]

        rows = exact_grouped[
            (layer, head)
        ]

        exact = np.asarray(
            [
                row["exact_effect"]
                for row in rows
            ],
            dtype=np.float64,
        )

        attr = np.asarray(
            [
                row["attribution"]
                for row in rows
            ],
            dtype=np.float64,
        )

        exact_summary.append(
            {
                "layer":
                    layer,
                "head":
                    head,
                "mean_attribution":
                    float(attr.mean()),
                "mean_exact_effect":
                    float(exact.mean()),
                "mean_abs_exact_effect":
                    float(
                        np.abs(exact).mean()
                    ),
                "std_exact_effect":
                    float(exact.std()),
                "sign_agreement":
                    float(
                        np.mean(
                            np.sign(attr)
                            == np.sign(exact)
                        )
                    ),
                "positive_exact_fraction":
                    float(
                        np.mean(exact > 0)
                    ),
                "n":
                    len(rows),
            }
        )

    mean_attr = np.asarray(
        [
            row["mean_attribution"]
            for row in exact_summary
        ]
    )

    mean_exact = np.asarray(
        [
            row["mean_exact_effect"]
            for row in exact_summary
        ]
    )

    rho, p_value = spearmanr(
        mean_attr,
        mean_exact,
    )

    # Selection rule.
    selected = [
        row
        for row in exact_summary
        if (
            row["mean_exact_effect"]
            >= 0.10
            and
            row["sign_agreement"]
            >= 0.75
        )
    ]

    selected.sort(
        key=lambda row:
            row["mean_exact_effect"],
        reverse=True,
    )

    heads = {}

    for row in selected:
        heads.setdefault(
            str(row["layer"]),
            [],
        ).append(
            row["head"]
        )

    for layer in heads:
        heads[layer] = sorted(
            heads[layer]
        )

    print()
    print("=" * 72)
    print("HEAD DISCOVERY SUMMARY")
    print("=" * 72)
    print(
        "Spearman attribution -> exact:"
    )
    print(f"  rho = {rho:.4f}")
    print(f"  p   = {p_value:.6g}")

    print()
    print("Selected heads:")

    for row in selected:
        print(
            f"  L{row['layer']:02d}"
            f"H{row['head']:02d}  "
            f"exact="
            f"{row['mean_exact_effect']:+.6f}  "
            f"sign="
            f"{row['sign_agreement']:.2f}"
        )

    result = {
        "model":
            args.model,
        "language_pair":
            f"en->{args.lang}",
        "activation_site":
            "attention head output before o_proj",
        "example_indices":
            [
                examples[0]["index"],
                examples[-1]["index"],
            ],
        "n_examples":
            len(examples),
        "top_k":
            args.top_k,
        "spearman_rho":
            float(rho),
        "spearman_p":
            float(p_value),
        "selection_rule": {
            "minimum_mean_exact_effect":
                0.10,
            "minimum_sign_agreement":
                0.75,
        },
        "heads":
            heads,
        "attribution_summary":
            attribution_summary,
        "exact_summary":
            exact_summary,
        "attribution_rows":
            attribution_rows,
        "exact_rows":
            exact_rows,
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
    print("Saved:", out)


if __name__ == "__main__":
    main()
