import argparse
import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
from scipy.stats import spearmanr

from common import (
    ROOT,
    build_position_matched_examples,
    capture_head_outputs,
    head_dim,
    language_metric,
    load_metric_tokens,
    load_model_and_tokenizer,
    model_inputs,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Reproduce source-to-destination attention-head "
            "path validation for the Qwen2.5 EN->ES pilot."
        )
    )

    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B",
    )

    p.add_argument(
        "--heads-json",
        default=(
            "discovery/qwen2.5-1.5b/results/"
            "qwen25_en_es_heads.json"
        ),
    )

    p.add_argument(
        "--metric-json",
        default=(
            "discovery/qwen2.5-1.5b/results/"
            "en_es_language_metric.json"
        ),
    )

    p.add_argument(
        "--start",
        type=int,
        default=200,
    )

    p.add_argument(
        "--n-examples",
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
        "--out",
        default=(
            "results/discovery_repro/"
            "path_validation.json"
        ),
    )

    return p.parse_args()


def load_heads(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    heads = []

    for layer, layer_heads in data["heads"].items():
        for head in layer_heads:
            heads.append(
                (int(layer), int(head))
            )

    heads.sort()

    return heads


def ordered_paths(heads):
    """
    All earlier-layer -> later-layer pairs among the frozen heads.
    """
    paths = []

    for source, destination in combinations(heads, 2):
        if source[0] < destination[0]:
            paths.append(
                (
                    source[0],
                    source[1],
                    destination[0],
                    destination[1],
                )
            )

    return paths


def slice_for_head(model, head):
    dh = head_dim(model)
    start = head * dh
    end = start + dh
    return start, end




def baseline_destination_gradients(
    model,
    inputs,
    destination_layers,
    english_ids,
    spanish_ids,
):
    """
    Run the untouched English baseline while retaining destination-head
    activations and their gradients for first-order path attribution.
    """
    store = {}
    handles = []

    for layer in destination_layers:
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
            store[layer] = hook_inputs[0]

        handles.append(
            o_proj.register_forward_pre_hook(
                hook
            )
        )

    model.zero_grad(set_to_none=True)

    try:
        outputs = model(
            **inputs,
            use_cache=False,
        )

        metric = language_metric(
            outputs.logits[:, -1, :],
            english_ids,
            spanish_ids,
        )[0]

        tensors = [
            store[layer]
            for layer in destination_layers
        ]

        grads = torch.autograd.grad(
            metric,
            tensors,
            retain_graph=False,
            create_graph=False,
        )

    finally:
        for handle in handles:
            handle.remove()

    activations = {
        layer: (
            store[layer]
            .detach()
            .clone()
        )
        for layer in destination_layers
    }

    gradients = {
        layer: (
            grad
            .detach()
            .clone()
        )
        for layer, grad in zip(
            destination_layers,
            grads,
        )
    }

    return (
        float(metric.detach()),
        activations,
        gradients,
    )


@torch.no_grad()
def patch_source_and_capture_destination(
    model,
    inputs,
    source_layer,
    source_head,
    clean_source_z,
    destination_layer,
    english_ids,
    spanish_ids,
):
    """
    Patch one clean Spanish source head into the English run, let the
    intervention propagate normally, and capture the resulting
    destination-head input to o_proj.
    """
    src_start, src_end = slice_for_head(
        model,
        source_head,
    )

    replacement = (
        clean_source_z[
            0,
            -1,
            src_start:src_end,
        ]
        .detach()
        .clone()
    )

    captured = {}

    source_o_proj = (
        model.model.layers[source_layer]
        .self_attn
        .o_proj
    )

    destination_o_proj = (
        model.model.layers[destination_layer]
        .self_attn
        .o_proj
    )

    def source_hook(
        module,
        hook_inputs,
    ):
        z = hook_inputs[0].clone()

        z[
            :,
            -1,
            src_start:src_end,
        ] = replacement.to(
            device=z.device,
            dtype=z.dtype,
        )

        return (z,) + tuple(
            hook_inputs[1:]
        )

    def destination_hook(
        module,
        hook_inputs,
    ):
        captured["z"] = (
            hook_inputs[0]
            .detach()
            .clone()
        )

    h1 = source_o_proj.register_forward_pre_hook(
        source_hook
    )

    h2 = destination_o_proj.register_forward_pre_hook(
        destination_hook
    )

    try:
        outputs = model(
            **inputs,
            use_cache=False,
        )

        metric = language_metric(
            outputs.logits[:, -1, :],
            english_ids,
            spanish_ids,
        )[0]

    finally:
        h1.remove()
        h2.remove()

    if "z" not in captured:
        raise RuntimeError(
            "Destination activation was not captured."
        )

    return (
        float(metric),
        captured["z"],
    )


@torch.no_grad()
def patch_destination_only(
    model,
    inputs,
    destination_layer,
    destination_head,
    replacement_z,
    english_ids,
    spanish_ids,
):
    """
    Patch only the destination head into an untouched English baseline.
    """
    dst_start, dst_end = slice_for_head(
        model,
        destination_head,
    )

    replacement = (
        replacement_z[
            0,
            -1,
            dst_start:dst_end,
        ]
        .detach()
        .clone()
    )

    o_proj = (
        model.model.layers[destination_layer]
        .self_attn
        .o_proj
    )

    def hook(
        module,
        hook_inputs,
    ):
        z = hook_inputs[0].clone()

        z[
            :,
            -1,
            dst_start:dst_end,
        ] = replacement.to(
            device=z.device,
            dtype=z.dtype,
        )

        return (z,) + tuple(
            hook_inputs[1:]
        )

    handle = o_proj.register_forward_pre_hook(
        hook
    )

    try:
        logits = model(
            **inputs,
            use_cache=False,
        ).logits

        metric = language_metric(
            logits[:, -1, :],
            english_ids,
            spanish_ids,
        )[0]

    finally:
        handle.remove()

    return float(metric)


def main():
    args = parse_args()

    metric_data = load_metric_tokens(
        ROOT / args.metric_json
    )

    english_ids = metric_data[
        "english_token_ids"
    ]

    spanish_ids = metric_data[
        "spanish_token_ids"
    ]

    heads = load_heads(
        ROOT / args.heads_json
    )

    paths = ordered_paths(heads)

    print("Frozen heads:")
    print(
        ", ".join(
            f"L{layer}H{head}"
            for layer, head in heads
        )
    )

    print("\nCandidate paths:", len(paths))

    for (
        src_layer,
        src_head,
        dst_layer,
        dst_head,
    ) in paths:
        print(
            f"  L{src_layer}H{src_head}"
            f" -> "
            f"L{dst_layer}H{dst_head}"
        )

    model, tok = load_model_and_tokenizer(
        args.model
    )

    device = next(
        model.parameters()
    ).device

    examples = build_position_matched_examples(
        tok,
        start=args.start,
        n_examples=args.n_examples,
        max_len=args.max_len,
        min_len=args.min_len,
    )

    if not examples:
        raise RuntimeError(
            "No usable examples."
        )

    destination_layers = sorted(
        {
            dst_layer
            for (
                _,
                _,
                dst_layer,
                _,
            ) in paths
        }
    )

    source_layers = sorted(
        {
            src_layer
            for (
                src_layer,
                _,
                _,
                _,
            ) in paths
        }
    )

    rows = []

    print()
    print("=" * 72)
    print("SOURCE -> DESTINATION PATH VALIDATION")
    print("=" * 72)
    print(
        "examples:",
        len(examples),
    )
    print(
        "indices:",
        examples[0]["index"],
        "-",
        examples[-1]["index"],
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

        clean_sources = capture_head_outputs(
            model,
            es_inputs,
            source_layers,
        )

        (
            baseline,
            baseline_destinations,
            baseline_gradients,
        ) = baseline_destination_gradients(
            model,
            en_inputs,
            destination_layers,
            english_ids,
            spanish_ids,
        )

        # A source intervention is identical for every path sharing
        # that source, so cache its propagated destination states.
        propagation_cache = {}

        for (
            src_layer,
            src_head,
            dst_layer,
            dst_head,
        ) in paths:

            cache_key = (
                src_layer,
                src_head,
                dst_layer,
            )

            if cache_key not in propagation_cache:
                (
                    source_metric,
                    destination_after_source,
                ) = (
                    patch_source_and_capture_destination(
                        model=model,
                        inputs=en_inputs,
                        source_layer=src_layer,
                        source_head=src_head,
                        clean_source_z=clean_sources[
                            src_layer
                        ],
                        destination_layer=dst_layer,
                        english_ids=english_ids,
                        spanish_ids=spanish_ids,
                    )
                )

                propagation_cache[
                    cache_key
                ] = (
                    source_metric,
                    destination_after_source,
                )

            (
                source_metric,
                destination_after_source,
            ) = propagation_cache[
                cache_key
            ]

            dst_start, dst_end = slice_for_head(
                model,
                dst_head,
            )

            baseline_dst = (
                baseline_destinations[
                    dst_layer
                ][
                    0,
                    -1,
                    dst_start:dst_end,
                ]
            )

            propagated_dst = (
                destination_after_source[
                    0,
                    -1,
                    dst_start:dst_end,
                ]
            )

            receiver_delta = (
                propagated_dst
                - baseline_dst
            )

            gradient = (
                baseline_gradients[
                    dst_layer
                ][
                    0,
                    -1,
                    dst_start:dst_end,
                ]
            )

            attribution = float(
                torch.dot(
                    gradient.float(),
                    receiver_delta.float(),
                )
            )

            patched_metric = patch_destination_only(
                model=model,
                inputs=en_inputs,
                destination_layer=dst_layer,
                destination_head=dst_head,
                replacement_z=destination_after_source,
                english_ids=english_ids,
                spanish_ids=spanish_ids,
            )

            exact_path_effect = (
                patched_metric
                - baseline
            )

            source_total_effect = (
                source_metric
                - baseline
            )

            rows.append(
                {
                    "flores_index":
                        example["index"],
                    "source_layer":
                        src_layer,
                    "source_head":
                        src_head,
                    "destination_layer":
                        dst_layer,
                    "destination_head":
                        dst_head,
                    "attribution":
                        attribution,
                    "exact_path_effect":
                        exact_path_effect,
                    "source_total_effect":
                        source_total_effect,
                    "receiver_delta_norm":
                        float(
                            torch.linalg.vector_norm(
                                receiver_delta.float()
                            )
                        ),
                }
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["source_layer"],
            row["source_head"],
            row["destination_layer"],
            row["destination_head"],
        )

        grouped[key].append(row)

    summary = []

    for (
        src_layer,
        src_head,
        dst_layer,
        dst_head,
    ), group in grouped.items():

        attr = np.asarray(
            [
                r["attribution"]
                for r in group
            ],
            dtype=np.float64,
        )

        exact = np.asarray(
            [
                r["exact_path_effect"]
                for r in group
            ],
            dtype=np.float64,
        )

        source = np.asarray(
            [
                r["source_total_effect"]
                for r in group
            ],
            dtype=np.float64,
        )

        mean_abs_exact = float(
            np.abs(exact).mean()
        )

        mean_abs_source = float(
            np.abs(source).mean()
        )

        ratio = (
            mean_abs_exact / mean_abs_source
            if mean_abs_source > 0
            else float("nan")
        )

        summary.append(
            {
                "src_layer":
                    src_layer,
                "src_head":
                    src_head,
                "dst_layer":
                    dst_layer,
                "dst_head":
                    dst_head,
                "n":
                    len(group),
                "mean_attribution":
                    float(attr.mean()),
                "mean_exact_path_effect":
                    float(exact.mean()),
                "mean_abs_path_effect":
                    mean_abs_exact,
                "std_path_effect":
                    float(exact.std()),
                "attribution_exact_sign_agreement":
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
                "mean_source_total_effect":
                    float(source.mean()),
                "mean_abs_source_total_effect":
                    mean_abs_source,
                "path_to_source_abs_ratio":
                    ratio,
            }
        )

    summary.sort(
        key=lambda x: abs(
            x["mean_exact_path_effect"]
        ),
        reverse=True,
    )

    all_attr = np.asarray(
        [
            row["attribution"]
            for row in rows
        ],
        dtype=np.float64,
    )

    all_exact = np.asarray(
        [
            row["exact_path_effect"]
            for row in rows
        ],
        dtype=np.float64,
    )

    per_example_rho, per_example_p = spearmanr(
        all_attr,
        all_exact,
    )

    edge_attr = np.asarray(
        [
            row["mean_attribution"]
            for row in summary
        ],
        dtype=np.float64,
    )

    edge_exact = np.asarray(
        [
            row["mean_exact_path_effect"]
            for row in summary
        ],
        dtype=np.float64,
    )

    edge_rho, edge_p = spearmanr(
        edge_attr,
        edge_exact,
    )

    print()
    print("=" * 72)
    print("PATH VALIDATION SUMMARY")
    print("=" * 72)

    print(
        "Per-example attribution/exact "
        f"rho = {per_example_rho:.6f}"
    )

    print(
        "Path-mean attribution/exact "
        f"rho = {edge_rho:.6f}"
    )

    print()

    for row in summary:
        print(
            f"L{row['src_layer']:02d}"
            f"H{row['src_head']:02d}"
            " -> "
            f"L{row['dst_layer']:02d}"
            f"H{row['dst_head']:02d}  "
            f"exact="
            f"{row['mean_exact_path_effect']:+.6f}  "
            f"attr="
            f"{row['mean_attribution']:+.6f}"
        )

    result = {
        "model":
            args.model,
        "language_pair":
            "en->es",
        "method": (
            "source-head clean activation patch, capture resulting "
            "destination-head output, then patch only that "
            "destination head into the corrupted baseline"
        ),
        "example_indices": [
            examples[0]["index"],
            examples[-1]["index"],
        ],
        "n_examples":
            len(examples),
        "n_candidate_paths":
            len(paths),
        "per_example_spearman": {
            "rho":
                float(per_example_rho),
            "p":
                float(per_example_p),
        },
        "edge_mean_spearman": {
            "rho":
                float(edge_rho),
            "p":
                float(edge_p),
        },
        "summary":
            summary,
        "rows":
            rows,
    }

    out = ROOT / args.out
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
