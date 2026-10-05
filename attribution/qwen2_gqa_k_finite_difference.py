import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

DEVICE = "cpu"
DTYPE = torch.float64

SEEDS = list(range(10))
H = 0.1
SIGNAL_FLOOR = 1e-4

N_QUERY_HEADS = 4
N_KV_HEADS = 2
HIDDEN_SIZE = 32
HEAD_DIM = HIDDEN_SIZE // N_QUERY_HEADS
LAYER_IDX = 0

TARGET_TOKEN = 7
FOIL_TOKEN = 11

SOURCE_POSITIONS = [1, 3, 5]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_IDS = torch.tensor(
    [[1, 5, 9, 13, 17, 21, 25, 29]],
    dtype=torch.long,
)

CORRUPTED_IDS = torch.tensor(
    [[1, 6, 10, 14, 18, 22, 26, 30]],
    dtype=torch.long,
)


def metric(logits):
    return (
        logits[0, -1, TARGET_TOKEN]
        - logits[0, -1, FOIL_TOKEN]
    )


def build_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=N_QUERY_HEADS,
        num_key_value_heads=N_KV_HEADS,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        _attn_implementation="eager",
    )

    model = Qwen2ForCausalLM(config).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    model.eval()

    return model


def get_kproj(model):
    return model.model.layers[LAYER_IDX].self_attn.k_proj


def capture_k(model, ids):
    captured = {}

    def hook(module, inputs, output):
        captured["k"] = output

    handle = get_kproj(model).register_forward_hook(hook)

    try:
        outputs = model(
            input_ids=ids,
            use_cache=False,
        )
    finally:
        handle.remove()

    return outputs, captured["k"]


def patched_metric(
    model,
    kv_head,
    source_position,
    delta,
    alpha,
):
    start = kv_head * HEAD_DIM
    end = start + HEAD_DIM

    def hook(module, inputs, output):
        patched = output.clone()

        patched[:, source_position, start:end] = (
            patched[:, source_position, start:end]
            + alpha * delta
        )

        return patched

    handle = get_kproj(model).register_forward_hook(hook)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=CORRUPTED_IDS,
                use_cache=False,
            )
            return float(metric(outputs.logits))
    finally:
        handle.remove()


rows = []

for seed in SEEDS:
    model = build_model(seed)

    with torch.no_grad():
        _, clean_k = capture_k(
            model,
            CLEAN_IDS,
        )

    corrupted_outputs, corrupted_k = capture_k(
        model,
        CORRUPTED_IDS,
    )

    base_metric = metric(
        corrupted_outputs.logits
    )

    base_value = float(
        base_metric.detach()
    )

    grad_k = torch.autograd.grad(
        outputs=base_metric,
        inputs=corrupted_k,
    )[0].detach()

    for kv_head in range(N_KV_HEADS):
        start = kv_head * HEAD_DIM
        end = start + HEAD_DIM

        for position in SOURCE_POSITIONS:
            delta = (
                clean_k[
                    0,
                    position,
                    start:end,
                ]
                - corrupted_k.detach()[
                    0,
                    position,
                    start:end,
                ]
            ).detach()

            autograd_derivative = float(
                torch.dot(
                    grad_k[
                        0,
                        position,
                        start:end,
                    ],
                    delta,
                )
            )

            if (
                abs(autograd_derivative)
                < SIGNAL_FLOOR
            ):
                continue

            plus = patched_metric(
                model,
                kv_head,
                position,
                delta,
                +H,
            )

            minus = patched_metric(
                model,
                kv_head,
                position,
                delta,
                -H,
            )

            finite_difference = (
                plus - minus
            ) / (2 * H)

            absolute_error = abs(
                finite_difference
                - autograd_derivative
            )

            relative_error = (
                absolute_error
                / abs(autograd_derivative)
            )

            sign_agreement = bool(
                np.sign(finite_difference)
                == np.sign(autograd_derivative)
            )

            symmetry_error = abs(
                plus
                + minus
                - 2 * base_value
            )

            rows.append(
                {
                    "seed": seed,
                    "kv_head": kv_head,
                    "source_position": position,
                    "autograd_derivative":
                        autograd_derivative,
                    "finite_difference":
                        finite_difference,
                    "absolute_error":
                        absolute_error,
                    "relative_error":
                        relative_error,
                    "sign_agreement":
                        sign_agreement,
                    "symmetry_error":
                        symmetry_error,
                }
            )

    del model


if not rows:
    raise RuntimeError(
        "No meaningful K cases found."
    )

relative_errors = np.array(
    [r["relative_error"] for r in rows]
)

sign_rate = float(
    np.mean(
        [r["sign_agreement"] for r in rows]
    )
)

summary = {
    "n_seeds": len(SEEDS),
    "n_meaningful_k_cases": len(rows),
    "signal_floor": SIGNAL_FLOOR,
    "finite_difference_h": H,
    "median_relative_error":
        float(np.median(relative_errors)),
    "p90_relative_error":
        float(np.percentile(relative_errors, 90)),
    "max_relative_error":
        float(np.max(relative_errors)),
    "sign_agreement_rate":
        sign_rate,
}

summary["pass"] = bool(
    summary["n_meaningful_k_cases"] >= 10
    and summary["median_relative_error"] < 0.01
    and summary["p90_relative_error"] < 0.01
    and sign_rate == 1.0
)

print("=" * 72)
print("QWEN2 GQA K-PROJECTION FINITE-DIFFERENCE CHECK")
print("=" * 72)

for key, value in summary.items():
    print(f"{key}: {value}")

with open(
    RESULTS_DIR / "qwen2_gqa_k_finite_difference.csv",
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(rows)

with open(
    RESULTS_DIR / "qwen2_gqa_k_finite_difference_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

if not summary["pass"]:
    raise SystemExit(
        "Finite-difference validation FAILED."
    )
