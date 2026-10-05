import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEVICE = "cpu"
DTYPE = torch.float64

MODEL_SEEDS = [42, 123, 999]
N_DIRECTIONS = 5

EPSILONS = [
    1e-1,
    3e-2,
    1e-2,
    3e-3,
    1e-3,
]

# Random directions whose gradient projection is almost zero make
# relative-error measurements ill-conditioned. We therefore require a
# modest nonzero projection while still keeping directions random.
MIN_ABS_COSINE = 0.08

TARGET_TOKEN = 7
FOIL_TOKEN = 11

INPUT_IDS = torch.tensor(
    [[1, 12, 9, 21, 5, 18, 7, 4]],
    dtype=torch.long,
    device=DEVICE,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_ctx=16,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )

    model = GPT2LMHeadModel(config).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    model.eval()

    return model, config


# ---------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------

def metric_from_logits(logits):
    return (
        logits[0, -1, TARGET_TOKEN]
        - logits[0, -1, FOIL_TOKEN]
    )


# ---------------------------------------------------------------------
# Capture residual stream and compute its gradient
# ---------------------------------------------------------------------

def baseline_and_gradient(model):
    # Residual stream between blocks 0 and 1.
    patch_module = model.transformer.h[1]

    captured = {}

    def capture_pre_hook(module, inputs):
        captured["hidden"] = inputs[0]

    model.zero_grad(set_to_none=True)

    handle = patch_module.register_forward_pre_hook(
        capture_pre_hook
    )

    outputs = model(
        input_ids=INPUT_IDS,
        use_cache=False,
    )

    metric = metric_from_logits(outputs.logits)
    baseline_metric = float(metric.detach())

    hidden = captured["hidden"]

    if not hidden.requires_grad:
        handle.remove()
        raise RuntimeError(
            "Captured residual stream does not require gradients."
        )

    hidden_grad = torch.autograd.grad(
        outputs=metric,
        inputs=hidden,
        retain_graph=False,
        create_graph=False,
    )[0]

    handle.remove()

    grad = hidden_grad[0, -1, :].detach().clone()

    grad_norm = torch.linalg.vector_norm(grad)

    if grad_norm.item() == 0:
        raise RuntimeError(
            "Gradient norm is zero."
        )

    return (
        patch_module,
        baseline_metric,
        grad,
        float(grad_norm),
    )


# ---------------------------------------------------------------------
# Random perturbation directions
# ---------------------------------------------------------------------

def sample_random_directions(
    grad,
    model_seed,
    n_directions=N_DIRECTIONS,
):
    grad_norm = torch.linalg.vector_norm(grad)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(model_seed + 100_000)

    accepted = []
    attempts = 0

    while len(accepted) < n_directions:
        attempts += 1

        if attempts > 10000:
            raise RuntimeError(
                "Could not find enough usable random directions."
            )

        raw = torch.randn(
            grad.shape,
            generator=generator,
            dtype=DTYPE,
            device=DEVICE,
        )

        norm = torch.linalg.vector_norm(raw)

        if norm.item() == 0:
            continue

        direction = raw / norm

        cosine = (
            torch.dot(grad, direction)
            / grad_norm
        ).item()

        if abs(cosine) < MIN_ABS_COSINE:
            continue

        accepted.append(
            {
                "direction": direction,
                "cosine": cosine,
            }
        )

    return accepted


# ---------------------------------------------------------------------
# Exact perturbation
# ---------------------------------------------------------------------

def exact_delta_for_intervention(
    model,
    patch_module,
    baseline_metric,
    delta_activation,
):
    def perturb_pre_hook(module, inputs):
        hidden = inputs[0]

        patched = hidden.clone()

        patched[:, -1, :] = (
            patched[:, -1, :]
            + delta_activation
        )

        return (patched,) + tuple(inputs[1:])

    handle = patch_module.register_forward_pre_hook(
        perturb_pre_hook
    )

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=INPUT_IDS,
                use_cache=False,
            )

            perturbed_metric = float(
                metric_from_logits(outputs.logits)
            )

    finally:
        handle.remove()

    return perturbed_metric - baseline_metric


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

all_rows = []
curve_rows = []
seed_summaries = []

for model_seed in MODEL_SEEDS:
    print()
    print("=" * 70)
    print(f"MODEL SEED {model_seed}")
    print("=" * 70)

    model, config = build_model(model_seed)

    (
        patch_module,
        baseline_metric,
        grad,
        grad_norm,
    ) = baseline_and_gradient(model)

    print(f"baseline metric : {baseline_metric:.10f}")
    print(f"gradient norm   : {grad_norm:.10f}")

    directions = sample_random_directions(
        grad=grad,
        model_seed=model_seed,
    )

    print(
        "accepted |cosine|:",
        [
            round(abs(d["cosine"]), 4)
            for d in directions
        ],
    )

    seed_curve_rows = []

    for direction_idx, item in enumerate(directions):
        direction = item["direction"]
        cosine = item["cosine"]

        for sign in [-1, 1]:
            intervention_rows = []

            for eps in EPSILONS:
                delta_activation = (
                    sign
                    * eps
                    * direction
                )

                # First-order prediction:
                #
                # Δm ≈ ∇m · Δh
                predicted_delta = float(
                    torch.dot(
                        grad,
                        delta_activation,
                    )
                )

                exact_delta = exact_delta_for_intervention(
                    model=model,
                    patch_module=patch_module,
                    baseline_metric=baseline_metric,
                    delta_activation=delta_activation,
                )

                absolute_error = abs(
                    predicted_delta - exact_delta
                )

                # Exact-effect relative error.
                relative_error = (
                    absolute_error
                    / max(
                        abs(exact_delta),
                        1e-15,
                    )
                )

                sign_agreement = bool(
                    np.sign(predicted_delta)
                    == np.sign(exact_delta)
                )

                row = {
                    "model_seed": model_seed,
                    "direction_idx": direction_idx,
                    "direction_cosine": cosine,
                    "sign": sign,
                    "epsilon": eps,
                    "predicted_delta": predicted_delta,
                    "exact_delta": exact_delta,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "sign_agreement": sign_agreement,
                }

                all_rows.append(row)
                intervention_rows.append(row)

            # Fit Taylor-error scaling using the four larger eps values.
            # This reduces sensitivity to floating-point floors.
            fit_rows = intervention_rows[:4]

            log_eps = np.log(
                [
                    r["epsilon"]
                    for r in fit_rows
                ]
            )

            log_error = np.log(
                [
                    max(
                        r["absolute_error"],
                        1e-30,
                    )
                    for r in fit_rows
                ]
            )

            slope = float(
                np.polyfit(
                    log_eps,
                    log_error,
                    1,
                )[0]
            )

            smallest = intervention_rows[-1]

            small_rows = [
                r
                for r in intervention_rows
                if r["epsilon"] <= 3e-3
            ]

            small_sign_agreement = all(
                r["sign_agreement"]
                for r in small_rows
            )

            curve = {
                "model_seed": model_seed,
                "direction_idx": direction_idx,
                "direction_cosine": cosine,
                "sign": sign,
                "taylor_error_slope": slope,
                "smallest_epsilon": smallest["epsilon"],
                "smallest_relative_error": smallest[
                    "relative_error"
                ],
                "smallest_absolute_error": smallest[
                    "absolute_error"
                ],
                "smallest_sign_agreement": smallest[
                    "sign_agreement"
                ],
                "small_epsilon_sign_agreement": (
                    small_sign_agreement
                ),
            }

            curve_rows.append(curve)
            seed_curve_rows.append(curve)

    seed_slopes = np.array(
        [
            r["taylor_error_slope"]
            for r in seed_curve_rows
        ]
    )

    seed_rel_errors = np.array(
        [
            r["smallest_relative_error"]
            for r in seed_curve_rows
        ]
    )

    seed_small_sign = np.mean(
        [
            r["small_epsilon_sign_agreement"]
            for r in seed_curve_rows
        ]
    )

    seed_summary = {
        "model_seed": model_seed,
        "n_curves": len(seed_curve_rows),
        "median_slope": float(
            np.median(seed_slopes)
        ),
        "fraction_slope_gt_1_5": float(
            np.mean(seed_slopes > 1.5)
        ),
        "median_smallest_relative_error": float(
            np.median(seed_rel_errors)
        ),
        "max_smallest_relative_error": float(
            np.max(seed_rel_errors)
        ),
        "small_epsilon_sign_agreement_rate": float(
            seed_small_sign
        ),
    }

    seed_summaries.append(seed_summary)

    print(
        f"median slope      : "
        f"{seed_summary['median_slope']:.4f}"
    )
    print(
        f"slope > 1.5       : "
        f"{seed_summary['fraction_slope_gt_1_5']:.1%}"
    )
    print(
        f"median rel error  : "
        f"{seed_summary['median_smallest_relative_error']:.6e}"
    )
    print(
        f"max rel error     : "
        f"{seed_summary['max_smallest_relative_error']:.6e}"
    )
    print(
        f"small-eps signs   : "
        f"{seed_summary['small_epsilon_sign_agreement_rate']:.1%}"
    )

    del model


# ---------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------

slopes = np.array(
    [
        r["taylor_error_slope"]
        for r in curve_rows
    ]
)

smallest_rel_errors = np.array(
    [
        r["smallest_relative_error"]
        for r in curve_rows
    ]
)

small_epsilon_sign_rate = float(
    np.mean(
        [
            r["small_epsilon_sign_agreement"]
            for r in curve_rows
        ]
    )
)

median_slope = float(
    np.median(slopes)
)

fraction_slope_gt_1_5 = float(
    np.mean(slopes > 1.5)
)

median_smallest_rel_error = float(
    np.median(smallest_rel_errors)
)

p90_smallest_rel_error = float(
    np.percentile(
        smallest_rel_errors,
        90,
    )
)

max_smallest_rel_error = float(
    np.max(smallest_rel_errors)
)


# These thresholds are fixed before looking at this run.
passed = bool(
    median_slope > 1.5
    and fraction_slope_gt_1_5 >= 0.80
    and median_smallest_rel_error < 0.01
    and p90_smallest_rel_error < 0.05
    and small_epsilon_sign_rate >= 0.98
)


summary = {
    "device": DEVICE,
    "dtype": str(DTYPE),
    "model_seeds": MODEL_SEEDS,
    "n_directions_per_model": N_DIRECTIONS,
    "signs": [-1, 1],
    "epsilons": EPSILONS,
    "min_abs_cosine": MIN_ABS_COSINE,

    "n_models": len(MODEL_SEEDS),
    "n_curves": len(curve_rows),
    "n_exact_interventions": len(all_rows),

    "median_taylor_error_slope": median_slope,
    "fraction_curves_slope_gt_1_5": (
        fraction_slope_gt_1_5
    ),

    "median_smallest_relative_error": (
        median_smallest_rel_error
    ),

    "p90_smallest_relative_error": (
        p90_smallest_rel_error
    ),

    "max_smallest_relative_error": (
        max_smallest_rel_error
    ),

    "small_epsilon_sign_agreement_rate": (
        small_epsilon_sign_rate
    ),

    "seed_summaries": seed_summaries,

    "pass_criteria": {
        "median_slope_gt": 1.5,
        "fraction_slope_gt_1_5_gte": 0.80,
        "median_smallest_relative_error_lt": 0.01,
        "p90_smallest_relative_error_lt": 0.05,
        "small_epsilon_sign_agreement_rate_gte": 0.98,
    },

    "pass": passed,
}


print()
print("=" * 70)
print("ROBUST PHASE 0 SUMMARY")
print("=" * 70)
print(f"models tested                    : {len(MODEL_SEEDS)}")
print(f"random directions/model          : {N_DIRECTIONS}")
print(f"signed perturbation curves       : {len(curve_rows)}")
print(f"exact interventions              : {len(all_rows)}")
print()
print(f"median Taylor-error slope         : {median_slope:.4f}")
print(
    f"fraction slopes > 1.5            : "
    f"{fraction_slope_gt_1_5:.1%}"
)
print(
    f"median rel. error @ eps=1e-3     : "
    f"{median_smallest_rel_error:.6e}"
)
print(
    f"90th pct rel. error @ eps=1e-3   : "
    f"{p90_smallest_rel_error:.6e}"
)
print(
    f"max rel. error @ eps=1e-3        : "
    f"{max_smallest_rel_error:.6e}"
)
print(
    f"small-epsilon sign agreement      : "
    f"{small_epsilon_sign_rate:.1%}"
)
print()
print(f"PASS: {passed}")


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

with open(
    RESULTS_DIR
    / "attribution_vs_exact_robust_interventions.csv",
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=all_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(all_rows)


with open(
    RESULTS_DIR
    / "attribution_vs_exact_robust_curves.csv",
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=curve_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(curve_rows)


with open(
    RESULTS_DIR
    / "attribution_vs_exact_robust_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


if not passed:
    raise SystemExit(
        "Robust Phase 0 check FAILED. "
        "Inspect individual directions before proceeding."
    )


print()
print(
    "Saved:"
)
print(
    "  phase0/results/"
    "attribution_vs_exact_robust_interventions.csv"
)
print(
    "  phase0/results/"
    "attribution_vs_exact_robust_curves.csv"
)
print(
    "  phase0/results/"
    "attribution_vs_exact_robust_summary.json"
)
