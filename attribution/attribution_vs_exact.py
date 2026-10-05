import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel


SEED = 42
DEVICE = "cpu"
DTYPE = torch.float64

torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Tiny random transformer
# ---------------------------------------------------------------------

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

    # Keep special-token IDs inside the toy vocabulary.
    bos_token_id=0,
    eos_token_id=0,
    pad_token_id=0,
)

model = GPT2LMHeadModel(config).to(
    device=DEVICE,
    dtype=DTYPE,
)
model.eval()

input_ids = torch.tensor(
    [[1, 12, 9, 21, 5, 18, 7, 4]],
    dtype=torch.long,
    device=DEVICE,
)

TARGET_TOKEN = 7
FOIL_TOKEN = 11


# We intervene on the residual stream BETWEEN block 0 and block 1.
# Therefore we hook the INPUT to block 1.
patch_module = model.transformer.h[1]

captured = {}


def capture_pre_hook(module, inputs):
    hidden = inputs[0]
    captured["hidden"] = hidden


# ---------------------------------------------------------------------
# Baseline forward pass + exact gradient
# ---------------------------------------------------------------------

model.zero_grad(set_to_none=True)

capture_handle = patch_module.register_forward_pre_hook(
    capture_pre_hook
)

outputs = model(
    input_ids=input_ids,
    use_cache=False,
)

metric = (
    outputs.logits[0, -1, TARGET_TOKEN]
    - outputs.logits[0, -1, FOIL_TOKEN]
)

baseline_metric = float(metric.detach())

hidden = captured["hidden"]

if not hidden.requires_grad:
    raise RuntimeError(
        "Captured residual stream does not require gradients."
    )

# Directly request d(metric) / d(hidden).
hidden_grad = torch.autograd.grad(
    outputs=metric,
    inputs=hidden,
    retain_graph=False,
    create_graph=False,
)[0]

capture_handle.remove()

grad = hidden_grad[0, -1, :].detach().clone()

grad_norm = torch.linalg.vector_norm(grad).item()

if grad_norm == 0:
    raise RuntimeError(
        "Gradient norm is zero; cannot construct perturbation direction."
    )

# Use normalized gradient direction so the linear effect is non-zero.
direction = grad / torch.linalg.vector_norm(grad)


# ---------------------------------------------------------------------
# Compare first-order attribution with exact intervention
# ---------------------------------------------------------------------

epsilons = [
    1e-1,
    3e-2,
    1e-2,
    3e-3,
    1e-3,
]

rows = []

for eps in epsilons:

    delta_activation = eps * direction

    # Linear / fast attribution prediction:
    #
    # Δm ≈ ∇m · Δh
    predicted_delta = torch.dot(
        grad,
        delta_activation,
    ).item()

    def perturb_pre_hook(module, inputs):
        hidden_in = inputs[0]

        patched = hidden_in.clone()

        patched[:, -1, :] = (
            patched[:, -1, :]
            + delta_activation
        )

        return (patched,) + tuple(inputs[1:])

    perturb_handle = patch_module.register_forward_pre_hook(
        perturb_pre_hook
    )

    with torch.no_grad():
        perturbed_outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )

        perturbed_metric = (
            perturbed_outputs.logits[0, -1, TARGET_TOKEN]
            - perturbed_outputs.logits[0, -1, FOIL_TOKEN]
        ).item()

    perturb_handle.remove()

    exact_delta = perturbed_metric - baseline_metric

    absolute_error = abs(
        predicted_delta - exact_delta
    )

    relative_error = (
        absolute_error
        / max(abs(exact_delta), 1e-15)
    )

    sign_agreement = (
        np.sign(predicted_delta)
        == np.sign(exact_delta)
    )

    rows.append(
        {
            "epsilon": eps,
            "predicted_delta": predicted_delta,
            "exact_delta": exact_delta,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "sign_agreement": bool(sign_agreement),
        }
    )


# ---------------------------------------------------------------------
# Test expected Taylor convergence
#
# For a correct first-order approximation:
#
# exact_delta = linear_term + O(epsilon^2)
#
# so absolute error should scale approximately as epsilon^2.
# ---------------------------------------------------------------------

fit_rows = rows[:4]

log_eps = np.log(
    [r["epsilon"] for r in fit_rows]
)

log_error = np.log(
    [
        max(r["absolute_error"], 1e-30)
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

smallest_relative_error = rows[-1][
    "relative_error"
]

all_signs_agree = all(
    r["sign_agreement"]
    for r in rows
)

passed = (
    slope > 1.5
    and smallest_relative_error < 0.01
    and all_signs_agree
)


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print(
    "\n=== Phase 0: Attribution vs Exact Intervention ==="
)
print(f"device:             {DEVICE}")
print(f"dtype:              {DTYPE}")
print(f"baseline metric:    {baseline_metric:.10f}")
print(f"gradient norm:      {grad_norm:.10f}")

print()

print(
    f"{'epsilon':>10} "
    f"{'predicted':>16} "
    f"{'exact':>16} "
    f"{'abs error':>16} "
    f"{'rel error':>14} "
    f"{'sign':>8}"
)

for row in rows:
    print(
        f"{row['epsilon']:10.1e} "
        f"{row['predicted_delta']:16.10e} "
        f"{row['exact_delta']:16.10e} "
        f"{row['absolute_error']:16.10e} "
        f"{row['relative_error']:14.6e} "
        f"{str(row['sign_agreement']):>8}"
    )

print()
print(
    f"log-log Taylor error slope: {slope:.4f}"
)
print(
    "smallest-epsilon relative error: "
    f"{smallest_relative_error:.6e}"
)
print(
    f"all signs agree: {all_signs_agree}"
)
print(
    f"PASS: {passed}"
)


# ---------------------------------------------------------------------
# Save machine-readable results
# ---------------------------------------------------------------------

results = {
    "seed": SEED,
    "device": DEVICE,
    "dtype": str(DTYPE),

    "model": {
        "architecture": "tiny random GPT-2",
        "layers": config.n_layer,
        "heads": config.n_head,
        "hidden_size": config.n_embd,
        "vocab_size": config.vocab_size,
    },

    "metric": (
        "target-token logit minus foil-token logit"
    ),

    "patch_site": (
        "residual stream between transformer "
        "blocks 0 and 1, final prompt position"
    ),

    "baseline_metric": baseline_metric,
    "gradient_norm": grad_norm,

    "rows": rows,

    "taylor_error_loglog_slope": slope,

    "smallest_relative_error": (
        smallest_relative_error
    ),

    "all_signs_agree": all_signs_agree,
    "pass": passed,
}

with open(
    RESULTS_DIR / "attribution_vs_exact.json",
    "w",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )

with open(
    RESULTS_DIR / "attribution_vs_exact.csv",
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


if not passed:
    raise SystemExit(
        "Phase 0 check FAILED. "
        "Inspect convergence before proceeding."
    )

print(
    "\nSaved results to "
    "phase0/results/attribution_vs_exact.{json,csv}"
)
