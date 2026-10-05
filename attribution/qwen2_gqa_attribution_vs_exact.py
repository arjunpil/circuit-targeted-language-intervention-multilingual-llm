import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


def rmsnorm_native_dtype(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return self.weight * (
        x * torch.rsqrt(variance + self.variance_epsilon)
    )


Qwen2RMSNorm.forward = rmsnorm_native_dtype

DEVICE = "cpu"
DTYPE = torch.float64

SEEDS = [42, 123, 999]

# Slightly stronger local sweep to avoid measuring ~1e-10 effects
EPSILONS = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3]
SIGNS = [-1, 1]

N_QUERY_HEADS = 4
N_KV_HEADS = 2
HIDDEN_SIZE = 32
HEAD_DIM = HIDDEN_SIZE // N_QUERY_HEADS
N_LAYERS = 2
VOCAB_SIZE = 64
LAYER_IDX = 0

SOURCE_POSITIONS = [1, 3, 5]

TARGET_TOKEN = 7
FOIL_TOKEN = 11

# Cases below this are reported but excluded from convergence pass/fail
SIGNAL_FLOOR = 0.0

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_IDS = torch.tensor(
    [[1, 5, 9, 13, 17, 21, 25, 29]],
    dtype=torch.long,
    device=DEVICE,
)

CORRUPTED_IDS = torch.tensor(
    [[1, 6, 10, 14, 18, 22, 26, 30]],
    dtype=torch.long,
    device=DEVICE,
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
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_QUERY_HEADS,
        num_key_value_heads=N_KV_HEADS,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        _attn_implementation="sdpa",
    )

    model = Qwen2ForCausalLM(config).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    model.eval()

    return model, config


def get_projection(model, projection_name):
    return getattr(
        model.model.layers[LAYER_IDX].self_attn,
        projection_name,
    )


def forward_and_capture(model, input_ids, projection_name):
    module = get_projection(model, projection_name)
    captured = {}

    def hook(module, inputs, output):
        captured["output"] = output

    handle = module.register_forward_hook(hook)

    try:
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )
    finally:
        handle.remove()

    return outputs, captured["output"]


def exact_patch_metric(
    model,
    projection_name,
    kv_head,
    source_position,
    delta,
    alpha,
):
    module = get_projection(model, projection_name)

    start = kv_head * HEAD_DIM
    end = start + HEAD_DIM

    def hook(module, inputs, output):
        patched = output.clone()
        patched[:, source_position, start:end] = (
            patched[:, source_position, start:end]
            + alpha * delta
        )
        return patched

    handle = module.register_forward_hook(hook)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=CORRUPTED_IDS,
                use_cache=False,
            )
            value = float(metric(outputs.logits))
    finally:
        handle.remove()

    return value


assert N_QUERY_HEADS % N_KV_HEADS == 0
QUERY_HEADS_PER_KV = N_QUERY_HEADS // N_KV_HEADS
QUERY_TO_KV = [
    q // QUERY_HEADS_PER_KV
    for q in range(N_QUERY_HEADS)
]

print("=" * 72)
print("TINY QWEN2 GQA CONFIGURATION")
print("=" * 72)
print(f"query heads       : {N_QUERY_HEADS}")
print(f"KV heads          : {N_KV_HEADS}")
print(f"query heads / KV  : {QUERY_HEADS_PER_KV}")
print(f"query -> KV map   : {QUERY_TO_KV}")

all_rows = []
curve_rows = []

for seed in SEEDS:
    model, config = build_model(seed)

    print()
    print("=" * 72)
    print(f"MODEL SEED {seed}")
    print("=" * 72)

    for projection_name in ["k_proj", "v_proj"]:
        with torch.no_grad():
            _, clean_projection = forward_and_capture(
                model,
                CLEAN_IDS,
                projection_name,
            )
            clean_projection = clean_projection.detach().clone()

        model.zero_grad(set_to_none=True)

        corrupted_outputs, corrupted_projection = forward_and_capture(
            model,
            CORRUPTED_IDS,
            projection_name,
        )

        corrupted_metric = metric(corrupted_outputs.logits)
        baseline_metric = float(corrupted_metric.detach())

        projection_grad = torch.autograd.grad(
            outputs=corrupted_metric,
            inputs=corrupted_projection,
            retain_graph=False,
            create_graph=False,
        )[0].detach()

        retained = 0
        below_floor = 0

        for kv_head in range(N_KV_HEADS):
            start = kv_head * HEAD_DIM
            end = start + HEAD_DIM

            for source_position in SOURCE_POSITIONS:
                delta = (
                    clean_projection[
                        0,
                        source_position,
                        start:end,
                    ]
                    - corrupted_projection.detach()[
                        0,
                        source_position,
                        start:end,
                    ]
                ).detach()

                grad_slice = projection_grad[
                    0,
                    source_position,
                    start:end,
                ]

                linear_effect_full = float(
                    torch.dot(grad_slice, delta)
                )

                meaningful = (
                    abs(linear_effect_full) >= SIGNAL_FLOOR
                )

                if meaningful:
                    retained += 1
                else:
                    below_floor += 1

                for sign in SIGNS:
                    intervention_rows = []

                    for eps in EPSILONS:
                        alpha = sign * eps
                        predicted_delta = (
                            alpha * linear_effect_full
                        )

                        patched_metric = exact_patch_metric(
                            model=model,
                            projection_name=projection_name,
                            kv_head=kv_head,
                            source_position=source_position,
                            delta=delta,
                            alpha=alpha,
                        )

                        exact_delta = (
                            patched_metric - baseline_metric
                        )

                        absolute_error = abs(
                            predicted_delta - exact_delta
                        )

                        relative_error = (
                            absolute_error
                            / max(abs(exact_delta), 1e-15)
                        )

                        sign_agreement = bool(
                            np.sign(predicted_delta)
                            == np.sign(exact_delta)
                        )

                        row = {
                            "seed": seed,
                            "projection": projection_name,
                            "kv_head": kv_head,
                            "source_position": source_position,
                            "sign": sign,
                            "epsilon": eps,
                            "linear_effect_full": linear_effect_full,
                            "meaningful_signal": meaningful,
                            "predicted_delta": predicted_delta,
                            "exact_delta": exact_delta,
                            "absolute_error": absolute_error,
                            "relative_error": relative_error,
                            "sign_agreement": sign_agreement,
                        }

                        all_rows.append(row)
                        intervention_rows.append(row)

                    log_eps = np.log([
                        r["epsilon"]
                        for r in intervention_rows[:4]
                    ])

                    log_error = np.log([
                        max(r["absolute_error"], 1e-30)
                        for r in intervention_rows[:4]
                    ])

                    slope = float(
                        np.polyfit(
                            log_eps,
                            log_error,
                            1,
                        )[0]
                    )

                    smallest = intervention_rows[-1]

                    curve_rows.append(
                        {
                            "seed": seed,
                            "projection": projection_name,
                            "kv_head": kv_head,
                            "source_position": source_position,
                            "sign": sign,
                            "linear_effect_full": linear_effect_full,
                            "meaningful_signal": meaningful,
                            "taylor_error_slope": slope,
                            "smallest_relative_error":
                                smallest["relative_error"],
                            "smallest_sign_agreement":
                                smallest["sign_agreement"],
                        }
                    )

        print(
            f"{projection_name}: meaningful={retained}, "
            f"below_signal_floor={below_floor}"
        )

    del model


def summarize_projection(name):
    rows = [
        r for r in curve_rows
        if r["projection"] == name
        and r["meaningful_signal"]
    ]

    slopes = np.array([
        r["taylor_error_slope"]
        for r in rows
    ])

    rel = np.array([
        r["smallest_relative_error"]
        for r in rows
    ])

    sign_rate = np.mean([
        r["smallest_sign_agreement"]
        for r in rows
    ])

    return {
        "n_meaningful_curves": len(rows),
        "median_slope": float(np.median(slopes)),
        "fraction_slope_gt_1_5":
            float(np.mean(slopes > 1.5)),
        "median_relative_error":
            float(np.median(rel)),
        "p90_relative_error":
            float(np.percentile(rel, 90)),
        "max_relative_error":
            float(np.max(rel)),
        "sign_agreement_rate":
            float(sign_rate),
    }


k_summary = summarize_projection("k_proj")
v_summary = summarize_projection("v_proj")

gqa_structure_ok = (
    QUERY_HEADS_PER_KV == 2
    and QUERY_TO_KV == [0, 0, 1, 1]
)

# Separate criteria
k_pass = (
    k_summary["n_meaningful_curves"] >= 4
    and k_summary["median_slope"] > 1.5
    and k_summary["median_relative_error"] < 0.05
    and k_summary["sign_agreement_rate"] >= 0.95
)

v_pass = (
    v_summary["n_meaningful_curves"] >= 10
    and v_summary["median_slope"] > 1.5
    and v_summary["median_relative_error"] < 0.01
    and v_summary["sign_agreement_rate"] >= 0.98
)

passed = bool(
    gqa_structure_ok
    and k_pass
    and v_pass
)

summary = {
    "signal_floor": SIGNAL_FLOOR,
    "epsilons": EPSILONS,
    "query_heads": N_QUERY_HEADS,
    "kv_heads": N_KV_HEADS,
    "query_to_kv": QUERY_TO_KV,
    "gqa_structure_ok": gqa_structure_ok,
    "k_proj": k_summary,
    "v_proj": v_summary,
    "k_pass": k_pass,
    "v_pass": v_pass,
    "pass": passed,
}

print()
print("=" * 72)
print("REVISED TINY QWEN2 GQA SUMMARY")
print("=" * 72)

print("\nK projection:")
for k, v in k_summary.items():
    print(f"  {k}: {v}")

print("\nV projection:")
for k, v in v_summary.items():
    print(f"  {k}: {v}")

print()
print(f"GQA structure OK: {gqa_structure_ok}")
print(f"K PASS: {k_pass}")
print(f"V PASS: {v_pass}")
print(f"OVERALL PASS: {passed}")

with open(
    RESULTS_DIR / "qwen2_gqa_revised_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

with open(
    RESULTS_DIR / "qwen2_gqa_revised_curves.csv",
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
    RESULTS_DIR / "qwen2_gqa_revised_interventions.csv",
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=all_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(all_rows)

if not passed:
    raise SystemExit(
        "Revised GQA check FAILED."
    )
