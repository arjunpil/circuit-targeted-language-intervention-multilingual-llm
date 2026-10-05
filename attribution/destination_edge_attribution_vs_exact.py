import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SEEDS = [42, 123, 999]
DEVICE = "cpu"
DTYPE = torch.float64

D_MODEL = 24
D_HEAD = 8
N_SRC = 3
N_DST = 3
VOCAB = 16

TARGET_TOKEN = 3
FOIL_TOKEN = 7

EPSILONS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
SIGNS = [-1, 1]

MIN_ABS_LINEAR_EFFECT = 1e-6
BROADCAST_ALPHA = 1e-1
DISTINCT_TOL = 1e-8

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EdgeExplicitToy(nn.Module):
    """
    Tiny transformer-like network with explicit source->destination messages.

    Each source node produces an activation s_i.
    Each directed edge i->j carries:
        m_ij = s_i @ W_ij

    Destination j receives only its incoming edge messages:
        z_j = base_j(x) + sum_i m_ij

    This lets us patch exactly one source->destination edge without
    modifying the source activation or its other outgoing edges.
    """

    def __init__(self):
        super().__init__()

        self.src_in = nn.Parameter(
            torch.randn(N_SRC, D_MODEL, D_HEAD) / math.sqrt(D_MODEL)
        )
        self.edge_w = nn.Parameter(
            torch.randn(N_SRC, N_DST, D_HEAD, D_HEAD) / math.sqrt(D_HEAD)
        )
        self.dst_base = nn.Parameter(
            torch.randn(N_DST, D_MODEL, D_HEAD) / math.sqrt(D_MODEL)
        )
        self.dst_out = nn.Parameter(
            torch.randn(N_DST, D_HEAD, D_MODEL) / math.sqrt(D_HEAD)
        )
        self.unembed = nn.Parameter(
            torch.randn(D_MODEL, VOCAB) / math.sqrt(D_MODEL)
        )

    def forward(
        self,
        x,
        edge_patch=None,
        source_patch=None,
        return_internal=False,
    ):
        src = torch.tanh(
            torch.einsum("bd,sdh->bsh", x, self.src_in)
        )

        # Broadcast/source control:
        # changing one source changes every outgoing edge from it.
        if source_patch is not None:
            src_idx, delta_src, alpha = source_patch
            src = src.clone()
            src[:, src_idx, :] = (
                src[:, src_idx, :] + alpha * delta_src
            )

        # Explicit edge messages:
        # [batch, source, destination, head_dim]
        edge_messages = torch.einsum(
            "bsh,sthk->bstk",
            src,
            self.edge_w,
        )

        # True destination-specific edge intervention:
        # modify exactly one s -> t message.
        if edge_patch is not None:
            src_idx, dst_idx, delta_edge, alpha = edge_patch
            edge_messages = edge_messages.clone()
            edge_messages[:, src_idx, dst_idx, :] = (
                edge_messages[:, src_idx, dst_idx, :]
                + alpha * delta_edge
            )

        base = torch.einsum(
            "bd,tdh->bth",
            x,
            self.dst_base,
        )

        dst_pre = base + edge_messages.sum(dim=1)
        dst = torch.tanh(dst_pre)

        dst_projected = torch.einsum(
            "bth,thd->btd",
            dst,
            self.dst_out,
        )

        residual = x + dst_projected.sum(dim=1)
        logits = residual @ self.unembed

        if return_internal:
            return logits, src, edge_messages

        return logits


def metric(logits):
    return (
        logits[0, TARGET_TOKEN]
        - logits[0, FOIL_TOKEN]
    )


def build_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = EdgeExplicitToy().to(
        device=DEVICE,
        dtype=DTYPE,
    )
    model.eval()
    return model


def make_inputs(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 12345)

    clean = torch.randn(
        1,
        D_MODEL,
        generator=generator,
        dtype=DTYPE,
        device=DEVICE,
    )

    corrupted = torch.randn(
        1,
        D_MODEL,
        generator=generator,
        dtype=DTYPE,
        device=DEVICE,
    )

    return clean, corrupted


all_rows = []
curve_rows = []
broadcast_rows = []
seed_summaries = []

for seed in SEEDS:
    model = build_model(seed)
    clean_x, corrupted_x = make_inputs(seed)

    with torch.no_grad():
        (
            clean_logits,
            clean_src,
            clean_edges,
        ) = model(
            clean_x,
            return_internal=True,
        )

    (
        corrupted_logits,
        corrupted_src,
        corrupted_edges,
    ) = model(
        corrupted_x,
        return_internal=True,
    )

    corrupted_metric = metric(corrupted_logits)
    baseline_metric = float(corrupted_metric.detach())

    edge_grad = torch.autograd.grad(
        outputs=corrupted_metric,
        inputs=corrupted_edges,
        retain_graph=False,
        create_graph=False,
    )[0].detach()

    retained_edges = 0
    seed_curves = []

    print()
    print("=" * 72)
    print(f"MODEL SEED {seed}")
    print("=" * 72)
    print(f"corrupted baseline metric: {baseline_metric:.10f}")

    for src_idx in range(N_SRC):
        delta_src = (
            clean_src[0, src_idx]
            - corrupted_src.detach()[0, src_idx]
        ).detach()

        for dst_idx in range(N_DST):
            delta_edge = (
                clean_edges[0, src_idx, dst_idx]
                - corrupted_edges.detach()[0, src_idx, dst_idx]
            ).detach()

            grad_edge = edge_grad[
                0,
                src_idx,
                dst_idx,
            ]

            linear_effect_full = float(
                torch.dot(
                    grad_edge,
                    delta_edge,
                )
            )

            if abs(linear_effect_full) < MIN_ABS_LINEAR_EFFECT:
                continue

            retained_edges += 1

            # ---------------------------------------------------------
            # Control:
            # compare patching one destination-specific edge against
            # changing the source itself, which alters every outgoing edge.
            # ---------------------------------------------------------

            with torch.no_grad():
                edge_control_metric = float(
                    metric(
                        model(
                            corrupted_x,
                            edge_patch=(
                                src_idx,
                                dst_idx,
                                delta_edge,
                                BROADCAST_ALPHA,
                            ),
                        )
                    )
                )

                source_control_metric = float(
                    metric(
                        model(
                            corrupted_x,
                            source_patch=(
                                src_idx,
                                delta_src,
                                BROADCAST_ALPHA,
                            ),
                        )
                    )
                )

            edge_control_delta = (
                edge_control_metric - baseline_metric
            )
            source_control_delta = (
                source_control_metric - baseline_metric
            )

            control_difference = abs(
                edge_control_delta
                - source_control_delta
            )

            broadcast_rows.append(
                {
                    "seed": seed,
                    "source": src_idx,
                    "destination": dst_idx,
                    "alpha": BROADCAST_ALPHA,
                    "destination_specific_delta": edge_control_delta,
                    "source_broadcast_delta": source_control_delta,
                    "absolute_difference": control_difference,
                    "distinct": bool(
                        control_difference > DISTINCT_TOL
                    ),
                }
            )

            # ---------------------------------------------------------
            # Fast edge attribution vs exact edge intervention.
            # ---------------------------------------------------------

            for sign in SIGNS:
                intervention_rows = []

                for eps in EPSILONS:
                    alpha = sign * eps

                    predicted_delta = (
                        alpha * linear_effect_full
                    )

                    with torch.no_grad():
                        patched_logits = model(
                            corrupted_x,
                            edge_patch=(
                                src_idx,
                                dst_idx,
                                delta_edge,
                                alpha,
                            ),
                        )

                        patched_metric = float(
                            metric(patched_logits)
                        )

                    exact_delta = (
                        patched_metric
                        - baseline_metric
                    )

                    absolute_error = abs(
                        predicted_delta
                        - exact_delta
                    )

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
                        "seed": seed,
                        "source": src_idx,
                        "destination": dst_idx,
                        "sign": sign,
                        "epsilon": eps,
                        "alpha": alpha,
                        "linear_effect_full": linear_effect_full,
                        "predicted_delta": predicted_delta,
                        "exact_delta": exact_delta,
                        "absolute_error": absolute_error,
                        "relative_error": relative_error,
                        "sign_agreement": sign_agreement,
                    }

                    all_rows.append(row)
                    intervention_rows.append(row)

                fit_rows = intervention_rows[:4]

                log_eps = np.log(
                    [
                        row["epsilon"]
                        for row in fit_rows
                    ]
                )

                log_error = np.log(
                    [
                        max(
                            row["absolute_error"],
                            1e-30,
                        )
                        for row in fit_rows
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
                    row
                    for row in intervention_rows
                    if row["epsilon"] <= 3e-3
                ]

                small_sign_agreement = all(
                    row["sign_agreement"]
                    for row in small_rows
                )

                curve = {
                    "seed": seed,
                    "source": src_idx,
                    "destination": dst_idx,
                    "sign": sign,
                    "linear_effect_full": linear_effect_full,
                    "taylor_error_slope": slope,
                    "smallest_epsilon": smallest["epsilon"],
                    "smallest_relative_error": smallest[
                        "relative_error"
                    ],
                    "smallest_absolute_error": smallest[
                        "absolute_error"
                    ],
                    "small_epsilon_sign_agreement": bool(
                        small_sign_agreement
                    ),
                }

                curve_rows.append(curve)
                seed_curves.append(curve)

    slopes = np.array(
        [
            row["taylor_error_slope"]
            for row in seed_curves
        ]
    )

    rel_errors = np.array(
        [
            row["smallest_relative_error"]
            for row in seed_curves
        ]
    )

    sign_rate = float(
        np.mean(
            [
                row["small_epsilon_sign_agreement"]
                for row in seed_curves
            ]
        )
    )

    seed_summary = {
        "seed": seed,
        "retained_edges": retained_edges,
        "curves": len(seed_curves),
        "median_slope": float(np.median(slopes)),
        "fraction_slope_gt_1_5": float(
            np.mean(slopes > 1.5)
        ),
        "median_smallest_relative_error": float(
            np.median(rel_errors)
        ),
        "max_smallest_relative_error": float(
            np.max(rel_errors)
        ),
        "small_epsilon_sign_agreement_rate": sign_rate,
    }

    seed_summaries.append(seed_summary)

    print(f"retained edges     : {retained_edges}/{N_SRC * N_DST}")
    print(
        f"median slope       : "
        f"{seed_summary['median_slope']:.4f}"
    )
    print(
        f"slope > 1.5        : "
        f"{seed_summary['fraction_slope_gt_1_5']:.1%}"
    )
    print(
        f"median rel error   : "
        f"{seed_summary['median_smallest_relative_error']:.6e}"
    )
    print(
        f"max rel error      : "
        f"{seed_summary['max_smallest_relative_error']:.6e}"
    )
    print(
        f"small-eps signs    : "
        f"{seed_summary['small_epsilon_sign_agreement_rate']:.1%}"
    )


slopes = np.array(
    [
        row["taylor_error_slope"]
        for row in curve_rows
    ]
)

smallest_rel_errors = np.array(
    [
        row["smallest_relative_error"]
        for row in curve_rows
    ]
)

median_slope = float(np.median(slopes))

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

small_sign_rate = float(
    np.mean(
        [
            row["small_epsilon_sign_agreement"]
            for row in curve_rows
        ]
    )
)

broadcast_distinct_rate = float(
    np.mean(
        [
            row["distinct"]
            for row in broadcast_rows
        ]
    )
)

passed = bool(
    len(curve_rows) >= 40
    and median_slope > 1.5
    and fraction_slope_gt_1_5 >= 0.90
    and median_smallest_rel_error < 0.01
    and p90_smallest_rel_error < 0.05
    and small_sign_rate >= 0.98
    and broadcast_distinct_rate >= 0.95
)

summary = {
    "device": DEVICE,
    "dtype": str(DTYPE),
    "seeds": SEEDS,
    "architecture": {
        "type": "edge-explicit transformer-like toy network",
        "d_model": D_MODEL,
        "d_head": D_HEAD,
        "n_source_nodes": N_SRC,
        "n_destination_nodes": N_DST,
    },
    "metric": "target logit minus foil logit",
    "edge_semantics": (
        "patch one explicit source->destination message only"
    ),
    "epsilons": EPSILONS,
    "signs": SIGNS,
    "minimum_abs_linear_effect": MIN_ABS_LINEAR_EFFECT,
    "n_curves": len(curve_rows),
    "n_exact_edge_interventions": len(all_rows),
    "median_taylor_error_slope": median_slope,
    "fraction_curves_slope_gt_1_5": fraction_slope_gt_1_5,
    "median_smallest_relative_error": median_smallest_rel_error,
    "p90_smallest_relative_error": p90_smallest_rel_error,
    "max_smallest_relative_error": max_smallest_rel_error,
    "small_epsilon_sign_agreement_rate": small_sign_rate,
    "destination_vs_broadcast_distinct_rate": broadcast_distinct_rate,
    "seed_summaries": seed_summaries,
    "pass": passed,
}

print()
print("=" * 72)
print("DESTINATION-SPECIFIC EDGE PHASE 0 SUMMARY")
print("=" * 72)
print(f"seeds tested                         : {len(SEEDS)}")
print(f"signed edge curves                   : {len(curve_rows)}")
print(f"exact destination-edge interventions : {len(all_rows)}")
print()
print(f"median Taylor-error slope             : {median_slope:.4f}")
print(
    f"fraction slopes > 1.5                : "
    f"{fraction_slope_gt_1_5:.1%}"
)
print(
    f"median rel. error @ eps=1e-3         : "
    f"{median_smallest_rel_error:.6e}"
)
print(
    f"90th pct rel. error @ eps=1e-3       : "
    f"{p90_smallest_rel_error:.6e}"
)
print(
    f"max rel. error @ eps=1e-3            : "
    f"{max_smallest_rel_error:.6e}"
)
print(
    f"small-epsilon sign agreement          : "
    f"{small_sign_rate:.1%}"
)
print(
    f"destination vs source-broadcast diff  : "
    f"{broadcast_distinct_rate:.1%}"
)
print()
print(f"PASS: {passed}")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv(
    RESULTS_DIR / "destination_edge_interventions.csv",
    all_rows,
)

write_csv(
    RESULTS_DIR / "destination_edge_curves.csv",
    curve_rows,
)

write_csv(
    RESULTS_DIR / "destination_edge_broadcast_controls.csv",
    broadcast_rows,
)

with open(
    RESULTS_DIR / "destination_edge_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

if not passed:
    raise SystemExit(
        "Destination-specific edge Phase 0 check FAILED. "
        "Inspect individual edges before proceeding."
    )

print()
print("Saved:")
print("  phase0/results/destination_edge_interventions.csv")
print("  phase0/results/destination_edge_curves.csv")
print("  phase0/results/destination_edge_broadcast_controls.csv")
print("  phase0/results/destination_edge_summary.json")
