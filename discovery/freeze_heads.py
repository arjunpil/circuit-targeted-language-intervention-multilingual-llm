import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Freeze a head-discovery run's selected heads into the "
            "compact heads.json format consumed by run_head_validation.py "
            "and the intervention/ harness."
        )
    )

    p.add_argument(
        "--discovery",
        required=True,
        help="Path to a run_head_discovery.py output JSON.",
    )

    p.add_argument(
        "--out",
        required=True,
        help="Path to write the frozen heads.json.",
    )

    p.add_argument(
        "--heldout-examples",
        type=int,
        default=50,
        help=(
            "Recorded alongside the frozen heads as the intended "
            "held-out validation sample size."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    with Path(args.discovery).open(
        "r",
        encoding="utf-8",
    ) as f:
        discovery = json.load(f)

    result = {
        "heads": discovery["heads"],
        "model": discovery["model"],
        "language_pair": discovery["language_pair"],
        "activation_site": discovery["activation_site"],
        "discovery_examples": discovery["n_examples"],
        "heldout_examples": args.heldout_examples,
        "discovery_indices": discovery["example_indices"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Saved:", out)
    print("Heads:", result["heads"])


if __name__ == "__main__":
    main()
