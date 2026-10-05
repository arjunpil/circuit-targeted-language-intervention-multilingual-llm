import argparse
import json
import math
from collections import Counter

from transformers import AutoTokenizer

from common import ROOT, flores


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build the Llama-3.2-1B EN/ES token-set metric "
            "from FLORES-200 dev."
        )
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
    )
    p.add_argument(
        "--n-language-tokens",
        type=int,
        default=200,
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=5,
    )
    p.add_argument(
        "--out",
        default=(
            "discovery/llama-3.2-1b/results/"
            "en_es_language_metric.json"
        ),
    )
    return p.parse_args()


def count_tokens(tokenizer, texts):
    counts = Counter()

    for text in texts:
        ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        counts.update(ids)

    return counts


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    en_counts = count_tokens(
        tokenizer,
        flores("en", "dev"),
    )

    es_counts = count_tokens(
        tokenizer,
        flores("es", "dev"),
    )

    rows = []

    for token_id in set(en_counts) | set(es_counts):
        en = en_counts[token_id]
        es = es_counts[token_id]

        if en + es < args.min_count:
            continue

        score = math.log((es + 1) / (en + 1))

        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([token_id]),
                "english_count": int(en),
                "spanish_count": int(es),
                "score": float(score),
            }
        )

    english = sorted(
        rows,
        key=lambda row: row["score"],
    )[:args.n_language_tokens]

    spanish = sorted(
        rows,
        key=lambda row: row["score"],
        reverse=True,
    )[:args.n_language_tokens]

    result = {
        "model": args.model,
        "selection": {
            "corpus": "FLORES-200 dev",
            "n_language_tokens": args.n_language_tokens,
            "minimum_total_token_count": args.min_count,
            "score": (
                "log((spanish_count + 1) / "
                "(english_count + 1))"
            ),
        },
        "english_token_ids": [
            row["token_id"] for row in english
        ],
        "spanish_token_ids": [
            row["token_id"] for row in spanish
        ],
        "english_preview": english[:20],
        "spanish_preview": spanish[:20],
    }

    out = ROOT / args.out
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Saved:", out)


if __name__ == "__main__":
    main()
