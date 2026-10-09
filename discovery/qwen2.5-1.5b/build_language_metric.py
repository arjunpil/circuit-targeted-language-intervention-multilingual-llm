import argparse
import json
import math
from collections import Counter

from transformers import AutoTokenizer

from common import LANG_NAMES, ROOT, default_metric_json, flores


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build the Qwen2.5-1.5B EN<->target token-set metric "
            "from FLORES-200 dev."
        )
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B",
    )
    p.add_argument(
        "--lang",
        default="es",
        help=(
            "Target language code (es, ru, zh, hi, ko, ...). "
            "Always paired against English."
        ),
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
        default=None,
        help=(
            "Output path. Defaults to "
            "discovery/qwen2.5-1.5b/results/en_<lang>_language_metric.json."
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

    target_name = LANG_NAMES.get(args.lang, args.lang)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    en_counts = count_tokens(
        tokenizer,
        flores("en", "dev"),
    )

    target_counts = count_tokens(
        tokenizer,
        flores(args.lang, "dev"),
    )

    rows = []

    for token_id in set(en_counts) | set(target_counts):
        en = en_counts[token_id]
        target = target_counts[token_id]

        if en + target < args.min_count:
            continue

        score = math.log((target + 1) / (en + 1))

        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([token_id]),
                "english_count": int(en),
                "target_count": int(target),
                "score": float(score),
            }
        )

    english = sorted(
        rows,
        key=lambda row: row["score"],
    )[:args.n_language_tokens]

    target = sorted(
        rows,
        key=lambda row: row["score"],
        reverse=True,
    )[:args.n_language_tokens]

    result = {
        "model": args.model,
        "target_lang": args.lang,
        "target_lang_name": target_name,
        "selection": {
            "corpus": "FLORES-200 dev",
            "n_language_tokens": args.n_language_tokens,
            "minimum_total_token_count": args.min_count,
            "score": (
                "log((target_count + 1) / "
                "(english_count + 1))"
            ),
        },
        "english_token_ids": [
            row["token_id"] for row in english
        ],
        "target_token_ids": [
            row["token_id"] for row in target
        ],
        "english_preview": english[:20],
        "target_preview": target[:20],
    }

    out = (
        ROOT / args.out
        if args.out is not None
        else default_metric_json(args.lang)
    )

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
