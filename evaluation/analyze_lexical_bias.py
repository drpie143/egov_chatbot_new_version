"""Measure how much the FAQ test set rewards pure lexical matching.

The questions are crawled from the public-service portal FAQ, where many are
phrased by quoting the official procedure title. A question that contains its
own answer's title is trivial for BM25 and says little about how retrieval
behaves on the paraphrased queries real users type. This script splits the test
set by question/title token overlap and reports each retrieval run per bucket,
so the benchmark's bias is visible instead of baked into a single average.

Usage:
    python evaluation/analyze_lexical_bias.py \
        --testset evaluation/testsets/dvc_faq_clean_v2.jsonl \
        --run bm25=evaluation/reports/fixed_bm25_per_sample.jsonl \
        --run hybrid=evaluation/reports/fusion_legacy_per_sample.jsonl \
        --run dense=evaluation/reports/fixed_dense_per_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egov_bot.utils.normalizer import tokenize_bm25  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def title_coverage(question: str, title: str) -> float:
    """Fraction of the expected title's tokens that already appear in the question."""
    q_tokens = set(tokenize_bm25(question))
    t_tokens = set(tokenize_bm25(title))
    if not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(t_tokens)


def metric_from_rank(rank: int | None, metric: str, k: int = 10) -> float:
    if rank is None or rank > k:
        return 0.0
    if metric.startswith("recall@"):
        return 1.0 if rank <= int(metric.split("@")[1]) else 0.0
    if metric == "mrr@10":
        return 1.0 / rank
    if metric == "ndcg@10":
        return 1.0 / math.log2(rank + 1)
    raise ValueError(metric)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Split retrieval results by lexical overlap.")
    parser.add_argument("--testset", default="evaluation/testsets/dvc_faq_clean_v2.jsonl")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Per-sample JSONL of a run, e.g. bm25=evaluation/reports/fixed_bm25_per_sample.jsonl",
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = read_jsonl(Path(args.testset))
    coverage = [
        title_coverage(s["question"], s["expected_procedure_title"]) for s in samples
    ]

    exact = sum(1 for c in coverage if c >= 0.999)
    high = sum(1 for c in coverage if c >= args.threshold)
    print(f"Questions: {len(samples)}")
    print(f"Mean title-token coverage in question: {mean(coverage):.3f}")
    print(f"Questions containing the FULL title    : {exact} ({exact/len(samples):.1%})")
    print(f"Questions with coverage >= {args.threshold:.2f}       : {high} ({high/len(samples):.1%})")
    print()
    print("A high-coverage question already spells out the answer's title, so it")
    print("measures string matching rather than retrieval of a user's real query.")
    print()

    runs: dict[str, list[int | None]] = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        rows = read_jsonl(Path(path))
        if len(rows) != len(samples):
            raise SystemExit(f"Run {name!r} has {len(rows)} rows, testset has {len(samples)}")
        runs[name] = [row.get("rank") for row in rows]

    buckets = {
        f"lexical (coverage >= {args.threshold})": [i for i, c in enumerate(coverage) if c >= args.threshold],
        f"paraphrased (coverage < {args.threshold})": [i for i, c in enumerate(coverage) if c < args.threshold],
    }

    payload = {"threshold": args.threshold, "mean_coverage": round(mean(coverage), 4), "buckets": {}}

    for bucket_name, idxs in buckets.items():
        print(f"--- {bucket_name}  (n={len(idxs)}) ---")
        if not idxs:
            print("  empty\n")
            continue
        print(f"{'run':<10}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'MRR@10':>9}{'nDCG@10':>10}")
        bucket_rows = {}
        for name, ranks in runs.items():
            selected = [ranks[i] for i in idxs]
            values = {
                m: mean([metric_from_rank(r, m) for r in selected])
                for m in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
            }
            bucket_rows[name] = {k: round(v, 4) for k, v in values.items()}
            print(
                f"{name:<10}{values['recall@1']:>9.4f}{values['recall@5']:>9.4f}"
                f"{values['recall@10']:>9.4f}{values['mrr@10']:>9.4f}{values['ndcg@10']:>10.4f}"
            )
        payload["buckets"][bucket_name] = {"n": len(idxs), "runs": bucket_rows}
        print()

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
