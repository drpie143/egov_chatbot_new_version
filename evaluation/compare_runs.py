"""Paired bootstrap comparison between two retrieval runs.

With 74 evaluation questions a one-question swing moves Recall@10 by 1.35
points, which is the same size as the gaps usually quoted between retrieval
modes. This script resamples the per-question results to say whether a gap
survives that noise, instead of reporting point estimates that cannot support
the comparison.

Usage:
    python evaluation/compare_runs.py \
        --baseline evaluation/reports/baseline_bm25_per_sample.jsonl \
        --candidate evaluation/reports/fixed_bm25_per_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

METRICS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")


def load_ranks(path: Path) -> list[int | None]:
    ranks: list[int | None] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                ranks.append(json.loads(line).get("rank"))
    return ranks


def per_sample_metric(rank: int | None, metric: str, k: int = 10) -> float:
    if rank is None or rank > k:
        return 0.0
    if metric.startswith("recall@"):
        return 1.0 if rank <= int(metric.split("@")[1]) else 0.0
    if metric == "mrr@10":
        return 1.0 / rank
    if metric == "ndcg@10":
        return 1.0 / math.log2(rank + 1)
    raise ValueError(f"Unknown metric: {metric}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap(
    baseline: list[float],
    candidate: list[float],
    rounds: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return (mean difference, ci_low, ci_high) for candidate - baseline."""
    rng = random.Random(seed)
    n = len(baseline)
    diffs: list[float] = []
    indices = range(n)
    for _ in range(rounds):
        sample = [rng.choice(indices) for _ in indices]
        diffs.append(
            mean([candidate[i] for i in sample]) - mean([baseline[i] for i in sample])
        )
    diffs.sort()
    low = diffs[int(0.025 * rounds)]
    high = diffs[min(int(0.975 * rounds), rounds - 1)]
    return mean(candidate) - mean(baseline), low, high


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison of two runs.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--testset",
        default=None,
        help="Required with --coverage-below/--coverage-at-least, to slice by lexical overlap.",
    )
    parser.add_argument("--coverage-below", type=float, default=None)
    parser.add_argument("--coverage-at-least", type=float, default=None)
    args = parser.parse_args()

    base_ranks = load_ranks(Path(args.baseline))
    cand_ranks = load_ranks(Path(args.candidate))
    if len(base_ranks) != len(cand_ranks):
        raise SystemExit(
            f"Runs cover different question counts: {len(base_ranks)} vs {len(cand_ranks)}"
        )

    if args.coverage_below is not None or args.coverage_at_least is not None:
        if not args.testset:
            raise SystemExit("--testset is required when slicing by coverage.")
        from analyze_lexical_bias import read_jsonl, title_coverage

        samples = read_jsonl(Path(args.testset))
        if len(samples) != len(base_ranks):
            raise SystemExit(
                f"Testset has {len(samples)} rows but runs have {len(base_ranks)}"
            )
        keep = []
        for i, sample in enumerate(samples):
            cov = title_coverage(sample["question"], sample["expected_procedure_title"])
            if args.coverage_below is not None and cov >= args.coverage_below:
                continue
            if args.coverage_at_least is not None and cov < args.coverage_at_least:
                continue
            keep.append(i)
        base_ranks = [base_ranks[i] for i in keep]
        cand_ranks = [cand_ranks[i] for i in keep]
        print(f"Sliced to {len(keep)} of {len(samples)} questions by title-token coverage.")

    n = len(base_ranks)
    if n == 0:
        raise SystemExit("No questions left after slicing.")
    print(f"Questions: {n}")
    print(f"Bootstrap rounds: {args.rounds:,}\n")
    print(f"{'metric':<12}{'baseline':>10}{'candidate':>11}{'delta':>9}{'95% CI':>20}  verdict")
    print("-" * 74)

    rows = []
    for metric in METRICS:
        base = [per_sample_metric(r, metric) for r in base_ranks]
        cand = [per_sample_metric(r, metric) for r in cand_ranks]
        delta, low, high = paired_bootstrap(base, cand, args.rounds, args.seed)
        significant = low > 0 or high < 0
        verdict = "significant" if significant else "within noise"
        print(
            f"{metric:<12}{mean(base):>10.4f}{mean(cand):>11.4f}{delta:>+9.4f}"
            f"{f'[{low:+.4f}, {high:+.4f}]':>20}  {verdict}"
        )
        rows.append(
            {
                "metric": metric,
                "baseline": round(mean(base), 4),
                "candidate": round(mean(cand), 4),
                "delta": round(delta, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "significant": significant,
            }
        )

    print(
        "\nA CI that spans zero means this test set cannot tell the two runs apart; "
        "it is not evidence that they are equal."
    )

    if args.output:
        payload = {
            "baseline": args.baseline,
            "candidate": args.candidate,
            "questions": n,
            "bootstrap_rounds": args.rounds,
            "seed": args.seed,
            "results": rows,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
