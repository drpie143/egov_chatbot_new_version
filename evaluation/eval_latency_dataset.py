"""Latency benchmark on FAQ testset.

Measures end-to-end API latency percentiles across FAQ test questions.

Usage:
    python evaluation/eval_latency_dataset.py \
        --base-url http://localhost:7860 \
        --testset evaluation/testsets/dvc_faq_qa_500.jsonl \
        --output evaluation/reports/faq_latency_metrics.json \
        --limit 100
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.utils.jsonl_io import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency benchmark on FAQ testset.")
    parser.add_argument("--base-url", default="http://localhost:7860")
    parser.add_argument("--testset", default="evaluation/testsets/dvc_faq_qa_500.jsonl")
    parser.add_argument("--output", default="evaluation/reports/faq_latency_metrics.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warm-up", type=int, default=2, help="Number of warm-up requests before measuring.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    samples = read_jsonl(args.testset, limit=args.limit)
    print(f"Loaded {len(samples)} test samples (limit={args.limit}).")
    print(f"Base URL: {base_url}")

    # Warm-up requests
    if args.warm_up > 0:
        print(f"Running {args.warm_up} warm-up request(s)...")
        for i in range(min(args.warm_up, len(samples))):
            try:
                requests.post(
                    f"{base_url}/chat",
                    json={"question": samples[i]["question"], "session_id": f"warmup-{i}"},
                    timeout=180,
                )
            except Exception:
                pass

    # Benchmark
    latencies: list[float] = []
    errors = 0
    print("\nRunning latency benchmark...")

    for i, sample in enumerate(samples):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={"question": sample["question"], "session_id": f"eval-latency-{i}"},
                timeout=180,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                latencies.append(latency_ms)
            else:
                errors += 1
        except Exception:
            errors += 1

        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{len(samples)} requests...")

    if not latencies:
        print("No successful requests. Cannot compute latency metrics.", file=sys.stderr)
        raise SystemExit(1)

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    metrics = {
        "count": n,
        "errors": errors,
        "success_rate": round(n / max(1, n + errors), 4),
        "latency_min_ms": round(sorted_lat[0], 1),
        "latency_mean_ms": round(statistics.mean(sorted_lat), 1),
        "latency_p50_ms": round(sorted_lat[n // 2], 1),
        "latency_p90_ms": round(sorted_lat[int(0.90 * n)], 1) if n > 1 else round(sorted_lat[0], 1),
        "latency_p95_ms": round(sorted_lat[int(0.95 * n)], 1) if n > 1 else round(sorted_lat[0], 1),
        "latency_p99_ms": round(sorted_lat[int(0.99 * n)], 1) if n > 1 else round(sorted_lat[0], 1),
        "latency_max_ms": round(sorted_lat[-1], 1),
        "run_type": "warm",
    }

    write_json(args.output, metrics)

    print(f"\n{'='*50}")
    print("Latency Benchmark Results")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
