"""Retrieval evaluation using procedure title matching.

Measures Recall@k, MRR@k, nDCG@k by checking if the expected procedure title
appears in the top-k retrieval results.

Usage:
    python evaluation/eval_retrieval_title.py \
        --testset evaluation/testsets/dvc_faq_qa_500.jsonl \
        --output evaluation/reports/faq_retrieval_metrics.json \
        --per-sample-output evaluation/reports/faq_retrieval_per_sample.jsonl \
        --k 10 \
        --mode hybrid
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluation.utils.jsonl_io import read_jsonl, write_json, write_jsonl  # noqa: E402
from evaluation.utils.text_normalize import normalize_title  # noqa: E402

from egov_bot.config import load_settings  # noqa: E402
from egov_bot.data.resource_loader import load_resources  # noqa: E402
from egov_bot.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


def get_result_title(result) -> str:
    """Extract the procedure title from a retrieval result."""
    for attr in ["title", "procedure_title"]:
        if hasattr(result, attr) and getattr(result, attr):
            return str(getattr(result, attr))

    metadata = getattr(result, "metadata", {}) or {}
    for key in ["ten_thu_tuc", "title", "procedure_title", "name"]:
        if metadata.get(key):
            return str(metadata[key])

    return ""


def title_match(predicted_title: str, expected_title: str) -> bool:
    """Check if two titles match after normalization."""
    return normalize_title(predicted_title) == normalize_title(expected_title)


def evaluate(
    samples: list[dict],
    retriever: HybridRetriever,
    k: int,
    mode: str,
) -> tuple[dict, list[dict]]:
    """Run retrieval evaluation and return (metrics, per_sample_results)."""
    per_sample: list[dict] = []
    ranks: list[int | None] = []
    latencies: list[float] = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        expected = sample["expected_procedure_title"]

        start = time.perf_counter()
        results = retriever.retrieve(question, top_k=k, mode=mode)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        # Find rank of first matching result
        rank = None
        top_titles: list[str] = []
        for idx, result in enumerate(results):
            result_title = get_result_title(result)
            top_titles.append(result_title)
            if rank is None and title_match(result_title, expected):
                rank = idx + 1  # 1-indexed

        ranks.append(rank)
        per_sample.append({
            "question": question,
            "expected_procedure_title": expected,
            "rank": rank,
            "hit@1": rank is not None and rank <= 1,
            "hit@3": rank is not None and rank <= 3,
            "hit@5": rank is not None and rank <= 5,
            "hit@10": rank is not None and rank <= 10,
            "top_titles": top_titles[:5],
            "latency_ms": round(latency_ms, 1),
        })

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(samples)} samples...")

    # Compute aggregate metrics
    n = len(samples)
    recall_at = {}
    for cutoff in [1, 3, 5, 10]:
        recall_at[cutoff] = sum(1 for r in ranks if r is not None and r <= cutoff) / max(1, n)

    mrr = sum((1.0 / r if r is not None and r <= k else 0.0) for r in ranks) / max(1, n)
    ndcg = sum((1.0 / math.log2(r + 1) if r is not None and r <= k else 0.0) for r in ranks) / max(1, n)

    sorted_latencies = sorted(latencies)
    metrics = {
        "mode": mode,
        "k": k,
        "count": n,
        "recall@1": round(recall_at.get(1, 0), 4),
        "recall@3": round(recall_at.get(3, 0), 4),
        "recall@5": round(recall_at.get(5, 0), 4),
        "recall@10": round(recall_at.get(10, 0), 4),
        "mrr@10": round(mrr, 4),
        "ndcg@10": round(ndcg, 4),
        "latency_p50_ms": round(sorted_latencies[len(sorted_latencies) // 2], 1) if sorted_latencies else 0,
        "latency_p95_ms": round(sorted_latencies[int(0.95 * len(sorted_latencies))] if sorted_latencies else 0, 1),
    }

    return metrics, per_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation with title matching.")
    parser.add_argument("--testset", default="evaluation/testsets/dvc_faq_qa_500.jsonl")
    parser.add_argument("--output", default="evaluation/reports/faq_retrieval_metrics.json")
    parser.add_argument("--per-sample-output", default="evaluation/reports/faq_retrieval_per_sample.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    settings = load_settings()
    print("Loading resources...")
    resources = load_resources(settings, load_models=True)
    retriever = HybridRetriever(
        settings,
        resources.procedure_store,
        metadatas=resources.metadatas,
        faiss_index=resources.faiss_index,
        bm25=resources.bm25,
        embedding_model=resources.embedding_model,
    )

    samples = read_jsonl(args.testset)
    print(f"Loaded {len(samples)} test samples.")
    print(f"Mode: {args.mode}, k: {args.k}")
    print()

    metrics, per_sample = evaluate(samples, retriever, args.k, args.mode)

    write_json(args.output, metrics)
    write_jsonl(args.per_sample_output, per_sample)

    print(f"\n{'='*50}")
    print(f"Retrieval Evaluation Results (mode={args.mode})")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved metrics to {args.output}")
    print(f"Saved per-sample results to {args.per_sample_output}")


if __name__ == "__main__":
    main()
