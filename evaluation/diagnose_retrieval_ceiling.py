"""Locate where retrieval loses the correct procedure.

Answers three questions before any architectural change is chosen:

1. How much headroom does a reranker have? Compare Recall@10 with Recall@100.
   A reranker can only reorder what candidate generation already returned, so
   the gap between the two is its hard ceiling.

2. How much of the top-k is wasted on duplicates? Chunks are scored
   individually, so several chunks of one procedure can occupy several slots
   and push other procedures out.

3. What does aggregating chunk scores up to the parent procedure recover?
   This is the cheap version of parent-document retrieval.

Usage:
    python evaluation/diagnose_retrieval_ceiling.py \
        --testset evaluation/testsets/dvc_faq_clean_v2.jsonl --mode bm25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from evaluation.analyze_lexical_bias import title_coverage  # noqa: E402
from evaluation.utils.jsonl_io import read_jsonl  # noqa: E402
from evaluation.utils.text_normalize import normalize_title  # noqa: E402

from egov_bot.config import load_settings  # noqa: E402
from egov_bot.data.resource_loader import load_resources  # noqa: E402
from egov_bot.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402

DEPTHS = (1, 3, 5, 10, 20, 50, 100)


def first_hit_rank(results, expected: str) -> int | None:
    target = normalize_title(expected)
    for position, result in enumerate(results, start=1):
        if normalize_title(result.title) == target:
            return position
    return None


def collapse_to_procedures(results):
    """Keep the best-scoring chunk per parent procedure, preserving order."""
    seen: set[str] = set()
    collapsed = []
    for result in results:
        key = normalize_title(result.title) or result.parent_id
        if key in seen:
            continue
        seen.add(key)
        collapsed.append(result)
    return collapsed


def recall_table(ranks: list[int | None], n: int) -> dict[int, float]:
    return {
        depth: sum(1 for r in ranks if r is not None and r <= depth) / max(1, n)
        for depth in DEPTHS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose retrieval loss before choosing a fix.")
    parser.add_argument("--testset", default="evaluation/testsets/dvc_faq_clean_v2.jsonl")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    settings = load_settings()
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
    n = len(samples)
    print(f"Test set: {args.testset}  (n={n}, mode={args.mode}, depth={args.depth})\n")

    chunk_ranks: list[int | None] = []
    proc_ranks: list[int | None] = []
    unique_in_10: list[int] = []
    paraphrased_flags: list[bool] = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        expected = sample["expected_procedure_title"]
        paraphrased_flags.append(title_coverage(question, expected) < args.coverage_threshold)

        results = retriever.retrieve(question, top_k=args.depth, mode=args.mode)
        chunk_ranks.append(first_hit_rank(results, expected))

        collapsed = collapse_to_procedures(results)
        proc_ranks.append(first_hit_rank(collapsed, expected))

        titles_in_10 = {normalize_title(r.title) for r in results[:10]}
        unique_in_10.append(len(titles_in_10))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}...")

    chunk_recall = recall_table(chunk_ranks, n)
    proc_recall = recall_table(proc_ranks, n)

    print("\n=== 1. Reranker headroom (chunk-level ranking, as shipped) ===")
    print(f"{'depth':>7}{'recall':>10}")
    for depth in DEPTHS:
        print(f"{depth:>7}{chunk_recall[depth]:>10.4f}")
    headroom = chunk_recall[args.depth] - chunk_recall[10]
    print(f"\n  Recall@10  = {chunk_recall[10]:.4f}")
    print(f"  Recall@{args.depth} = {chunk_recall[args.depth]:.4f}")
    print(f"  -> A perfect reranker over {args.depth} candidates could add at most {headroom:+.4f}")
    print(f"  -> {1 - chunk_recall[args.depth]:.1%} of questions never retrieve the right procedure at all;")
    print("     no reranker can recover those.")

    print("\n=== 2. Duplicate crowding in top-10 ===")
    mean_unique = sum(unique_in_10) / max(1, len(unique_in_10))
    print(f"  Distinct procedures per top-10: {mean_unique:.2f} of 10")
    print(f"  -> {10 - mean_unique:.2f} slots per query are spent re-showing a procedure already listed.")

    print("\n=== 3. After collapsing chunks to their parent procedure ===")
    print(f"{'depth':>7}{'chunk':>10}{'parent':>10}{'delta':>9}")
    for depth in DEPTHS:
        delta = proc_recall[depth] - chunk_recall[depth]
        print(f"{depth:>7}{chunk_recall[depth]:>10.4f}{proc_recall[depth]:>10.4f}{delta:>+9.4f}")

    para_idx = [i for i, flag in enumerate(paraphrased_flags) if flag]
    if para_idx:
        print(f"\n=== 4. Paraphrased questions only (n={len(para_idx)}) ===")
        para_chunk = recall_table([chunk_ranks[i] for i in para_idx], len(para_idx))
        para_proc = recall_table([proc_ranks[i] for i in para_idx], len(para_idx))
        print(f"{'depth':>7}{'chunk':>10}{'parent':>10}{'delta':>9}")
        for depth in DEPTHS:
            delta = para_proc[depth] - para_chunk[depth]
            print(f"{depth:>7}{para_chunk[depth]:>10.4f}{para_proc[depth]:>10.4f}{delta:>+9.4f}")
        print(f"\n  Ceiling on paraphrased queries: Recall@{args.depth} = {para_chunk[args.depth]:.4f}")

    if args.output:
        payload = {
            "testset": args.testset,
            "mode": args.mode,
            "n": n,
            "depth": args.depth,
            "chunk_recall": {str(k): round(v, 4) for k, v in chunk_recall.items()},
            "parent_recall": {str(k): round(v, 4) for k, v in proc_recall.items()},
            "mean_unique_in_top10": round(mean_unique, 2),
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
