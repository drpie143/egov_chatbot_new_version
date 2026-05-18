"""Clean raw FAQ data and match expected_procedure_title to corpus titles.

Usage:
    python evaluation/clean_dvc_faq_testset.py \
        --input evaluation/testsets/dvc_faq_raw.jsonl \
        --corpus static/data/toan_bo_du_lieu_final.json \
        --output evaluation/testsets/dvc_faq_clean_full.jsonl \
        --manual-review-output evaluation/testsets/dvc_faq_manual_review.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.utils.jsonl_io import read_jsonl, write_jsonl  # noqa: E402
from evaluation.utils.text_normalize import normalize_text, normalize_title  # noqa: E402
from evaluation.utils.title_matching import (  # noqa: E402
    build_title_map,
    load_corpus_titles,
    match_title,
)

# Minimum lengths for valid samples
MIN_QUESTION_LEN = 15
MIN_ANSWER_LEN = 30
MIN_TITLE_LEN = 10

# Generic/empty answers to reject
REJECT_ANSWERS = [
    "đang cập nhật",
    "không có thông tin",
    "vui lòng xem thủ tục liên quan",
    "đang xử lý",
]


def _is_valid_sample(sample: dict) -> bool:
    """Check if a sample meets minimum quality requirements."""
    question = sample.get("question", "")
    answer = sample.get("reference_answer", "")
    title = sample.get("expected_procedure_title", "")

    if len(question) < MIN_QUESTION_LEN:
        return False
    if len(answer) < MIN_ANSWER_LEN:
        return False
    if len(title) < MIN_TITLE_LEN:
        return False

    answer_lower = answer.lower()
    for pattern in REJECT_ANSWERS:
        if pattern in answer_lower:
            return False

    return True


def _deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicate questions, keeping the one with the longer answer."""
    seen: dict[str, dict] = {}
    for sample in samples:
        key = normalize_title(sample["question"])
        existing = seen.get(key)
        if existing is None:
            seen[key] = sample
        elif len(sample.get("reference_answer", "")) > len(existing.get("reference_answer", "")):
            seen[key] = sample
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw FAQ testset and match titles to corpus.")
    parser.add_argument("--input", required=True, help="Path to raw JSONL file.")
    parser.add_argument("--corpus", default="static/data/toan_bo_du_lieu_final.json", help="Path to corpus JSON.")
    parser.add_argument("--output", required=True, help="Output clean JSONL path.")
    parser.add_argument("--manual-review-output", default=None, help="Output CSV for unmatched samples.")
    parser.add_argument("--fuzzy-threshold", type=float, default=92.0, help="Fuzzy match threshold.")
    args = parser.parse_args()

    # Load raw data
    raw_samples = read_jsonl(args.input)
    print(f"Loaded {len(raw_samples)} raw samples.")

    # Filter valid samples
    valid = [s for s in raw_samples if _is_valid_sample(s)]
    print(f"After validation filter: {len(valid)} samples.")

    # Deduplicate
    valid = _deduplicate(valid)
    print(f"After deduplication: {len(valid)} samples.")

    # Load corpus titles
    corpus_titles = load_corpus_titles(args.corpus)
    title_map = build_title_map(corpus_titles)
    print(f"Loaded {len(corpus_titles)} corpus titles ({len(title_map)} unique normalized).")

    # Match titles
    matched: list[dict] = []
    unmatched: list[dict] = []
    stats = {"exact": 0, "fuzzy": 0, "none": 0}

    for sample in valid:
        expected = sample["expected_procedure_title"]
        matched_title, method, score = match_title(expected, title_map, args.fuzzy_threshold)
        stats[method] += 1

        if matched_title is not None:
            clean_sample = {
                "question": normalize_text(sample["question"]),
                "reference_answer": normalize_text(sample["reference_answer"]),
                "expected_procedure_title": matched_title,
                "match_method": method,
                "match_score": round(score, 2),
            }
            # Preserve debug fields if present
            for field in ["faq_id", "url"]:
                if field in sample:
                    clean_sample[field] = sample[field]
            matched.append(clean_sample)
        else:
            unmatched.append({
                "question": sample["question"],
                "reference_answer": sample["reference_answer"],
                "expected_procedure_title": expected,
                "best_candidate_title": "",
                "fuzzy_score": round(score, 2),
            })

    print("\nTitle matching results:")
    print(f"  Exact:     {stats['exact']}")
    print(f"  Fuzzy:     {stats['fuzzy']}")
    print(f"  Unmatched: {stats['none']}")
    print(f"  Total matched: {len(matched)}")

    # Save clean output
    write_jsonl(args.output, matched)
    print(f"\nWrote {len(matched)} clean samples to {args.output}")

    # Save manual review CSV
    if args.manual_review_output and unmatched:
        review_path = Path(args.manual_review_output)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "reference_answer", "expected_procedure_title",
                "best_candidate_title", "fuzzy_score",
            ])
            writer.writeheader()
            writer.writerows(unmatched)
        print(f"Wrote {len(unmatched)} unmatched samples to {args.manual_review_output}")


if __name__ == "__main__":
    main()
