"""Export a final 3-field FAQ evaluation testset from the clean full JSONL.

Usage:
    python evaluation/export_faq_eval_testset.py \
        --input evaluation/testsets/dvc_faq_clean_full.jsonl \
        --output evaluation/testsets/dvc_faq_qa_500.jsonl \
        --limit 500 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.utils.jsonl_io import read_jsonl, write_jsonl  # noqa: E402

REQUIRED_KEYS = {"question", "reference_answer", "expected_procedure_title"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final 3-field FAQ testset.")
    parser.add_argument("--input", required=True, help="Path to clean full JSONL.")
    parser.add_argument("--output", required=True, help="Output testset JSONL path.")
    parser.add_argument("--limit", type=int, default=500, help="Max number of samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    print(f"Loaded {len(samples)} clean samples.")

    # Shuffle and limit
    random.seed(args.seed)
    random.shuffle(samples)
    samples = samples[:args.limit]

    # Strip to only 3 fields
    final: list[dict] = []
    for sample in samples:
        final.append({
            "question": sample["question"],
            "reference_answer": sample["reference_answer"],
            "expected_procedure_title": sample["expected_procedure_title"],
        })

    write_jsonl(args.output, final)
    print(f"Wrote {len(final)} samples to {args.output}")
    print("Fields per sample: question, reference_answer, expected_procedure_title")


if __name__ == "__main__":
    main()
