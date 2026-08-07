"""Measure how much of each query is administrative boilerplate.

Vietnamese public-service questions are usually wrapped in fixed politeness and
request phrasing ("Xin hỏi...", "Đề nghị cho biết...", "Trình tự thực hiện thủ
tục hành chính..."). Those tokens carry no retrieval signal but they do carry
weight, in BM25 term counts and in the dense embedding alike. This reports how
much of the average query is boilerplate before any rewriting is attempted.

Usage:
    python evaluation/analyze_query_boilerplate.py \
        --testset evaluation/testsets/dvc_faq_clean_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from egov_bot.utils.normalizer import tokenize_bm25  # noqa: E402

# Lead-in phrases that request information without describing it.
BOILERPLATE_PREFIXES = [
    r"đề nghị cho biết",
    r"xin (?:cho )?hỏi",
    r"cho (?:tôi )?hỏi",
    r"xin cho biết",
    r"cho biết",
    r"tôi muốn hỏi",
    r"tôi muốn biết",
    r"vui lòng cho biết",
    r"trình tự thực hiện thủ tục hành chính",
    r"trình tự thực hiện thủ tục",
    r"trình tự thực hiện",
    r"thành phần hồ sơ (?:của )?thủ tục",
    r"cơ quan (?:nào )?có thẩm quyền",
    r"thủ tục hành chính",
]

TRAILING_NOISE = [
    r"là gì\s*\??$",
    r"như thế nào\s*\??$",
    r"ra sao\s*\??$",
    r"được không\s*\??$",
    r"\?+$",
]

PREFIX_RE = re.compile("|".join(BOILERPLATE_PREFIXES), re.IGNORECASE)
TRAILING_RE = re.compile("|".join(TRAILING_NOISE), re.IGNORECASE)


def strip_boilerplate(question: str) -> str:
    text = question.strip()
    # Strip repeatedly: questions often stack two lead-ins.
    for _ in range(3):
        stripped = PREFIX_RE.sub(" ", text, count=1).strip(" ,:;.-")
        if stripped == text:
            break
        text = stripped
    text = TRAILING_RE.sub(" ", text).strip(" ,:;.-")
    return text or question.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify boilerplate in evaluation queries.")
    parser.add_argument("--testset", default="evaluation/testsets/dvc_faq_clean_v2.jsonl")
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.testset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    before_lens, after_lens, changed = [], [], 0
    hit_counter: Counter[str] = Counter()
    examples = []

    for row in rows:
        question = row["question"]
        stripped = strip_boilerplate(question)
        n_before = len(tokenize_bm25(question))
        n_after = len(tokenize_bm25(stripped))
        before_lens.append(n_before)
        after_lens.append(n_after)
        if stripped != question.strip():
            changed += 1
            match = PREFIX_RE.search(question)
            if match:
                hit_counter[match.group(0).lower()] += 1
            if len(examples) < args.show:
                examples.append((question, stripped))

    n = len(rows)
    mean_before = sum(before_lens) / n
    mean_after = sum(after_lens) / n

    print(f"Questions: {n}")
    print(f"Queries containing boilerplate : {changed} ({changed / n:.1%})")
    print(f"Mean query length (tokens)     : {mean_before:.2f} -> {mean_after:.2f}")
    print(f"Mean tokens removed            : {mean_before - mean_after:.2f} ({1 - mean_after / mean_before:.1%})")

    print("\nMost common lead-ins:")
    for phrase, count in hit_counter.most_common(10):
        print(f"  {count:>4}  {phrase!r}")

    print("\nExamples:")
    for original, stripped in examples:
        print(f"  before: {original[:100]}")
        print(f"  after : {stripped[:100]}")
        print()

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "questions": n,
                    "with_boilerplate": changed,
                    "mean_tokens_before": round(mean_before, 2),
                    "mean_tokens_after": round(mean_after, 2),
                    "top_prefixes": hit_counter.most_common(15),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
