"""Validate a final FAQ testset for correctness.

Usage:
    python evaluation/validate_faq_testset.py \
        --testset evaluation/testsets/dvc_faq_qa_500.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_KEYS = {"question", "reference_answer", "expected_procedure_title"}
MIN_QUESTION_LEN = 15
MIN_ANSWER_LEN = 30
MIN_TITLE_LEN = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FAQ testset integrity.")
    parser.add_argument("--testset", required=True, help="Path to testset JSONL.")
    args = parser.parse_args()

    path = Path(args.testset)
    if not path.exists():
        print(f"FAIL: File not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    questions: set[str] = set()
    valid_count = 0

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Check for empty lines
        if not stripped:
            errors.append(f"Line {line_num}: empty line")
            continue

        # Check JSON parse
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_num}: invalid JSON: {exc}")
            continue

        # Check exactly 3 keys
        keys = set(row.keys())
        if keys != REQUIRED_KEYS:
            extra = keys - REQUIRED_KEYS
            missing = REQUIRED_KEYS - keys
            msg = f"Line {line_num}: wrong keys."
            if extra:
                msg += f" Extra: {extra}."
            if missing:
                msg += f" Missing: {missing}."
            errors.append(msg)

        # Check missing values
        for key in REQUIRED_KEYS:
            if not row.get(key):
                errors.append(f"Line {line_num}: empty value for '{key}'")

        # Check minimum lengths
        q = row.get("question", "")
        a = row.get("reference_answer", "")
        t = row.get("expected_procedure_title", "")

        if len(q) < MIN_QUESTION_LEN:
            errors.append(f"Line {line_num}: question too short ({len(q)} < {MIN_QUESTION_LEN})")
        if len(a) < MIN_ANSWER_LEN:
            errors.append(f"Line {line_num}: reference_answer too short ({len(a)} < {MIN_ANSWER_LEN})")
        if len(t) < MIN_TITLE_LEN:
            errors.append(f"Line {line_num}: expected_procedure_title too short ({len(t)} < {MIN_TITLE_LEN})")

        # Check duplicate questions
        if q in questions:
            errors.append(f"Line {line_num}: duplicate question")
        questions.add(q)

        valid_count += 1

    # Report
    print(f"File: {path}")
    print(f"Total lines: {len(lines)}")
    print(f"Valid samples: {valid_count}")
    print(f"Unique questions: {len(questions)}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for err in errors[:20]:
            print(f"  {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        raise SystemExit(1)
    else:
        print("\nPASS: All checks passed.")


if __name__ == "__main__":
    main()
