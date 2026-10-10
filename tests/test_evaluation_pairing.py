import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "compare_runs", Path(__file__).resolve().parents[1] / "evaluation" / "compare_runs.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def write_run(path, question, rank):
    path.write_text(json.dumps({"question": question, "expected_procedure_title": "Khai sinh", "rank": rank}), encoding="utf-8")


def test_equal_counts_do_not_make_different_questions_a_paired_benchmark(tmp_path):
    baseline, candidate = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_run(baseline, "Question A", 1)
    write_run(candidate, "Question B", 1)
    with pytest.raises(ValueError, match="not paired"):
        comparison.load_paired_ranks(baseline, candidate)


def test_invalid_rank_cannot_inflate_recall_or_break_mrr(tmp_path):
    baseline, candidate = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_run(baseline, "Question", 1)
    write_run(candidate, "Question", 0)
    with pytest.raises(ValueError, match="Invalid retrieval rank"):
        comparison.load_paired_ranks(baseline, candidate)
