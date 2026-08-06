"""Fail loudly when a declared dependency is missing from the environment.

`rapidfuzz` was listed in requirements.txt but absent from the environment that
built the evaluation set. The caller treated the resulting ImportError as "no
match found" and dropped every question whose title was not a verbatim corpus
match, which silently cut the test set from 167 questions to 74 and inflated
every metric computed from it.

These imports are cheap and run in CI, so the same class of failure surfaces as
a red build instead of as a quietly better-looking number.
"""

import importlib

import pytest

# Modules whose absence changes results rather than crashing the process.
SILENT_FAILURE_DEPENDENCIES = [
    ("bm25s", "sparse retrieval backend"),
    ("rapidfuzz", "fuzzy title matching when building the evaluation set"),
    ("faiss", "dense retrieval index"),
    ("sentence_transformers", "embedding model and cross-encoder reranker"),
    ("scipy", "sparse matrix backend used by bm25s"),
]


@pytest.mark.parametrize("module_name,purpose", SILENT_FAILURE_DEPENDENCIES)
def test_dependency_is_importable(module_name: str, purpose: str):
    try:
        importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only on a broken env
        pytest.fail(
            f"{module_name!r} is required for {purpose} but cannot be imported: {exc}. "
            "Retrieval and evaluation degrade silently without it."
        )


def test_fuzzy_matching_raises_instead_of_returning_no_match():
    """The evaluation cleaner must not treat a missing dependency as 'no match'."""
    from evaluation.utils.title_matching import fuzzy_match

    matched, score = fuzzy_match("Đăng ký khai sinh", {"đăng ký khai sinh": "Đăng ký khai sinh"})
    assert matched == "Đăng ký khai sinh"
    assert score >= 92.0
