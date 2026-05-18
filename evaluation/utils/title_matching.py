"""Title matching utilities: exact normalized match and fuzzy fallback with RapidFuzz."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evaluation.utils.text_normalize import normalize_title

logger = logging.getLogger(__name__)


def load_corpus_titles(corpus_path: str | Path) -> list[str]:
    """Load procedure titles from the corpus JSON file."""
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    titles: list[str] = []
    for record in records:
        for field in ["ten_thu_tuc", "title", "procedure_title", "name"]:
            value = record.get(field)
            if value and isinstance(value, str) and value.strip():
                titles.append(value.strip())
                break
    return titles


def build_title_map(titles: list[str]) -> dict[str, str]:
    """Build a normalized-title -> original-title mapping."""
    title_map: dict[str, str] = {}
    for title in titles:
        key = normalize_title(title)
        if key and key not in title_map:
            title_map[key] = title
    return title_map


def exact_match(expected_title: str, title_map: dict[str, str]) -> str | None:
    """Return the corpus title if exact normalized match is found."""
    key = normalize_title(expected_title)
    return title_map.get(key)


def fuzzy_match(
    expected_title: str,
    title_map: dict[str, str],
    threshold: float = 92.0,
) -> tuple[str | None, float]:
    """Fuzzy match using RapidFuzz. Returns (matched_title, score) or (None, score).

    Only returns a match if the fuzzy score >= threshold.
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        logger.warning("rapidfuzz is not installed; fuzzy matching is disabled.")
        return None, 0.0

    key = normalize_title(expected_title)
    candidates = list(title_map.keys())
    if not candidates:
        return None, 0.0

    best_key, score, _ = process.extractOne(
        key,
        candidates,
        scorer=fuzz.token_set_ratio,
    )
    if score >= threshold:
        return title_map[best_key], score
    return None, score


def match_title(
    expected_title: str,
    title_map: dict[str, str],
    fuzzy_threshold: float = 92.0,
) -> tuple[str | None, str, float]:
    """Try exact match first, then fuzzy. Returns (matched_title, method, score)."""
    result = exact_match(expected_title, title_map)
    if result is not None:
        return result, "exact", 100.0

    result, score = fuzzy_match(expected_title, title_map, fuzzy_threshold)
    if result is not None:
        return result, "fuzzy", score

    return None, "none", score


def title_match(predicted_title: str, expected_title: str) -> bool:
    """Check if two titles match after normalization (for evaluation metrics)."""
    return normalize_title(predicted_title) == normalize_title(expected_title)
