"""Text normalization utilities for Vietnamese FAQ benchmark evaluation."""

from __future__ import annotations

import html
import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[.,;:!?'\"\"\"''()\[\]{}]")


def normalize_text(s: str) -> str:
    """General-purpose text normalization: unescape HTML, NFC, collapse whitespace."""
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\xa0", " ")
    s = _SPACE_RE.sub(" ", s)
    s = s.strip()
    return s


def normalize_title(s: str) -> str:
    """Normalize a procedure title for matching: lowercase, remove punctuation."""
    s = normalize_text(s).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def strip_vietnamese_accents(s: str) -> str:
    """Remove Vietnamese diacritics (used only for fuzzy fallback, not primary matching)."""
    normalized = unicodedata.normalize("NFD", s)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")
