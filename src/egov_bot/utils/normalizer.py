from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return _SPACE_RE.sub(" ", text)


def strip_vietnamese_accents(text: str | None) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def searchable_text(text: str | None) -> str:
    return normalize_text(strip_vietnamese_accents(text))

