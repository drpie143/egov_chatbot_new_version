from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")

# BM25 tokenisation must be identical when the index is built and when it is
# queried. Keep `/` and `-` inside a token so legal document codes such as
# `38/2015/tt-btc` survive as a single searchable term.
_BM25_SPLIT_RE = re.compile(r"[^\w/-]+", re.UNICODE)
_BM25_TRIM_RE = re.compile(r"^[/-]+|[/-]+$")

#: Bumped whenever `tokenize_bm25` changes, so a stale index can be detected
#: instead of silently degrading retrieval quality.
BM25_TOKENIZER_VERSION = "v2-lower-punct"


def tokenize_bm25(text: str | None) -> list[str]:
    """Tokenize text for BM25 retrieval.

    The previous implementation used a bare ``str.split()`` on both sides, which
    left punctuation attached to 78.9% of the indexed vocabulary (``định;`` and
    ``định`` were counted as different terms) and kept 27.4% of terms reachable
    only in their capitalised form.
    """
    if not text:
        return []
    lowered = unicodedata.normalize("NFKC", str(text)).lower()
    tokens: list[str] = []
    for chunk in _BM25_SPLIT_RE.sub(" ", lowered).split():
        token = _BM25_TRIM_RE.sub("", chunk)
        if token:
            tokens.append(token)
    return tokens


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

