"""Rule-based query preprocessing for retrieval.

Public-service questions arrive wrapped in request phrasing ("Xin hỏi...",
"Đề nghị cho biết...", "Trình tự thực hiện thủ tục hành chính..."). Measured on
the 167-question FAQ set, 91% of queries carry such a lead-in, averaging 2.22 of
25.76 tokens. Those tokens are near-uniformly distributed across the corpus, so
they add BM25 term matches and dense-embedding mass without discriminating
between procedures.

This rewriting is applied to the retrieval query only. The original question is
what reaches the answer prompt, since the phrasing carries intent the model
should still see.
"""

from __future__ import annotations

import re

_BOILERPLATE_PREFIXES = [
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

_TRAILING_NOISE = [
    r"là gì\s*\??$",
    r"như thế nào\s*\??$",
    r"ra sao\s*\??$",
    r"được không\s*\??$",
    r"\?+$",
]

_PREFIX_RE = re.compile("|".join(_BOILERPLATE_PREFIXES), re.IGNORECASE)
_TRAILING_RE = re.compile("|".join(_TRAILING_NOISE), re.IGNORECASE)

_MIN_TOKENS_AFTER_STRIP = 3


def strip_boilerplate(question: str | None) -> str:
    """Remove request phrasing, keeping the substantive part of the question.

    Falls back to the original text when stripping would leave too little to
    retrieve on, so a query that is nothing but a lead-in still works.
    """
    if not question:
        return ""
    original = question.strip()
    text = original
    for _ in range(3):  # lead-ins are often stacked two deep
        stripped = _PREFIX_RE.sub(" ", text, count=1).strip(" ,:;.-")
        if stripped == text:
            break
        text = stripped
    text = _TRAILING_RE.sub(" ", text).strip(" ,:;.-")

    if len(text.split()) < _MIN_TOKENS_AFTER_STRIP:
        return original
    return text
