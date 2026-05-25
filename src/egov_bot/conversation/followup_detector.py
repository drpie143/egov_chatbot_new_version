from __future__ import annotations

import re

from egov_bot.utils.normalizer import searchable_text

CONTEXT_REFERENCE_PATTERNS = [
    r"\b(thu tuc|ho so|viec|noi dung)\s+(nay|do|tren|vua roi)\b",
    r"\b(no|viec nay|ho so nay|noi tren|cai nay)\b",
]

DETAIL_FOLLOWUP_PATTERNS = [
    r"\b(co quan nao|o dau|nop o dau|lam o dau)\b",
    r"\b(phi|le phi|phi bao nhieu|bao nhieu tien|can bao nhieu tien|het bao nhieu|dong)\b",
    r"\b(mat bao lau|bao lau|thoi han|thoi gian)\b",
    r"\b(trinh tu|ho so|giay to|can gi|can giay to gi|dieu kien)\b",
]

NEW_TOPIC_PATTERNS = [
    r"\b(thu tuc)\s+\w+",
    r"\b(dang ky|cap|xin|doi|lam)\s+(giay|thu tuc|khai sinh|ket hon|ho chieu|can cuoc|giay phep)\b",
    r"\b(giay phep|ho chieu|khai sinh|ket hon|can cuoc cong dan|ly lich tu phap)\b",
]


def is_followup(text: str, has_history: bool = True) -> bool:
    if not has_history:
        return False
    value = searchable_text(text)
    if not value:
        return False
    if any(re.search(pattern, value) for pattern in CONTEXT_REFERENCE_PATTERNS):
        return True
    if any(re.search(pattern, value) for pattern in NEW_TOPIC_PATTERNS):
        return False
    if any(re.search(pattern, value) for pattern in DETAIL_FOLLOWUP_PATTERNS):
        return True
    return len(value.split()) <= 8
