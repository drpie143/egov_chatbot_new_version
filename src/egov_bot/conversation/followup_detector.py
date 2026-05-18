from __future__ import annotations

import re

from egov_bot.utils.normalizer import normalize_text

FOLLOWUP_PATTERNS = [
    r"\b(thu tuc|thủ tục)\s+(nay|này|do|đó|tren|trên)\b",
    r"\b(no|nó|viec nay|việc này|ho so nay|hồ sơ này)\b",
    r"\b(co quan nao|cơ quan nào|phi bao nhieu|phí bao nhiêu|mat bao lau|mất bao lâu)\b",
    r"\b(trinh tu|trình tự|ho so|hồ sơ|le phi|lệ phí|dieu kien|điều kiện)\b",
]

NEW_TOPIC_PATTERNS = [
    r"\b(thu tuc|thủ tục)\s+\w+",
    r"\b(dang ky|đăng ký|cap|cấp|xin|doi|đổi)\b",
    r"\b(giay phep|giấy phép|ho chieu|hộ chiếu|khai sinh|ket hon|kết hôn)\b",
]


def is_followup(text: str, has_history: bool = True) -> bool:
    if not has_history:
        return False
    value = normalize_text(text)
    if not value:
        return False
    if any(re.search(pattern, value) for pattern in FOLLOWUP_PATTERNS):
        return True
    if any(re.search(pattern, value) for pattern in NEW_TOPIC_PATTERNS):
        return False
    return len(value.split()) <= 8

