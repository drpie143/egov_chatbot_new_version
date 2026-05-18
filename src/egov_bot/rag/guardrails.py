from __future__ import annotations

from egov_bot.schemas.common import Source
from egov_bot.utils.normalizer import searchable_text

DOMAIN_TERMS = {
    "thu tuc",
    "hanh chinh",
    "ho so",
    "giay",
    "cap",
    "dang ky",
    "co quan",
    "le phi",
    "khai sinh",
    "ket hon",
    "ho chieu",
    "dich vu cong",
    "doanh nghiep",
    "giay phep",
}


def looks_in_domain(question: str, sources: list[Source]) -> bool:
    if sources:
        return True
    q = searchable_text(question)
    return any(term in q for term in DOMAIN_TERMS)


def out_of_domain_answer() -> str:
    return (
        "Mình chỉ hỗ trợ tra cứu thủ tục hành chính Việt Nam dựa trên dữ liệu đã thu thập. "
        "Bạn hãy hỏi rõ tên thủ tục, hồ sơ, cơ quan thực hiện, lệ phí hoặc trình tự thực hiện "
        "để mình tìm nguồn phù hợp hơn."
    )

