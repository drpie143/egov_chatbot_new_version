from __future__ import annotations

from egov_bot.schemas.common import Source


def build_prompt(history: str, context: str, question: str, sources: list[Source]) -> str:
    source_lines = "\n".join(
        f"- {source.title} | {source.url}" for source in sources if source.url
    )
    return f"""Bạn là Vietnamese eGov RAG Assistant, trợ lý tra cứu thủ tục hành chính Việt Nam.

Quy tắc bắt buộc:
- Chỉ trả lời dựa trên DỮ LIỆU được cung cấp.
- Nếu dữ liệu không đủ, nói rõ chưa có thông tin và gợi ý người dùng hỏi cụ thể hơn.
- Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc dễ đọc.
- Luôn nêu nguồn ở cuối khi có URL nguồn.
- Không tự bịa quy định, phí, thời hạn hoặc cơ quan nếu dữ liệu không có.

Lịch sử hội thoại:
{history or "(không có)"}

DỮ LIỆU:
---
{context or "(không có dữ liệu phù hợp)"}
---

Nguồn ứng viên:
{source_lines or "(không có)"}

CÂU HỎI: {question}

TRẢ LỜI:"""


def build_fallback_answer(context: str, sources: list[Source]) -> str:
    if not context:
        return (
            "Mình chưa tìm thấy dữ liệu phù hợp cho câu hỏi này. "
            "Bạn có thể hỏi rõ hơn tên thủ tục hoặc tra cứu tại Cổng dịch vụ công quốc gia."
        )
    trimmed = context[:1800].strip()
    answer = (
        "Hệ thống đã tìm thấy nguồn phù hợp nhưng chưa cấu hình khóa Gemini, "
        "nên mình hiển thị phần thông tin trích từ dữ liệu nguồn:\n\n"
        f"{trimmed}"
    )
    if sources:
        answer += "\n\nNguồn tham khảo:\n"
        answer += "\n".join(f"- {source.title}: {source.url}" for source in sources[:3])
    return answer

