from egov_bot.rag.prompt_builder import build_prompt
from egov_bot.schemas.common import Source


def test_prompt_contains_question_context_and_source():
    prompt = build_prompt(
        history="User: Xin chào",
        context="Ten thu tuc: Đăng ký khai sinh",
        question="Cần giấy tờ gì?",
        sources=[Source(title="Đăng ký khai sinh", url="https://example.com")],
    )
    assert "Cần giấy tờ gì?" in prompt
    assert "Đăng ký khai sinh" in prompt
    assert "https://example.com" in prompt

