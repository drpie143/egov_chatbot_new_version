from egov_bot.utils.normalizer import normalize_text, searchable_text


def test_normalize_text_collapses_spacing():
    assert normalize_text("  Đăng   Ký  ") == "đăng ký"


def test_searchable_text_removes_vietnamese_accents():
    assert searchable_text("Đăng ký khai sinh") == "dang ky khai sinh"

