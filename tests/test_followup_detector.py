from egov_bot.conversation.followup_detector import is_followup


def test_short_contextual_question_is_followup():
    assert is_followup("Lệ phí bao nhiêu?", has_history=True)


def test_specific_procedure_starts_new_topic():
    assert not is_followup("Thủ tục cấp hộ chiếu cần gì?", has_history=True)


def test_no_history_is_never_followup():
    assert not is_followup("Lệ phí bao nhiêu?", has_history=False)

