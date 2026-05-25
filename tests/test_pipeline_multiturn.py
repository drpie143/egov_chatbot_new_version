from egov_bot.config import Settings
from egov_bot.conversation.session_manager import SessionManager
from egov_bot.data.procedure_store import ProcedureStore
from egov_bot.rag.pipeline import RAGPipeline
from egov_bot.retrieval.hybrid_retriever import HybridRetriever
from egov_bot.storage.db import Database
from egov_bot.storage.repositories import AppRepository


def build_pipeline(tmp_path):
    store = ProcedureStore(
        [
            {
                "ten_thu_tuc": "Đăng ký khai sinh",
                "nguon": "https://example.com/khai-sinh",
                "co_quan_thuc_hien": "Ủy ban nhân dân cấp xã",
                "thanh_phan_ho_so": "Tờ khai đăng ký khai sinh, giấy chứng sinh.",
                "cach_thuc_thuc_hien": "Lệ phí: miễn lệ phí nếu đăng ký đúng hạn; 75.000 đồng nếu đăng ký quá hạn.",
            }
        ]
    )
    settings = Settings(
        google_api_key=None,
        google_api_key_2=None,
        cache_dir=tmp_path,
        data_dir=tmp_path,
        sqlite_path=tmp_path / "test.db",
    )
    sessions = SessionManager()
    repository = AppRepository(Database(settings.sqlite_path))
    retriever = HybridRetriever(settings=settings, procedure_store=store)
    return RAGPipeline(settings, store, retriever, sessions, repository), sessions


def test_followup_reuses_previous_procedure_context(tmp_path):
    pipeline, _ = build_pipeline(tmp_path)

    first = pipeline.answer("Tôi muốn làm giấy khai sinh", session_id="s1", request_id="req-1")
    followup = pipeline.answer("Cần bao nhiêu tiền?", session_id="s1", request_id="req-2")

    assert first.context_source == "https://example.com/khai-sinh"
    assert followup.context_source == first.context_source
    assert "75.000" in followup.answer


def test_cached_answer_restores_session_context_for_followup(tmp_path):
    pipeline, sessions = build_pipeline(tmp_path)

    pipeline.answer("Tôi muốn làm giấy khai sinh", session_id="s1", request_id="req-1")
    sessions.clear("s1")

    cached_first = pipeline.answer("Tôi muốn làm giấy khai sinh", session_id="s1", request_id="req-2")
    followup = pipeline.answer("Cần bao nhiêu tiền?", session_id="s1", request_id="req-3")

    assert cached_first.cached
    assert followup.context_source == "https://example.com/khai-sinh"
    assert "75.000" in followup.answer
