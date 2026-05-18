from egov_bot.config import Settings
from egov_bot.data.procedure_store import ProcedureStore
from egov_bot.retrieval.hybrid_retriever import HybridRetriever


def test_search_falls_back_to_procedure_store(tmp_path):
    store = ProcedureStore(
        [
            {
                "ten_thu_tuc": "Thủ tục đăng ký khai sinh",
                "nguon": "https://example.com/khai-sinh",
                "co_quan_thuc_hien": "Ủy ban nhân dân",
                "thanh_phan_ho_so": "Tờ khai đăng ký khai sinh",
            }
        ]
    )
    settings = Settings(cache_dir=tmp_path, data_dir=tmp_path, sqlite_path=tmp_path / "test.db")
    retriever = HybridRetriever(settings=settings, procedure_store=store)
    results = retriever.search("đăng ký khai sinh", limit=3)
    assert results
    assert results[0].url == "https://example.com/khai-sinh"

