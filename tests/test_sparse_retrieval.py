"""Regression tests for the sparse retrieval path.

Each case here corresponds to a failure that ran silently in production: the
BM25 index and the query path tokenized differently, the fusion labelled
"Reciprocal Rank Fusion" was dominated by its score term, and a sparse index
built against a different corpus still returned confident-looking results.
"""

from egov_bot.config import Settings
from egov_bot.data.procedure_store import ProcedureStore
from egov_bot.retrieval.hybrid_retriever import (
    RetrievalResult,
    _collapse_to_procedures,
    _fuse_legacy,
    _fuse_rrf,
    _fuse_weighted,
)
from egov_bot.retrieval.query_rewriter import strip_boilerplate
from egov_bot.retrieval.sparse_index import BACKEND_BM25S, SparseIndex
from egov_bot.utils.normalizer import BM25_TOKENIZER_VERSION, tokenize_bm25

# --------------------------------------------------------------------- tokenizer


def test_tokenizer_lowercases():
    assert tokenize_bm25("Đăng Ký KHAI SINH") == ["đăng", "ký", "khai", "sinh"]


def test_tokenizer_strips_attached_punctuation():
    # The shipped index had punctuation attached to 78.9% of its vocabulary,
    # so "định;" and "định" were counted as unrelated terms.
    assert tokenize_bm25("quy định; sửa đổi, bổ sung.") == [
        "quy", "định", "sửa", "đổi", "bổ", "sung",
    ]


def test_tokenizer_keeps_legal_document_codes_intact():
    assert tokenize_bm25("Thông tư 38/2015/TT-BTC,") == ["thông", "tư", "38/2015/tt-btc"]


def test_tokenizer_matches_across_case_and_punctuation():
    indexed = tokenize_bm25("Đăng ký khai sinh.")
    queried = tokenize_bm25("đăng ký khai sinh?")
    assert indexed == queried


def test_tokenizer_handles_empty_input():
    assert tokenize_bm25("") == []
    assert tokenize_bm25(None) == []


# ------------------------------------------------------------------ sparse index


def test_sparse_index_roundtrip_and_query(tmp_path):
    texts = [
        "Thủ tục đăng ký khai sinh cho trẻ em",
        "Cấp căn cước công dân lần đầu",
        "Đăng ký kết hôn có yếu tố nước ngoài",
    ]
    index = SparseIndex.build(texts)
    assert index.kind == BACKEND_BM25S
    assert index.doc_count == 3

    path = tmp_path / "bm25.pkl.gz"
    index.save(path)

    import gzip
    import pickle

    with gzip.open(path, "rb") as file:
        payload = pickle.load(file)
    loaded = SparseIndex.load(payload)

    assert loaded is not None
    assert loaded.doc_count == 3
    assert loaded.tokenizer_version == BM25_TOKENIZER_VERSION

    hits = loaded.top_n("đăng ký khai sinh", 3)
    assert hits, "query should match the indexed corpus"
    assert hits[0][0] == 0


def test_sparse_index_query_survives_case_and_punctuation():
    index = SparseIndex.build(["Thủ tục đăng ký khai sinh", "Cấp hộ chiếu phổ thông"])
    lowered = index.top_n("đăng ký khai sinh", 2)
    shouted = index.top_n("ĐĂNG KÝ KHAI SINH?!", 2)
    assert lowered and shouted
    assert lowered[0][0] == shouted[0][0] == 0


def test_sparse_index_clamps_k_above_corpus_size():
    # bm25s raises when k exceeds the corpus, which would surface as an
    # exception swallowed into an empty result set.
    index = SparseIndex.build(["một tài liệu duy nhất"])
    assert index.top_n("tài liệu", 500)


def test_sparse_index_returns_nothing_for_unknown_terms():
    index = SparseIndex.build(["đăng ký khai sinh"])
    assert index.top_n("zzzz qqqq", 5) == []


def test_sparse_index_load_rejects_unknown_payload():
    assert SparseIndex.load(None) is None
    assert SparseIndex.load(object()) is None


# ------------------------------------------------------------------------ fusion


def _ranked(*scores: float) -> dict[int, float]:
    return dict(enumerate(scores))


def test_rrf_ignores_score_magnitude():
    """RRF must depend on rank only, otherwise scale differences leak in."""
    dense = _ranked(0.9, 0.8)
    sparse = _ranked(0.05, 0.04)
    scaled_up = {k: v * 1000 for k, v in sparse.items()}
    assert _fuse_rrf(dense, sparse) == _fuse_rrf(dense, scaled_up)


def test_legacy_fusion_is_dominated_by_its_score_term():
    """The 'RRF' that shipped ranks by score, not rank.

    The rank term maxes out at 1/61 while the score term reaches 0.35, so a
    document that is last by rank but top by score still wins.
    """
    dense = {0: 0.0, 1: 1.0}  # doc 0 ranks first only because dict order
    sparse: dict[int, float] = {}
    legacy = _fuse_legacy(dense, sparse)
    assert legacy[1] > legacy[0]

    rrf = _fuse_rrf(dense, sparse)
    assert rrf[1] > rrf[0]  # doc 1 also ranks first on score-sorted rank order


def test_weighted_fusion_respects_alpha():
    dense = _ranked(1.0)
    sparse = _ranked(0.0)
    assert _fuse_weighted(dense, sparse, alpha=1.0)[0] == 1.0
    assert _fuse_weighted(dense, sparse, alpha=0.0)[0] == 0.0


# --------------------------------------------------------------- parent grouping


def _result(index: int, parent: str, title: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        index=index, parent_id=parent, title=title, url=parent, text="", score=score
    )


def test_collapse_keeps_best_chunk_per_procedure():
    results = [
        _result(0, "p1", "Đăng ký khai sinh", 0.9),
        _result(1, "p1", "Đăng ký khai sinh", 0.8),
        _result(2, "p2", "Cấp hộ chiếu", 0.7),
    ]
    collapsed = _collapse_to_procedures(results)
    assert [r.parent_id for r in collapsed] == ["p1", "p2"]
    assert collapsed[0].score == 0.9
    assert collapsed[0].chunk_matches == 2
    assert collapsed[1].chunk_matches == 1


def test_collapse_by_title_merges_provincial_copies():
    results = [
        _result(0, "p-hanoi", "Đăng ký khai sinh", 0.9),
        _result(1, "p-hcm", "Đăng ký khai sinh", 0.8),
    ]
    assert len(_collapse_to_procedures(results, "parent")) == 2
    assert len(_collapse_to_procedures(results, "title")) == 1


def test_collapse_preserves_rank_order():
    results = [
        _result(0, "p2", "B", 0.9),
        _result(1, "p1", "A", 0.8),
        _result(2, "p2", "B", 0.7),
    ]
    assert [r.parent_id for r in _collapse_to_procedures(results)] == ["p2", "p1"]


# ------------------------------------------------------------- query preprocessing


def test_strip_boilerplate_removes_lead_in():
    assert strip_boilerplate("Xin hỏi thẩm quyền phê duyệt phương án ứng phó thiên tai") == (
        "thẩm quyền phê duyệt phương án ứng phó thiên tai"
    )


def test_strip_boilerplate_keeps_query_when_nothing_substantive_remains():
    question = "Cho biết là gì?"
    assert strip_boilerplate(question) == question


def test_strip_boilerplate_handles_empty():
    assert strip_boilerplate("") == ""
    assert strip_boilerplate(None) == ""


# ------------------------------------------------------------- alignment guard


def test_retriever_warns_when_index_and_corpus_disagree(tmp_path, caplog):
    """A misaligned index returns confident results about the wrong documents."""
    from egov_bot.retrieval.hybrid_retriever import HybridRetriever

    settings = Settings(cache_dir=tmp_path, data_dir=tmp_path, sqlite_path=tmp_path / "t.db")
    index = SparseIndex.build(["a", "b", "c"])  # 3 docs
    metadatas = [{"text": "a", "ten_thu_tuc": "A", "nguon": "u"}]  # but 1 chunk

    with caplog.at_level("ERROR"):
        HybridRetriever(
            settings=settings,
            procedure_store=ProcedureStore([]),
            metadatas=metadatas,
            bm25=index,
        )
    assert any("different corpus" in record.message for record in caplog.records)
