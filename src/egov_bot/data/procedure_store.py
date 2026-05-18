from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from egov_bot.schemas.common import Source
from egov_bot.utils.normalizer import normalize_text, searchable_text

FIELD_LABELS = {
    "ten_thu_tuc": "Tên thủ tục",
    "co_quan_thuc_hien": "Cơ quan thực hiện",
    "yeu_cau_dieu_kien": "Yêu cầu, điều kiện",
    "thanh_phan_ho_so": "Thành phần hồ sơ",
    "trinh_tu_thuc_hien": "Trình tự thực hiện",
    "cach_thuc_thuc_hien": "Cách thức thực hiện",
    "thu_tuc_lien_quan": "Thủ tục liên quan",
    "nguon": "Nguồn",
}


@dataclass(frozen=True)
class Procedure:
    title: str
    url: str
    agency: str = ""
    fields: dict[str, Any] | None = None

    def to_source(self, score: float = 0.0, snippet: str = "") -> Source:
        return Source(
            title=self.title or "Thu tuc hanh chinh",
            url=self.url,
            agency=self.agency or "",
            score=score,
            snippet=snippet,
        )


class ProcedureStore:
    def __init__(self, procedures: list[dict[str, Any]] | None = None) -> None:
        self.records = procedures or []
        self.by_url: dict[str, dict[str, Any]] = {}
        self.by_title: dict[str, dict[str, Any]] = {}
        self._search_rows: list[tuple[int, str]] = []
        self._index_records()

    def _index_records(self) -> None:
        for index, record in enumerate(self.records):
            url = str(record.get("nguon") or "")
            title = str(record.get("ten_thu_tuc") or "")
            if url:
                self.by_url[url] = record
            if title:
                self.by_title[normalize_text(title)] = record
            haystack = " ".join(
                str(record.get(key) or "")
                for key in [
                    "ten_thu_tuc",
                    "co_quan_thuc_hien",
                    "thanh_phan_ho_so",
                    "trinh_tu_thuc_hien",
                    "cach_thuc_thuc_hien",
                    "yeu_cau_dieu_kien",
                ]
            )
            self._search_rows.append((index, searchable_text(haystack)))

    def __len__(self) -> int:
        return len(self.records)

    def get(self, parent_id: str | None) -> dict[str, Any] | None:
        if not parent_id:
            return None
        return self.by_url.get(parent_id) or self.by_title.get(normalize_text(parent_id))

    def to_procedure(self, record: dict[str, Any] | None) -> Procedure | None:
        if not record:
            return None
        return Procedure(
            title=str(record.get("ten_thu_tuc") or "Thu tuc hanh chinh"),
            url=str(record.get("nguon") or ""),
            agency=str(record.get("co_quan_thuc_hien") or ""),
            fields=record,
        )

    def source_for(self, parent_id: str | None, score: float = 0.0, snippet: str = "") -> Source | None:
        procedure = self.to_procedure(self.get(parent_id))
        if procedure is None:
            return None
        return procedure.to_source(score=score, snippet=snippet)

    def format_record(self, record: dict[str, Any] | None) -> str:
        if not record:
            return ""
        parts: list[str] = []
        for key in [
            "ten_thu_tuc",
            "co_quan_thuc_hien",
            "yeu_cau_dieu_kien",
            "thanh_phan_ho_so",
            "trinh_tu_thuc_hien",
            "cach_thuc_thuc_hien",
            "thu_tuc_lien_quan",
            "nguon",
        ]:
            value = record.get(key)
            if value:
                parts.append(f"{FIELD_LABELS[key]}:\n{str(value).strip()}")
        return "\n\n".join(parts)

    def format_procedure(self, parent_id: str | None) -> str:
        return self.format_record(self.get(parent_id))

    def search(self, query: str, limit: int = 10) -> list[Source]:
        q = searchable_text(query)
        if not q:
            return []
        terms = [term for term in q.split() if len(term) > 1]
        if not terms:
            terms = [q]

        scored: list[tuple[float, int]] = []
        for index, haystack in self._search_rows:
            score = 0.0
            for term in terms:
                if term in haystack:
                    score += 2.0 if haystack.startswith(term) else 1.0
            if score:
                title = searchable_text(self.records[index].get("ten_thu_tuc", ""))
                score += sum(1.5 for term in terms if term in title)
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[Source] = []
        for score, index in scored[:limit]:
            record = self.records[index]
            procedure = self.to_procedure(record)
            if procedure is None:
                continue
            snippet = self._make_snippet(record, terms)
            results.append(procedure.to_source(score=score, snippet=snippet))
        return results

    def _make_snippet(self, record: dict[str, Any], terms: list[str]) -> str:
        for key in ["thanh_phan_ho_so", "trinh_tu_thuc_hien", "cach_thuc_thuc_hien", "yeu_cau_dieu_kien"]:
            value = str(record.get(key) or "").strip()
            if not value:
                continue
            searchable = searchable_text(value)
            if any(term in searchable for term in terms):
                return value[:280]
        return str(record.get("co_quan_thuc_hien") or "")[:280]
