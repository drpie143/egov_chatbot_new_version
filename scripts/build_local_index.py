from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def chunk_record(record: dict, max_chars: int = 1000) -> list[dict]:
    title = str(record.get("ten_thu_tuc") or "")
    url = str(record.get("nguon") or "")
    fields = [
        "co_quan_thuc_hien",
        "yeu_cau_dieu_kien",
        "thanh_phan_ho_so",
        "trinh_tu_thuc_hien",
        "cach_thuc_thuc_hien",
        "thu_tuc_lien_quan",
    ]
    chunks: list[dict] = []
    for field in fields:
        value = str(record.get(field) or "").strip()
        if not value:
            continue
        for start in range(0, len(value), max_chars):
            text = value[start : start + max_chars]
            chunks.append(
                {
                    "parent_id": url,
                    "nguon": url,
                    "ten_thu_tuc": title,
                    "field": field,
                    "text": f"{title}\n{text}",
                }
            )
    if not chunks and title:
        chunks.append({"parent_id": url, "nguon": url, "ten_thu_tuc": title, "field": "ten_thu_tuc", "text": title})
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local FAISS and BM25 index from procedure JSON.")
    parser.add_argument("--input", default="static/data/toan_bo_du_lieu_final.json")
    parser.add_argument("--output-dir", default=".cache/egov_data")
    parser.add_argument("--embedding-model", default="AITeamVN/Vietnamese_Embedding")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = [chunk for record in records for chunk in chunk_record(record)]
    texts = [chunk["text"] for chunk in chunks]

    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype(
        "float32"
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with gzip.open(output_dir / "metas.pkl.gz", "wb") as file:
        pickle.dump({"corpus": chunks}, file)
    with gzip.open(output_dir / "bm25.pkl.gz", "wb") as file:
        pickle.dump(BM25Okapi([text.split() for text in texts]), file)
    (output_dir / "toan_bo_du_lieu_final.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    print(f"Built {len(chunks)} chunks into {output_dir}")


if __name__ == "__main__":
    main()

