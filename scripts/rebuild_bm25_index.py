"""Rebuild the BM25 sparse index with the shared tokenizer.

The published index was built with a bare ``text.split()``, which left
punctuation attached to most of the vocabulary and split each term across its
punctuated variants. This script rebuilds it from the existing chunk metadata,
so the FAISS index and embeddings are reused as-is.

Usage:
    python scripts/rebuild_bm25_index.py
    python scripts/rebuild_bm25_index.py --output .cache/egov_data/bm25.pkl.gz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Must happen before load_settings(): HF_REPO_ID and DATA_DIR decide which
# corpus the index is built against, and an index built against the wrong
# corpus silently returns unrelated documents.
load_dotenv(ROOT / ".env")

from egov_bot.config import load_settings  # noqa: E402
from egov_bot.data.resource_loader import load_resources  # noqa: E402
from egov_bot.retrieval.sparse_index import SparseIndex  # noqa: E402
from egov_bot.utils.normalizer import BM25_TOKENIZER_VERSION, tokenize_bm25  # noqa: E402


def chunk_text(metadata: dict) -> str:
    return str(metadata.get("text") or metadata.get("raw") or metadata.get("content") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the BM25 index with the shared tokenizer.")
    parser.add_argument("--output", default=None, help="Defaults to <DATA_DIR>/bm25.pkl.gz")
    args = parser.parse_args()

    settings = load_settings()
    output = Path(args.output) if args.output else settings.data_dir / "bm25.pkl.gz"

    print("Loading chunk metadata (no embedding model needed)...")
    resources = load_resources(settings, load_models=False)
    texts = [chunk_text(meta) for meta in resources.metadatas]
    if not texts:
        raise SystemExit("No chunk metadata found. Check DATA_SOURCE / cache state.")

    print(f"Corpus: {settings.hf_repo_id} (data_dir={settings.data_dir})")
    print(f"Chunks: {len(texts):,}")
    print(f"Tokenizer: {BM25_TOKENIZER_VERSION}")

    sample_tokens = tokenize_bm25(texts[0])[:12]
    print(f"Sample tokens: {sample_tokens}")

    start = time.perf_counter()
    index = SparseIndex.build(texts)
    elapsed = time.perf_counter() - start
    print(f"Built {index.kind} index over {index.doc_count:,} docs in {elapsed:.1f}s")

    index.save(output)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Saved -> {output}  ({size_mb:.1f} MB)")
    print("\nThe resource loader prefers DATA_DIR over the Hugging Face copy,")
    print("so the app and evaluation scripts will pick this up on next start.")


if __name__ == "__main__":
    main()
