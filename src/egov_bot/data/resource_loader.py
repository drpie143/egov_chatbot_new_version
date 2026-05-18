from __future__ import annotations

import gzip
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egov_bot.config import Settings
from egov_bot.data.procedure_store import ProcedureStore

logger = logging.getLogger(__name__)


RESOURCE_FILES = {
    "faiss": "index.faiss",
    "metas": "metas.pkl.gz",
    "bm25": "bm25.pkl.gz",
    "raw": "toan_bo_du_lieu_final.json",
}


@dataclass
class Resources:
    procedure_store: ProcedureStore
    procedures: list[dict[str, Any]]
    metadatas: list[dict[str, Any]]
    faiss_index: Any = None
    bm25: Any = None
    embedding_model: Any = None
    load_errors: list[str] | None = None

    @property
    def loaded(self) -> bool:
        return bool(self.procedures)

    @property
    def index_loaded(self) -> bool:
        return self.faiss_index is not None and bool(self.metadatas)

    @property
    def embedding_loaded(self) -> bool:
        return self.embedding_model is not None


def load_resources(settings: Settings, load_models: bool = True) -> Resources:
    errors: list[str] = []
    local_paths = _resolve_resource_paths(settings, errors)
    procedures = _load_raw_json(local_paths.get("raw"), errors)
    metadatas = _load_metadatas(local_paths.get("metas"), errors)
    faiss_index = _load_faiss(local_paths.get("faiss"), errors)
    bm25 = _load_pickle(local_paths.get("bm25"), "BM25", errors)
    embedding_model = _load_embedding_model(settings, errors) if load_models else None

    store = ProcedureStore(procedures)
    resources = Resources(
        procedure_store=store,
        procedures=procedures,
        metadatas=metadatas,
        faiss_index=faiss_index,
        bm25=bm25,
        embedding_model=embedding_model,
        load_errors=errors,
    )
    logger.info(
        "Resources loaded: procedures=%s metas=%s faiss=%s bm25=%s embedding=%s",
        len(procedures),
        len(metadatas),
        faiss_index is not None,
        bm25 is not None,
        embedding_model is not None,
    )
    for error in errors:
        logger.warning("Resource load warning: %s", error)
    return resources


def _resolve_resource_paths(settings: Settings, errors: list[str]) -> dict[str, Path]:
    if settings.is_local_data:
        paths = {key: settings.data_dir / filename for key, filename in RESOURCE_FILES.items()}
        raw_fallback = Path("static/data/toan_bo_du_lieu_final.json")
        if not paths["raw"].exists() and raw_fallback.exists():
            paths["raw"] = raw_fallback
        return paths

    paths: dict[str, Path] = {}
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        errors.append(f"huggingface_hub is not available: {exc}")
        hf_hub_download = None

    for key, filename in RESOURCE_FILES.items():
        local_candidate = settings.data_dir / filename
        if local_candidate.exists():
            paths[key] = local_candidate
            continue

        if hf_hub_download is not None:
            try:
                downloaded = hf_hub_download(
                    repo_id=settings.hf_repo_id,
                    filename=filename,
                    repo_type=settings.hf_repo_type,
                    cache_dir=str(settings.cache_dir / "huggingface"),
                    local_files_only=settings.hf_local_files_only,
                )
                paths[key] = Path(downloaded)
                continue
            except Exception as exc:
                mode = "cache-only" if settings.hf_local_files_only else "download/cache"
                errors.append(f"Could not resolve {filename} from Hugging Face ({mode}): {exc}")

        if key == "raw":
            raw_fallback = Path("static/data/toan_bo_du_lieu_final.json")
            if raw_fallback.exists():
                paths[key] = raw_fallback
    return paths


def _load_raw_json(path: Path | None, errors: list[str]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        errors.append("Raw procedure JSON is not available.")
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        errors.append(f"Raw JSON is not a list: {path}")
    except Exception as exc:
        errors.append(f"Failed to load raw JSON {path}: {exc}")
    return []


def _load_metadatas(path: Path | None, errors: list[str]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        with gzip.open(path, "rb") as file:
            loaded = pickle.load(file)
        if isinstance(loaded, dict):
            loaded = loaded.get("corpus", loaded.get("metadatas", []))
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
        errors.append(f"Metadata file has unsupported shape: {path}")
    except Exception as exc:
        errors.append(f"Failed to load metadata {path}: {exc}")
    return []


def _load_pickle(path: Path | None, label: str, errors: list[str]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as file:
            return pickle.load(file)
    except Exception as exc:
        errors.append(f"Failed to load {label} {path}: {exc}")
        return None


def _load_faiss(path: Path | None, errors: list[str]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        import faiss

        return faiss.read_index(str(path))
    except Exception as exc:
        errors.append(f"Failed to load FAISS index {path}: {exc}")
        return None


def _load_embedding_model(settings: Settings, errors: list[str]) -> Any:
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(settings.emb_model, device=device)
    except Exception as exc:
        errors.append(f"Failed to load embedding model {settings.emb_model}: {exc}")
        return None
