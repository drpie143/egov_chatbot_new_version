from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    google_api_key_2: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY_2"))
    hf_repo_id: str = field(default_factory=lambda: os.getenv("HF_REPO_ID", "HungBB/egov-bot-data"))
    hf_repo_type: str = field(default_factory=lambda: os.getenv("HF_REPO_TYPE", "dataset"))
    emb_model: str = field(default_factory=lambda: os.getenv("EMB_MODEL", "AITeamVN/Vietnamese_Embedding"))
    genai_model: str = field(default_factory=lambda: os.getenv("GENAI_MODEL", "gemini-2.5-flash"))
    port: int = field(default_factory=lambda: _int_env("PORT", 7860))
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "local"))
    flask_env: str = field(default_factory=lambda: os.getenv("FLASK_ENV", "production"))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", ".cache/egov_data")))
    sqlite_path: Path = field(default_factory=lambda: Path(os.getenv("SQLITE_PATH", "user_data/egov_bot.db")))
    data_source: str = field(default_factory=lambda: os.getenv("DATA_SOURCE", "hf").strip().lower())
    hf_local_files_only: bool = field(default_factory=lambda: _bool_env("HF_LOCAL_FILES_ONLY", False))
    cors_origins: list[str] = field(default_factory=lambda: _csv_env("CORS_ORIGINS", "http://localhost:7860,http://127.0.0.1:7860"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    enable_debug_retrieval: bool = field(default_factory=lambda: _bool_env("ENABLE_DEBUG_RETRIEVAL", False))
    top_k: int = field(default_factory=lambda: _int_env("TOP_K", 3))
    search_limit: int = field(default_factory=lambda: _int_env("SEARCH_LIMIT", 10))
    faiss_candidates: int = field(default_factory=lambda: _int_env("FAISS_CANDIDATES", 50))
    bm25_candidates: int = field(default_factory=lambda: _int_env("BM25_CANDIDATES", 50))
    answer_cache_ttl_seconds: int = field(default_factory=lambda: _int_env("ANSWER_CACHE_TTL_SECONDS", 3600))
    answer_cache_max_items: int = field(default_factory=lambda: _int_env("ANSWER_CACHE_MAX_ITEMS", 512))

    def ensure_directories(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_local_data(self) -> bool:
        return self.data_source == "local"

    @property
    def google_api_keys(self) -> list[str]:
        return [key for key in [self.google_api_key, self.google_api_key_2] if key]


def load_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()

    cache_root = settings.cache_dir.resolve()
    hf_home = cache_root / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return settings
