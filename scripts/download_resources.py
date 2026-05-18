from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egov_bot.config import load_settings  # noqa: E402
from egov_bot.data.resource_loader import load_resources  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    resources = load_resources(settings, load_models=False)
    print(
        {
            "procedures": len(resources.procedures),
            "metadatas": len(resources.metadatas),
            "faiss_loaded": resources.faiss_index is not None,
            "bm25_loaded": resources.bm25 is not None,
            "load_errors": resources.load_errors,
        }
    )


if __name__ == "__main__":
    main()

