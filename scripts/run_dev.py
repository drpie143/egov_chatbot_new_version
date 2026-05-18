from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egov_bot.app import create_app  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    app = create_app()
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")


if __name__ == "__main__":
    main()

