from __future__ import annotations

import os

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:7860").rstrip("/")


def main() -> None:
    health = requests.get(f"{BASE_URL}/health", timeout=20)
    print("/health", health.status_code, health.text[:500])
    health.raise_for_status()

    search = requests.get(f"{BASE_URL}/search", params={"q": "đăng ký khai sinh", "limit": 3}, timeout=30)
    print("/search", search.status_code, search.text[:500])
    search.raise_for_status()

    payload = {"question": "Đăng ký khai sinh cần giấy tờ gì?", "session_id": "smoke-test"}
    chat = requests.post(f"{BASE_URL}/chat", json=payload, timeout=180)
    print("/chat", chat.status_code, chat.text[:800])
    chat.raise_for_status()

    feedback = requests.post(
        f"{BASE_URL}/feedback",
        json={"session_id": "smoke-test", "rating": "neutral", "comment": "smoke"},
        timeout=20,
    )
    print("/feedback", feedback.status_code, feedback.text[:300])
    feedback.raise_for_status()


if __name__ == "__main__":
    main()

