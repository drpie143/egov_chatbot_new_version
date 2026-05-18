from __future__ import annotations

import logging

from egov_bot.config import Settings

logger = logging.getLogger(__name__)


class GeminiAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._genai = None

    @property
    def available(self) -> bool:
        return bool(self.settings.google_api_keys)

    def generate(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("GOOGLE_API_KEY is not configured.")

        last_error: Exception | None = None
        for api_key in self.settings.google_api_keys:
            try:
                genai = self._load_genai()
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    self.settings.genai_model,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 2048,
                    },
                )
                response = model.generate_content(prompt)
                text = getattr(response, "text", None)
                return text if text else str(response)
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "429" in message or "quota" in message or "rate" in message:
                    logger.warning("Gemini quota/rate error; trying fallback key if available.")
                    continue
                raise
        raise RuntimeError(f"Gemini generation failed: {last_error}")

    def _load_genai(self):
        if self._genai is None:
            import google.generativeai as genai

            self._genai = genai
        return self._genai
