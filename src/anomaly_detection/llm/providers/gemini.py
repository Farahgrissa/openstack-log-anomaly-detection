"""Provider Google Gemini.

Avantages :
  - 1500 req/jour gratuit, no credit card requis
  - Sortie JSON native (response_mime_type)
  - Contexte 1M tokens
  - Modèles modernes : gemini-2.5-flash, gemini-2.0-flash

Obtenir une clé : https://aistudio.google.com/apikey
"""
import json
import time
import urllib.request
import urllib.error

from ..base import LLMClient, LLMResponse
from ...utils.logging import logger


class GeminiClient(LLMClient):
    """Client Gemini via l'API REST native (pas de SDK lourd)."""

    provider_name = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def generate_json(self, system_prompt: str, user_prompt: str,
                       schema_hint: dict = None) -> LLMResponse:
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        # Gemini : combine system + user en 1 user message
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        body = {
            "contents": [
                {"role": "user", "parts": [{"text": full_prompt}]}
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        if schema_hint:
            body["generationConfig"]["responseSchema"] = schema_hint

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logger.error(f"[Gemini] HTTP {e.code}: {err_body[:300]}")
            raise RuntimeError(f"Gemini HTTP {e.code}: {err_body[:200]}")
        except Exception as e:
            logger.error(f"[Gemini] error: {e}")
            raise

        latency_ms = int((time.time() - start) * 1000)

        # Extraire le texte de réponse
        try:
            content = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Gemini unexpected response: {payload}")

        usage = payload.get("usageMetadata", {})
        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            raw=payload,
        )
