"""Provider Groq (Llama 3.3 70B ultra-rapide via LPU).

Avantages :
  - 300+ tokens/sec (le plus rapide en 2026)
  - 14400 req/jour gratuit, 6000 tokens/min
  - API OpenAI-compatible
  - Modèles : llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b

Obtenir une clé : https://console.groq.com/keys
"""
import json
import time
import urllib.request
import urllib.error

from ..base import LLMClient, LLMResponse
from ...utils.logging import logger


class GroqClient(LLMClient):
    """Client Groq via l'endpoint chat completions (OpenAI-compatible)."""

    provider_name = "groq"
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def generate_json(self, system_prompt: str, user_prompt: str,
                       schema_hint: dict = None) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Force JSON output (Groq supporte json_object mode)
            "response_format": {"type": "json_object"},
        }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logger.error(f"[Groq] HTTP {e.code}: {err_body[:300]}")
            raise RuntimeError(f"Groq HTTP {e.code}: {err_body[:200]}")
        except Exception as e:
            logger.error(f"[Groq] error: {e}")
            raise

        latency_ms = int((time.time() - start) * 1000)

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Groq unexpected response: {payload}")

        usage = payload.get("usage", {})
        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw=payload,
        )
