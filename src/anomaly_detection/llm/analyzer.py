"""Orchestrateur LLM : prompt + provider + failover + cache.

Utilisation simple :
    analyzer = LLMAnalyzer.from_config()
    analysis = analyzer.analyze(alert_dict)

Le provider primaire est tenté en premier. Si échec → fallback. Si tous
les LLM échouent → template déterministe.
"""
import os
from pathlib import Path
from typing import Optional

from .base import LLMClient, LLMResponse
from .providers import GeminiClient, GroqClient, TemplateClient
from .prompt_builder import (
    SYSTEM_PROMPT, format_alert_for_prompt, get_response_schema
)
from .validator import build_analysis
from .cache import LLMCache
from .schema import LLMAnalysis, LLMHypothesis
from ..utils.config import get_config, resolve_path
from ..utils.logging import logger


# Mapping nom -> classe
PROVIDER_CLASSES = {
    "gemini": GeminiClient,
    "groq": GroqClient,
    "template": TemplateClient,
}

# Mapping nom -> variable d'environnement de la cle API
PROVIDER_ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "template": None,
}


def load_dotenv_if_present(env_path=None) -> None:
    """Charge les variables d'environnement depuis un fichier .env si present.

    Implementation maison (pas de dependance python-dotenv).
    """
    if env_path is None:
        env_path = resolve_path(".env")
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _build_provider(name: str, model: str, **kwargs) -> Optional[LLMClient]:
    """Instancie un provider a partir de son nom."""
    if name not in PROVIDER_CLASSES:
        logger.error(f"Unknown provider: {name}")
        return None
    cls = PROVIDER_CLASSES[name]
    env_key = PROVIDER_ENV_KEYS.get(name)
    api_key = os.environ.get(env_key, "") if env_key else ""
    if name != "template" and not api_key:
        logger.warning(f"No API key found for {name} (env var {env_key} unset)")
        return None
    try:
        return cls(api_key=api_key, model=model, **kwargs)
    except Exception as e:
        logger.error(f"Failed to instantiate {name}: {e}")
        return None


class LLMAnalyzer:
    """Orchestrateur LLM : essaie primary, puis fallback, puis template."""

    def __init__(
        self,
        primary: Optional[LLMClient],
        fallback: Optional[LLMClient] = None,
        template: Optional[LLMClient] = None,
        cache: Optional[LLMCache] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.template = template or TemplateClient()
        self.cache = cache

    @classmethod
    def from_config(cls, config: dict = None) -> "LLMAnalyzer":
        """Construit un analyzer depuis la config YAML."""
        load_dotenv_if_present()
        if config is None:
            config = get_config()
        llm_cfg = config.get("llm", {})
        if not llm_cfg.get("enabled", True):
            logger.info("LLM disabled in config -- using template-only")
            return cls(primary=None, fallback=None, template=TemplateClient())

        primary_name = llm_cfg.get("primary_provider", "gemini")
        primary_model = llm_cfg.get("primary_model", "gemini-2.0-flash")
        fallback_name = llm_cfg.get("fallback_provider", "groq")
        fallback_model = llm_cfg.get("fallback_model", "llama-3.3-70b-versatile")

        common = {
            "timeout_seconds": llm_cfg.get("timeout_seconds", 30),
            "temperature": llm_cfg.get("temperature", 0.1),
            "max_tokens": llm_cfg.get("max_tokens_response", 1500),
        }

        primary = _build_provider(primary_name, primary_model, **common)
        fallback = _build_provider(fallback_name, fallback_model, **common)
        template = TemplateClient()

        cache = None
        if llm_cfg.get("cache_enabled", True):
            cache_dir = resolve_path("data/llm_cache")
            ttl = llm_cfg.get("cache_ttl_hours", 24) * 3600
            cache = LLMCache(cache_dir, ttl_seconds=ttl)

        if primary:
            logger.info(f"LLM primary: {primary}")
        if fallback:
            logger.info(f"LLM fallback: {fallback}")
        if not primary and not fallback:
            logger.warning("No LLM provider available -- using template only")

        return cls(primary=primary, fallback=fallback, template=template, cache=cache)

    def analyze(self, alert: dict) -> LLMAnalysis:
        """Analyse une alerte. Retourne LLMAnalysis (jamais None)."""
        alert_id = alert.get("alert_id", "")
        n_top_logs = len(alert.get("top_logs", []))

        # Cache hit ?
        if self.cache and alert_id:
            cached = self.cache.get(alert_id)
            if cached is not None:
                logger.debug(f"LLM cache hit for {alert_id}")
                return self._analysis_from_cached(cached)

        user_prompt = format_alert_for_prompt(alert)
        schema_hint = get_response_schema()

        for client, is_fallback in self._provider_chain():
            if client is None or not client.is_available():
                continue
            response: LLMResponse = client.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema_hint=schema_hint,
            )
            if response.error:
                logger.warning(f"[{client.provider_name}] failed: {response.error}")
                continue

            tokens_total = (response.tokens_input or 0) + (response.tokens_output or 0)
            analysis = build_analysis(
                response_text=response.text,
                n_top_logs=n_top_logs,
                provider=response.provider,
                model=response.model,
                latency_ms=response.latency_ms,
                tokens_used=tokens_total if tokens_total > 0 else None,
                fallback_used=is_fallback,
            )

            if self.cache and alert_id and analysis.provider != "template":
                try:
                    self.cache.set(alert_id, analysis.to_dict())
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")

            return analysis

        return build_analysis(
            response_text="",
            n_top_logs=n_top_logs,
            provider="none",
            model="none",
            latency_ms=0,
        )

    def _provider_chain(self):
        chain = []
        if self.primary:
            chain.append((self.primary, False))
        if self.fallback:
            chain.append((self.fallback, True))
        chain.append((self.template, True))
        return chain

    @staticmethod
    def _analysis_from_cached(cached: dict) -> LLMAnalysis:
        primary = cached.get("primary_hypothesis", {})
        return LLMAnalysis(
            summary=cached.get("summary", ""),
            primary_hypothesis=LLMHypothesis(
                cause=primary.get("cause", ""),
                confidence=primary.get("confidence", "low"),
                evidence_log_indices=primary.get("evidence_log_indices", []),
            ),
            alternative_hypotheses=[
                LLMHypothesis(
                    cause=h.get("cause", ""),
                    confidence=h.get("confidence", "low"),
                    evidence_log_indices=h.get("evidence_log_indices", []),
                )
                for h in cached.get("alternative_hypotheses", [])
            ],
            affected_services=cached.get("affected_services", []),
            suggested_investigations=cached.get("suggested_investigations", []),
            severity_assessment=cached.get("severity_assessment", "medium"),
            requires_human_attention=cached.get("requires_human_attention", True),
            provider=cached.get("provider", ""),
            model=cached.get("model", ""),
            latency_ms=cached.get("latency_ms", 0.0),
            tokens_used=cached.get("tokens_used"),
            fallback_used=cached.get("fallback_used", False),
            grounding_valid=cached.get("grounding_valid", True),
        )
