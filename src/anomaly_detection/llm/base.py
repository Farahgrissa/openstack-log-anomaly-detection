"""Interface abstraite pour les providers LLM."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Reponse normalisee d'un appel LLM."""
    text: str                       # Reponse brute (texte ou JSON-string)
    model: str                      # Nom du modele utilise
    provider: str                   # Nom du provider
    latency_ms: float               # Temps total d'appel
    tokens_input: int = None
    tokens_output: int = None
    error: str = None               # Message d'erreur si echec


class LLMClient(ABC):
    """Interface abstraite pour les providers LLM."""

    provider_name: str = "abstract"

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 30,
                 temperature: float = 0.1, max_tokens: int = 1500):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint=None,
    ) -> LLMResponse:
        ...

    def is_available(self) -> bool:
        return bool(self.api_key)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.provider_name}/{self.model})>"
