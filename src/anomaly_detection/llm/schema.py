"""Schema de l'alerte enrichie (AnomalyAlert + analyse LLM)."""
from dataclasses import dataclass, field, asdict


@dataclass
class LLMHypothesis:
    """Une hypothese de cause racine produite par le LLM."""
    cause: str
    confidence: str  # low | medium | high
    evidence_log_indices: list = field(default_factory=list)


@dataclass
class LLMAnalysis:
    """Analyse LLM structuree d'une alerte."""
    summary: str
    primary_hypothesis: LLMHypothesis
    alternative_hypotheses: list = field(default_factory=list)
    affected_services: list = field(default_factory=list)
    suggested_investigations: list = field(default_factory=list)
    severity_assessment: str = "medium"
    requires_human_attention: bool = True

    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    tokens_used: int = None
    fallback_used: bool = False
    grounding_valid: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["primary_hypothesis"] = asdict(self.primary_hypothesis)
        d["alternative_hypotheses"] = [asdict(h) for h in self.alternative_hypotheses]
        return d


@dataclass
class EnrichedAlert:
    """AnomalyAlert + analyse LLM."""
    alert: dict
    llm_analysis: LLMAnalysis

    def to_dict(self) -> dict:
        return {
            "alert": self.alert,
            "llm_analysis": self.llm_analysis.to_dict(),
        }
