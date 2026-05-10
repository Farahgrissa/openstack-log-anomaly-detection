"""Schéma d'alerte versionné (AnomalyAlert, FeatureContribution, LogReference)."""
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class FeatureContribution:
    """Une feature qui contribue au score d'anomalie."""
    feature: str
    value: float
    baseline_value: float
    contribution: float       # signe + magnitude
    contribution_pct: float    # % du total des contribs

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LogReference:
    """Un log représentatif inclus dans l'alerte (avec contexte)."""
    timestamp: str
    programname: str
    log_level: str | None
    severity: int
    payload: str
    selection_reason: str
    request_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    python_module: str | None = None
    http_method: str | None = None
    http_status: int | None = None
    http_duration_ms: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnomalyAlert:
    """Alerte d'anomalie détectée pour une fenêtre temporelle."""
    # Métadonnées
    schema_version: str
    alert_id: str
    detected_at: str

    # Contexte temporel
    window_start: str
    window_end: str

    # Scoring
    final_score: float
    ml_score: float
    ml_proba: float
    rules_max_score: float
    severity: str             # low | medium | high | critical
    decision_source: str      # ml_only | rules_only | both

    # Explicabilité
    top_features: list[FeatureContribution]
    triggered_rules: list[str]

    # Logs représentatifs
    top_logs: list[LogReference]

    # Stats fenêtre
    window_stats: dict        # count_total, services_actifs, etc.

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @staticmethod
    def compute_alert_id(window_start: str, top_features: list, decision_source: str) -> str:
        """Calcule un alert_id stable (pour déduplication).

        Fingerprint = hash(window_start + decision_source + top 3 feature names + ranges)
        Ainsi deux fenêtres très similaires (même top features, même décision)
        auront le même alert_id et peuvent être dédupliquées.
        """
        top_feat_names = ",".join(sorted(f.feature for f in top_features[:3]))
        fingerprint = f"{window_start}|{decision_source}|{top_feat_names}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def compute_severity(final_score: float, triggered_rules: list[str]) -> str:
    """Calcule le niveau de sévérité d'une alerte.

    Combine le score numérique et la nature des règles déclenchées.
    """
    # Critical si CRITICAL log ou 5xx
    if "any_error_or_critical" in triggered_rules and final_score > 0.95:
        return "critical"
    if "http_5xx_present" in triggered_rules:
        return "critical"

    # High si score très haut OU règles dangereuses
    if final_score > 0.9:
        return "high"
    if "any_error_or_critical" in triggered_rules:
        return "high"
    if "service_silence" in triggered_rules:
        return "high"
    if "unknown_templates_present" in triggered_rules:
        return "high"

    # Medium si score modéré ou règles modérées
    if final_score > 0.7:
        return "medium"
    if "high_latency" in triggered_rules:
        return "medium"
    if "http_4xx_burst" in triggered_rules:
        return "medium"

    return "low"
