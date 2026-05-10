"""Détecteur hybride : Isolation Forest + règles métier déterministes.
"""
import pandas as pd
import numpy as np
from typing import Optional

from ..utils.logging import logger


class Rule:
    """Règle de détection : nom, fonction de scoring, sévérité."""
    def __init__(self, name: str, severity: str = "medium"):
        self.name = name
        self.severity = severity

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class ErrorRule(Rule):
    """Alerte sur ERROR/CRITICAL seulement au-dessus d'un seuil baseline.

    Le baseline normal inclut ~30 ERRORs/fenêtre (cinder-backup heartbeat).
    Le seuil par défaut (50) prend en compte cette activité normale.
    Un seul CRITICAL déclenche toujours, sans seuil.
    """
    def __init__(self, error_threshold: int = 50):
        super().__init__("any_error_or_critical", "high")
        self.error_threshold = error_threshold

    def score(self, X: pd.DataFrame) -> np.ndarray:
        n_err = X.get("count_severity_error", pd.Series(0, index=X.index)).fillna(0).values
        n_crit = X.get("count_severity_critical", pd.Series(0, index=X.index)).fillna(0).values
        crit_score = np.where(n_crit > 0, 0.95, 0.0)
        excess = np.maximum(n_err - self.error_threshold, 0)
        err_score = np.where(excess > 0, 0.7 + 0.3 * (1 - np.exp(-excess / 20)), 0.0)
        return np.maximum(err_score, crit_score)


class HttpServerErrorRule(Rule):
    """Tout 5xx HTTP > 0 → score élevé."""
    def __init__(self):
        super().__init__("http_5xx_present", "high")

    def score(self, X: pd.DataFrame) -> np.ndarray:
        n_5xx = X.get("count_status_5xx", pd.Series(0, index=X.index)).fillna(0).values
        return np.where(n_5xx > 0, 0.8 + 0.2 * (1 - np.exp(-n_5xx / 3)), 0.0)


class ServiceSilenceRule(Rule):
    """Activité présente mais nombre de services anormalement bas."""
    def __init__(self, min_expected_services: int = 13):
        super().__init__("service_silence", "high")
        self.min_expected = min_expected_services

    def score(self, X: pd.DataFrame) -> np.ndarray:
        active = X.get("count_total", pd.Series(0, index=X.index)).fillna(0).values >= 100
        n_svc = X.get("unique_services", pd.Series(0, index=X.index)).fillna(0).values
        deficit = np.maximum(self.min_expected - n_svc, 0)
        return np.where(active, 1 - np.exp(-deficit / 1.5), 0.0)


class HighLatencyRule(Rule):
    """Latence P95 ou P99 anormalement haute."""
    def __init__(self, p95_threshold_ms: float = 200.0, p99_threshold_ms: float = 500.0):
        super().__init__("high_latency", "medium")
        self.p95_thresh = p95_threshold_ms
        self.p99_thresh = p99_threshold_ms

    def score(self, X: pd.DataFrame) -> np.ndarray:
        p95 = X.get("latency_p95", pd.Series(0, index=X.index)).fillna(0).values
        p99 = X.get("latency_p99", pd.Series(0, index=X.index)).fillna(0).values
        score_p95 = np.clip((p95 - self.p95_thresh) / 1000, 0, 1)
        score_p99 = np.clip((p99 - self.p99_thresh) / 5000, 0, 1)
        return np.maximum(score_p95, score_p99) * 0.8


class HttpClientErrorBurstRule(Rule):
    """Burst de 4xx (>5 par fenêtre)."""
    def __init__(self, threshold: int = 5):
        super().__init__("http_4xx_burst", "medium")
        self.threshold = threshold

    def score(self, X: pd.DataFrame) -> np.ndarray:
        n = X.get("count_status_4xx", pd.Series(0, index=X.index)).fillna(0).values
        return np.where(n >= self.threshold,
                        0.5 + 0.5 * (1 - np.exp(-(n - self.threshold) / 10)), 0.0)


class HybridDetector:
    """Combine un AnomalyDetector (ML) avec des règles métier.

    Score final = max(score_ML_proba, max(scores_rules))
    """

    def __init__(self, ml_detector=None, rules: Optional[list[Rule]] = None):
        self.ml_detector = ml_detector
        self.rules = rules or self._default_rules()

    @staticmethod
    def _default_rules() -> list[Rule]:
        return [
            ErrorRule(),
            HttpServerErrorRule(),
            ServiceSilenceRule(),
            HighLatencyRule(),
            HttpClientErrorBurstRule(),
        ]

    def score(self, X: pd.DataFrame) -> dict:
        """Retourne un dict détaillé : score ML, score par règle, score final."""
        out = {}

        if self.ml_detector is not None:
            ml_scores = self.ml_detector.score(X)
            ml_probs = self.ml_detector.score_to_probability(ml_scores)
            out["ml_score"] = ml_scores
            out["ml_proba"] = ml_probs
        else:
            out["ml_score"] = np.zeros(len(X))
            out["ml_proba"] = np.zeros(len(X))

        rule_scores_dict = {}
        for rule in self.rules:
            rule_scores_dict[rule.name] = rule.score(X)
        out["rule_scores"] = rule_scores_dict

        rules_max = np.zeros(len(X))
        for s in rule_scores_dict.values():
            rules_max = np.maximum(rules_max, s)
        out["rules_max"] = rules_max
        out["final_score"] = np.maximum(out["ml_proba"], rules_max)

        return out

    def predict(self, X: pd.DataFrame, ml_threshold: float = 0.90,
                rules_threshold: float = 0.5) -> dict:
        """Prédit anomalie selon deux seuils découplés.

        Une fenêtre est anormale si :
          - ml_proba > ml_threshold, OU
          - une règle métier dépasse rules_threshold

        Retourne aussi les contributions ML/rules pour explicabilité.
        """
        sc = self.score(X)

        ml_pos = sc["ml_proba"] > ml_threshold
        rules_pos = sc["rules_max"] > rules_threshold
        sc["predicted"] = (ml_pos | rules_pos).astype(int)
        sc["decision_source"] = np.where(
            ml_pos & rules_pos, "both",
            np.where(ml_pos, "ml_only",
                     np.where(rules_pos, "rules_only", "none"))
        )

        triggered_rules = []
        for i in range(len(X)):
            triggered = [
                rule.name
                for rule in self.rules
                if sc["rule_scores"][rule.name][i] > rules_threshold
            ]
            triggered_rules.append(triggered)
        sc["triggered_rules"] = triggered_rules

        return sc
