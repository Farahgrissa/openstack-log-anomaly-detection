"""Scaling des features par groupe.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple

from ..utils.logging import logger


# Préfixes/noms de features par type
COUNT_PREFIXES = ("count_", "unique_", "delta_count_", "delta_unique_")
RATIO_PREFIXES = ("ratio_", "delta_ratio_")
LATENCY_PREFIXES = ("latency_", "delta_latency_")
# Le reste : binaires (is_*), cycliques (hour_sin/cos, dow_sin/cos), entropies, scalaires
# → ne sont PAS scalés
NO_SCALE_PREFIXES = ("hour_", "dow_", "is_", "minute_", "max_severity",
                     "severity_entropy", "template_entropy", "top_template_dominance",
                     "new_templates_count", "unknown_templates_count")


def categorize_features(feature_cols: list[str]) -> dict[str, list[str]]:
    """Catégorise les features par type pour scaling adapté."""
    cats = {"counts": [], "ratios": [], "latency": [], "no_scale": []}
    for col in feature_cols:
        if col.startswith(LATENCY_PREFIXES):
            cats["latency"].append(col)
        elif col.startswith(RATIO_PREFIXES):
            cats["ratios"].append(col)
        elif col.startswith(NO_SCALE_PREFIXES) or col in (
            "max_severity", "severity_entropy", "template_entropy",
            "top_template_dominance", "new_templates_count",
            "unknown_templates_count", "is_weekend", "is_business_hours",
            "minute_of_day", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ):
            cats["no_scale"].append(col)
        elif col.startswith(COUNT_PREFIXES):
            cats["counts"].append(col)
        else:
            # Default : counts (sécurité)
            cats["counts"].append(col)
    return cats


class FeatureScaler:
    """Encapsule 3 scalers + colonnes non-scalées.

    Permet fit + transform + sérialisation.
    """

    def __init__(self):
        self.scaler_counts = RobustScaler()
        self.scaler_ratios = RobustScaler()
        self.scaler_latency = RobustScaler()
        self.categories: dict[str, list[str]] = {}
        self.fitted = False

    def fit(self, X: pd.DataFrame, feature_cols: list[str]) -> "FeatureScaler":
        self.categories = categorize_features(feature_cols)
        if self.categories["counts"]:
            self.scaler_counts.fit(X[self.categories["counts"]].astype(float))
        if self.categories["ratios"]:
            self.scaler_ratios.fit(X[self.categories["ratios"]].astype(float))
        if self.categories["latency"]:
            self.scaler_latency.fit(X[self.categories["latency"]].astype(float))
        self.fitted = True
        logger.info(f"FeatureScaler fitted: "
                    f"{len(self.categories['counts'])} counts, "
                    f"{len(self.categories['ratios'])} ratios, "
                    f"{len(self.categories['latency'])} latency, "
                    f"{len(self.categories['no_scale'])} no-scale")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforme X et retourne un numpy array (ordre : counts, ratios, latency, no_scale)."""
        if not self.fitted:
            raise RuntimeError("FeatureScaler not fitted")

        parts = []
        ordered_cols = []
        if self.categories["counts"]:
            parts.append(self.scaler_counts.transform(X[self.categories["counts"]].astype(float)))
            ordered_cols.extend(self.categories["counts"])
        if self.categories["ratios"]:
            parts.append(self.scaler_ratios.transform(X[self.categories["ratios"]].astype(float)))
            ordered_cols.extend(self.categories["ratios"])
        if self.categories["latency"]:
            parts.append(self.scaler_latency.transform(X[self.categories["latency"]].astype(float)))
            ordered_cols.extend(self.categories["latency"])
        if self.categories["no_scale"]:
            parts.append(X[self.categories["no_scale"]].astype(float).values)
            ordered_cols.extend(self.categories["no_scale"])

        return np.hstack(parts), ordered_cols

    def fit_transform(self, X: pd.DataFrame, feature_cols: list[str]) -> Tuple[np.ndarray, list[str]]:
        self.fit(X, feature_cols)
        return self.transform(X)
