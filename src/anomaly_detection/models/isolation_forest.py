"""Wrapper Isolation Forest : scaling, entraînement par CV, calibration du seuil."""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
import joblib
from pathlib import Path
from typing import Optional

from ..utils.config import get_config
from ..utils.logging import logger
from .scaling import FeatureScaler


class AnomalyDetector:
    """Wrapper Isolation Forest + scaling + calibration."""

    def __init__(self, name: str = "global", **iforest_kwargs):
        """
        Args:
            name : nom du modèle (utile pour logs et sauvegarde)
            **iforest_kwargs : params à passer à IsolationForest (override config)
        """
        config = get_config()
        defaults = config["models"]["isolation_forest"].copy()
        defaults.update(iforest_kwargs)
        self.iforest_params = defaults

        self.name = name
        self.model = IsolationForest(**self.iforest_params)
        self.scaler = FeatureScaler()
        self.feature_cols: list[str] = []
        self.threshold: Optional[float] = None
        self.threshold_percentile: float = config["models"]["threshold_percentile"]
        # Distribution des scores du training (pour calibration empirique)
        self.train_scores: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, feature_cols: list[str]) -> "AnomalyDetector":
        """Entraîne le modèle.

        Étapes :
        1. fit_transform du scaler
        2. fit IsolationForest
        3. cross-validation 5-fold pour calculer scores sur training (sans biais d'overfitting)
        4. calcul du threshold au percentile P
        """
        logger.info(f"[{self.name}] Training on {len(X)} samples × {len(feature_cols)} features")
        self.feature_cols = feature_cols

        # 1. Fit scaler + transform
        X_scaled, _ = self.scaler.fit_transform(X, feature_cols)

        # 2. Cross-validation pour scores non-biaisés
        # On entraîne 5 modèles sur 4/5 des données, on score sur le 1/5 restant
        # Cela donne des scores réalistes (pas du score "sur soi-même")
        cv = KFold(n_splits=5, shuffle=True, random_state=self.iforest_params["random_state"])
        cv_scores = np.zeros(len(X_scaled))
        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_scaled)):
            tmp_model = IsolationForest(**self.iforest_params)
            tmp_model.fit(X_scaled[tr_idx])
            # decision_function : haut = normal, bas = anormal
            # On inverse : haut = anormal
            cv_scores[te_idx] = -tmp_model.decision_function(X_scaled[te_idx])
        self.train_scores = np.sort(cv_scores)
        logger.info(f"[{self.name}] CV scores - "
                    f"P50={np.percentile(cv_scores, 50):.4f} "
                    f"P95={np.percentile(cv_scores, 95):.4f} "
                    f"P99.5={np.percentile(cv_scores, 99.5):.4f}")

        # 3. Fit modèle final sur tout le training
        self.model.fit(X_scaled)

        # 4. Calibration du threshold au percentile P
        self.threshold = float(np.percentile(cv_scores, self.threshold_percentile))
        logger.info(f"[{self.name}] Threshold @ P{self.threshold_percentile}: {self.threshold:.4f}")

        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les scores d'anomalie (haut = plus anormal)."""
        if not self.fitted:
            raise RuntimeError(f"[{self.name}] Not fitted")
        # Les colonnes manquantes (présentes en train mais pas en test) → 0
        # Les colonnes en plus (présentes en test mais pas en train) → ignorées
        for col in self.feature_cols:
            if col not in X.columns:
                X = X.assign(**{col: 0})
        X_scaled, _ = self.scaler.transform(X[self.feature_cols])
        # decision_function : haut = normal → on inverse
        return -self.model.decision_function(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne 1 si anomalie (score > threshold), 0 sinon."""
        scores = self.score(X)
        return (scores > self.threshold).astype(int)

    def score_to_probability(self, scores: np.ndarray) -> np.ndarray:
        """Convertit un score brut en pseudo-probabilité [0, 1].

        Méthode : map au percentile dans la distribution des scores du training.
        Score = 0 → 0% (jamais vu de pire normal). Score >= max_train → 100%.
        """
        if self.train_scores is None:
            raise RuntimeError(f"[{self.name}] No train_scores stored")
        # Pour chaque score, son rang dans train_scores normalisé
        ranks = np.searchsorted(self.train_scores, scores, side="right")
        return ranks / len(self.train_scores)

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modèle complet (sklearn obj + scaler + threshold + meta)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "name": self.name,
            "iforest_params": self.iforest_params,
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "train_scores": self.train_scores,
            "fitted": self.fitted,
        }
        joblib.dump(state, path)
        logger.info(f"[{self.name}] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        path = Path(path)
        state = joblib.load(path)
        det = cls(name=state["name"])
        det.iforest_params = state["iforest_params"]
        det.model = state["model"]
        det.scaler = state["scaler"]
        det.feature_cols = state["feature_cols"]
        det.threshold = state["threshold"]
        det.threshold_percentile = state["threshold_percentile"]
        det.train_scores = state["train_scores"]
        det.fitted = state["fitted"]
        logger.info(f"[{det.name}] Loaded from {path}")
        return det
