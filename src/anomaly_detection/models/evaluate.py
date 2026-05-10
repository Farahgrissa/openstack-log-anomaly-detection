"""Évaluation des modèles sur le test set anormal.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from ..utils.config import resolve_path
from ..utils.logging import logger
from ..utils.io import load_df
from .isolation_forest import AnomalyDetector


def load_test_data(features_path: str | Path,
                   labels_path: str | Path) -> pd.DataFrame:
    """Charge le test set + labels et merge sur window_id."""
    features = load_df(features_path)
    labels = pd.read_csv(labels_path)
    df = features.merge(
        labels[["window_id", "is_anomaly", "anomaly_type", "anomaly_severity"]],
        on="window_id", how="left"
    )
    df["is_anomaly"] = df["is_anomaly"].fillna(0).astype(int)
    df["anomaly_type"] = df["anomaly_type"].fillna("")
    return df


def evaluate_model(
    model: AnomalyDetector,
    test_df: pd.DataFrame,
    label_col: str = "is_anomaly",
    type_col: str = "anomaly_type",
    only_active: bool = True,
) -> dict:
    """Évalue un modèle sur un test set labellisé.

    Returns : dict avec AUC, Precision, Recall, F1, métriques par type, scores bruts.
    """
    # Optionnel : filtrer fenêtres actives (sinon on évalue sur le bruit nuit)
    if only_active and "count_total" in test_df.columns:
        df = test_df[test_df["count_total"] > 0].reset_index(drop=True)
        logger.info(f"[{model.name}] Active windows: {len(df)}/{len(test_df)}")
    else:
        df = test_df.reset_index(drop=True)

    if len(df) == 0:
        return {"error": "no active windows"}

    y_true = df[label_col].values
    scores = model.score(df)
    y_pred = (scores > model.threshold).astype(int)

    n_anom = int(y_true.sum())
    n_normal = len(y_true) - n_anom
    logger.info(f"[{model.name}] Test : {n_anom} anomaly + {n_normal} normal windows")

    if n_anom == 0:
        logger.warning(f"[{model.name}] No anomaly windows in test set")
        return {"error": "no anomalies"}

    # Métriques globales
    metrics = {
        "name": model.name,
        "n_total": len(df),
        "n_anomaly": n_anom,
        "n_normal": n_normal,
        "threshold": model.threshold,
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "auc_pr": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_predicted_anomaly": int(y_pred.sum()),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "false_positive_rate": float(fp / max(1, fp + tn)),
        })

    # Métriques par TYPE d'anomalie
    by_type = {}
    for atype in df.loc[df[label_col] == 1, type_col].unique():
        if atype == "":
            continue
        # Recall = TP / (TP + FN) sur cette catégorie
        mask_type = (df[type_col] == atype)
        n_in_type = int(mask_type.sum())
        n_detected_in_type = int(((df[type_col] == atype) & (y_pred == 1)).sum())
        recall_type = n_detected_in_type / max(1, n_in_type)
        by_type[atype] = {
            "n_windows": n_in_type,
            "n_detected": n_detected_in_type,
            "recall": float(recall_type),
            "score_p50": float(np.percentile(scores[mask_type], 50)),
            "score_p95": float(np.percentile(scores[mask_type], 95)),
        }
    metrics["by_type"] = by_type

    return metrics


def evaluate_all(
    models_dir: str | Path,
    test_features_path: str | Path,
    test_labels_path: str | Path,
    only_active: bool = True,
) -> dict:
    """Évalue tous les modèles sur le test set."""
    models_dir = Path(models_dir)
    logger.info(f"Loading test data...")
    test_df = load_test_data(test_features_path, test_labels_path)

    n_anom_total = int(test_df["is_anomaly"].sum())
    logger.info(f"Test set: {len(test_df)} windows, {n_anom_total} anomaly windows")
    logger.info(f"Anomaly types: {sorted(test_df.loc[test_df['is_anomaly']==1, 'anomaly_type'].unique())}")

    results = {}

    # Modèle global
    global_path = models_dir / "model_global.joblib"
    if global_path.exists():
        logger.info("=" * 60)
        logger.info("Evaluating GLOBAL model (ML only)")
        logger.info("=" * 60)
        global_model = AnomalyDetector.load(global_path)
        results["global"] = evaluate_model(global_model, test_df, only_active=only_active)
        _log_metrics(results["global"])

    # Modèles par groupe
    for group_path in sorted(models_dir.glob("model_group_*.joblib")):
        group_name = group_path.stem.replace("model_group_", "")
        logger.info("=" * 60)
        logger.info(f"Evaluating GROUP model: {group_name} (ML only)")
        logger.info("=" * 60)
        model = AnomalyDetector.load(group_path)
        results[f"group_{group_name}"] = evaluate_model(model, test_df, only_active=only_active)
        _log_metrics(results[f"group_{group_name}"])

    # Ensemble
    if len(results) > 1:
        logger.info("=" * 60)
        logger.info("Evaluating ENSEMBLE (max of all model scores)")
        logger.info("=" * 60)
        results["ensemble_max"] = _evaluate_ensemble(models_dir, test_df, only_active=only_active)
        _log_metrics(results["ensemble_max"])

    # Détecteur HYBRIDE (ML + règles métier)
    if global_path.exists():
        logger.info("=" * 60)
        logger.info("Evaluating HYBRID detector (ML + business rules)")
        logger.info("=" * 60)
        results["hybrid"] = _evaluate_hybrid(global_path, test_df, only_active=only_active)
        _log_metrics(results["hybrid"])

    return results


def _evaluate_hybrid(
    ml_model_path: Path,
    test_df: pd.DataFrame,
    only_active: bool = True,
) -> dict:
    """Évalue le détecteur hybride (ML + rules)."""
    from .hybrid import HybridDetector

    if only_active and "count_total" in test_df.columns:
        df = test_df[test_df["count_total"] > 0].reset_index(drop=True)
    else:
        df = test_df.reset_index(drop=True)

    if len(df) == 0:
        return {"error": "no active windows"}

    y_true = df["is_anomaly"].values
    if y_true.sum() == 0:
        return {"error": "no anomalies"}

    ml_model = AnomalyDetector.load(ml_model_path)
    hybrid = HybridDetector(ml_detector=ml_model)
    sc = hybrid.predict(df, ml_threshold=0.90, rules_threshold=0.5)

    y_pred = sc["predicted"]
    final_score = sc["final_score"]

    metrics = {
        "name": "hybrid",
        "n_total": len(df),
        "n_anomaly": int(y_true.sum()),
        "n_normal": int(len(y_true) - y_true.sum()),
        "auc_roc": float(roc_auc_score(y_true, final_score)),
        "auc_pr": float(average_precision_score(y_true, final_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_predicted_anomaly": int(y_pred.sum()),
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "false_positive_rate": float(fp / max(1, fp + tn)),
        })

    # Per-type
    by_type = {}
    for atype in df.loc[df["is_anomaly"] == 1, "anomaly_type"].unique():
        if atype == "":
            continue
        mask = (df["anomaly_type"] == atype)
        n = int(mask.sum())
        n_det = int((mask & (y_pred == 1)).sum())
        by_type[atype] = {
            "n_windows": n,
            "n_detected": n_det,
            "recall": float(n_det / max(1, n)),
        }
    metrics["by_type"] = by_type

    # Source de décision
    sources = pd.Series(sc["decision_source"]).value_counts().to_dict()
    metrics["decision_sources"] = {k: int(v) for k, v in sources.items()}

    return metrics


def _log_metrics(metrics: dict) -> None:
    """Log lisible des métriques."""
    if "error" in metrics:
        logger.warning(f"  Skip: {metrics['error']}")
        return
    logger.info(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR  : {metrics['auc_pr']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall   : {metrics['recall']:.3f}")
    logger.info(f"  F1       : {metrics['f1']:.3f}")
    if "fp" in metrics:
        logger.info(f"  Confusion: TN={metrics['tn']} FP={metrics['fp']} "
                    f"FN={metrics['fn']} TP={metrics['tp']}")
        logger.info(f"  FPR      : {metrics['false_positive_rate']:.4f}")
    if "by_type" in metrics:
        logger.info("  Per-anomaly-type recall:")
        for atype, sub in sorted(metrics["by_type"].items()):
            logger.info(f"    {atype:<32s} | n={sub['n_windows']:>3d} "
                        f"| detected={sub['n_detected']:>3d} "
                        f"| recall={sub['recall']:.3f}")


def _evaluate_ensemble(
    models_dir: Path,
    test_df: pd.DataFrame,
    only_active: bool = True,
) -> dict:
    """Évalue un ensemble : score = max des scores normalisés des modèles."""
    if only_active and "count_total" in test_df.columns:
        df = test_df[test_df["count_total"] > 0].reset_index(drop=True)
    else:
        df = test_df.reset_index(drop=True)

    y_true = df["is_anomaly"].values
    if len(df) == 0 or y_true.sum() == 0:
        return {"error": "empty or no anomalies"}

    # Charger tous les modèles et calculer scores normalisés
    all_scores = []
    triggered_per_window = np.zeros(len(df), dtype=int)
    for path in sorted(models_dir.glob("model_*.joblib")):
        model = AnomalyDetector.load(path)
        scores = model.score(df)
        # Normaliser : (score - threshold) / (max(scores) - threshold) clip à [0, 1+]
        # Plus simple : convertir en proba via score_to_probability
        probs = model.score_to_probability(scores)
        all_scores.append(probs)
        triggered_per_window += (scores > model.threshold).astype(int)

    # Score d'ensemble = max des probabilités
    ensemble_scores = np.max(all_scores, axis=0)
    # Prediction d'ensemble : flagged si AU MOINS UN modèle a flaggé (= triggered_per_window > 0)
    y_pred = (triggered_per_window > 0).astype(int)

    metrics = {
        "name": "ensemble_max",
        "n_total": len(df),
        "n_anomaly": int(y_true.sum()),
        "n_normal": int(len(y_true) - y_true.sum()),
        "auc_roc": float(roc_auc_score(y_true, ensemble_scores)),
        "auc_pr": float(average_precision_score(y_true, ensemble_scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_predicted_anomaly": int(y_pred.sum()),
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "false_positive_rate": float(fp / max(1, fp + tn)),
        })

    by_type = {}
    for atype in df.loc[df["is_anomaly"] == 1, "anomaly_type"].unique():
        if atype == "":
            continue
        mask = (df["anomaly_type"] == atype)
        n_in_type = int(mask.sum())
        n_detected = int((mask & (y_pred == 1)).sum())
        by_type[atype] = {
            "n_windows": n_in_type,
            "n_detected": n_detected,
            "recall": float(n_detected / max(1, n_in_type)),
        }
    metrics["by_type"] = by_type

    return metrics
