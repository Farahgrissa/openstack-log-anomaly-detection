"""Inference : détection + feature attribution + génération d'alertes structurées."""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from ..utils.logging import logger
from ..utils.io import load_df
from ..models.isolation_forest import AnomalyDetector
from ..models.hybrid import HybridDetector
from ..explain.explain import feature_attribution, select_top_k_logs
from .schema import AnomalyAlert, FeatureContribution, LogReference, compute_severity


def detect_anomalies(
    features_df: pd.DataFrame,
    enriched_logs_df: pd.DataFrame,
    detector: HybridDetector,
    ml_threshold: float = 0.90,
    rules_threshold: float = 0.5,
    top_k_features: int = 10,
    top_k_logs: int = 10,
) -> list[AnomalyAlert]:
    """Détecte les anomalies dans un batch de fenêtres et génère les alertes.
    """
    if features_df.empty:
        return []

    # Score toutes les fenêtres en batch
    sc = detector.predict(features_df, ml_threshold=ml_threshold,
                           rules_threshold=rules_threshold)

    # Extraire les indices des fenêtres anormales
    anomaly_indices = np.where(sc["predicted"] == 1)[0]
    logger.info(f"Detected {len(anomaly_indices)} anomalies out of {len(features_df)} windows")

    alerts = []
    for idx in anomaly_indices:
        try:
            alert = _build_alert(
                features_df, enriched_logs_df, idx, sc, detector,
                top_k_features=top_k_features, top_k_logs=top_k_logs,
            )
            alerts.append(alert)
        except Exception as e:
            logger.warning(f"Failed to build alert for window {idx}: {e}")
            continue

    logger.info(f"Generated {len(alerts)} structured alerts")
    return alerts


def _build_alert(
    features_df: pd.DataFrame,
    enriched_logs_df: pd.DataFrame,
    idx: int,
    sc: dict,
    detector: HybridDetector,
    top_k_features: int = 10,
    top_k_logs: int = 10,
) -> AnomalyAlert:
    """Construit une AnomalyAlert pour une fenêtre donnée."""
    row = features_df.iloc[idx]
    window_start = row["window_start"]
    window_end = row["window_end"]
    if hasattr(window_start, "tz_localize") and window_start.tz is None:
        window_start = window_start.tz_localize("UTC")
        window_end = window_end.tz_localize("UTC")

    # 1. Feature attribution ML (uniquement si on a un ML detector)
    top_features_list: list[FeatureContribution] = []
    if detector.ml_detector is not None:
        single_row_df = features_df.iloc[[idx]].reset_index(drop=True)
        feats = feature_attribution(
            detector.ml_detector,
            single_row_df,
            top_k=top_k_features,
        )
        # Ne garder que les contributions POSITIVES (vraiment causes de l'anomalie)
        feats = [f for f in feats if f["contribution"] > 0]
        top_features_list = [
            FeatureContribution(
                feature=f["feature"],
                value=f["value"],
                baseline_value=f["baseline_value"],
                contribution=f["contribution"],
                contribution_pct=f["contribution_pct"],
            )
            for f in feats
        ]

    # 1.b Compléter avec les features qui ont déclenché les règles métier
    # (ces features sont essentielles mais peuvent ne pas apparaître dans le ML
    # parce qu'elles sont constantes en training)
    triggered = sc.get("triggered_rules", [[]])[idx] if idx < len(sc.get("triggered_rules", [])) else []
    rule_features = _features_from_triggered_rules(triggered, row)
    # Ajouter en tête, sans dupliquer
    existing_names = {f.feature for f in top_features_list}
    for rf in rule_features:
        if rf.feature not in existing_names:
            top_features_list.insert(0, rf)
            existing_names.add(rf.feature)
    # Tronquer au top_k_features
    top_features_list = top_features_list[:top_k_features]

    # 2. Top K logs (basé sur les top features)
    top_logs_list: list[LogReference] = []
    if not enriched_logs_df.empty and top_features_list:
        feat_dicts = [f.to_dict() for f in top_features_list]
        selected_logs = select_top_k_logs(
            enriched_logs_df,
            window_start=window_start,
            window_end=window_end,
            top_features=feat_dicts,
            k=top_k_logs,
        )
        for _, log_row in selected_logs.iterrows():
            top_logs_list.append(LogReference(
                timestamp=log_row["timestamp"].isoformat() if pd.notna(log_row["timestamp"]) else "",
                programname=str(log_row.get("programname", "")),
                log_level=log_row.get("log_level") if pd.notna(log_row.get("log_level")) else None,
                severity=int(log_row.get("severity", 0)),
                payload=str(log_row.get("payload", ""))[:500],
                selection_reason=str(log_row.get("selection_reason", "")),
                request_id=log_row.get("request_id") if pd.notna(log_row.get("request_id")) else None,
                user_id=log_row.get("user_id") if pd.notna(log_row.get("user_id")) else None,
                tenant_id=log_row.get("tenant_id") if pd.notna(log_row.get("tenant_id")) else None,
                python_module=log_row.get("python_module") if pd.notna(log_row.get("python_module")) else None,
                http_method=log_row.get("http_method") if pd.notna(log_row.get("http_method")) else None,
                http_status=int(log_row["http_status"]) if pd.notna(log_row.get("http_status")) else None,
                http_duration_ms=float(log_row["http_duration_ms"]) if pd.notna(log_row.get("http_duration_ms")) else None,
            ))

    # 3. Décision source
    decision_source = sc["decision_source"][idx] if "decision_source" in sc else "unknown"

    # 4. Severity
    severity = compute_severity(float(sc["final_score"][idx]), triggered)

    # 5. Window stats
    window_stats = {
        "count_total": int(row.get("count_total", 0)),
        "unique_services": int(row.get("unique_services", 0)),
        "unique_templates": int(row.get("unique_templates", 0)),
        "count_severity_warning": int(row.get("count_severity_warning", 0)),
        "count_severity_error": int(row.get("count_severity_error", 0)),
        "count_severity_critical": int(row.get("count_severity_critical", 0)),
        "count_status_4xx": int(row.get("count_status_4xx", 0)),
        "count_status_5xx": int(row.get("count_status_5xx", 0)),
        "latency_p95_ms": float(row.get("latency_p95", 0)),
        "latency_p99_ms": float(row.get("latency_p99", 0)),
        "latency_max_ms": float(row.get("latency_max", 0)),
        "unknown_templates_count": int(row.get("unknown_templates_count", 0)),
    }

    # 6. Alert ID (fingerprint pour déduplication)
    alert_id = AnomalyAlert.compute_alert_id(
        window_start=window_start.isoformat(),
        top_features=top_features_list,
        decision_source=decision_source,
    )

    alert = AnomalyAlert(
        schema_version="1.0",
        alert_id=alert_id,
        detected_at=datetime.now(timezone.utc).isoformat(),
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        final_score=float(sc["final_score"][idx]),
        ml_score=float(sc["ml_score"][idx]),
        ml_proba=float(sc["ml_proba"][idx]),
        rules_max_score=float(sc["rules_max"][idx]),
        severity=severity,
        decision_source=decision_source,
        top_features=top_features_list,
        triggered_rules=triggered,
        top_logs=top_logs_list,
        window_stats=window_stats,
    )

    return alert


def _features_from_triggered_rules(triggered: list[str], row: pd.Series) -> list[FeatureContribution]:
   
    rule_to_features = {
        "any_error_or_critical": ["count_severity_error", "count_severity_critical"],
        "http_5xx_present": ["count_status_5xx"],
        # "unknown_templates_present": ["unknown_templates_count"],  # DÉSACTIVÉ : templates retirés
        "service_silence": ["unique_services", "count_total"],
        "high_latency": ["latency_p95", "latency_p99", "latency_max"],
        "http_4xx_burst": ["count_status_4xx"],
    }
    out = []
    for rule_name in triggered:
        if rule_name not in rule_to_features:
            continue
        for feat_name in rule_to_features[rule_name]:
            if feat_name not in row.index:
                continue
            val = float(row[feat_name])
            if val <= 0:
                continue
            out.append(FeatureContribution(
                feature=feat_name,
                value=val,
                baseline_value=0.0,
                contribution=1.0,  # marqueur "rule-detected" (pas une vraie contrib ML)
                contribution_pct=100.0,
            ))
    return out


def detect_from_files(
    features_path: str | Path,
    enriched_logs_path: str | Path,
    models_dir: str | Path,
    output_path: str | Path = None,
    only_active: bool = True,
    ml_threshold: float = 0.90,
    rules_threshold: float = 0.5,
    deduplicate: bool = True,
) -> tuple[list[AnomalyAlert], list[dict]]:
    """Pipeline complet : charge les fichiers et lance la détection.

    Returns : (alerts, incidents)
      - alerts : liste de AnomalyAlert (toutes les fenêtres détectées)
      - incidents : liste d'incidents (alertes groupées par dédup)
    """
    features_path = Path(features_path)
    enriched_logs_path = Path(enriched_logs_path)
    models_dir = Path(models_dir)

    logger.info(f"Loading features from {features_path}")
    features_df = load_df(features_path)
    if only_active and "count_total" in features_df.columns:
        features_df = features_df[features_df["count_total"] > 0].reset_index(drop=True)

    logger.info(f"Loading enriched logs from {enriched_logs_path}")
    enriched_logs_df = load_df(enriched_logs_path)

    logger.info(f"Loading models from {models_dir}")
    ml_model = AnomalyDetector.load(models_dir / "model_global.joblib")
    hybrid = HybridDetector(ml_detector=ml_model)

    alerts = detect_anomalies(
        features_df, enriched_logs_df, hybrid,
        ml_threshold=ml_threshold, rules_threshold=rules_threshold,
    )

    # Déduplication
    incidents = []
    if deduplicate:
        from .dedup import deduplicate_alerts
        incidents = deduplicate_alerts(alerts, max_gap_minutes=5)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(output_path, "w") as f:
            json.dump(
                [a.to_dict() for a in alerts],
                f, indent=2, default=str
            )
        logger.info(f"Saved {len(alerts)} alerts to {output_path}")

        if incidents:
            incidents_path = output_path.with_name(output_path.stem + "_incidents.json")
            with open(incidents_path, "w") as f:
                json.dump(incidents, f, indent=2, default=str)
            logger.info(f"Saved {len(incidents)} incidents to {incidents_path}")

    return alerts, incidents
