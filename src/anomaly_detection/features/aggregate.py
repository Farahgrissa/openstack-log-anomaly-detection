"""Pipeline d'agrégation : DataFrame enrichi → features par fenêtre."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from ..utils.config import resolve_path, get_all_groups, get_config
from ..utils.logging import logger
from ..utils.io import load_df, save_df
from .windowing import generate_windows, windows_to_df
from .calculators import (
    compute_all_features,
    add_delta_features,
)


def aggregate_features(
    enriched_path: Path | str,
    output_path: Path | str,
    window_seconds: int = 300,
    step_seconds: int = 60,
) -> dict:
    """Pipeline complet d'agrégation."""
    enriched_path = Path(enriched_path) if not isinstance(enriched_path, Path) else enriched_path
    output_path = Path(output_path) if not isinstance(output_path, Path) else output_path

    logger.info(f"Loading enriched data from {enriched_path}")
    df = load_df(enriched_path)
    logger.info(f"Loaded {len(df):,} enriched logs")

    # Vérifier que timestamp est tz-aware
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Générer les fenêtres
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    logger.info(f"Time range: {t_min} → {t_max}")

    windows = generate_windows(t_min.to_pydatetime(), t_max.to_pydatetime(),
                                window_seconds, step_seconds)
    logger.info(f"Generated {len(windows)} windows of {window_seconds}s, step={step_seconds}s")

    # Index temporel pour recherche rapide par fenêtre
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Utiliser DatetimeIndex pour searchsorted (gère les timezones)
    ts_index = pd.DatetimeIndex(df["timestamp"])

    # Préparer la liste de tous les groupes (pour features volume par groupe)
    all_groups = get_all_groups()

    # Calculer les features pour chaque fenêtre
    logger.info("Computing features for each window...")
    rows = []

    for w_id, (w_start, w_end) in enumerate(windows):
        w_start_ts = pd.Timestamp(w_start)
        w_end_ts = pd.Timestamp(w_end)
        if w_start_ts.tz is None:
            w_start_ts = w_start_ts.tz_localize("UTC")
            w_end_ts = w_end_ts.tz_localize("UTC")
        i_start = ts_index.searchsorted(w_start_ts, side="left")
        i_end = ts_index.searchsorted(w_end_ts, side="left")
        df_window = df.iloc[i_start:i_end]

        feats = compute_all_features(
            df_window=df_window,
            window_id=w_id,
            window_start=w_start_ts,
            window_end=w_end_ts,
            all_groups=all_groups,
        )
        rows.append(feats)

        if (w_id + 1) % 500 == 0:
            logger.info(f"  Processed {w_id+1}/{len(windows)} windows")

    logger.info(f"Computed features for {len(rows)} windows")

    # DataFrame final
    feat_df = pd.DataFrame(rows)

    # Niveau 7 : features delta sur les colonnes intéressantes
    delta_cols = [
        "count_total",
        "ratio_warning",
        "ratio_error",
        "latency_p95",
        "latency_p99",
        "unique_services",
        "unique_templates",
    ]
    feat_df = add_delta_features(feat_df, delta_cols)

    # Tri final + reset index
    feat_df = feat_df.sort_values("window_id").reset_index(drop=True)

    # Stats avant sauvegarde
    n_features = len([c for c in feat_df.columns if c not in ["window_id", "window_start", "window_end"]])
    logger.info(f"Final features DataFrame: {feat_df.shape[0]} windows × {n_features} features")

    # Vérifier qu'il n'y a aucun NaN dans les features
    feat_cols = [c for c in feat_df.columns if c not in ["window_id", "window_start", "window_end"]]
    nan_counts = feat_df[feat_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        logger.warning(f"Columns with NaN values: {cols_with_nan.to_dict()}")
        # Imputer à 0 (pas de NaN dans les features)
        feat_df[feat_cols] = feat_df[feat_cols].fillna(0.0)

    # Sauvegarde
    saved = save_df(feat_df, output_path)
    logger.info(f"Saved features DataFrame: {saved}")

    return {
        "n_windows": len(feat_df),
        "n_features": n_features,
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "output_path": str(saved),
        "feature_columns": feat_cols,
    }
