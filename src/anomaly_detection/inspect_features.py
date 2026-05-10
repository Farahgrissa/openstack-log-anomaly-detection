"""Inspection du DataFrame de features."""
import pandas as pd
import numpy as np

from .utils.config import resolve_path
from .utils.logging import logger
from .utils.io import load_df
from .features.selection import (
    get_feature_columns, get_constant_columns, filter_active_windows
)


def inspect_features(input_file: str = "data/features/windows_5min") -> dict:
    """Affiche un rapport détaillé du DataFrame de features."""
    path = resolve_path(input_file)
    logger.info(f"Loading features from {path}")
    df = load_df(path)

    logger.info(f"Shape: {df.shape}")

    # Time range
    logger.info(f"Time range: {df['window_start'].min()} → {df['window_end'].max()}")

    # Total NaN
    n_nan = df.isna().sum().sum()
    logger.info(f"Total NaN: {n_nan}")

    # Features split
    all_feats = get_feature_columns(df, drop_constant=False)
    useful_feats = get_feature_columns(df, drop_constant=True)
    constant_feats = get_constant_columns(df)
    logger.info(f"Numeric features: {len(all_feats)} (useful: {len(useful_feats)}, constant: {len(constant_feats)})")

    if constant_feats:
        logger.info(f"Constant features: {constant_feats}")

    # Fenêtres actives vs vides
    active = filter_active_windows(df, min_count=1)
    logger.info(f"Active windows (count_total > 0): {len(active)} / {len(df)} ({100*len(active)/len(df):.1f}%)")

    # Distribution count_total
    if len(active) > 0:
        logger.info("=== count_total stats (active windows) ===")
        stats = active["count_total"].describe([0.5, 0.95, 0.99])
        logger.info(f"  min/p50/p95/p99/max: {stats['min']:.0f}/{stats['50%']:.0f}/{stats['95%']:.0f}/{stats['99%']:.0f}/{stats['max']:.0f}")

        logger.info("=== latency_p95 stats (active windows) ===")
        stats = active["latency_p95"].describe([0.5, 0.95, 0.99])
        logger.info(f"  min/p50/p95/p99/max: {stats['min']:.0f}/{stats['50%']:.0f}/{stats['95%']:.0f}/{stats['99%']:.0f}/{stats['max']:.0f}")

        logger.info("=== ratio_warning stats (active windows) ===")
        stats = active["ratio_warning"].describe([0.5, 0.95, 0.99])
        logger.info(f"  min/p50/p95/p99/max: {stats['min']:.4f}/{stats['50%']:.4f}/{stats['95%']:.4f}/{stats['99%']:.4f}/{stats['max']:.4f}")

    return {
        "shape": df.shape,
        "n_features_useful": len(useful_feats),
        "n_active_windows": len(active),
    }
