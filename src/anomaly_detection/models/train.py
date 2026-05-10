"""Entraînement : 1 modèle global + N modèles spécialisés par groupe métier."""
import pandas as pd
from pathlib import Path
from typing import Iterable

from ..utils.config import get_config, resolve_path, get_all_groups
from ..utils.logging import logger
from ..utils.io import load_df
from ..features.selection import (
    get_feature_columns, get_features_for_group, filter_active_windows
)
from .data_split import split_using_config
from .isolation_forest import AnomalyDetector


# Sous-groupes métiers à entraîner (pas tous - on évite "ui" et "api_gateway"
# qui sont transverses ou peu pertinents seuls)
GROUP_MODELS = ["compute", "network", "storage", "identity", "orchestration", "infra"]


def train_all(
    features_path: str | Path,
    output_dir: str | Path,
    filter_active: bool = True,
) -> dict:
    """Entraîne le modèle global + N modèles par groupe.

    Args:
        features_path : chemin vers le DataFrame de features (Phase 2)
        output_dir : dossier où sauver les modèles
        filter_active : si True, n'entraîne que sur les fenêtres actives (count_total > 0)

    Returns : dict de stats
    """
    features_path = Path(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    logger.info(f"Loading features from {features_path}")
    df = load_df(features_path)
    logger.info(f"Loaded {len(df)} windows × {df.shape[1]} columns")

    # 2. Split temporel
    train_df, val_df, holdout_df = split_using_config(df)

    # 3. Filtrer fenêtres actives (la nuit, count_total = 0, on n'entraîne pas dessus)
    if filter_active:
        n_before = len(train_df)
        train_df = filter_active_windows(train_df, min_count=1)
        n_after = len(train_df)
        logger.info(f"Filtered active training windows: {n_after}/{n_before} kept")

    if len(train_df) < 50:
        raise ValueError(f"Not enough training samples after filtering: {len(train_df)}")

    # 4. Entraîner le modèle GLOBAL
    logger.info("=" * 60)
    logger.info("Training GLOBAL model")
    logger.info("=" * 60)
    feature_cols_global = get_feature_columns(train_df, drop_constant=True)
    logger.info(f"Global features: {len(feature_cols_global)}")

    global_model = AnomalyDetector(name="global")
    global_model.fit(train_df, feature_cols_global)
    global_model.save(output_dir / "model_global.joblib")

    # 5. Entraîner les modèles par groupe
    group_models = {}
    for group in GROUP_MODELS:
        logger.info("=" * 60)
        logger.info(f"Training GROUP model: {group}")
        logger.info("=" * 60)

        feature_cols_group = get_features_for_group(train_df, group, drop_constant=True)
        logger.info(f"  Features for '{group}': {len(feature_cols_group)}")

        # Vérifier que la colonne count_group_<group> existe (sinon group n'a pas d'activité)
        if f"count_group_{group}" not in train_df.columns:
            logger.warning(f"  Skipping {group}: no count_group_{group} column")
            continue

        # Filtrer sur fenêtres avec activité dans ce groupe (sinon on entraîne sur du vide)
        sub_df = train_df[train_df[f"count_group_{group}"] > 0].reset_index(drop=True)
        logger.info(f"  Active samples in group '{group}': {len(sub_df)}")

        if len(sub_df) < 30:
            logger.warning(f"  Skipping {group}: only {len(sub_df)} samples")
            continue

        model = AnomalyDetector(name=f"group_{group}")
        model.fit(sub_df, feature_cols_group)
        model.save(output_dir / f"model_group_{group}.joblib")
        group_models[group] = model

    # 6. Validation rapide sur val_df (jour 6 = samedi)
    logger.info("=" * 60)
    logger.info("Validation on val set (sain attendu)")
    logger.info("=" * 60)

    val_active = filter_active_windows(val_df, min_count=1) if filter_active else val_df
    if len(val_active) > 0:
        val_scores = global_model.score(val_active)
        n_flagged = (val_scores > global_model.threshold).sum()
        fp_rate = n_flagged / len(val_active)
        logger.info(f"  Global model on val: {n_flagged}/{len(val_active)} "
                    f"flagged ({fp_rate:.2%} FP rate, target <1%)")

        # Per-group
        for group, model in group_models.items():
            sub_val = val_active[val_active[f"count_group_{group}"] > 0]
            if len(sub_val) > 0:
                scores = model.score(sub_val)
                fp_rate = (scores > model.threshold).sum() / len(sub_val)
                logger.info(f"  group_{group}: {fp_rate:.2%} FP rate on val")

    return {
        "n_train_samples": len(train_df),
        "n_val_samples": len(val_df),
        "n_features_global": len(feature_cols_global),
        "global_threshold": global_model.threshold,
        "n_group_models": len(group_models),
        "groups_trained": list(group_models.keys()),
    }
