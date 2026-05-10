"""Sélection et catégorisation des features.
"""
import pandas as pd
import numpy as np


# Métadonnées non-features
META_COLS = ["window_id", "window_start", "window_end"]


def get_feature_columns(df: pd.DataFrame, drop_constant: bool = True) -> list[str]:
    """Retourne la liste des colonnes utilisables comme features ML.
    """
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        # Doit être numérique
        if df[c].dtype.kind not in ("i", "u", "f", "b"):
            continue
        # Si demandé, exclure constantes
        if drop_constant and df[c].nunique(dropna=False) <= 1:
            continue
        cols.append(c)
    return cols


def get_features_for_group(df: pd.DataFrame, group: str,
                           drop_constant: bool = True) -> list[str]:
    """Retourne les features pertinentes pour un sous-modèle par groupe.

    Approche : on garde
      - features liées au groupe (count_group_X, ratio_group_X, latency_p95_<service du groupe>)
      - features globales pertinentes (severity_*, latency_*, http_*, sémantique, temporel)
      - features delta globales

    On n'inclut PAS les count_group d'autres groupes (sinon le modèle "compute"
    voit aussi le volume "storage", ce qu'on veut éviter).
    """
    all_feats = get_feature_columns(df, drop_constant=drop_constant)

    keep = []
    for f in all_feats:
        # Features spécifiques au groupe
        if f.startswith(f"count_group_{group}") or f.startswith(f"ratio_group_{group}"):
            keep.append(f)
            continue
        # Features d'autres groupes : EXCLURE
        if f.startswith("count_group_") or f.startswith("ratio_group_"):
            continue
        # Sinon, features globales : on garde
        keep.append(f)

    return keep


def get_constant_columns(df: pd.DataFrame) -> list[str]:
    """Liste des colonnes constantes (variance = 0)."""
    return [
        c for c in df.columns
        if c not in META_COLS
        and df[c].dtype.kind in ("i", "u", "f", "b")
        and df[c].nunique(dropna=False) <= 1
    ]


def filter_active_windows(df: pd.DataFrame, min_count: int = 1) -> pd.DataFrame:
    """Filtre les fenêtres actives (count_total >= min_count).

    Utile pour entraîner uniquement sur les périodes d'activité réelle.
    """
    return df[df["count_total"] >= min_count].copy().reset_index(drop=True)
