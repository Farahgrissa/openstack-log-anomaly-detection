"""Split temporel des données (train/val/holdout par date).
"""
import pandas as pd
from typing import Tuple

from ..utils.config import get_config
from ..utils.logging import logger


def split_by_date(
    df: pd.DataFrame,
    train_dates: list[str],
    val_dates: list[str],
    holdout_dates: list[str] = None,
    timestamp_col: str = "window_start",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split un DataFrame de features par date.

    Args:
        df : DataFrame de features avec colonne window_start
        train_dates : liste des dates "YYYY-MM-DD" pour le training
        val_dates : liste des dates pour la validation
        holdout_dates : optionnel, dates pour le holdout

    Returns:
        (train_df, val_df, holdout_df)
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df[timestamp_col]).dt.strftime("%Y-%m-%d")

    train_df = df[df["_date"].isin(train_dates)].drop(columns=["_date"]).reset_index(drop=True)
    val_df = df[df["_date"].isin(val_dates)].drop(columns=["_date"]).reset_index(drop=True)

    if holdout_dates:
        holdout_df = df[df["_date"].isin(holdout_dates)].drop(columns=["_date"]).reset_index(drop=True)
    else:
        holdout_df = pd.DataFrame()

    logger.info(f"Split: train={len(train_df)} | val={len(val_df)} | holdout={len(holdout_df)}")
    return train_df, val_df, holdout_df


def split_using_config(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split selon les dates définies dans config.yaml."""
    config = get_config()
    return split_by_date(
        df,
        train_dates=config["data_split"]["train_dates"],
        val_dates=config["data_split"]["val_dates"],
        holdout_dates=config["data_split"].get("holdout_dates"),
    )


def filter_active(df: pd.DataFrame, min_count: int = 1) -> pd.DataFrame:
    """Garde uniquement les fenêtres avec count_total >= min_count."""
    if "count_total" not in df.columns:
        return df
    return df[df["count_total"] >= min_count].reset_index(drop=True)
