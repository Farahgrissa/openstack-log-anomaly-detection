"""Génération de fenêtres temporelles glissantes.

Une fenêtre = (start, end) où end = start + window_size.
Step = décalage entre deux fenêtres consécutives.


"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Iterator


def generate_windows(
    start: datetime,
    end: datetime,
    window_seconds: int,
    step_seconds: int,
) -> list[tuple[datetime, datetime]]:
    """Génère la liste des fenêtres (start, end) entre start et end.

    """
    windows = []
    # On commence à la première borne de step alignée sur la minute pleine
    # (pour des fenêtres reproductibles)
    cur_start = start.replace(second=0, microsecond=0)
    while cur_start + timedelta(seconds=window_seconds) <= end + timedelta(seconds=step_seconds):
        cur_end = cur_start + timedelta(seconds=window_seconds)
        windows.append((cur_start, cur_end))
        cur_start += timedelta(seconds=step_seconds)
    return windows


def assign_windows_to_df(
    df: pd.DataFrame,
    window_seconds: int,
    step_seconds: int,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, list[tuple[datetime, datetime]]]:
    """Pour chaque log, calcule la liste des window_ids qui le contiennent.
    """
    if df.empty:
        return df.copy(), []

    t_min = df[timestamp_col].min()
    t_max = df[timestamp_col].max()

    # Fenêtre de référence : on aligne sur la minute pleine
    origin = t_min.replace(second=0, microsecond=0)
    windows = generate_windows(t_min, t_max, window_seconds, step_seconds)

    # Pour chaque log, calculer [w_min, w_max] (intervalle d'IDs de fenêtres qui le contiennent)
    # En secondes depuis origin
    delta_sec = (df[timestamp_col] - origin).dt.total_seconds().to_numpy()

    # w_id couvre log si window_start <= t < window_end
    # window_start = origin + w_id * step
    # window_end = window_start + window_size
    # → t - window_size < origin + w_id * step <= t
    # → (t - window_size - origin) / step < w_id <= (t - origin) / step
    w_id_max = np.floor(delta_sec / step_seconds).astype(np.int64)
    w_id_min = np.floor((delta_sec - window_seconds) / step_seconds).astype(np.int64) + 1

    # Borner aux fenêtres existantes
    n_windows = len(windows)
    w_id_max = np.minimum(w_id_max, n_windows - 1)
    w_id_min = np.maximum(w_id_min, 0)

    # Pour chaque log, on génère (log_idx, w_id) pour chaque w_id dans [w_id_min, w_id_max]
    # Approche vectorisée : on crée une longue liste flat
    n_per_log = (w_id_max - w_id_min + 1).clip(0)  # nombre de fenêtres par log
    n_per_log[w_id_max < w_id_min] = 0

    total_pairs = n_per_log.sum()

    # Indices de log dans le résultat exploded
    log_indices = np.repeat(np.arange(len(df)), n_per_log)

    # window_id correspondants
    window_ids = np.zeros(total_pairs, dtype=np.int64)
    pos = 0
    for i in range(len(df)):
        n = n_per_log[i]
        if n > 0:
            window_ids[pos:pos + n] = np.arange(w_id_min[i], w_id_max[i] + 1)
            pos += n

    # Construire le DataFrame explodé
    df_exploded = df.iloc[log_indices].copy()
    df_exploded["window_id"] = window_ids
    df_exploded.reset_index(drop=True, inplace=True)

    return df_exploded, windows


def windows_to_df(windows: list[tuple[datetime, datetime]]) -> pd.DataFrame:
    """Convertit la liste de fenêtres en DataFrame avec window_id, start, end."""
    return pd.DataFrame({
        "window_id": np.arange(len(windows)),
        "window_start": [w[0] for w in windows],
        "window_end": [w[1] for w in windows],
    })
