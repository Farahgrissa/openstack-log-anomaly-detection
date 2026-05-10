"""Tampon glissant en mémoire pour l'agrégation temps réel.
"""
import bisect
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import pandas as pd

from ..utils.logging import logger


class WindowBuffer:
    """Tampon glissant de logs enrichis, organisé par timestamp.
    """

    def __init__(
        self,
        window_seconds: int = 300,
        step_seconds: int = 60,
        retention_minutes: int = 10,
        safety_margin_seconds: int = 60,
    ):
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.retention = timedelta(minutes=retention_minutes)
        self.safety_margin = timedelta(seconds=safety_margin_seconds)

        # Liste triée par timestamp (insertion via bisect)
        # On utilise une liste Python plutôt qu'un deque pour pouvoir bisect
        # côté gauche (purge) ET au milieu (insertion désordonnée). Les
        # opérations restent O(log n) + O(n) au pire pour les insertions
        # tardives, ce qui reste négligeable à 10 min × ~quelques milliers de logs.
        self._logs: list[dict] = []
        self._timestamps: list[datetime] = []  # parallel array pour bisect

        # Mémorise les fenêtres déjà émises pour ne pas les rejouer.
        # Clé = window_start ISO string (déterministe et compact).
        self._emitted_windows: set[str] = set()

    # API publique

    def append(self, enriched_logs: list[dict]) -> int:
        """Ajoute une liste de logs enrichis au buffer.

        Les logs peuvent arriver dans le désordre (Fluentd peut envoyer en
        rafale, OpenSearch peut renvoyer plusieurs shards) : on insère
        à la bonne position pour garder le buffer trié.

        Args :
            enriched_logs : liste de dicts (sortie de pipeline.enrich.enrich_log)
                            avec une clé `timestamp` (datetime tz-aware).

        Returns :
            Nombre de logs effectivement ajoutés (logs sans timestamp ignorés).
        """
        if not enriched_logs:
            return 0

        n_added = 0
        for log in enriched_logs:
            ts = log.get("timestamp")
            if ts is None:
                continue
            # Normaliser en datetime tz-aware UTC
            ts = _to_utc_datetime(ts)
            if ts is None:
                continue

            # Insertion triée
            pos = bisect.bisect_right(self._timestamps, ts)
            self._timestamps.insert(pos, ts)
            self._logs.insert(pos, log)
            n_added += 1

        return n_added

    def cleanup(self, now: Optional[datetime] = None) -> int:
        """Purge les logs trop anciens (au-delà de `retention_minutes`).

        Args :
            now : datetime de référence (défaut = utcnow tz-aware).

        Returns :
            Nombre de logs purgés.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        if not self._timestamps:
            return 0
        cutoff = now - self.retention
        # bisect_left pour garder les logs >= cutoff
        idx = bisect.bisect_left(self._timestamps, cutoff)
        if idx <= 0:
            return 0
        del self._timestamps[:idx]
        del self._logs[:idx]

        # Purger aussi les window_starts émis trop anciens (limite mémoire)
        # On garde une marge x2 pour être tranquille
        emitted_cutoff = (now - self.retention * 2).isoformat()
        self._emitted_windows = {w for w in self._emitted_windows if w >= emitted_cutoff}

        return idx

    def get_completed_windows(
        self,
        now: Optional[datetime] = None,
    ) -> Iterator[tuple[datetime, datetime, list[dict]]]:
        """Itère sur les fenêtres "complètes" non encore émises.

        Une fenêtre [w_start, w_end] est complète si :
          - w_end <= now - safety_margin (assez de recul pour Fluentd)
          - elle n'a pas déjà été émise
          - le buffer contient au moins une partie de [w_start, w_end]

        Args :
            now : datetime de référence (défaut = utcnow tz-aware).

        Yields :
            (window_start, window_end, logs_dans_la_fenetre)
            La liste de logs est une COPIE (le caller peut la consommer
            librement sans affecter le buffer).
        """
        if now is None:
            now = datetime.now(timezone.utc)
        if not self._logs:
            return

        # Borne supérieure : aucune fenêtre dont la fin est > (now - safety) ne peut sortir
        latest_end = now - self.safety_margin

        # Borne inférieure : on regarde uniquement les fenêtres qui contiennent
        # au moins un log encore en buffer
        oldest_log_ts = self._timestamps[0]

        # On démarre depuis la première fenêtre dont window_start est <= oldest_log_ts
        # Aligné sur la minute pleine (cohérence avec generate_windows existant)
        first_start = _floor_to_minute(oldest_log_ts)
        # Calcul vectorisé : on va boucler sur les starts par pas de step_seconds
        cur_start = first_start
        last_start = latest_end - timedelta(seconds=self.window_seconds)

        while cur_start <= last_start:
            cur_end = cur_start + timedelta(seconds=self.window_seconds)
            key = cur_start.isoformat()
            if key in self._emitted_windows:
                cur_start += timedelta(seconds=self.step_seconds)
                continue

            # Slice des logs dans [cur_start, cur_end) via bisect
            i_start = bisect.bisect_left(self._timestamps, cur_start)
            i_end = bisect.bisect_left(self._timestamps, cur_end)
            window_logs = self._logs[i_start:i_end]

            # On émet même si la fenêtre est vide : permet au worker de
            # détecter `service_silence` ou de simplement avancer son curseur
            self._emitted_windows.add(key)
            yield (cur_start, cur_end, list(window_logs))

            cur_start += timedelta(seconds=self.step_seconds)

    # Helpers d'introspection (debug / monitoring)

    def __len__(self) -> int:
        return len(self._logs)

    @property
    def n_emitted(self) -> int:
        return len(self._emitted_windows)

    def stats(self) -> dict:
        """Stats résumées (utile pour le state dump du worker)."""
        if not self._logs:
            return {
                "n_logs": 0, "n_emitted_windows": self.n_emitted,
                "oldest_ts": None, "newest_ts": None,
            }
        return {
            "n_logs": len(self._logs),
            "n_emitted_windows": self.n_emitted,
            "oldest_ts": self._timestamps[0].isoformat(),
            "newest_ts": self._timestamps[-1].isoformat(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Snapshot pandas du contenu actuel (utile pour appeler les
        calculateurs de features sans réécrire de logique d'agrégation)."""
        if not self._logs:
            return pd.DataFrame()
        df = pd.DataFrame(self._logs)
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        return df


# Helpers internes

def _to_utc_datetime(ts) -> Optional[datetime]:
    """Convertit un timestamp (datetime, pd.Timestamp, str ISO) en datetime UTC."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()
    if isinstance(ts, str):
        try:
            d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _floor_to_minute(ts: datetime) -> datetime:
    """Tronque un datetime à la minute pleine (cohérent avec generate_windows)."""
    return ts.replace(second=0, microsecond=0)
