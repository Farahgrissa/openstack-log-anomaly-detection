"""LLM async — décorrèle la latence Gemini/Groq de la boucle de détection."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Optional

from ..utils.logging import logger


# Sentinelle envoyée pour signaler la fin du flux à consumer
_STOP_SENTINEL = object()


class LLMWorker:
    """Thread asynchrone qui consomme une queue d'alertes et écrit l'enrichissement.
    """

    def __init__(
        self,
        analyzer,                           # LLMAnalyzer (typage évité pour ne pas
                                            # forcer l'import des SDK LLM côté worker)
        output_path: Path,
        max_queue_size: int = 200,
        name: str = "llm-worker",
    ):
        self.analyzer = analyzer
        self.output_path = Path(output_path)
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._stopped = threading.Event()

        # Stats (lues par le worker principal pour le state heartbeat)
        self.n_submitted = 0
        self.n_processed = 0
        self.n_dropped = 0
        self.n_errors = 0

        # Préparer le fichier de sortie
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # API publique

    def start(self) -> None:
        """Démarre le thread consommateur."""
        if not self._thread.is_alive():
            self._thread.start()
            logger.info(f"LLM async worker started (queue maxsize={self._queue.maxsize})")

    def submit(self, alert: dict) -> bool:
        """Soumet une alerte à analyser.

        Non-bloquant. Si la queue est pleine, on drop et on log un warning :
        le LLM est un bonus, pas un bloqueur. La détection brute reste
        disponible dans `alerts.jsonl`.

        Returns :
            True si l'alerte a été ajoutée, False si droppée.
        """
        if self._stopped.is_set():
            return False
        try:
            self._queue.put_nowait(alert)
            self.n_submitted += 1
            return True
        except Full:
            self.n_dropped += 1
            logger.warning(
                f"LLM queue saturée ({self._queue.qsize()} alertes en attente) — "
                f"alerte droppée (alert_id={alert.get('alert_id', '?')})"
            )
            return False

    def stop(self, timeout: float = 30.0) -> None:
        """Demande l'arrêt et attend la fin du traitement en cours."""
        if self._stopped.is_set():
            return
        self._stopped.set()
        # Pousser la sentinelle pour réveiller le thread s'il est en attente
        try:
            self._queue.put_nowait(_STOP_SENTINEL)
        except Full:
            # Queue pleine : on attendra simplement le timeout
            pass
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                f"LLM worker n'a pas terminé en {timeout}s "
                f"(en queue : {self._queue.qsize()})"
            )
        else:
            logger.info(
                f"LLM worker arrêté : {self.n_processed} alertes analysées, "
                f"{self.n_dropped} droppées, {self.n_errors} erreurs"
            )

    def stats(self) -> dict:
        """Stats pour le state heartbeat."""
        return {
            "queue_size": self._queue.qsize(),
            "n_submitted": self.n_submitted,
            "n_processed": self.n_processed,
            "n_dropped": self.n_dropped,
            "n_errors": self.n_errors,
        }

    # Boucle interne du thread

    def _run(self) -> None:
        """Boucle de consommation : prend une alerte, l'analyse, écrit le résultat."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                # Pas d'alerte à traiter : check si on doit s'arrêter
                if self._stopped.is_set() and self._queue.empty():
                    return
                continue

            if item is _STOP_SENTINEL:
                # Si on a vidé la queue, on sort
                if self._queue.empty():
                    return
                continue

            alert = item
            try:
                analysis = self.analyzer.analyze(alert)
                self._append_enriched(alert, analysis)
                self.n_processed += 1
            except Exception as e:
                self.n_errors += 1
                logger.error(
                    f"LLM worker : analyse échouée pour "
                    f"alert_id={alert.get('alert_id', '?')} : {e}"
                )
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

    def _append_enriched(self, alert: dict, analysis) -> None:
        """Append une ligne JSONL `{alert, llm_analysis, enriched_at}`.

        Format identique à celui produit par la commande `analyze` (Phase 6),
        plus un champ `enriched_at` indiquant l'heure réelle de l'analyse
        (différente de `detected_at` à cause de l'asynchronisme).
        """
        try:
            ana_dict = analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis)
        except Exception:
            ana_dict = {"_raw": str(analysis)}

        payload = {
            "alert": alert,
            "llm_analysis": ana_dict,
            "enriched_at": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
