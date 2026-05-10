"""Worker temps réel : daemon principal du pipeline de détection en streaming."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

from ..utils.config import get_config, resolve_path, get_all_groups
from ..utils.logging import logger
from ..pipeline.enrich import enrich_log, load_template_dict, stream_ndjson
from ..features.calculators import compute_all_features
from ..models.isolation_forest import AnomalyDetector
from ..models.hybrid import HybridDetector
from ..alerting.detect import detect_anomalies
from ..alerting.schema import AnomalyAlert

from .opensearch_client import OpenSearchClient, OpenSearchError
from .window_buffer import WindowBuffer


# Colonnes attendues par les calculateurs de features (pour DataFrame vide)
ENRICHED_LOG_COLUMNS = [
    "timestamp", "programname", "log_level", "severity", "log_kind", "groups",
    "http_method", "http_status", "http_status_class", "http_duration_ms",
    "http_bytes", "endpoint_template",
    "payload_template_id", "payload_template",
    "request_id", "user_id", "tenant_id",
    "python_module", "payload",
]


def _empty_enriched_df() -> pd.DataFrame:
    """Retourne un DataFrame vide avec les colonnes attendues par les calculateurs de features."""
    return pd.DataFrame({col: pd.Series(dtype="object") for col in ENRICHED_LOG_COLUMNS})



class RealtimeWorker:
    """Daemon de détection d'anomalies en quasi-temps réel.

    Instanciation : `RealtimeWorker.from_config()`.
    Exécution : `worker.run()` (boucle infinie, Ctrl+C pour stop propre).
    """

    def __init__(
        self,
        # OpenSearch
        client: Optional[OpenSearchClient],
        # Modèles + templates
        detector: HybridDetector,
        template_dict: dict[str, int],
        # I/O
        alerts_path: Path,
        enriched_alerts_path: Path,
        state_path: Path,
        # Config running
        poll_interval_seconds: int = 30,
        lookback_window_minutes: int = 5,
        window_seconds: int = 300,
        step_seconds: int = 60,
        max_logs_per_iter: int = 10_000,
        seen_ids_max: int = 100_000,
        llm_min_severity: str = "high",
        llm_enabled: bool = True,
        ml_threshold: float = 0.90,
        rules_threshold: float = 0.5,
        # Mode replay
        replay_path: Optional[Path] = None,
        replay_speed: float = 1.0,
    ):
        self.client = client
        self.detector = detector
        self.template_dict = template_dict
        self.alerts_path = Path(alerts_path)
        self.enriched_alerts_path = Path(enriched_alerts_path)
        self.state_path = Path(state_path)

        self.poll_interval = poll_interval_seconds
        self.lookback = timedelta(minutes=lookback_window_minutes)
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.max_logs_per_iter = max_logs_per_iter
        self.seen_ids_max = seen_ids_max

        self.llm_min_severity = llm_min_severity
        self.llm_enabled = llm_enabled
        self.ml_threshold = ml_threshold
        self.rules_threshold = rules_threshold

        # Mode replay
        self.replay_path = Path(replay_path) if replay_path else None
        self.replay_speed = float(replay_speed)
        self._replay_iter: Optional[Iterator[dict]] = None
        self._replay_cur_ts: Optional[datetime] = None  # curseur dans le NDJSON
        self._replay_started_at: Optional[datetime] = None  # début wall-clock

        # Buffer glissant
        self.buffer = WindowBuffer(
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            retention_minutes=10,
            safety_margin_seconds=60,
        )

        # État runtime
        self.last_timestamp_seen: Optional[datetime] = None
        self.seen_ids: set[str] = set()
        self.windows_emitted_count = 0
        self.alerts_generated_count = 0
        self.iter_count = 0
        self.started_at: Optional[datetime] = None

        # LLM async worker (lazy-loaded, optionnel)
        self._llm_worker: Optional[Any] = None

        # Pour arrêt propre
        self._stop_requested = False

        # Préparation des outputs (mkdir + ouverture en append)
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        self.enriched_alerts_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)


    @classmethod
    def from_config(
        cls,
        replay: Optional[str] = None,
        speed: float = 1.0,
    ) -> "RealtimeWorker":
        """Construit un worker depuis configs/config.yaml + .env."""
        cfg = get_config()
        rt_cfg = cfg.get("realtime", {}) or {}

        if not rt_cfg.get("enabled", True):
            logger.warning("realtime.enabled=false dans config — le worker s'exécute quand même mais c'est inhabituel")

        # 1. Modèles
        models_dir = resolve_path("data/models")
        ml_model_path = models_dir / "model_global.joblib"
        if not ml_model_path.exists():
            raise FileNotFoundError(
                f"Modèle global introuvable : {ml_model_path}. "
                f"Lance `python -m anomaly_detection.cli train` d'abord."
            )
        logger.info(f"Chargement du modèle ML : {ml_model_path}")
        ml = AnomalyDetector.load(ml_model_path)
        detector = HybridDetector(ml_detector=ml)

        # 2. Templates frozen (Phase 1)
        templates_path = resolve_path("data/processed/templates.tsv")
        if not templates_path.exists():
            raise FileNotFoundError(
                f"Templates introuvables : {templates_path}. "
                f"Lance `python -m anomaly_detection.cli enrich` d'abord."
            )
        template_dict = load_template_dict(templates_path)
        logger.info(f"Chargement de {len(template_dict)} templates frozen depuis {templates_path}")

        # 3. Client OpenSearch (sauf en replay)
        client: Optional[OpenSearchClient] = None
        if replay is None:
            os_cfg = rt_cfg.get("opensearch", {}) or {}
            host = os_cfg.get("host", "localhost")
            port = int(os_cfg.get("port", 9200))
            username = os.environ.get("OPENSEARCH_USER") or os_cfg.get("username")
            password = os.environ.get("OPENSEARCH_PASSWORD") or os_cfg.get("password")
            client = OpenSearchClient(
                host=host,
                port=port,
                scheme=os_cfg.get("scheme", "http"),
                username=username,
                password=password,
                index_pattern=os_cfg.get("index_pattern", "flog-*"),
                verify_certs=bool(os_cfg.get("verify_certs", False)),
                timeout_seconds=int(os_cfg.get("timeout_seconds", 30)),
            )
            if not client.ping():
                logger.warning(
                    "OpenSearch ne répond pas au ping initial. "
                    "Le worker démarre quand même (avec retry exponentiel à chaque tour)."
                )

        # 4. Chemins de sortie
        alerts_path = resolve_path(rt_cfg.get("alerts_output", "data/alerts/alerts.jsonl"))
        enriched_path = resolve_path(rt_cfg.get("enriched_output", "data/alerts/alerts_enriched.jsonl"))
        state_path = resolve_path(rt_cfg.get("state_file", "data/realtime_state.json"))

        # 5. Paramètres windowing (alignés sur features.windowing.primary)
        win_cfg = cfg.get("windowing", {}).get("primary", {})
        window_seconds = int(win_cfg.get("window_size_seconds", 300))
        step_seconds = int(win_cfg.get("step_seconds", 60))

        # 6. LLM
        llm_enabled = bool(cfg.get("llm", {}).get("enabled", True))

        return cls(
            client=client,
            detector=detector,
            template_dict=template_dict,
            alerts_path=alerts_path,
            enriched_alerts_path=enriched_path,
            state_path=state_path,
            poll_interval_seconds=int(rt_cfg.get("poll_interval_seconds", 30)),
            lookback_window_minutes=int(rt_cfg.get("lookback_window_minutes", 5)),
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            max_logs_per_iter=int(rt_cfg.get("max_logs_per_iter", 10_000)),
            llm_min_severity=str(rt_cfg.get("llm_min_severity", "high")),
            llm_enabled=llm_enabled,
            replay_path=Path(replay) if replay else None,
            replay_speed=speed,
        )


    def run(self) -> None:
        """Lance la boucle infinie. Ctrl+C pour stopper proprement."""
        self.started_at = datetime.now(timezone.utc)
        self._setup_signal_handlers()
        self._restore_state()

        mode = "replay" if self.replay_path else "live (OpenSearch)"
        logger.info(f"=== Worker temps réel démarré (mode {mode}) ===")
        logger.info(f"  poll_interval = {self.poll_interval}s")
        logger.info(f"  lookback      = {self.lookback}")
        logger.info(f"  window        = {self.window_seconds}s, step = {self.step_seconds}s")
        logger.info(f"  alerts_path   = {self.alerts_path}")
        logger.info(f"  state_path    = {self.state_path}")
        if self.replay_path:
            logger.info(f"  replay_path   = {self.replay_path} (speed={self.replay_speed}x)")

        # Mode replay : pré-charger l'itérateur sur le NDJSON
        if self.replay_path:
            self._setup_replay()

        try:
            while not self._stop_requested:
                self.iter_count += 1
                t_iter_start = time.time()
                try:
                    self._run_one_iteration()
                except Exception as e:
                    logger.exception(f"Itération {self.iter_count} a échoué : {e}")

                # Persistance d'état (même en cas d'erreur en cours de cycle)
                self._save_state()

                # Sleep jusqu'au prochain tour
                elapsed = time.time() - t_iter_start
                sleep_s = max(0.0, self.poll_interval - elapsed)
                logger.debug(f"Itération {self.iter_count} terminée en {elapsed:.1f}s, sleep {sleep_s:.1f}s")
                self._sleep_interruptible(sleep_s)
        finally:
            # Arrêt propre du thread LLM async (s'il a été instancié)
            if self._llm_worker is not None:
                logger.info("Arrêt du thread LLM async (vidange de la queue)...")
                self._llm_worker.stop(timeout=30.0)
            self._save_state()
            logger.info(
                f"=== Worker arrêté ({self.windows_emitted_count} fenêtres, "
                f"{self.alerts_generated_count} alertes, {self.iter_count} itérations) ==="
            )


    def _run_one_iteration(self) -> None:
        """Une seule itération de la boucle (poll + détection)."""
        # 1. Récupérer les logs bruts depuis la source (OpenSearch ou replay)
        raw_logs = list(self._fetch_raw_logs())
        n_raw = len(raw_logs)
        logger.info(f"[iter {self.iter_count}] {n_raw} logs récupérés")

        if n_raw == 0:
            self.buffer.cleanup(now=self._now())
            return

        # 2. Dédup par `_os_id` (en mode live) — en replay les logs n'ont pas d'_os_id
        deduped = []
        for log in raw_logs:
            os_id = log.get("_os_id")
            if os_id:
                if os_id in self.seen_ids:
                    continue
                self.seen_ids.add(os_id)
            deduped.append(log)
        n_dedup = n_raw - len(deduped)
        if n_dedup > 0:
            logger.info(f"[iter {self.iter_count}] {n_dedup} doublons _os_id ignorés")

        # 3. Enrichir avec le pipeline Phase 1 (templates FROZEN)
        enriched: list[dict] = []
        n_enrich_failed = 0
        for log in deduped:
            try:
                e = enrich_log(log, self.template_dict, freeze_templates=True)
                if e.get("timestamp") is None:
                    n_enrich_failed += 1
                    continue
                enriched.append(e)
            except Exception:
                n_enrich_failed += 1
                continue
        if n_enrich_failed > 0:
            logger.warning(f"[iter {self.iter_count}] {n_enrich_failed} logs n'ont pas pu être enrichis")
        logger.info(f"[iter {self.iter_count}] {len(enriched)} logs enrichis")

        # 4. Mettre à jour le curseur de timestamp (max vu)
        for log in enriched:
            ts = log["timestamp"]
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if self.last_timestamp_seen is None or ts > self.last_timestamp_seen:
                self.last_timestamp_seen = ts

        # 5. Append au buffer
        self.buffer.append(enriched)

        # 6. Récupérer les fenêtres complètes
        now = self._now()
        completed = list(self.buffer.get_completed_windows(now=now))
        logger.info(f"[iter {self.iter_count}] {len(completed)} fenêtres complètes prêtes à scorer")

        if completed:
            self._score_and_alert(completed)

        # 7. Cleanup
        self.buffer.cleanup(now=now)
        self._cleanup_seen_ids()


    def _fetch_raw_logs(self) -> Iterator[dict]:
        """Yield des logs bruts. Mode live : poll OpenSearch. Mode replay : lit le NDJSON."""
        if self.replay_path:
            yield from self._fetch_from_replay()
            return

        if self.client is None:
            return

        now = datetime.now(timezone.utc)
        if self.last_timestamp_seen is None:
            # Premier tour : on prend une fenêtre courte (lookback)
            start = now - self.lookback
        else:
            # Tours suivants : on relit légèrement en arrière pour
            # absorber la latence Fluentd (logs qui apparaissent après coup)
            start = self.last_timestamp_seen - timedelta(minutes=1)
        end = now
        try:
            logger.debug(f"OpenSearch fetch [{start.isoformat()} → {end.isoformat()}]")
            n = 0
            for log in self.client.fetch_logs_between(
                start, end, max_logs=self.max_logs_per_iter
            ):
                n += 1
                yield log
        except OpenSearchError as e:
            logger.error(f"Erreur OpenSearch (cycle ignoré, retry au prochain tour) : {e}")


    def _setup_replay(self) -> None:
        """Initialise l'itérateur sur le NDJSON de replay et fixe la borne de départ."""
        if not self.replay_path or not self.replay_path.exists():
            raise FileNotFoundError(f"Fichier de replay introuvable : {self.replay_path}")
        logger.info(f"Préparation du replay depuis {self.replay_path}")

        # On charge un itérateur paresseux (le fichier peut faire 100k+ logs)
        self._replay_iter = stream_ndjson(self.replay_path)
        # Curseur initial : on lit le premier log pour connaître la borne
        try:
            first_log = next(self._replay_iter)
        except StopIteration:
            raise ValueError(f"NDJSON de replay vide : {self.replay_path}")
        first_ts = _parse_fluentd_ts(first_log.get("@timestamp", ""))
        if first_ts is None:
            raise ValueError("Impossible de parser le @timestamp du premier log de replay")
        self._replay_cur_ts = first_ts
        self._replay_started_at = datetime.now(timezone.utc)
        # On bufferise ce premier log pour ne pas le perdre
        self._replay_pending: list[dict] = [first_log]

    def _fetch_from_replay(self) -> Iterator[dict]:
        """Itère sur les logs du NDJSON jusqu'au timestamp simulé courant."""
        if self._replay_iter is None:
            return

        # Calcul du timestamp simulé courant
        # = first_ts + (wall_time_écoulé × speed)
        now_wall = datetime.now(timezone.utc)
        elapsed_wall = (now_wall - self._replay_started_at).total_seconds()
        sim_now = self._replay_cur_ts + timedelta(seconds=elapsed_wall * self.replay_speed)

        n = 0
        # 1. Vidange du buffer pending
        remaining_pending: list[dict] = []
        for log in self._replay_pending:
            ts = _parse_fluentd_ts(log.get("@timestamp", ""))
            if ts is None:
                continue
            if ts <= sim_now:
                yield log
                n += 1
                if n >= self.max_logs_per_iter:
                    self._replay_pending = remaining_pending
                    return
            else:
                remaining_pending.append(log)
        self._replay_pending = remaining_pending

        # 2. Lecture du NDJSON tant qu'on est avant sim_now
        while True:
            try:
                log = next(self._replay_iter)
            except StopIteration:
                logger.info("Fin du fichier de replay atteinte — arrêt du worker dans 2 cycles")
                self._stop_requested = True  # arrêt après 2 cycles pour vider le buffer
                return
            ts = _parse_fluentd_ts(log.get("@timestamp", ""))
            if ts is None:
                continue
            if ts <= sim_now:
                yield log
                n += 1
                if n >= self.max_logs_per_iter:
                    return
            else:
                # On stocke ce log pour le prochain tour
                self._replay_pending.append(log)
                break


    def _score_and_alert(self, completed: list[tuple[datetime, datetime, list[dict]]]) -> None:
        """Pour chaque fenêtre complète, calcule features → score → AnomalyAlert."""
        all_groups = get_all_groups()

        feature_rows: list[dict] = []
        per_window_logs: list[pd.DataFrame] = []
        for w_start, w_end, logs in completed:
            self.windows_emitted_count += 1
            if logs:
                df_window = pd.DataFrame(logs)
                if "timestamp" in df_window.columns:
                    df_window["timestamp"] = pd.to_datetime(df_window["timestamp"], utc=True, errors="coerce")
                for col in ENRICHED_LOG_COLUMNS:
                    if col not in df_window.columns:
                        df_window[col] = None
            else:
                df_window = _empty_enriched_df()
            feats = compute_all_features(
                df_window=df_window,
                window_id=self.windows_emitted_count,
                window_start=pd.Timestamp(w_start),
                window_end=pd.Timestamp(w_end),
                all_groups=all_groups,
            )
            feature_rows.append(feats)
            per_window_logs.append(df_window)

        if not feature_rows:
            return

        feat_df = pd.DataFrame(feature_rows)
        # Delta features : fixées à 0 en streaming (pas d'historique inter-sessions)
        delta_cols = [
            "delta_count_total", "delta_ratio_warning", "delta_ratio_error",
            "delta_latency_p95", "delta_latency_p99",
            "delta_unique_services", "delta_unique_templates",
        ]
        for col in delta_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df.fillna(0.0)

        # 2. Concaténer tous les logs des fenêtres pour le top-K logs
        non_empty = [df for df in per_window_logs if not df.empty]
        if non_empty:
            all_logs_df = pd.concat(non_empty, ignore_index=True)
        else:
            all_logs_df = pd.DataFrame()

        # 3. Détection batch
        alerts = detect_anomalies(
            features_df=feat_df,
            enriched_logs_df=all_logs_df,
            detector=self.detector,
            ml_threshold=self.ml_threshold,
            rules_threshold=self.rules_threshold,
        )

        if not alerts:
            logger.info(f"[iter {self.iter_count}] Aucune anomalie détectée sur ce batch")
            return

        # 4. Append en JSONL
        for alert in alerts:
            self.alerts_generated_count += 1
            self._append_alert(alert)

            # 5. Hook LLM pour les alertes sévères
            if self.llm_enabled and self._severity_meets(alert.severity, self.llm_min_severity):
                try:
                    self._enrich_with_llm(alert)
                except Exception as e:
                    logger.warning(f"Enrichissement LLM échoué pour {alert.alert_id} : {e}")

        logger.info(
            f"[iter {self.iter_count}] {len(alerts)} alertes générées "
            f"(total cumulé : {self.alerts_generated_count})"
        )


    def _append_alert(self, alert: AnomalyAlert) -> None:
        """Append une alerte au fichier JSONL (append-only, safe pour lecture concurrente)."""
        line = json.dumps(_clean_for_json(alert.to_dict()), ensure_ascii=False, default=str)
        with open(self.alerts_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _append_enriched_alert(self, payload: dict) -> None:
        """Append une alerte enrichie LLM."""
        line = json.dumps(_clean_for_json(payload), ensure_ascii=False, default=str)
        with open(self.enriched_alerts_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


    def _enrich_with_llm(self, alert: AnomalyAlert) -> None:
        """Soumet une alerte sévère au thread LLM async (non-bloquant)."""
        worker = self._ensure_llm_worker()
        if worker is None:
            return
        worker.submit(alert.to_dict())

    def _ensure_llm_worker(self):
        """Lazy-init du thread LLM async (créé au premier usage seulement)."""
        if self._llm_worker is not None:
            return self._llm_worker
        try:
            from ..llm import LLMAnalyzer
            from .llm_worker import LLMWorker
            analyzer = LLMAnalyzer.from_config()
            self._llm_worker = LLMWorker(
                analyzer=analyzer,
                output_path=self.enriched_alerts_path,
            )
            self._llm_worker.start()
        except Exception as e:
            logger.error(f"Initialisation LLM async échouée : {e}")
            self.llm_enabled = False
            return None
        return self._llm_worker

    @staticmethod
    def _severity_meets(actual: str, minimum: str) -> bool:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return order.get(actual, 0) >= order.get(minimum, 2)


    def _save_state(self) -> None:
        """Écrit l'état courant dans state_path (heartbeat)."""
        state = {
            "schema_version": "1.0",
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "checkpoint_at": datetime.now(timezone.utc).isoformat(),
            "iter_count": self.iter_count,
            "windows_emitted_count": self.windows_emitted_count,
            "alerts_generated_count": self.alerts_generated_count,
            "last_timestamp_seen": (
                self.last_timestamp_seen.isoformat() if self.last_timestamp_seen else None
            ),
            "buffer_stats": self.buffer.stats(),
            "n_seen_ids": len(self.seen_ids),
            "mode": "replay" if self.replay_path else "live",
            "llm_worker_stats": (
                self._llm_worker.stats() if self._llm_worker is not None else None
            ),
        }
        # Écriture atomique : write to .tmp then rename (Windows-safe)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)

    def _restore_state(self) -> None:
        """Tente de restaurer l'état depuis state_path (reprise après crash)."""
        if not self.state_path.exists():
            logger.info(f"Pas d'état précédent à restaurer ({self.state_path})")
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            logger.warning(f"Impossible de lire {self.state_path} : {e}")
            return

        last_ts = state.get("last_timestamp_seen")
        if last_ts:
            try:
                self.last_timestamp_seen = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                logger.info(f"État restauré : last_timestamp_seen = {self.last_timestamp_seen}")
            except Exception:
                logger.warning(f"last_timestamp_seen non parsable : {last_ts}")
        # On ne restaure PAS les compteurs (on repart de zéro pour cette session)
        # mais on log le total de la session précédente pour info
        prev_alerts = state.get("alerts_generated_count", 0)
        prev_iters = state.get("iter_count", 0)
        if prev_alerts or prev_iters:
            logger.info(
                f"Session précédente : {prev_iters} itérations, "
                f"{prev_alerts} alertes générées"
            )


    def _now(self) -> datetime:
        """Datetime courant (UTC). En mode replay, c'est le temps simulé."""
        if self.replay_path and self._replay_started_at and self._replay_cur_ts:
            elapsed_wall = (datetime.now(timezone.utc) - self._replay_started_at).total_seconds()
            return self._replay_cur_ts + timedelta(seconds=elapsed_wall * self.replay_speed)
        return datetime.now(timezone.utc)

    def _cleanup_seen_ids(self) -> None:
        """Limite la taille du set seen_ids (anti-fuite mémoire)."""
        if len(self.seen_ids) > self.seen_ids_max:
            # On vide la moitié au hasard (le risque de re-traiter un log
            # rare est négligeable face au bénéfice mémoire)
            to_drop = len(self.seen_ids) - self.seen_ids_max // 2
            for _ in range(to_drop):
                self.seen_ids.pop()
            logger.info(f"Cleanup seen_ids : drop {to_drop}, kept {len(self.seen_ids)}")

    def _setup_signal_handlers(self) -> None:
        """Capture SIGINT/SIGTERM pour arrêt propre."""
        def _handler(signum, frame):
            logger.info(f"Signal {signum} reçu — arrêt en cours...")
            self._stop_requested = True
        try:
            signal.signal(signal.SIGINT, _handler)
            # SIGTERM peut ne pas exister sur Windows
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):
            # Si on n'est pas dans le main thread, on ignore
            pass

    def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep découpé en tranches de 0.5s pour répondre vite à un Ctrl+C."""
        end = time.time() + seconds
        while time.time() < end and not self._stop_requested:
            time.sleep(min(0.5, end - time.time()))



def _clean_for_json(obj: Any) -> Any:
    """Nettoie un objet pour la sérialisation JSON (numpy/pandas/datetime)."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _parse_fluentd_ts(ts: str) -> Optional[datetime]:
    """Parse un @timestamp Fluentd (ISO8601 avec ns) → datetime UTC.

    Délégation à pipeline.enrich.parse_timestamp pour cohérence stricte.
    """
    from ..pipeline.enrich import parse_timestamp
    d = parse_timestamp(ts)
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def run_worker(replay: Optional[str] = None, speed: float = 1.0) -> None:
    """Point d'entrée appelé par la commande CLI `live`."""
    worker = RealtimeWorker.from_config(replay=replay, speed=speed)
    worker.run()
