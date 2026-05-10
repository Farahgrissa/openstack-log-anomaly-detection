"""Couche d'accès aux données pour le dashboard.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


# Résolution des chemins (depuis la config)

@st.cache_resource
def get_paths() -> dict:
    """Résout les chemins depuis configs/config.yaml.

    Caché en `cache_resource` pour ne lire la config qu'une seule fois par
    session Streamlit.
    """
    from anomaly_detection.utils.config import get_config, resolve_path
    cfg = get_config()
    rt = cfg.get("realtime", {}) or {}
    return {
        "alerts": resolve_path(rt.get("alerts_output", "data/alerts/alerts.jsonl")),
        "enriched": resolve_path(rt.get("enriched_output", "data/alerts/alerts_enriched.jsonl")),
        "state": resolve_path(rt.get("state_file", "data/realtime_state.json")),
        "eval_results": resolve_path("data/models/eval_results.json"),
        "eval_report_png": resolve_path("data/models/eval_report.png"),
    }


# Lecture incrémentale JSONL

def _load_jsonl_incremental(path: Path, state_key: str) -> list[dict]:
    """Lit un fichier JSONL en append-only et retourne la liste cumulée.

    Mémorise dans `st.session_state[state_key]` :
      - "pos"   : dernier offset byte lu
      - "items" : liste cumulée des objets parsés

    Si le fichier a été tronqué (taille < pos), on relit tout depuis 0.
    Si le fichier n'existe pas, on retourne [].
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = {"pos": 0, "items": []}
    state = st.session_state[state_key]

    if not path.exists():
        return state["items"]

    size = path.stat().st_size
    if size < state["pos"]:
        # Fichier tronqué (rotation, suppression, etc.) → on relit tout
        state["pos"] = 0
        state["items"] = []

    if size == state["pos"]:
        return state["items"]

    new_items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        f.seek(state["pos"])
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                new_items.append(json.loads(line))
            except json.JSONDecodeError:
                # Ligne incomplète (le worker écrit peut-être en parallèle)
                # → on n'avance pas le pointeur sur cette ligne, on retentera
                break
        state["pos"] = f.tell()

    state["items"].extend(new_items)
    return state["items"]


def load_alerts() -> list[dict]:
    """Toutes les alertes brutes depuis le démarrage de la session Streamlit."""
    return _load_jsonl_incremental(get_paths()["alerts"], "_alerts_state")


def load_enriched_alerts() -> list[dict]:
    """Toutes les alertes enrichies LLM depuis le démarrage de la session."""
    return _load_jsonl_incremental(get_paths()["enriched"], "_enriched_state")


def reset_cache() -> None:
    """Invalide les caches incrémentaux (utile après un nettoyage de fichiers)."""
    for k in ("_alerts_state", "_enriched_state"):
        if k in st.session_state:
            del st.session_state[k]


# State du worker (heartbeat)

def load_worker_state() -> Optional[dict]:
    """Lit le state.json du worker (heartbeat). Retourne None si indisponible."""
    p = get_paths()["state"]
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def worker_status_summary(state: Optional[dict]) -> tuple[str, str, str]:
    """Retourne (statut, couleur, message) pour l'indicateur de tête.

    - "live"     : checkpoint < 90s          → vert
    - "stale"    : 90s ≤ checkpoint < 5min   → jaune
    - "down"     : checkpoint ≥ 5min ou pas de fichier → rouge
    """
    if state is None or not state.get("checkpoint_at"):
        return ("down", "🔴", "Worker arrêté ou jamais lancé")

    try:
        cp = datetime.fromisoformat(state["checkpoint_at"].replace("Z", "+00:00"))
    except Exception:
        return ("down", "🔴", "checkpoint_at invalide")

    age = (datetime.now(timezone.utc) - cp).total_seconds()
    if age < 90:
        return ("live", "🟢", f"Worker actif (heartbeat il y a {int(age)}s)")
    elif age < 300:
        return ("stale", "🟡", f"Worker en retard (heartbeat il y a {int(age)}s)")
    else:
        return ("down", "🔴", f"Worker injoignable (dernier heartbeat il y a {int(age // 60)}min)")


# Conversion alertes → DataFrame pandas (pratique pour les tabs)

def alerts_to_dataframe(alerts: list[dict]) -> pd.DataFrame:
    """Convertit la liste d'alertes JSON en DataFrame plat pour analyse.

    Colonnes principales : alert_id, window_start, window_end, severity,
    decision_source, final_score, ml_proba, rules_max_score,
    triggered_rules (str CSV), n_top_logs, n_top_features.
    """
    if not alerts:
        return pd.DataFrame(columns=[
            "alert_id", "window_start", "window_end", "severity",
            "decision_source", "final_score", "ml_proba", "rules_max_score",
            "triggered_rules", "n_top_logs", "n_top_features",
        ])

    rows = []
    for a in alerts:
        rows.append({
            "alert_id": a.get("alert_id", ""),
            "detected_at": pd.Timestamp(a.get("detected_at")) if a.get("detected_at") else pd.NaT,
            "window_start": pd.Timestamp(a.get("window_start")) if a.get("window_start") else pd.NaT,
            "window_end": pd.Timestamp(a.get("window_end")) if a.get("window_end") else pd.NaT,
            "severity": a.get("severity", "low"),
            "decision_source": a.get("decision_source", "unknown"),
            "final_score": float(a.get("final_score", 0.0)),
            "ml_proba": float(a.get("ml_proba", 0.0)),
            "rules_max_score": float(a.get("rules_max_score", 0.0)),
            "triggered_rules": ",".join(a.get("triggered_rules", []) or []),
            "n_top_logs": len(a.get("top_logs", []) or []),
            "n_top_features": len(a.get("top_features", []) or []),
            "count_total": int((a.get("window_stats") or {}).get("count_total", 0)),
            "count_severity_error": int((a.get("window_stats") or {}).get("count_severity_error", 0)),
            "count_severity_critical": int((a.get("window_stats") or {}).get("count_severity_critical", 0)),
            "count_status_5xx": int((a.get("window_stats") or {}).get("count_status_5xx", 0)),
            "latency_p95_ms": float((a.get("window_stats") or {}).get("latency_p95_ms", 0.0)),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("window_start").reset_index(drop=True)
    return df


# Index alert_id → enriched (pour rejoindre rapidement)

def index_enriched_by_alert_id(enriched: list[dict]) -> dict[str, dict]:
    """Map alert_id → dict(alert + llm_analysis) (dernière version gagne)."""
    out: dict[str, dict] = {}
    for entry in enriched:
        aid = entry.get("alert", {}).get("alert_id")
        if aid:
            out[aid] = entry
    return out
