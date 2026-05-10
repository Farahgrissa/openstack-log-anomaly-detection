"""Calcul des features par fenêtre temporelle.
"""
import numpy as np
import pandas as pd
from typing import Iterable

from ..utils.config import get_all_groups


CRITICAL_ENDPOINTS = [
    ("POST", "/v3/auth/tokens"),
    ("POST", "/v2.1/servers"),
    ("DELETE", "/v2.1/servers/<UUID>"),
    ("POST", "/v3/volumes"),
    ("DELETE", "/v3/volumes/<UUID>"),
    ("POST", "/v2.0/networks"),
    ("POST", "/v2.0/subnets"),
    ("POST", "/auth/login/"),
]

CRITICAL_LATENCY_SERVICES = [
    "keystone-uwsgi",
    "nova-api-uwsgi",
    "neutron-server-uwsgi",
    "cinder-api-uwsgi",
    "horizon-uwsgi",
]


# --- Niveau 1 : Volume ---

def compute_volume_features(df_window: pd.DataFrame, all_groups: list[str]) -> dict:
    """Features de volume : total, par groupe, ratios."""
    feats = {}
    n_total = len(df_window)
    feats["count_total"] = n_total

    if n_total == 0:
        feats["unique_services"] = 0
        feats["unique_python_modules"] = 0
        feats["unique_request_ids"] = 0
        feats["unique_users"] = 0
        feats["unique_tenants"] = 0
        for group in all_groups:
            feats[f"count_group_{group}"] = 0
            feats[f"ratio_group_{group}"] = 0.0
        for kind in ["http_access", "app_log", "infra_log", "agent_log"]:
            feats[f"ratio_kind_{kind}"] = 0.0
        return feats

    feats["unique_services"] = df_window["programname"].nunique()
    feats["unique_python_modules"] = df_window["python_module"].nunique() if "python_module" in df_window.columns else 0
    feats["unique_request_ids"] = df_window["request_id"].dropna().nunique()
    feats["unique_users"] = df_window["user_id"].dropna().nunique() if "user_id" in df_window.columns else 0
    feats["unique_tenants"] = df_window["tenant_id"].dropna().nunique() if "tenant_id" in df_window.columns else 0

    for group in all_groups:
        count = df_window["groups"].apply(lambda gs: group in gs if isinstance(gs, list) else False).sum()
        feats[f"count_group_{group}"] = int(count)
        feats[f"ratio_group_{group}"] = count / n_total

    kind_counts = df_window["log_kind"].value_counts()
    for kind in ["http_access", "app_log", "infra_log", "agent_log"]:
        feats[f"ratio_kind_{kind}"] = kind_counts.get(kind, 0) / n_total

    return feats


# --- Niveau 2 : Log levels ---

def compute_level_features(df_window: pd.DataFrame) -> dict:
    """Features sur log_level / severity."""
    feats = {}
    n_total = len(df_window)

    if n_total == 0:
        feats["count_severity_info"] = 0
        feats["count_severity_warning"] = 0
        feats["count_severity_error"] = 0
        feats["count_severity_critical"] = 0
        feats["ratio_warning"] = 0.0
        feats["ratio_error"] = 0.0
        feats["ratio_critical"] = 0.0
        feats["severity_entropy"] = 0.0
        feats["max_severity"] = 0
        return feats

    sev_counts = df_window["severity"].value_counts()
    feats["count_severity_info"] = int(sev_counts.get(1, 0) + sev_counts.get(0, 0) + sev_counts.get(2, 0))
    feats["count_severity_warning"] = int(sev_counts.get(3, 0))
    feats["count_severity_error"] = int(sev_counts.get(4, 0))
    feats["count_severity_critical"] = int(sev_counts.get(5, 0))

    feats["ratio_warning"] = feats["count_severity_warning"] / n_total
    feats["ratio_error"] = feats["count_severity_error"] / n_total
    feats["ratio_critical"] = feats["count_severity_critical"] / n_total

    probs = sev_counts.values / n_total
    probs = probs[probs > 0]
    feats["severity_entropy"] = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0
    feats["max_severity"] = int(df_window["severity"].max())

    return feats


# --- Niveau 3 : HTTP ---

def compute_http_features(df_window: pd.DataFrame) -> dict:
    """Features sur les logs HTTP (uniquement http_access)."""
    feats = {}

    http = df_window[df_window["log_kind"] == "http_access"]
    n_http = len(http)
    feats["count_http"] = n_http

    if n_http == 0:
        for cls in ["2xx", "3xx", "4xx", "5xx"]:
            feats[f"count_status_{cls}"] = 0
            feats[f"ratio_status_{cls}"] = 0.0
        for m in ["GET", "POST", "PUT", "DELETE", "OPTIONS"]:
            feats[f"count_method_{m}"] = 0
        feats["ratio_modifying"] = 0.0
        feats["ratio_healthcheck"] = 0.0
        for method, path in CRITICAL_ENDPOINTS:
            key = f"count_endpoint_{method}_{path.replace('/', '_').replace('<', '').replace('>', '')}"
            feats[key] = 0
        return feats

    sc_counts = http["http_status_class"].value_counts()
    for cls in ["2xx", "3xx", "4xx", "5xx"]:
        c = int(sc_counts.get(cls, 0))
        feats[f"count_status_{cls}"] = c
        feats[f"ratio_status_{cls}"] = c / n_http

    method_counts = http["http_method"].value_counts()
    for m in ["GET", "POST", "PUT", "DELETE", "OPTIONS"]:
        feats[f"count_method_{m}"] = int(method_counts.get(m, 0))

    n_modify = int(method_counts.get("POST", 0) + method_counts.get("PUT", 0) + method_counts.get("DELETE", 0))
    feats["ratio_modifying"] = n_modify / n_http

    n_health = int((http["endpoint_template"] == "GET /healthcheck").sum())
    feats["ratio_healthcheck"] = n_health / n_http

    for method, path in CRITICAL_ENDPOINTS:
        key = f"count_endpoint_{method}_{path.replace('/', '_').replace('<', '').replace('>', '')}"
        ep = f"{method} {path}"
        feats[key] = int((http["endpoint_template"] == ep).sum())

    return feats


# --- Niveau 4 : Latence ---

def compute_latency_features(df_window: pd.DataFrame) -> dict:
    """Features de latence HTTP (globales + par service critique)."""
    feats = {}

    http = df_window[df_window["log_kind"] == "http_access"]
    durations = http["http_duration_ms"].dropna()

    if len(durations) == 0:
        feats["latency_p50"] = 0.0
        feats["latency_p95"] = 0.0
        feats["latency_p99"] = 0.0
        feats["latency_max"] = 0.0
        feats["latency_mean"] = 0.0
        feats["latency_std"] = 0.0
        feats["count_latency_over_1s"] = 0
        feats["count_latency_over_5s"] = 0
    else:
        feats["latency_p50"] = float(np.percentile(durations, 50))
        feats["latency_p95"] = float(np.percentile(durations, 95))
        feats["latency_p99"] = float(np.percentile(durations, 99))
        feats["latency_max"] = float(durations.max())
        feats["latency_mean"] = float(durations.mean())
        feats["latency_std"] = float(durations.std()) if len(durations) > 1 else 0.0
        feats["count_latency_over_1s"] = int((durations > 1000).sum())
        feats["count_latency_over_5s"] = int((durations > 5000).sum())

    for svc in CRITICAL_LATENCY_SERVICES:
        svc_durations = http[http["programname"] == svc]["http_duration_ms"].dropna()
        svc_key = svc.replace('-', '_')
        if len(svc_durations) > 0:
            feats[f"latency_p95_{svc_key}"] = float(np.percentile(svc_durations, 95))
            feats[f"latency_max_{svc_key}"] = float(svc_durations.max())
        else:
            feats[f"latency_p95_{svc_key}"] = 0.0
            feats[f"latency_max_{svc_key}"] = 0.0

    return feats


# --- Niveau 5 : Sémantique (templates) ---

def compute_semantic_features(
    df_window: pd.DataFrame,
    known_template_ids: set[int],
) -> dict:
    """Features sur les templates Payload.

    known_template_ids : set des template_ids vus dans les fenêtres précédentes.
    """
    feats = {}
    n_total = len(df_window)

    if n_total == 0:
        feats["unique_templates"] = 0
        feats["new_templates_count"] = 0
        feats["unknown_templates_count"] = 0
        feats["top_template_dominance"] = 0.0
        feats["template_entropy"] = 0.0
        return feats

    template_ids = df_window["payload_template_id"]
    template_counts = template_ids.value_counts()
    feats["unique_templates"] = int(len(template_counts))

    # template_id == -1 : template inconnu en mode freeze (signal production)
    n_unknown = int((template_ids == -1).sum())
    feats["unknown_templates_count"] = n_unknown

    current_template_ids = set(template_ids.unique())
    new_templates = current_template_ids - known_template_ids - {-1}
    feats["new_templates_count"] = len(new_templates)

    top_count = template_counts.iloc[0]
    feats["top_template_dominance"] = float(top_count / n_total)

    probs = template_counts.values / n_total
    probs = probs[probs > 0]
    feats["template_entropy"] = float(-np.sum(probs * np.log2(probs)))

    return feats


# --- Niveau 6 : Temporel cyclique ---

def compute_temporal_features(window_start: pd.Timestamp) -> dict:
    """Features cycliques de l'heure et du jour de la semaine."""
    feats = {}

    hour = window_start.hour + window_start.minute / 60.0
    feats["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    feats["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))

    dow = window_start.weekday()
    feats["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    feats["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))

    feats["is_weekend"] = int(dow >= 5)
    feats["is_business_hours"] = int(8 <= window_start.hour < 18)
    feats["minute_of_day"] = window_start.hour * 60 + window_start.minute

    return feats


# --- Niveau 7 : Inter-fenêtres (deltas) ---

def add_delta_features(df_features: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """Ajoute des features delta (valeur actuelle - valeur précédente).

    Calculé sur le DataFrame de features ordonné par window_id.
    """
    df = df_features.sort_values("window_id").reset_index(drop=True)
    for col in target_cols:
        if col in df.columns:
            df[f"delta_{col}"] = df[col].diff().fillna(0.0)
    return df


# --- Orchestrateur ---

def compute_all_features(
    df_window: pd.DataFrame,
    window_id: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    all_groups: list[str],
) -> dict:
    """Calcule toutes les features pour une fenêtre donnée.
    """
    feats = {
        "window_id": window_id,
        "window_start": window_start,
        "window_end": window_end,
    }

    feats.update(compute_volume_features(df_window, all_groups))
    feats.update(compute_level_features(df_window))
    feats.update(compute_http_features(df_window))
    feats.update(compute_latency_features(df_window))

    # Niveau 5 : figé à 0 pour compatibilité de schéma
    feats["unique_templates"] = 0
    feats["new_templates_count"] = 0
    feats["unknown_templates_count"] = 0
    feats["top_template_dominance"] = 0.0
    feats["template_entropy"] = 0.0

    feats.update(compute_temporal_features(window_start))

    return feats
