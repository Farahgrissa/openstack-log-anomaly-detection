"""Onglet Logs : explorateur full-text des top_logs.

Recherche transverse à toutes les alertes : tu peux chercher un programname,
un user_id, un endpoint, un fragment de payload… Les résultats te ramènent
à l'alerte d'origine.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render(alerts: list[dict]) -> None:
    if not alerts:
        st.info("Aucune alerte n'a encore été générée — pas de logs à explorer.")
        return

    # 1. Construire un DataFrame plat de tous les top_logs
    rows = []
    for alert in alerts:
        aid = alert.get("alert_id", "")
        ws = alert.get("window_start", "")
        sev = alert.get("severity", "low")
        for log in alert.get("top_logs", []) or []:
            rows.append({
                "alert_id": aid,
                "alert_severity": sev,
                "window_start": ws,
                "timestamp": log.get("timestamp", ""),
                "programname": log.get("programname", ""),
                "log_level": log.get("log_level") or "",
                "severity_num": int(log.get("severity", 0)),
                "http_method": log.get("http_method") or "",
                "http_status": log.get("http_status") if log.get("http_status") else "",
                "http_duration_ms": log.get("http_duration_ms") if log.get("http_duration_ms") else "",
                "request_id": log.get("request_id") or "",
                "user_id": log.get("user_id") or "",
                "tenant_id": log.get("tenant_id") or "",
                "payload": log.get("payload", "") or "",
                "selection_reason": log.get("selection_reason", ""),
            })
    if not rows:
        st.info("Les alertes ne contiennent pas de top_logs.")
        return
    df_logs = pd.DataFrame(rows)

    # 2. Barre de recherche + filtres
    st.markdown("##### Filtres")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input(
            "🔎 Recherche full-text (programname, payload, user_id, endpoint, …)",
            placeholder="ex: cinder-volume, ERROR, /v3/auth/tokens, …",
        )
    with col2:
        services = sorted(df_logs["programname"].dropna().unique().tolist())
        selected_services = st.multiselect("Services", options=services)
    with col3:
        severities = ["low", "medium", "high", "critical"]
        selected_sev = st.multiselect(
            "Sévérité (alerte)", options=severities, default=severities,
        )

    # 3. Application des filtres
    df = df_logs.copy()
    if query:
        # Recherche dans plusieurs colonnes texte (case-insensitive)
        q = query.lower()
        mask = (
            df["programname"].str.lower().str.contains(q, na=False)
            | df["payload"].str.lower().str.contains(q, na=False)
            | df["user_id"].str.lower().str.contains(q, na=False)
            | df["request_id"].str.lower().str.contains(q, na=False)
            | df["log_level"].astype(str).str.lower().str.contains(q, na=False)
        )
        df = df[mask]
    if selected_services:
        df = df[df["programname"].isin(selected_services)]
    if selected_sev:
        df = df[df["alert_severity"].isin(selected_sev)]

    st.markdown(f"**{len(df)} logs** correspondent (sur {len(df_logs)} au total).")

    if df.empty:
        st.info("Aucun log ne correspond à ces critères.")
        return

    # 4. Affichage du tableau (avec extrait de payload)
    df_display = df[[
        "timestamp", "programname", "log_level", "http_method",
        "http_status", "http_duration_ms",
        "payload", "alert_severity", "alert_id",
    ]].copy()
    df_display["payload"] = df_display["payload"].str.slice(0, 200)
    df_display.columns = [
        "Heure", "Service", "Niv.", "Méthode", "Status", "Lat. ms",
        "Payload", "Sév. alerte", "Alerte ID",
    ]
    df_display["Heure"] = df_display["Heure"].astype(str).str[11:19]
    st.dataframe(
        df_display.head(500),  # cap à 500 lignes pour rester fluide
        use_container_width=True,
        hide_index=True,
        height=600,
    )
    if len(df) > 500:
        st.caption(f"⚠ Tableau tronqué à 500 lignes (total : {len(df)}). "
                   "Affine les filtres pour voir le reste.")
