"""Onglet Live : KPIs en temps réel + time series + heatmap.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import streamlit as st


def render(df_alerts: pd.DataFrame, alerts: list[dict], worker_state: Optional[dict]) -> None:
    if df_alerts.empty:
        _render_empty_state(worker_state)
        return

    # Row 1 : KPIs principaux (4 metrics + 1 caption)
    now = datetime.now(timezone.utc)
    last_hour = now - timedelta(hours=1)
    last_day = now - timedelta(hours=24)

    df_h = df_alerts[df_alerts["window_start"] >= pd.Timestamp(last_hour)]
    df_d = df_alerts[df_alerts["window_start"] >= pd.Timestamp(last_day)]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Alertes (1h)",
            len(df_h),
            delta=f"{len(df_h[df_h['severity'].isin(['high', 'critical'])])} sévères",
            delta_color="inverse" if len(df_h) > 0 else "off",
        )
    with col2:
        st.metric(
            "Alertes (24h)",
            len(df_d),
            delta=f"{len(df_d[df_d['severity'].isin(['high', 'critical'])])} sévères",
            delta_color="inverse" if len(df_d) > 0 else "off",
        )
    with col3:
        score_max_h = float(df_h["final_score"].max()) if len(df_h) > 0 else 0.0
        st.metric(
            "Score max (1h)",
            f"{score_max_h:.2f}",
            delta="≥ 0.90 = anomalie ML forte" if score_max_h >= 0.9 else "ras",
            delta_color="off",
        )
    with col4:
        decision_counts = df_d["decision_source"].value_counts()
        rules_pct = 100 * decision_counts.get("rules_only", 0) / max(1, len(df_d))
        ml_pct = 100 * decision_counts.get("ml_only", 0) / max(1, len(df_d))
        both_pct = 100 * decision_counts.get("both", 0) / max(1, len(df_d))
        st.metric(
            "Source décision (24h)",
            f"{both_pct:.0f}% both",
            delta=f"{ml_pct:.0f}% ML / {rules_pct:.0f}% rules",
            delta_color="off",
        )

    st.divider()

    # Row 2 : Time series (alertes par minute) + Severity breakdown
    col_ts, col_sev = st.columns([2, 1])

    with col_ts:
        st.markdown("##### Alertes par fenêtre temporelle")
        ts_chart = _build_timeseries_chart(df_alerts)
        if ts_chart is not None:
            st.altair_chart(ts_chart, use_container_width=True)
        else:
            st.info("Pas assez de données pour le graphique temporel.")

    with col_sev:
        st.markdown("##### Répartition par sévérité")
        sev_counts = df_alerts["severity"].value_counts().reindex(
            ["critical", "high", "medium", "low"], fill_value=0
        )
        # Bar chart Streamlit natif (pas de dépendance plotly)
        st.bar_chart(sev_counts, color="#c2410c")
        # Petit complément textuel
        for sev, n in sev_counts.items():
            css_class = f"severity-{sev}"
            st.markdown(f"<span class='{css_class}'>{sev}</span> : **{int(n)}**",
                        unsafe_allow_html=True)

    st.divider()

    # Row 3 : Heatmap règles × heure de la journée
    st.markdown("##### Heatmap : règles métier déclenchées par heure")
    heatmap = _build_rules_heatmap(df_alerts)
    if heatmap is not None:
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("Pas assez d'alertes pour la heatmap.")

    # Row 4 : Liste des dernières alertes (table compacte)
    st.markdown("##### Dernières alertes")
    n_show = min(10, len(df_alerts))
    df_recent = df_alerts.tail(n_show).iloc[::-1].copy()  # plus récente en haut
    df_recent_display = df_recent[[
        "window_start", "severity", "decision_source",
        "final_score", "triggered_rules", "alert_id",
    ]].copy()
    df_recent_display.columns = [
        "Fenêtre", "Sévérité", "Source", "Score", "Règles", "ID",
    ]
    df_recent_display["Fenêtre"] = df_recent_display["Fenêtre"].dt.strftime("%Y-%m-%d %H:%M")
    df_recent_display["Score"] = df_recent_display["Score"].round(3)
    st.dataframe(
        df_recent_display,
        use_container_width=True,
        hide_index=True,
        height=min(400, 50 + 35 * n_show),
    )


# Helpers

def _render_empty_state(worker_state: Optional[dict]) -> None:
    """Affiche un message d'accueil quand il n'y a pas encore d'alertes."""
    if worker_state is None:
        st.info(
            "👋 **Aucune donnée pour le moment.**\n\n"
            "Le worker temps réel ne semble pas tourner. Lance-le avec :\n\n"
            "```\npython -m anomaly_detection.cli live\n```\n\n"
            "Ou en mode replay :\n\n"
            "```\npython -m anomaly_detection.cli live --replay data/test_anomalies/test_day.json --speed 60\n```"
        )
    else:
        st.success(
            "✅ Le worker tourne mais n'a pas encore détecté d'anomalies. "
            "Le dashboard se mettra à jour automatiquement dès qu'une alerte est émise."
        )
        st.caption(
            f"Itérations : **{worker_state.get('iter_count', 0)}**, "
            f"fenêtres émises : **{worker_state.get('windows_emitted_count', 0)}**"
        )


def _build_timeseries_chart(df_alerts: pd.DataFrame):
    """Bar chart des alertes par fenêtre, coloré par sévérité.

    Utilise Altair (déjà inclus dans Streamlit) plutôt que st.bar_chart
    pour avoir le stacking par sévérité.
    """
    try:
        import altair as alt
    except ImportError:
        return None

    if df_alerts.empty:
        return None

    df = df_alerts.copy()
    # Granularité adaptative selon la plage temporelle
    span = df["window_start"].max() - df["window_start"].min()
    if span < pd.Timedelta(hours=2):
        df["bucket"] = df["window_start"].dt.floor("1min")
        time_unit = "yearmonthdatehoursminutes"
    elif span < pd.Timedelta(days=2):
        df["bucket"] = df["window_start"].dt.floor("5min")
        time_unit = "yearmonthdatehoursminutes"
    else:
        df["bucket"] = df["window_start"].dt.floor("1h")
        time_unit = "yearmonthdatehours"

    grouped = (
        df.groupby(["bucket", "severity"]).size().reset_index(name="count")
    )

    severity_order = ["low", "medium", "high", "critical"]
    severity_colors = ["#9ca3af", "#d97706", "#c2410c", "#b91c1c"]

    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("bucket:T", title="Temps", axis=alt.Axis(format="%H:%M")),
            y=alt.Y("count:Q", title="Nombre d'alertes"),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(domain=severity_order, range=severity_colors),
                sort=severity_order,
                legend=alt.Legend(title="Sévérité"),
            ),
            tooltip=["bucket:T", "severity:N", "count:Q"],
        )
        .properties(height=240)
    )
    return chart


def _build_rules_heatmap(df_alerts: pd.DataFrame):
    """Heatmap (règle × heure) du nombre d'alertes."""
    try:
        import altair as alt
    except ImportError:
        return None

    if df_alerts.empty:
        return None

    # Une ligne par (alert, rule) — explose triggered_rules en lignes
    rows = []
    for _, alert in df_alerts.iterrows():
        rules = alert["triggered_rules"]
        if not rules:
            continue
        for r in rules.split(","):
            r = r.strip()
            if r:
                rows.append({"hour": alert["window_start"].floor("1h"), "rule": r})

    if not rows:
        return None

    df_rules = pd.DataFrame(rows)
    grouped = (
        df_rules.groupby(["hour", "rule"]).size().reset_index(name="count")
    )

    chart = (
        alt.Chart(grouped)
        .mark_rect()
        .encode(
            x=alt.X("hour:T", title="Heure", axis=alt.Axis(format="%m-%d %H:%M")),
            y=alt.Y("rule:N", title="Règle métier", sort="-x"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="orangered"),
                            legend=alt.Legend(title="N alertes")),
            tooltip=["hour:T", "rule:N", "count:Q"],
        )
        .properties(height=200)
    )
    return chart
