"""Onglet Incidents : timeline + détail (LLM compris).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


def render(df_alerts: pd.DataFrame, alerts: list[dict],
           enriched_index: dict[str, dict]) -> None:
    if df_alerts.empty:
        st.info("Aucune alerte n'a encore été générée.")
        return

    # 1. Reconstruire les incidents (groupement temporel par signature)
    incidents = _build_incidents(alerts)

    if not incidents:
        st.info("Aucun incident à afficher.")
        return

    # 2. Timeline en barres horizontales (Gantt-style)
    st.markdown("##### Timeline des incidents")
    chart = _build_timeline_chart(incidents)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("(Altair indisponible — passage en mode tableau)")

    st.divider()

    # 3. Tableau des incidents + sélection
    st.markdown("##### Liste des incidents")

    df_inc = pd.DataFrame([{
        "ID": inc["incident_id"][:12],
        "Début": pd.Timestamp(inc["first_window_start"]).strftime("%Y-%m-%d %H:%M"),
        "Durée (min)": inc["duration_minutes"],
        "N alertes": inc["n_alerts"],
        "Sévérité max": inc["max_severity"],
        "Score max": round(inc["max_score"], 3),
        "Règles": ", ".join(inc["triggered_rules"]) if inc["triggered_rules"] else "—",
    } for inc in incidents])

    df_inc = df_inc.sort_values("Début", ascending=False).reset_index(drop=True)
    st.dataframe(df_inc, use_container_width=True, hide_index=True, height=300)

    # 4. Détail d'un incident sélectionné
    incident_ids = [inc["incident_id"] for inc in incidents]
    incident_labels = [
        f"{pd.Timestamp(inc['first_window_start']).strftime('%H:%M')} "
        f"— {inc['max_severity']} — {inc['n_alerts']} alertes"
        for inc in incidents
    ]
    selected_idx = st.selectbox(
        "Sélectionne un incident pour voir le détail :",
        options=list(range(len(incidents))),
        format_func=lambda i: incident_labels[i],
        index=0,
    )

    if selected_idx is not None:
        _render_incident_detail(incidents[selected_idx], alerts, enriched_index)


# Helpers

def _build_incidents(alerts: list[dict]) -> list[dict]:
    """Reconstruit les incidents à partir des alertes brutes.

    Réutilise alerting.dedup.deduplicate_alerts pour la cohérence avec
    le pipeline batch. On reconstruit temporairement des AnomalyAlert
    depuis les dicts JSON.
    """
    if not alerts:
        return []

    try:
        from anomaly_detection.alerting.dedup import deduplicate_alerts
        from anomaly_detection.alerting.schema import AnomalyAlert
    except ImportError:
        return []

    # Reconstruire les AnomalyAlert depuis les dicts
    rebuilt = []
    for a in alerts:
        try:
            # AnomalyAlert n'expose pas de from_dict, on instancie manuellement
            # avec les champs flat (les listes top_features/top_logs sont des
            # dicts simples à ce stade — c'est OK pour le dedup qui ne lit
            # que alert_id, window_start/end, severity, decision_source,
            # final_score, triggered_rules)
            rebuilt.append(_DictAlertProxy(a))
        except Exception:
            continue

    return deduplicate_alerts(rebuilt, max_gap_minutes=5)


class _DictAlertProxy:
    """Vue minimale d'une alerte (depuis dict JSON) pour deduplicate_alerts.

    Reproduit l'interface AnomalyAlert nécessaire au déduplicateur.
    """
    def __init__(self, d: dict):
        self._d = d
        self.alert_id = d.get("alert_id", "")
        self.window_start = d.get("window_start", "")
        self.window_end = d.get("window_end", "")
        self.severity = d.get("severity", "low")
        self.decision_source = d.get("decision_source", "unknown")
        self.final_score = float(d.get("final_score", 0.0))
        self.triggered_rules = list(d.get("triggered_rules", []) or [])

    def to_dict(self) -> dict:
        return self._d


def _build_timeline_chart(incidents: list[dict]):
    """Construit une timeline Altair (barres horizontales par incident)."""
    try:
        import altair as alt
    except ImportError:
        return None

    if not incidents:
        return None

    rows = []
    for inc in incidents:
        rows.append({
            "incident_id": inc["incident_id"][:12],
            "start": pd.Timestamp(inc["first_window_start"]),
            "end": pd.Timestamp(inc["last_alert_window_end"]),
            "severity": inc["max_severity"],
            "n_alerts": inc["n_alerts"],
            "rules": ", ".join(inc["triggered_rules"]) if inc["triggered_rules"] else "—",
        })
    df = pd.DataFrame(rows)

    severity_order = ["low", "medium", "high", "critical"]
    severity_colors = ["#9ca3af", "#d97706", "#c2410c", "#b91c1c"]

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("start:T", title="Temps", axis=alt.Axis(format="%H:%M")),
            x2="end:T",
            y=alt.Y("incident_id:N", title=None, sort="-x"),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(domain=severity_order, range=severity_colors),
                legend=alt.Legend(title="Sévérité max"),
            ),
            tooltip=[
                alt.Tooltip("incident_id:N", title="ID"),
                alt.Tooltip("start:T", title="Début"),
                alt.Tooltip("end:T", title="Fin"),
                alt.Tooltip("severity:N", title="Sévérité"),
                alt.Tooltip("n_alerts:Q", title="Nb alertes"),
                alt.Tooltip("rules:N", title="Règles"),
            ],
        )
        .properties(height=max(200, 30 * len(df)))
    )
    return chart


def _render_incident_detail(incident: dict, alerts: list[dict],
                              enriched_index: dict[str, dict]) -> None:
    """Affiche le détail d'un incident : métadonnées + alertes + LLM."""
    st.markdown("---")
    st.markdown(f"### Détail de l'incident `{incident['incident_id'][:12]}…`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sévérité", incident["max_severity"])
    with col2:
        st.metric("Durée", f"{incident['duration_minutes']:.1f} min")
    with col3:
        st.metric("Alertes groupées", incident["n_alerts"])

    st.markdown(f"**Règles déclenchées :** "
                f"{', '.join('`'+r+'`' for r in incident['triggered_rules']) if incident['triggered_rules'] else '_aucune_'}")
    st.markdown(f"**Source décision :** `{incident['decision_sources'][0]}`")
    st.markdown(f"**Plage temporelle :** `{incident['first_window_start']}` → "
                f"`{incident['last_alert_window_end']}`")

    # Détail de la première alerte de l'incident
    first_alert = incident["first_alert"]
    st.markdown("##### Top features de la première alerte")
    top_feats = first_alert.get("top_features", [])
    if top_feats:
        df_f = pd.DataFrame([{
            "Feature": f["feature"],
            "Valeur": round(f["value"], 3),
            "Baseline": round(f["baseline_value"], 3),
            "Contribution": round(f["contribution"], 3),
            "%": round(f["contribution_pct"], 1),
        } for f in top_feats[:8]])
        st.dataframe(df_f, use_container_width=True, hide_index=True)
    else:
        st.caption("(Pas de top_features)")

    # Top logs représentatifs
    st.markdown("##### Top logs représentatifs")
    top_logs = first_alert.get("top_logs", [])
    if top_logs:
        df_l = pd.DataFrame([{
            "Heure": l.get("timestamp", "")[11:19],
            "Service": l.get("programname", ""),
            "Niv.": l.get("log_level", "") or "",
            "Status": l.get("http_status", "") or "",
            "Lat. ms": l.get("http_duration_ms", "") or "",
            "Payload": (l.get("payload", "") or "")[:120],
            "Raison": l.get("selection_reason", ""),
        } for l in top_logs[:10]])
        st.dataframe(df_l, use_container_width=True, hide_index=True)
    else:
        st.caption("(Pas de top_logs)")

    # Analyse LLM enrichie (si disponible)
    aid = first_alert.get("alert_id")
    enriched = enriched_index.get(aid) if aid else None
    if enriched:
        analysis = enriched.get("llm_analysis", {})
        st.markdown("##### 🤖 Analyse LLM")
        provider = analysis.get("provider", "?")
        st.caption(f"Provider : **{provider}** · "
                   f"latence {analysis.get('latency_ms', 0):.0f}ms · "
                   f"grounding {'✓' if analysis.get('grounding_valid') else '⚠'}")
        st.markdown(f"> _{analysis.get('summary', '_pas de résumé_')}_")

        primary = analysis.get("primary_hypothesis", {}) or {}
        if primary.get("cause"):
            st.markdown(f"**Cause probable** ({primary.get('confidence', 'low')} confiance) : "
                        f"{primary['cause']}")

        alts = analysis.get("alternative_hypotheses", []) or []
        if alts:
            with st.expander(f"Hypothèses alternatives ({len(alts)})"):
                for h in alts:
                    st.markdown(f"- _{h.get('cause', '')}_ ({h.get('confidence', 'low')})")

        invs = analysis.get("suggested_investigations", []) or []
        if invs:
            with st.expander(f"Investigations suggérées ({len(invs)})"):
                for i in invs:
                    st.markdown(f"- {i}")
    else:
        st.caption("ℹ️ Pas d'analyse LLM pour cette alerte (sévérité ou enrichissement non encore disponible).")
