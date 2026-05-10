"""Dashboard Streamlit — 4 onglets (Live, Incidents, Logs, Modèle), auto-refresh."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="OpenStack Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from anomaly_detection.dashboard import data_layer
from anomaly_detection.dashboard.tabs import live_tab, incidents_tab, logs_tab, metrics_tab


st.markdown("""
<style>
.severity-critical { color: #b91c1c; font-weight: 600; }
.severity-high     { color: #c2410c; font-weight: 600; }
.severity-medium   { color: #b45309; font-weight: 500; }
.severity-low      { color: #4b5563; font-weight: 500; }
.muted             { color: #6b7280; font-size: 0.85em; }
[data-testid="stMetricValue"] { font-size: 2rem; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.title("🛰️ Anomaly Detection")
    st.caption("OpenStack Logs - Real-time Monitoring")

    # Statut du worker
    worker_state = data_layer.load_worker_state()
    status, badge, message = data_layer.worker_status_summary(worker_state)
    st.markdown(f"### {badge} Statut du worker")
    st.caption(message)

    if worker_state is not None:
        with st.expander("État détaillé du worker", expanded=False):
            st.write(f"**Itérations** : {worker_state.get('iter_count', '?')}")
            st.write(f"**Fenêtres émises** : {worker_state.get('windows_emitted_count', '?')}")
            st.write(f"**Alertes générées** : {worker_state.get('alerts_generated_count', '?')}")
            st.write(f"**Mode** : {worker_state.get('mode', '?')}")
            buf = worker_state.get("buffer_stats", {}) or {}
            st.write(f"**Buffer** : {buf.get('n_logs', 0)} logs, {buf.get('n_emitted_windows', 0)} fenêtres émises")
            llm_stats = worker_state.get("llm_worker_stats")
            if llm_stats:
                st.write(
                    f"**LLM** : {llm_stats.get('n_processed', 0)} analysées / "
                    f"{llm_stats.get('queue_size', 0)} en file / "
                    f"{llm_stats.get('n_dropped', 0)} droppées"
                )
            if worker_state.get("last_timestamp_seen"):
                st.write(f"**Dernier log vu** : `{worker_state['last_timestamp_seen']}`")

    st.divider()

    # Contrôles
    st.markdown("### Contrôles")
    auto_refresh = st.toggle("Auto-refresh", value=True, help="Recharger les données périodiquement")
    refresh_interval = st.slider("Intervalle (s)", 5, 60, 10, step=5, disabled=not auto_refresh)

    if st.button("🔄 Recharger maintenant", use_container_width=True):
        data_layer.reset_cache()
        st.rerun()

    st.divider()
    st.caption(
        "Dashboard lit en append-only :\n"
        "- `data/alerts/alerts.jsonl`\n"
        "- `data/alerts/alerts_enriched.jsonl`\n"
        "- `data/realtime_state.json`"
    )



col_title, col_count = st.columns([3, 1])
with col_title:
    st.title("Détection d'anomalies — OpenStack")

# Charger les données (incrémental → rapide)
alerts = data_layer.load_alerts()
enriched = data_layer.load_enriched_alerts()
df_alerts = data_layer.alerts_to_dataframe(alerts)
enriched_index = data_layer.index_enriched_by_alert_id(enriched)

with col_count:
    n_critical = int((df_alerts["severity"] == "critical").sum()) if not df_alerts.empty else 0
    n_high = int((df_alerts["severity"] == "high").sum()) if not df_alerts.empty else 0
    st.metric(
        label="Alertes critiques + sévères",
        value=f"{n_critical + n_high}",
        delta=f"sur {len(df_alerts)} au total" if len(df_alerts) > 0 else None,
        delta_color="off",
    )

# Onglets
tab_live, tab_incidents, tab_logs, tab_metrics = st.tabs([
    "📊 Live",
    "📅 Incidents",
    "🔎 Logs",
    "🎯 Modèle",
])

with tab_live:
    live_tab.render(df_alerts, alerts, worker_state)

with tab_incidents:
    incidents_tab.render(df_alerts, alerts, enriched_index)

with tab_logs:
    logs_tab.render(alerts)

with tab_metrics:
    metrics_tab.render()



if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()
