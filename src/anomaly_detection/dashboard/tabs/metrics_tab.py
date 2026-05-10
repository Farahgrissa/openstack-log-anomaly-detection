"""Onglet Modèle : métriques offline (Phase 4 - évaluation).
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def render() -> None:
    from anomaly_detection.dashboard import data_layer
    paths = data_layer.get_paths()

    eval_path = paths["eval_results"]
    if not eval_path.exists():
        st.warning(
            f"Le fichier `{eval_path.name}` n'existe pas encore.\n\n"
            "Lance d'abord :\n```\n"
            "python -m anomaly_detection.cli train\n"
            "python -m anomaly_detection.cli evaluate\n"
            "```"
        )
        return

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        st.error(f"Erreur de lecture de `{eval_path}` : {e}")
        return

    # 1. Métriques globales
    st.markdown("##### Métriques globales (modèle ML seul)")

    g = results.get("global", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("AUC-ROC", f"{g.get('auc_roc', 0):.3f}", "Cible : >0.85")
    with col2:
        st.metric("AUC-PR", f"{g.get('auc_pr', 0):.3f}", "Cible : >0.70")
    with col3:
        st.metric("Précision", f"{g.get('precision', 0):.2%}")
    with col4:
        st.metric("Recall", f"{g.get('recall', 0):.2%}")
    with col5:
        st.metric("F1", f"{g.get('f1', 0):.3f}")

    col_fp1, col_fp2 = st.columns(2)
    with col_fp1:
        st.metric("Faux positifs",
                  f"{g.get('fp', 0)} / {g.get('n_normal', 0)}",
                  f"FPR = {100 * g.get('false_positive_rate', 0):.2f}%")
    with col_fp2:
        st.metric("Vrais positifs",
                  f"{g.get('tp', 0)} / {g.get('n_anomaly', 0)}",
                  f"sur {g.get('n_total', 0)} fenêtres au total")

    st.caption(
        "ℹ️ Ces métriques mesurent uniquement le modèle ML pur (Isolation Forest). "
        "Le **détecteur hybride en production** combine ML + 6 règles métier "
        "et atteint **F1 = 0.84** (vs 0.62 ici) — gain documenté dans `docs/architecture.md`."
    )

    st.divider()

    # 2. Rapport visuel ROC/PR (image générée par scripts/evaluate)
    eval_report = paths["eval_report_png"]
    if eval_report.exists():
        st.markdown("##### Rapport visuel ROC / PR")
        st.image(str(eval_report), use_container_width=True)
    else:
        st.caption(f"_(rapport visuel `{eval_report.name}` non disponible)_")

    st.divider()

    # 3. Recall par type d'anomalie
    st.markdown("##### Recall par type d'anomalie injectée")
    by_type = g.get("by_type", {})
    if by_type:
        rows = []
        for anomaly_type, m in by_type.items():
            rows.append({
                "Type": anomaly_type,
                "Fenêtres": m.get("n_windows", 0),
                "Détectées (ML)": m.get("n_detected", 0),
                "Recall (ML)": f"{100 * m.get('recall', 0):.0f}%",
                "Score P50": round(m.get("score_p50", 0), 3),
                "Score P95": round(m.get("score_p95", 0), 3),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "💡 Les types `error_burst_cinder` et `new_error_templates` ont un "
            "recall ML proche de 0 parce que les features `count_severity_error` "
            "et `unknown_templates_count` sont **constantes en training**. "
            "C'est précisément le rôle des règles métier de les rattraper "
            "(architecture hybride détaillée dans `docs/architecture.md`)."
        )
    else:
        st.caption("_(pas de breakdown par type dans eval_results.json)_")

    # 4. Modèles par groupe (s'ils existent)
    group_models = {k: v for k, v in results.items() if k != "global"}
    if group_models:
        st.divider()
        st.markdown("##### Modèles par groupe métier")
        rows = []
        for name, m in group_models.items():
            rows.append({
                "Groupe": name,
                "AUC-ROC": round(m.get("auc_roc", 0), 3),
                "AUC-PR": round(m.get("auc_pr", 0), 3),
                "Précision": f"{100 * m.get('precision', 0):.0f}%",
                "Recall": f"{100 * m.get('recall', 0):.0f}%",
                "F1": round(m.get("f1", 0), 3),
                "Features": m.get("n_features", "?"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
