"""Explicabilité : feature attribution (leave-one-out) et sélection des top-K logs représentatifs."""
import pandas as pd
import numpy as np
from typing import Optional

from ..utils.logging import logger
from ..models.isolation_forest import AnomalyDetector


def feature_attribution(
    detector: AnomalyDetector,
    window_features: pd.DataFrame,
    baseline: Optional[dict] = None,
    top_k: int = 10,
) -> list[dict]:
    """Calcule la contribution de chaque feature au score d'anomalie.

    Méthode : leave-one-out — on remplace la valeur d'une feature par sa
    valeur baseline (médiane training par défaut), on score, on compare.

    """
    if not detector.fitted:
        raise RuntimeError(f"[{detector.name}] not fitted")
    if len(window_features) != 1:
        raise ValueError(f"window_features must have exactly 1 row, got {len(window_features)}")

    # Score de référence
    base_score = detector.score(window_features)[0]

    # Baseline : médiane des features training (calculée depuis le scaler interne)
    # On utilise les valeurs d'origine (non scaled), donc on doit reconstruire
    # depuis le DataFrame de training.
    # Plus pragmatique : pour chaque feature, baseline = 0 (centre du RobustScaler)
    # ou la médiane reconstituée.
    if baseline is None:
        # Approche : remplacer chaque feature par 0 (qui est environ la médiane après RobustScaler)
        # NB: pour les features non-scaled (binaires, cycliques), 0 n'est pas le centre,
        # mais leur impact reste limité.
        baseline_values = {}
        for col in detector.feature_cols:
            # Pour les binaires/cycliques, valeur médiane plausible
            if col in ("is_weekend", "is_business_hours"):
                baseline_values[col] = 1  # heure ouvrée
            elif col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
                baseline_values[col] = window_features[col].iloc[0]  # garder
            elif col == "minute_of_day":
                baseline_values[col] = window_features[col].iloc[0]  # garder
            else:
                # Pour tout le reste : valeur 0 (approxime la médiane après RobustScaler)
                baseline_values[col] = 0.0
        baseline = baseline_values

    # Pour chaque feature : remplacer par baseline, score, diff
    # Optimisation : on skip d'abord les features dont la valeur ≈ baseline
    # (rien à expliquer), ce qui réduit drastiquement le nombre de scoring.
    contributions = []
    candidates = []
    for col in detector.feature_cols:
        if col not in window_features.columns:
            continue
        original_val = float(window_features[col].iloc[0])
        baseline_val = float(baseline.get(col, 0.0))
        if abs(original_val - baseline_val) < 1e-9:
            continue
        candidates.append((col, original_val, baseline_val))

    # Score reference + scoring batch des candidats
    # Pour optimiser : on construit une matrice où chaque ligne a UNE feature mise à baseline,
    # puis on score tout en une fois.
    if not candidates:
        return []

    n_cand = len(candidates)
    # Réplication de la fenêtre n_cand fois
    df_batch = pd.concat([window_features] * n_cand, ignore_index=True)
    for i, (col, _, baseline_val) in enumerate(candidates):
        df_batch.at[i, col] = baseline_val
    # Score batch
    modified_scores = detector.score(df_batch)

    for i, (col, original_val, baseline_val) in enumerate(candidates):
        contribution = base_score - modified_scores[i]
        contributions.append({
            "feature": col,
            "value": original_val,
            "baseline_value": baseline_val,
            "contribution": float(contribution),
        })

    # Trier par contribution positive décroissante
    
    contributions.sort(key=lambda c: c["contribution"], reverse=True)

    # Calculer pct (sur la somme des top contribs positives pour normalisation)
    pos_contribs = [c for c in contributions[:top_k] if c["contribution"] > 0]
    total_pos = sum(c["contribution"] for c in pos_contribs)
    if total_pos > 0:
        for c in contributions[:top_k]:
            c["contribution_pct"] = float(c["contribution"] / total_pos * 100) if c["contribution"] > 0 else 0.0
    else:
        for c in contributions[:top_k]:
            c["contribution_pct"] = 0.0

    return contributions[:top_k]


def select_top_k_logs(
    enriched_logs: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    top_features: list[dict],
    k: int = 10,
) -> pd.DataFrame:
    """Sélectionne les K logs les plus représentatifs de l'anomalie.

    Stratégie :
      Pour chaque top feature, identifier la "raison" et les logs candidats :
        - count_severity_error → logs avec severity == 4
        - count_severity_critical → logs avec severity == 5
        - count_status_5xx → logs avec http_status_class == '5xx'
        - count_status_4xx → logs avec http_status_class == '4xx'
        - unknown_templates_count → logs avec payload_template_id == -1
        - latency_p95* / latency_max* / count_latency_over_* → top latences
        - count_endpoint_X → logs sur cet endpoint
        - count_total → top logs (par volume de leur template)

      Puis on prend max K logs de l'union, en diversifiant par programname.


    """
    # Filtrer les logs de la fenêtre
    if enriched_logs["timestamp"].dt.tz is None:
        enriched_logs = enriched_logs.assign(timestamp=enriched_logs["timestamp"].dt.tz_localize("UTC"))
    if window_start.tz is None:
        window_start = window_start.tz_localize("UTC")
        window_end = window_end.tz_localize("UTC")

    win_logs = enriched_logs[
        (enriched_logs["timestamp"] >= window_start) &
        (enriched_logs["timestamp"] < window_end)
    ].copy()

    if win_logs.empty:
        return win_logs

    # Pour chaque top feature, identifier les logs candidats avec leur raison
    candidates = []  # liste de (idx, reason, priority_score)

    for feat in top_features:
        col = feat["feature"]
        # Si la contribution est négligeable, on skip
        if abs(feat["contribution"]) < 0.001:
            continue

        # === Mapping feature → critère de sélection ===
        if col == "count_severity_error":
            mask = win_logs["severity"] == 4
            reason = f"ERROR log (count={int(feat['value'])})"
            priority = abs(feat["contribution"])
        elif col == "count_severity_critical":
            mask = win_logs["severity"] == 5
            reason = f"CRITICAL log (count={int(feat['value'])})"
            priority = abs(feat["contribution"]) * 1.5  # CRITICAL > ERROR
        elif col == "count_severity_warning":
            mask = win_logs["severity"] == 3
            reason = f"WARNING log (count={int(feat['value'])})"
            priority = abs(feat["contribution"]) * 0.5
        elif col == "count_status_5xx":
            mask = win_logs["http_status_class"] == "5xx"
            reason = f"HTTP 5xx (count={int(feat['value'])})"
            priority = abs(feat["contribution"]) * 1.5
        elif col == "count_status_4xx":
            mask = win_logs["http_status_class"] == "4xx"
            reason = f"HTTP 4xx (count={int(feat['value'])})"
            priority = abs(feat["contribution"])
        elif col == "unknown_templates_count":
            mask = win_logs["payload_template_id"] == -1
            reason = f"Unknown template (count={int(feat['value'])})"
            priority = abs(feat["contribution"]) * 1.2
        elif col.startswith("latency_") or col.startswith("count_latency_"):
            # Top latences (jusqu'au top 5)
            if "http_duration_ms" in win_logs.columns:
                lat_logs = win_logs.dropna(subset=["http_duration_ms"]).nlargest(
                    5, "http_duration_ms"
                )
                for idx in lat_logs.index:
                    candidates.append((
                        idx,
                        f"High latency ({lat_logs.loc[idx, 'http_duration_ms']:.0f}ms) — feature {col}",
                        abs(feat["contribution"]),
                    ))
            continue
        elif col.startswith("count_endpoint_"):
            # Extraire l'endpoint depuis le nom du feature
            endpoint_key = col.replace("count_endpoint_", "")
            # Approximation : on prend les logs http_access avec method dans le nom
            mask = pd.Series(False, index=win_logs.index)
            for method in ("GET", "POST", "PUT", "DELETE"):
                if endpoint_key.startswith(method + "_"):
                    mask = win_logs["http_method"] == method
                    break
            reason = f"Endpoint anomaly: {endpoint_key} (count={int(feat['value'])})"
            priority = abs(feat["contribution"])
        elif col == "count_total":
            # Pour count_total anormal, on prend des logs représentatifs (par template fréquent)
            # On échantillonne max 3 logs par template dominant
            tpl_counts = win_logs["payload_template_id"].value_counts().head(3)
            for tpl_id in tpl_counts.index:
                tpl_logs = win_logs[win_logs["payload_template_id"] == tpl_id].head(3)
                for idx in tpl_logs.index:
                    candidates.append((
                        idx,
                        f"Volume anomaly (template_id={tpl_id} count={tpl_counts[tpl_id]})",
                        abs(feat["contribution"]) * 0.7,
                    ))
            continue
        elif col.startswith("count_group_"):
            group = col.replace("count_group_", "")
            mask = win_logs["groups"].apply(lambda gs: group in gs if isinstance(gs, list) else False)
            reason = f"Group {group} volume (count={int(feat['value'])})"
            priority = abs(feat["contribution"]) * 0.6
        elif col.startswith("count_method_"):
            method = col.replace("count_method_", "")
            mask = win_logs["http_method"] == method
            reason = f"HTTP {method} count anomaly (count={int(feat['value'])})"
            priority = abs(feat["contribution"]) * 0.7
        elif col == "unique_services":
            # Service silence : pas de log spécifique à pointer, on échantillonne le top
            tpl_counts = win_logs["programname"].value_counts().head(5)
            for prog in tpl_counts.index:
                prog_log = win_logs[win_logs["programname"] == prog].head(1)
                for idx in prog_log.index:
                    candidates.append((
                        idx,
                        f"Service diversity anomaly (only {int(feat['value'])} services active)",
                        abs(feat["contribution"]) * 0.5,
                    ))
            continue
        else:
            # Feature non gérée explicitement : on skip
            continue

        # Ajouter les logs qui matchent ce mask
        matching = win_logs[mask]
        for idx in matching.index:
            candidates.append((idx, reason, priority))

    if not candidates:
        # Fallback : top 5 logs par sévérité dans la fenêtre
        top = win_logs.nlargest(min(5, len(win_logs)), "severity")
        return top.assign(selection_reason="fallback_max_severity")

    # Dédupliquer par index, en gardant la raison/priorité la plus haute
    cand_df = pd.DataFrame(candidates, columns=["idx", "reason", "priority"])
    cand_df = cand_df.sort_values("priority", ascending=False).drop_duplicates("idx")

    # Diversification : prendre les K en assurant un mix de programnames
    # Stratégie : tri par priority desc, puis pour chaque candidat, on l'inclut
    # tant qu'on n'a pas dépassé K et qu'on diversifie (cap de N par programname)
    selected = []
    selected_indices = set()
    program_counts = {}
    cap_per_program = max(1, k // 3)  # max k/3 logs du même programname

    for _, row in cand_df.iterrows():
        if len(selected) >= k:
            break
        idx = row["idx"]
        if idx in selected_indices:
            continue
        prog = win_logs.loc[idx, "programname"]
        if program_counts.get(prog, 0) >= cap_per_program and len(selected) < k:
            continue
        selected.append((idx, row["reason"], row["priority"]))
        selected_indices.add(idx)
        program_counts[prog] = program_counts.get(prog, 0) + 1

    # Si pas assez : compléter sans contrainte de diversité
    if len(selected) < k:
        for _, row in cand_df.iterrows():
            if len(selected) >= k:
                break
            idx = row["idx"]
            if idx in selected_indices:
                continue
            selected.append((idx, row["reason"], row["priority"]))
            selected_indices.add(idx)

    # Construire DataFrame de sortie
    result_idx = [s[0] for s in selected]
    result = win_logs.loc[result_idx].copy()
    result["selection_reason"] = [s[1] for s in selected]
    result["selection_priority"] = [s[2] for s in selected]
    return result.sort_values("timestamp").reset_index(drop=True)
