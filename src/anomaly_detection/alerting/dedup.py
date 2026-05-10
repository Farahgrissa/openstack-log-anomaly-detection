"""Déduplication temporelle d'alertes : groupe les fenêtres adjacentes en incidents."""
import pandas as pd
from datetime import timedelta
from typing import Iterable

from .schema import AnomalyAlert
from ..utils.logging import logger


def deduplicate_alerts(
    alerts: list[AnomalyAlert],
    max_gap_minutes: int = 5,
) -> list[dict]:
    """Groupe les alertes adjacentes en incidents.
    """
    if not alerts:
        return []

    # Trier par window_start
    sorted_alerts = sorted(alerts, key=lambda a: a.window_start)

    incidents = []
    current_incident: dict | None = None

    for alert in sorted_alerts:
        alert_start = pd.Timestamp(alert.window_start)
        alert_end = pd.Timestamp(alert.window_end)

        # Test de groupement avec l'incident en cours
        can_group = False
        if current_incident is not None:
            last_end = pd.Timestamp(current_incident["last_alert_window_end"])
            gap = (alert_start - last_end).total_seconds() / 60.0
            same_rules = (set(alert.triggered_rules) ==
                          set(current_incident["triggered_rules"]))
            same_decision = (alert.decision_source ==
                             current_incident["decision_sources"][0])
            # Group si : même fingerprint exact, OU mêmes rules et fenêtre adjacente
            if alert.alert_id == current_incident["fingerprint"]:
                can_group = True
            elif same_rules and same_decision and gap <= max_gap_minutes:
                can_group = True

        if can_group:
            current_incident["n_alerts"] += 1
            current_incident["last_alert_window_end"] = alert.window_end
            current_incident["max_score"] = max(current_incident["max_score"], alert.final_score)
            current_incident["alert_ids"].append(alert.alert_id)
            # Mise à jour severity (max)
            sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if sev_order.get(alert.severity, 0) > sev_order.get(current_incident["max_severity"], 0):
                current_incident["max_severity"] = alert.severity
        else:
            # Démarrer un nouvel incident
            if current_incident is not None:
                incidents.append(_finalize_incident(current_incident))
            current_incident = {
                "incident_id": alert.alert_id,
                "fingerprint": alert.alert_id,
                "first_alert": alert.to_dict(),
                "first_window_start": alert.window_start,
                "last_alert_window_end": alert.window_end,
                "n_alerts": 1,
                "max_severity": alert.severity,
                "max_score": alert.final_score,
                "decision_sources": [alert.decision_source],
                "triggered_rules": alert.triggered_rules,
                "alert_ids": [alert.alert_id],
            }

    if current_incident is not None:
        incidents.append(_finalize_incident(current_incident))

    logger.info(f"Deduplication: {len(alerts)} alerts → {len(incidents)} incidents")
    return incidents


def _finalize_incident(incident: dict) -> dict:
    """Finalise un incident en calculant la durée."""
    start = pd.Timestamp(incident["first_window_start"])
    end = pd.Timestamp(incident["last_alert_window_end"])
    duration_min = (end - start).total_seconds() / 60.0
    incident["duration_minutes"] = round(duration_min, 1)
    # Cleanup
    incident.pop("fingerprint", None)
    return incident
