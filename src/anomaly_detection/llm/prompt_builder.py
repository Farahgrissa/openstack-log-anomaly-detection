"""Construction de prompts grounded pour analyse d'alertes.

Strategie anti-hallucination :
  1. SYSTEM prompt strict avec regles de grounding
  2. Donnees fournies en format structure (pas de texte libre)
  3. Logs numerotes pour citation par index
  4. Schema JSON impose
"""
from typing import Any


SYSTEM_PROMPT = """You are an OpenStack incident analyst.

Your task: analyze ONE detected anomaly and produce a concise, factual root-cause hypothesis based ONLY on the data provided.

CRITICAL RULES:
1. Use ONLY information from the alert below. Do NOT invent facts, services, or causes.
2. Every claim MUST cite specific log indices (0-based) from the LOGS list when relevant.
3. If evidence is insufficient for a confident hypothesis, say so explicitly and set confidence to "low".
4. Output MUST be valid JSON matching the schema (no markdown fences, no extra text).
5. Keep summary < 200 chars. Be specific, avoid generic phrases like "investigate the issue".
6. suggested_investigations must be CONCRETE actions (e.g., "Check disk usage on /var/lib/nova"), not vague advice.

Output language: French (the user is a French OpenStack admin)."""


def format_alert_for_prompt(alert: dict) -> str:
    """Formate une AnomalyAlert (dict) en prompt structure."""
    parts = []

    parts.append("ANOMALY ALERT:")
    parts.append(f"- Window: {alert['window_start']} to {alert['window_end']}")
    parts.append(f"- Severity: {alert['severity']}")
    parts.append(f"- Final score: {alert['final_score']:.3f} (ML proba: {alert['ml_proba']:.3f}, rules max: {alert['rules_max_score']:.3f})")
    parts.append(f"- Decision source: {alert['decision_source']}")
    parts.append(f"- Triggered rules: {alert['triggered_rules']}")

    parts.append("\nWINDOW STATISTICS:")
    stats = alert.get("window_stats", {})
    for k, v in stats.items():
        parts.append(f"- {k}: {v}")

    parts.append("\nTOP CONTRIBUTING FEATURES:")
    for f in alert.get("top_features", []):
        parts.append(
            f"- {f['feature']}: value={f['value']:.2f}, "
            f"baseline={f['baseline_value']:.2f}, "
            f"contribution={f['contribution']:+.3f}"
        )

    parts.append("\nREPRESENTATIVE LOGS (cite by index 0..N-1):")
    for i, log in enumerate(alert.get("top_logs", [])):
        ts = log.get("timestamp", "")[:19]
        prog = log.get("programname", "?")
        lvl = log.get("log_level") or "-"
        payload = log.get("payload", "")[:200]
        reason = log.get("selection_reason", "")
        parts.append(f"[{i}] {ts} [{prog}] {lvl} | {payload}")
        if reason:
            parts.append(f"     reason: {reason}")

    parts.append("\nTASK: Produce a JSON analysis with this exact schema:")
    parts.append("""{
  "summary": "1 phrase < 200 chars, factuelle",
  "primary_hypothesis": {
    "cause": "string decrivant la cause probable",
    "confidence": "low" | "medium" | "high",
    "evidence_log_indices": [int]
  },
  "alternative_hypotheses": [
    {
      "cause": "string",
      "confidence": "low" | "medium" | "high",
      "evidence_log_indices": [int]
    }
  ],
  "affected_services": ["service1", "service2"],
  "suggested_investigations": [
    "action concrete 1",
    "action concrete 2",
    "action concrete 3"
  ],
  "severity_assessment": "low" | "medium" | "high" | "critical",
  "requires_human_attention": true | false
}""")

    return "\n".join(parts)


def get_response_schema() -> dict:
    """Schema JSON Gemini-compatible pour forcer la structure de sortie."""
    return {
        "type": "OBJECT",
        "properties": {
            "summary": {"type": "STRING"},
            "primary_hypothesis": {
                "type": "OBJECT",
                "properties": {
                    "cause": {"type": "STRING"},
                    "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]},
                    "evidence_log_indices": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                    },
                },
                "required": ["cause", "confidence", "evidence_log_indices"],
            },
            "alternative_hypotheses": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "cause": {"type": "STRING"},
                        "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]},
                        "evidence_log_indices": {
                            "type": "ARRAY",
                            "items": {"type": "INTEGER"},
                        },
                    },
                    "required": ["cause", "confidence", "evidence_log_indices"],
                },
            },
            "affected_services": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "suggested_investigations": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "severity_assessment": {
                "type": "STRING",
                "enum": ["low", "medium", "high", "critical"],
            },
            "requires_human_attention": {"type": "BOOLEAN"},
        },
        "required": [
            "summary", "primary_hypothesis", "affected_services",
            "suggested_investigations", "severity_assessment",
            "requires_human_attention",
        ],
    }
