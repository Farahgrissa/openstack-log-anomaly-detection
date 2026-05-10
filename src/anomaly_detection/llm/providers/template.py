"""Provider de fallback deterministe (sans LLM).

Utilise pour :
  - demos offline
  - si tous les providers LLM echouent
  - tests unitaires

Ne fait PAS d'appel reseau. Genere une analyse basee sur les
triggered_rules extraites du prompt.
"""
import json
import re
from ..base import LLMClient, LLMResponse


# Mapping regle -> cause probable + investigations
RULE_KNOWLEDGE = {
    "any_error_or_critical": {
        "cause": "Service produit des erreurs anormales (ERROR/CRITICAL logs)",
        "investigations": [
            "Examiner les payloads des logs ERROR pour identifier le service en panne",
            "Verifier la sante du backend concerne (DB, stockage, reseau)",
            "Consulter l'historique recent des deploiements",
        ],
    },
    "http_5xx_present": {
        "cause": "Erreurs serveur HTTP 5xx -- probleme cote backend OpenStack",
        "investigations": [
            "Identifier l'endpoint concerne dans les logs",
            "Verifier la disponibilite de la base de donnees (MariaDB)",
            "Verifier les logs de l'API qui retourne le 5xx",
        ],
    },
    "unknown_templates_present": {
        "cause": "Nouveaux types de messages jamais vus en training -- symptome potentiel d'un nouveau bug ou changement de configuration",
        "investigations": [
            "Examiner le contenu des logs au template inconnu",
            "Verifier si une mise a jour recente a ete deployee",
            "Comparer aux templates historiques du service concerne",
        ],
    },
    "service_silence": {
        "cause": "Un ou plusieurs services ne loggent plus alors qu'ils devraient -- service potentiellement crashe ou bloque",
        "investigations": [
            "Lister les services attendus vs services actifs",
            "Verifier l'etat des conteneurs (docker ps) ou systemd",
            "Examiner les derniers logs avant le silence",
        ],
    },
    "high_latency": {
        "cause": "Latences API anormalement elevees -- saturation backend ou contention",
        "investigations": [
            "Identifier le service le plus lent (latency_p95_<service>)",
            "Verifier la charge CPU/RAM/IO de la VM",
            "Examiner les requetes SQL lentes en base",
        ],
    },
    "http_4xx_burst": {
        "cause": "Burst de codes 4xx -- possible attaque, utilisateur mal configure, ou bug client",
        "investigations": [
            "Identifier les IP sources et endpoints vises",
            "Si 401/403 repetes : tentative de brute-force possible",
            "Si 404 : revoir le routage / config DNS",
        ],
    },
}


class TemplateClient(LLMClient):
    """Provider deterministe sans appel LLM."""

    provider_name = "template"

    def __init__(self, api_key: str = "", model: str = "rules-template", **kwargs):
        super().__init__(api_key=api_key or "n/a", model=model, **kwargs)

    def is_available(self) -> bool:
        return True

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint=None,
    ) -> LLMResponse:
        # Extraire les triggered_rules du user_prompt
        triggered = []
        m = re.search(r"Triggered rules:\s*\[(.*?)\]", user_prompt)
        if m:
            inner = m.group(1)
            triggered = [
                r.strip().strip("'").strip('"')
                for r in inner.split(",")
                if r.strip()
            ]

        # Construire l'analyse basee sur les regles
        primary_cause = "Anomalie detectee"
        confidence = "low"
        investigations = ["Examiner les top_logs pour identifier la cause exacte"]

        if triggered:
            first_rule = triggered[0]
            knowledge = RULE_KNOWLEDGE.get(first_rule, {})
            primary_cause = knowledge.get(
                "cause", f"Regle '{first_rule}' declenchee"
            )
            confidence = "medium"
            investigations = knowledge.get("investigations", investigations)

        # Mentionner les regles supplementaires
        all_causes = []
        for rule in triggered:
            k = RULE_KNOWLEDGE.get(rule, {})
            if k.get("cause") and k["cause"] != primary_cause:
                all_causes.append(k["cause"])

        analysis = {
            "summary": f"Anomalie detectee par {len(triggered)} regle(s) metier ; explication automatique sans LLM.",
            "primary_hypothesis": {
                "cause": primary_cause,
                "confidence": confidence,
                "evidence_log_indices": list(range(min(3, 10))),
            },
            "alternative_hypotheses": [
                {
                    "cause": c,
                    "confidence": "low",
                    "evidence_log_indices": [],
                }
                for c in all_causes[:2]
            ],
            "affected_services": [],
            "suggested_investigations": investigations,
            "severity_assessment": "medium",
            "requires_human_attention": True,
        }

        return LLMResponse(
            text=json.dumps(analysis, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            latency_ms=0.5,
        )
