"""Validation des reponses LLM.

Effectue :
  1. Parse JSON robuste (gere les markdown fences si LLM les ajoute)
  2. Validation du schema (champs obligatoires, types)
  3. Verification du grounding : les indices cites existent dans top_logs
  4. Construction de l'objet LLMAnalysis
"""
import json
import re
from typing import Optional, Tuple

from .schema import LLMAnalysis, LLMHypothesis
from ..utils.logging import logger


def parse_json_robust(text: str) -> Optional[dict]:
    """Parse une reponse LLM en JSON, en tolerant les markdown fences."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def validate_grounding(analysis_dict: dict, n_top_logs: int) -> Tuple[bool, list]:
    """Verifie que les indices cites existent."""
    warnings = []

    def check_indices(indices, where):
        for idx in indices:
            if not isinstance(idx, int):
                warnings.append(f"{where}: non-int index {idx!r}")
                continue
            if idx < 0 or idx >= n_top_logs:
                warnings.append(f"{where}: index {idx} out of range [0..{n_top_logs - 1}]")

    primary = analysis_dict.get("primary_hypothesis", {})
    check_indices(primary.get("evidence_log_indices", []), "primary_hypothesis")
    for i, alt in enumerate(analysis_dict.get("alternative_hypotheses", [])):
        check_indices(alt.get("evidence_log_indices", []), f"alternative_hypotheses[{i}]")

    is_valid = len(warnings) == 0
    return is_valid, warnings


def build_analysis(
    response_text: str,
    n_top_logs: int,
    provider: str,
    model: str,
    latency_ms: float,
    tokens_used: Optional[int] = None,
    fallback_used: bool = False,
) -> LLMAnalysis:
    """Parse + valide + construit LLMAnalysis."""
    parsed = parse_json_robust(response_text)
    if parsed is None:
        logger.warning(f"[{provider}] JSON parse failed, returning empty analysis")
        return _empty_analysis(provider, model, latency_ms, tokens_used,
                               fallback_used,
                               error_summary="JSON parse failed")

    is_valid, warnings = validate_grounding(parsed, n_top_logs)
    if warnings:
        for w in warnings:
            logger.warning(f"[{provider}] grounding: {w}")

    primary = parsed.get("primary_hypothesis", {})
    primary_indices = [
        i for i in primary.get("evidence_log_indices", [])
        if isinstance(i, int) and 0 <= i < n_top_logs
    ]

    primary_hyp = LLMHypothesis(
        cause=primary.get("cause", "Cause non specifiee"),
        confidence=primary.get("confidence", "low"),
        evidence_log_indices=primary_indices,
    )

    alternatives = []
    for alt in parsed.get("alternative_hypotheses", [])[:3]:
        alt_indices = [
            i for i in alt.get("evidence_log_indices", [])
            if isinstance(i, int) and 0 <= i < n_top_logs
        ]
        alternatives.append(LLMHypothesis(
            cause=alt.get("cause", ""),
            confidence=alt.get("confidence", "low"),
            evidence_log_indices=alt_indices,
        ))

    return LLMAnalysis(
        summary=str(parsed.get("summary", ""))[:300],
        primary_hypothesis=primary_hyp,
        alternative_hypotheses=alternatives,
        affected_services=[str(s) for s in parsed.get("affected_services", [])][:10],
        suggested_investigations=[
            str(s) for s in parsed.get("suggested_investigations", [])
        ][:6],
        severity_assessment=parsed.get("severity_assessment", "medium"),
        requires_human_attention=bool(parsed.get("requires_human_attention", True)),
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        fallback_used=fallback_used,
        grounding_valid=is_valid,
    )


def _empty_analysis(provider, model, latency_ms, tokens, fallback,
                    error_summary):
    return LLMAnalysis(
        summary=f"Analyse LLM non disponible ({error_summary})",
        primary_hypothesis=LLMHypothesis(
            cause="Voir les top_features et top_logs de l'alerte",
            confidence="low",
        ),
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        tokens_used=tokens,
        fallback_used=fallback,
        grounding_valid=False,
    )
