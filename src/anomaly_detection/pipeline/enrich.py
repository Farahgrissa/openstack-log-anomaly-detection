"""Pipeline d'enrichissement : NDJSON brut → DataFrame enrichi."""
import json
from pathlib import Path
from datetime import datetime
from typing import Iterator
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Fallback minimal si tqdm non installé
    def tqdm(iterable, **kwargs):
        return iterable

from ..utils.config import (
    get_groups_for_service, get_severity, resolve_path
)
from ..utils.logging import logger
from ..utils.io import save_df
from .parsers import (
    parse_uwsgi_payload, status_class, normalize_endpoint,
    detect_log_kind, template_payload
)


def parse_timestamp(ts: str) -> datetime | None:
    """Parse @timestamp ISO format → datetime UTC."""
    if not ts:
        return None
    try:
        # Format: "2026-04-27T08:42:00.000000000+00:00"
        # On enlève les nanos pour datetime (max microseconds = 6 digits)
        s = ts
        # Replace timezone separator : -> rien (Python <3.11 parfois capricieux)
        if "+" in s or s.count("-") > 2:
            # Garder les microsecs max
            if "." in s:
                base, rest = s.split(".", 1)
                # Trouver fin des chiffres décimaux
                i = 0
                while i < len(rest) and rest[i].isdigit():
                    i += 1
                # Microsecondes (max 6 digits)
                micros = rest[:i][:6].ljust(6, "0")
                tz = rest[i:]
                s = f"{base}.{micros}{tz}"
            return datetime.fromisoformat(s)
        return datetime.fromisoformat(s)
    except Exception:
        return None


def stream_ndjson(path: Path) -> Iterator[dict]:
    """Yield each line of a NDJSON file as dict."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def enrich_log(log: dict, template_dict: dict[str, int],
               freeze_templates: bool = False) -> dict:
    """Enrichit un log avec les métadonnées dérivées.

    Args:
        log : log brut JSON
        template_dict : dictionnaire de templates (modifié in-place sauf si frozen)
        freeze_templates : si True, n'ajoute PAS de nouveaux templates.
                           Les nouveaux messages auront template_id = -1.
    """
    payload = log.get("Payload", "") or ""
    programname = log.get("programname", "unknown")
    log_level = log.get("log_level")

    # Parsing uWSGI (si applicable)
    uwsgi_data = parse_uwsgi_payload(payload)
    has_uwsgi = uwsgi_data is not None

    # Endpoint normalisé
    if has_uwsgi:
        endpoint = normalize_endpoint(uwsgi_data["http_method"], uwsgi_data["http_path"])
        http_method = uwsgi_data["http_method"]
        http_status = uwsgi_data["http_status"]
        http_status_cls = status_class(http_status)
        http_duration_ms = uwsgi_data["http_duration_ms"]
        http_bytes = uwsgi_data["http_bytes"]
    else:
        endpoint = None
        http_method = None
        http_status = None
        http_status_cls = "none"
        http_duration_ms = None
        http_bytes = None

    # Templating - avec gestion frozen
    template = template_payload(payload)
    if template in template_dict:
        template_id = template_dict[template]
    elif freeze_templates:
        # Mode frozen : on n'ajoute pas, et on signale avec id=-1
        template_id = -1
    else:
        template_id = len(template_dict)
        template_dict[template] = template_id

    # Severity
    severity = get_severity(log_level)

    # Log kind
    log_kind = detect_log_kind(programname, payload, has_uwsgi)

    # Service groups (multi-groupes)
    groups = get_groups_for_service(programname)

    # Timestamp
    ts_raw = log.get("@timestamp", "")
    ts_parsed = parse_timestamp(ts_raw)

    # request_id et user_id (utiles pour la corrélation)
    request_id = log.get("request_id") or log.get("global_request_id") or None
    if request_id == "-":
        request_id = None
    user_id = log.get("user_id")
    if user_id == "-" or user_id == "":
        user_id = None
    tenant_id = log.get("tenant_id")
    if tenant_id == "-" or tenant_id == "":
        tenant_id = None

    return {
        "timestamp": ts_parsed,
        "programname": programname,
        "log_level": log_level,
        "severity": severity,
        "log_kind": log_kind,
        "groups": groups,
        # HTTP
        "http_method": http_method,
        "http_status": http_status,
        "http_status_class": http_status_cls,
        "http_duration_ms": http_duration_ms,
        "http_bytes": http_bytes,
        "endpoint_template": endpoint,
        # Templating
        "payload_template_id": template_id,
        "payload_template": template,
        # Identité
        "request_id": request_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        # Original (pour debug + sélection top K logs en Phase 5)
        "python_module": log.get("python_module"),
        "payload": payload,
    }


def enrich_file(input_path: Path, output_path: Path,
                template_dict: dict[str, int] | None = None,
                freeze_templates: bool = False) -> dict:
    """Enrichit un fichier NDJSON et sauvegarde le résultat.

    Args:
        input_path : NDJSON source
        output_path : sortie parquet/pickle
        template_dict : dictionnaire de templates initial (peut être pré-rempli)
        freeze_templates : si True, n'ajoute PAS de nouveaux templates au dict.
                           Les nouveaux messages auront template_id = -1
                           (signal d'alerte précieux pour la détection !).

    Returns : dict de stats.
    """
    if template_dict is None:
        template_dict = {}

    logger.info(f"Enriching: {input_path}")
    if freeze_templates:
        logger.info(f"Templates FROZEN at {len(template_dict)} entries (new templates → id=-1)")

    enriched_rows = []
    n_total = 0
    n_dropped = 0
    n_unknown_templates = 0  # compte de logs avec template inconnu (mode frozen)

    pbar = tqdm(stream_ndjson(input_path), unit=" logs", desc="parsing")
    for log in pbar:
        n_total += 1
        try:
            enriched = enrich_log(log, template_dict, freeze_templates=freeze_templates)
            if enriched.get("payload_template_id") == -1:
                n_unknown_templates += 1
        except Exception as e:
            n_dropped += 1
            if n_dropped <= 3:
                logger.warning(f"Drop log {n_total}: {e}")
            continue
        enriched_rows.append(enriched)

    logger.info(f"Parsed {n_total} logs, dropped {n_dropped}")
    if freeze_templates:
        logger.info(f"Logs with UNKNOWN template (id=-1): {n_unknown_templates}")

    df = pd.DataFrame(enriched_rows)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    saved = save_df(df, output_path)
    logger.info(f"Saved enriched DataFrame: {saved}")

    stats = {
        "n_logs": len(df),
        "n_dropped": n_dropped,
        "n_unique_services": df["programname"].nunique(),
        "n_unique_templates": len(template_dict),
        "n_unknown_templates": n_unknown_templates,
        "dates": sorted(df["date"].dropna().unique().tolist()),
        "log_kinds": df["log_kind"].value_counts().to_dict(),
        "severity_dist": df["severity"].value_counts().sort_index().to_dict(),
        "output_path": str(saved),
    }
    return stats


def save_template_dict(template_dict: dict[str, int], path: Path) -> None:
    """Sauvegarde le dictionnaire des templates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        # Format : id\ttemplate
        for tpl, tid in sorted(template_dict.items(), key=lambda x: x[1]):
            f.write(f"{tid}\t{tpl}\n")
    logger.info(f"Saved {len(template_dict)} templates to {path}")


def load_template_dict(path: Path) -> dict[str, int]:
    """Charge un dictionnaire de templates sauvegardé."""
    template_dict: dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                tid_str, tpl = parts
                template_dict[tpl] = int(tid_str)
    return template_dict
