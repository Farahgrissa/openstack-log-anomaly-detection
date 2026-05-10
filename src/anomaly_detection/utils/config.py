"""Configuration loader and accessor."""
from pathlib import Path
from typing import Any
import yaml
from functools import lru_cache


# Chemin par défaut de la config (relatif au root du projet)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = None) -> dict[str, Any]:
    """Charge la config YAML (cachée)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config() -> dict[str, Any]:
    """Récupère la config chargée."""
    return load_config()


def project_root() -> Path:
    """Racine du projet (3 niveaux au-dessus de ce fichier)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(rel_path: str) -> Path:
    """Résout un chemin relatif au root du projet."""
    return project_root() / rel_path


# Helpers métier dérivés de la config

@lru_cache(maxsize=1)
def get_service_to_group() -> dict[str, list[str]]:
    """Mapping inverse : programname -> [groupes]."""
    config = get_config()
    service_to_groups: dict[str, list[str]] = {}
    for group, services in config["service_groups"].items():
        for svc in services:
            service_to_groups.setdefault(svc, []).append(group)
    return service_to_groups


def get_groups_for_service(programname: str) -> list[str]:
    """Retourne les groupes d'un programname (peut être plusieurs ex: nova-api-uwsgi → compute + api_gateway)."""
    return get_service_to_group().get(programname, ["unknown"])


def get_severity(log_level: str | None) -> int:
    """Mappe un log_level texte vers un entier de sévérité."""
    if not log_level or log_level in ("NONE", "-"):
        return 1  # INFO par défaut
    config = get_config()
    return config["severity_map"].get(log_level, 1)


def get_all_groups() -> list[str]:
    """Liste de tous les groupes définis."""
    config = get_config()
    return list(config["service_groups"].keys())
