"""Extraction des champs structurés depuis le Payload des logs OpenStack."""
import re
from typing import Optional


# Regex pré-compilées

# uWSGI access log pattern :
# "GET /healthcheck => generated 2 bytes in 36 msecs (HTTP/1.0 200) 3 headers in 98 bytes ..."
RE_UWSGI = re.compile(
    r'^(?P<method>\w+)\s+'
    r'(?P<path>\S+)\s+=>\s+generated\s+'
    r'(?P<bytes>\d+)\s+bytes\s+in\s+'
    r'(?P<duration>\d+)\s+msecs\s+'
    r'\(HTTP/(?P<http_ver>[\d.]+)\s+(?P<status>\d+)\)'
)

# UUID v4 pattern
RE_UUID = re.compile(
    r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
    re.IGNORECASE
)

# IP address
RE_IP = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

# Numeric IDs in paths (/v2.1/servers/12345)
RE_NUM_IN_PATH = re.compile(r'/\d+(?=[/?]|$)')

# Query strings
RE_QUERY = re.compile(r'\?[^"\s]*')

# Timestamps embedded in messages (multiple formats)
RE_TS_ISO = re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[\.,]\d+)?(?:[+-]\d{2}:?\d{2}|Z)?')
RE_TS_SIMPLE = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}')

# Hex strings (hashes, request_ids partiels)
RE_HEX_LONG = re.compile(r'\b[0-9a-f]{16,}\b', re.IGNORECASE)

# Floating point numbers
RE_FLOAT = re.compile(r'\b\d+\.\d+\b')

# Integer numbers (mais pas les codes HTTP courants)
RE_INT = re.compile(r'\b\d+\b')


# Fonctions d'extraction

def parse_uwsgi_payload(payload: str) -> Optional[dict]:
    """Parse un payload uWSGI access log.

    Retourne dict avec http_method, path, status, bytes, duration_ms, http_version
    ou None si le payload ne match pas.
    """
    m = RE_UWSGI.match(payload)
    if not m:
        return None
    return {
        "http_method": m.group("method"),
        "http_path": m.group("path"),
        "http_status": int(m.group("status")),
        "http_bytes": int(m.group("bytes")),
        "http_duration_ms": int(m.group("duration")),
        "http_version": m.group("http_ver"),
    }


def status_class(status: int | None) -> str:
    """Retourne la classe HTTP (2xx, 3xx, 4xx, 5xx)."""
    if status is None:
        return "none"
    if 200 <= status < 300:
        return "2xx"
    if 300 <= status < 400:
        return "3xx"
    if 400 <= status < 500:
        return "4xx"
    if 500 <= status < 600:
        return "5xx"
    return "other"


def normalize_endpoint(method: str, path: str) -> str:
    """Normalise un endpoint pour grouper les requêtes similaires.

    Ex: 'GET /v2.1/servers/abc-def-...' -> 'GET /v2.1/servers/<UUID>'
        'PUT /allocations/123' -> 'PUT /allocations/<ID>'
        'POST /v3/auth/tokens?...' -> 'POST /v3/auth/tokens'
    """
    # Retirer query string
    p = RE_QUERY.sub("", path)
    # UUIDs -> <UUID>
    p = RE_UUID.sub("<UUID>", p)
    # IDs numériques dans path -> <ID>
    p = RE_NUM_IN_PATH.sub("/<ID>", p)
    return f"{method} {p}"


def detect_log_kind(programname: str, payload: str, has_uwsgi_match: bool) -> str:
    """Catégorise le type de log.

    Returns : http_access | app_log | infra_log | agent_log | other
    """
    if has_uwsgi_match or programname.endswith("-uwsgi"):
        return "http_access"
    if programname in ("rabbit", "mariadb", "fluent",
                       "openvswitch", "openvswitchdb", "libvirt"):
        return "infra_log"
    if "agent" in programname or programname == "neutron-openvswitch-agent":
        return "agent_log"
    return "app_log"


# Templating de Payload

def template_payload(payload: str) -> str:
    """Normalise un payload en template en remplaçant les valeurs variables.

    L'idée : 2 logs avec des UUIDs/timestamps différents mais même structure
    doivent avoir le même template.

    Exemple :
      "POST /v2.1/servers/abc-123 => generated 380 bytes in 1247 msecs (HTTP/1.1 202)"
      -> "POST /v2.1/servers/<UUID> => generated <N> bytes in <N> msecs (HTTP/<V> <N>)"
    """
    if not payload:
        return ""

    s = payload

    # 1. Timestamps complets (avant tout, les patterns numériques peuvent matcher dedans)
    s = RE_TS_ISO.sub("<TS>", s)
    s = RE_TS_SIMPLE.sub("<TS>", s)

    # 2. UUIDs
    s = RE_UUID.sub("<UUID>", s)

    # 3. IPs
    s = RE_IP.sub("<IP>", s)

    # 4. HTTP version (HTTP/1.1 → HTTP/<V>)
    s = re.sub(r'HTTP/\d+\.\d+', 'HTTP/<V>', s)

    # 5. Hex longs
    s = RE_HEX_LONG.sub("<HEX>", s)

    # 6. Floats
    s = RE_FLOAT.sub("<F>", s)

    # 7. Integers (en dernier, pour éviter de dégrader les patterns au-dessus)
    s = RE_INT.sub("<N>", s)

    # 8. Normaliser les espaces multiples
    s = re.sub(r'\s+', ' ', s).strip()

    # 9. Tronquer pour éviter explosion mémoire (rare mais possible)
    if len(s) > 500:
        s = s[:500] + "..."

    return s


def template_id_from_template(template: str, template_dict: dict[str, int]) -> int:
    """Retourne un ID stable pour un template texte.

    template_dict est mis à jour (modification in-place) avec les nouveaux templates.
    """
    if template in template_dict:
        return template_dict[template]
    new_id = len(template_dict)
    template_dict[template] = new_id
    return new_id
