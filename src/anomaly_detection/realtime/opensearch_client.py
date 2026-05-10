"""Client HTTP minimaliste pour OpenSearch (urllib stdlib uniquement).
"""
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from ..utils.logging import logger


# Exceptions

class OpenSearchError(Exception):
    """Erreur générique de communication avec OpenSearch."""
    pass


# Client

class OpenSearchClient:
    """Client OpenSearch minimal, basé sur urllib stdlib.

    Usage :
        client = OpenSearchClient(
            host="192.168.91.128", port=9200,
            username="admin", password="admin",
        )
        if client.ping():
            for log in client.fetch_logs_between(start_ts, end_ts):
                process(log)
    """

    def __init__(
        self,
        host: str,
        port: int = 9200,
        scheme: str = "http",
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_pattern: str = "flog-*",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        verify_certs: bool = False,
        page_size: int = 1000,
    ):
        self.host = host
        self.port = port
        self.scheme = scheme
        self.username = username
        self.password = password
        self.index_pattern = index_pattern
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.verify_certs = verify_certs
        self.page_size = page_size

        self.base_url = f"{scheme}://{host}:{port}"
        self._auth_header: Optional[str] = None
        if username and password:
            token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
            self._auth_header = f"Basic {token}"

        # Pour HTTPS auto-signé : optionnellement désactiver la vérification
        # (volontairement isolé dans un opener custom au lieu de modifier le contexte global)
        if scheme == "https" and not verify_certs:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            self._ssl_context = ctx
        else:
            self._ssl_context = None

    # HTTP layer (avec retry exponentiel)

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Effectue une requête HTTP avec retry exponentiel.

        Raises :
            OpenSearchError si toutes les tentatives échouent.
        """
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        data = None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        if self._auth_header:
            headers["Authorization"] = self._auth_header

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                if self._ssl_context is not None:
                    resp = urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_context)
                else:
                    resp = urllib.request.urlopen(req, timeout=self.timeout)
                with resp:
                    raw = resp.read()
                    if not raw:
                        return {}
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as e:
                # 4xx : erreur côté client - inutile de retry
                if 400 <= e.code < 500:
                    body_err = ""
                    try:
                        body_err = e.read().decode("utf-8", errors="replace")[:500]
                    except Exception:
                        pass
                    raise OpenSearchError(
                        f"HTTP {e.code} on {method} {path} : {e.reason} | {body_err}"
                    ) from e
                last_error = e
                logger.warning(f"OpenSearch HTTP {e.code} on attempt {attempt + 1}/{self.max_retries}: {e.reason}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = e
                logger.warning(f"OpenSearch network error on attempt {attempt + 1}/{self.max_retries}: {e}")
            except Exception as e:
                last_error = e
                logger.warning(f"OpenSearch unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")

            # Backoff exponentiel avant la prochaine tentative (sauf si dernière)
            if attempt < self.max_retries - 1:
                sleep_s = 2 ** attempt
                logger.info(f"Retry in {sleep_s}s...")
                time.sleep(sleep_s)

        raise OpenSearchError(
            f"All {self.max_retries} attempts failed for {method} {path}: {last_error}"
        )

    # API publique : diagnostic

    def ping(self) -> bool:
        """Test de connectivité. Retourne True si le cluster répond."""
        try:
            resp = self._request("GET", "/")
            cluster_name = resp.get("cluster_name", "?")
            version = resp.get("version", {}).get("number", "?")
            logger.info(f"OpenSearch OK — cluster={cluster_name} version={version}")
            return True
        except OpenSearchError as e:
            logger.error(f"OpenSearch ping failed : {e}")
            return False

    def list_indices(self, pattern: Optional[str] = None) -> list[str]:
        """Liste les indices correspondant au pattern (utile pour debug)."""
        pat = pattern or self.index_pattern
        try:
            resp = self._request("GET", f"/_cat/indices/{urllib.parse.quote(pat)}",
                                 params={"format": "json", "h": "index"})
            if isinstance(resp, list):
                return sorted([row["index"] for row in resp if "index" in row])
            return []
        except OpenSearchError as e:
            logger.error(f"list_indices failed : {e}")
            return []

    # API publique : recherche

    def search(self, query: dict, size: int = 1000, sort: Optional[list] = None,
               search_after: Optional[list] = None) -> dict:
        """Search brute. Retourne le payload OpenSearch complet."""
        body: dict[str, Any] = {"size": size, "query": query}
        if sort is not None:
            body["sort"] = sort
        if search_after is not None:
            body["search_after"] = search_after
        path = f"/{urllib.parse.quote(self.index_pattern)}/_search"
        return self._request("POST", path, body=body)

    def fetch_logs_between(
        self,
        start_ts: datetime,
        end_ts: datetime,
        max_logs: int = 50_000,
        timestamp_field: str = "@timestamp",
    ) -> Iterator[dict]:
        """Itère sur les logs dont le timestamp est dans [start_ts, end_ts).

        Pagination via search_after (tiebreaker _id) pour rester stable même
        si plusieurs logs partagent le même timestamp.

        Args :
            start_ts : datetime tz-aware (inclus)
            end_ts   : datetime tz-aware (exclus)
            max_logs : cap dur (anti-DDoS si OpenSearch en avalanche)
            timestamp_field : nom du champ timestamp dans les docs (par défaut Fluentd)

        Yields :
            dict log_source enrichi de la clé `_os_id` (= _id OpenSearch).
        """
        # Forcer UTC ISO strict
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=timezone.utc)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=timezone.utc)

        query = {
            "range": {
                timestamp_field: {
                    "gte": start_ts.isoformat(),
                    "lt": end_ts.isoformat(),
                }
            }
        }
        # Tri sur timestamp puis _id (tiebreaker) pour garantir search_after stable
        sort = [{timestamp_field: "asc"}, {"_id": "asc"}]

        n_yielded = 0
        last_sort: Optional[list] = None
        page = 0
        while True:
            try:
                payload = self.search(query=query, size=self.page_size,
                                      sort=sort, search_after=last_sort)
            except OpenSearchError as e:
                logger.error(f"fetch_logs_between : {e}")
                return

            hits = payload.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                src = hit.get("_source") or {}
                # Empreinte d'unicité côté worker
                src["_os_id"] = hit.get("_id", "")
                yield src
                n_yielded += 1
                if n_yielded >= max_logs:
                    logger.warning(
                        f"fetch_logs_between : cap {max_logs} atteint, troncature "
                        f"(window {start_ts.isoformat()} → {end_ts.isoformat()})"
                    )
                    return

            # Préparer la page suivante
            last_sort = hits[-1].get("sort")
            page += 1
            if last_sort is None:
                break
            # Heuristique : si la page n'est pas pleine, c'est terminé
            if len(hits) < self.page_size:
                break

        logger.debug(
            f"fetch_logs_between : {n_yielded} logs sur "
            f"{(end_ts - start_ts).total_seconds():.0f}s "
            f"({page + 1} pages)"
        )
