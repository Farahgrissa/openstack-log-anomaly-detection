"""Cache LRU sur disque pour les analyses LLM.

Économise tokens et coût en évitant de ré-analyser les mêmes alertes.
La clé de cache est l'alert_id (fingerprint) — donc deux alertes très
similaires partageront leur analyse.

Format : un fichier JSON par alert_id, dans data/llm_cache/
"""
import json
import time
from pathlib import Path
from typing import Optional

from ..utils.logging import logger


class LLMCache:
    """Cache disque simple, indexé par alert_id."""

    def __init__(self, cache_dir, ttl_seconds: int = 86400 * 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _path_for(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in key)[:64]
        return self.cache_dir / f"{safe}.json"

    def get(self, alert_id: str) -> Optional[dict]:
        path = self._path_for(alert_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        ts = entry.get("_cached_at", 0)
        if time.time() - ts > self.ttl_seconds:
            try:
                path.unlink()
            except OSError:
                pass
            return None
        return entry.get("analysis")

    def set(self, alert_id: str, analysis: dict) -> None:
        path = self._path_for(alert_id)
        entry = {
            "_cached_at": time.time(),
            "alert_id": alert_id,
            "analysis": analysis,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2, default=str)
        except OSError as e:
            logger.warning(f"Cache write failed: {e}")

    def clear(self) -> int:
        n = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
        return n
