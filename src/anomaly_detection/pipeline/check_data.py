"""Vérifie que les données brutes sont présentes et bien formées."""
import json
from pathlib import Path
from collections import Counter

from ..utils.config import resolve_path
from ..utils.logging import logger


def run_check() -> dict:
    """Vérifie l'intégrité du dataset dans data/raw/."""
    raw_dir = resolve_path("data/raw")
    logger.info(f"Checking raw data in: {raw_dir}")

    if not raw_dir.exists():
        logger.error(f"Raw data directory does not exist: {raw_dir}")
        raise FileNotFoundError(raw_dir)

    # Cherche tous les fichiers .json (NDJSON)
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        logger.error(f"No .json files found in {raw_dir}")
        raise FileNotFoundError("No raw data files")

    logger.info(f"Found {len(files)} file(s)")

    summary = {"files": [], "total_lines": 0, "dates": Counter(), "errors": 0}

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  - {f.name}: {size_mb:.1f} MB")

        # Lecture rapide pour compter et extraire les dates
        n_lines = 0
        n_errors = 0
        local_dates = Counter()
        with open(f, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    obj = json.loads(line)
                    ts = obj.get("@timestamp", "")
                    if ts and len(ts) >= 10:
                        local_dates[ts[:10]] += 1
                except json.JSONDecodeError:
                    n_errors += 1

        summary["files"].append({
            "name": f.name,
            "size_mb": round(size_mb, 1),
            "lines": n_lines,
            "errors": n_errors,
            "dates": dict(local_dates),
        })
        summary["total_lines"] += n_lines
        summary["errors"] += n_errors
        for d, c in local_dates.items():
            summary["dates"][d] += c

    logger.info(f"=== Summary ===")
    logger.info(f"Total lines: {summary['total_lines']:,}")
    logger.info(f"Malformed lines: {summary['errors']}")
    logger.info(f"Date range:")
    for d, c in sorted(summary["dates"].items()):
        logger.info(f"  {d}: {c:,} lines")

    return summary
