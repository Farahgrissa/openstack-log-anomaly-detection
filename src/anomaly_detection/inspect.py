"""Inspection rapide du DataFrame enrichi - utile en debug."""
import pandas as pd
from pathlib import Path

from .utils.config import resolve_path
from .utils.logging import logger
from .utils.io import load_df


def inspect(input_file: str = "data/processed/enriched") -> dict:
    """Affiche un rapport résumé du DataFrame enrichi."""
    path = resolve_path(input_file)
    logger.info(f"Loading {path}")
    df = load_df(path)

    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Volumes par jour
    logger.info("=== Volumes par jour ===")
    for date, count in df.groupby("date").size().items():
        logger.info(f"  {date}: {count:,}")

    # Services
    logger.info("=== Top 10 services ===")
    for prog, count in df["programname"].value_counts().head(10).items():
        logger.info(f"  {prog}: {count:,}")

    # log_kinds
    logger.info("=== Log kinds ===")
    for kind, count in df["log_kind"].value_counts().items():
        logger.info(f"  {kind}: {count:,}")

    # Severity
    logger.info("=== Severity ===")
    for sev, count in df["severity"].value_counts().sort_index().items():
        logger.info(f"  {sev}: {count:,}")

    # HTTP stats
    http = df[df["log_kind"] == "http_access"]
    logger.info(f"=== HTTP ({len(http):,} logs) ===")
    logger.info(f"  Status classes: {http['http_status_class'].value_counts().to_dict()}")
    logger.info(f"  Methods: {http['http_method'].value_counts().to_dict()}")
    logger.info(f"  Latency P50/P95/P99/Max: "
                f"{http['http_duration_ms'].quantile(0.5):.0f}/"
                f"{http['http_duration_ms'].quantile(0.95):.0f}/"
                f"{http['http_duration_ms'].quantile(0.99):.0f}/"
                f"{http['http_duration_ms'].max():.0f} ms")

    # Templates
    n_templates = df["payload_template_id"].nunique()
    logger.info(f"=== Templates: {n_templates} unique ===")

    # NaN check
    logger.info("=== Null check ===")
    for col in ["timestamp", "programname", "groups", "payload_template_id", "log_kind"]:
        n = df[col].isna().sum()
        if n > 0:
            logger.warning(f"  {col}: {n} NaN ⚠️")
        else:
            logger.info(f"  {col}: OK")

    return {
        "shape": df.shape,
        "n_services": df["programname"].nunique(),
        "n_templates": n_templates,
        "dates": sorted(df["date"].unique().tolist()),
    }
