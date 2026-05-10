"""CLI du système de détection d'anomalies OpenStack."""
import click
from pathlib import Path
import numpy as np

from .utils.config import get_config, resolve_path
from .utils.logging import setup_logging, logger


@click.group()
@click.option("--log-level", default=None, help="Override log level")
@click.pass_context
def cli(ctx, log_level):
    """OpenStack Anomaly Detection System CLI."""
    setup_logging(level=log_level)
    ctx.ensure_object(dict)


@cli.command()
def info():
    """Affiche les infos du projet et la config."""
    config = get_config()
    logger.info("=== Anomaly Detection System ===")
    logger.info(f"Project root: {resolve_path('.')}")
    logger.info(f"Service groups: {list(config['service_groups'].keys())}")
    logger.info(f"Window size: {config['windowing']['primary']['window_size_seconds']}s")
    logger.info(f"Window step: {config['windowing']['primary']['step_seconds']}s")
    logger.info(f"Train dates: {config['data_split']['train_dates']}")


@cli.command()
def check_data():
    """Vérifie que les données brutes sont présentes et lisibles."""
    from .pipeline.check_data import run_check
    run_check()


@cli.command()
@click.option("--input", "-i", "input_file", default="data/raw/dataset.json",
              help="Fichier NDJSON d'entrée")
@click.option("--output", "-o", "output_file", default="data/processed/enriched",
              help="Fichier de sortie (sans extension)")
@click.option("--templates", "-t", "templates_file", default=None,
              help="Fichier templates.tsv à charger")
@click.option("--freeze-templates", is_flag=True,
              help="Figer le dictionnaire de templates (les inconnus auront id=-1)")
@click.option("--save-templates/--no-save-templates", default=True,
              help="Sauver le dict de templates en fin (défaut: oui)")
def enrich(input_file: str, output_file: str, templates_file: str,
           freeze_templates: bool, save_templates: bool):
    """Phase 1 : enrichit les logs bruts (parsing, templating, métadonnées).

    Modes :
      - Training : enrich --input training.json (crée templates.tsv)
      - Inference : enrich --input test.json --templates templates.tsv --freeze-templates
    """
    from .pipeline.enrich import enrich_file, save_template_dict, load_template_dict

    in_path = resolve_path(input_file)
    out_path = resolve_path(output_file)

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise SystemExit(1)

    if templates_file:
        tpl_path = resolve_path(templates_file)
        if tpl_path.exists():
            template_dict = load_template_dict(tpl_path)
            logger.info(f"Loaded {len(template_dict)} existing templates from {tpl_path}")
        else:
            logger.warning(f"Templates file not found: {tpl_path}, starting fresh")
            template_dict = {}
    else:
        template_dict = {}

    initial_count = len(template_dict)
    stats = enrich_file(in_path, out_path, template_dict, freeze_templates=freeze_templates)

    if save_templates and not freeze_templates:
        tpl_save_path = resolve_path("data/processed/templates.tsv")
        save_template_dict(template_dict, tpl_save_path)

    logger.info("=== Enrichment Summary ===")
    logger.info(f"  Logs processed: {stats['n_logs']:,} (dropped: {stats['n_dropped']})")
    logger.info(f"  Unique services: {stats['n_unique_services']}")
    if freeze_templates:
        logger.info(f"  Templates: {len(template_dict)} (frozen, started at {initial_count})")
        logger.info(f"  Logs with UNKNOWN template: {stats['n_unknown_templates']:,}")
    else:
        logger.info(f"  Templates: {len(template_dict)} (was {initial_count})")
    logger.info(f"  Dates covered: {stats['dates']}")
    logger.info(f"  Severity distribution: {stats['severity_dist']}")
    logger.info(f"  Output: {stats['output_path']}")


@cli.command()
@click.option("--input", "-i", "input_file", default="data/processed/enriched",
              help="Fichier enrichi à inspecter")
def inspect(input_file: str):
    """Inspecte un DataFrame enrichi (stats détaillées)."""
    from .inspect import inspect as run_inspect
    run_inspect(input_file)


@cli.command()
@click.option("--input", "-i", "input_file", default="data/processed/enriched",
              help="Fichier enrichi (Phase 1)")
@click.option("--output", "-o", "output_file", default="data/features/windows_5min",
              help="Fichier de sortie features par fenêtre")
@click.option("--window", default=None, type=int,
              help="Taille fenêtre en secondes (défaut: config)")
@click.option("--step", default=None, type=int,
              help="Step en secondes (défaut: config)")
def features(input_file: str, output_file: str, window: int, step: int):
    """Phase 2 : agrégation en fenêtres + calcul de ~60 features."""
    from .features.aggregate import aggregate_features

    config = get_config()
    if window is None:
        window = config["windowing"]["primary"]["window_size_seconds"]
    if step is None:
        step = config["windowing"]["primary"]["step_seconds"]

    in_path = resolve_path(input_file)
    out_path = resolve_path(output_file)

    if not in_path.exists():
        for ext in (".pkl.gz", ".parquet"):
            if Path(str(in_path) + ext).exists():
                in_path = Path(str(in_path) + ext)
                break
        else:
            logger.error(f"Input file not found: {in_path}")
            raise SystemExit(1)

    stats = aggregate_features(in_path, out_path, window, step)

    logger.info("=== Features Aggregation Summary ===")
    logger.info(f"  Windows: {stats['n_windows']}")
    logger.info(f"  Features per window: {stats['n_features']}")
    logger.info(f"  Window size: {stats['window_seconds']}s, step: {stats['step_seconds']}s")
    logger.info(f"  Output: {stats['output_path']}")


@cli.command(name="inspect-features")
@click.option("--input", "-i", "input_file", default="data/features/windows_5min",
              help="Fichier features à inspecter")
def inspect_features_cmd(input_file: str):
    """Inspecte un DataFrame de features (Phase 2)."""
    from .inspect_features import inspect_features
    inspect_features(input_file)


@cli.command()
@click.option("--input", "-i", "input_file", default="data/features/windows_5min",
              help="Fichier features (Phase 2)")
@click.option("--output", "-o", "output_dir", default="data/models",
              help="Dossier où sauver les modèles")
@click.option("--no-filter-active", is_flag=True,
              help="Ne pas filtrer les fenêtres actives (par défaut: filtré)")
def train(input_file: str, output_dir: str, no_filter_active: bool):
    """Phase 4 : entraîne le modèle global + N modèles par groupe."""
    from .models.train import train_all

    in_path = resolve_path(input_file)
    out_path = resolve_path(output_dir)

    if not in_path.exists():
        for ext in (".pkl.gz", ".parquet"):
            if Path(str(in_path) + ext).exists():
                in_path = Path(str(in_path) + ext)
                break
        else:
            logger.error(f"Input file not found: {in_path}")
            raise SystemExit(1)

    stats = train_all(in_path, out_path, filter_active=not no_filter_active)

    logger.info("=== Training Summary ===")
    logger.info(f"  Training samples: {stats['n_train_samples']}")
    logger.info(f"  Features (global model): {stats['n_features_global']}")
    logger.info(f"  Global threshold: {stats['global_threshold']:.4f}")
    logger.info(f"  Group models: {stats['n_group_models']} ({stats['groups_trained']})")
    logger.info(f"  Models saved in: {out_path}")


@cli.command()
@click.option("--features", "test_features", default="data/test_anomalies/test_features",
              help="Test features file")
@click.option("--labels", "test_labels", default="data/test_anomalies/labels.csv",
              help="Test labels file (CSV)")
@click.option("--models", "models_dir", default="data/models",
              help="Models directory")
@click.option("--output", "output_file", default="data/models/eval_results.json",
              help="Output JSON results file")
def evaluate(test_features: str, test_labels: str, models_dir: str, output_file: str):
    """Phase 4 : évalue les modèles sur le test set anormal."""
    from .models.evaluate import evaluate_all
    import json

    feat_path = resolve_path(test_features)
    labels_path = resolve_path(test_labels)
    models_path = resolve_path(models_dir)
    out_path = resolve_path(output_file)

    if not feat_path.exists():
        for ext in (".pkl.gz", ".parquet"):
            if Path(str(feat_path) + ext).exists():
                feat_path = Path(str(feat_path) + ext)
                break

    results = evaluate_all(models_path, feat_path, labels_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(x) for x in obj]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    logger.info(f"Results saved to: {out_path}")


@cli.command()
@click.option("--features", "test_features", default="data/test_anomalies/test_features",
              help="Features file (Phase 2 output)")
@click.option("--logs", "enriched_logs", default="data/test_anomalies/test_enriched",
              help="Enriched logs file (Phase 1 output)")
@click.option("--models", "models_dir", default="data/models",
              help="Models directory")
@click.option("--output", "output_file", default="data/alerts/alerts.json",
              help="Output JSON file with alerts")
@click.option("--ml-threshold", default=0.90, type=float,
              help="ML probability threshold")
@click.option("--rules-threshold", default=0.5, type=float,
              help="Rules score threshold")
def detect(test_features: str, enriched_logs: str, models_dir: str,
           output_file: str, ml_threshold: float, rules_threshold: float):
    """Phase 5 : détection + explication + génération d'alertes structurées."""
    from .alerting.detect import detect_from_files
    from collections import Counter

    feat_path = resolve_path(test_features)
    logs_path = resolve_path(enriched_logs)
    models_path = resolve_path(models_dir)
    out_path = resolve_path(output_file)

    for p, name in [(feat_path, "features"), (logs_path, "enriched logs")]:
        if not p.exists():
            for ext in (".pkl.gz", ".parquet"):
                if Path(str(p) + ext).exists():
                    if name == "features":
                        feat_path = Path(str(p) + ext)
                    else:
                        logs_path = Path(str(p) + ext)
                    break

    alerts, incidents = detect_from_files(
        feat_path, logs_path, models_path, out_path,
        ml_threshold=ml_threshold, rules_threshold=rules_threshold,
    )

    sev_counts = Counter(a.severity for a in alerts)
    logger.info("=== Detection Summary ===")
    logger.info(f"  Total alerts (windows): {len(alerts)}")
    for sev in ["critical", "high", "medium", "low"]:
        logger.info(f"    {sev}: {sev_counts.get(sev, 0)}")
    logger.info(f"  Total incidents (deduped): {len(incidents)}")
    inc_sev_counts = Counter(i.get("max_severity", "unknown") for i in incidents)
    for sev in ["critical", "high", "medium", "low"]:
        logger.info(f"    {sev}: {inc_sev_counts.get(sev, 0)}")
    logger.info(f"  Output: {out_path}")


@cli.command()
@click.option("--input", "input_file", default="data/alerts/alerts.json",
              help="Fichier d'alertes JSON à analyser")
@click.option("--output", "output_file", default="data/alerts/alerts_enriched.json",
              help="Fichier de sortie avec analyses LLM")
@click.option("--max-alerts", default=None, type=int,
              help="Limiter le nombre d'alertes (debug)")
@click.option("--severity-min", default=None,
              help="Analyser seulement les alertes >= cette sévérité (low|medium|high|critical)")
def analyze(input_file: str, output_file: str, max_alerts: int, severity_min: str):
    """Phase 6 : analyse les alertes avec LLM (Gemini/Groq + fallback template)."""
    from .llm import LLMAnalyzer
    import json
    from collections import Counter

    in_path = resolve_path(input_file)
    out_path = resolve_path(output_file)

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise SystemExit(1)

    with open(in_path, "r") as f:
        alerts = json.load(f)
    logger.info(f"Loaded {len(alerts)} alerts from {in_path}")

    sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    if severity_min:
        min_lvl = sev_order.get(severity_min.lower(), 0)
        alerts = [a for a in alerts if sev_order.get(a.get("severity", "low"), 0) >= min_lvl]
        logger.info(f"Filtered to {len(alerts)} alerts with severity >= {severity_min}")

    if max_alerts:
        alerts = alerts[:max_alerts]
        logger.info(f"Limited to first {max_alerts} alerts")

    analyzer = LLMAnalyzer.from_config()

    enriched = []
    sources = Counter()
    for i, alert in enumerate(alerts):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Analyzing alert {i + 1}/{len(alerts)}...")
        analysis = analyzer.analyze(alert)
        sources[analysis.provider] += 1
        enriched.append({
            "alert": alert,
            "llm_analysis": analysis.to_dict(),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved {len(enriched)} enriched alerts to {out_path}")

    logger.info("=== LLM Analysis Summary ===")
    logger.info(f"  Total enriched: {len(enriched)}")
    for prov, n in sources.most_common():
        logger.info(f"  Provider {prov}: {n} ({100 * n / len(enriched):.0f}%)")


@cli.command()
@click.option("--replay", default=None, type=click.Path(),
              help="Mode replay : rejoue un NDJSON local au lieu d'OpenSearch")
@click.option("--speed", default=1.0, type=float,
              help="Vitesse de simulation en mode replay (1=temps réel, 60=1 min/s)")
def live(replay, speed):
    """Phase 7 : worker temps réel — streame OpenSearch et détecte les anomalies en continu.

    Exemples :
      python -m anomaly_detection.cli live
      python -m anomaly_detection.cli live --replay data/test_anomalies/test_day.json --speed 60
    """
    from .realtime.worker import run_worker
    run_worker(replay=replay, speed=speed)


@cli.command()
@click.option("--port", default=8501, type=int, help="Port HTTP du dashboard")
@click.option("--no-browser", is_flag=True, help="Ne pas ouvrir le navigateur automatiquement")
def dashboard(port, no_browser):
    """Phase 8 : lance le dashboard Streamlit pour visualiser les alertes en temps réel.

    Exemples :
      python -m anomaly_detection.cli dashboard
      python -m anomaly_detection.cli dashboard --port 8888 --no-browser
    """
    import subprocess, sys
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    if not app_path.exists():
        logger.error(f"App Streamlit introuvable : {app_path}")
        raise SystemExit(1)
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
    ]
    if no_browser:
        cmd += ["--server.headless", "true"]
    logger.info(f"Lancement du dashboard : {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("Streamlit n'est pas installé. Lance : pip install streamlit")
        raise SystemExit(1)
    except KeyboardInterrupt:
        logger.info("Dashboard arrêté")


if __name__ == "__main__":
    cli()
