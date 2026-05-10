# Pipeline complet bout-en-bout 

# Stop au premier echec
$ErrorActionPreference = "Stop"

# Se placer a la racine du projet
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Print-Phase {
    param([string]$Title)
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Blue
    Write-Host "  $Title" -ForegroundColor Blue
    Write-Host "========================================================================" -ForegroundColor Blue
}

function Print-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

# Verifier que le dataset est present
if (-not (Test-Path "data/raw/dataset.json")) {
    Write-Host "[!] data/raw/dataset.json not found." -ForegroundColor Yellow
    Write-Host "    Please copy your training NDJSON file there before running this script."
    exit 1
}

# Phase 1: Enrichment
Print-Phase "PHASE 1 - Enrichissement des logs bruts"
python -m anomaly_detection.cli enrich `
    --input data/raw/dataset.json `
    --output data/processed/enriched
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Print-Ok "Phase 1 complete"

# Phase 2: Feature engineering
Print-Phase "PHASE 2 - Feature engineering (fenetres 5min/step 1min)"
python -m anomaly_detection.cli features `
    --input data/processed/enriched `
    --output data/features/windows_5min
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Print-Ok "Phase 2 complete"

# Phase 3: Test set with anomalies
Print-Phase "PHASE 3 - Generation du test set avec 7 anomalies labellisees"

python scripts/generate_test_set.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Print-Ok "Test set generated"

# Enrich test set with FROZEN templates (production-like)
python -m anomaly_detection.cli enrich `
    --input data/test_anomalies/test_day.json `
    --output data/test_anomalies/test_enriched `
    --templates data/processed/templates.tsv `
    --freeze-templates
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Generate features for test set
python -m anomaly_detection.cli features `
    --input data/test_anomalies/test_enriched `
    --output data/test_anomalies/test_features
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Print-Ok "Phase 3 complete"

# Phase 4: Train models + evaluate
if ($env:SKIP_TRAIN -ne "1") {
    Print-Phase "PHASE 4a - Entrainement (1 global + N par groupe)"
    python -m anomaly_detection.cli train
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Print-Ok "Phase 4a complete"

    Print-Phase "PHASE 4b - Evaluation sur test set anormal"
    python -m anomaly_detection.cli evaluate
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Print-Ok "Phase 4b complete"
} else {
    Write-Host "Skipping Phase 4 (SKIP_TRAIN=1)" -ForegroundColor Yellow
}

# Phase 5: Detection + alerts
Print-Phase "PHASE 5 - Detection + generation des alertes structurees"
python -m anomaly_detection.cli detect
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Print-Ok "Phase 5 complete"

# Summary
Print-Phase "PIPELINE COMPLET - RESUME"
Write-Host ""
Write-Host "Donnees produites :"
Write-Host "  * data/processed/enriched.{pkl.gz,parquet}    Logs enrichis (Phase 1)"
Write-Host "  * data/processed/templates.tsv                Dictionnaire templates"
Write-Host "  * data/features/windows_5min.{pkl.gz,parquet} Features par fenetre (Phase 2)"
Write-Host "  * data/test_anomalies/                        Test set + labels (Phase 3)"
Write-Host "  * data/models/                                Modeles entraines (Phase 4)"
Write-Host "  * data/models/eval_report.png                 Rapport visuel d'evaluation"
Write-Host "  * data/alerts/alerts.json                     Alertes brutes (Phase 5)"
Write-Host "  * data/alerts/alerts_incidents.json           Alertes dedupliquees en incidents"
Write-Host ""
Print-Ok "Tout est pret !"
