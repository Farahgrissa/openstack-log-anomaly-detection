# OpenStack Anomaly Detection

Détection automatique d'incidents en quasi-temps réel sur des logs OpenStack hétérogènes (Kolla-Ansible all-in-one), avec alertes structurées, explicabilité, enrichissement LLM et dashboard Streamlit.

## Pipeline

```
NDJSON brut Fluentd
      ↓
[Phase 1] Enrichissement      → parsing HTTP, templating, mapping service→groupe
      ↓
[Phase 2] Feature engineering → fenêtres glissantes 5min/1min, ~85 features
      ↓
[Phase 4] Entraînement        → 1 modèle global + 5 modèles par groupe (architecture hybride ML + règles)
      ↓
[Phase 5] Inference           → scoring + explicabilité + top-K logs + déduplication
      ↓
[Phase 6] Couche LLM          → Gemini → Groq → Template 
      ↓
[Phase 7] Worker temps réel   → streaming OpenSearch + LLM async + heartbeat
      ↓
[Phase 8] Dashboard Streamlit → 4 onglets (live/incidents/logs/modèle), auto-refresh
```

## Performances (test set — 7 anomalies labellisées)

| Métrique | Valeur | Cible |
|----------|--------|-------|
| AUC-ROC  | **0.974** | > 0.85 |
| AUC-PR   | **0.912** | > 0.70 |
| Precision | **0.86** | > 0.70 |
| Recall   | **0.82**  | > 0.70 |
| F1       | **0.84**  | > 0.50 |
| FPR (validation sain) | **0.4 – 2.2%** | < 5% |

7/7 incidents détectés, dont 5 à recall 100%.

## Démarrage rapide

```bash
# Installation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Dataset → data/raw/dataset.json, puis :
./scripts/run_pipeline.sh

# Worker temps réel
python -m anomaly_detection.cli live

# Dashboard (autre terminal)
python -m anomaly_detection.cli dashboard
```

## CLI

```bash
python -m anomaly_detection.cli --help
python -m anomaly_detection.cli info            # infos + config
python -m anomaly_detection.cli check-data      # valide data/raw/
python -m anomaly_detection.cli enrich          # Phase 1
python -m anomaly_detection.cli features        # Phase 2
python -m anomaly_detection.cli train           # Phase 4 — entraînement
python -m anomaly_detection.cli evaluate        # Phase 4 — évaluation
python -m anomaly_detection.cli detect          # Phase 5 — alertes batch
python -m anomaly_detection.cli analyze         # Phase 6 — enrichissement LLM
python -m anomaly_detection.cli live            # Phase 7 — worker temps réel
python -m anomaly_detection.cli dashboard       # Phase 8 — dashboard Streamlit
```

---

*Projet académique — mai 2026.*
