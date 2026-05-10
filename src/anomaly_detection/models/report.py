"""Rapport visuel d'évaluation : courbes ROC, PR, analyse multi-seuils."""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from ..utils.logging import logger
from .isolation_forest import AnomalyDetector
from .evaluate import load_test_data


def plot_evaluation(
    models_dir: str | Path,
    test_features_path: str | Path,
    test_labels_path: str | Path,
    output_path: str | Path = "data/models/eval_report.png",
    only_active: bool = True,
) -> None:
    """Produit un graphique complet :
       - ROC, PR pour chaque modèle
       - Distribution des scores sain vs anormal
       - Tableau multi-seuils
       - Recall par type d'anomalie
    """
    models_dir = Path(models_dir)
    test_df = load_test_data(test_features_path, test_labels_path)
    if only_active and "count_total" in test_df.columns:
        test_df = test_df[test_df["count_total"] > 0].reset_index(drop=True)

    y_true = test_df["is_anomaly"].values

    # Charger tous les modèles
    models = {}
    for path in sorted(models_dir.glob("model_*.joblib")):
        m = AnomalyDetector.load(path)
        models[m.name] = m

    fig = plt.figure(figsize=(20, 14))

    # 1. ROC curves
    ax1 = plt.subplot(2, 3, 1)
    for name, model in models.items():
        scores = model.score(test_df)
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", lw=1.5)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    # 2. PR curves
    ax2 = plt.subplot(2, 3, 2)
    for name, model in models.items():
        scores = model.score(test_df)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(rec, prec)
        ax2.plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})", lw=1.5)
    baseline = y_true.mean()
    ax2.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.3)

    # 3. Distribution des scores - modèle global
    ax3 = plt.subplot(2, 3, 3)
    if "global" in models:
        scores = models["global"].score(test_df)
        sain = scores[y_true == 0]
        anom = scores[y_true == 1]
        ax3.hist(sain, bins=40, alpha=0.6, label=f"Normal (n={len(sain)})", color="steelblue", density=True)
        ax3.hist(anom, bins=40, alpha=0.6, label=f"Anomaly (n={len(anom)})", color="crimson", density=True)
        ax3.axvline(models["global"].threshold, color="black", linestyle="--",
                    label=f"Threshold (P95)")
        ax3.set_xlabel("Anomaly Score")
        ax3.set_ylabel("Density")
        ax3.set_title("Score Distribution (Global Model)")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

    # 4. Multi-threshold analysis (modèle global)
    ax4 = plt.subplot(2, 3, 4)
    if "global" in models:
        model = models["global"]
        scores = model.score(test_df)
        thresholds_pct = [99.5, 99.0, 98.0, 95.0, 90.0, 85.0, 80.0, 75.0]
        precs, recs, f1s, fpr_list = [], [], [], []
        for pct in thresholds_pct:
            t = np.percentile(model.train_scores, pct)
            y_pred = (scores > t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            fpr = fp / max(1, fp + tn)
            precs.append(prec); recs.append(rec); f1s.append(f1); fpr_list.append(fpr)
        ax4.plot(thresholds_pct, precs, "o-", label="Precision")
        ax4.plot(thresholds_pct, recs, "s-", label="Recall")
        ax4.plot(thresholds_pct, f1s, "^-", label="F1")
        ax4.plot(thresholds_pct, fpr_list, "x-", label="FPR")
        ax4.invert_xaxis()  # Plus à droite = seuil plus strict
        ax4.set_xlabel("Threshold percentile (P)")
        ax4.set_ylabel("Score")
        ax4.set_title("Metric vs Threshold (Global Model)")
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        ax4.set_ylim(-0.05, 1.05)

    # 5. Recall par type d'anomalie
    ax5 = plt.subplot(2, 3, 5)
    if "global" in models:
        model = models["global"]
        scores = model.score(test_df)
        y_pred = (scores > model.threshold).astype(int)

        recalls_by_type = {}
        for atype in test_df.loc[test_df["is_anomaly"] == 1, "anomaly_type"].unique():
            if atype == "":
                continue
            mask = (test_df["anomaly_type"] == atype)
            n = mask.sum()
            n_detected = ((mask) & (y_pred == 1)).sum()
            recalls_by_type[atype] = (n_detected / n if n > 0 else 0, n)

        types = list(recalls_by_type.keys())
        recalls = [recalls_by_type[t][0] for t in types]
        ns = [recalls_by_type[t][1] for t in types]
        labels = [f"{t}\n(n={n})" for t, n in zip(types, ns)]
        colors = ['green' if r >= 0.7 else 'orange' if r >= 0.3 else 'red' for r in recalls]
        ax5.barh(labels, recalls, color=colors)
        ax5.set_xlabel("Recall")
        ax5.set_xlim(0, 1.05)
        ax5.set_title("Per-anomaly-type Recall (Global Model @ current threshold)")
        ax5.grid(alpha=0.3, axis="x")
        for i, r in enumerate(recalls):
            ax5.text(r + 0.02, i, f"{r:.0%}", va="center", fontsize=8)

    # 6. Tableau métriques
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    table_data = []
    for name, model in models.items():
        scores = model.score(test_df)
        y_pred = (scores > model.threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        roc_auc_val = auc(*roc_curve(y_true, scores)[:2])
        table_data.append([name, f"{roc_auc_val:.3f}", f"{prec:.2f}",
                           f"{rec:.2f}", f"{f1:.2f}", f"{tp}/{tp+fn}", f"{fp}"])

    table = ax6.table(
        cellText=table_data,
        colLabels=["Model", "AUC-ROC", "Precision", "Recall", "F1", "TP/Anom", "FP"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax6.set_title("Summary metrics @ calibrated threshold", pad=20)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=80, bbox_inches="tight")
    logger.info(f"Saved evaluation report: {output_path}")
