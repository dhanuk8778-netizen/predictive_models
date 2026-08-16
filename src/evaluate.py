"""
evaluate.py
-----------
Full evaluation of the trained models on the held-out test set:
  - Confusion matrix at default 0.5 threshold
  - ROC curve + ROC-AUC
  - Precision-Recall curve + PR-AUC  (the more informative curve here,
    given the 3.4% failure rate)
  - "Operating point" table: at a few chosen recall levels, what precision
    (false-alarm rate) do we get? This is the number a maintenance team
    actually cares about: "if we want to catch 90% of failures, how many
    false alarms do we generate?"
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
FIG_DIR = BASE / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_test_set():
    df = pd.read_csv(MODEL_DIR / "test_set.csv")
    y = df["machine_failure"]
    X = df.drop(columns=["machine_failure"])
    return X, y


def operating_points(y_true, proba, recall_targets=(0.7, 0.8, 0.9, 0.95)):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    rows = []
    for target in recall_targets:
        # Find the highest-precision point that still achieves >= target recall
        idx_candidates = np.where(recall[:-1] >= target)[0]
        if len(idx_candidates) == 0:
            continue
        idx = idx_candidates[np.argmax(precision[idx_candidates])]
        rows.append({
            "recall_target": target,
            "achieved_recall": recall[idx],
            "precision": precision[idx],
            "threshold": thresholds[idx],
        })
    return pd.DataFrame(rows)


def evaluate_model(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_test, preds, target_names=["No Failure", "Failure"]))

    cm = confusion_matrix(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")

    op = operating_points(y_test, proba)
    print("\nOperating points (best precision at each recall target):")
    print(op.to_string(index=False))

    return {"name": name, "proba": proba, "cm": cm, "roc_auc": roc_auc,
            "pr_auc": pr_auc, "operating_points": op}


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        cm = res["cm"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No Failure", "Failure"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No Failure", "Failure"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(res["name"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(results, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        axes[0].plot(fpr, tpr, label=f"{res['name']} (AUC={res['roc_auc']:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, res["proba"])
        axes[1].plot(rec, prec, label=f"{res['name']} (AUC={res['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend()

    baseline = y_test.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", alpha=0.4,
                     label=f"Baseline ({baseline:.1%})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def main():
    X_test, y_test = load_test_set()
    rf = joblib.load(MODEL_DIR / "random_forest.joblib")
    xgb = joblib.load(MODEL_DIR / "xgboost.joblib")

    results = [
        evaluate_model("Random Forest", rf, X_test, y_test),
        evaluate_model("XGBoost", xgb, X_test, y_test),
    ]

    plot_confusion_matrices(results)
    plot_roc_pr_curves(results, y_test)
    print(f"\n[evaluate] Saved evaluation plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
