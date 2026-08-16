"""
shap_analysis.py
-----------------
Model interpretation using SHAP (SHapley Additive exPlanations).

Why this matters for predictive maintenance specifically: a maintenance
engineer won't trust (or act on) a "failure probability: 0.87" number
unless they can see *why*. SHAP breaks each prediction down into how much
each feature pushed the prediction up or down, which lets us:
  1. Rank global feature importance (summary plot)
  2. Explain individual high-risk predictions (waterfall plot)
  3. Sanity-check that the model is learning something physically sensible
     (e.g. does tool wear + torque actually drive overstrain-style
     predictions, matching the dataset's own known failure logic?)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
FIG_DIR = BASE / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    xgb = joblib.load(MODEL_DIR / "xgboost.joblib")
    test_df = pd.read_csv(MODEL_DIR / "test_set.csv")
    y_test = test_df["machine_failure"]
    X_test = test_df.drop(columns=["machine_failure"])

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_test)

    # 1. Global summary plot -- which features matter most, overall?
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Bar plot of mean |SHAP value| -- simpler global ranking
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Waterfall plot for the single highest-risk TRUE FAILURE case --
    # explains "why did the model correctly flag this one?"
    proba = xgb.predict_proba(X_test)[:, 1]
    failure_idx = np.where(y_test.values == 1)[0]
    top_failure_idx = failure_idx[np.argmax(proba[failure_idx])]

    plt.figure()
    shap.plots.waterfall(shap_values[top_failure_idx], show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print a compact text summary of global importance for the console/README
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False)
    print("[shap] Global feature importance (mean |SHAP value|):")
    print(importance.to_string())
    importance.to_csv(MODEL_DIR / "shap_feature_importance.csv", header=["mean_abs_shap"])

    print(f"\n[shap] Saved plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
