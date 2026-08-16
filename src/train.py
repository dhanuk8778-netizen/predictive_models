"""
train.py
--------
Trains and compares two models for failure prediction:
  1. Random Forest  (class_weight='balanced' to handle the 3.4% failure rate)
  2. XGBoost        (scale_pos_weight to handle the same imbalance)

Why care about imbalance at all? With only 3.4% positives, a model that
always predicts "no failure" is already 96.6% "accurate" -- which is
useless. That's why we evaluate with PR-AUC / recall / precision instead
of plain accuracy, and why both models are configured to weight the
minority (failure) class more heavily during training.

We use stratified 5-fold cross-validation + a small randomized
hyperparameter search on the training set only, then do one final
evaluation on the untouched test set.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from data_prep import load_clean
from features import build_model_table, split

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def tune_random_forest(X_train, y_train):
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 6, 10, 16],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=15, scoring="average_precision",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"[train] RF best CV PR-AUC: {search.best_score_:.4f}")
    print(f"[train] RF best params: {search.best_params_}")
    return search.best_estimator_


def tune_xgboost(X_train, y_train):
    # scale_pos_weight ~ (negatives / positives) tells XGBoost how much
    # more to weight a missed failure vs. a false alarm
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    xgb = XGBClassifier(
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=15, scoring="average_precision",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"[train] XGB best CV PR-AUC: {search.best_score_:.4f}")
    print(f"[train] XGB best params: {search.best_params_}")
    return search.best_estimator_


def main():
    df = load_clean()
    X, y = build_model_table(df)
    X_train, X_test, y_train, y_test = split(X, y)

    print("\n=== Tuning Random Forest ===")
    best_rf = tune_random_forest(X_train, y_train)

    print("\n=== Tuning XGBoost ===")
    best_xgb = tune_xgboost(X_train, y_train)

    # Quick held-out test check (full evaluation happens in evaluate.py)
    for name, model in [("Random Forest", best_rf), ("XGBoost", best_xgb)]:
        proba = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, proba)
        print(f"[train] {name} held-out test PR-AUC: {pr_auc:.4f}")

    joblib.dump(best_rf, MODEL_DIR / "random_forest.joblib")
    joblib.dump(best_xgb, MODEL_DIR / "xgboost.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")
    # Persist the split for evaluate.py / shap_analysis.py so results are
    # computed on the exact same test set
    X_test.assign(machine_failure=y_test.values).to_csv(
        MODEL_DIR / "test_set.csv", index=False
    )
    X_train.assign(machine_failure=y_train.values).to_csv(
        MODEL_DIR / "train_set.csv", index=False
    )
    print(f"\n[train] Saved models and data splits to {MODEL_DIR}")


if __name__ == "__main__":
    main()
