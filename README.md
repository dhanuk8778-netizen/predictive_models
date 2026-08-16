# Predictive Maintenance & Failure Risk Modelling

Predicts industrial machine failure from live sensor readings (temperature,
rotational speed, torque, tool wear) using the **AI4I 2020 Predictive
Maintenance** dataset (10,000 samples, 3.39% failure rate).

## Why this project is harder than it looks

This is not a "load data, call `.fit()`" exercise. The real problem is
**severe class imbalance**: 96.6% of samples are healthy machines, so a
model that always predicts "no failure" is already 96.6% accurate and
completely useless. Everything here — metric choice, model configuration,
threshold selection — is built around that constraint.

## Pipeline

```
src/
├── data_prep.py       # Load + clean data, sanity-check labels
├── eda.py              # Class balance, distributions, correlations
├── features.py         # Leakage-safe feature engineering + train/test split
├── train.py             # RF + XGBoost, CV, hyperparameter search, imbalance handling
├── evaluate.py          # Confusion matrix, ROC/PR curves, operating points
└── shap_analysis.py     # Model interpretability
```

Run in order:
```bash
pip install -r requirements.txt
cd src
python data_prep.py
python eda.py
python features.py
python train.py
python evaluate.py
python shap_analysis.py
```

## Key decisions

- **Leakage control**: the dataset includes 5 failure-subtype flags
  (TWF, HDF, PWF, OSF, RNF) that are only known *after* a failure occurs.
  These are used only in EDA to validate the target column, then dropped
  before modeling.
- **Engineered features**: `temp_diff_k` (process − air temp),
  `power_proxy` (torque × rpm), and `torque_wear_product` — chosen
  because the dataset's own failure-generating logic is defined around
  exactly these physical relationships (verified by SHAP below).
- **Imbalance handling**: `class_weight="balanced"` for Random Forest,
  `scale_pos_weight` for XGBoost — both up-weight the rare failure class
  during training instead of resampling the data.
- **Model selection metric**: PR-AUC (not accuracy or plain ROC-AUC),
  since it's far more sensitive to how well a model finds the minority
  class.
- **Stratified 5-fold CV + RandomizedSearchCV** for hyperparameter tuning,
  with a completely untouched test set for final numbers.

## Results (held-out test set, 2,000 samples)

| Model | ROC-AUC | PR-AUC | Precision @ 0.5 | Recall @ 0.5 |
|---|---|---|---|---|
| Random Forest | 0.977 | 0.875 | 0.95 | 0.79 |
| XGBoost | 0.986 | 0.888 | 0.84 | 0.82 |

**Operating points** (XGBoost) — the numbers a maintenance team would
actually use to pick a threshold:

| Target recall | Achieved recall | Precision |
|---|---|---|
| 70% | 71% | 100% |
| 80% | 81% | 93% |
| 90% | 91% | 43% |
| 95% | 96% | 32% |

This is the real trade-off: catching 80% of failures costs very few false
alarms (93% precision), but pushing to 95% recall means accepting 2 false
alarms for every real failure caught — a genuine business decision, not a
modeling one.

## Interpretability (SHAP)

Global feature importance matches the dataset's own known failure
mechanics almost exactly: `tool_wear_min`, `rotational_speed_rpm`, and the
engineered `power_proxy` dominate — consistent with how power failure and
overstrain failure are actually defined in the underlying data generating
process. This is a good sanity check that the model learned real physics,
not spurious correlations.

See `output/figures/shap_summary.png` for the full breakdown and
`output/figures/shap_waterfall_example.png` for a single-prediction
explanation.

## Next steps

- Try LightGBM and compare against XGBoost
- Cost-sensitive threshold selection (assign $ cost to false alarm vs.
  missed failure, pick threshold that minimizes expected cost)
- Wrap the best model in a small FastAPI service for real-time scoring
  (natural extension into the "Production ML System" project)

## Data source

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset),
S. Matzka, *"Explainable Artificial Intelligence for Predictive Maintenance
Applications,"* 2020. Synthetic dataset modeled on a real milling machine.
