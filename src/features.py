"""
features.py
-----------
Feature engineering for the predictive maintenance model.

Important design decision: we DROP the five failure-subtype columns
(twf, hdf, pwf, osf, rnf) before modeling. They are only known *after* a
failure happens, so including them would be data leakage -- the model
would be "cheating" by looking at the answer. We only use them earlier,
in EDA, to sanity-check the target column.

We also engineer a few physically-motivated features based on how the
dataset's own failure modes are defined (temperature difference relates
to heat dissipation failure; torque * tool wear relates to overstrain
failure). This mirrors how a domain expert would think about the problem,
not just throwing raw columns at a model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

LEAKY_COLS = ["twf", "hdf", "pwf", "osf", "rnf"]
ID_COLS = ["udi", "product_id"]
TARGET = "machine_failure"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Temperature gap -- small gaps are associated with heat dissipation failure
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]

    # Power proxy (torque * rotational speed) -- power failure is defined
    # around a power threshold in the dataset's own generating process
    df["power_proxy"] = df["torque_nm"] * df["rotational_speed_rpm"]

    # Torque * tool wear -- overstrain failure threshold depends on this product
    df["torque_wear_product"] = df["torque_nm"] * df["tool_wear_min"]

    # One-hot encode product type (L/M/H)
    df = pd.get_dummies(df, columns=["product_type"], prefix="type")

    return df


def build_model_table(df: pd.DataFrame):
    """Return (X, y) ready for modeling, with leaky/id columns removed."""
    df = engineer_features(df)
    drop_cols = LEAKY_COLS + ID_COLS + [TARGET]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]
    return X, y


def split(X, y, test_size=0.2, random_state=42):
    """Stratified split -- important given the ~3.4% failure rate, so both
    train and test sets keep the same failure proportion."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    from data_prep import load_clean

    df = load_clean()
    X, y = build_model_table(df)
    print("Feature columns:", list(X.columns))
    print("X shape:", X.shape, " y positive rate:", y.mean())
    X_train, X_test, y_train, y_test = split(X, y)
    print("Train:", X_train.shape, " Test:", X_test.shape)
    print("Train failure rate:", y_train.mean(), " Test failure rate:", y_test.mean())
