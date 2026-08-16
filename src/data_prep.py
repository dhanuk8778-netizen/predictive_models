"""
data_prep.py
------------
Load the AI4I 2020 Predictive Maintenance dataset and clean it up.

The raw file has a few quirks that are worth handling explicitly (this is a
good habit to build early -- real datasets are never perfectly tidy):
  - A UTF-8 BOM character on the first column name.
  - Windows-style line endings.
  - Column names with spaces and units (e.g. "Air temperature [K]"),
    which are annoying to type in code, so we rename them.
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"

# Map ugly raw column names -> clean, code-friendly names
COLUMN_RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "product_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

FAILURE_SUBTYPES = ["twf", "hdf", "pwf", "osf", "rnf"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and apply clean column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_RENAME)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning + sanity checks. Doesn't drop anything from AI4I since the
    dataset is already fairly clean, but this is where you'd handle nulls,
    duplicates, and obvious outliers on a messier dataset.
    """
    df = df.copy()

    # Duplicates
    n_dupes = df.duplicated(subset=["product_id"]).sum()
    if n_dupes:
        df = df.drop_duplicates(subset=["product_id"])

    # Nulls -- AI4I has none, but we check anyway (defensive programming)
    n_nulls = df.isnull().sum().sum()

    # Sanity check: machine_failure should be 1 whenever any subtype fired
    subtype_any = df[FAILURE_SUBTYPES].sum(axis=1) > 0
    mismatch = (subtype_any != df["machine_failure"].astype(bool)).sum()

    print(f"[data_prep] rows={len(df)}  duplicates_removed={n_dupes}  "
          f"nulls={n_nulls}  label_mismatches={mismatch}")

    return df


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    return basic_clean(load_raw(path))


if __name__ == "__main__":
    data = load_clean()
    print(data.head())
    print(data["machine_failure"].value_counts(normalize=True))
