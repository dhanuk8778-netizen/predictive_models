"""
eda.py
------
Exploratory Data Analysis for the predictive maintenance dataset.
Generates and saves a handful of plots that matter for this problem:
  1. Class balance (failure rate)
  2. Feature distributions split by failure/no-failure
  3. Correlation heatmap of numeric features
  4. Failure rate by product type (L/M/H)
"""

import matplotlib
matplotlib.use("Agg")  # headless -- we're saving to file, not showing a window
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from data_prep import load_clean

FIG_DIR = Path(__file__).resolve().parents[1] / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
    "air_temp_k", "process_temp_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min",
]

sns.set_theme(style="whitegrid")


def plot_class_balance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["machine_failure"].value_counts().sort_index()
    labels = ["No Failure", "Failure"]
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f"{v} ({v/len(df):.1%})", ha="center")
    ax.set_title("Class Balance: Machine Failure")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        sns.kdeplot(data=df, x=col, hue="machine_failure", ax=axes[i],
                    common_norm=False, fill=True, alpha=0.4)
        axes[i].set_title(col)
    axes[-1].axis("off")
    fig.suptitle("Feature Distributions by Failure Status", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df[NUMERIC_COLS + ["machine_failure"]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_failure_by_type(df: pd.DataFrame):
    rate = df.groupby("product_type")["machine_failure"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(5, 4))
    rate.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Failure Rate by Product Type")
    ax.set_ylabel("Failure Rate")
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.001, f"{v:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "failure_rate_by_type.png", dpi=150)
    plt.close(fig)


def run_eda():
    df = load_clean()
    plot_class_balance(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_failure_by_type(df)
    print(f"[eda] Saved 4 figures to {FIG_DIR}")


if __name__ == "__main__":
    run_eda()
