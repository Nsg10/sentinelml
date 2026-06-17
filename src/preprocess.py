"""
SentinelML — Data loading and exploration utilities.
Run this file directly to print a full EDA report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "creditcard.csv"


def load_data() -> pd.DataFrame:
    """Load the raw creditcard CSV and return as DataFrame."""
    df = pd.read_csv(DATA_PATH)
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Print a structured EDA report to stdout."""

    print("=" * 60)
    print("SENTINELML — EDA REPORT")
    print("=" * 60)

    # --- Shape ---
    print(f"\n[1] SHAPE")
    print(f"    Rows: {df.shape[0]:,}")
    print(f"    Columns: {df.shape[1]}")

    # --- Column names ---
    print(f"\n[2] COLUMNS")
    print(f"    {list(df.columns)}")

    # --- Missing values ---
    print(f"\n[3] MISSING VALUES")
    missing = df.isnull().sum().sum()
    print(f"    Total missing cells: {missing}")

    # --- Class distribution ---
    print(f"\n[4] CLASS DISTRIBUTION")
    counts = df["Class"].value_counts()
    total = len(df)
    fraud_count = counts[1]
    legit_count = counts[0]
    fraud_pct = fraud_count / total * 100
    print(f"    Legitimate (0): {legit_count:,}  ({100 - fraud_pct:.4f}%)")
    print(f"    Fraud (1):      {fraud_count:,}  ({fraud_pct:.4f}%)")
    print(f"    Imbalance ratio: {legit_count // fraud_count}:1  (legit:fraud)")

    # --- Feature statistics ---
    print(f"\n[5] AMOUNT FEATURE")
    print(f"    Min:    ${df['Amount'].min():.2f}")
    print(f"    Max:    ${df['Amount'].max():.2f}")
    print(f"    Mean:   ${df['Amount'].mean():.2f}")
    print(f"    Median: ${df['Amount'].median():.2f}")
    print(f"    Std:    ${df['Amount'].std():.2f}")

    print(f"\n[6] TIME FEATURE")
    print(f"    Min:  {df['Time'].min():.0f}s")
    print(f"    Max:  {df['Time'].max():.0f}s")
    print(f"    Span: {df['Time'].max() / 3600:.1f} hours of transactions")

    # --- Amount by class ---
    print(f"\n[7] AMOUNT BY CLASS")
    print(f"    Mean amount — Legitimate: ${df[df['Class']==0]['Amount'].mean():.2f}")
    print(f"    Mean amount — Fraud:      ${df[df['Class']==1]['Amount'].mean():.2f}")
    print(f"    Max fraud amount:         ${df[df['Class']==1]['Amount'].max():.2f}")

    # --- V feature ranges (PCA components) ---
    print(f"\n[8] PCA FEATURE RANGES (V1–V28)")
    v_cols = [f"V{i}" for i in range(1, 29)]
    v_stats = df[v_cols].describe().loc[["min", "max", "mean", "std"]]
    print(f"    Mean of means: {v_stats.loc['mean'].mean():.4f}  (expect ~0, PCA-centered)")
    print(f"    Std range:     {v_stats.loc['std'].min():.2f} – {v_stats.loc['std'].max():.2f}")

    # --- Fraud amount distribution insight ---
    print(f"\n[9] FRAUD AMOUNT BUCKETS")
    fraud_df = df[df["Class"] == 1]["Amount"]
    print(f"    Fraud txns under $100:   {(fraud_df < 100).sum()} ({(fraud_df < 100).mean()*100:.1f}%)")
    print(f"    Fraud txns $100–$1000:   {((fraud_df >= 100) & (fraud_df < 1000)).sum()}")
    print(f"    Fraud txns over $1000:   {(fraud_df >= 1000).sum()}")

    print(f"\n[10] SCALING NEEDED?")
    print(f"    Amount std: {df['Amount'].std():.2f}  ← very different from V features")
    print(f"    V1 std:     {df['V1'].std():.2f}")
    print(f"    → Amount and Time must be scaled before training.")

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


def save_class_distribution_plot(df: pd.DataFrame) -> None:
    """Save a class distribution bar chart to data/eda_class_dist.png"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    counts = df["Class"].value_counts()
    axes[0].bar(["Legitimate", "Fraud"], counts.values,
                color=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.5)
    axes[0].set_title("Class Distribution (Raw Count)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=10)

    # Amount distribution by class
    axes[1].hist(df[df["Class"]==0]["Amount"], bins=50, alpha=0.6,
                 color="#2ecc71", label="Legitimate", density=True)
    axes[1].hist(df[df["Class"]==1]["Amount"], bins=50, alpha=0.6,
                 color="#e74c3c", label="Fraud", density=True)
    axes[1].set_title("Amount Distribution by Class")
    axes[1].set_xlabel("Amount ($)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].set_xlim(0, 2000)

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "data" / "eda_class_dist.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()
    run_eda(df)
    save_class_distribution_plot(df)


# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Separate features from label and split into train/test sets.
    MUST be called before SMOTE — never apply SMOTE before splitting.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # preserve fraud ratio in both splits
    )

    print(f"\n[SPLIT]")
    print(f"  Train size: {len(X_train):,}  | Fraud in train: {y_train.sum():,}")
    print(f"  Test size:  {len(X_test):,}   | Fraud in test:  {y_test.sum():,}")
    print(f"  Train fraud rate: {y_train.mean()*100:.4f}%")
    print(f"  Test fraud rate:  {y_test.mean()*100:.4f}%")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale Amount and Time using StandardScaler.
    V1-V28 are already PCA-transformed and centered — no scaling needed.
    Fit scaler ONLY on training data, then transform both train and test.
    This prevents data leakage from test statistics into training.
    """
    scaler = StandardScaler()
    cols_to_scale = ["Amount", "Time"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print(f"\n[SCALING]")
    print(f"  Scaled columns: {cols_to_scale}")
    print(f"  Amount mean after scaling: {X_train['Amount'].mean():.4f}  (expect ~0)")
    print(f"  Amount std after scaling:  {X_train['Amount'].std():.4f}   (expect ~1)")

    return X_train, X_test, scaler


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to training data only.
    Generates synthetic fraud samples by interpolating between
    real fraud cases in feature space.
    Target: 50/50 balance between classes after oversampling.
    """
    print(f"\n[SMOTE — BEFORE]")
    print(f"  Legitimate: {(y_train == 0).sum():,}")
    print(f"  Fraud:      {(y_train == 1).sum():,}")
    print(f"  Ratio:      {(y_train == 0).sum() // (y_train == 1).sum()}:1")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n[SMOTE — AFTER]")
    print(f"  Legitimate: {(y_resampled == 0).sum():,}")
    print(f"  Fraud:      {(y_resampled == 1).sum():,}")
    print(f"  Ratio:      1:1  (balanced)")
    print(f"  New training size: {len(X_resampled):,}")

    return X_resampled, y_resampled


def run_preprocessing_pipeline(df: pd.DataFrame):
    """
    Full pipeline: split → scale → SMOTE.
    Returns everything needed for model training.
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    print(f"\n[PIPELINE COMPLETE]")
    print(f"  Training data shape (post-SMOTE): {X_train_resampled.shape}")
    print(f"  Test data shape (untouched):      {X_test.shape}")

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler
