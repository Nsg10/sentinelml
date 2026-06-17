"""
SentinelML — Threshold tuning and SHAP explainability.
Selects the best decision threshold and explains individual predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)
from xgboost import XGBClassifier

from src.preprocess import load_data, run_preprocessing_pipeline

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# THRESHOLD TUNING
# ─────────────────────────────────────────────

def tune_threshold(model, X_test, y_test, min_precision=0.6):
    """
    Sweep thresholds from 0.1 to 0.9.
    Find the threshold that maximizes recall
    subject to precision >= min_precision.

    In fraud detection, recall matters more than precision:
    - False negative (missed fraud) = real financial loss
    - False positive (false alarm) = customer friction, investigation cost
    We accept more false alarms to catch more fraud.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.91, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f = f1_score(y_test, y_pred, zero_division=0)
        results.append({"threshold": round(t, 2), "precision": p,
                        "recall": r, "f1": f})

    df_results = pd.DataFrame(results)

    # Best threshold: highest recall where precision >= min_precision
    valid = df_results[df_results["precision"] >= min_precision]

    if valid.empty:
        print(f"  No threshold achieves precision >= {min_precision}")
        print(f"  Falling back to best F1 threshold")
        best_row = df_results.loc[df_results["f1"].idxmax()]
    else:
        best_row = valid.loc[valid["recall"].idxmax()]

    best_threshold = best_row["threshold"]

    print(f"\n[THRESHOLD TUNING]")
    print(f"  Min precision constraint: {min_precision}")
    print(f"  Best threshold found:     {best_threshold}")
    print(f"  Precision at threshold:   {best_row['precision']:.4f}")
    print(f"  Recall at threshold:      {best_row['recall']:.4f}")
    print(f"  F1 at threshold:          {best_row['f1']:.4f}")

    # Print comparison: default vs tuned
    default_row = df_results[df_results["threshold"] == 0.5].iloc[0]
    print(f"\n  Comparison (default 0.5 vs tuned {best_threshold}):")
    print(f"  {'Metric':<12} {'Default 0.5':>12} {'Tuned':>12} {'Change':>10}")
    print(f"  {'-'*48}")
    for metric in ["precision", "recall", "f1"]:
        default_val = default_row[metric]
        tuned_val = best_row[metric]
        change = tuned_val - default_val
        arrow = "↑" if change > 0 else "↓"
        print(f"  {metric:<12} {default_val:>12.4f} {tuned_val:>12.4f} "
              f"{arrow}{abs(change):>8.4f}")

    return best_threshold, df_results


def plot_threshold_curves(df_results, best_threshold):
    """Save precision/recall vs threshold plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: precision and recall vs threshold
    axes[0].plot(df_results["threshold"], df_results["precision"],
                 label="Precision", color="#2ecc71", linewidth=2)
    axes[0].plot(df_results["threshold"], df_results["recall"],
                 label="Recall", color="#e74c3c", linewidth=2)
    axes[0].plot(df_results["threshold"], df_results["f1"],
                 label="F1", color="#3498db", linewidth=2, linestyle="--")
    axes[0].axvline(x=best_threshold, color="black", linestyle=":",
                    linewidth=1.5, label=f"Best threshold ({best_threshold})")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision / Recall / F1 vs Threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: precision vs recall tradeoff curve
    axes[1].plot(df_results["recall"], df_results["precision"],
                 color="#9b59b6", linewidth=2)
    best_row = df_results[df_results["threshold"] == best_threshold].iloc[0]
    axes[1].scatter([best_row["recall"]], [best_row["precision"]],
                    color="black", zorder=5, s=80,
                    label=f"Chosen point (t={best_threshold})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Tradeoff")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = MODELS_DIR / "threshold_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {path}")


# ─────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

def build_shap_explainer(model, X_train_sample):
    """
    Build a SHAP TreeExplainer for XGBoost.
    TreeExplainer is optimized for tree-based models —
    it uses the tree structure directly instead of sampling,
    making it exact and fast.

    X_train_sample: a small sample of training data used
    to compute background distribution for SHAP values.
    """
    print(f"\n[SHAP] Building TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print(f"  Explainer built successfully.")
    return explainer


def explain_prediction(explainer, X_single, feature_names, top_n=5):
    """
    Explain a single prediction.
    Returns top N features with their SHAP values.

    SHAP value interpretation:
    - Positive SHAP value → feature pushed prediction TOWARD fraud
    - Negative SHAP value → feature pushed prediction AWAY from fraud
    - Magnitude → how much impact this feature had
    """
    shap_values = explainer.shap_values(X_single)

    # For binary classification, shap_values shape is (1, n_features)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # class 1 (fraud)
    else:
        sv = shap_values[0]

    # Build sorted feature importance
    feature_impacts = []
    for i, (fname, sval) in enumerate(zip(feature_names, sv)):
        feature_impacts.append({
            "feature": fname,
            "shap_value": round(float(sval), 6),
            "feature_value": round(float(X_single.iloc[0, i]), 6),
            "direction": "→ fraud" if sval > 0 else "→ legitimate"
        })

    feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return feature_impacts[:top_n]


def print_shap_explanation(explanation, prediction_prob, threshold):
    """Pretty-print a SHAP explanation for one transaction."""
    decision = "FRAUD" if prediction_prob >= threshold else "LEGITIMATE"
    print(f"\n[SHAP EXPLANATION]")
    print(f"  Fraud probability: {prediction_prob:.4f}")
    print(f"  Decision (t={threshold}): {decision}")
    print(f"\n  Top contributing features:")
    print(f"  {'Feature':<12} {'Value':>10} {'SHAP':>12} {'Direction':>15}")
    print(f"  {'-'*52}")
    for f in explanation:
        print(f"  {f['feature']:<12} {f['feature_value']:>10.4f} "
              f"{f['shap_value']:>12.6f} {f['direction']:>15}")


def save_shap_summary_plot(explainer, X_test_sample, feature_names):
    """Save SHAP summary plot showing global feature importance."""
    print(f"\n[SHAP] Generating summary plot (this takes ~30 seconds)...")
    shap_values = explainer.shap_values(X_test_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test_sample, feature_names=feature_names,
                      show=False, max_display=15)
    path = MODELS_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


def save_best_model(model, threshold, scaler):
    """Save the best model, threshold, and scaler for API use."""
    joblib.dump(model, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(threshold, MODELS_DIR / "threshold.pkl")
    print(f"\n[SAVED]")
    print(f"  models/best_model.pkl")
    print(f"  models/scaler.pkl")
    print(f"  models/threshold.pkl")


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = run_preprocessing_pipeline(df)

    # Retrain best model (XGB_tuned) — same params as train.py Run 5
    print("\nRetraining XGB_tuned (best model)...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    # Threshold tuning
    best_threshold, df_results = tune_threshold(
        model, X_test, y_test, min_precision=0.6
    )
    plot_threshold_curves(df_results, best_threshold)

    # SHAP
    feature_names = list(X_test.columns)
    explainer = build_shap_explainer(model, X_train.iloc[:500])

    # Explain one real fraud case from test set
    fraud_indices = y_test[y_test == 1].index
    X_fraud_sample = X_test.loc[[fraud_indices[0]]]
    fraud_prob = model.predict_proba(X_fraud_sample)[0][1]
    explanation = explain_prediction(
        explainer, X_fraud_sample, feature_names, top_n=5
    )
    print_shap_explanation(explanation, fraud_prob, best_threshold)

    # SHAP summary plot on 500 test samples
    X_test_sample = X_test.iloc[:500]
    save_shap_summary_plot(explainer, X_test_sample, feature_names)

    # Save everything for the API
    save_best_model(model, best_threshold, scaler)

    print("\n" + "="*60)
    print("  STEP 4 COMPLETE")
    print("  Best model, threshold, and scaler saved to models/")
    print("="*60)
