"""
SentinelML — Model training with MLflow experiment tracking.
Runs 5 experiments: LogReg, RF x2, XGBoost x2.
Each run logs parameters, metrics, and the trained model artifact.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

from src.preprocess import load_data, run_preprocessing_pipeline

EXPERIMENT_NAME = "sentinelml_fraud_detection"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def compute_metrics(model, X_test, y_test, threshold=0.5):
    """
    Compute all evaluation metrics at a given decision threshold.
    Default threshold is 0.5 — we'll tune this separately.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_test, y_prob),
        "threshold": threshold,
    }


def save_confusion_matrix(model, X_test, y_test, run_name, threshold=0.5):
    """Save confusion matrix as PNG artifact for this run."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"])
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="black")

    ax.set_title(f"Confusion Matrix — {run_name}\n(threshold={threshold})")
    plt.tight_layout()

    path = MODELS_DIR / f"cm_{run_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def run_experiment(run_name, model, params, X_train, X_test,
                   y_train, y_test, threshold=0.5):
    """
    Single MLflow run: train model, log params + metrics + artifacts.
    """
    print(f"\n{'='*50}")
    print(f"  RUN: {run_name}")
    print(f"{'='*50}")

    with mlflow.start_run(run_name=run_name):

        # --- Log parameters ---
        mlflow.log_params(params)

        # --- Train ---
        print(f"  Training...")
        model.fit(X_train, y_train)

        # --- Evaluate ---
        metrics = compute_metrics(model, X_test, y_test, threshold)
        mlflow.log_metrics(metrics)

        # --- Log model artifact ---
        if "XGB" in run_name:
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # --- Save + log confusion matrix ---
        cm_path = save_confusion_matrix(model, X_test, y_test,
                                        run_name, threshold)
        mlflow.log_artifact(cm_path)

        # --- Print results ---
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    return metrics


def run_all_experiments(X_train, X_test, y_train, y_test):
    """
    Run all 5 experiments under one MLflow experiment namespace.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}

    # --- Run 1: Logistic Regression (baseline) ---
    params_lr = {
        "model_type": "LogisticRegression",
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
    }
    model_lr = LogisticRegression(
        C=params_lr["C"],
        max_iter=params_lr["max_iter"],
        solver=params_lr["solver"],
        class_weight=params_lr["class_weight"],
        random_state=42,
        n_jobs=-1,
    )
    results["LR_baseline"] = run_experiment(
        "LR_baseline", model_lr, params_lr,
        X_train, X_test, y_train, y_test
    )

    # --- Run 2: Random Forest default ---
    params_rf1 = {
        "model_type": "RandomForest",
        "n_estimators": 100,
        "max_depth": "None",
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    }
    model_rf1 = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    results["RF_default"] = run_experiment(
        "RF_default", model_rf1, params_rf1,
        X_train, X_test, y_train, y_test
    )

    # --- Run 3: Random Forest tuned ---
    params_rf2 = {
        "model_type": "RandomForest",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
    }
    model_rf2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    results["RF_tuned"] = run_experiment(
        "RF_tuned", model_rf2, params_rf2,
        X_train, X_test, y_train, y_test
    )

    # --- Run 4: XGBoost default ---
    # scale_pos_weight handles class imbalance in XGBoost
    # set to ratio of negative/positive in ORIGINAL data (before SMOTE)
    # We already applied SMOTE so data is balanced — set to 1
    params_xgb1 = {
        "model_type": "XGBoost",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": 1,
        "subsample": 0.8,
    }
    model_xgb1 = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1,
        subsample=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    results["XGB_default"] = run_experiment(
        "XGB_default", model_xgb1, params_xgb1,
        X_train, X_test, y_train, y_test
    )

    # --- Run 5: XGBoost tuned ---
    params_xgb2 = {
        "model_type": "XGBoost",
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "scale_pos_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
    }
    model_xgb2 = XGBClassifier(
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
    results["XGB_tuned"] = run_experiment(
        "XGB_tuned", model_xgb2, params_xgb2,
        X_train, X_test, y_train, y_test
    )

    return results


def print_summary(results):
    """Print a comparison table of all 5 runs."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Run':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print(f"  {'-'*60}")
    for run_name, metrics in results.items():
        print(f"  {run_name:<20} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} "
              f"{metrics['auc_roc']:>10.4f}")


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = run_preprocessing_pipeline(df)

    print("\nStarting MLflow experiments...")
    results = run_all_experiments(X_train, X_test, y_train, y_test)
    print_summary(results)

    print(f"\nTo view results in MLflow UI, run:")
    print(f"  mlflow ui")
    print(f"  Then open: http://127.0.0.1:5000")
