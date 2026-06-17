"""
SentinelML — FastAPI application.
Loads model at startup, serves fraud predictions with SHAP explanations.
"""

import numpy as np
import pandas as pd
import joblib
import shap

from fastapi import FastAPI, HTTPException
from pathlib import Path
from xgboost import XGBClassifier

from api.schemas import TransactionRequest, PredictionResponse, HealthResponse, SHAPFeature

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="SentinelML — Fraud Detection API",
    description=(
        "XGBoost fraud detection model trained on 284k credit card transactions. "
        "Returns fraud probability, binary decision, and SHAP feature explanations."
    ),
    version="1.0.0",
)

MODELS_DIR = Path(__file__).parent.parent / "models"

# ─────────────────────────────────────────────
# Model loading — runs once at startup
# ─────────────────────────────────────────────

model = None
scaler = None
threshold = None
explainer = None
feature_names = None


@app.on_event("startup")
async def load_model():
    """
    Load model artifacts once when the server starts.
    Loading on every request would add 2-3 seconds per call.
    Startup loading means the model is always in memory, ready instantly.
    """
    global model, scaler, threshold, explainer, feature_names

    try:
        model = joblib.load(MODELS_DIR / "best_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        threshold = joblib.load(MODELS_DIR / "threshold.pkl")

        # Build SHAP explainer once at startup
        explainer = shap.TreeExplainer(model)

        # Feature names must match training order exactly
        feature_names = (
            ["Time"] +
            [f"V{i}" for i in range(1, 29)] +
            ["Amount"]
        )

        print(f"✓ Model loaded successfully")
        print(f"✓ Decision threshold: {threshold}")
        print(f"✓ Features: {len(feature_names)}")

    except FileNotFoundError as e:
        print(f"✗ Model file not found: {e}")
        print(f"  Run src/explain.py first to generate model artifacts.")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check — confirms model is loaded and ready."""
    return HealthResponse(
        status="ok" if model is not None else "model not loaded",
        model_loaded=model is not None,
        threshold=threshold if threshold is not None else 0.0,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """
    Predict whether a transaction is fraudulent.

    - Accepts all 30 transaction features as JSON
    - Scales Amount and Time using the training scaler
    - Returns fraud probability, binary decision, and top 5 SHAP features
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/explain.py first."
        )

    # Build DataFrame in exact training column order
    data = transaction.model_dump()
    df = pd.DataFrame([data])[feature_names]

    # Scale Amount and Time — same as training
    df_scaled = df.copy()
    df_scaled[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

    # Predict
    fraud_prob = float(model.predict_proba(df_scaled)[0][1])
    is_fraud = fraud_prob >= threshold

    # SHAP explanation
    shap_values = explainer.shap_values(df_scaled)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Build top 5 features by absolute SHAP value
    feature_impacts = []
    for fname, sval, fval in zip(feature_names, sv, df_scaled.iloc[0]):
        feature_impacts.append(
            SHAPFeature(
                feature=fname,
                shap_value=round(float(sval), 6),
                feature_value=round(float(fval), 6),
                direction="→ fraud" if sval > 0 else "→ legitimate",
            )
        )
    feature_impacts.sort(key=lambda x: abs(x.shap_value), reverse=True)
    top_features = feature_impacts[:5]

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 6),
        is_fraud=is_fraud,
        decision_threshold=threshold,
        top_features=top_features,
    )
