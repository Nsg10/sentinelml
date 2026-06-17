"""
SentinelML — Pydantic request and response schemas.
Defines the exact shape of API input and output.
"""

from pydantic import BaseModel, Field
from typing import List


class TransactionRequest(BaseModel):
    """
    A single credit card transaction.
    All 30 features required — Time, V1-V28, Amount.
    All must be floats.
    """
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount in dollars", ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 406.0,
                "V1": -2.3122,
                "V2": 1.9519,
                "V3": -1.6097,
                "V4": 3.9979,
                "V5": -0.5220,
                "V6": -1.4265,
                "V7": -2.5374,
                "V8": 1.3918,
                "V9": -2.7700,
                "V10": -2.7722,
                "V11": 3.2020,
                "V12": -2.8992,
                "V13": -0.5950,
                "V14": -4.2895,
                "V15": 0.3898,
                "V16": -1.1407,
                "V17": -2.8300,
                "V18": -0.0168,
                "V19": 0.4165,
                "V20": 0.1267,
                "V21": 0.5173,
                "V22": -0.0355,
                "V23": -0.4654,
                "V24": 0.3799,
                "V25": 0.1455,
                "V26": -0.0751,
                "V27": 0.1279,
                "V28": 0.1021,
                "Amount": 149.62
            }
        }
    }


class SHAPFeature(BaseModel):
    """One feature's contribution to the prediction."""
    feature: str
    shap_value: float
    feature_value: float
    direction: str


class PredictionResponse(BaseModel):
    """
    API response for a fraud prediction.
    Returns probability, binary decision, and top SHAP features.
    """
    fraud_probability: float = Field(..., description="Model confidence 0-1")
    is_fraud: bool = Field(..., description="True if fraud probability >= threshold")
    decision_threshold: float = Field(..., description="Threshold used for decision")
    top_features: List[SHAPFeature] = Field(..., description="Top 5 SHAP features")
    model_version: str = Field(default="xgb_tuned_v1")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    threshold: float
