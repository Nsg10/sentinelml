# SentinelML — Fraud Detection & MLOps Pipeline

End-to-end machine learning pipeline for credit card fraud detection.
Trained on 284,807 transactions, deployed as a live REST API with
per-prediction SHAP explainability.

**Live API:** https://sentinelml-db9j.onrender.com  
**Interactive docs:** https://sentinelml-db9j.onrender.com/docs

---

## What it does

- Detects fraudulent credit card transactions with **97.8% AUC-ROC**
- Returns fraud probability, binary decision, and top 5 contributing features per prediction
- Decision threshold tuned to 0.63 to maximize recall on a 577:1 imbalanced dataset

---

## Architecture
Raw CSV (284k rows)

↓

Preprocessing: train/test split → StandardScaler → SMOTE (1:1 balance)

↓

MLflow Experiments: 5 runs across LogReg / Random Forest / XGBoost

↓

Best Model: XGBoost (AUC-ROC 0.980, threshold=0.63)

↓

SHAP Explainability: top contributing features per prediction

↓

FastAPI REST endpoint → Render (live)

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Models | XGBoost, Random Forest, Logistic Regression |
| Imbalance handling | SMOTE (imbalanced-learn) |
| Experiment tracking | MLflow |
| Explainability | SHAP TreeExplainer |
| API | FastAPI + Pydantic |
| Deployment | Render |
| Dataset | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## Dataset

- **Source:** ULB Machine Learning Group via Kaggle
- **Size:** 284,807 transactions, 492 fraud cases
- **Fraud rate:** 0.172% (577:1 imbalance ratio)
- **Features:** Time, V1–V28 (PCA-anonymized), Amount

Not included in this repo. Download from Kaggle and place at `data/creditcard.csv`.

---

## Model results

| Run | Precision | Recall | F1 | AUC-ROC |
|-----|-----------|--------|----|---------|
| LR_baseline | 0.058 | 0.918 | 0.109 | 0.970 |
| RF_default | 0.845 | 0.837 | 0.841 | 0.973 |
| RF_tuned | 0.750 | 0.827 | 0.786 | 0.970 |
| XGB_default | 0.357 | 0.878 | 0.507 | 0.975 |
| **XGB_tuned** | **0.601** | **0.878** | **0.714** | **0.980** |

All runs tracked in MLflow with full parameter, metric, and artifact logging.

**Model selection:** XGB_tuned chosen for highest AUC-ROC (0.980) — best ranking
ability across all thresholds, independent of the threshold decision.

**Threshold tuning:** swept 0.1–0.9, selected 0.63 to maximize recall
subject to precision ≥ 0.6. Missing fraud has higher cost than false alarms.

---

## API usage

### Health check
```bash
curl https://sentinelml-db9j.onrender.com/health
```

```json
{"status": "ok", "model_loaded": true, "threshold": 0.63}
```

### Fraud prediction
```bash
curl -X POST https://sentinelml-db9j.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0, "V1": -2.3122, "V2": 1.9519, "V3": -1.6097,
    "V4": 3.9979, "V5": -0.5220, "V6": -1.4265, "V7": -2.5374,
    "V8": 1.3918, "V9": -2.7700, "V10": -2.7722, "V11": 3.2020,
    "V12": -2.8992, "V13": -0.5950, "V14": -4.2895, "V15": 0.3898,
    "V16": -1.1407, "V17": -2.8300, "V18": -0.0168, "V19": 0.4165,
    "V20": 0.1267, "V21": 0.5173, "V22": -0.0355, "V23": -0.4654,
    "V24": 0.3799, "V25": 0.1455, "V26": -0.0751, "V27": 0.1279,
    "V28": 0.1021, "Amount": 149.62
  }'
```

```json
{
  "fraud_probability": 0.999138,
  "is_fraud": true,
  "decision_threshold": 0.63,
  "top_features": [
    {"feature": "V14", "shap_value": 4.709767, "feature_value": -4.2895, "direction": "→ fraud"},
    {"feature": "V10", "shap_value": 1.650971, "feature_value": -2.7722, "direction": "→ fraud"},
    {"feature": "V17", "shap_value": 0.819522, "feature_value": -2.83,   "direction": "→ fraud"},
    {"feature": "V4",  "shap_value": 0.761664, "feature_value": 3.9979,  "direction": "→ fraud"},
    {"feature": "Time","shap_value": -0.753565,"feature_value": -1.989523,"direction": "→ legitimate"}
  ],
  "model_version": "xgb_tuned_v1"
}
```

---

## Run locally

```bash
# Clone and setup
git clone https://github.com/Nsg10/sentinelml.git
cd sentinelml
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download dataset from Kaggle → place at data/creditcard.csv

# Run preprocessing + training
PYTHONPATH=. python src/preprocess.py
PYTHONPATH=. python src/train.py
PYTHONPATH=. python src/explain.py

# Start API
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

---

## Project structure
sentinelml/

├── data/                  # Dataset (gitignored — download from Kaggle)

├── src/

│   ├── preprocess.py      # EDA, split, scaling, SMOTE

│   ├── train.py           # MLflow experiment tracking (5 runs)

│   └── explain.py         # Threshold tuning + SHAP explainability

├── api/

│   ├── main.py            # FastAPI application

│   └── schemas.py         # Pydantic request/response models

├── models/                # Saved artifacts (best_model.pkl, scaler, threshold)

├── requirements.txt

├── render.yaml

└── README.md

---

## Author

Niharika G · [GitHub](https://github.com/Nsg10) ·
Built as part of a 90-day FAANG preparation sprint.
