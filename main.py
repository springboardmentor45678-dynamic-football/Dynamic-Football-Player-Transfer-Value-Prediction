from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# -----------------------------------
# Create FastAPI app
# -----------------------------------
app = FastAPI(
    title="Transfermarkt Player Value Prediction API",
    description="XGBoost model for football player market value prediction",
    version="1.0"
)

# -----------------------------------
# Load Model & Feature List
# -----------------------------------
model = joblib.load("xgboost_model.joblib")
features = joblib.load("model_features.joblib")

print("✅ Model loaded")
print("✅ Features:", features)

# -----------------------------------
# Request Schema (Dynamic Input)
# -----------------------------------
class PlayerInput(BaseModel):
    Rank: float | None = 0
    Rank_perf: float | None = 0
    Age: float | None = 0
    Age_perf: float | None = 0
    Matches: float | None = 0
    Goals: float | None = 0
    goal_contribution: float | None = 0
    Sentiment: float | None = 0

# -----------------------------------
# Health Check
# -----------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "XGBoost",
        "features": features
    }

# -----------------------------------
# Demo Prediction (Browser-friendly)
# -----------------------------------
@app.get("/predict_demo")
def predict_demo():
    # Create input with all features = 0
    input_data = {feature: 0 for feature in features}

    # Change ONE feature for demo
    input_data["Rank"] = 10

    df = pd.DataFrame([input_data])

    pred_log = model.predict(df)[0]
    pred_euro = 10 ** pred_log

    return {
        "market_value_log": round(float(pred_log), 4),
        "market_value_euro": round(float(pred_euro), 2)
    }

# -----------------------------------
# Main Prediction Endpoint
# -----------------------------------
@app.post("/predict")
def predict(data: PlayerInput):
    # Convert input to dict
    user_data = data.dict()

    # Ensure all features exist
    input_data = {feature: 0 for feature in features}

    # Update with provided values
    for key, value in user_data.items():
        if key in input_data:
            input_data[key] = value

    df = pd.DataFrame([input_data])

    pred_log = model.predict(df)[0]
    pred_euro = 10 ** pred_log

    return {
        "market_value_log": round(float(pred_log), 4),
        "market_value_euro": round(float(pred_euro), 2)
    }
