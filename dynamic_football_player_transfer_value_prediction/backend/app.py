from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from model_loader import load_models

app = FastAPI(title="Player Market Value Prediction API")

# Load models once at startup
rf_model, lgb_model, selected_features = load_models()

class PlayerInput(BaseModel):
    data: dict

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict_value(input_data: PlayerInput):
    """
    Input:
    {
        "data": {
            "age": 25,
            "height": 180,
            ...
        }
    }
    """

    df = pd.DataFrame([input_data.data])

    # Ensure correct feature order
    df = df.reindex(columns=selected_features, fill_value=0)

    prediction = lgb_model.predict(df)[0]

    return {
        "predicted_market_value": float(prediction)
    }
