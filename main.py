from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TransferIQ API")

# Allow CORS for frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Decision Tree model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/decision_tree_regressor_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Load feature names
X_val = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_val_new.csv"))
FEATURES = X_val.columns.tolist()

@app.get("/")
def home():
    return {"message": "TransferIQ API (Decision Tree) is running successfully"}

@app.post("/predict")
def predict(data: dict):
    try:
        input_df = pd.DataFrame([data])

        # Fill missing features with 0 for model compatibility
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure correct column order
        input_df = input_df[FEATURES]

        # Convert all columns to numeric if possible (for one-hot and numeric features)
        input_df = input_df.apply(pd.to_numeric, errors='ignore')

        prediction = model.predict(input_df)[0]

        return {"predicted_transfer_value": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
