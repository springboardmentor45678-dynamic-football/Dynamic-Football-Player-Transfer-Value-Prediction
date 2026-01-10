# main.py (FastAPI Backend)
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import lightgbm as lgb

app = FastAPI()

# Load the trained LightGBM model
model = lgb.Booster(model_file='lightgbm_model.txt')

class PlayerData(BaseModel):
    age: float
    most_recent_transfer_fee: float
    total_career_goals: float
    total_career_assists: float
    days_since_joined: float
    total_transfers: float
    vader_polarity: float
    tb_polarity: float
    num_unique_teammates: float
    total_career_minutes_played: float
    total_value_at_transfer: float
    remaining_contract_duration: float
    days_since_last_transfer: float
    total_career_matches: float
    total_transfer_fees: float
    citizenship_freq_encoded: float
    club_prestige: float

@app.post("/predict")
def predict_market_value(data: PlayerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # --- REPLICATE FEATURE ENGINEERING FROM NOTEBOOK ---
    # Performance Age Ratio
    df['performance_age_ratio'] = (df['total_career_goals'] + df['total_career_assists']) / (df['age'] - 15).clip(lower=1)
    
    # Loyalty Index
    df['loyalty_index'] = df['days_since_joined'] / (df['total_transfers'] + 1)
    
    # Market Visibility
    df['market_visibility'] = (df['vader_polarity'] + df['tb_polarity']) * df['num_unique_teammates']
    
    # Career Momentum
    df['career_momentum'] = df['most_recent_transfer_fee'] / (df['age'] - 17).clip(lower=1)
    
    # Efficiency Index
    df['efficiency_index'] = (df['total_career_goals'] + df['total_career_assists']) / (df['total_career_minutes_played'] + 1)

    # Select the Gold Features used for training
    gold_features = [
        'club_prestige', 'total_value_at_transfer', 'most_recent_transfer_fee',
        'career_momentum', 'remaining_contract_duration', 'days_since_last_transfer',
        'total_career_matches', 'performance_age_ratio', 'total_career_minutes_played',
        'total_transfer_fees', 'market_visibility', 'loyalty_index',
        'citizenship_freq_encoded', 'efficiency_index'
    ]
    
    X_input = df[gold_features]

    # Predict (Inverse log transform as used in notebook)
    log_prediction = model.predict(X_input)
    real_prediction = np.expm1(log_prediction)[0]

    return {"predicted_value": float(real_prediction)}