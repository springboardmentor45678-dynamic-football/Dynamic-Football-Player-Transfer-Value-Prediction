import pandas as pd
import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- CONFIGURATION ---
MODEL_PATH = "final_football_model_log.joblib"
FEATURES_PATH = "model_features_list.joblib"

app = FastAPI(title="Football Valuation Engine (Hybrid Expert)", version="8.0")

# --- GLOBAL VARIABLES ---
model = None
model_features = None

# --- LOAD MODEL ---
@app.on_event("startup")
def load_artifacts():
    global model, model_features
    try:
        model = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)
        print("✅ Model Loaded Successfully")
    except Exception as e:
        print(f"🔴 Error loading model: {e}")

class PlayerData(BaseModel):
    goals: int
    assists: int
    minutes_played: int
    age_momentum: float
    prev_value: float
    days_injured: int
    country: str
    position: str

def manual_scale(value, avg, std):
    if std == 0: return 0
    return (value - avg) / std

@app.post("/predict")
def predict_value(data: PlayerData):
    if not model: raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        print(f"\n--- NEW REQUEST: {data.position} ({data.country}) ---")

        # 1. PREPARE INPUT DATAFRAME FOR ML MODEL
        input_df = pd.DataFrame(0, index=[0], columns=model_features)

        # 2. APPLY SCALING (Standardizing inputs for the AI)
        input_df['goals'] = manual_scale(data.goals, 4, 5) 
        input_df['assists'] = manual_scale(data.assists, 3, 4)
        input_df['minutes_played'] = manual_scale(data.minutes_played, 1800, 800)
        
        mom = data.age_momentum
        if abs(mom) > 1.0: mom = mom / 100.0
        input_df['value_momentum'] = mom 

        input_df['value_lag_1'] = manual_scale(data.prev_value, 2000000, 5000000)
        input_df['value_lag_2'] = manual_scale(data.prev_value, 2000000, 5000000)
        input_df['total_days_injured'] = manual_scale(data.days_injured, 15, 40)

        # 3. CATEGORICAL MATCHING
        country_col = f"x1_{data.country}"
        position_col = f"x0_{data.position}"
        if country_col in input_df.columns: input_df[country_col] = 1
        if position_col in input_df.columns: input_df[position_col] = 1

        # 4. GET BASE ML PREDICTION
        log_prediction = model.predict(input_df.values)[0]
        base_ml_value = np.expm1(log_prediction)

        # ---------------------------------------------------------
        # 5. 🔥 THE HYBRID LOGIC LAYER (The "Fix" for Responsiveness)
        # This layer guarantees inputs affect the final price.
        # ---------------------------------------------------------

        # A. Position Multiplier (Base Demand)
        pos_multiplier = 1.0
        if "Attack" in data.position: pos_multiplier = 1.15
        elif "Midfield" in data.position: pos_multiplier = 1.08
        elif "Defender" in data.position: pos_multiplier = 0.95
        elif "Goalkeeper" in data.position: pos_multiplier = 0.90
        
        # B. Nationality Premium (League Reputation)
        # Brazilians, English, French often cost more due to hype/Homegrown rules
        nat_multiplier = 1.0
        if data.country in ["Brazil", "England", "France"]: nat_multiplier = 1.10
        elif data.country in ["Spain", "Germany", "Argentina"]: nat_multiplier = 1.05
        
        # C. Performance Impact Calculator (Direct Euro value for stats)
        # This ensures Goals/Assists/Minutes ALWAYS change the price
        
        # Determine "Value per Goal" based on player tier
        tier_multiplier = max(data.prev_value / 1000000, 1) # Higher value players get bigger bonuses
        
        goal_bonus = data.goals * (50000 * tier_multiplier)  # e.g., €50k per goal for small players, more for stars
        assist_bonus = data.assists * (30000 * tier_multiplier)
        
        # Minutes Logic: Reward for playing > 2000 mins, Penalty for < 500
        mins_impact = 0
        if data.minutes_played > 2000:
            mins_impact = (data.minutes_played - 2000) * (500 * tier_multiplier)
        elif data.minutes_played < 1000:
            mins_impact = (data.minutes_played - 1000) * (1000 * tier_multiplier) # Steep penalty
            
        # Injury Penalty: Direct value loss per day
        injury_penalty = data.days_injured * (2000 * tier_multiplier)

        # D. COMBINE EVERYTHING
        adjusted_value = (base_ml_value * pos_multiplier * nat_multiplier) + goal_bonus + assist_bonus + mins_impact - injury_penalty
        
        # E. Final Safety Checks
        # Prevent negative values or impossible explosions
        if adjusted_value < 10000: adjusted_value = 10000
        
        # Sanity Clamp: Don't let value explode more than 3x previous (unless rookie)
        if data.prev_value > 1000000 and adjusted_value > (data.prev_value * 3):
            adjusted_value = data.prev_value * 3

        # ---------------------------------------------------------
        # 6. GENERATE REPORT DATA
        # ---------------------------------------------------------
        
        # Drivers (Explain "Why")
        drivers = []
        if data.goals >= 15: drivers.append("🔥 Elite Goalscoring Form (+)")
        elif data.goals >= 5: drivers.append("⚽ Consistent Goal Output (+)")
        
        if data.assists >= 10: drivers.append("🎯 Top Playmaker Stats (+)")
        
        if data.minutes_played > 2500: drivers.append("🛡️ Reliable Starter (+)")
        elif data.minutes_played < 800: drivers.append("⚠️ Low Playing Time (-)")
        
        if mom > 0.1: drivers.append("📈 High Growth Trend (+)")
        elif mom < -0.1: drivers.append("📉 Declining Form (-)")
        
        if data.days_injured > 60: drivers.append(f"🏥 Injury Impact (-€{injury_penalty:,.0f})")
        
        if nat_multiplier > 1.0: drivers.append(f"🌍 {data.country} Market Premium (+)")
        
        if not drivers: drivers.append("⚖️ Consistent Market Performance")

        # Similar Players Logic (Visual Context)
        val_m = adjusted_value / 1000000
        comparison = "Unknown"
        if val_m > 120: comparison = "Mbappe, Haaland, Vinicius Jr"
        elif val_m > 80: comparison = "Kane, Bellingham, Saka"
        elif val_m > 50: comparison = "Salah, Bruno Fernandes, Diaz"
        elif val_m > 30: comparison = "Watkins, Maddison, Gvardiol"
        elif val_m > 15: comparison = "Solid Top 5 League Starters"
        elif val_m > 5: comparison = "Squad Rotation / Championship Stars"
        else: comparison = "Academy Graduates / Emerging Pros"

        # Radar Chart Stats (0-100)
        # Normalize stats relative to an "Elite" player
        radar_stats = {
            "Attacking": min((data.goals * 3 + data.assists * 4), 100), 
            "Stamina": min((data.minutes_played / 3500) * 100, 100),
            "Availability": max(100 - (data.days_injured / 2), 0),
            "Potential": min(((mom * 100) + 50), 100)
        }

        print(f"💰 Final Calculation: €{adjusted_value:,.0f}")
        
        return {
            "status": "success",
            "market_value_euro": round(adjusted_value, 0),
            "range_min": round(adjusted_value * 0.90, 0),
            "range_max": round(adjusted_value * 1.10, 0),
            "drivers": drivers,
            "similar_players": comparison,
            "radar_stats": radar_stats
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)