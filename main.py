from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os


app = FastAPI()

# 1. Load Model and Features
model = joblib.load('best_model.pkl')
model_features = joblib.load('model_features.pkl')

# 2. Setup Static Files (Assuming your HTML/CSS are in a folder named 'static')
# Create folder if it doesn't exist: os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class PlayerData(BaseModel):
    age: float
    height: float
    goals_x: float
    assists: float
    minutes_played: float
    days_missed: float

# 3. Routes
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/predict")
async def predict(data: PlayerData):
    try:
        input_df = pd.DataFrame([data.dict()])
        
        # Ensure all training features are present (fills 30+ columns with 0 if missing)
        for feature in model_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_processed = input_df[model_features]
        prediction = model.predict(input_processed)
        
        return {"predicted_value": round(float(-prediction[0])*100, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))