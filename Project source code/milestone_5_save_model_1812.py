import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

print("--- TRAINING LOG-TRANSFORMED MODEL WITH SCALER ---")
df = pd.read_csv('8_FINAL_TRAINING_DATA_V2.csv')        

optimized_features = [
    'x0_Midfield - Left Midfield', 'x0_Defender - Right-Back', 'total_days_injured', 
    'x1_Spain', 'x0_Midfield - Right Midfield', 'x1_Brazil', 'x0_Defender - Centre-Back', 
    'x0_Goalkeeper', 'x0_Defender - Left-Back', 'x1_0', 'x1_Switzerland', 'minutes_played', 
    'value_lag_2', 'x0_Attack - Centre-Forward', 'x1_Mexico', 'x1_Türkiye', 'x1_Hungary', 
    'x1_Austria', 'x0_Attack - Right Winger', 'x1_Scotland', 'value_lag_1', 'x1_Germany', 
    'x1_France', 'x1_Poland', 'yellow_cards', 'x0_Midfield - Attacking Midfield', 
    'x0_Attack - Left Winger', 'x1_England', 'value_momentum', 'x1_Italy', 
    'goals', 'assists', 'sentiment'
]

X = df[optimized_features].values
y = df['market_value'].values 

y_log = np.log1p(y) 

print("Training Random Forest...")
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X, y_log)

print("Training Complete.")
joblib.dump(model, 'final_football_model_log.joblib')
joblib.dump(optimized_features, 'model_features_list.joblib')
print("SUCCESS: Model Saved.")
