
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

folder_path = '/content/gdrive/MyDrive/TransferIQ_proj/data'
model_path = os.path.join(folder_path, 'lightgbm_model.pkl')

FEATURE_NAMES = [
    'total_career_matches', 'total_career_goals', 'total_career_assists',
    'total_career_minutes_played', 'avg_season_goals', 'avg_season_assists',
    'total_career_yellow_cards', 'avg_rolling_avg_3_seasons_minutes_played',
    'overall_avg_recovery_time', 'total_transfer_fees', 'total_value_at_transfer',
    'most_recent_transfer_fee', 'age', 'days_since_joined',
    'avg_ppg_with_teammates', 'total_joint_goal_participation',
    'value_per_goal', 'value_per_minute_played', 'market_value_to_age_ratio'
]

# --- Load Model --- #
@st.cache_resource
def load_model():
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

loaded_model = load_model()

st.set_page_config(page_title="Player Market Value Predictor", layout="wide")
st.title("⚽Player Market Value Predictor")
st.markdown("Enter player statistics to predict their market value.")

st.sidebar.header("Player Features Input")

input_data = {}
for feature in FEATURE_NAMES:
    if 'market_value' in feature or 'transfer_fee' in feature or 'value_per' in feature:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    elif 'days' in feature or 'minutes' in feature:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", min_value=0.0, value=365.0, step=1.0, format="%.0f")
    elif 'age' in feature:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", min_value=15.0, max_value=45.0, value=25.0, step=1.0, format="%.0f")
    else:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", min_value=0.0, value=50.0, step=1.0, format="%.0f")

if st.sidebar.button("Predict Market Value"):
    input_df = pd.DataFrame([input_data])

    input_df = input_df[FEATURE_NAMES]

    prediction = loaded_model.predict(input_df)[0]

    st.subheader("Predicted Market Value")
    st.success(f"The predicted market value for the player is: €{prediction:,.2f}")

st.sidebar.markdown("---")
