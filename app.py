import streamlit as st
import requests

st.title("TransferIQ ")
st.subheader("Football Player Transfer Value Prediction")
st.markdown("Enter the player's details below to predict their market value.")
# Replace with your actual feature names and defaults
feature_defaults = {
    "age": 25,
    "goals": 10,
    "assists": 5,
    "matches": 30,
    "injury_days": 15,
    "sentiment": 0.1,
    "market_value_last": 1.5,
}

user_input = {}
for feature, default in feature_defaults.items():
    if isinstance(default, (int, float)):
        user_input[feature] = st.number_input(f"Enter {feature}", value=default)
    else:
        user_input[feature] = st.text_input(f"Enter {feature}", value=default)

if st.button("Predict"):
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=user_input,
        )
        if response.status_code == 200:
            result = response.json()["predicted_transfer_value"]
            st.success(f"Predicted Market Value: €{result}M")
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Request failed: {e}")
