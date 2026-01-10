import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Player Value Predictor", layout="centered")

st.title("⚽ Player Market Value Prediction")
st.write("Enter player statistics below")

age = st.number_input("Age", 16, 45, 25)
height = st.number_input("Height (cm)", 150, 210, 180)
goals = st.number_input("Goals", 0, 50, 5)
assists = st.number_input("Assists", 0, 30, 3)
minutes = st.number_input("Minutes Played", 0, 5000, 1200)

if st.button("Predict Market Value"):
    payload = {
        "data": {
            "age": age,
            "height": height,
            "goals": goals,
            "assists": assists,
            "minutes_played": minutes
        }
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        value = response.json()["predicted_market_value"]
        st.success(f"💰 Predicted Market Value: €{value:,.0f}")
    else:
        st.error("Prediction failed. Check backend.")
