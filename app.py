
        # app.py (Streamlit Frontend)
import streamlit as st
import requests

st.set_page_config(page_title="AI Football Scout", layout="wide")

st.title("⚽ Football Player Market Value Predictor")
st.write("Enter player statistics to estimate current market value based on AI analysis.")

# Create columns for organized input
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=15, max_value=45, value=23)
    club_prestige = st.number_input("Club Prestige (Avg Squad Value)", value=25000000)
    most_recent_transfer_fee = st.number_input("Most Recent Transfer Fee (€)", value=50000000)
    total_transfer_fees = st.number_input("Total Career Transfer Fees (€)", value=60000000)

with col2:
    total_career_goals = st.number_input("Total Career Goals", value=15)
    total_career_assists = st.number_input("Total Career Assists", value=5)
    total_career_matches = st.number_input("Total Career Matches", value=60)
    total_career_minutes_played = st.number_input("Total Career Minutes", value=5000)

with col3:
    remaining_contract_duration = st.number_input("Contract Remaining (Years)", value=3)
    days_since_joined = st.number_input("Days at Current Club", value=400)
    total_transfers = st.number_input("Total Career Transfers", value=2)
    days_since_last_transfer = st.number_input("Days Since Last Transfer", value=200)

# Hidden/Static values for demo (can be added as sliders/inputs)
vader_polarity = 0.2
tb_polarity = 0.1
num_unique_teammates = 30
total_value_at_transfer = 40000000
citizenship_freq_encoded = 0.05

if st.button("Predict Market Value"):
    payload = {
        "age": age,
        "most_recent_transfer_fee": most_recent_transfer_fee,
        "total_career_goals": total_career_goals,
        "total_career_assists": total_career_assists,
        "days_since_joined": days_since_joined,
        "total_transfers": total_transfers,
        "vader_polarity": vader_polarity,
        "tb_polarity": tb_polarity,
        "num_unique_teammates": num_unique_teammates,
        "total_career_minutes_played": total_career_minutes_played,
        "total_value_at_transfer": total_value_at_transfer,
        "remaining_contract_duration": remaining_contract_duration,
        "days_since_last_transfer": days_since_last_transfer,
        "total_career_matches": total_career_matches,
        "total_transfer_fees": total_transfer_fees,
        "citizenship_freq_encoded": citizenship_freq_encoded,
        "club_prestige": club_prestige
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        prediction = response.json()["predicted_value"]
        st.success(f"### Estimated Market Value: €{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
