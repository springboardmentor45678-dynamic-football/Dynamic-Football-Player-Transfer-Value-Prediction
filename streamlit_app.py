import streamlit as st
import requests
import math

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="AI Transfer IQ",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽AI Transfer IQ")
st.caption("AI-based real-world market value estimation (₹ Crore)")

# ----------------------------------
# USER INPUTS (TYPED)
# ----------------------------------
player_name = st.text_input("Player Name", placeholder="e.g. Lamine Yamal")

age = st.number_input("Age", min_value=15, max_value=45, value=18)
matches = st.number_input("Matches Played", min_value=0, value=35)
goals = st.number_input("Goals", min_value=0, value=12)
assists = st.number_input("Assists", min_value=0, value=10)

performance = st.number_input(
    "Overall Performance Rating (0–10)",
    min_value=0.0,
    max_value=10.0,
    value=9.5
)

# ----------------------------------
# INTERNAL FEATURE MAPPING (HIDDEN)
# ----------------------------------
Rank = max(1, 100 - int(goals * 4 + assists * 3))
Rank_perf = performance / 10
goal_contribution = (goals + assists) / max(matches, 1)
Age_perf = 1.2 if age <= 20 else 1.0
Sentiment = Rank_perf

payload = {
    "Rank": Rank,
    "Rank_perf": Rank_perf,
    "Age": age,
    "Age_perf": Age_perf,
    "Matches": matches,
    "Goals": goals,
    "goal_contribution": goal_contribution,
    "Sentiment": Sentiment
}

# ----------------------------------
# PREDICT
# ----------------------------------
if st.button("🔮 Estimate Market Value"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            res = response.json()
            base_euro_value = res["market_value_euro"]

            # ----------------------------------
            # SOFT REAL-WORLD CALIBRATION
            # ----------------------------------
            if age <= 20:
                age_factor = 1 + (21 - age) * 0.9
            elif age <= 24:
                age_factor = 1 + (25 - age) * 0.4
            else:
                age_factor = 1.2

            perf_factor = 1 + (performance / 10) ** 1.7
            scarcity_bonus = 1 + math.log1p(goal_contribution * 8)

            final_euro_value = base_euro_value * age_factor * perf_factor * scarcity_bonus

            # Soft ceiling ≈ ₹1600–1700 Cr
            final_euro_value = min(final_euro_value, 200_000_000)

            # ----------------------------------
            # EURO → INR CRORE CONVERSION
            # ----------------------------------
            EURO_TO_INR = 90
            inr_value = final_euro_value * EURO_TO_INR
            inr_crore = inr_value / 10_000_000

            name = player_name.strip() if player_name.strip() else "the player"

            st.success(
                f"Estimated real-world market value for **{name}** is\n\n"
                f"### 🇮🇳 ₹ {inr_crore:,.0f} Crore"
            )

            st.caption(
                "Value shown in Indian Rupees based on market-adjusted estimation."
            )

        else:
            st.error("Prediction failed. Ensure backend is running.")

    except Exception:
        st.error("Unable to connect to backend. Please start the FastAPI server.")
