import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import shap


st.set_page_config(
    page_title="Football Market Value Predictor",
    layout="wide"
)


with open("lightgbm_best_model.pkl", "rb") as f:
    model = pickle.load(f)

final_features = [
    "mv_pct_from_peak",
    "current_club_name_te",
    "team_club_name_te",
    "team_competition_id_te",
    "mv_gap_from_peak",
    "team_country_name_te",
    "team_competitions_count",
    "team_team_seasons_count",
    "team_latest_season",
    "pv_mv_count",
    "age",
    "tm_joint_goal_participation",
    "tm_unique_teammates",
    "tr_last_transfer_fee",
    "tr_total_transfer_fees",
    "tr_first_transfer_date_te",
    "tr_years_since_last_transfer",
    "perf_assists_rolling_5",
    "inj_total_injuries",
    "tr_last_transfer_date_te",
    "perf_assists",
    "perf_assists_rolling_10",
    "inj_mean_days_out",
    "inj_seasons_with_injury"
]


@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_shap_explainer(model)



st.sidebar.header("⚽ Player Inputs")

age = st.sidebar.number_input("Age", 16, 40, 24)
perf_assists = st.sidebar.number_input("Total Goal Assists", 0, 100, 10)
perf_assists_rolling_10 = st.sidebar.number_input("Assists (Last 10 Matches)", 0, 20, 4)
inj_total_injuries = st.sidebar.number_input("Total Injuries", 0, 20, 1)
tr_years_since_last_transfer = st.sidebar.slider(
    "Years Since Last Transfer", 0.0, 10.0, 1.5
)
tr_last_transfer_fee = st.sidebar.number_input(
    "Last Transfer Fee (€)", 0.0, 2e8, 5e6
)
mv_pct_from_peak = st.sidebar.slider(
    "Market Value % from Peak", 0.0, 1.0, 0.7
)


input_data = {
    "mv_pct_from_peak": mv_pct_from_peak,
    "current_club_name_te": 0.45,
    "team_club_name_te": 0.40,
    "team_competition_id_te": 0.50,
    "mv_gap_from_peak": 2_000_000,
    "team_country_name_te": 0.35,
    "team_competitions_count": 3,
    "team_team_seasons_count": 2,
    "team_latest_season": 2024,
    "pv_mv_count": 15,
    "age": age,
    "tm_joint_goal_participation": perf_assists * 2,
    "tm_unique_teammates": 60,
    "tr_last_transfer_fee": tr_last_transfer_fee,
    "tr_total_transfer_fees": tr_last_transfer_fee * 1.5,
    "tr_first_transfer_date_te": 0.30,
    "tr_years_since_last_transfer": tr_years_since_last_transfer,
    "perf_assists_rolling_5": perf_assists_rolling_10 / 2,
    "inj_total_injuries": inj_total_injuries,
    "tr_last_transfer_date_te": 0.60,
    "perf_assists": perf_assists,
    "perf_assists_rolling_10": perf_assists_rolling_10,
    "inj_mean_days_out": inj_total_injuries * 15,
    "inj_seasons_with_injury": min(inj_total_injuries, 3)
}

input_df = pd.DataFrame([input_data])[final_features]


st.title("⚽ Football Player Market Value Prediction")
st.markdown(
    "Predict player market value using a **LightGBM-based machine learning model** "
    "trained on performance, transfer, injury, and market data."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Feature Importance (Model Level)")

    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": final_features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    fig_imp = px.bar(
        imp_df.tail(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Driving Market Value"
    )
    st.plotly_chart(fig_imp, use_container_width=True)


with col2:
    st.subheader("💰 Predicted Market Value")

    if st.button("Predict Market Value"):
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Market Value: €{prediction:,.0f}")
    else:
        st.info("Adjust inputs from the sidebar and click **Predict Market Value**.")


st.divider()
st.subheader("🧠 Model Explainability (Why this prediction?)")

st.markdown(
    "The charts below explain **which factors influenced the prediction**. "
    "Positive impact increases market value, while negative impact reduces it."
)

with st.expander("🎯 Explanation for this player", expanded=True):

    shap_vals_local = explainer.shap_values(input_df)[0]

    local_df = pd.DataFrame({
        "Feature": final_features,
        "SHAP Impact": shap_vals_local
    }).sort_values(by="SHAP Impact")

    fig_local = px.bar(
        local_df.tail(10),
        x="SHAP Impact",
        y="Feature",
        orientation="h",
        title="How Each Feature Affected This Player"
    )
    st.plotly_chart(fig_local, use_container_width=True)

    st.caption(
        "Bars to the right increase predicted value; bars to the left decrease it."
    )

st.caption(
    "⚙️ Powered by LightGBM | Football Market Value Prediction Project"
)
 