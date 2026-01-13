import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="TransferIQ", page_icon="⚽", layout="wide")

st.title("⚽ TransferIQ")
st.subheader("Dynamic Player Transfer Value Prediction using AI")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "data/final_master_player_dataset.csv",
        encoding="latin1"
    )

df = load_data()

# ---------------- CONFIG ----------------
PLAYER_COL = "player_clean"

FEATURES = [
    "age",
    "overall_rating",
    "potential",
    "international_reputation(1-5)",
    "skill_moves(1-5)",
    "weak_foot(1-5)",
    "total_events"
]

df = df.dropna(subset=FEATURES)

# ---------------- SYNTHETIC TARGET ----------------
df["transfer_value"] = (
    (df["overall_rating"] * 1_200_000) +
    (df["potential"] * 800_000) +
    (df["international_reputation(1-5)"] * 2_000_000) +
    (df["skill_moves(1-5)"] * 1_000_000) +
    (df["weak_foot(1-5)"] * 700_000) +
    (df["total_events"] * 50_000) -
    (df["age"] * 300_000)
)

df["transfer_value"] = df["transfer_value"].clip(lower=500_000)

# ---------------- MODEL ----------------
X = df[FEATURES]
y = np.log1p(df["transfer_value"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    random_state=42
)
model.fit(X_scaled, y)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎮 Player Controls")

player = st.sidebar.selectbox(
    "Select Player",
    sorted(df[PLAYER_COL].unique())
)

player_row = df[df[PLAYER_COL] == player].iloc[0]

input_data = {}
for f in FEATURES:
    min_v = float(df[f].min())
    max_v = float(df[f].max())
    if min_v == max_v:
        min_v -= 1
        max_v += 1

    input_data[f] = st.sidebar.slider(
        f.replace("_", " ").title(),
        min_value=min_v,
        max_value=max_v,
        value=float(player_row[f])
    )

# ---------------- PREDICT ----------------
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

log_pred = model.predict(input_scaled)[0]
prediction = np.expm1(log_pred)

# ---------------- OUTPUT ----------------
st.markdown("## 🔮 Predicted Transfer Value")
st.markdown(
    f"<h1 style='color:#27AE60;'>€{int(prediction):,}</h1>",
    unsafe_allow_html=True
)

st.markdown("### 📊 Player Attributes")
st.dataframe(input_df)

