import streamlit as st
import requests
import time
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro Scout AI | Valuation Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CSS ---
st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 20px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        color: white;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #1F2937;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #10B981;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Driver Tags */
    .driver-tag-pos {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        padding: 5px 12px;
        border-radius: 15px;
        border: 1px solid #10B981;
        font-size: 0.85em;
        margin: 3px;
        display: inline-block;
    }
    .driver-tag-neg {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        padding: 5px 12px;
        border-radius: 15px;
        border: 1px solid #EF4444;
        font-size: 0.85em;
        margin: 3px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (Inputs) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d3/Soccerball.svg", width=50)
    st.markdown("## **Player Profile**")
    
    country = st.selectbox("🌍 Nationality", ["Spain", "Germany", "France", "Brazil", "England", "Italy", "Portugal", "Argentina", "Netherlands"])
    
    pos_map = {
        "Striker": "Attack - Centre-Forward",
        "Winger (Left)": "Attack - Left Winger",
        "Winger (Right)": "Attack - Right Winger",
        "Attacking Midfield": "Midfield - Attacking Midfield",
        "Defensive Midfield": "Midfield - Defensive Midfield",
        "Centre-Back": "Defender - Centre-Back",
        "Full-Back (Left)": "Defender - Left-Back",
        "Full-Back (Right)": "Defender - Right-Back",
        "Goalkeeper": "Goalkeeper"
    }
    position_display = st.selectbox("🎽 Position", list(pos_map.keys()))
    position_backend = pos_map[position_display]

    st.markdown("---")
    st.markdown("### 📊 Season Performance")
    col1, col2 = st.columns(2)
    with col1: goals = st.number_input("Goals", 0, 100, 12)
    with col2: assists = st.number_input("Assists", 0, 100, 8)
    minutes = st.slider("⏱️ Minutes Played", 0, 5000, 2400, step=50)
    
    st.markdown("### 🏥 Fitness & History")
    injured = st.slider("Days Injured", 0, 365, 0)
    prev_val = st.number_input("💰 Previous Value (€)", min_value=0, max_value=200000000, value=15000000, step=500000, format="%d")
    momentum = st.slider("📈 Momentum (%)", -50.0, 50.0, 10.0)

# --- MAIN CONTENT ---
st.title("⚽ Pro Scout AI: Valuation Engine")
st.caption("Powered by Hybrid Machine Learning (Random Forest + Expert Logic)")
st.markdown("---")

# Prediction Button
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    predict_btn = st.button("🚀 PREDICT MARKET VALUE")

if predict_btn:
    with st.spinner('🤖 Analyzing Stats & Market Trends...'):
        time.sleep(0.5) 
        
        payload = {
            "goals": goals, "assists": assists, "minutes_played": minutes,
            "age_momentum": momentum, "prev_value": prev_val,
            "days_injured": injured, "country": country, "position": position_backend
        }
        
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()
            
            if result['status'] == 'success':
                val = result['market_value_euro']
                val_min = result['range_min']
                val_max = result['range_max']
                drivers = result['drivers']
                similar = result['similar_players']
                radar = result['radar_stats']
                
                # --- SECTION 1: HERO METRIC ---
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color:#9CA3AF; margin:0; font-size: 16px; text-transform: uppercase; letter-spacing: 1px;">Estimated Market Value</h2>
                    <h1 style="color:#10B981; font-size: 52px; margin: 10px 0; font-weight: 800; text-shadow: 0px 0px 20px rgba(16, 185, 129, 0.4);">€ {val:,.0f}</h1>
                    <p style="color:#D1D5DB; font-size: 14px;">Confidence Range: <span style="color:white; font-weight:bold;">€{val_min:,.0f} - €{val_max:,.0f}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # --- SECTION 2: VISUALS (RADAR CHART & DETAILS) ---
                col_graph, col_info = st.columns([1.2, 1])
                
                with col_graph:
                    # Create Radar Chart
                    categories = list(radar.keys())
                    values = list(radar.values())
                    # Close the loop
                    categories.append(categories[0])
                    values.append(values[0])

                    fig = go.Figure(data=go.Scatterpolar(
                      r=values,
                      theta=categories,
                      fill='toself',
                      line_color='#10B981',
                      fillcolor='rgba(16, 185, 129, 0.3)'
                    ))
                    fig.update_layout(
                      polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color='gray')),
                        angularaxis=dict(tickfont=dict(color='white', size=14))
                      ),
                      showlegend=False,
                      margin=dict(l=40, r=40, t=30, b=30),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_info:
                    st.markdown("### 🧬 Analysis Breakdown")
                    st.info(f"**Market Tier:** {similar}")
                    
                    # Growth Logic
                    diff = val - prev_val
                    pct = (diff / prev_val) * 100 if prev_val > 0 else 0
                    if diff > 0:
                        st.success(f"📈 **Growth:** +€{diff:,.0f} (+{pct:.1f}%)")
                    else:
                        st.error(f"📉 **Decline:** -€{abs(diff):,.0f} ({pct:.1f}%)")
                    
                    st.markdown("#### 🔑 Valuation Drivers")
                    for d in drivers:
                        if "(+)" in d:
                            st.markdown(f'<span class="driver-tag-pos">{d}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="driver-tag-neg">{d}</span>', unsafe_allow_html=True)

            else:
                st.error("Error from backend.")
                
        except Exception as e:
            st.error(f"⚠️ Connection Error! Is backend running? {e}")

else:
    st.info("👈 Enter player stats to generate an AI Valuation Report.")