# ⚽ Pro Scout AI: Football Transfer Valuation Engine

**Pro Scout AI** is an advanced Machine Learning application designed to predict football player transfer fees with high accuracy. Unlike traditional black-box models, this system uses a **Hybrid Expert Architecture**—combining a **Log-Transformed Random Forest Regressor** with a **Rule-Based Business Logic Layer**—to provide objective, explainable, and market-aware valuations.

This tool helps scouts, managers, and analysts identify undervalued talent and understand the specific drivers behind a player's price tag (e.g., "Premier League Tax", "Injury Risk", "Goalscorer Premium").

---

## 📂 Project Structure

```text
/
├── Project Report/
│   ├── FINAL PROJECT REPORT REVISED.docx   # Full Technical & Business Report
│   └── Milestone Reports (1-5)             # Development history
│
└── Project source code/
    ├── backend_api.py              # FastAPI Backend (The "Brain" & Logic Layer)
    ├── frontend_dashboard.py       # Streamlit Frontend (The UI & Charts)
    ├── milestone_2_pipeline.py     # Data Cleaning & Feature Engineering Pipeline
    ├── milestone_5_save_model_1812.py # Model Training Script
    ├── data_preparation.ipynb      # Notebook for initial data exploration
    ├── requirements.txt            # List of library dependencies
    └── README.md                   # Project Documentation

```

---

## 🚀 Key Features

### 1. Hybrid Expert Engine

We moved beyond simple AI guessing. This project uses a two-step process:

* **Step 1 (The Math):** A **Random Forest Regressor** predicts the "Base Value" based on statistical patterns from 900,000+ player records.
* **Step 2 (The Expert):** A **Python Logic Layer** adjusts this base value using real-world football rules:
* **Position Multipliers:** Strikers get a +15% premium; Goalkeepers get a -10% discount.
* **Injury Penalties:** Direct value deduction for every day injured.
* **Nationality Hype:** Premium applied to players from Tier 1 nations (Brazil, England, France).



### 2. "Billionaire Bias" Fix

Standard models fail to value low-tier players (e.g., €50k vs €100k) because they are skewed by superstars (Mbappe, Haaland).

* **Solution:** We implemented **Logarithmic Transformation (`np.log1p`)** on the target variable. This ensures the model treats a €10k difference for a rookie as seriously as a €10M difference for a star.

### 3. Professional Dashboard (Full Stack)

* **Interactive UI:** Built with **Streamlit**, featuring a dark-mode professional theme.
* **Radar Charts:** Uses **Plotly** to visualize player strengths (Attacking, Stamina, Availability) on a spider graph.
* **Real-Time Responsiveness:** Changing "Goals" or "Minutes" instantly updates the valuation using our "Performance Impact Calculator."

### 4. Explainable AI (XAI)

The system doesn't just give a number; it explains **WHY**:

* **Valuation Drivers:** Tags like `🔥 Elite Form`, `🏥 Injury Risk`, or `🛡️ Reliable Starter` appear based on the input data.
* **Confidence Intervals:** Displays a Min-Max range (e.g., €18M - €22M) to show statistical certainty.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Machine Learning:** Scikit-Learn (Random Forest, StandardScaler)
* **Backend:** FastAPI, Uvicorn (High-performance API)
* **Frontend:** Streamlit (Reactive Dashboard)
* **Visualization:** Plotly Graph Objects (Interactive Radar Charts)
* **Data Handling:** Pandas, NumPy, Joblib

---

## ⚙️ Installation & Setup

Follow these steps to run the project on your local machine:

### 1. Prerequisites

Ensure you have Python installed. Install the required libraries:

```bash
pip install -r requirements.txt

```

### 2. Train the Model (First Run Only)

Run the training script to generate the model artifact (`.joblib`) from the source code.

```bash
python milestone_5_save_model_1812.py

```

### 3. Start the Backend (The Engine)

Open a terminal inside `Project source code` and run the API server:

```bash
python backend_api.py

```

*You should see: `Uvicorn running on http://127.0.0.1:8000*`

### 4. Start the Frontend (The Dashboard)

Open a **new** terminal window inside `Project source code` and launch the UI:

```bash
streamlit run frontend_dashboard.py

```

*The application will open in your web browser automatically.*

---

## 🎮 How to Use the Demo

1. **Enter Player Stats:** Use the sidebar to select Nationality, Position, Goals, Assists, etc.
2. **Adjust Momentum:** Use the slider to simulate "Hype" (Positive Momentum) or "Decline" (Negative Momentum).
3. **Click "Generate Valuation":**
* The **Estimated Market Value** card will appear.
* A **Radar Chart** will plot the player's profile.
* **"Similar Players"** (e.g., "Similar to: Mbappe") will be suggested.
* **Driver Tags** (Green/Red) will explain the logic behind the price.



---

## 📊 Logic & Business Rules Implemented

| Logic Rule | Implementation | Effect on Value |
| --- | --- | --- |
| **Striker Premium** | `if "Attack": multiplier = 1.15` | **+15% Increase** |
| **Goalkeeper Discount** | `if "Goalkeeper": multiplier = 0.90` | **-10% Decrease** |
| **Injury Penalty** | `days_injured * Tier_Multiplier` | **Direct Deduction** (€) |
| **Goal Bonus** | `goals * Tier_Multiplier` | **Direct Addition** (€) |
| **Nationality Hype** | `if "Brazil" or "England": x1.10` | **+10% Increase** |

---

## 👤 Credits

**Project by:** Sanket Kurve
**Developed for:** Dynamic Football Player Transfer Value Prediction Project
**Special Thanks:** Mentor Farman Sir for guidance on Business Logic validation.

```

```