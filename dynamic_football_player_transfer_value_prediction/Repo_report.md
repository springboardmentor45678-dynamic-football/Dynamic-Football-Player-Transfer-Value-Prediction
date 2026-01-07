
# **TransferIQ – Explainable AI System for Player Transfer Value Prediction**


## **1. Project Motivation (Why this problem matters)**

In modern football, player transfer values are influenced by multiple factors such as:

* On-field performance (goals, minutes played, consistency)
* Age and career stage
* Injury history and availability
* Contract status and club influence

However, **current valuation methods are mostly subjective**, relying on:

* Human scouts
* Media hype
* Manual comparisons

This creates:

* Overpriced players
* Risky transfers
* Financial losses for clubs

### 🎯 **Objective of this project**

To build a **data-driven, explainable AI model** that predicts player market value accurately and can be **used in real decision-making** by football clubs.


## **2. Dataset Overview (What data we used)**

The dataset contains **~50,000 player records** with **160+ features**, including:

### a) Performance Data

* Goals
* Assists
* Minutes played
* Matches played
* Position information

### b) Player Profile Data

* Age (derived)
* Height
* Preferred foot
* Playing position

### c) Contract & Club Data

* Current club ID
* Contract expiry year
* Agent ID
* Loan status

### d) Injury Information

* Days missed
* Games missed
* Injury flag (engineered)

📌 **Target Variable**:
`value` → player transfer market value


## **3. Notebook Flow Explanation (C1 → C4)**

This project was implemented **step-by-step**, not in a single script:

| Notebook | Purpose                  | Explanation                             |
| -------- | ------------------------ | --------------------------------------- |
| **C1**   | Data loading & merging   | Combined data from multiple sources     |
| **C2**   | Cleaning & preprocessing | Removed noise, handled missing values   |
| **C3**   | Model experimentation    | Tested multiple ML approaches           |
| **C4**   | Final pipeline           | Optimized RF + LightGBM with evaluation |

This demonstrates **incremental learning and refinement**, which is important academically.


## **4. Milestone-wise Detailed Explanation**

---

## 🟩 **Milestone 1 – Week 1: Data Collection & Exploration**

### What was done

* Loaded player performance and market value data
* Explored:

  * Data types
  * Missing values
  * Distributions of market value
* Identified that:

  * Market value is **highly skewed**
  * Many text/date columns are not directly usable

### Why this step is important

* Helps understand **data quality**
* Prevents wrong assumptions during modeling
* Identifies features that need engineering

📌 *Business view:*
Clubs also start with raw data but **cannot directly use it** without understanding quality.


## 🟩 **Milestone 2 – Week 2: Data Cleaning & Feature Engineering**

### Cleaning performed

* Removed:

  * ID columns (no predictive value)
  * Text fields (tweets, raw text)
  * Columns with 100% missing values
* Removed rows with missing target values

### Feature engineering (very important)

You created **meaningful football-related features**:

#### 1️⃣ Age

```python
df["age"] = current_year - date_of_birth
```

✔ Represents experience vs decline
✔ Strong factor in transfer value

#### 2️⃣ Injury flag

```python
is_injured = 0 / 1
```

✔ Captures availability risk
✔ Clubs avoid injury-prone players

#### 3️⃣ Goals per match

```python
goals / appearances
```

✔ Measures efficiency, not just totals
✔ Separates consistent players from occasional scorers

📌 *Business relevance:*
These features directly match **how clubs negotiate prices**.


## 🟩 **Milestone 3 – Weeks 3–4: Advanced Feature Engineering & Sentiment Decision**

### Sentiment analysis (important justification)

* Sentiment data was explored
* But finally **not used** because:

  * High sparsity
  * No strong correlation
  * Increased noise

📌 **Very important academic point**

> Removing a feature is also a **data-driven decision**, not a failure.

### Final dataset

* ~48,000 rows
* ~150 clean, meaningful features
* Ready for modeling


## 🟨 **Milestone 4 – Week 5: Model Selection Decision**

### Why NOT LSTM?

Although LSTM was proposed initially:

| Issue                  | Explanation                             |
| ---------------------- | --------------------------------------- |
| Not pure time series   | Player values don’t have long sequences |
| Sparse historical data | Not every player has continuous seasons |
| Poor explainability    | Hard to justify predictions             |

### Why LightGBM instead?

| Reason                  | Benefit                  |
| ----------------------- | ------------------------ |
| Handles non-linear data | Football data is complex |
| Fast training           | Practical for deployment |
| Feature importance      | Explainable              |
| Industry standard       | Used in finance & sports |

📌 *This is a **justified design evolution***.


## 🟩 **Milestone 5 – Week 6: Ensemble Strategy**

Instead of stacking many models, you used a **clean ensemble design**:

### 🔹 Step 1: Random Forest

* Trained on full dataset
* Extracted **feature importance**
* Selected top **32 most influential features**

### 🔹 Step 2: LightGBM

* Trained only on selected features
* Reduced noise
* Improved stability

📌 *Business view:*
This mimics how analysts **filter key indicators before final decision**.


## 🟩 **Milestone 6 – Week 7: Evaluation & Validation**

### Evaluation methodology (VERY IMPORTANT)

You used **three levels of evaluation**:

#### 1️⃣ Cross-Validation

* 5-Fold CV
* Measures training stability

#### 2️⃣ Validation Set

* Used for model selection
* Detects overfitting

#### 3️⃣ Test Set

* Completely unseen data
* Measures real-world performance

### Your actual results (explained)

#### Random Forest

* **Test R² ≈ 0.896**
* Very strong predictive power
* Slight overfitting (train > validation)

#### LightGBM (RF-selected)

* **Test R² ≈ 0.836**
* More stable
* Better generalization

📌 *Interpretation:*
Random Forest is strong, LightGBM is **safer for deployment**.


## 🟩 **Milestone 7 – Week 8: Deployment & Visualization**

### Deployment architecture

* **FastAPI** → prediction backend
* **Streamlit** → user interface
* **Joblib** → model persistence

### Why this matters

* Model is not just academic
* It is **production-ready**
* Can be used by:

  * Scouts
  * Analysts
  * Club management


## **5. Code Explanation (High-Level)**

### Step-by-step flow

1. Load dataset
2. Clean & engineer features
3. Split into train/val/test
4. Impute missing values
5. Train Random Forest
6. Select important features
7. Train LightGBM
8. Evaluate on all datasets
9. Save models
10. Deploy

📌 **This follows the standard ML lifecycle used in industry.**


## **6. Business Justification (How this convinces stakeholders)**

### For clubs

* Avoid overpaying players
* Identify undervalued talent
* Reduce injury risk in transfers

### For analysts

* Explainable feature importance
* Data-backed negotiation tool

### For management

* Financial risk reduction
* Faster decision making


## **7. Final Conclusion (Strong Closing)**

> This project demonstrates a complete, explainable, and deployable AI system for predicting football player transfer values.
> By combining Random Forest feature selection with LightGBM prediction, the model achieves high accuracy while remaining interpretable and business-relevant.