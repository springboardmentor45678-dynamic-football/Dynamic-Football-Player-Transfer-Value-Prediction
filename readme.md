# Football Player Market Value Prediction ⚽📈

A machine learning project to **predict football players’ market value** using performance, availability, physical attributes, and sentiment-aware features. The project emphasizes **data quality, feature semantics, explainability, and deployment readiness**.

---

## 📌 Project Overview

* Built on a large, well-structured Kaggle football dataset
* Aggregates session-level data into **player-level representations**
* Uses **non-linear models** to capture real-world valuation dynamics
* Focuses on **interpretability** via feature importance and SHAP analysis

---

## 🗂️ Dataset

* Source: [*5.7M+ Records – Most Comprehensive Football Dataset* (Kaggle)
 ](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets/code)
 
* Integrated data:

  * Player performance
  * Injury history
  * Market value
  * Player profiles
  * Sentiment (mentor-provided)
* Final dataset: **clean, player-centric, modeling-ready** 

---

## 🔧 Methodology

### Data Processing

* Aggregation of session-level stats (minutes, goals, cards, injuries)
* Removal of non-informative identifiers and artifacts
* Semantic handling of missing target values
* Pipeline-based preprocessing (imputation, encoding, scaling)

### Modeling

* Baseline: Linear Regression
* Non-linear models:

  * Decision Tree
  * Random Forest
  * LightGBM
  * CatBoost
* Tree-based models achieved **R² ≈ 0.99**, validating feature quality

### Feature Selection

* Cross-model feature importance validation
* Reduced to **16 high-signal features**
* Target variable log-transformed for stability

### Explainability

* SHAP used for:

  * Global feature importance
  * Non-linear dependency analysis

---

## 🚀 Deployment

* **Backend:** FastAPI
* **Frontend:** Streamlit
* Serialized model + preprocessing pipeline for reproducible inference

---

## 📊 Key Takeaways

* Player valuation is **non-linear and interaction-driven**
* Feature quality > feature quantity
* Interpretable ML is feasible without sacrificing performance

---

## 🧠 Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `LightGBM` · `CatBoost` · `SHAP` · `FastAPI` · `Streamlit`

---

