import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_models():
    """
    Loads trained ML models and selected feature list
    """
    rf_model = joblib.load(
        os.path.join(MODEL_DIR, "random_forest_final.joblib")
    )

    lgb_model = joblib.load(
        os.path.join(MODEL_DIR, "lightgbm_rf_selected_final.joblib")
    )

    feature_df = pd.read_csv(
        os.path.join(MODEL_DIR, "rf_selected_features.csv")
    )

    selected_features = feature_df["feature"].tolist()

    return rf_model, lgb_model, selected_features
