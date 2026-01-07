import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

print("--- STARTING MILESTONE 2: PREPROCESSING PIPELINE ---")

df = pd.read_csv('5_master_training_set.csv')

X = df.drop(columns=['market_value', 'player_id', 'player_name'])
y = df['market_value']

print(f"Data Loaded. Features: {X.shape[1]} columns. Rows: {X.shape[0]}")

numeric_features = ['goals', 'assists', 'minutes_played', 'yellow_cards', 'total_days_injured', 'sentiment']
categorical_features = ['position', 'country_of_birth']

numeric_pipe = make_pipeline(
    SimpleImputer(strategy='constant', fill_value=0), 
    StandardScaler() 
)

categorical_pipe = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='Unknown'),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse=False helps us see the data easier
)

def extract_date_features(df_date):
    dt = pd.to_datetime(df_date.iloc[:, 0])
    return pd.DataFrame({
        'year': dt.dt.year,
        'month': dt.dt.month
    })

date_pipe = make_pipeline(
    FunctionTransformer(extract_date_features, validate=False),
    StandardScaler()
)

preprocessor = make_column_transformer(
    (numeric_pipe, numeric_features),
    (categorical_pipe, categorical_features),
    (date_pipe, ['date']),
    remainder='drop' 
)

print("Running Preprocessing Pipeline...")
X_processed = preprocessor.fit_transform(X)

new_columns = (numeric_features + 
               list(preprocessor.named_transformers_['pipeline-2'].named_steps['onehotencoder'].get_feature_names_out()) + 
               ['year', 'month'])

df_processed = pd.DataFrame(X_processed, columns=new_columns)

df_processed['market_value'] = y.values

df_processed.to_csv('6_clean_training_data_v1.csv', index=False)
print(f"-> DONE. Saved '6_clean_training_data_v1.csv'. This is ready for AI training.")
print("First 5 rows of clean data:")
print(df_processed.head())

