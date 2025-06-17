import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import os

# Load dataset
df = pd.read_csv("Datasets/crop_production_india.csv")

#  Preprocessing
text_columns = ['State_Name', 'District_Name', 'Crop', 'Season']
for col in text_columns:
    df[col] = df[col].str.strip().str.lower()

# Filter valid values
df = df[(df['Area'] > 0) & (df['Production'] > 0)]

# Define inputs and outputs
features = ['State_Name', 'District_Name', 'Crop', 'Season']
X = df[features]
y_area = df['Area']
y_prod = df['Production']

# One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Train-test split
X_train_area, _, y_train_area, _ = train_test_split(X_encoded, y_area, test_size=0.2, random_state=42)
X_train_prod, _, y_train_prod, _ = train_test_split(X_encoded, y_prod, test_size=0.2, random_state=42)

# Train models
model_area = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model_prod = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

model_area.fit(X_train_area, y_train_area)
model_prod.fit(X_train_prod, y_train_prod)

# Save models and encoder
os.makedirs("models", exist_ok=True)

joblib.dump(encoder, "trained_models/encoder.pkl")
joblib.dump(model_area, "trained_models/area_model.pkl")
joblib.dump(model_prod, "trained_models/production_model.pkl")

print("Models and encoder saved successfully to 'trained_models/' directory.")
