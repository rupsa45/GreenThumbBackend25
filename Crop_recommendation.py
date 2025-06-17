import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib  # for saving .pkl files

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Use selected features
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]
y = df['label']

# Encode crop labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save the trained model and label encoder
joblib.dump(model, "trained_models/crop_recommendation_model.pkl")
joblib.dump(le, "trained_models/label_encoder.pkl")

# Recommend crops function
def recommend_crops(N, P, K, temperature, humidity, rainfall):
    user_input = pd.DataFrame([[N, P, K, temperature, humidity, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'rainfall'])
    probabilities = model.predict_proba(user_input)[0]
    top5_indices = np.argsort(probabilities)[::-1][:5]
    top5_crops = le.inverse_transform(top5_indices)
    return top5_crops

# Run with user input
if __name__ == "__main__":
    try:
        N = int(input("Enter Nitrogen (N): "))
        P = int(input("Enter Phosphorus (P): "))
        K = int(input("Enter Potassium (K): "))
        temp = float(input("Enter Temperature (°C): "))
        hum = float(input("Enter Humidity (%): "))
        rain = float(input("Enter Rainfall (mm): "))

        results = recommend_crops(N, P, K, temp, hum, rain)

        print("\nRecommended Top 5 Crops based on Inputs:")
        for i, crop in enumerate(results, 1):
            print(f"{i}. {crop}")

    except ValueError:
        print("Invalid input. Please enter numeric values.")
