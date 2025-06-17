import pandas as pd

# Load dataset
df = pd.read_csv("Datasets/fertilizer_recommendation.csv").dropna()
df.columns = df.columns.str.strip()

# Clean text fields
df['Crop Type'] = df['Crop Type'].str.strip().str.lower()
df['Fertilizer Name'] = df['Fertilizer Name'].str.strip()

# === USER INPUT ===
print("=== Fertilizer Information System ===")
user_crop = input("Enter Crop Type: ").strip().lower()

# Filter rows for that crop
matched = df[df['Crop Type'] == user_crop]

# If no match found
if matched.empty:
    print(f"Error: No data found for crop '{user_crop}'.")
else:
    print(f"\n=== Recommendations for '{user_crop.title()}' ===")
    for i, row in matched.iterrows():
        print(f"\n--- Match {i+1} ---")
        print(f"Fertilizer     : {row['Fertilizer Name']}")
        if row['Nitrogen'] != 0:
            print(f"Nitrogen (N)   : {row['Nitrogen']}")
        if row['Phosphorus'] != 0:
            print(f"Phosphorus (P) : {row['Phosphorus']}")
        if row['Potassium'] != 0:
            print(f"Potassium (K)  : {row['Potassium']}")
        print(f"Soil Moisture  : {row['Soil Moisture']}")
