import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================================
# CLAIMWISE INSURANCE PREDICTION — STEP 4
# FEATURE SCALING (STANDARD SCALER)
# Reads : clean_label_encode_M2.csv
# Saves : clean_minmax_stand_norma_M2.csv
# ==========================================================

BASE_DIR = "/Users/padaltiruvinayak/Desktop/Credit ML"
INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "clean_label_encode_M2.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Dataset", "clean_minmax_stand_norma_M2.csv")

if not os.path.exists(INPUT_FILE):
    print("Input dataset file not found at:")
    print(INPUT_FILE)
    raise FileNotFoundError(INPUT_FILE)

print("Input dataset file found:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)
data = df.copy()

print("=" * 70)
print("STEP 4: FEATURE SCALING (clean_label_encode_M2.csv -> clean_minmax_stand_norma_M2.csv)")
print("=" * 70)

# Remove 'id' column if present
if "id" in data.columns:
    data = data.drop(columns=["id"])

continuous_features = [col for col in data.columns if col.startswith("cont")]
target_col = "loss"

print("Continuous Features to Scale:", continuous_features)
print("Target Column              :", target_col)

# Apply StandardScaler
scaler = StandardScaler()
scaled_data = data.copy()
scaled_data[continuous_features] = scaler.fit_transform(data[continuous_features])

# Save output
scaled_data.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Scaled Dataset Saved To :", OUTPUT_FILE)
print("Final Scaled Data Shape :", scaled_data.shape)
