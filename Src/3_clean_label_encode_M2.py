import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# CLAIMWISE INSURANCE PREDICTION — STEP 3
# CATEGORICAL LABEL ENCODING
# Reads : 02_cleaned_dedup_median_imputed.csv
# Saves : 03_encoded_label.csv
# ==========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Project root: Credit ML
INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "02_cleaned_dedup_median_imputed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Dataset", "03_encoded_label.csv")

if not os.path.exists(INPUT_FILE):
    print("Input dataset file not found at:")
    print(INPUT_FILE)
    raise FileNotFoundError(INPUT_FILE)

print("Input dataset file found:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)
data = df.copy()

print("=" * 70)
print("STEP 3: LABEL ENCODING (02_cleaned_dedup_median_imputed.csv -> 03_encoded_label.csv)")
print("=" * 70)
print(data.head())
print("Dataset Shape:", data.shape)

# Identify categorical features
categorical_cols = data.select_dtypes(include=["object", "str", "category"]).columns.tolist()

print(f"\nFound {len(categorical_cols)} categorical text columns for Label Encoding.")

# Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("Label Encoding completed successfully for all categorical columns.")

# Save result
data.to_csv(OUTPUT_FILE, index=False)

print("=" * 70)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Label Encoded Dataset Saved To:", OUTPUT_FILE)
print("Final Encoded Dataset Shape   :", data.shape)
