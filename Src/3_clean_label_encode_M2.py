import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# CLAIMWISE INSURANCE PREDICTION — STEP 3
# CATEGORICAL LABEL ENCODING
# Reads : clean_del_median_model_M2.csv
# Saves : clean_label_encode_M2.csv
# ==========================================================

BASE_DIR = "/Users/padaltiruvinayak/Desktop/Credit ML"
INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "clean_del_median_model_M2.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Dataset", "clean_label_encode_M2.csv")

if not os.path.exists(INPUT_FILE):
    print("Input dataset file not found at:")
    print(INPUT_FILE)
    raise FileNotFoundError(INPUT_FILE)

print("Input dataset file found:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)
data = df.copy()

print("=" * 70)
print("STEP 3: LABEL ENCODING (clean_del_median_model_M2.csv -> clean_label_encode_M2.csv)")
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
