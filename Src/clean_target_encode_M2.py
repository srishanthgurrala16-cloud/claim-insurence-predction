import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# CLAIMWISE INSURANCE PREDICTION — TARGET ENCODING (ALTERNATIVE)
# Reads : 01_raw_claimwise_50000.csv or 02_cleaned_dedup_median_imputed.csv
# Saves : Dataset/target_encoded.csv (if used)
# Note : This was previously incomplete (5 lines). Implemented as alternative
#        encoding for high-cardinality cats, not in main pipeline.
#        Main pipeline uses label encoding (03_encoded_label.csv).
# ==========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "02_cleaned_dedup_median_imputed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Dataset", "03c_target_encoded_alternative.csv")

if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "01_raw_claimwise_50000.csv")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

print(f"Reading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# Drop id if exists
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Simple target encoding example: mean of loss per category
target = "loss"
if target in df.columns:
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"Target encoding {len(cat_cols)} categorical columns using mean of '{target}'")
    for col in cat_cols:
        means = df.groupby(col)[target].mean()
        df[col] = df[col].map(means)
        print(f"  - {col}: {len(means)} categories mapped")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved target-encoded dataset to: {OUTPUT_FILE}")
else:
    print("Target column 'loss' not found, skipping target encoding.")
