import os
import pandas as pd
import numpy as np

# ============================================================
# CLAIMWISE INSURANCE PREDICTION — STEP 2
# MISSING VALUES & DUPLICATES CLEANING
# Reads : 01_raw_claimwise_50000.csv
# Saves : 02_cleaned_dedup_median_imputed.csv
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Project root: Credit ML
INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "01_raw_claimwise_50000.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Dataset", "02_cleaned_dedup_median_imputed.csv")

if not os.path.exists(INPUT_FILE):
    print("Input dataset file not found at:")
    print(INPUT_FILE)
    raise FileNotFoundError(INPUT_FILE)

print("Input dataset file found:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STEP 2: CLEANING & IMPUTATION (01_raw_claimwise_50000.csv -> 02_cleaned_dedup_median_imputed.csv)")
print("=" * 70)
print(df.head())
print("\nOriginal Dataset Shape:", df.shape)

# ------------------------------------------------------------
# 1. CHECK MISSING VALUES & DUPLICATES
# ------------------------------------------------------------

print("\nMissing Values in Original Dataset:", df.isnull().sum().sum())
print("Duplicate Records in Original Dataset:", df.duplicated().sum())

df_clean = df.copy()

# ------------------------------------------------------------
# 2. DUPLICATE REMOVAL
# ------------------------------------------------------------

before_duplicates = len(df_clean)
df_clean = df_clean.drop_duplicates()
after_duplicates = len(df_clean)

print("\n" + "=" * 70)
print("1. DUPLICATE REMOVAL")
print("=" * 70)
print("Original rows        :", before_duplicates)
print("Rows after deletion  :", after_duplicates)
print("Rows deleted         :", before_duplicates - after_duplicates)

# ------------------------------------------------------------
# 3. NUMERICAL MEDIAN IMPUTATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. MEDIAN IMPUTATION FOR NUMERICAL FEATURES")
print("=" * 70)

numeric_columns = df_clean.select_dtypes(include=np.number).columns.tolist()

for col in numeric_columns:
    if df_clean[col].isnull().any():
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)
        print(f"Median used for '{col}' = {median_value}")

# ------------------------------------------------------------
# 4. CATEGORICAL MODE IMPUTATION & TEXT STRIPPING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. MODE IMPUTATION & TEXT STRIPPING FOR CATEGORICAL FEATURES")
print("=" * 70)

categorical_columns = df_clean.select_dtypes(exclude=np.number).columns.tolist()

for col in categorical_columns:
    # Remove leading and trailing spaces (preserve NaN: don't convert NaN to "nan" string)
    df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    if df_clean[col].isnull().any():
        mode_value = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_value)
        print(f"Mode used for '{col}' = {mode_value}")

# ------------------------------------------------------------
# 5. REMOVE UNNECESSARY COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. REMOVE UNNECESSARY COLUMNS")
print("=" * 70)

# Remove customer/claim ID column because it is only an identifier
if "id" in df_clean.columns:
    df_clean = df_clean.drop(columns=["id"])
    print("Removed column: 'id' (Reason: Unique identifier with no predictive value).")

# ------------------------------------------------------------
# 6. SAVE INTERMEDIATE CLEAN DATASET
# ------------------------------------------------------------

df_clean.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Clean Dataset Saved To  :", OUTPUT_FILE)
print("Final Dataset Shape     :", df_clean.shape)
print("Remaining Missing Values:", df_clean.isnull().sum().sum())
print("Remaining Duplicates    :", df_clean.duplicated().sum())
