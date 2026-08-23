import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CLAIMWISE INSURANCE PREDICTION — DATASET EXPLORATION & ANALYSIS
# Module 1: Load and Understand Dataset
# ============================================================

print("=" * 70)
print("1. LOAD DATASET")
print("=" * 70)

BASE_DIR = "/Users/padaltiruvinayak/Desktop/Credit ML"
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "claimwise_50000.csv")

if not os.path.exists(DATASET_PATH):
    print("Dataset file not found:")
    print(DATASET_PATH)
    raise FileNotFoundError(DATASET_PATH)

print("Dataset found:")
print(DATASET_PATH)

try:
    df = pd.read_csv(DATASET_PATH)
    print("ClaimWise dataset loaded successfully!")

    print("\n2. Number of Rows and Columns:")
    print("-----------------------------------")
    print("Rows    :", df.shape[0])
    print("Columns :", df.shape[1])

    print("\n3. Column Names:")
    print("-----------------------------------")
    for i, column in enumerate(df.columns[:30], start=1):
        print(f"{i:2}. {column}")
    print("... (Total", len(df.columns), "columns)")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n" + "=" * 70)
    print("4. FIRST 10 RECORDS")
    print("=" * 70)
    print(df.head(10))

    print("\n" + "=" * 70)
    print("5. LAST 10 RECORDS")
    print("=" * 70)
    print(df.tail(10))

    print("\n" + "=" * 70)
    print("6. DATA TYPES")
    print("=" * 70)
    print(df.dtypes.value_counts())

    print("\nColumn Name                 Data Type")
    print("-" * 45)
    for column in df.columns[:20]:
        print(f"{column:<28} {df[column].dtype}")

    print("\n" + "=" * 70)
    print("7. DATASET INFORMATION")
    print("=" * 70)
    df.info()

    print("\n" + "=" * 70)
    print("8. NUMERICAL COLUMNS")
    print("=" * 70)
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    print(numeric_columns.tolist())

    print("\n" + "=" * 70)
    print("9. CATEGORICAL COLUMNS")
    print("=" * 70)
    categorical_columns = df.select_dtypes(include=["object", "str", "category"]).columns
    print(f"Total Categorical Columns: {len(categorical_columns)}")

    print("\n" + "=" * 70)
    print("10. MISSING VALUES IN EACH COLUMN")
    print("=" * 70)
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")
    print("\nTotal Missing Values:", df.isnull().sum().sum())

    print("\n" + "=" * 70)
    print("11. DUPLICATE RECORDS")
    print("=" * 70)
    print("Number of Duplicate Records:", df.duplicated().sum())

    print("\n" + "=" * 70)
    print("12. STATISTICAL SUMMARY")
    print("=" * 70)
    print(df.describe())

    print("\n" + "=" * 70)
    print("13. TARGET COLUMN OVERVIEW (loss)")
    print("=" * 70)
    if "loss" in df.columns:
        print(df["loss"].describe())

    print("\n" + "=" * 70)
    print("CLAIMWISE DATASET ANALYSIS COMPLETED")
    print("=" * 70)

except Exception as e:
    print("\nERROR:", e)