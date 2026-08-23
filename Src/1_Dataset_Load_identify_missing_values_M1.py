import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. Load Dataset & Identify Missing Values
# ============================================
# Project: ClaimWise Insurance Prediction
# Module 1: Dataset Exploration and Initial Audit
# Reads: claimwise_50000.csv
# ============================================

BASE_DIR = "/Users/padaltiruvinayak/Desktop/Credit ML"
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "claimwise_50000.csv")

if not os.path.exists(DATASET_PATH):
    print("Dataset file not found at:")
    print(DATASET_PATH)
    raise FileNotFoundError(DATASET_PATH)

print("Dataset file found successfully:")
print(DATASET_PATH)

print("=" * 70)
print("1. LOAD DATASET (claimwise_50000.csv)")
print("=" * 70)

try:
    df = pd.read_csv(DATASET_PATH)

    print("\n-----------------------------------")
    print("1. Dataset Contents (First 5 Rows):")
    print("-----------------------------------")
    print(df.head())

    print("\n-----------------------------------")
    print("2. Number of Rows and Columns:")
    print("-----------------------------------")
    print(df.shape)
    print("Total Rows   :", df.shape[0])
    print("Total Columns:", df.shape[1])

    print("\n-----------------------------------")
    print("3. Column Names:")
    print("-----------------------------------")
    print(df.columns.tolist()[:30], "... (Total", len(df.columns), "columns)")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n-----------------------------------")
    print("4. First 10 Records:")
    print("-----------------------------------")
    print(df.head(10))

    print("\n-----------------------------------")
    print("5. Last 10 Records:")
    print("-----------------------------------")
    print(df.tail(10))

    print("\n-----------------------------------")
    print("6. Data Types Summary:")
    print("-----------------------------------")
    print(df.dtypes.value_counts())

    print("\n" + "=" * 60)
    print("7. Dataset Information:")
    print("=" * 60)
    df.info()

    # Numerical columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    print("\n-----------------------------------")
    print("8. Numerical Columns Count:", len(numeric_df.columns))
    print("-----------------------------------")

    print("\n-----------------------------------")
    print("9. Missing Values in Numerical Columns:")
    print("-----------------------------------")
    print(numeric_df.isnull().sum())
    print("Total Missing Numerical Values:", numeric_df.isnull().sum().sum())

    # Categorical columns
    categorical_df = df.select_dtypes(include=["object", "str", "category"])
    print("\n-----------------------------------")
    print("10. Categorical Columns Count:", len(categorical_df.columns))
    print("-----------------------------------")

    print("\n-----------------------------------")
    print("Missing Values in Categorical Columns:")
    print("-----------------------------------")
    print(categorical_df.isnull().sum())
    print("Total Missing Categorical Values:", categorical_df.isnull().sum().sum())

    print("\n-----------------------------------")
    print("11. Total Missing Values Across Dataset:", df.isnull().sum().sum())
    print("12. Duplicate Records in Dataset        :", df.duplicated().sum())

    print("\n-----------------------------------")
    print("13. Statistical Summary:")
    print("-----------------------------------")
    print(df.describe())

    print("\n" + "=" * 70)
    print("STEP 1: DATASET UNDERSTANDING COMPLETED SUCCESSFULLY")
    print("=" * 70)

except Exception as e:
    print(f"An error occurred: {e}")
