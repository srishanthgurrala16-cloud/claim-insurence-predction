import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CLAIMWISE INSURANCE PREDICTION — EDA & VISUALIZATION
# Reads preprocessed dataset: 05_preprocessed_final.csv
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Project root: Credit ML
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "05_preprocessed_final.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Outputs", "EDA_Analysis_outputs")

# Verify preprocessed dataset file exists
if not os.path.exists(DATASET_PATH):
    print("Preprocessed dataset file not found at:")
    print(DATASET_PATH)
    raise FileNotFoundError(DATASET_PATH)

print("Preprocessed dataset file found:")
print(DATASET_PATH)

# Automatically create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("CLAIMWISE INSURANCE PREDICTION - EDA ANALYSIS")
print("=" * 70)
print("First Five Records:")
print(df.head())
print("\nDataset Shape :", df.shape)
print("\nData Types Summary:")
print(df.dtypes.value_counts())
print("\nMissing Values Total:", df.isnull().sum().sum())
print("Duplicate Rows      :", df.duplicated().sum())

target = "loss"

# 1. Target Distribution Plot
if target in df.columns:
    print("\nGenerating Target Distribution Plot (01_target_distribution.png)...")
    plt.figure(figsize=(10, 5))
    sns.histplot(df[target], bins=50, kde=True, color="teal")
    plt.title("Distribution of Claim Loss Amount (loss)")
    plt.xlabel("Loss Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "01_target_distribution.png"))
    plt.close()

# 2. Numerical Feature Distributions (Histograms)
continuous_cols = [col for col in df.columns if col.startswith("cont")]
print("\nGenerating Feature Distribution Plots (02_feature_distribution_*.png)...")
for col in continuous_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, color="steelblue")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"02_feature_distribution_{col}.png"))
    plt.close()

# 3. Individual Boxplots for Outlier Detection (One PNG file per feature)
print("\nGenerating Individual Boxplot PNG Images (03_boxplot_*.png)...")
for col in continuous_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col], color="orange")
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"03_boxplot_{col}.png"))
    plt.close()

# 4. Correlation Matrix Heatmap
print("\nGenerating Correlation Heatmap (04_correlation_heatmap.png)...")
numeric_df = df[continuous_cols + [target]] if target in df.columns else df[continuous_cols]
corr = numeric_df.corr()
corr.to_csv(os.path.join(OUTPUT_FOLDER, "Correlation_Matrix.csv"))

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (Continuous Features & Loss)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "04_correlation_heatmap.png"))
plt.close()

# 5. Categorical Count Plots
categorical_columns = [col for col in df.columns if col.startswith("cat")]
print("\nGenerating Categorical Feature Count Plots (05_countplot_*.png)...")
for col in categorical_columns[:5]:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=col, color="teal")
    plt.title(f"Count Plot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"05_countplot_{col}.png"))
    plt.close()

# 6. Missing Value Heatmap
print("\nGenerating Missing Values Heatmap (06_missing_values_heatmap.png)...")
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "06_missing_values_heatmap.png"))
plt.close()

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)
print("All figures are saved in:", OUTPUT_FOLDER)
