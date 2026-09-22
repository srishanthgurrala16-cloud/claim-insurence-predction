import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CLAIMWISE INSURANCE PREDICTION
# CORRELATION MATRIX, HEATMAP, AND INDIVIDUAL BOXPLOTS
# Module: Correlation_Matrix_heatmap_boxplots_M1.py
# Reads preprocessed dataset: 05_preprocessed_final.csv
# Output Folder: Outputs/Boxplots_correlation/
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Project root: Credit ML
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "05_preprocessed_final.csv")
BOXPLOT_OUTPUT_FOLDER = os.path.join(BASE_DIR, "Outputs", "Boxplots_correlation")

# Verify preprocessed dataset file exists
if not os.path.exists(DATASET_PATH):
    print("Preprocessed dataset file not found at:")
    print(DATASET_PATH)
    raise FileNotFoundError(DATASET_PATH)

print("Preprocessed dataset found:")
print(DATASET_PATH)

# Automatically create the Boxplots_correlation output directory
os.makedirs(BOXPLOT_OUTPUT_FOLDER, exist_ok=True)

print("=" * 70)
print("1. LOAD PREPROCESSED DATASET FOR BOXPLOTS & CORRELATION ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

# Select continuous numerical features (cont1 to cont14) and target (loss)
continuous_cols = [col for col in df.columns if col.startswith("cont")]
target_col = "loss"

numeric_cols = continuous_cols + ([target_col] if target_col in df.columns else [])
numeric_df = df[numeric_cols]

print("Numerical features for boxplots and correlation:")
print(numeric_df.columns.tolist())

# ------------------------------------------------------------
# 2. GENERATE CORRELATION MATRIX & HEATMAP
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. GENERATE CORRELATION MATRIX & HEATMAP")
print("=" * 70)

corr_matrix = numeric_df.corr()

# Save Correlation Matrix CSV inside Boxplots_correlation folder
corr_csv_path = os.path.join(BOXPLOT_OUTPUT_FOLDER, "Correlation_Matrix.csv")
corr_matrix.to_csv(corr_csv_path)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (Numerical Features & Loss)")
plt.tight_layout()

heatmap_path = os.path.join(BOXPLOT_OUTPUT_FOLDER, "correlation_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

print("Correlation Matrix saved to :", corr_csv_path)
print("Correlation Heatmap saved to:", heatmap_path)

# ------------------------------------------------------------
# 3. GENERATE INDIVIDUAL BOXPLOTS FOR EACH NUMERICAL FEATURE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. GENERATE INDIVIDUAL FEATURE BOXPLOTS")
print("=" * 70)

# Generate one boxplot PNG file for each continuous numerical feature
for column in continuous_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column], color="orange")
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.tight_layout()
    
    boxplot_path = os.path.join(BOXPLOT_OUTPUT_FOLDER, f"boxplot_{column}.png")
    plt.savefig(boxplot_path)
    plt.close()
    
    print(f"Saved boxplot for '{column}' -> {boxplot_path}")

print("\n" + "=" * 70)
print("BOXPLOT AND CORRELATION ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)
print("All generated files saved in:", BOXPLOT_OUTPUT_FOLDER)
