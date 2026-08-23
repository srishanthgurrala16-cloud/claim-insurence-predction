import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================
# CLAIMWISE INSURANCE PREDICTION — STEP 5
# FINAL PREPROCESSING & TRAIN-TEST SPLIT
# Reads : clean_minmax_stand_norma_M2.csv
# Saves : claimwise_preprocessed.csv
#         X_train.csv, X_test.csv
#         y_train.csv, y_test.csv
# ============================================

BASE_DIR = "/Users/padaltiruvinayak/Desktop/Credit ML"
INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "clean_minmax_stand_norma_M2.csv")

PREPROCESSED_DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "claimwise_preprocessed.csv")
X_TRAIN_PATH = os.path.join(BASE_DIR, "Dataset", "X_train.csv")
X_TEST_PATH = os.path.join(BASE_DIR, "Dataset", "X_test.csv")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "Dataset", "y_train.csv")
Y_TEST_PATH = os.path.join(BASE_DIR, "Dataset", "y_test.csv")

if not os.path.exists(INPUT_FILE):
    print("Input dataset file not found at:")
    print(INPUT_FILE)
    raise FileNotFoundError(INPUT_FILE)

print("Input dataset file found:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)
processed_df = df.copy()

print("=" * 70)
print("STEP 5: FINAL PREPROCESSING & SPLITTING")
print("=" * 70)
print("Input Scaled Dataset Shape:", processed_df.shape)

# --------------------------------------------
# 1. Remove ID Column (if still present)
# --------------------------------------------
if "id" in processed_df.columns:
    processed_df.drop(columns=["id"], inplace=True)

# --------------------------------------------
# 2. Separate Features & Target
# --------------------------------------------
target_col = "loss"
X = processed_df.drop(columns=[target_col])
y = processed_df[target_col]

print(f"\nFeature Matrix (X) Shape: {X.shape}")
print(f"Target Vector (y) Shape  : {y.shape}")

# --------------------------------------------
# 3. Train-Test Split (80% Train, 20% Test)
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training Features (X_train) Shape: {X_train.shape}")
print(f"Testing Features  (X_test)  Shape: {X_test.shape}")
print(f"Training Target   (y_train) Shape: {y_train.shape}")
print(f"Testing Target    (y_test)  Shape: {y_test.shape}")

# --------------------------------------------
# 4. Save Final Datasets
# --------------------------------------------
processed_df.to_csv(PREPROCESSED_DATASET_PATH, index=False)
X_train.to_csv(X_TRAIN_PATH, index=False)
X_test.to_csv(X_TEST_PATH, index=False)
y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Saved Preprocessed Dataset:", PREPROCESSED_DATASET_PATH)
print("Saved X_train              :", X_TRAIN_PATH)
print("Saved X_test               :", X_TEST_PATH)
print("Saved y_train              :", Y_TRAIN_PATH)
print("Saved y_test               :", Y_TEST_PATH)
