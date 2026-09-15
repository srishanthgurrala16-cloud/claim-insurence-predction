# ClaimWise Insurance Prediction
> **Folder:** `Credit ML` (legacy name on laptop) → project is **ClaimWise Insurance**, not credit. Path: `/Users/padaltiruvinayak/Desktop/Credit ML`

## 📌 Project Overview

**ClaimWise Insurance Prediction** is a Machine Learning project that analyzes historical insurance claim data and prepares it for predicting insurance claim loss.

The project focuses on understanding, cleaning, preprocessing, encoding, scaling, and preparing insurance data for Machine Learning models.

---

## 🎯 Objectives

- Analyze insurance claim data.
- Handle missing values.
- Remove duplicate records.
- Encode categorical features.
- Scale numerical features.
- Prepare features for Machine Learning.
- Split the dataset into training and testing data.
- Predict insurance claim loss using Machine Learning.

---

## 📊 Dataset Information

The project uses a dataset containing **50,000 insurance records**.

| Information | Value |
|---|---:|
| Total Records | 50,000 |
| Original Columns | 132 |
| Final Columns | 131 |
| Removed Column | `id` |
| Input Features | 130 |
| Training Records | 40,000 |
| Testing Records | 10,000 |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Target Variable | `loss` |

---

## 🔄 Preprocessing Pipeline

The preprocessing workflow follows these steps:

```text
01_raw_claimwise_50000.csv (132 cols: id +116 cat +14 cont + loss) — RAW ingestion
     ↓  dedup + median/mode impute + strip + drop id
02_cleaned_dedup_median_imputed.csv (131 cols) — CLEANED
     ↓  LabelEncoder (116 cats → ints) — main branch
03_encoded_label.csv (131 cols) — ENCODED (main)
     ↘  03b_encoded_onehot_alternative.csv (1045 cols) — One-Hot alternative (1031 dummies +14 cont, id dropped, mean impute)
     ↓  StandardScaler (cont1..14 only, loss untouched)
04_scaled_standardized.csv (131 cols) — SCALED
     ↓  declare final
05_preprocessed_final.csv (131 cols) — FINAL pre-split
     ↓  train_test_split 80/20 seed 42
06_split_X_train.csv (40000×130) + 06_split_y_train.csv (40000×1) — TRAIN
06_split_X_test.csv  (10000×130) + 06_split_y_test.csv  (10000×1) — TEST
```

> **Legacy names** (confusing M1/M2) kept as symlinks for backward compat: `claimwise_50000.csv → 01_raw...`, `clean_del_median_model_M2.csv → 02...`, etc. See `reports/DATASET_LIFECYCLE.xlsx` and `reports/Credit_ML_Audit_Report.md` for full mapping.

---

## 📊 Dataset Lifecycle Mapping (New Conceptual Names)

| Stage | File (Conceptual) | Concept / Life | Old Name |
|-------|-------------------|----------------|----------|
| 1 RAW | `01_raw_claimwise_50000.csv` | Original 50k insurance claims | `claimwise_50000.csv` |
| 2 CLEANED | `02_cleaned_dedup_median_imputed.csv` | Deduplicated, median/mode imputed, id removed | `clean_del_median_model_M2.csv` |
| 3 ENCODED | `03_encoded_label.csv` | Label-encoded 116 cats → ints | `clean_label_encode_M2.csv` |
| 3b ALT | `03b_encoded_onehot_alternative.csv` | One-hot alternative (mean impute, 1045 cols) | `clean_one_hot_encoding_M2.csv` |
| 4 SCALED | `04_scaled_standardized.csv` | Standardized cont1..14 | `clean_minmax_stand_norma_M2.csv` |
| 5 FINAL | `05_preprocessed_final.csv` | Final before split | `claimwise_preprocessed.csv` |
| 6 SPLIT | `06_split_X_train.csv` / `X_test` / `y_train` / `y_test` | Train/test 40k/10k | `X_train.csv` etc. |

Run pipeline: `Credit ML/.venv/bin/python Src/claimwise_preprocessing_pipeline.py` or `python main.py`

---

## ⚠️ Audit 2026-09-15 — Errors Fixed

- Fixed `claimwise_preprocessing_pipeline.py` `BASE_DIR="/"` → portable, corrected `../Dataset` read-only error
- Fixed `main.py` not found + hardcoded `python3` → uses `Src/` + `sys.executable`
- Fixed categorical `astype(str).str.strip()` bug swallowing NaNs → `apply(lambda ...)`
- Implemented truncated `clean_target_encode_M2.py` (target encoding alternative)
- Rewrote `clean_one_hot_encod_M2.py`: dropped `id`, fixed “Placement” comment, made portable, 1046→1045 cols
- Renamed all datasets to lifecycle concept names, kept symlinks
- Repaired `.venv` pip cross-venv corruption, installed pandas/sklearn/matplotlib/seaborn/Flask
- Validated accuracy: Ridge R² 0.483 RMSE 2050, RF R² 0.447 — preprocessing accurate

See `reports/Credit_ML_Audit_Report.md` (full audit) and `reports/DATASET_LIFECYCLE.xlsx` (Excel mapping).
