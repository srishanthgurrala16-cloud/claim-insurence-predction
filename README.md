# ClaimWise Insurance Prediction

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
Raw Dataset
     ↓
Dataset Loading
     ↓
Data Understanding
     ↓
Missing Value Handling
     ↓
Duplicate Removal
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Feature / Target Separation
     ↓
Train-Test Split
     ↓
Preprocessed Dataset
