# ClaimWise Insurance Prediction

A machine learning-based insurance analytics system for predicting insurance claim loss using customer, policy, claim, coverage, and other insurance-related information.

---

## 📌 Project Overview

**ClaimWise Insurance Prediction** is a machine learning project developed to analyze insurance-related data and prepare it for predicting insurance claim loss.

The project follows a structured machine learning workflow that includes data understanding, data cleaning, preprocessing, feature preparation, exploratory data analysis, train-test splitting, model development, model evaluation, and web application integration.

The system works with insurance-related information such as customer details, policy information, claim information, coverage details, and other relevant attributes.

The primary prediction target of the project is the **`loss`** variable.

---

## 🎯 Objectives

The main objectives of ClaimWise Insurance Prediction are:

- Analyze insurance-related data.
- Understand the structure and characteristics of the dataset.
- Identify and handle missing values.
- Remove duplicate records.
- Encode categorical variables.
- Scale and normalize numerical features.
- Prepare clean and consistent data for machine learning.
- Separate input features from the target variable.
- Split the dataset into training and testing datasets.
- Perform exploratory data analysis.
- Develop suitable regression models.
- Evaluate and compare model performance.
- Provide a professional web-based interface.
- Integrate the trained machine learning model for future real-time prediction.

---

## 🧠 Machine Learning Problem

### Problem Type

**Regression**

### Target Variable

`loss`

The target variable `loss` represents the insurance claim loss that the machine learning system is designed to predict.

### Input Features

After preprocessing, the dataset contains:

- **130 input features**
- **1 target variable**
- **Target:** `loss`

The `id` column is removed because it is an identifier and does not provide meaningful predictive information.

# claim-insurence-predction

---

## 📊 Dataset Information

The project uses an insurance dataset containing **50,000 records** with **132 original columns**.

| Dataset Property | Value |
|---|---:|
| Total Records | 50,000 |
| Original Columns | 132 |
| Final Records | 50,000 |
| Final Columns | 131 |
| Input Features | 130 |
| Target Variable | `loss` |
| Removed Column | `id` |
| Training Records | 40,000 |
| Testing Records | 10,000 |
| Missing Values After Preprocessing | 0 |
| Duplicate Records After Preprocessing | 0 |

---

## 🔄 Machine Learning Workflow

```text
Raw Insurance Dataset
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Missing Value Handling
        ↓
Categorical Feature Encoding
        ↓
Feature Scaling
        ↓
Feature Normalization
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
Exploratory Data Analysis
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Claim Loss Prediction
        ↓
Web Application
