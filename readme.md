# 🔬 Breast Cancer Tumor Classification with Logistic Regression

This project is a web-based machine learning app that predicts whether a breast tumor is **Benign (non-cancerous)** or **Malignant (cancerous)** using tumor measurement data. The app is built using **Python**, **scikit-learn**, and **Streamlit**.

---

## 📌 Problem Statement

Can we build a machine learning model that accurately classifies breast tumors based on basic features like radius, texture, and area?

---

## 🧠 Machine Learning Approach

- **Algorithm Used**: Logistic Regression
- **Type**: Supervised Binary Classification
- **Model Evaluation**: Accuracy, Confusion Matrix, Classification Report

---

## 📊 Dataset

- **Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Size**: 569 samples × 32 columns
- **Target Column**: `diagnosis` (B = Benign, M = Malignant → encoded as 0 and 1)

### 📌 Features Used in Model:
- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `symmetry_mean`

---

## 📁 Project Structure

