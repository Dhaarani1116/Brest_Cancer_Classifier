import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model, scaler, and dataset
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("data.csv")

# Preprocess dataset for graph
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Define selected features
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "symmetry_mean"
]

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
st.title("🔬 Breast Cancer Tumor Prediction")
st.markdown("Enter tumor measurements below to predict whether it's **Benign (0)** or **Malignant (1)**.")

# --- Clear flag logic BEFORE widget creation ---
if "clear" in st.session_state and st.session_state["clear"]:
    for feature in features:
        st.session_state[feature] = 0.0
    st.session_state["clear"] = False

# ----------------- FORM UI -----------------
with st.form("prediction_form"):
    st.subheader("📋 Enter Tumor Feature Values")

    col1, col2 = st.columns(2)
    with col1:
        radius_mean = st.number_input("🔹 Radius Mean", min_value=0.0, format="%.2f", key="radius_mean")
        perimeter_mean = st.number_input("🔹 Perimeter Mean", min_value=0.0, format="%.2f", key="perimeter_mean")
        smoothness_mean = st.number_input("🔹 Smoothness Mean", min_value=0.0, format="%.5f", key="smoothness_mean")
        concavity_mean = st.number_input("🔹 Concavity Mean", min_value=0.0, format="%.5f", key="concavity_mean")

    with col2:
        texture_mean = st.number_input("🔹 Texture Mean", min_value=0.0, format="%.2f", key="texture_mean")
        area_mean = st.number_input("🔹 Area Mean", min_value=0.0, format="%.2f", key="area_mean")
        compactness_mean = st.number_input("🔹 Compactness Mean", min_value=0.0, format="%.5f", key="compactness_mean")
        symmetry_mean = st.number_input("🔹 Symmetry Mean", min_value=0.0, format="%.5f", key="symmetry_mean")

    st.markdown("---")
    col1_btn, col2_btn = st.columns([1, 1])
    predict_btn = col1_btn.form_submit_button("🚀 Predict Tumor Type")
    clear_btn = col2_btn.form_submit_button("🔄 Clear All Fields")

# ----------------- PREDICTION LOGIC -----------------
if predict_btn:
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                            smoothness_mean, compactness_mean, concavity_mean, symmetry_mean]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ The tumor is likely **Malignant**.")
    else:
        st.success("✅ The tumor is likely **Benign**.")

# ----------------- CLEAR LOGIC -----------------
if clear_btn:
    st.session_state["clear"] = True
    st.rerun()

# ----------------- GRAPHS -----------------
with st.expander("📊 Show Data Insights & Graphs"):
    st.subheader("📌 Diagnosis Class Distribution")
    diag_counts = df['diagnosis'].value_counts().rename({0: 'Benign', 1: 'Malignant'})
    fig1, ax1 = plt.subplots()
    sns.barplot(x=diag_counts.index, y=diag_counts.values, palette="Set2", ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_title("Benign vs Malignant")
    st.pyplot(fig1)

    st.subheader("📌 Feature Correlation Heatmap")
    selected_df = df[features + ['diagnosis']]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(selected_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    ax2.set_title("Feature Correlation")
    st.pyplot(fig2)

# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("🧠 Built by Dhaarani • Model: Logistic Regression • Features Used: 8")
