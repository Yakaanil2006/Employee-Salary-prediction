import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

# Set Streamlit page config
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")

# Sidebar for input
st.sidebar.header("Enter Employee Details")

def user_input_features():
    age = st.sidebar.slider("Age", 17, 76, 30)
    workclass = st.sidebar.selectbox("Workclass", [0, 1, 2, 3, 4, 5, 6])
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=50000)
    educational_num = st.sidebar.slider("Education Number", 5, 16, 10)
    marital_status = st.sidebar.selectbox("Marital Status", [0, 1, 2])
    occupation = st.sidebar.selectbox("Occupation", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    relationship = st.sidebar.selectbox("Relationship", [0, 1, 2, 3, 4])
    race = st.sidebar.selectbox("Race", [0, 1, 2, 3, 4])
    gender = st.sidebar.selectbox("Gender", [0, 1])  # 0: Female, 1: Male
    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
    native_country = st.sidebar.selectbox("Native Country", list(range(0, 42)))  # Adjust based on your encoding

    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

st.subheader("📊 User Input:")
st.write(input_df)

# Make prediction
if st.button("Predict Salary Category"):
    try:
        prediction = model.predict(input_df)
        result = "Income >50K" if prediction[0] == 1 else "Income <=50K"
        st.success(f"💰 Predicted Salary Category: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
