import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# Set page config
st.set_page_config(page_title="Employee Salary Classifier", page_icon="💼", layout="centered")

# Native Country Map
native_country_map = {
    0: "Cambodia", 1: "Canada", 2: "China", 3: "Columbia", 4: "Cuba",
    5: "Dominican-Republic", 6: "Ecuador", 7: "El-Salvador", 8: "England",
    9: "France", 10: "Germany", 11: "Greece", 12: "Guatemala", 13: "Haiti",
    14: "Holand-Netherlands", 15: "Honduras", 16: "Hong", 17: "Hungary",
    18: "India", 19: "Iran", 20: "Ireland", 21: "Italy", 22: "Jamaica",
    23: "Japan", 24: "Laos", 25: "Mexico", 26: "Nicaragua", 27: "Outlying-US(Guam-USVI-etc)",
    28: "Peru", 29: "Philippines", 30: "Poland", 31: "Portugal", 32: "Puerto-Rico",
    33: "Scotland", 34: "South", 35: "Taiwan", 36: "Thailand", 37: "Trinadad&Tobago",
    38: "United-States", 39: "Vietnam", 40: "Yugoslavia", 41: "Others"
}

# Title
st.markdown("<h1 style='text-align: center; color: navy;'>💼 Employee Salary Classification App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict whether an employee earns >50K or <=50K based on demographics & job info</h4><br>", unsafe_allow_html=True)

st.sidebar.markdown("## 🔧 Input Employee Details")

def user_input_features():
    st.sidebar.markdown("### 👤 Demographics")
    age = st.sidebar.slider("Age", 17, 76, 30)
    gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: {0: "Female", 1: "Male"}[x])
    race = st.sidebar.selectbox("Race", [0, 1, 2, 3, 4],
                                 format_func=lambda x: {0: "Amer-Indian-Eskimo", 1: "Asian-Pac-Islander", 2: "Black", 3: "Other", 4: "White"}[x])
    relationship = st.sidebar.selectbox("Relationship", [0, 1, 2, 3, 4],
                                        format_func=lambda x: {0: "Husband", 1: "Not-in-family", 2: "Other-relative", 3: "Own-child", 4: "Unmarried"}[x])
    native_country = st.sidebar.selectbox("Native Country", list(native_country_map.keys()),
                                          format_func=lambda x: f"{native_country_map[x]} ({x})")

    st.sidebar.markdown("### 🧑‍💼 Work Info")
    workclass = st.sidebar.selectbox("Workclass", [0, 1, 2, 3, 4, 5, 6],
                                     format_func=lambda x: {
                                         0: "Federal-gov", 1: "Local-gov", 2: "Private", 3: "Self-emp-inc",
                                         4: "Self-emp-not-inc", 5: "State-gov", 6: "Others"
                                     }[x])
    occupation = st.sidebar.selectbox("Occupation", list(range(14)),
                                      format_func=lambda x: {
                                          0: "Adm-clerical", 1: "Armed-Forces", 2: "Craft-repair", 3: "Exec-managerial",
                                          4: "Farming-fishing", 5: "Handlers-cleaners", 6: "Machine-op-inspct",
                                          7: "Others", 8: "Priv-house-serv", 9: "Prof-specialty", 10: "Protective-serv",
                                          11: "Sales", 12: "Tech-support", 13: "Transport-moving"
                                      }[x])
    marital_status = st.sidebar.selectbox("Marital Status", [0, 1, 2],
                                          format_func=lambda x: {0: "Divorced", 1: "Married", 2: "Never-married"}[x])
    educational_num = st.sidebar.slider("Educational Number", 5, 16, 10)
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 1000000, 50000)

    st.sidebar.markdown("### 💰 Financial")
    capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.sidebar.number_input("Capital Loss", 0, 99999, 0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)

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

# Get input
input_df = user_input_features()

# Input display
st.markdown("### 📋 Input Summary")
st.dataframe(input_df, use_container_width=True)

# Predict
if st.button("🔍 Predict Salary"):
    try:
        prediction = model.predict(input_df)
        result = "💰 Income >50K" if prediction[0] == 1 else "🪙 Income <=50K"
        if prediction[0] == 1:
            st.success(f"🎯 **Prediction:** {result}")
        else:
            st.error(f"🎯 **Prediction:** {result}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Country Code Info
with st.expander("🌍 Native Country Code Reference"):
    st.dataframe(pd.DataFrame(native_country_map.items(), columns=["Code", "Country"]))
