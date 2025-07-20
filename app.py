import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# Set page config
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Enter the employee details below to predict their salary category.")

# === Native Country Mapping ===
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

# === Sidebar for input ===
st.sidebar.header("🧾 Input Employee Details")

def user_input_features():
    age = st.sidebar.slider("Age", 17, 76, 30)

    workclass = st.sidebar.selectbox(
        "Workclass (Label Encoded)",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            0: "Federal-gov", 1: "Local-gov", 2: "Private", 3: "Self-emp-inc",
            4: "Self-emp-not-inc", 5: "State-gov", 6: "Others"
        }[x]
    )

    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=50000)

    educational_num = st.sidebar.slider("Education Number", 5, 16, 10)

    marital_status = st.sidebar.selectbox(
        "Marital Status (Label Encoded)",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Divorced", 1: "Married", 2: "Never-married"}[x]
    )

    occupation = st.sidebar.selectbox(
        "Occupation (Label Encoded)",
        options=list(range(14)),
        format_func=lambda x: {
            0: "Adm-clerical", 1: "Armed-Forces", 2: "Craft-repair", 3: "Exec-managerial",
            4: "Farming-fishing", 5: "Handlers-cleaners", 6: "Machine-op-inspct",
            7: "Others", 8: "Priv-house-serv", 9: "Prof-specialty", 10: "Protective-serv",
            11: "Sales", 12: "Tech-support", 13: "Transport-moving"
        }[x]
    )

    relationship = st.sidebar.selectbox(
        "Relationship (Label Encoded)",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "Husband", 1: "Not-in-family", 2: "Other-relative", 3: "Own-child", 4: "Unmarried"
        }[x]
    )

    race = st.sidebar.selectbox(
        "Race (Label Encoded)",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "Amer-Indian-Eskimo", 1: "Asian-Pac-Islander", 2: "Black", 3: "Other", 4: "White"
        }[x]
    )

    gender = st.sidebar.selectbox(
        "Gender (Label Encoded)",
        options=[0, 1],
        format_func=lambda x: {0: "Female", 1: "Male"}[x]
    )

    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)

    native_country = st.sidebar.selectbox(
        "Native Country (Label Encoded)",
        options=list(native_country_map.keys()),
        format_func=lambda x: f"{x} = {native_country_map[x]}"
    )

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

# === Get Input ===
input_df = user_input_features()

st.subheader("📋 User Input Summary:")
st.write(input_df)

# === Prediction ===
if st.button("🔍 Predict Salary Category"):
    try:
        prediction = model.predict(input_df)
        result = "Income >50K" if prediction[0] == 1 else "Income <=50K"
        st.success(f"✅ Predicted Salary Category: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# === Label Encoding Reference ===
with st.expander("📌 Label Encoding Reference for Native Country"):
    st.write(pd.DataFrame(list(native_country_map.items()), columns=["Code", "Country"]))
