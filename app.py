import streamlit as st
import pandas as pd
import joblib

# Load the trained model (Pipeline is recommended)
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# User Inputs
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Construct DataFrame from input
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Check feature compatibility
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Predict button
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")
    except ValueError as e:
        st.error("❌ Prediction failed due to input mismatch.")
        st.code(str(e))
        st.warning("Check if input column names and types match model training data.")

# Batch Prediction Section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.write(batch_data.head())

        # Align columns if model has feature names
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            batch_data = batch_data.reindex(columns=expected_features, fill_value=0)

        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds

        st.success("✅ Batch Prediction Completed:")
        st.write(batch_data.head())

        # Download option
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

    except Exception as e:
        st.error("❌ Batch prediction failed.")
        st.code(str(e))
