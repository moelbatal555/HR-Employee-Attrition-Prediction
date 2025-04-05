import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('final_model.pkl')

st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("🔍 HR Employee Attrition Prediction")
st.write("Enter employee details to predict the likelihood of attrition 🧑‍💼")

# Form to collect user input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)
        job_role = st.selectbox("Job Role", ["Finance", "Healthcare", "Technology", "Education", "Media"])
        work_life_balance = st.selectbox("Work-Life Balance", ["Poor", "Below Average", "Good", "Excellent"])
        job_satisfaction = st.selectbox("Job Satisfaction", ["Very Low", "Low", "Medium", "High"])

    with col2:
        performance_rating = st.selectbox("Performance Rating", ["Low", "Below Average", "Average", "High"])
        num_promotions = st.number_input("Number of Promotions", min_value=0, max_value=10, value=1)
        distance_from_home = st.number_input("Distance from Home (miles)", min_value=0, value=10)
        education_level = st.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
        job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

    submitted = st.form_submit_button("🔎 Predict")

if submitted:
    input_data = pd.DataFrame.from_dict({
        "Age": [age],
        "Gender": [gender],
        "Years at Company": [years_at_company],
        "Monthly Income": [monthly_income],
        "Job Role": [job_role],
        "Work-Life Balance": [work_life_balance],
        "Job Satisfaction": [job_satisfaction],
        "Performance Rating": [performance_rating],
        "Number of Promotions": [num_promotions],
        "Distance from Home": [distance_from_home],
        "Education Level": [education_level],
        "Marital Status": [marital_status],
        "Job Level": [job_level],
        "Company Size": [company_size],
    })

    # Make prediction and get probability
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of attrition
    percentage = round(prediction_proba * 100, 2)

    if prediction == 1:
        st.warning(f"⚠️ The employee is likely to leave the company. Probability: {percentage}%")
    else:
        st.success(f"✅ The employee is unlikely to leave. Probability: {percentage}%")
