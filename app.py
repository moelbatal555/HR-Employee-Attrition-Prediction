import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load("final_model_joblib.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("HR Employee Attrition Prediction Dashboard")

# Sidebar inputs
st.sidebar.header("Employee Details")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 60, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
    job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative"])
    monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    work_life_balance = st.sidebar.selectbox("Work-Life Balance", ["Bad", "Good", "Better", "Best"])
    job_satisfaction = st.sidebar.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
    performance_rating = st.sidebar.selectbox("Performance Rating", ["Low", "Good", "Excellent", "Outstanding"])
    number_of_promotions = st.sidebar.slider("Number of Promotions", 0, 10, 1)
    overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
    distance_from_home = st.sidebar.slider("Distance from Home", 1, 50, 10)
    education_level = st.sidebar.selectbox("Education Level", ["Below College", "College", "Bachelor", "Master", "Doctor"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    number_of_dependents = st.sidebar.slider("Number of Dependents", 0, 10, 2)
    job_level = st.sidebar.slider("Job Level", 1, 5, 2)
    company_size = st.sidebar.selectbox("Company Size", ["Small", "Medium", "Large"])
    company_tenure = st.sidebar.slider("Company Tenure", 0, 40, 5)
    remote_work = st.sidebar.selectbox("Remote Work", ["Yes", "No"])
    leadership_opportunities = st.sidebar.selectbox("Leadership Opportunities", ["Yes", "No"])
    innovation_opportunities = st.sidebar.selectbox("Innovation Opportunities", ["Yes", "No"])
    company_reputation = st.sidebar.slider("Company Reputation (0-10)", 0, 10, 5)
    employee_recognition = st.sidebar.selectbox("Employee Recognition", ["Yes", "No"])
    salary_perf_ratio = monthly_income / (performance_rating.index(performance_rating) + 1)
    tenure_group = "Low" if years_at_company < 3 else ("Medium" if years_at_company < 7 else "High")
    worklife_score = work_life_balance.index(work_life_balance) + job_satisfaction.index(job_satisfaction)
    income_joblevel_ratio = monthly_income / job_level

    data = {
        'Age': [age],
        'Gender': [1 if gender == "Male" else 0],
        'Years_at_Company': [years_at_company],
        'Job_Role': [job_role],
        'Monthly_Income': [monthly_income],
        'Work-Life_Balance': [work_life_balance],
        'Job_Satisfaction': [job_satisfaction],
        'Performance_Rating': [performance_rating],
        'Number_of_Promotions': [number_of_promotions],
        'Overtime': [1 if overtime == "Yes" else 0],
        'Distance_from_Home': [distance_from_home],
        'Education_Level': [education_level],
        'Marital_Status': [marital_status],
        'Number_of_Dependents': [number_of_dependents],
        'Job_Level': [job_level],
        'Company_Size': [company_size],
        'Company_Tenure': [company_tenure],
        'Remote_Work': [1 if remote_work == "Yes" else 0],
        'Leadership_Opportunities': [1 if leadership_opportunities == "Yes" else 0],
        'Innovation_Opportunities': [1 if innovation_opportunities == "Yes" else 0],
        'Company_Reputation': [company_reputation],
        'Employee_Recognition': [1 if employee_recognition == "Yes" else 0],
        'Salary_Performance_Ratio': [salary_perf_ratio],
        'Tenure_Group': [tenure_group],
        'WorkLife_Satisfaction_Score': [worklife_score],
        'Income_JobLevel_Ratio': [income_joblevel_ratio],
        'Attrition_Num': [0]  # Placeholder, not used for prediction
    }
    return pd.DataFrame(data)

input_df = user_input_features()

# Prediction
input_df = input_df[model.feature_name_]
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

# Display
st.subheader("Prediction")
attrition_result = "Yes" if prediction == 1 else "No"
st.write(f"Attrition: **{attrition_result}**")
st.write(f"Probability of Leaving: **{round(prediction_proba * 100, 2)}%**")
