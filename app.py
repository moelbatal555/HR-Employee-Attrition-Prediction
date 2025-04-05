import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

model = load_model()

st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("HR Employee Attrition Prediction")

st.markdown("---")

# Input form
with st.form("input_form"):
    st.subheader("Enter Employee Information")
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    monthly_income = st.number_input("Monthly Income", 1000, 100000, 5000)
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    performance_rating = st.slider("Performance Rating", 1, 4, 3)
    num_promotions = st.slider("Number of Promotions", 0, 10, 1)
    distance_from_home = st.slider("Distance from Home (km)", 1, 50, 10)
    education_level = st.slider("Education Level", 1, 5, 3)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_dependents = st.slider("Number of Dependents", 0, 10, 1)
    job_level = st.slider("Job Level", 1, 5, 2)
    company_size = st.slider("Company Size (No. of Employees in 1000s)", 1, 100, 10)
    company_tenure = st.slider("Company Tenure (Years)", 0, 40, 5)
    remote_work = st.selectbox("Remote Work", ["Yes", "No"])
    leadership_opportunities = st.selectbox("Leadership Opportunities", ["Yes", "No"])
    innovation_opportunities = st.selectbox("Innovation Opportunities", ["Yes", "No"])
    company_reputation = st.slider("Company Reputation (1-5)", 1, 5, 3)
    employee_recognition = st.selectbox("Employee Recognition", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Raw input data
    input_dict = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Years_at_Company": years_at_company,
        "Monthly_Income": monthly_income,
        "Job_Role": hash(job_role) % 1000,  # Simplified encoding
        "Work-Life_Balance": work_life_balance,
        "Job_Satisfaction": job_satisfaction,
        "Performance_Rating": performance_rating,
        "Number_of_Promotions": num_promotions,
        "Distance_from_Home": distance_from_home,
        "Education_Level": education_level,
        "Marital_Status": hash(marital_status) % 1000,
        "Number_of_Dependents": num_dependents,
        "Job_Level": job_level,
        "Company_Size": company_size,
        "Company_Tenure": company_tenure,
        "Remote_Work": 1 if remote_work == "Yes" else 0,
        "Leadership_Opportunities": 1 if leadership_opportunities == "Yes" else 0,
        "Innovation_Opportunities": 1 if innovation_opportunities == "Yes" else 0,
        "Company_Reputation": company_reputation,
        "Employee_Recognition": 1 if employee_recognition == "Yes" else 0,
    }

    # Engineered features
    input_dict["Salary_Performance_Ratio"] = monthly_income / (performance_rating + 1e-5)
    input_dict["Tenure_Group"] = 1 if years_at_company < 3 else (2 if years_at_company < 10 else 3)
    input_dict["WorkLife_Satisfaction_Score"] = (work_life_balance + job_satisfaction) / 2
    input_dict["Income_JobLevel_Ratio"] = monthly_income / (job_level + 1e-5)
    input_dict["Attrition_Num"] = 0  # dummy value just for model structure

    input_df = pd.DataFrame([input_dict])

    # Reorder columns to match model input
    input_df = input_df[model.feature_name_]

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]
    percentage = round(prediction_proba * 100, 2)

    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Attrition Likelihood:** {percentage}%")
    st.write(f"**Prediction:** {'Will Leave' if prediction == 1 else 'Will Stay'}")
