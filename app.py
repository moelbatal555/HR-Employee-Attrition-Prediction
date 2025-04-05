import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="HR Employee Attrition Prediction", layout="wide")
st.title("HR Employee Attrition Prediction Dashboard")

# Load the trained model
model = joblib.load("final_model.pkl")

# Define input form
with st.form("attrition_form"):
    st.header("Enter Employee Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        job_role = st.selectbox("Job Role", ["Education", "Media", "Healthcare", "Technology", "Business", "Engineering", "Human Resources", "Sales", "Other"])
        monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
        work_life_balance = st.selectbox("Work-Life Balance", ["Excellent", "Good", "Fair", "Poor"])
        job_satisfaction = st.selectbox("Job Satisfaction", ["Very High", "High", "Medium", "Low"])
        performance_rating = st.selectbox("Performance Rating", ["Excellent", "High", "Average", "Low"])

    with col2:
        number_of_promotions = st.number_input("Number of Promotions", 0, 10, 0)
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        distance_from_home = st.slider("Distance from Home (km)", 0, 100, 10)
        education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        number_of_dependents = st.slider("Number of Dependents", 0, 10, 0)
        job_level = st.selectbox("Job Level", ["Junior", "Mid", "Senior", "Lead", "Executive"])
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

    with col3:
        company_tenure = st.slider("Company Tenure (Years)", 0, 40, 5)
        remote_work = st.selectbox("Remote Work", ["Yes", "No"])
        leadership_opportunities = st.selectbox("Leadership Opportunities", ["Yes", "No"])
        innovation_opportunities = st.selectbox("Innovation Opportunities", ["Yes", "No"])
        company_reputation = st.selectbox("Company Reputation", ["Excellent", "Good", "Fair", "Poor"])
        employee_recognition = st.selectbox("Employee Recognition", ["High", "Medium", "Low"])

    submitted = st.form_submit_button("Predict Attrition")

# Encoding maps
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
category_maps = {
    "Work-Life Balance": {"Excellent": 1.0, "Good": 0.6667, "Fair": 0.3333, "Poor": 0.0},
    "Job Satisfaction": {"Very High": 1.0, "High": 0.6667, "Medium": 0.3333, "Low": 0.0},
    "Performance Rating": {"Excellent": 1.0, "High": 0.6667, "Average": 0.3333, "Low": 0.0},
    "Job Level": {"Junior": 0.0, "Mid": 0.25, "Senior": 0.5, "Lead": 0.75, "Executive": 1.0},
    "Company Size": {"Small": 0.0, "Medium": 0.5, "Large": 1.0},
    "Company Reputation": {"Excellent": 1.0, "Good": 0.6667, "Fair": 0.3333, "Poor": 0.0},
    "Employee Recognition": {"High": 1.0, "Medium": 0.5, "Low": 0.0},
    "Marital Status": {"Single": 0, "Married": 1, "Divorced": 2},
    "Education Level": {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3, "Other": 4},
    "Job Role": {"Education": 0.0, "Media": 0.25, "Healthcare": 0.5, "Technology": 0.75, "Business": 1.0,
                  "Engineering": 0.125, "Human Resources": 0.375, "Sales": 0.625, "Other": 0.875}
}

if submitted:
    input_dict = {
        "Age": age / 60,
        "Gender": binary_map[gender],
        "Years_at_Company": years_at_company / 40,
        "Job_Role": category_maps["Job Role"][job_role],
        "Monthly_Income": monthly_income / 50000,
        "Work-Life_Balance": category_maps["Work-Life Balance"][work_life_balance],
        "Job_Satisfaction": category_maps["Job Satisfaction"][job_satisfaction],
        "Performance_Rating": category_maps["Performance Rating"][performance_rating],
        "Number_of_Promotions": number_of_promotions / 10,
        "Overtime": binary_map[overtime],
        "Distance_from_Home": distance_from_home / 100,
        "Education_Level": category_maps["Education Level"][education_level],
        "Marital_Status": category_maps["Marital Status"][marital_status],
        "Number_of_Dependents": number_of_dependents / 10,
        "Job_Level": category_maps["Job Level"][job_level],
        "Company_Size": category_maps["Company Size"][company_size],
        "Company_Tenure": company_tenure / 40,
        "Remote_Work": binary_map[remote_work],
        "Leadership_Opportunities": binary_map[leadership_opportunities],
        "Innovation_Opportunities": binary_map[innovation_opportunities],
        "Company_Reputation": category_maps["Company Reputation"][company_reputation],
        "Employee_Recognition": category_maps["Employee Recognition"][employee_recognition]
    }

    # Additional features
    input_dict["Salary_Performance_Ratio"] = input_dict["Monthly_Income"] / (input_dict["Performance_Rating"] + 0.1)
    input_dict["Tenure_Group"] = input_dict["Years_at_Company"] // 0.2
    input_dict["WorkLife_Satisfaction_Score"] = (input_dict["Work-Life_Balance"] + input_dict["Job_Satisfaction"]) / 2
    input_dict["Income_JobLevel_Ratio"] = input_dict["Monthly_Income"] / (input_dict["Job_Level"] + 0.1)
    input_dict["Attrition_Num"] = 0  # placeholder

    input_df = pd.DataFrame([input_dict])

    try:
        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction = "Employee is likely to leave" if prediction_proba > 0.5 else "Employee is likely to stay"
        st.success(f"Prediction: {prediction} ({prediction_proba*100:.2f}%)")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
