import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("final_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("HR Employee Attrition Prediction")

# === Input Fields === #
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative"
    ])
    monthly_income = st.number_input("Monthly Income", min_value=1000, step=100)
    work_life_balance = st.selectbox("Work-Life Balance", ["Bad", "Good", "Better", "Best"])
    job_satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
    performance_rating = st.selectbox("Performance Rating", ["Low", "Good", "Excellent", "Outstanding"])
    number_of_promotions = st.slider("Number of Promotions", 0, 10, 1)
    overtime = st.selectbox("Overtime", ["Yes", "No"])

with col2:
    distance_from_home = st.slider("Distance from Home (km)", 0, 60, 10)
    education_level = st.selectbox("Education Level", ["Below College", "College", "Bachelor", "Master", "Doctor"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    number_of_dependents = st.slider("Number of Dependents", 0, 5, 0)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    company_tenure = st.slider("Company Tenure (years)", 0, 40, 5)
    remote_work = st.selectbox("Remote Work", ["Yes", "No"])
    leadership_opportunities = st.selectbox("Leadership Opportunities", ["Yes", "No"])
    innovation_opportunities = st.selectbox("Innovation Opportunities", ["Yes", "No"])
    company_reputation = st.slider("Company Reputation Score", 0, 10, 5)
    employee_recognition = st.selectbox("Employee Recognition", ["Yes", "No"])

# === Encoding Maps (Example mappings, adapt to your training notebook if needed) === #
map_yes_no = {"Yes": 1, "No": 0}
map_gender = {"Male": 1, "Female": 0}
map_job_role = {
    "Sales Executive": 0,
    "Research Scientist": 0.75,
    "Laboratory Technician": 0.5,
    "Manufacturing Director": 1.0,
    "Healthcare Representative": 0.25,
}
map_education = {
    "Below College": 0,
    "College": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctor": 4,
}
map_wlb = {"Bad": 0, "Good": 0.3333, "Better": 0.6666, "Best": 1.0}
map_satisfaction = {"Low": 0.0, "Medium": 0.3333, "High": 0.6666, "Very High": 1.0}
map_perf = {"Low": 0.0, "Good": 0.3333, "Excellent": 0.6666, "Outstanding": 1.0}
map_marital = {"Single": 0, "Married": 1, "Divorced": 2}
map_size = {"Small": 0, "Medium": 1, "Large": 2}

# === Feature Engineering === #
encoded_input = {
    "Age": age / 60,
    "Gender": map_gender[gender],
    "Years_at_Company": years_at_company / 40,
    "Job_Role": map_job_role[job_role],
    "Monthly_Income": monthly_income / 20000,
    "Work-Life_Balance": map_wlb[work_life_balance],
    "Job_Satisfaction": map_satisfaction[job_satisfaction],
    "Performance_Rating": map_perf[performance_rating],
    "Number_of_Promotions": number_of_promotions / 10,
    "Overtime": map_yes_no[overtime],
    "Distance_from_Home": distance_from_home / 60,
    "Education_Level": map_education[education_level] / 4,
    "Marital_Status": map_marital[marital_status] / 2,
    "Number_of_Dependents": number_of_dependents / 5,
    "Job_Level": job_level / 5,
    "Company_Size": map_size[company_size] / 2,
    "Company_Tenure": company_tenure / 40,
    "Remote_Work": map_yes_no[remote_work],
    "Leadership_Opportunities": map_yes_no[leadership_opportunities],
    "Innovation_Opportunities": map_yes_no[innovation_opportunities],
    "Company_Reputation": company_reputation / 10,
    "Employee_Recognition": map_yes_no[employee_recognition],
}

# Derived features
encoded_input["Salary_Performance_Ratio"] = encoded_input["Monthly_Income"] / (encoded_input["Performance_Rating"] + 0.1)
encoded_input["Tenure_Group"] = 1 if years_at_company > 5 else 0
encoded_input["WorkLife_Satisfaction_Score"] = (encoded_input["Work-Life_Balance"] + encoded_input["Job_Satisfaction"]) / 2
encoded_input["Income_JobLevel_Ratio"] = encoded_input["Monthly_Income"] / (encoded_input["Job_Level"] + 0.1)
encoded_input["Attrition_Num"] = 0  # dummy for prediction

input_df = pd.DataFrame([encoded_input])

# Match order
input_df = input_df[model.feature_name_]

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    st.markdown("---")
    st.subheader(f"Attrition Probability: {probability:.2f}%")
    if prediction == 1:
        st.error("⚠️ This employee is **likely to leave** the company.")
    else:
        st.success("✅ This employee is **likely to stay** with the company.")
