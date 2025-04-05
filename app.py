import streamlit as st
import pandas as pd
import joblib

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

model = load_model()

# Page config
st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("🔍 HR Employee Attrition Prediction")
st.write("Enter employee details to predict the likelihood of attrition.")

# Input form
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

    encoding_maps = {
        "Gender": {"Male": 1, "Female": 0},
        "Job Role": {"Finance": 0, "Healthcare": 1, "Technology": 2, "Education": 3, "Media": 4},
        "Work-Life Balance": {"Poor": 0, "Below Average": 1, "Good": 2, "Excellent": 3},
        "Job Satisfaction": {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3},
        "Performance Rating": {"Low": 0, "Below Average": 1, "Average": 2, "High": 3},
        "Education Level": {
            "High School": 0,
            "Associate Degree": 1,
            "Bachelor’s Degree": 2,
            "Master’s Degree": 3,
            "PhD": 4
        },
        "Marital Status": {"Divorced": 0, "Married": 1, "Single": 2},
        "Job Level": {"Entry": 0, "Mid": 1, "Senior": 2},
        "Company Size": {"Small": 0, "Medium": 1, "Large": 2},
    }

    for col, mapping in encoding_maps.items():
        input_data[col] = input_data[col].map(mapping)

    feature_order = [
        'Age', 'Gender', 'Years at Company', 'Monthly Income', 'Job Role',
        'Work-Life Balance', 'Job Satisfaction', 'Performance Rating',
        'Number of Promotions', 'Distance from Home', 'Education Level',
        'Marital Status', 'Job Level', 'Company Size'
    ]
    input_data = input_data[feature_order].astype(float)

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    percentage = round(prediction_proba * 100, 2)

    if prediction == 1:
        st.warning(f"⚠️ The employee is likely to leave. Probability: {percentage}%")
    else:
        st.success(f"✅ The employee is unlikely to leave. Probability: {percentage}%")
