import streamlit as st

import pickle
import pandas as pd

# Load model and encoders once
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)



st.title("💰 PaySignal")
st.write("Know your worth in the AI job market")

experience = st.selectbox(
    "Experience Level",
    ["EN", "MI", "SE", "EX"]
)

st.write("You selected:", experience)



work_year = st.selectbox(
    "Work Year",
    [2024, 2023, 2022, 2021, 2020]
)

job_title = st.selectbox(
    "Job Title",
    ["Data Engineer", "Data Scientist", "ML Engineer", 
     "Analytics Engineer", "Data Analyst", "AI Engineer"]
)

remote_ratio = st.selectbox(
    "Remote Work",
    [0, 50, 100],
    format_func=lambda x: {0: "Onsite", 50: "Hybrid", 100: "Remote"}[x]
)

company_location = st.selectbox(
    "Company Location",
    ["US", "GB", "IN", "CA", "DE", "FR", "ES"]
)

company_size = st.selectbox(
    "Company Size",
    ["S", "M", "L"],
    format_func=lambda x: {
        "S": "Small (startup)", 
        "M": "Medium", 
        "L": "Large (FAANG/enterprise)"
    }[x]
)

if st.button("Predict My Salary"):
    
    # Encode inputs
    exp  = encoders['experience_level'].transform([experience])[0]
    job  = encoders['job_title'].transform([job_title])[0]
    loc  = encoders['company_location'].transform([company_location])[0]
    size = encoders['company_size'].transform([company_size])[0]

    # Build input
    sample_df = pd.DataFrame([{
        'work_year': work_year,
        'experience_level': exp,
        'job_title': job,
        'remote_ratio': remote_ratio,
        'company_location': loc,
        'company_size': size
    }])

    # Predict
    predicted = model.predict(sample_df)[0]

    low  = predicted * 0.85
    high = predicted * 1.15
    if remote_ratio == 100 and company_location == "US":
        st.info("🌍 Remote US roles pay 40-60% more than same roles in other locations")
    
    if experience == "EN" and company_location == "IN":
        st.warning("⚠️ Entry level salaries in India are significantly impacted by AI automation in 2024-2025")

    if experience == "EX" or experience == "SE":
        st.info("🤖 Senior+ engineers with AI skills earn 14-18% more than peers without")

    st.success(f"Estimated Salary Range: ${low:,.0f} — ${high:,.0f}")
    st.write(f"Market midpoint: ${predicted:,.0f}")

