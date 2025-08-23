import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("credit_model_pipeline (1).pkl")

st.title("Credit Rating Prediction App")
st.write("Enter customer details to predict their credit rating category.")

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", value=1000)
day = st.number_input("Day of Contact", min_value=1, max_value=31, value=5)
duration = st.number_input("Call Duration (seconds)", value=120)
campaign = st.number_input("Campaign Contacts", value=1)
pdays = st.number_input("Days Since Last Contact", value=-1)
previous = st.number_input("Previous Contacts", value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

job = st.selectbox("Job", [
    "blue-collar", "entrepreneur", "housemaid", "management", 
    "retired", "self-employed", "services", "student", 
    "technician", "unemployed", "unknown"
])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Default", ["yes", "no"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
contact = st.selectbox("Contact", ["telephone", "cellular", "unknown"])
month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
poutcome = st.selectbox("Previous Outcome", ["success", "failure", "other", "unknown"])
y = st.selectbox("Subscribed (y)", ["yes", "no"])

# Create dataframe for raw input
input_data = pd.DataFrame([{
    "age": age,
    "balance": balance,
    "day": day,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "credit_score": credit_score,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "poutcome": poutcome,
    "y": y
}])

# Predict
if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0]

    st.success(f"Predicted Credit Rating: {prediction}")

    # Show probabilities per class
    st.write("### Prediction Probabilities:")
    st.write({cls: round(p, 3) for cls, p in zip(pipeline.classes_, proba)})

# Feature importance visualization
if st.checkbox("Show Feature Importance (Random Forest)"):
    importances = pipeline.named_steps["classifier"].feature_importances_
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:10]

    st.write("### Top 10 Important Features")
    st.bar_chart(feat_imp)
