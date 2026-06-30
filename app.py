import streamlit as st
import numpy as np
import joblib

# Load saved models
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
edu_encoder = joblib.load("education_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="AI Resume Screening", layout="centered")

st.title("🤖 AI-Based Hiring Prediction System")
st.write("Predict whether a candidate will be **Hire or Reject** based on resume data")

# ---------------- INPUT FIELDS ---------------- #

skills = st.text_input("Skills (comma separated)")
certifications = st.text_input("Certifications")
job_role = st.text_input("Job Role")

experience = st.number_input("Experience (Years)", min_value=0.0, step=0.5)

education = st.selectbox("Education", edu_encoder.classes_)

salary = st.number_input("Salary Expectation ($)", min_value=0.0, step=1000.0)

projects = st.number_input("Projects Count", min_value=0, step=1)

# ---------------- PREDICTION ---------------- #

if st.button("Predict Recruiter Decision"):

    # Encode education (safe handling)
    education_encoded = edu_encoder.transform([education])[0]

    # Combine text fields (matches training)
    combined_text = f"{skills} {certifications} {job_role}"
    text_vector = tfidf.transform([combined_text]).toarray()

    # Numeric features (must match train_model order)
    numeric_features = np.array([[
        experience,
        education_encoded,
        salary,
        projects
    ]])

    # Final feature vector
    final_input = np.hstack((text_vector, numeric_features))

    # Prediction
    prediction = model.predict(final_input)[0]
    result = target_encoder.inverse_transform([prediction])[0]

    # Output
    st.subheader("Result:")

    if result == "Hire":
        st.success("✅ Candidate is SELECTED (Hire)")
    else:
        st.error("❌ Candidate is REJECTED")