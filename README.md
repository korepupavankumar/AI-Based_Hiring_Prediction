# 🤖 AI-Based Hiring_Prediction_System (Machine Learning + Streamlit)

## 📌 Project Overview
This project is an **AI-Based Hiring Prediction System** that predicts whether a candidate should be **Hired or Rejected** based on their resume details such as skills, experience, education, certifications, job role, salary expectation, and project count.

It uses **Machine Learning (Random Forest)** and **Natural Language Processing (TF-IDF)** and is deployed using **Streamlit web interface**.

---

## 🎯 Objective
To automate the initial resume screening process and help recruiters quickly identify suitable candidates using AI.

---

## 📊 Dataset Description

The dataset contains the following features:

- Resume_ID (ignored in training)
- Name (ignored in training)
- Skills
- Experience (Years)
- Education
- Certifications
- Job Role
- Salary Expectation ($)
- Projects Count
- AI Score (ignored to avoid data leakage)
- Recruiter Decision (Target: Hire / Reject)

---

## 🧠 Machine Learning Approach

### ✔ Feature Engineering
- Text features combined: Skills + Certifications + Job Role
- Numerical features:
  - Experience (Years)
  - Education (Label Encoded)
  - Salary Expectation
  - Projects Count

### ✔ Techniques Used
- TF-IDF Vectorization (for text data)
- Label Encoding (for Education & Target)
- Random Forest Classifier

---

## 🏗️ Project Structure    
AI-Based Hiring_Prediction_System/
│
├── app.py # Streamlit web app
├── train_model.py # Model training script
├── AI_Resume_Screening.csv
│
├── model.pkl # Trained ML model
├── tfidf.pkl # TF-IDF vectorizer
├── education_encoder.pkl # Education encoder
├── target_encoder.pkl # Target encoder
│
└── README.md

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn joblib


python train_model.py

python -m streamlit run app.py

🌐 Features
📌 User-friendly web interface
📌 Real-time prediction
📌 Supports multiple resume inputs
📌 Fast ML-based decision system
📌 Simple and lightweight design

Input:
Skills: Python, TensorFlow
Experience: 5 years
Education: B.Tech
Job Role: AI Engineer

Output:
✅ Hire


📈 Model Performance
Algorithm: Random Forest Classifier
Input Type: Hybrid (Text + Numeric)
Accuracy: High (depends on dataset split)

🛠️ Technologies Used
Python 🐍
Pandas
NumPy
Scikit-learn
Streamlit
Joblib

📌 Future Improvements
Resume PDF upload support
Deep Learning model integration
Advanced dashboard analytics
Deployment on cloud (Streamlit Cloud / AWS)
