import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("AI_Resume_Screening.csv")

df.drop(columns=["Resume_ID", "Name", "AI Score (0-100)"], inplace=True)
df.fillna("None", inplace=True)

df["combined_text"] = df["Skills"] + " " + df["Certifications"] + " " + df["Job Role"]

edu_encoder = LabelEncoder()
df["Education"] = edu_encoder.fit_transform(df["Education"])

target_encoder = LabelEncoder()
df["Recruiter Decision"] = target_encoder.fit_transform(df["Recruiter Decision"])

tfidf = TfidfVectorizer(max_features=200)
X_text = tfidf.fit_transform(df["combined_text"]).toarray()

X_num = df[["Experience (Years)", "Education", "Salary Expectation ($)", "Projects Count"]].values

X = np.hstack((X_text, X_num))
y = df["Recruiter Decision"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(edu_encoder, "education_encoder.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Model trained successfully!")