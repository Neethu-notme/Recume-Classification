import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import pdfplumber
import docx
import matplotlib.pyplot as plt

model = pickle.load(open("model_1.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def extract_docx(file):
    doc = docx.Document(file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

st.set_page_config(page_title="Resume Classifier", layout="wide")

st.title("Resume Classification System")

st.write(
"""
This system predicts the **job category of a resume** using a trained **Machine Learning model (SVM)**  
and **TF-IDF feature extraction**.
"""
)

st.subheader("Upload Resume OR Paste Text")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx"])

resume_text = st.text_area("Or Paste Resume Text Here")

if st.button("Predict Category"):
    text = ""
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_docx(uploaded_file)
    else:
        text = resume_text
    if text.strip() == "":
        st.warning("Please upload or paste resume content")
    else:
        clean = preprocess(text)
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = np.max(probs) * 100
        st.success("Prediction Complete")
        st.subheader("Prediction Result")
        st.write("Predicted Category:", prediction)
        st.write(f"Confidence Score: {confidence:.2f}%")
        st.subheader("Category Probabilities")
        categories = model.classes_
        prob_df = pd.DataFrame({
            "Category": categories,
            "Probability": probs
        })
        st.dataframe(prob_df)

        fig, ax = plt.subplots()
        ax.bar(categories, probs)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Category")
        ax.set_title("Prediction Probability Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)