!pip install streamlit
!pip install "numpy<2" && pip install git+https://github.com/NicolasHug/Surprise.git
import streamlit as st
import pickle
import pandas as pd

st.title("Adaptive Learning Recommendation System")

model = pickle.load(open("adaptiveLearning/model/trained_model.pkl", "rb"))
trainset = pickle.load(open("adaptiveLearning/model/train_data.pkl", "rb"))
data = pd.read_csv("adaptiveLearning/data/adaptive_learning.csv")

def recommend_courses(model, trainset, student_id, df, top_n=5):
    all_courses = df["Course_Name"].unique()
    taken_courses = df[df["Student_ID"] == student_id]["Course_Name"].unique()
    preds = []
    for course in all_courses:
        if course not in taken_courses:
            est = model.predict(student_id, course).est
            preds.append((course, est))
    top = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return top

student_id = st.selectbox("Select a Student ID", sorted(data["Student_ID"].unique()))
top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend"):
    recs = recommend_courses(model, trainset, student_id, data, top_n)
    st.subheader(f"Top {top_n} Recommended Courses for {student_id}:")
    for course, score in recs:
        st.write(f"{course} — Predicted Rating: {score:.2f}")

