!pip install "numpy<2" && pip install git+https://github.com/NicolasHug/Surprise.git

from flask import Flask, request, jsonify
import pickle
import pandas as pd
from surprise import SVD

app = Flask(__name__)

model = pickle.load(open("adaptiveLearning/model/train_data.pkl", "rb"))
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
    return [{"Course Name": c, "Predicted Score": round(s, 2)} for c, s in top]

@app.route("/")
def home():
    return jsonify({"message": "Adaptive Learning Recommendation API is running!"})

@app.route("/recommend", methods=["GET"])
def recommend():
    student_id = request.args.get("student_id")
    top_n = int(request.args.get("top_n", 5))

    if student_id not in data["Student_ID"].values:
        return jsonify({"error": "Student ID not found"}), 404

    recommendations = recommend_courses(model, trainset, student_id, data, top_n)
    return jsonify({"Student id": student_id, "Recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)

