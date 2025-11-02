# adaptiveLearning

This project implements a **personalized learning recommendation system** that suggests suitable learning materials for students using collaborative filtering. It leverages the **Surprise** library for model training, along with **Pandas**, **NumPy**, and **Streamlit** for data handling, analysis, and visualization.

## Features
- Collaborative filtering using the Surprise library (SVD / KNNBaseline)
- Adaptive rating system combining engagement, feedback, and dropout likelihood
- Course recommendations tailored to student learning styles
- Model persistence for reuse and evaluation

## Tech Stack
- **Python**  
- **Pandas**, **NumPy**, **scikit-surprise**  
- **Pickle** for model persistence

## Dataset

The model expects a dataset with the following columns:

 - Student_ID - Unique ID for each student
 - Age - Age of the student
 - Gender - Gender of the student
 Education - Education level
 Course_Name - Name of the course
 Time_Spent_on_Videos - Total time spent watching course videos
 Quiz_Attempts - Number of quizzes attempted
 Quiz_Scores - Average quiz score
 Forum_Participation - Forum activity level
 Assignment_Completion_Rate - Percentage of assignments completed
 Final_Exam_Score - Final exam performance
 Engagement_Level - Engagement rating
 Learning_Style - Visual, Auditory, or Kinesthetic
 Feedback_Score - Overall feedback score
 Dropout_Likelihood - Likelihood of dropping out (0–1 scale)

## Project Structure
