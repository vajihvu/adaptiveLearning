from typing import Optional, List, Tuple, Dict
import os
import pickle
import numpy as np
import pandas as pd
from surprise import Dataset, Reader
from surprise import accuracy


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def preprocess_df(
    df: pd.DataFrame,
    feedback_col: str = "Feedback_Score",
    engagement_col: str = "Engagement_Level",
    dropout_col: str = "Dropout_Likelihood",
    time_col: Optional[str] = None,
    time_decay_half_life_days: Optional[float] = None,
) -> pd.DataFrame:
    
    df = df.copy()

    for c in [feedback_col, engagement_col, dropout_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    df["_adaptive_raw"] = (
        df[feedback_col].astype(float)
        * df[engagement_col].astype(float)
        * (1.0 - df[dropout_col].astype(float))
    )

    if time_col and time_decay_half_life_days is not None and time_col in df.columns:
        if not np.issubdtype(df[time_col].dtype, np.datetime64):
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        now = df[time_col].max()
        age_days = (now - df[time_col]).dt.total_seconds() / (3600 * 24)
        age_days = age_days.fillna(age_days.max() if not age_days.isna().all() else 0)
        decay = 0.5 ** (age_days / float(time_decay_half_life_days))
        df["_decay"] = decay
        df["_adaptive_raw"] = df["_adaptive_raw"] * df["_decay"]

    min_r = df["_adaptive_raw"].min()
    max_r = df["_adaptive_raw"].max()
    if pd.isna(min_r) or pd.isna(max_r) or max_r == min_r:
        if feedback_col in df.columns:
            fb_min, fb_max = df[feedback_col].min(), df[feedback_col].max()
            if fb_max == fb_min:
                df["Adaptive_Rating"] = 3.0
            else:
                df["Adaptive_Rating"] = 1 + 4 * (df[feedback_col] - fb_min) / (fb_max - fb_min)
        else:
            df["Adaptive_Rating"] = 3.0
    else:
        df["Adaptive_Rating"] = 1 + 4 * (df["_adaptive_raw"] - min_r) / (max_r - min_r)

    df["Adaptive_Rating"] = df["Adaptive_Rating"].clip(1.0, 5.0).round(2)

    df = df.drop(columns=[c for c in ["_adaptive_raw", "_decay"] if c in df.columns])
    return df


def build_surprise_dataset(
    df: pd.DataFrame, user_col: str = "Student_ID", item_col: str = "Course_Name", rating_col: str = "Adaptive_Rating",
) -> Dataset:
    if user_col not in df.columns or item_col not in df.columns or rating_col not in df.columns:
        raise KeyError("One of user/item/rating columns not found in DataFrame")
    reader = Reader(rating_scale=(1.0, 5.0))
    dataset = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
    return dataset


def save_model(model, trainset, model_path: str = "adaptiveLearning/model/trained_model.pkl", trainset_path: str = "adaptiveLearning/model/trainset.pkl") -> None:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(trainset_path, "wb") as f:
        pickle.dump(trainset, f)


def load_model(model_path: str = "adaptiveLearning/model/trained_model.pkl", trainset_path: str = "adaptiveLearning/model/trainset.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(trainset_path):
        raise FileNotFoundError("Model or trainset file not found")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(trainset_path, "rb") as f:
        trainset = pickle.load(f)
    return model, trainset


def recommend_courses(
    model,
    trainset,
    df: pd.DataFrame,
    student_id: str,
    top_n: int = 5,
    user_col: str = "Student_ID",
    item_col: str = "Course_Name",
    filter_learning_style: Optional[str] = None,
    min_engagement: Optional[float] = None,
) -> List[Dict[str, object]]:

    items = df[item_col].unique().tolist()

    taken = set(df.loc[df[user_col] == student_id, item_col].unique()) if student_id in df[user_col].values else set()

    candidates = [it for it in items if it not in taken]
    if filter_learning_style and "Learning_Style" in df.columns:
        allowed = set(df[df["Learning_Style"] == filter_learning_style][item_col].unique())
        candidates = [c for c in candidates if c in allowed]
    if min_engagement is not None and "Engagement_Level" in df.columns:
        high_eng = set(df[df["Engagement_Level"] >= float(min_engagement)][item_col].unique())
        candidates = [c for c in candidates if c in high_eng]

    preds = []
    for iid in candidates:
        try:
            est = model.predict(student_id, iid).est
        except Exception:
            try:
                est = trainset.global_mean
            except Exception:
                est = 3.0
        preds.append((iid, float(est)))

    top = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"Course_Name": c, "Predicted_Score": round(s, 3)} for c, s in top]



def evaluate_on_testset(model, testset) -> Dict[str, float]:
    preds = model.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)
    return {"rmse": rmse, "mae": mae, "n": len(testset)}


def precision_at_k(model, trainset, test_df: pd.DataFrame, user_col: str = "Student_ID", item_col: str = "Course_Name", k: int = 10) -> float:
    
    actual = test_df.groupby(user_col)[item_col].apply(set).to_dict()
    users = list(test_df[user_col].unique())
    items_all = test_df[item_col].unique().tolist()

    precisions = []
    for u in users:
        try:
            topk = [d["Course_Name"] for d in recommend_courses(model, trainset, test_df, u, top_n=k)]
        except Exception:
            topk = []
        act = actual.get(u, set())
        if len(topk) == 0:
            continue
        hit = len([i for i in topk if i in act])
        precisions.append(hit / k)

    return float(np.mean(precisions)) if len(precisions) > 0 else 0.0


def get_student_profile(df: pd.DataFrame, student_id: str) -> pd.Series:
    if student_id not in df["Student_ID"].values:
        raise KeyError("Student not found in data")
    sub = df[df["Student_ID"] == student_id]
    numeric = sub.select_dtypes(include=["number"]).agg(["mean", "median"]).T
    categorical = {}
    for c in sub.select_dtypes(include=["object", "category"]).columns:
        if sub[c].mode().size > 0:
            categorical[c] = sub[c].mode().iloc[0]
    profile = pd.concat([numeric.squeeze(), pd.Series(categorical)])
    return profile


if __name__ == "__main__":
    df = load_data("adaptiveLearning/data/adaptive_learning.csv")

    df = preprocess_df(df)
    print("Preprocessed — sample:")
    print(df.head())

    data = build_surprise_dataset(df)
    trainset = data.build_full_trainset()
    try:
        from surprise import SVD
        algo = SVD(n_factors=20, n_epochs=8, random_state=42)
        algo.fit(trainset)
        print("Trained quick SVD demo model.")
        student = df["Student_ID"].sample(1, random_state=1).iloc[0]
        recs = recommend_courses(algo, trainset, df, student, top_n=5)
        print(f"Recommendations for {student}:")
        print(recs)
    except Exception as e:
        print("Surprise not available in the environment or train failed:", e)