import pickle

import numpy as np
import pandas as pd
import streamlit as st

from training_config import FEATURE_COLUMNS


st.set_page_config(page_title="Fitness Calories Predictor", layout="wide")

EXERCISE_FILE = "Exercise.csv"
DIET_PLAN_FILE = "diet_recommendations_dataset.csv"

model = pickle.load(open("model.pkl", "rb"))


@st.cache_data
def load_data():
    exercise_df = pd.read_csv(EXERCISE_FILE)
    diet_plan_df = pd.read_csv(DIET_PLAN_FILE)

    exercise_numeric_columns = [
        "Calories_Burned",
        "Session_Duration (hours)",
        "Workout_Frequency (days/week)",
        "Experience_Level",
        "BMI",
    ]
    diet_numeric_columns = ["Daily_Caloric_Intake", "Age", "BMI"]

    for column in exercise_numeric_columns:
        if column in exercise_df.columns:
            exercise_df[column] = pd.to_numeric(exercise_df[column], errors="coerce")

    for column in diet_numeric_columns:
        if column in diet_plan_df.columns:
            diet_plan_df[column] = pd.to_numeric(diet_plan_df[column], errors="coerce")

    return exercise_df, diet_plan_df


def get_activity_level(calories_burned):
    if calories_burned < 250:
        return "Sedentary"
    if calories_burned < 500:
        return "Moderate"
    return "Active"


def predict_calories(model_object, input_data):
    if isinstance(model_object, dict) and model_object.get("model_type") == "multi_dataset_ensemble":
        input_frame = pd.DataFrame(input_data, columns=model_object.get("feature_columns", FEATURE_COLUMNS))
        predictions = [trained_model.predict(input_frame) for trained_model in model_object["models"].values()]
        return np.mean(predictions, axis=0)

    return model_object.predict(input_data)


def recommend_exercises(
    predicted_calories, gender, experience_level, workout_days, bmi, exercise_df
):
    candidates = exercise_df.copy()

    if "Gender" in candidates.columns:
        gender_matches = candidates[candidates["Gender"].astype(str).str.lower() == gender.lower()]
        if not gender_matches.empty:
            candidates = gender_matches

    if "Experience_Level" in candidates.columns:
        experience_matches = candidates[
            candidates["Experience_Level"].between(experience_level - 1, experience_level + 1)
        ]
        if not experience_matches.empty:
            candidates = experience_matches

    if "Workout_Frequency (days/week)" in candidates.columns:
        frequency_matches = candidates[
            candidates["Workout_Frequency (days/week)"].between(workout_days - 1, workout_days + 1)
        ]
        if not frequency_matches.empty:
            candidates = frequency_matches

    if "BMI" in candidates.columns:
        bmi_matches = candidates[candidates["BMI"].between(bmi - 3, bmi + 3)]
        if not bmi_matches.empty:
            candidates = bmi_matches

    candidates = candidates.dropna(subset=["Calories_Burned", "Workout_Type"])
    candidates["calorie_gap"] = (candidates["Calories_Burned"] - predicted_calories).abs()
    closest_matches = candidates.nsmallest(40, "calorie_gap")

    summary = (
        closest_matches.groupby("Workout_Type", as_index=False)
        .agg(
            Avg_Calories=("Calories_Burned", "mean"),
            Avg_Duration_Hours=("Session_Duration (hours)", "mean"),
        )
        .sort_values(["Avg_Calories", "Avg_Duration_Hours"], ascending=[False, False])
        .head(3)
    )

    return summary


def recommend_diet_plan(calorie_intake, gender, age, activity_level, diet_plan_df):
    candidates = diet_plan_df.copy()

    if "Gender" in candidates.columns:
        gender_matches = candidates[candidates["Gender"].astype(str).str.lower() == gender.lower()]
        if not gender_matches.empty:
            candidates = gender_matches

    if "Physical_Activity_Level" in candidates.columns:
        activity_matches = candidates[
            candidates["Physical_Activity_Level"].astype(str).str.lower() == activity_level.lower()
        ]
        if not activity_matches.empty:
            candidates = activity_matches

    if "Age" in candidates.columns:
        age_matches = candidates[candidates["Age"].between(age - 10, age + 10)]
        if not age_matches.empty:
            candidates = age_matches

    candidates = candidates.dropna(subset=["Daily_Caloric_Intake", "Diet_Recommendation"])
    candidates["calorie_gap"] = (candidates["Daily_Caloric_Intake"] - calorie_intake).abs()

    top_rows = candidates.nsmallest(5, "calorie_gap")

    plan_summary = (
        top_rows.groupby("Diet_Recommendation", as_index=False)
        .agg(
            Avg_Intake=("Daily_Caloric_Intake", "mean"),
            Restriction=("Dietary_Restrictions", "first"),
        )
        .sort_values(["Avg_Intake"], ascending=[True])
    )

    best_row = top_rows.iloc[0] if not top_rows.empty else None
    return plan_summary, best_row


def get_meal_guidance(diet_recommendation):
    guidance_map = {
        "Balanced": [
            "Include a lean protein source, whole grains, and vegetables in each main meal.",
            "Choose fruit, yogurt, or nuts for snacks instead of fried or sugary foods.",
        ],
        "Low_Carb": [
            "Prioritize protein, salad, eggs, paneer, tofu, or grilled chicken.",
            "Reduce refined rice, bread, sugary drinks, and high-sugar snacks.",
        ],
        "Low_Sodium": [
            "Use fresh foods more often and reduce packaged, canned, and highly processed meals.",
            "Flavor meals with herbs, lemon, ginger, and spices instead of extra salt.",
        ],
    }
    return guidance_map.get(
        diet_recommendation,
        [
            "Focus on whole foods, enough protein, and regular meal timing.",
            "Keep portions steady and limit highly processed foods.",
        ],
    )


st.title("Fitness Calories Predictor")
st.write(
    "Predict calories burned, then get exercise suggestions and diet recommendations "
    "matched to your daily calorie intake."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=80, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0)
    height = st.number_input("Height (m)", min_value=1.4, max_value=2.2, value=1.70)
    daily_calorie_intake = st.number_input(
        "Daily Calorie Intake (kcal)", min_value=1000, max_value=5000, value=2200
    )

with col2:
    workout_days = st.slider("Workout Days per Week", min_value=0, max_value=7, value=3)
    session_duration = st.number_input(
        "Session Duration (hours)", min_value=0.5, max_value=3.0, value=1.0
    )
    experience = st.selectbox("Experience Level", [1, 2, 3], index=0)

bmi = weight / (height * height)
st.caption(f"Calculated BMI: {bmi:.2f}")

exercise_df, diet_plan_df = load_data()
gender_val = 0 if gender == "Male" else 1

if st.button("Predict Calories Burned"):
    input_data = np.array(
        [[age, gender_val, weight, height, bmi, workout_days, session_duration, experience]]
    )
    prediction = predict_calories(model, input_data)
    predicted_calories = float(prediction[0])
    activity_level = get_activity_level(predicted_calories)

    st.success(f"Estimated Calories Burned: {predicted_calories:.2f} kcal")
    st.info(f"Suggested activity band: {activity_level}")

    exercise_summary = recommend_exercises(
        predicted_calories, gender, experience, workout_days, bmi, exercise_df
    )
    diet_summary, best_diet_match = recommend_diet_plan(
        daily_calorie_intake, gender, age, activity_level, diet_plan_df
    )

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.subheader("Recommended Exercise Types")
        if exercise_summary.empty:
            st.warning("No close exercise matches were found in the dataset.")
        else:
            formatted_exercises = exercise_summary.copy()
            formatted_exercises["Avg_Calories"] = formatted_exercises["Avg_Calories"].round(1)
            formatted_exercises["Avg_Duration_Hours"] = formatted_exercises[
                "Avg_Duration_Hours"
            ].round(2)
            st.dataframe(formatted_exercises, use_container_width=True)

    with rec_col2:
        st.subheader("Diet Recommendation")
        if diet_summary.empty or best_diet_match is None:
            st.warning("No diet recommendation match was found in the dataset.")
        else:
            best_plan = str(best_diet_match["Diet_Recommendation"]).replace("_", " ")
            dietary_restrictions = best_diet_match.get("Dietary_Restrictions", "None")

            st.write(f"Recommended plan: **{best_plan}**")
            st.write(f"Dietary restriction in matched sample: **{dietary_restrictions}**")

            formatted_diet_summary = diet_summary.copy()
            formatted_diet_summary["Avg_Intake"] = formatted_diet_summary["Avg_Intake"].round(0)
            st.dataframe(formatted_diet_summary, use_container_width=True)

            st.subheader("Meal Guidance")
            for tip in get_meal_guidance(best_diet_match["Diet_Recommendation"]):
                st.write(f"- {tip}")
