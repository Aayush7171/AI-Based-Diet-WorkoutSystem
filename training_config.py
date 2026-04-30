FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Weight (kg)",
    "Height (m)",
    "BMI",
    "Workout_Frequency (days/week)",
    "Session_Duration (hours)",
    "Experience_Level",
]

TARGET_COLUMN = "Calories_Burned"

DATASET_CONFIGS = [
    {
        "name": "clean_fitness",
        "path": "clean_fitness_dataset.csv",
        "model_name": "linear_regression",
    },
    {
        "name": "exercise",
        "path": "Exercise.csv",
        "model_name": "random_forest",
    },
]
