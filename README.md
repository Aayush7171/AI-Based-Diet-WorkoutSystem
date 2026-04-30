# Fitness Calories Predictor

A Streamlit app that predicts calories burned and gives simple exercise and diet recommendations based on user input.

## Features

- Predicts `Calories_Burned`
- Suggests exercise types from `Exercise.csv`
- Recommends a diet plan from `diet_recommendations_dataset.csv`
- Shows meal guidance based on the selected diet recommendation
- Supports a combined saved model in `model.pkl`

## Project Files

- [app.py](/c:/Users/Aayush/OneDrive/Attachments/Desktop/AI%20Project/app.py): Streamlit application
- [model.ipynb](/c:/Users/Aayush/OneDrive/Attachments/Desktop/AI%20Project/model.ipynb): model training notebook
- [training_config.py](/c:/Users/Aayush/OneDrive/Attachments/Desktop/AI%20Project/training_config.py): shared feature and dataset config
- `model.pkl`: saved trained model

## Datasets Used

- `clean_fitness_dataset.csv`: calorie prediction training data
- `Exercise.csv`: calorie prediction training data and exercise recommendation data
- `diet_recommendations_dataset.csv`: diet recommendation lookup data

## Requirements

Install the required packages:

```powershell
py -m pip install streamlit pandas numpy scikit-learn
```

## Run the App

From the project folder:

```powershell
py -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Train or Retrain the Model

If you want to regenerate `model.pkl`, open and run [model.ipynb](/c:/Users/Aayush/OneDrive/Attachments/Desktop/AI%20Project/model.ipynb).

The notebook trains models using:

- `clean_fitness_dataset.csv`
- `Exercise.csv`

It then saves the trained output to `model.pkl`.

## Inputs in the App

- Age
- Gender
- Weight
- Height
- Daily calorie intake
- Workout days per week
- Session duration
- Experience level

## Output from the App

- Estimated calories burned
- Suggested activity band
- Recommended exercise types
- Diet recommendation
- Meal guidance