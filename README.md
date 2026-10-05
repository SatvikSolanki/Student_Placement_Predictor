# 🎓 Student Placement Predictor

A Streamlit web app that predicts whether a student is likely to be **Placed** or **Not Placed**, using a **Decision Tree** classifier trained on 10,000 student records.

**Live demo:** _add your Streamlit link here_

## Features

- **Predict:** enter a student's profile and get a verdict with a placement probability gauge.
- **Explainable:** see the exact decision-tree rules that led to the answer, and how the student compares with the average placed student.
- **Data insights:** feature importance, placed vs. not placed split, and box plots for any feature.
- **Model report:** accuracy, precision, recall, F1-score, confusion matrix and the tuned hyperparameters.
- Dark themed UI.

## Input features

| Feature | Description |
|---|---|
| Gender, Branch | Male/Female; CSE, IT, ECE, Mechanical, Civil, Commerce |
| Study hours, Sleep hours, Internet usage | Hours per day |
| Attendance | Percent |
| Assignments completed | Count |
| Previous score, Exam score | Marks out of 100 |
| Extracurricular | Yes / No |

## How the model is built

1. **Cleaning:** removed duplicates and missing rows, dropped `student_id`, standardised gender and branch spellings, and clipped unrealistic study, sleep and internet hours.
2. **Encoding:** one-hot encoding for branch, 0/1 for gender and extracurricular.
3. **Split:** 80% train and 20% test (`random_state=42`).
4. **Tuning:** `RandomizedSearchCV` (5-fold) over `max_depth`, `min_samples_leaf` and `criterion`.

The model trains automatically from the CSV when the app first starts, then stays cached.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
├── app.py                              # Streamlit app
├── requirements.txt
├── student_performance_dataset.csv
├── .streamlit/config.toml              # dark theme
└── Decision_Tree.ipynb                 # training notebook (optional)
```

## Tech stack

Python, pandas, NumPy, scikit-learn, Plotly, Streamlit

## Disclaimer

This is a learning project. Predictions are statistical estimates, not a guarantee of any real placement outcome.
