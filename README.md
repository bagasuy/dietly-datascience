# Dietly Data Science

This repository contains the data science and machine learning components for **Dietly**, a web application for diet and weight tracking.

The machine learning component estimates a user's future weight based on recent weight history and dietary records.

## Project Overview

Dietly combines a full-stack web application with a machine learning model.

The data science component is responsible for:

- Data understanding and assessment
- Data cleaning and preparation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model development
- Model evaluation
- Model export for backend integration
- Interactive Streamlit dashboard

## Business Problem

Dietly needs a way to provide users with an estimated future weight based on their recorded weight history and dietary information.

The main business question is:

> Based on a user's historical weight and dietary records, what is the estimated future weight?

The prediction is intended as an estimate generated from the available historical data. It is not a medical diagnosis or health recommendation.

## Dataset

The model was developed using the **DietDiary** dataset.

The dataset contains repeated weight observations together with dietary records for breakfast, lunch, and supper.

The dataset used in this project contains:

- **5,251 records**
- **611 participants**
- Weight observations
- Dietary records for breakfast, lunch, and supper
- Observation period from **2017-09-08 to 2021-04-14**

The original dataset documentation is available in:

```text
data/raw/readme.txt
```

The project uses the dataset for machine learning experimentation and model development. Dataset licensing and redistribution conditions should be reviewed according to the original dataset source before redistribution.

## Machine Learning Approach

The final model uses:

**Ridge Regression + TF-IDF**

### Numerical Features

The model uses three weight-history features:

- `previous_weight`
- `weight`
- `historical_weight_change`

### Text Features

Dietary records are converted into text features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

Dietary text includes meal information such as:

```text
breakfast ...
lunch ...
supper ...
```

The numerical and TF-IDF features are combined before being passed to the Ridge Regression model.

## Model Evaluation

The final model was evaluated using a held-out participant test set.

| Metric | Result |
|---|---:|
| MAE | 0.30 kg |
| RMSE | 0.41 kg |
| R² | 0.9986 |

These metrics describe model performance on the evaluation data and should not be interpreted as a guarantee of prediction accuracy for individual users.

## Model Artifact

The trained model is exported as:

```text
model/dietly_weight_model.joblib
```

The Joblib bundle contains:

- Trained Ridge Regression model
- TF-IDF vectorizer
- Numerical feature definitions

## Prediction Function

`predict.py` provides the prediction interface used by the application.

The main function is:

```python
predict_future_weight(input_data)
```

### Required Input

The prediction function requires the following fields:

```text
previous_weight
weight
historical_weight_change
dietary_text
```

### Output

The function returns:

```text
Predicted future weight in kilograms
```

### Example

```python
from predict import predict_future_weight

result = predict_future_weight({
    "previous_weight": 68.3,
    "weight": 68.8,
    "historical_weight_change": 0.5,
    "dietary_text": (
        "breakfast egg tomato rice "
        "lunch cabbage soup "
        "supper chicken"
    ),
})

print(result)
```

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard in:

```text
app.py
```

The dashboard provides:

1. Dataset overview
2. Weight distribution analysis
3. Weight trend analysis
4. Dietary record analysis
5. Model performance metrics
6. Prediction demonstration
7. Project conclusion

## Project Structure

```text
dietly-datascience/
├── app.py
├── predict.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│       ├── data.csv
│       └── readme.txt
│
├── model/
│   └── dietly_weight_model.joblib
│
└── notebooks/
    └── 02_dietly_dietdiary_ml.ipynb
```

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Run Prediction Test

Run the prediction module directly:

```bash
python predict.py
```

A successful execution produces a predicted future weight, for example:

```text
Predicted future weight: 68.56 kg
```

## Run Streamlit Dashboard

Start the dashboard with:

```bash
streamlit run app.py
```

The Streamlit application will then be available through the local URL displayed by Streamlit.

## Backend Integration

The exported model is also used by the Dietly Django REST Framework backend.

The integration flow is:

```text
Dietly User
    ↓
Weight History + Diet Entries
    ↓
Django REST API
    ↓
Prediction Service
    ↓
dietly_weight_model.joblib
    ↓
Ridge Regression + TF-IDF
    ↓
Predicted Future Weight
    ↓
Prediction API Response
    ↓
Dietly Frontend
```

The backend loads the same model bundle and prepares the required numerical and dietary text features before performing inference.

## Reproducibility

The machine learning workflow is documented in:

```text
notebooks/02_dietly_dietdiary_ml.ipynb
```

The notebook contains the data understanding, assessment, preparation, exploratory analysis, feature engineering, model development, evaluation, and model export workflow.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy
- Joblib
- Streamlit
- Jupyter Notebook

## Project Context

Dietly is a capstone project combining:

- Frontend web development
- Backend REST API development
- Database persistence
- Data science
- Machine learning
- Machine learning inference integration

The machine learning component is designed to be consumed by the Dietly backend rather than functioning as an isolated model.