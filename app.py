import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from predict import predict_future_weight


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Dietly Data Science",
    page_icon="⚖️",
    layout="wide",
)


# ============================================================
# Load Dataset
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "data.csv"

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

meal_columns = [
    "breakfast",
    "lunch",
    "supper",
]


# ============================================================
# Header
# ============================================================

st.title("Dietly Weight Prediction")

st.write(
    """
    This dashboard presents exploratory insights from the
    DietDiary dataset and demonstrates the machine learning
    model used by Dietly to estimate future weight.
    """
)


# ============================================================
# Dataset Overview
# ============================================================

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Participants",
        f"{df['ID'].nunique():,}"
    )

with col3:
    st.metric(
        "Average Weight",
        f"{df['weight'].mean():.2f} kg"
    )

with col4:
    st.metric(
        "Median Weight",
        f"{df['weight'].median():.2f} kg"
    )


st.write(
    f"Observation period: "
    f"{df['date'].min().date()} to "
    f"{df['date'].max().date()}"
)


# ============================================================
# Weight Analysis
# ============================================================

st.header("Weight Analysis")

tab1, tab2 = st.tabs([
    "Weight Distribution",
    "Weight Trend",
])


with tab1:

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["weight"],
        bins=30,
    )

    ax.set_title("Distribution of Recorded Weight")
    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)


with tab2:

    daily_weight = (
        df.groupby("date")["weight"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        daily_weight["date"],
        daily_weight["weight"],
    )

    ax.set_title("Average Recorded Weight Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Weight (kg)")

    plt.xticks(rotation=45)

    st.pyplot(fig)


# ============================================================
# Dietary Record Analysis
# ============================================================

st.header("Dietary Record Analysis")

st.write(
    """
    DietDiary contains dietary records for breakfast, lunch,
    and supper. These records provide textual dietary information
    that is transformed into TF-IDF features for the prediction
    model.
    """
)

meal_presence = pd.DataFrame({
    "Meal": ["Breakfast", "Lunch", "Supper"],
    "Records": [
        df["breakfast"].notna().sum(),
        df["lunch"].notna().sum(),
        df["supper"].notna().sum(),
    ],
})

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    meal_presence["Meal"],
    meal_presence["Records"],
)

ax.set_title("Dietary Records by Meal")
ax.set_xlabel("Meal")
ax.set_ylabel("Number of Records")

st.pyplot(fig)


# ============================================================
# Model Performance
# ============================================================

st.header("Model Performance")

st.write(
    """
    The final model uses Ridge Regression with numerical
    weight-history features and TF-IDF representations of
    dietary text. Evaluation was performed on participants
    that were not included in the training set.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        "0.30 kg"
    )

with col2:
    st.metric(
        "RMSE",
        "0.41 kg"
    )

with col3:
    st.metric(
        "R²",
        "0.9986"
    )

st.caption(
    "Evaluation metrics are based on the held-out participant test set."
)


# ============================================================
# Prediction Demo
# ============================================================

st.header("Prediction Demo")

st.write(
    """
    Enter recent weight history and dietary information to
    generate an estimated future weight using the trained model.
    """
)

col1, col2 = st.columns(2)

with col1:

    previous_weight = st.number_input(
        "Previous Weight (kg)",
        min_value=20.0,
        max_value=300.0,
        value=68.3,
        step=0.1,
    )

    current_weight = st.number_input(
        "Current Weight (kg)",
        min_value=20.0,
        max_value=300.0,
        value=68.8,
        step=0.1,
    )

    historical_change = st.number_input(
        "Historical Weight Change (kg)",
        value=0.5,
        step=0.1,
    )


with col2:

    dietary_text = st.text_area(
        "Dietary Records",
        value=(
            "breakfast egg tomato rice "
            "lunch cabbage soup "
            "supper chicken"
        ),
        height=150,
    )


if st.button(
    "Predict Future Weight",
    type="primary",
):

    input_data = {
        "previous_weight": previous_weight,
        "weight": current_weight,
        "historical_weight_change": historical_change,
        "dietary_text": dietary_text,
    }

    try:

        prediction = predict_future_weight(
            input_data
        )

        predicted_change = (
            prediction - current_weight
        )

        st.success(
            f"Predicted future weight: "
            f"{prediction:.2f} kg"
        )

        st.metric(
            "Predicted Weight Change",
            f"{predicted_change:+.2f} kg",
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# Conclusion
# ============================================================

st.header("Conclusion")

st.write(
    """
    The analysis shows that the DietDiary dataset provides
    repeated weight observations together with dietary records.
    Recent weight history provides strong predictive information,
    while dietary text is incorporated through TF-IDF features.

    The final Ridge Regression model is exported as a Joblib
    model bundle and is used by the Dietly backend for prediction.
    """
)