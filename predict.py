from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack


# Load trained Dietly model bundle
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "dietly_weight_model.joblib"
)

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
tfidf = bundle["tfidf"]

NUMERIC_FEATURES = [
    "previous_weight",
    "weight",
    "historical_weight_change",
]


def predict_future_weight(input_data):
    """
    Predict future weight from recent weight history
    and dietary information.

    Parameters
    ----------
    input_data : dict
        Required keys:
        - previous_weight
        - weight
        - historical_weight_change
        - dietary_text

    Returns
    -------
    float
        Predicted future weight in kilograms.
    """

    required_features = NUMERIC_FEATURES + ["dietary_text"]

    missing_features = [
        feature
        for feature in required_features
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    numeric_values = [
        float(input_data[feature])
        for feature in NUMERIC_FEATURES
    ]

    numeric_features = np.array(
        numeric_values,
        dtype=float
    ).reshape(1, -1)

    dietary_text = str(input_data["dietary_text"])

    text_features = tfidf.transform([dietary_text])

    features = hstack([
        numeric_features,
        text_features,
    ])

    prediction = model.predict(features)

    return float(prediction[0])


if __name__ == "__main__":
    sample_input = {
        "previous_weight": 68.3,
        "weight": 68.8,
        "historical_weight_change": 0.5,
        "dietary_text": (
            "breakfast egg tomato rice "
            "lunch cabbage soup "
            "supper chicken"
        ),
    }

    result = predict_future_weight(sample_input)

    print(f"Predicted future weight: {result:.2f} kg")
