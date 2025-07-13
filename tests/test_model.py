import joblib
import pandas as pd
from src.config.paths import MODEL_FILE
from src.config.config import config

def test_model_exists():
    assert MODEL_FILE.exists(), "Модель не найдена"

def test_model_predicts():
    model = joblib.load(MODEL_FILE)

    sample = pd.DataFrame([{
        "AIRLINE": "AA",
        "ORIGIN_AIRPORT": "JFK",
        "DESTINATION_AIRPORT": "LAX",
        "SCHEDULED_DEPARTURE": 700,
        "DISTANCE": 2475,
        "SCHEDULED_TIME": 300.0,
        "MONTH": 1,
        "DAY": 15,
        "DAY_OF_WEEK": 3
    }])

    for col in config.data.categorical_features:
        if col in sample.columns:
            sample[col] = sample[col].astype("category")

    prediction = model.predict(sample)
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]

def test_model_probability():
    model = joblib.load(MODEL_FILE)

    sample = pd.DataFrame([{
        "AIRLINE": "AA",
        "ORIGIN_AIRPORT": "JFK",
        "DESTINATION_AIRPORT": "LAX",
        "SCHEDULED_DEPARTURE": 700,
        "DISTANCE": 2475,
        "SCHEDULED_TIME": 300.0,
        "MONTH": 1,
        "DAY": 15,
        "DAY_OF_WEEK": 3
    }])

    for col in config.data.categorical_features:
        if col in sample.columns:
            sample[col] = sample[col].astype("category")

    probabilities = model.predict_proba(sample)
    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0][1] <= 1