import joblib
import pandas as pd
from src.config.paths import MODEL_FILE
from src.data.preprocess import CATEGORICAL_FEATURES

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

    for col in CATEGORICAL_FEATURES:
        sample[col] = sample[col].astype("category")

    prediction = model.predict(sample)
    assert prediction.shape == (1,)
