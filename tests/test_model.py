import joblib
import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "lgbm_model.pkl"

def test_model_loading():
    assert os.path.exists(MODEL_FILE), "Модель не найдена"
    model = joblib.load(MODEL_FILE)
    assert hasattr(model, "predict")

def test_model_prediction_shape():
    model = joblib.load(MODEL_FILE)

    sample = pd.DataFrame([{
        "AIRLINE": "AA",
        "ORIGIN_AIRPORT": "JFK",
        "DESTINATION_AIRPORT": "LAX",
        "SCHEDULED_DEPARTURE": 700,
        "SCHEDULED_ARRIVAL": 900,
        "SCHEDULED_TIME": 300.0,
        "MONTH": 1,
        "DAY": 15,
        "DAY_OF_WEEK": 3
    }])

    categorical = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    for col in categorical:
        sample[col] = sample[col].astype("category")

    pred = model.predict(sample)
    assert pred.shape == (1,)

def test_model_prediction_output_values():
    model = joblib.load(MODEL_FILE)

    sample = pd.DataFrame([{
        "AIRLINE": "AA",
        "ORIGIN_AIRPORT": "JFK",
        "DESTINATION_AIRPORT": "LAX",
        "SCHEDULED_DEPARTURE": 700,
        "SCHEDULED_ARRIVAL": 900,
        "SCHEDULED_TIME": 300.0,
        "MONTH": 1,
        "DAY": 15,
        "DAY_OF_WEEK": 3
    }])

    for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        sample[col] = sample[col].astype("category")

    prob = model.predict_proba(sample)[0][1]
    assert 0.0 <= prob <= 1.0, "Вероятность вне допустимого диапазона"
