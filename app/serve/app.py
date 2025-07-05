from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgbm_model.pkl"

CATEGORICAL = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

model = None


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model


class FlightInput(BaseModel):
    MONTH: int
    DAY: int
    DAY_OF_WEEK: int
    AIRLINE: str
    ORIGIN_AIRPORT: str
    DESTINATION_AIRPORT: str
    SCHEDULED_DEPARTURE: int
    DISTANCE: float
    SCHEDULED_TIME: float


@app.post("/predict")
def predict_delay(flight: FlightInput):
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model is not available")

    data = flight.model_dump()
    df = pd.DataFrame([data])

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")

    prob = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    return {
        "delay_probability": round(prob, 4),
        "will_be_delayed": bool(prediction)
    }
