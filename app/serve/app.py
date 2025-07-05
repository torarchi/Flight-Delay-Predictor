from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgbm_model.pkl"
model = joblib.load(MODEL_PATH)


CATEGORICAL = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]


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


if __name__ == "__main__":
    uvicorn.run("serve.app:app", host="127.0.0.1", port=8000, reload=True)
