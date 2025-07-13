from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.config.paths import MODEL_FILE
from src.model.persist import load_model
from src.data.preprocess import CATEGORICAL_FEATURES
from src.utils.logger import get_logger
from src.serve.schemas import FlightInput

logger = get_logger("serve")

app = FastAPI(title="Flight Delay Predictor")

@app.post("/predict")
def predict_delay(flight: FlightInput):
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    data = flight.model_dump()
    df = pd.DataFrame([data])

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    prob = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    return {
        "delay_probability": round(prob, 4),
        "will_be_delayed": bool(prediction)
    }
