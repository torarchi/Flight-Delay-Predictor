from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.config.paths import MODEL_FILE
from src.config.config import config
from src.model.persist import load_model
from src.utils.logger import get_logger
from src.serve.schemas import FlightInput

logger = get_logger("serve")

app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version
)

@app.get("/")
def root():
    return {
        "message": "Flight Delay Predictor API",
        "version": config.api.version,
        "delay_threshold": config.data.delay_threshold
    }

@app.get("/config")
def get_config():
    return {
        "model_type": config.model.type,
        "model_params": config.model.params,
        "delay_threshold": config.data.delay_threshold,
        "features": config.data.feature_columns,
        "categorical_features": config.data.categorical_features
    }

@app.post("/predict")
def predict_delay(flight: FlightInput):
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    data = flight.model_dump()
    df = pd.DataFrame([data])

    for col in config.data.categorical_features:
        if col in df.columns:
            df[col] = df[col].astype("category")

    prob = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    return {
        "delay_probability": round(prob, 4),
        "will_be_delayed": bool(prediction),
        "threshold_minutes": config.data.delay_threshold,
        "model_type": config.model.type
    }

@app.get("/health")
def health_check():
    try:
        model = load_model()
        return {"status": "healthy", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "unhealthy", "model_loaded": False}