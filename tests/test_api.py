from fastapi.testclient import TestClient
from src.serve.api import app
from src.config.config import config

client = TestClient(app)

valid_payload = {
    "AIRLINE": "AA",
    "ORIGIN_AIRPORT": "JFK",
    "DESTINATION_AIRPORT": "LAX",
    "SCHEDULED_DEPARTURE": 700,
    "DISTANCE": 2475,
    "SCHEDULED_TIME": 300.0,
    "MONTH": 1,
    "DAY": 15,
    "DAY_OF_WEEK": 3,
}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["delay_threshold"] == config.data.delay_threshold

def test_config_endpoint():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == config.model.type
    assert data["delay_threshold"] == config.data.delay_threshold
    assert "features" in data
    assert "categorical_features" in data

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_success():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "delay_probability" in data
    assert "will_be_delayed" in data
    assert "threshold_minutes" in data
    assert "model_type" in data
    assert data["threshold_minutes"] == config.data.delay_threshold

def test_missing_fields():
    response = client.post("/predict", json={"AIRLINE": "AA"})
    assert response.status_code == 422

def test_wrong_types():
    response = client.post(
        "/predict",
        json={
            "AIRLINE": 123,
            "ORIGIN_AIRPORT": "JFK",
            "DESTINATION_AIRPORT": "LAX",
            "SCHEDULED_DEPARTURE": "seven",
            "DISTANCE": "far",
            "SCHEDULED_TIME": "fast",
            "MONTH": "Jan",
            "DAY": "Monday",
            "DAY_OF_WEEK": "first",
        },
    )
    assert response.status_code == 422