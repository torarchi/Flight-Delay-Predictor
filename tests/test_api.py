from fastapi.testclient import TestClient
from src.serve.api import app

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

def test_predict_success():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "delay_probability" in data
    assert "will_be_delayed" in data

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
