from fastapi.testclient import TestClient
from serve.app import app

client = TestClient(app)

def test_predict_success():
    response = client.post("/predict", json={
        "AIRLINE": "AA",
        "ORIGIN_AIRPORT": "JFK",
        "DESTINATION_AIRPORT": "LAX",
        "SCHEDULED_DEPARTURE": 700,
        "SCHEDULED_ARRIVAL": 900,
        "DISTANCE": 2475,
        "SCHEDULED_TIME": 300.0,
        "MONTH": 1,
        "DAY": 15,
        "DAY_OF_WEEK": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert "delay_probability" in data
    assert "will_be_delayed" in data

def test_predict_invalid():
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422

def test_missing_fields():
    response = client.post("/predict", json={
        "AIRLINE": "AA",
        "ORIGIN_AIRPORT": "JFK"
        # остальных полей нет
    })
    assert response.status_code == 422

def test_wrong_data_types():
    response = client.post("/predict", json={
        "AIRLINE": 123,  # должно быть строкой
        "ORIGIN_AIRPORT": "JFK",
        "DESTINATION_AIRPORT": "LAX",
        "SCHEDULED_DEPARTURE": "not_a_number",
        "DISTANCE": "far",
        "SCHEDULED_TIME": "long",
        "MONTH": "Jan",
        "DAY": "Monday",
        "DAY_OF_WEEK": "Three"
    })
    assert response.status_code == 422

def test_predict_labels():
    response = client.post("/predict", json={
        "AIRLINE": "DL",
        "ORIGIN_AIRPORT": "ATL",
        "DESTINATION_AIRPORT": "ORD",
        "SCHEDULED_DEPARTURE": 1200,
        "DISTANCE": 800,
        "SCHEDULED_TIME": 120.0,
        "MONTH": 6,
        "DAY": 10,
        "DAY_OF_WEEK": 4
    })
    json_data = response.json()
    assert isinstance(json_data["delay_probability"], float)
    assert isinstance(json_data["will_be_delayed"], bool)

def test_predict_boundary_values():
    response = client.post("/predict", json={
        "AIRLINE": "AA",
        "ORIGIN_AIRPORT": "JFK",
        "DESTINATION_AIRPORT": "LAX",
        "SCHEDULED_DEPARTURE": 0,
        "SCHEDULED_ARRIVAL": 1440,
        "DISTANCE": 0,
        "SCHEDULED_TIME": 0.0,
        "MONTH": 1,
        "DAY": 1,
        "DAY_OF_WEEK": 1
    })
    assert response.status_code == 200
    assert "delay_probability" in response.json()

def test_predict_with_missing_fields():
    response = client.post("/predict", json={
        "AIRLINE": "AA"
    })
    assert response.status_code == 422

def test_predict_with_wrong_types():
    response = client.post("/predict", json={
        "AIRLINE": 123,  # ожидалась строка
        "ORIGIN_AIRPORT": "JFK",
        "DESTINATION_AIRPORT": "LAX",
        "SCHEDULED_DEPARTURE": "seven",
        "DISTANCE": "far",
        "SCHEDULED_ARRIVAL": "soon",
        "SCHEDULED_TIME": "fast",
        "MONTH": "Jan",
        "DAY": "Monday",
        "DAY_OF_WEEK": "first"
    })
    assert response.status_code == 422
