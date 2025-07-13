import pandas as pd
from src.data.preprocess import preprocess_flights
from src.config.config import config

def test_preprocess_output_shapes():
    df = pd.DataFrame({
        "AIRLINE": ["AA", "DL"],
        "ORIGIN_AIRPORT": ["JFK", "LAX"],
        "DESTINATION_AIRPORT": ["LAX", "ORD"],
        "SCHEDULED_DEPARTURE": [700, 800],
        "SCHEDULED_TIME": [300.0, 200.0],
        "DISTANCE": [2475, 1745],
        "MONTH": [1, 1],
        "DAY": [15, 15],
        "DAY_OF_WEEK": [3, 3],
        "ARRIVAL_DELAY": [16, 0]
    })

    X, y = preprocess_flights(df)
    assert X.shape[0] == y.shape[0] == 2
    assert "AIRLINE" in X.columns
    
    expected_delays = [1 if delay > config.data.delay_threshold else 0 for delay in [16, 0]]
    assert y.tolist() == expected_delays

def test_categorical_features_processing():
    df = pd.DataFrame({
        "AIRLINE": ["AA", "DL"],
        "ORIGIN_AIRPORT": ["JFK", "LAX"],
        "DESTINATION_AIRPORT": ["LAX", "ORD"],
        "SCHEDULED_DEPARTURE": [700, 800],
        "SCHEDULED_TIME": [300.0, 200.0],
        "DISTANCE": [2475, 1745],
        "MONTH": [1, 1],
        "DAY": [15, 15],
        "DAY_OF_WEEK": [3, 3],
        "ARRIVAL_DELAY": [16, 0]
    })

    X, y = preprocess_flights(df)
    
    for col in config.data.categorical_features:
        if col in X.columns:
            assert X[col].dtype.name == "category"

def test_configurable_threshold():
    df = pd.DataFrame({
        "AIRLINE": ["AA"],
        "ORIGIN_AIRPORT": ["JFK"],
        "DESTINATION_AIRPORT": ["LAX"],
        "SCHEDULED_DEPARTURE": [700],
        "SCHEDULED_TIME": [300.0],
        "DISTANCE": [2475],
        "MONTH": [1],
        "DAY": [15],
        "DAY_OF_WEEK": [3],
        "ARRIVAL_DELAY": [config.data.delay_threshold + 1]
    })

    X, y = preprocess_flights(df)
    assert y.iloc[0] == 1
    
    df["ARRIVAL_DELAY"] = [config.data.delay_threshold - 1]
    X, y = preprocess_flights(df)
    assert y.iloc[0] == 0