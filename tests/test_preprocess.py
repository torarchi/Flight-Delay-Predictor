import pandas as pd
from etl.train_flow import preprocess

def test_preprocess_output_shapes():
    df = pd.DataFrame({
        "AIRLINE": ["AA", "DL"],
        "ORIGIN_AIRPORT": ["JFK", "LAX"],
        "DESTINATION_AIRPORT": ["LAX", "ORD"],
        "SCHEDULED_DEPARTURE": [700, 800],
        "SCHEDULED_ARRIVAL": [900, 1000],
        "DISTANCE": [2475, 1745],
        "SCHEDULED_TIME": [300.0, 200.0],
        "MONTH": [1, 1],
        "DAY": [15, 15],
        "DAY_OF_WEEK": [3, 3],
        "ARRIVAL_DELAY": [16, 0]
    })

    X, y = preprocess(df)

    assert X.shape[0] == y.shape[0] == 2
    assert "AIRLINE" in X.columns
    assert y.tolist() == [1, 0]

def test_preprocess_with_missing_values():
    df = pd.DataFrame({
        "AIRLINE": ["AA", None],
        "ORIGIN_AIRPORT": ["JFK", "LAX"],
        "DESTINATION_AIRPORT": ["LAX", "ORD"],
        "SCHEDULED_DEPARTURE": [700, None],
        "SCHEDULED_ARRIVAL": [900, 1000],
        "DISTANCE": [2475, 1745],
        "SCHEDULED_TIME": [300.0, 200.0],
        "MONTH": [1, 1],
        "DAY": [15, 15],
        "DAY_OF_WEEK": [3, 3],
        "ARRIVAL_DELAY": [16, 0]
    })

    X, y = preprocess(df.dropna())
    assert X.shape[0] == y.shape[0] == 1
