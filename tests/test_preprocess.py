import pandas as pd
from src.data.preprocess import preprocess_flights

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
    assert y.tolist() == [1, 0]
