import pandas as pd

CATEGORICAL_FEATURES = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
FEATURE_COLUMNS = [
    "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE",
    "MONTH", "DAY", "DAY_OF_WEEK"
]

def preprocess_flights(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=["ARRIVAL_DELAY"])
    df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] > 15).astype(int)

    X = df[FEATURE_COLUMNS].copy()
    y = df["IS_DELAYED"]

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    return X, y
