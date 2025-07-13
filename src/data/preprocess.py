import pandas as pd
from src.config.config import config

def preprocess_flights(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=["ARRIVAL_DELAY"])
    
    df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] > config.data.delay_threshold).astype(int)

    X = df[config.data.feature_columns].copy()
    y = df["IS_DELAYED"]

    for col in config.data.categorical_features:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y

CATEGORICAL_FEATURES = config.data.categorical_features
FEATURE_COLUMNS = config.data.feature_columns