import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DIR / "flights_prepared.csv"

def preprocess_flights():
    df = pd.read_csv(RAW_DIR / "flights.csv", low_memory=False)

    # Целевая переменная: задержка прилёта > 15 минут
    df["arr_delayed"] = df["ARRIVAL_DELAY"] > 15

    cols = [
        "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE",
        "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE", "DISTANCE", "SCHEDULED_TIME",
        "ARRIVAL_DELAY", "arr_delayed"
    ]
    df = df[cols]

    df = df.dropna()

    df["SCHEDULED_DEPARTURE"] = df["SCHEDULED_DEPARTURE"].astype(int)
    df["SCHEDULED_TIME"] = df["SCHEDULED_TIME"].astype(float)
    df["DISTANCE"] = df["DISTANCE"].astype(float)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

    print(f"Предобработка завершена: {len(df):,} строк")
    print(f"Сохранено в {PROCESSED_FILE}")

if __name__ == "__main__":
    preprocess_flights()
