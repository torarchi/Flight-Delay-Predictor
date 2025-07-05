import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")


def load_datasets():
    flights = pd.read_csv(RAW_DIR / "flights.csv")
    airlines = pd.read_csv(RAW_DIR / "airlines.csv")
    airports = pd.read_csv(RAW_DIR / "airports.csv")

    print(f"Загружено: {len(flights):,} рейсов")
    print(f"Авиакомпаний: {len(airlines)}")
    print(f"Аэропортов: {len(airports)}")

    print("\nПример данных о рейсах:")
    print(flights.head(3).to_string(index=False))


if __name__ == "__main__":
    load_datasets()
