import pandas as pd
from src.config.paths import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_datasets() -> dict[str, pd.DataFrame]:
    flights = pd.read_csv(RAW_DATA_DIR / "flights.csv")
    airlines = pd.read_csv(RAW_DATA_DIR / "airlines.csv")
    airports = pd.read_csv(RAW_DATA_DIR / "airports.csv")

    logger.info(f"Загружено {len(flights):,} рейсов")
    logger.info(f"Авиакомпаний: {len(airlines)}")
    logger.info(f"Аэропортов: {len(airports)}")

    return {
        "flights": flights,
        "airlines": airlines,
        "airports": airports
    }
