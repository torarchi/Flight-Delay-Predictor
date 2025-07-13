import pandas as pd
from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_FILE
from src.config.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_flights_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "flights.csv", low_memory=False)

    df["arr_delayed"] = df["ARRIVAL_DELAY"] > config.data.delay_threshold

    df = df[config.data.selected_columns].dropna()
    
    df["SCHEDULED_DEPARTURE"] = df["SCHEDULED_DEPARTURE"].astype(int)
    df["SCHEDULED_TIME"] = df["SCHEDULED_TIME"].astype(float)
    df["DISTANCE"] = df["DISTANCE"].astype(float)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

    logger.info(f"Предобработка завершена: {len(df):,} строк")
    logger.info(f"Порог задержки: {config.data.delay_threshold} минут")
    logger.info(f"Сохранено в {PROCESSED_FILE}")
    
    return df

if __name__ == "__main__":
    prepare_flights_data()