import pandas as pd
from src.data.prepare import prepare_flights_data
from src.data.preprocess import preprocess_flights
from src.model.train import train_lgbm_model
from src.model.evaluate import evaluate_model
from src.model.persist import save_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def full_training_pipeline():
    logger.info("Запуск полного пайплайна обучения")

    df = prepare_flights_data()

    X, y = preprocess_flights(df)

    model = train_lgbm_model()

    evaluate_model(model, X, y)

    save_model(model)

    logger.info("Пайплайн завершён")
