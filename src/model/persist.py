import joblib
from src.config.paths import MODEL_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_model(model) -> None:
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    logger.info(f"Модель сохранена: {MODEL_FILE}")

def load_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_FILE}")
