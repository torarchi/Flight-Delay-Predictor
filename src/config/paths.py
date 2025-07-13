from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_FILE = PROCESSED_DATA_DIR / "flights_prepared.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "lgbm_model.pkl"

MLRUNS_DIR = BASE_DIR / "mlruns"
