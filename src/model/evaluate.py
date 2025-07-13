from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import lightgbm as lgb
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    roc = roc_auc_score(y, y_prob)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"ROC AUC:  {roc:.4f}")

    return acc, roc
