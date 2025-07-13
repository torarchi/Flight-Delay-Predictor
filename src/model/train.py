import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.lightgbm

from src.config.paths import PROCESSED_FILE, MLRUNS_DIR
from src.data.preprocess import CATEGORICAL_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_lgbm_model() -> lgb.LGBMClassifier:
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("flight_delay_prediction")

    df = pd.read_csv(PROCESSED_FILE)
    y = df["arr_delayed"]
    X = df.drop(columns=["ARRIVAL_DELAY", "arr_delayed"])

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": 42,
        })

        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.lightgbm.log_model(model, "model")

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"ROC AUC:  {auc:.4f}")

    return model

if __name__ == "__main__":
    train_lgbm_model()