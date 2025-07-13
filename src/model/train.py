import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.lightgbm

from src.config.paths import PROCESSED_FILE, MLRUNS_DIR
from src.config.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_lgbm_model() -> lgb.LGBMClassifier:
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(config.mlflow.experiment_name)

    df = pd.read_csv(PROCESSED_FILE)
    y = df["arr_delayed"]
    X = df.drop(columns=["ARRIVAL_DELAY", "arr_delayed"])

    for col in config.data.categorical_features:
        if col in X.columns:
            X[col] = X[col].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.data.test_size, 
        random_state=config.data.random_state
    )

    model = lgb.LGBMClassifier(**config.model.params)

    with mlflow.start_run(run_name=config.mlflow.run_name):
        mlflow.log_params(config.model.params)
        mlflow.log_param("delay_threshold", config.data.delay_threshold)
        mlflow.log_param("test_size", config.data.test_size)
        
        model.fit(
            X_train, y_train, 
            categorical_feature=config.data.categorical_features
        )
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        mlflow.lightgbm.log_model(model, "model")

        logger.info(f"Модель обучена: {config.model.type}")
        logger.info(f"Точность: {acc:.4f}")
        logger.info(f"ROC AUC: {auc:.4f}")
        logger.info(f"Параметры: {config.model.params}")

    return model

if __name__ == "__main__":
    train_lgbm_model()