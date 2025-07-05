import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path
import joblib
import mlflow
import mlflow.lightgbm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MLRUNS_PATH = BASE_DIR / "mlruns"
DATA_FILE = BASE_DIR.parent / "data" / "processed" / "flights_prepared.csv"
MODEL_FILE = BASE_DIR / "models" / "lgbm_model.pkl"


def train_model():
    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    mlflow.set_experiment("flight_delay_prediction")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Целевая переменная
    y = df["arr_delayed"]
    X = df.drop(columns=["ARRIVAL_DELAY", "arr_delayed"])

    categorical = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    for col in categorical:
        X[col] = X[col].astype("category")

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Лог
        mlflow.log_params(
            {
                "model": "LightGBM",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": 42,
            }
        )

        # Обучение
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train, categorical_feature=categorical)

        # Предсказание
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Метрики
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        mlflow.lightgbm.log_model(model, "model")
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_FILE)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC:  {auc:.4f}")
        print(f"Модель сохранена: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()
