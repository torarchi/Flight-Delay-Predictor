from prefect import flow, task
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "flights_prepared.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "lgbm_model.pkl")


@task
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df


@task
def preprocess(df):
    df = df.dropna(subset=["ARRIVAL_DELAY"])
    df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] > 15).astype(int)

    X = df.loc[
        :,
        [
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "SCHEDULED_DEPARTURE",
            "SCHEDULED_TIME",
            "DISTANCE",
            "MONTH",
            "DAY",
            "DAY_OF_WEEK",
        ],
    ].copy()

    y = df["IS_DELAYED"]

    categorical = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    for col in categorical:
        X[col] = X[col].astype("category")

    return X, y


@task
def train_model(X, y):
    model = lgb.LGBMClassifier()
    model.fit(X, y)
    return model


@task
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    roc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc:.4f}")
    return acc, roc


@task
def save_model(model):
    joblib.dump(model, MODEL_FILE)
    print(f"Модель сохранена: {MODEL_FILE}")


@flow
def train_flow():
    df = load_data()
    X, y = preprocess(df)
    model = train_model(X, y)
    evaluate_model(model, X, y)
    save_model(model)


if __name__ == "__main__":
    train_flow()
