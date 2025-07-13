#  Flight Delay Predictor

ML-сервис, предсказывающий вероятность задержки рейса на основе табличных признаков: авиакомпания, аэропорты, время вылета, расстояние и др.  
Модель обучена с помощью LightGBM.

Данные: [US DOT Flight Delays – Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays?resource=download)

---

-  Python 3.13  
-  ML: LightGBM, scikit-learn  
-  Data: pandas, joblib  
-  API: FastAPI  
-  Оркестрация: Prefect  
-  Тестирование: pytest  
-  Отслеживание: MLflow  
-  Валидация: Pydantic v2  
-  ASGI сервер: Uvicorn

---


##  Как запустить

### 1. Установка зависимостей

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Подготовка данных

```bash
python -m src.data.prepare
```

### 3. Запуск полного пайплайна обучения (через Prefect)

```bash
python main.py
```

Модель будет сохранена в `models/lgbm_model.pkl`.

### 3.1 Обучение вручную через MLflow

```bash
python -m src.model.train
```

MLflow логирует параметры, метрики и модель в директорию `mlruns/`.

Для запуска интерфейса отслеживания MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns
```

Далее перейти по адресу: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

##  Запуск API

```bash
uvicorn src.serve.api:app --reload
```

Документация доступна по адресу:  
📎 http://127.0.0.1:8000/docs

---

##  Запуск тестов

```bash
pytest -v
```

---

Модель обучается только на **признаках, известных до вылета**

---