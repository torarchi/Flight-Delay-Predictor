# Flight Delay Predictor

  FastAPI-сервис с ML-моделью для прогнозирования вероятности задержки авиарейса. 
  В основе лежит LightGBM, модель обучена на признаках рейса, включая авиакомпанию, аэропорты, время вылета и дистанцию. 
  
  
Данные взял с kaggle - 
https://www.kaggle.com/datasets/usdot/flight-delays?resource=download

---

Залил на render.com. Можно протестировать api

https://flight-delay-predictor-mvhf.onrender.com/docs

Пример запроса:

```
{
  "MONTH": 7,
  "DAY": 15,
  "DAY_OF_WEEK": 1,
  "AIRLINE": "AA",
  "ORIGIN_AIRPORT": "JFK",
  "DESTINATION_AIRPORT": "LAX",
  "SCHEDULED_DEPARTURE": 830,
  "DISTANCE": 2475,
  "SCHEDULED_TIME": 360.0
}
```

пример ответа:

{
  "delay_probability": 0.1662,
  "will_be_delayed": false
}

## Стек

- Python 3.13  
- ML: LightGBM, scikit-learn  
- Data: pandas, joblib, kaggle  
- API: FastAPI  
- Оркестрация: Prefect  
- Тестирование: pytest  
- Отслеживание: MLflow  
- DevTools: Pydantic v2, uvicorn


---

## Как запустить

### 1. Установка зависимостей

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Подготовка данных
```bash
python app/preprocess/prepare_training_data.py
```

### 3. Обучение модели через Prefect

#### Выполняет весь pipeline: загрузка, препроцессинг, обучение и сохранение модели.
```bash
python app/etl/train_flow.py
```
Модель будет сохранена в models/lgbm_model.pkl.


### 3.1 Обучение модели вручную
```bash
python app/train/train_model.py
```

### 4. Запуск API
```bash
uvicorn serve.app:app --reload
```
 http://127.0.0.1:8000/docs - swagger

### 5. tests
```bash
pytest -v
```
