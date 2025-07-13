import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from src.config.paths import BASE_DIR

@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]

@dataclass
class DataConfig:
    delay_threshold: int
    test_size: float
    random_state: int
    categorical_features: list
    feature_columns: list
    selected_columns: list

@dataclass
class MLflowConfig:
    experiment_name: str
    run_name: str

@dataclass
class APIConfig:
    title: str
    description: str
    version: str
    host: str
    port: int

@dataclass
class LoggingConfig:
    level: str
    format: str

@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    mlflow: MLflowConfig
    api: APIConfig
    logging: LoggingConfig

def load_config(config_path: str = None) -> Config:
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    
    return Config(
        model=ModelConfig(**config_dict['model']),
        data=DataConfig(**config_dict['data']),
        mlflow=MLflowConfig(**config_dict['mlflow']),
        api=APIConfig(**config_dict['api']),
        logging=LoggingConfig(**config_dict['logging'])
    )

config = load_config()