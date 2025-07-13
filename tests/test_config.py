import pytest
import yaml
import tempfile
from pathlib import Path
from src.config.config import load_config, Config

def test_load_config():
    config = load_config()
    
    assert hasattr(config, 'model')
    assert hasattr(config, 'data')
    assert hasattr(config, 'mlflow')
    assert hasattr(config, 'api')
    assert hasattr(config, 'logging')
    
    assert config.model.type == "lightgbm"
    assert config.data.delay_threshold == 15
    assert config.data.test_size == 0.2
    assert config.mlflow.experiment_name == "flight_delay_prediction"

def test_config_validation():
    config = load_config()
    
    assert isinstance(config.model.params, dict)
    assert isinstance(config.data.categorical_features, list)
    assert isinstance(config.data.feature_columns, list)
    assert isinstance(config.data.delay_threshold, int)
    assert isinstance(config.data.test_size, float)
    
    assert 0 < config.data.test_size < 1
    assert config.data.delay_threshold > 0
    assert len(config.data.categorical_features) > 0
    assert len(config.data.feature_columns) > 0

def test_custom_config_file():
    custom_config = {
        'model': {
            'type': 'lightgbm',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'random_state': 123
            }
        },
        'data': {
            'delay_threshold': 20,
            'test_size': 0.3,
            'random_state': 123,
            'categorical_features': ['AIRLINE'],
            'feature_columns': ['AIRLINE', 'DISTANCE'],
            'selected_columns': ['AIRLINE', 'DISTANCE', 'ARRIVAL_DELAY', 'arr_delayed']
        },
        'mlflow': {
            'experiment_name': 'test_experiment',
            'run_name': 'test_run'
        },
        'api': {
            'title': 'Test API',
            'description': 'Test Description',
            'version': '1.0.0',
            'host': '0.0.0.0',
            'port': 9000
        },
        'logging': {
            'level': 'DEBUG',
            'format': 'TEST: %(message)s'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(custom_config, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        
        assert config.model.params['n_estimators'] == 200
        assert config.data.delay_threshold == 20
    finally:
        Path(temp_path).unlink(missing_ok=True)