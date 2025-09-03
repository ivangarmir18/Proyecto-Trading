"""
Tests unitarios para el módulo AI training
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.ai_train import AITrainer

@pytest.fixture
def ai_trainer():
    storage = Mock()
    return AITrainer(storage, lookback_days=30, test_size=0.2)

def test_ai_trainer_initialization(ai_trainer):
    """Test de inicialización del AI Trainer"""
    assert ai_trainer.lookback_days == 30
    assert ai_trainer.test_size == 0.2
    assert ai_trainer.model is None
    assert ai_trainer.scaler is None

@patch('core.ai_train.pd.read_sql_query')
@patch('core.ai_train.AITrainer._get_scores')
@patch('core.ai_train.AITrainer._get_indicators') 
@patch('core.ai_train.AITrainer._merge_dataframes')
def test_load_training_data(mock_merge, mock_indicators, mock_scores, mock_read_sql, ai_trainer):
    """Test de carga de datos de entrenamiento con mocks"""
    # Mock de datos de entrenamiento
    mock_data = pd.DataFrame({
        'ts': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'open': [100, 101], 'high': [105, 106], 'low': [95, 96],
        'close': [102, 103], 'volume': [1000, 1001],
        'ema9': [101, 102], 'ema40': [100, 101], 'atr': [1.0, 1.1],
        'rsi': [50, 55], 'support': [95, 96], 'resistance': [105, 106]
    })
    
    # Configurar mocks
    mock_merge.return_value = mock_data
    mock_scores.return_value = pd.DataFrame()
    mock_indicators.return_value = pd.DataFrame()
    mock_read_sql.return_value = mock_data
    
    # Ejecutar prueba
    df = ai_trainer._load_training_data("BTCUSDT", "5m")
    
    # Verificaciones
    assert not df.empty
    assert len(df) == 2
    assert 'open' in df.columns
    assert 'close' in df.columns
    assert 'rsi' in df.columns

def test_calculate_features(ai_trainer):
    """Test de cálculo de features técnicas"""
    # Datos de prueba
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105],
        'high': [105, 106, 107, 108, 109, 110],
        'low': [95, 96, 97, 98, 99, 100],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500]
    })
    
    result = ai_trainer._calculate_features(df)
    
    # Verificar que se calcularon features
    assert 'returns' in result.columns
    assert 'log_returns' in result.columns
    assert 'volatility_20' in result.columns
    assert 'sma_20' in result.columns
    assert 'rsi' in result.columns

@patch('core.ai_train.lgb.train')
@patch('core.ai_train.train_test_split')
def test_train_model(mock_split, mock_lgb, ai_trainer):
    """Test de entrenamiento del modelo"""
    # Mock de datos
    X = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature2': [0.5, 0.4, 0.3, 0.2, 0.1]
    })
    y = pd.Series([0, 1, 0, 1, 0])
    
    # Mock de split
    mock_split.return_value = (X, X, y, y)
    
    # Mock de LightGBM
    mock_model = Mock()
    mock_model.predict.return_value = [0, 1, 0, 1, 0]
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    mock_lgb.return_value = mock_model
    
    # Ejecutar entrenamiento
    metrics = ai_trainer.train_model(X, y)
    
    # Verificaciones
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert 'f1_score' in metrics
    assert 0 <= metrics['accuracy'] <= 1

@patch('core.ai_train.joblib.dump')
@patch('core.ai_train.datetime')
def test_save_model(mock_datetime, mock_dump, ai_trainer):
    """Test de guardado del modelo"""
    mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
    
    # Mock del modelo
    ai_trainer.model = Mock()
    ai_trainer.scaler = Mock()
    ai_trainer.feature_columns = ['feature1', 'feature2']
    
    metrics = {'accuracy': 0.85, 'roc_auc': 0.90}
    result = ai_trainer.save_model("BTCUSDT", "5m", metrics)
    
    # Verificar que se llamó a joblib.dump
    assert mock_dump.call_count >= 2
    assert 'model_path' in result
    assert 'metrics' in result

@patch('core.ai_train.AITrainer.prepare_dataset')
@patch('core.ai_train.AITrainer.train_model')
@patch('core.ai_train.AITrainer.save_model')
def test_full_training_pipeline(mock_save, mock_train, mock_prepare, ai_trainer):
    """Test del pipeline completo de entrenamiento"""
    # Mock de datos
    X = pd.DataFrame({'feature1': [0.1, 0.2], 'feature2': [0.3, 0.4]})
    y = pd.Series([0, 1])
    
    mock_prepare.return_value = (X, y, ['feature1', 'feature2'])
    mock_train.return_value = {'accuracy': 0.85, 'roc_auc': 0.90}
    mock_save.return_value = {'model_path': 'test.model', 'metrics': {'accuracy': 0.85}}
    
    # Ejecutar pipeline
    result = ai_trainer.full_training_pipeline("BTCUSDT", "5m")
    
    # Verificaciones
    assert 'model_path' in result
    assert 'metrics' in result
    assert result['metrics']['accuracy'] == 0.85