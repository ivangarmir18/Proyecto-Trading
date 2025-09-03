"""
Tests unitarios para el módulo score
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.score import AIScoringSystem

@pytest.fixture
def scoring_system():
    storage = Mock()
    config = {
        "weights": {
            "trend": 0.25,
            "support": 0.20,
            "momentum": 0.15,
            "volatility": 0.10,
            "volume": 0.10,
            "ai_confidence": 0.20
        },
        "ai": {
            "model_dir": "models",
            "default_confidence": 0.5
        }
    }
    return AIScoringSystem(storage, config)

def test_scoring_system_initialization(scoring_system):
    assert scoring_system.weights["trend"] == 0.25
    assert scoring_system.weights["support"] == 0.20
    assert scoring_system.ai_config["default_confidence"] == 0.5

def test_calculate_technical_indicators(scoring_system):
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'ema9': [99, 100, 101, 102, 103],
        'ema40': [98, 99, 100, 101, 102],
        'support': [95, 96, 97, 98, 99],
        'resistance': [105, 106, 107, 108, 109],
        'atr': [1.0, 1.1, 1.2, 1.3, 1.4],
        'rsi': [50, 55, 60, 65, 70],
        'macd_hist': [0.1, 0.2, 0.3, 0.4, 0.5],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })

    result = scoring_system.calculate_technical_indicators(df)

    assert 'trend_score' in result.columns
    assert 'support_proximity_score' in result.columns
    assert 'momentum_score' in result.columns
    assert 'volatility_score' in result.columns
    assert 'volume_score' in result.columns

@patch('core.score.joblib.load')
def test_calculate_ai_confidence(mock_load, scoring_system):
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6]])
    mock_load.return_value = mock_model

    df = pd.DataFrame({
        'feature1': [0.1, 0.2],
        'feature2': [0.3, 0.4]
    })

    # Mock del modelo
    scoring_system.ai_models = {
        'BTCUSDT_5m': {
            'model': mock_model,
            'scaler': None,
            'metadata': {'feature_columns': ['feature1', 'feature2']}
        }
    }

    confidence = scoring_system.calculate_ai_confidence(df, "BTCUSDT", "5m")

    assert len(confidence) == 2
    assert all(0 <= c <= 1 for c in confidence)