"""
Tests de integración para el orchestrator
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

from core.orchestrator import TradingOrchestrator

@pytest.fixture
def mock_config():
    return {
        "app": {"default_limit": 100},
        "api": {
            "binance": {"api_key": "test", "api_secret": "test", "rate_limit_per_min": 1200},
            "finnhub": {"keys": ["key1", "key2"]}
        },
        "assets": {
            "crypto": ["BTCUSDT", "ETHUSDT"],
            "stocks": ["AAPL", "TSLA"]
        },
        "scheduler": {"processing_interval_minutes": 15},
        "ai": {"enabled": True, "retrain_days": 30},
        "retention_days": {"5m": 30, "1h": 90, "1d": 365}
    }

@patch('core.orchestrator.load_config')
@patch('core.orchestrator.make_storage_from_env')
@patch('core.orchestrator.Fetcher')
def test_orchestrator_integration(mock_fetcher, mock_storage, mock_load_config, mock_config):
    mock_load_config.return_value = mock_config
    
    # Mocks para storage y fetcher
    mock_storage_instance = Mock()
    mock_storage.return_value = mock_storage_instance
    
    mock_fetcher_instance = Mock()
    mock_fetcher.return_value = mock_fetcher_instance
    
    # Mock de datos
    mock_df = pd.DataFrame({
        'ts': [datetime.now() - timedelta(minutes=i) for i in range(100)],
        'open': [100 + i for i in range(100)],
        'high': [105 + i for i in range(100)],
        'low': [95 + i for i in range(100)],
        'close': [102 + i for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    })
    
    mock_storage_instance.get_ohlcv.return_value = mock_df
    mock_fetcher_instance.fetch_ohlcv.return_value = mock_df
    
    # Crear orchestrator
    orchestrator = TradingOrchestrator("test_config.json")
    
    # Probar fetch de datos
    results = orchestrator.fetch_data(["BTCUSDT", "ETHUSDT"], "5m")
    assert len(results) == 2
    assert "BTCUSDT" in results
    assert "ETHUSDT" in results
    
    # Probar procesamiento de asset
    with patch('core.orchestrator.compute_and_save_scores') as mock_scores:
        with patch('core.orchestrator.train_ai_model') as mock_ai:
            mock_scores.return_value = 25
            mock_ai.return_value = {"accuracy": 0.85}
            
            result = orchestrator.process_asset("BTCUSDT", "5m")
            assert result is True
            
            mock_scores.assert_called_once()
            mock_ai.assert_called_once()

