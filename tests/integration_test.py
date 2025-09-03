"""
tests/integration_test.py - Tests de integración para el sistema completo
Verifica que todos los componentes trabajen j correctamente
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import tempfile
import json
import os

from core.orchestrator import TradingOrchestrator, run_full_pipeline
from core.storage_postgres import PostgresStorage
from core.fetch import Fetcher

@pytest.fixture
def mock_config():
    """Configuración de prueba"""
    return {
        "app": {
            "default_limit": 100
        },
        "api": {
            "binance": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "rate_limit_per_min": 1200
            },
            "finnhub": {
                "keys": ["test_key_1", "test_key_2"]
            }
        },
        "assets": {
            "crypto": ["BTCUSDT", "ETHUSDT"],
            "stocks": ["AAPL", "TSLA"]
        },
        "scheduler": {
            "processing_interval_minutes": 15
        },
        "ai": {
            "enabled": True,
            "retrain_days": 30
        },
        "retention_days": {
            "5m": 30,
            "1h": 90,
            "1d": 365
        }
    }

@pytest.fixture
def mock_storage():
    """Mock de storage para pruebas"""
    storage = Mock(spec=PostgresStorage)
    storage.make_save_callback.return_value = Mock()
    
    # Mock de datos de ejemplo
    mock_df = pd.DataFrame({
        'ts': [datetime.now() - timedelta(minutes=i) for i in range(100)],
        'open': [100 + i for i in range(100)],
        'high': [105 + i for i in range(100)],
        'low': [95 + i for i in range(100)],
        'close': [102 + i for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    })
    
    storage.get_ohlcv.return_value = mock_df
    return storage

@pytest.fixture
def mock_fetcher():
    """Mock de fetcher para pruebas"""
    fetcher = Mock(spec=Fetcher)
    
    # Mock de datos de ejemplo
    mock_df = pd.DataFrame({
        'ts': [datetime.now() - timedelta(minutes=i) for i in range(50)],
        'open': [100 + i for i in range(50)],
        'high': [105 + i for i in range(50)],
        'low': [95 + i for i in range(50)],
        'close': [102 + i for i in range(50)],
        'volume': [1000 + i * 10 for i in range(50)]
    })
    
    fetcher.fetch_ohlcv.return_value = mock_df
    return fetcher

def test_orchestrator_initialization(mock_config, mock_storage, mock_fetcher):
    """Test de inicialización del orchestrator"""
    with patch('core.orchestrator.load_config', return_value=mock_config):
        with patch('core.orchestrator.make_storage_from_env', return_value=mock_storage):
            with patch('core.orchestrator.Fetcher', return_value=mock_fetcher):
                orchestrator = TradingOrchestrator("test_config.json")
                
                assert orchestrator is not None
                assert orchestrator.storage == mock_storage
                assert orchestrator.fetcher == mock_fetcher
                assert len(orchestrator.assets["crypto"]) == 2
                assert len(orchestrator.assets["stocks"]) == 2

def test_fetch_data(mock_config, mock_storage, mock_fetcher):
    """Test de obtención de datos"""
    with patch('core.orchestrator.load_config', return_value=mock_config):
        with patch('core.orchestrator.make_storage_from_env', return_value=mock_storage):
            with patch('core.orchestrator.Fetcher', return_value=mock_fetcher):
                orchestrator = TradingOrchestrator("test_config.json")
                
                assets = ["BTCUSDT", "ETHUSDT"]
                results = orchestrator.fetch_data(assets, "5m")
                
                assert len(results) == 2
                assert "BTCUSDT" in results
                assert "ETHUSDT" in results
                assert mock_fetcher.fetch_ohlcv.call_count == 2

def test_process_asset(mock_config, mock_storage, mock_fetcher):
    """Test de procesamiento de asset"""
    with patch('core.orchestrator.load_config', return_value=mock_config):
        with patch('core.orchestrator.make_storage_from_env', return_value=mock_storage):
            with patch('core.orchestrator.Fetcher', return_value=mock_fetcher):
                with patch('core.orchestrator.compute_and_save_scores') as mock_scores:
                    with patch('core.orchestrator.train_ai_model') as mock_ai:
                        orchestrator = TradingOrchestrator("test_config.json")
                        
                        # Mock de funciones externas
                        mock_scores.return_value = 25
                        mock_ai.return_value = {"accuracy": 0.85}
                        
                        # Procesar asset
                        result = orchestrator.process_asset("BTCUSDT", "5m")
                        
                        assert result == True
                        mock_scores.assert_called_once()
                        mock_ai.assert_called_once()

def test_system_status(mock_config, mock_storage, mock_fetcher):
    """Test de obtención de estado del sistema"""
    with patch('core.orchestrator.load_config', return_value=mock_config):
        with patch('core.orchestrator.make_storage_from_env', return_value=mock_storage):
            with patch('core.orchestrator.Fetcher', return_value=mock_fetcher):
                orchestrator = TradingOrchestrator("test_config.json")
                
                status = orchestrator.get_system_status()
                
                assert status["running"] == False
                assert status["storage_connected"] == True
                assert status["fetcher_configured"] == True
                assert "crypto" in status["assets_loaded"]
                assert "stocks" in status["assets_loaded"]

def test_full_pipeline(mock_config, mock_storage, mock_fetcher):
    """Test del pipeline completo"""
    with patch('core.orchestrator.load_config', return_value=mock_config):
        with patch('core.orchestrator.make_storage_from_env', return_value=mock_storage):
            with patch('core.orchestrator.Fetcher', return_value=mock_fetcher):
                with patch('core.orchestrator.TradingOrchestrator.run_processing_cycle') as mock_processing:
                    with patch('core.orchestrator.TradingOrchestrator.train_ai_model') as mock_ai:
                        # Configurar mocks
                        mock_processing.return_value = (4, 4)  # Todos exitosos
                        mock_ai.return_value = {"metrics": {"accuracy": 0.85}}
                        
                        # Ejecutar pipeline
                        results = run_full_pipeline("test_config.json")
                        
                        assert "fetch" in results
                        assert "processing" in results
                        assert "ai_training" in results
                        assert results["processing"]["success_rate"] == 1.0

# Tests de manejo de errores
def test_error_handling(mock_config, mock_storage, mock_fetcher):
    """Test de manejo de errores en el orchestrator"""
    with patch('core.orchestrator.load_config', return_value=mock_config):
        with patch('core.orchestrator.make_storage_from_env', return_value=mock_storage):
            with patch('core.orchestrator.Fetcher', return_value=mock_fetcher):
                orchestrator = TradingOrchestrator("test_config.json")
                
                # Forzar error en fetch
                mock_fetcher.fetch_ohlcv.side_effect = Exception("API Error")
                
                # Debe manejar el error gracefullmente
                results = orchestrator.fetch_data(["BTCUSDT"], "5m")
                assert results["BTCUSDT"].empty
                
                # Restaurar mocks
                mock_fetcher.fetch_ohlcv.side_effect = None

if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v"])