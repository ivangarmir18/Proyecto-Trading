"""
Tests unitarios para el módulo fetch
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import time

from core.fetch import Fetcher, FinnhubKeyManager, RateLimiter

@pytest.fixture
def fetcher():
    return Fetcher(
        binance_api_key="test_key",
        binance_secret="test_secret",
        finnhub_keys=["key1", "key2"],
        rate_limit_per_min=1200
    )

def test_rate_limiter():
    limiter = RateLimiter(rate=10, per_seconds=60)
    assert limiter.consume(5) is True
    assert limiter.consume(6) is False  # Excede el límite

def test_finnhub_key_manager():
    keys = ["key1", "key2", "key3"]
    manager = FinnhubKeyManager(keys)
    
    # Test de rotación de keys
    used_keys = set()
    for _ in range(5):
        key = manager.get_key()
        used_keys.add(key)
        time.sleep(0.1)  # Pequeña pausa
    
    assert len(used_keys) > 1  # Debería usar múltiples keys

@patch('core.fetch.ccxt.binance')
def test_fetcher_initialization(mock_binance):
    mock_exchange = Mock()
    mock_binance.return_value = mock_exchange
    
    fetcher = Fetcher(
        exchange_name="binance",
        binance_api_key="test_key",
        binance_secret="test_secret"
    )
    
    assert fetcher._exchange is not None
    mock_binance.assert_called_once()

@patch('core.fetch.requests.get')
def test_finnhub_fetch(mock_get, fetcher):
    # Mock de respuesta de Finnhub
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "s": "ok",
        "t": [1609459200, 1609459260],
        "o": [100.0, 101.0],
        "h": [105.0, 106.0],
        "l": [95.0, 96.0],
        "c": [102.0, 103.0],
        "v": [1000, 1100]
    }
    mock_get.return_value = mock_response
    
    # Mock del key manager
    fetcher.finnhub_key_manager = Mock()
    fetcher.finnhub_key_manager.get_key.return_value = "test_key"
    
    df = fetcher._fetch_finnhub_ohlcv("AAPL", "5", 60)
    
    assert not df.empty
    assert len(df) == 2
    assert "open" in df.columns
    assert "high" in df.columns

@patch('core.fetch.requests.get')
@patch('core.fetch.yf.download')  # Mock también yfinance
def test_yfinance_fallback(mock_yf_download, mock_get, fetcher):
    """Test de fallback a yfinance cuando Finnhub falla"""
    # Mock de error de Finnhub
    mock_response = Mock()
    mock_response.status_code = 401  # Unauthorized
    mock_get.return_value = mock_response
    
    # Mock de yfinance exitoso
    mock_data = pd.DataFrame({
        "Open": [100, 101], "High": [105, 106], "Low": [95, 96], 
        "Close": [102, 103], "Volume": [1000, 1100]
    })
    mock_yf_download.return_value = mock_data
    
    # Forzar que use Finnhub primero pero falle
    with patch.object(fetcher, '_determine_data_source', return_value=fetcher.DataSource.FINNHUB):
        df = fetcher.fetch_ohlcv("AAPL", interval="5m")
    
    assert not df.empty
    assert len(df) == 2
    mock_yf_download.assert_called_once()