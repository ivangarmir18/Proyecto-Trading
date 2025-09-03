"""
Tests que simulan la base de datos para CI/CD
"""
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd

@pytest.fixture
def mock_storage():
    """Mock del storage para tests"""
    storage = Mock()
    
    # Mock de datos de ejemplo
    mock_df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=100, freq='H'),
        'open': range(100, 200),
        'high': range(105, 205),
        'low': range(95, 195),
        'close': range(102, 202),
        'volume': range(1000, 1100)
    })
    
    storage.get_ohlcv.return_value = mock_df
    storage.make_save_callback.return_value = Mock()
    
    return storage

def test_fetch_with_mock(mock_storage):
    """Test de fetch con storage mockeado"""
    from core.fetch import Fetcher
    
    fetcher = Fetcher()
    # Probar que se puede crear sin errores
    assert fetcher is not None
    print("✅ Fetcher creado exitosamente")

def test_score_calculation(mock_storage):
    """Test de cálculo de scores con mock"""
    from core.score import AIScoringSystem
    
    scoring_system = AIScoringSystem(mock_storage, {})
    df = mock_storage.get_ohlcv('AAPL', '1h')
    
    # Probar que puede calcular features
    result = scoring_system.calculate_technical_indicators(df)
    assert not result.empty
    print("✅ Cálculo de scores funciona")