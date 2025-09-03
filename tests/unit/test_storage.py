"""
Tests unitarios para el módulo storage
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import sqlite3

from core.storage_postgres import PostgresStorage

@pytest.fixture
def mock_storage():
    with patch('core.storage_postgres.psycopg2') as mock_psycopg2:
        mock_conn = Mock()
        mock_cur = Mock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        
        storage = PostgresStorage(
            host="localhost",
            port=5432,
            dbname="testdb",
            user="testuser",
            password="testpass"
        )
        storage._pool = Mock()
        storage._pool.getconn.return_value = mock_conn
        yield storage, mock_conn, mock_cur

def test_storage_initialization(mock_storage):
    storage, mock_conn, mock_cur = mock_storage
    assert storage is not None

def test_save_candles(mock_storage):
    storage, mock_conn, mock_cur = mock_storage
    
    # Datos de prueba
    df = pd.DataFrame({
        'ts': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'open': [100, 101],
        'high': [105, 106],
        'low': [95, 96],
        'close': [102, 103],
        'volume': [1000, 1001]
    })
    
    storage.save_candles(df, "BTCUSDT", "5m")
    
    # Verificar que se llamó a executemany
    assert mock_cur.executemany.called

def test_get_ohlcv(mock_storage):
    storage, mock_conn, mock_cur = mock_storage
    
    # Mock de respuesta de base de datos
    mock_cur.fetchall.return_value = [
        (1640995200000, 100, 105, 95, 102, 1000),
        (1641081600000, 101, 106, 96, 103, 1001)
    ]
    
    df = storage.get_ohlcv("BTCUSDT", "5m")
    
    assert len(df) == 2
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns