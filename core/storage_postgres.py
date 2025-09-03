# core/storage_postgres.py
"""
Postgres storage backend mejorado con:
- Políticas de retención configurables por intervalo
- Bulk upserts optimizados
- Pool de conexiones eficiente
- Métricas de rendimiento
"""

from __future__ import annotations
import os
import time
import math
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger("core.storage_postgres")

# Configuración de retención por defecto (días)
DEFAULT_RETENTION_POLICY = {
    "1m": 7,    # 1 semana para 1m
    "5m": 30,   # 1 mes para 5m
    "15m": 60,  # 2 meses para 15m
    "30m": 90,  # 3 meses para 30m
    "1h": 180,  # 6 meses para 1h
    "4h": 365,  # 1 año para 4h
    "1d": 730,  # 2 años para 1d
    "1w": 1825  # 5 años para 1w
}

class PostgresStorage:
    def __init__(
        self,
        dsn: Optional[str] = None,
        minconn: int = 1,
        maxconn: int = 10,
        retention_policy: Optional[Dict[str, int]] = None
    ):
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise ValueError("Se requiere DATABASE_URL o cadena de conexión")
        
        self.minconn = minconn
        self.maxconn = maxconn
        self.retention_policy = retention_policy or DEFAULT_RETENTION_POLICY
        
        self._pool = ThreadedConnectionPool(
            minconn, maxconn, self.dsn,
            application_name="trading-storage"
        )
        
        self._batch_size = 1000
        self._metrics = {
            "writes": 0,
            "reads": 0,
            "deletes": 0
        }
        self._lock = threading.RLock()

    @contextmanager
    def get_connection(self):
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def init_db(self):
        """Inicializa tablas con particionamiento por intervalo"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Tabla principal particionada
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                        asset TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        ts BIGINT NOT NULL,
                        open DECIMAL(18, 8) NOT NULL,
                        high DECIMAL(18, 8) NOT NULL,
                        low DECIMAL(18, 8) NOT NULL,
                        close DECIMAL(18, 8) NOT NULL,
                        volume DECIMAL(18, 8) NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (asset, interval, ts)
                    ) PARTITION BY LIST (interval);
                """)
                
                # Crear particiones para cada intervalo
                for interval in self.retention_policy.keys():
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS candles_{interval}
                        PARTITION OF candles FOR VALUES IN ('{interval}');
                    """)
                
                # Índices optimizados
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_candles_ts 
                    ON candles (ts DESC);
                    
                    CREATE INDEX IF NOT EXISTS idx_candles_asset 
                    ON candles (asset, ts DESC);
                """)

    def save_candles(self, df: pd.DataFrame, asset: str, interval: str):
        """Inserta velas usando COPY para máximo rendimiento"""
        if df.empty:
            return 0

        # Convertir a formato PostgreSQL
        records = []
        for _, row in df.iterrows():
            records.append((
                asset,
                interval,
                int(row['ts'].timestamp() * 1000),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Usar COPY para inserción masiva
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO candles (asset, interval, ts, open, high, low, close, volume)
                    VALUES %s
                    ON CONFLICT (asset, interval, ts) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                    """,
                    records,
                    page_size=self._batch_size
                )
                
                with self._lock:
                    self._metrics["writes"] += cur.rowcount
                
                return cur.rowcount

    def apply_retention_policy(self):
        """Aplica políticas de retención automáticamente"""
        deleted_total = 0
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for interval, days in self.retention_policy.items():
                    cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                    
                    cur.execute("""
                        DELETE FROM candles 
                        WHERE interval = %s AND ts < %s
                    """, (interval, cutoff_ts))
                    
                    with self._lock:
                        self._metrics["deletes"] += cur.rowcount
                    
                    deleted_total += cur.rowcount
                    logger.info(f"Retención {interval}: eliminadas {cur.rowcount} velas")
        
        return deleted_total

    def get_metrics(self) -> Dict[str, Any]:
        """Devuelve métricas de rendimiento"""
        with self._lock:
            return self._metrics.copy()

    def close(self):
        """Cierra el pool de conexiones"""
        self._pool.closeall()

# Uso recomendado en producción:
storage = PostgresStorage(
    dsn=os.getenv("DATABASE_URL"),
    retention_policy={
        "5m": 70,   # 70 días para velas de 5m
        "1h": 70,   # 70 días para velas de 1h
        "4h": 365,  # 1 año para velas de 4h
        "1d": 365   # 1 año para velas de 1d
    }
)