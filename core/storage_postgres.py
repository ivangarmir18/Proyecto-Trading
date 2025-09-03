# core/storage_postgres.py
"""
Postgres storage backend mejorado con:
- Soporte completo para Render.com
- Políticas de retención configurables por intervalo
- Bulk upserts optimizados
- Pool de conexiones eficiente
- Métricas de rendimiento
- Manejo robusto de errores
"""

from __future__ import annotations
import os
import time
import math
import logging
import threading
import urllib.parse
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
    "5m": 70,   # 70 días para 5m
    "15m": 90,  # 3 meses para 15m
    "30m": 120, # 4 meses para 30m
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
        # Obtener configuración de environment variables
        self.host = os.getenv("POSTGRES_HOST")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.dbname = os.getenv("POSTGRES_DB", "trading")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "")
        
        # Prioridad: parámetros individuales > DATABASE_URL > parámetro dsn
        if all([self.host, self.dbname, self.user]):
            # Construir DSN desde parámetros individuales
            self.dsn = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
            logger.info("Using individual PostgreSQL parameters")
        else:
            # Usar DATABASE_URL de Render
            self.dsn = os.getenv("DATABASE_URL") or dsn
            if self.dsn:
                # Asegurar formato correcto para psycopg2
                if self.dsn.startswith("postgresql://"):
                    self.dsn = self.dsn.replace("postgresql://", "postgres://")
                logger.info("Using DATABASE_URL from environment")
            else:
                raise ValueError("Se requiere configuración de PostgreSQL (DATABASE_URL o parámetros individuales)")
        
        if not self.dsn:
            raise ValueError("No se pudo determinar la configuración de PostgreSQL")
        
        self.minconn = minconn
        self.maxconn = maxconn
        self.retention_policy = retention_policy or DEFAULT_RETENTION_POLICY
        
        # Configuración de operaciones
        self._batch_size = 500
        self._metrics = {
            "writes": 0,
            "reads": 0,
            "deletes": 0,
            "errors": 0
        }
        self._lock = threading.RLock()
        self._pool = None
        
        # Inicializar pool de conexiones
        self._ensure_pool()
        
        logger.info(f"PostgresStorage inicializado para {self.dbname}")

    def _ensure_pool(self):
        """Crea el pool de conexiones si no existe"""
        if self._pool:
            return
            
        try:
            self._pool = ThreadedConnectionPool(
                self.minconn, self.maxconn, self.dsn,
                application_name="trading-storage"
            )
            logger.info(f"Pool de conexiones creado (min={self.minconn}, max={self.maxconn})")
        except Exception as e:
            logger.error(f"Error creando pool de conexiones: {e}")
            # Intentar conexión directa para diagnóstico
            self._test_connection()
            raise

    def _test_connection(self):
        """Prueba de conexión simple para diagnóstico"""
        try:
            import psycopg2
            conn = psycopg2.connect(self.dsn)
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()
            logger.info(f"PostgreSQL version: {version[0]}")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Prueba de conexión fallida: {e}")
            return False

    @contextmanager
    def get_conn(self):
        """Context manager para obtener conexiones del pool"""
        if not self._pool:
            self._ensure_pool()
            
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            with self._lock:
                self._metrics["errors"] += 1
            logger.error(f"Error en conexión: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    def init_db(self):
        """Inicializa tablas con particionamiento por intervalo"""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                # Tabla principal
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                        id BIGSERIAL,
                        asset TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        ts BIGINT NOT NULL,
                        open DOUBLE PRECISION NOT NULL,
                        high DOUBLE PRECISION NOT NULL,
                        low DOUBLE PRECISION NOT NULL,
                        close DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (asset, interval, ts)
                    );
                """)
                
                # Tabla de scores
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scores (
                        id BIGSERIAL PRIMARY KEY,
                        asset TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        ts BIGINT NOT NULL,
                        score DOUBLE PRECISION NOT NULL,
                        range_min DOUBLE PRECISION,
                        range_max DOUBLE PRECISION,
                        stop DOUBLE PRECISION,
                        target DOUBLE PRECISION,
                        p_ml DOUBLE PRECISION,
                        multiplier DOUBLE PRECISION,
                        signal_quality DOUBLE PRECISION,
                        created_at BIGINT NOT NULL,
                        UNIQUE (asset, interval, ts)
                    );
                """)
                
                # Tabla de indicadores
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS indicators (
                        id BIGSERIAL PRIMARY KEY,
                        asset TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        ts BIGINT NOT NULL,
                        ema9 DOUBLE PRECISION,
                        ema40 DOUBLE PRECISION,
                        atr DOUBLE PRECISION,
                        macd_line DOUBLE PRECISION,
                        macd_signal DOUBLE PRECISION,
                        macd_hist DOUBLE PRECISION,
                        rsi DOUBLE PRECISION,
                        support DOUBLE PRECISION,
                        resistance DOUBLE PRECISION,
                        fibonacci_levels JSONB,
                        UNIQUE (asset, interval, ts)
                    );
                """)
                
                # Índices para mejorar rendimiento
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_candles_asset_interval_ts 
                    ON candles (asset, interval, ts DESC);
                    
                    CREATE INDEX IF NOT EXISTS idx_scores_asset_interval_ts 
                    ON scores (asset, interval, ts DESC);
                    
                    CREATE INDEX IF NOT EXISTS idx_indicators_asset_interval_ts 
                    ON indicators (asset, interval, ts DESC);
                """)
                
                logger.info("Tablas de PostgreSQL inicializadas correctamente")

    def save_candles(self, df: pd.DataFrame, asset: str, interval: str, meta: Optional[Dict[str, Any]] = None) -> int:
        """Inserta velas usando execute_values para máximo rendimiento"""
        if df is None or df.empty:
            return 0

        # Preparar datos
        records = []
        for _, row in df.iterrows():
            # Convertir timestamp a ms
            if hasattr(row['ts'], 'timestamp'):
                ts_ms = int(row['ts'].timestamp() * 1000)
            else:
                ts_ms = int(row['ts'])
                
            records.append((
                asset,
                interval,
                ts_ms,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else None
            ))

        # Insertar en lotes
        inserted = 0
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(records), self._batch_size):
                    batch = records[i:i + self._batch_size]
                    try:
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
                            batch,
                            page_size=len(batch)
                        )
                        inserted += cur.rowcount
                    except Exception as e:
                        logger.error(f"Error insertando lote {i}: {e}")
                        with self._lock:
                            self._metrics["errors"] += 1
        
        with self._lock:
            self._metrics["writes"] += inserted
            
        logger.info(f"Insertadas {inserted} velas para {asset} {interval}")
        return inserted

    # --- INICIO: NUEVA FUNCIÓN AÑADIDA ---
    def has_data(self, asset: str, interval: str) -> bool:
        """Chequea si existe alguna vela para un activo/intervalo."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM candles WHERE asset = %s AND interval = %s LIMIT 1;",
                        (asset, interval)
                    )
                    return cur.fetchone() is not None
        except Exception as e:
            logger.exception("Error checking for data: %s", e)
            return False
    # --- FIN: NUEVA FUNCIÓN AÑADIDA ---
    
    def get_ohlcv(self, asset: str, interval: str, start_ms: Optional[int] = None, 
                 end_ms: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Recupera velas desde PostgreSQL"""
        query = """
            SELECT ts, open, high, low, close, volume 
            FROM candles 
            WHERE asset = %s AND interval = %s
        """
        params = [asset, interval]
        
        if start_ms is not None:
            query += " AND ts >= %s"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND ts <= %s"
            params.append(end_ms)
            
        query += " ORDER BY ts ASC"
        
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)
        
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    
                    if not rows:
                        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
                    
                    # Convertir a DataFrame
                    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
                    
                    # Convertir timestamp a datetime
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                    
                    with self._lock:
                        self._metrics["reads"] += len(df)
                        
                    return df
                    
        except Exception as e:
            logger.error(f"Error obteniendo OHLCV: {e}")
            with self._lock:
                self._metrics["errors"] += 1
            return pd.DataFrame()

    def apply_retention_policy(self):
        """Aplica políticas de retención automáticamente"""
        deleted_total = 0
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    for interval, days in self.retention_policy.items():
                        cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                        
                        cur.execute("""
                            DELETE FROM candles 
                            WHERE interval = %s AND ts < %s
                        """, (interval, cutoff_ts))
                        
                        deleted = cur.rowcount
                        deleted_total += deleted
                        
                        with self._lock:
                            self._metrics["deletes"] += deleted
                            
                        logger.info(f"Retención {interval}: eliminadas {deleted} velas")
            
            return deleted_total
            
        except Exception as e:
            logger.error(f"Error aplicando retención: {e}")
            with self._lock:
                self._metrics["errors"] += 1
            return 0

    def save_scores(self, df_scores: pd.DataFrame, asset: str, interval: str) -> int:
        """Guarda scores en la base de datos"""
        if df_scores is None or df_scores.empty:
            return 0
            
        records = []
        for _, row in df_scores.iterrows():
            # Convertir timestamp a ms
            if hasattr(row['ts'], 'timestamp'):
                ts_ms = int(row['ts'].timestamp() * 1000)
            else:
                ts_ms = int(row['ts'])
                
            records.append((
                asset,
                interval,
                ts_ms,
                float(row['score']) if 'score' in row else 0.0,
                float(row['range_min']) if 'range_min' in row and not pd.isna(row['range_min']) else None,
                float(row['range_max']) if 'range_max' in row and not pd.isna(row['range_max']) else None,
                float(row['stop']) if 'stop' in row and not pd.isna(row['stop']) else None,
                float(row['target']) if 'target' in row and not pd.isna(row['target']) else None,
                float(row['p_ml']) if 'p_ml' in row and not pd.isna(row['p_ml']) else None,
                float(row['multiplier']) if 'multiplier' in row and not pd.isna(row['multiplier']) else None,
                float(row['signal_quality']) if 'signal_quality' in row and not pd.isna(row['signal_quality']) else None,
                int(time.time() * 1000)  # created_at
            ))
        
        # Insertar en lotes
        inserted = 0
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(records), self._batch_size):
                    batch = records[i:i + self._batch_size]
                    try:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO scores (asset, interval, ts, score, range_min, range_max, 
                                              stop, target, p_ml, multiplier, signal_quality, created_at)
                            VALUES %s
                            ON CONFLICT (asset, interval, ts) DO UPDATE SET
                                score = EXCLUDED.score,
                                range_min = EXCLUDED.range_min,
                                range_max = EXCLUDED.range_max,
                                stop = EXCLUDED.stop,
                                target = EXCLUDED.target,
                                p_ml = EXCLUDED.p_ml,
                                multiplier = EXCLUDED.multiplier,
                                signal_quality = EXCLUDED.signal_quality,
                                created_at = EXCLUDED.created_at
                            """,
                            batch,
                            page_size=len(batch)
                        )
                        inserted += cur.rowcount
                    except Exception as e:
                        logger.error(f"Error insertando scores lote {i}: {e}")
                        with self._lock:
                            self._metrics["errors"] += 1
        
        with self._lock:
            self._metrics["writes"] += inserted
            
        logger.info(f"Insertados {inserted} scores para {asset} {interval}")
        return inserted

    def get_metrics(self) -> Dict[str, Any]:
        """Devuelve métricas de rendimiento"""
        with self._lock:
            return self._metrics.copy()

    def close(self):
        """Cierra el pool de conexiones"""
        if self._pool:
            try:
                self._pool.closeall()
                logger.info("Pool de conexiones cerrado")
            except Exception as e:
                logger.error(f"Error cerrando pool: {e}")
            finally:
                self._pool = None

    def make_save_callback(self):
        """Crea un callback para guardar datos"""
        def save_callback(df, asset, interval, meta=None):
            return self.save_candles(df, asset, interval, meta)
        return save_callback

# Función de conveniencia para crear storage desde environment variables
def make_storage_from_env() -> PostgresStorage:
    """Crea una instancia de PostgresStorage desde variables de entorno"""
    return PostgresStorage(
        minconn=int(os.getenv("POSTGRES_POOL_MIN", "1")),
        maxconn=int(os.getenv("POSTGRES_POOL_MAX", "10")),
        retention_policy=DEFAULT_RETENTION_POLICY
    )

# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Prueba de conexión
    storage = make_storage_from_env()
    
    try:
        storage.init_db()
        print("Base de datos inicializada correctamente")
        
        # Prueba de inserción
        test_df = pd.DataFrame([{
            "ts": datetime.now(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000.0
        }])
        
        inserted = storage.save_candles(test_df, "TEST", "1m")
        print(f"Insertadas {inserted} filas de prueba")
        
        # Prueba de lectura
        df = storage.get_ohlcv("TEST", "1m")
        print(f"Leídas {len(df)} filas")
        print(df)
        
    finally:
        storage.close()