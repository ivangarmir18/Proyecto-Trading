# core/storage_postgres.py
"""
Módulo de storage para PostgreSQL para el proyecto Watchlist.
Reemplaza a storage.py cuando se usa en entornos de producción.
"""

from __future__ import annotations
import os
import psycopg2
from contextlib import contextmanager
from typing import Optional, Iterator
import pandas as pd
import time
from datetime import datetime
import json

# Obtener URL de la base de datos desde variable de entorno
DATABASE_URL = os.getenv('DATABASE_URL')

DDL = """
-- Tabla de velas
CREATE TABLE IF NOT EXISTS candles (
    id SERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    interval TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset, interval, ts)
);

-- Índices para velas
CREATE INDEX IF NOT EXISTS idx_candles_asset_interval_ts ON candles (asset, interval, ts);
CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles (ts);

-- Tabla de indicadores
CREATE TABLE IF NOT EXISTS indicators (
    id SERIAL PRIMARY KEY,
    candle_id INTEGER NOT NULL REFERENCES candles(id) ON DELETE CASCADE,
    ema9 DOUBLE PRECISION,
    ema40 DOUBLE PRECISION,
    atr DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    support DOUBLE PRECISION,
    resistance DOUBLE PRECISION,
    fibonacci_levels JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para indicadores
CREATE INDEX IF NOT EXISTS idx_indicators_candle_id ON indicators (candle_id);

-- Tabla de scores
CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    interval TEXT NOT NULL,
    ts INTEGER NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    range_min DOUBLE PRECISION,
    range_max DOUBLE PRECISION,
    stop DOUBLE PRECISION,
    target DOUBLE PRECISION,
    p_ml DOUBLE PRECISION,
    multiplier DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset, interval, ts)
);

-- Índices para scores
CREATE INDEX IF NOT EXISTS idx_scores_asset_interval_ts ON scores (asset, interval, ts);
CREATE INDEX IF NOT EXISTS idx_scores_ts ON scores (ts);
"""


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    """
    Context manager que yield una conexión a PostgreSQL
    """
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL no está configurada en las variables de entorno")
    
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Crear esquema si no existe"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Ejecutar DDL
            cur.execute(DDL)
        conn.commit()


# -------------------------
# Candles
# -------------------------
def save_candles(df: pd.DataFrame, batch: int = 500, asset: Optional[str] = None, 
                 interval: Optional[str] = None) -> int:
    """
    Inserta o ignora velas en la tabla candles.
    Devuelve el número de filas insertadas.
    """
    if df is None or df.empty:
        return 0

    df = df.copy()

    # Permite llamar con asset/interval separados
    if asset is not None and 'asset' not in df.columns:
        df['asset'] = asset
    if interval is not None and 'interval' not in df.columns:
        df['interval'] = interval

    required = {'ts', 'open', 'high', 'low', 'close'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame de candles debe contener columnas: {required}")

    # Normalizar ts a segundos enteros
    if pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = (df['ts'].astype('int64') // 10**9).astype(int)
    else:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        if df['ts'].isna().any():
            raise ValueError("Columna 'ts' debe contener timestamps unix (segundos)")
        if df['ts'].max() > 1e12:
            df['ts'] = (df['ts'] // 1000).astype(int)
        else:
            df['ts'] = df['ts'].astype(int)

    # Asegurar tipos para columnas numéricas
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Asegurar columnas asset/interval
    if 'asset' not in df.columns:
        df['asset'] = ''
    if 'interval' not in df.columns:
        df['interval'] = ''

    # Preparar filas
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get('asset', '')),
            str(r.get('interval', '')),
            int(r['ts']),
            float(r['open']),
            float(r['high']),
            float(r['low']),
            float(r['close']),
            float(r['volume']) if pd.notna(r.get('volume')) else None
        ))

    inserted_count = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO candles (asset, interval, ts, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (asset, interval, ts) DO NOTHING
            RETURNING id
            """
            
            for i in range(0, len(rows), batch):
                batch_rows = rows[i:i+batch]
                cur.executemany(sql, batch_rows)
                inserted_count += sum(1 for _ in cur.fetchall() if cur.rowcount > 0)
    
    return inserted_count


def load_candles(asset: str, interval: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Carga velas desde PostgreSQL
    """
    with get_connection() as conn:
        sql = """
        SELECT ts, open, high, low, close, volume, asset, interval 
        FROM candles 
        WHERE asset = %s AND interval = %s 
        ORDER BY ts
        """
        params = (asset, interval)
        
        if limit:
            sql += f" LIMIT {int(limit)}"
            
        df = pd.read_sql_query(sql, conn, params=params)
        
        if not df.empty:
            df['ts'] = pd.to_numeric(df['ts']).astype(int)
            df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        
        return df


# -------------------------
# Indicators
# -------------------------
def save_indicators(df_ind: pd.DataFrame, batch: int = 200) -> int:
    """
    Guarda indicadores en PostgreSQL.
    Devuelve el número de filas insertadas.
    """
    if df_ind is None or df_ind.empty:
        return 0

    if 'ts' not in df_ind.columns:
        raise ValueError("df_ind debe contener columna 'ts'")

    df_ind = df_ind.copy()
    
    # Normalizar ts
    if pd.api.types.is_datetime64_any_dtype(df_ind['ts']):
        df_ind['ts'] = (df_ind['ts'].astype('int64') // 10**9).astype(int)
    else:
        df_ind['ts'] = pd.to_numeric(df_ind['ts'], errors='coerce').astype('Int64')

    inserted_count = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Obtener IDs de velas
            candle_ids = {}
            unique_keys = set()
            
            for _, r in df_ind.iterrows():
                asset_k = str(r.get('asset', ''))
                int_k = str(r.get('interval', ''))
                ts_k = int(r['ts'])
                unique_keys.add((asset_k, int_k, ts_k))
            
            # Buscar IDs de velas en lote
            if unique_keys:
                placeholders = ','.join(['%s'] * len(unique_keys) * 3)
                query = f"""
                SELECT id, asset, interval, ts 
                FROM candles 
                WHERE (asset, interval, ts) IN ({placeholders})
                """
                
                flat_params = []
                for asset, interval, ts in unique_keys:
                    flat_params.extend([asset, interval, ts])
                
                cur.execute(query, flat_params)
                for row in cur.fetchall():
                    candle_ids[(row[1], row[2], row[3])] = row[0]

            # Preparar filas de indicadores
            rows = []
            for _, r in df_ind.iterrows():
                asset_k = str(r.get('asset', ''))
                int_k = str(r.get('interval', ''))
                ts_k = int(r['ts'])
                candle_id = candle_ids.get((asset_k, int_k, ts_k))
                
                if candle_id is None:
                    continue
                    
                # Convertir niveles Fibonacci a JSON si es un dict
                fib_levels = r.get('fibonacci_levels')
                if isinstance(fib_levels, dict):
                    fib_levels = json.dumps(fib_levels)
                
                rows.append((
                    int(candle_id),
                    float(r.get('ema9')) if pd.notna(r.get('ema9')) else None,
                    float(r.get('ema40')) if pd.notna(r.get('ema40')) else None,
                    float(r.get('atr')) if pd.notna(r.get('atr')) else None,
                    float(r.get('macd')) if pd.notna(r.get('macd')) else None,
                    float(r.get('macd_signal')) if pd.notna(r.get('macd_signal')) else None,
                    float(r.get('rsi')) if pd.notna(r.get('rsi')) else None,
                    float(r.get('support')) if pd.notna(r.get('support')) else None,
                    float(r.get('resistance')) if pd.notna(r.get('resistance')) else None,
                    fib_levels
                ))

            # Insertar indicadores
            if rows:
                sql = """
                INSERT INTO indicators 
                (candle_id, ema9, ema40, atr, macd, macd_signal, rsi, support, resistance, fibonacci_levels)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (candle_id) DO UPDATE SET
                    ema9 = EXCLUDED.ema9,
                    ema40 = EXCLUDED.ema40,
                    atr = EXCLUDED.atr,
                    macd = EXCLUDED.macd,
                    macd_signal = EXCLUDED.macd_signal,
                    rsi = EXCLUDED.rsi,
                    support = EXCLUDED.support,
                    resistance = EXCLUDED.resistance,
                    fibonacci_levels = EXCLUDED.fibonacci_levels
                """
                
                for i in range(0, len(rows), batch):
                    cur.executemany(sql, rows[i:i+batch])
                    inserted_count += cur.rowcount
    
    return inserted_count


def load_indicators(asset: str, interval: str) -> pd.DataFrame:
    """
    Carga indicadores desde PostgreSQL
    """
    with get_connection() as conn:
        query = """
        SELECT c.ts, i.ema9, i.ema40, i.atr, i.macd, i.macd_signal, 
               i.rsi, i.support, i.resistance, i.fibonacci_levels
        FROM candles c 
        JOIN indicators i ON c.id = i.candle_id 
        WHERE c.asset = %s AND c.interval = %s 
        ORDER BY c.ts
        """
        
        df = pd.read_sql_query(query, conn, params=(asset, interval))
        
        if not df.empty:
            df['ts'] = pd.to_numeric(df['ts']).astype(int)
            
            # Parsear JSON de fibonacci_levels si existe
            if 'fibonacci_levels' in df.columns:
                df['fibonacci_levels'] = df['fibonacci_levels'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else x
                )
        
        return df


# -------------------------
# Scores
# -------------------------
def save_scores(df_scores: pd.DataFrame, batch: int = 300) -> int:
    """
    Guarda scores en PostgreSQL.
    Devuelve el número de filas insertadas/actualizadas.
    """
    if df_scores is None or df_scores.empty:
        return 0

    df = df_scores.copy()
    
    # Normalizar ts
    if 'ts' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = (df['ts'].astype('int64') // 10**9).astype(int)
    elif 'ts' in df.columns:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype(int)

    inserted_count = 0
    now_ts = int(time.time())
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            rows = []
            for _, r in df.iterrows():
                rows.append((
                    str(r.get('asset', '')),
                    str(r.get('interval', '')),
                    int(r['ts']),
                    float(r.get('score', 0.0)),
                    float(r.get('range_min')) if pd.notna(r.get('range_min')) else None,
                    float(r.get('range_max')) if pd.notna(r.get('range_max')) else None,
                    float(r.get('stop')) if pd.notna(r.get('stop')) else None,
                    float(r.get('target')) if pd.notna(r.get('target')) else None,
                    float(r.get('p_ml')) if pd.notna(r.get('p_ml')) else None,
                    float(r.get('multiplier')) if pd.notna(r.get('multiplier')) else None,
                    int(r.get('created_at', now_ts))
                ))

            sql = """
            INSERT INTO scores 
            (asset, interval, ts, score, range_min, range_max, stop, target, p_ml, multiplier, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (asset, interval, ts) DO UPDATE SET
                score = EXCLUDED.score,
                range_min = EXCLUDED.range_min,
                range_max = EXCLUDED.range_max,
                stop = EXCLUDED.stop,
                target = EXCLUDED.target,
                p_ml = EXCLUDED.p_ml,
                multiplier = EXCLUDED.multiplier,
                created_at = EXCLUDED.created_at
            """
            
            for i in range(0, len(rows), batch):
                cur.executemany(sql, rows[i:i+batch])
                inserted_count += cur.rowcount
    
    return inserted_count


def load_scores(asset: str, interval: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Carga scores desde PostgreSQL
    """
    with get_connection() as conn:
        sql = """
        SELECT ts, score, range_min, range_max, stop, target, p_ml, multiplier, created_at
        FROM scores 
        WHERE asset = %s AND interval = %s 
        ORDER BY ts
        """
        params = (asset, interval)
        
        if limit:
            sql += f" LIMIT {int(limit)}"
            
        df = pd.read_sql_query(sql, conn, params=params)
        
        if not df.empty:
            df['ts'] = pd.to_numeric(df['ts']).astype(int)
        
        return df


# -------------------------
# Funciones de Mantenimiento
# -------------------------
def prune_old_data(max_days: int = 30) -> dict:
    """
    Elimina datos antiguos manteniendo solo los de los últimos `max_days` días.
    Devuelve estadísticas de eliminación.
    """
    cutoff_ts = int(time.time()) - (max_days * 24 * 3600)
    stats = {'candles_deleted': 0, 'scores_deleted': 0}
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Eliminar scores antiguos
            cur.execute("DELETE FROM scores WHERE ts < %s", (cutoff_ts,))
            stats['scores_deleted'] = cur.rowcount
            
            # Eliminar candles antiguos (esto activará la eliminación en cascada de indicators)
            cur.execute("DELETE FROM candles WHERE ts < %s", (cutoff_ts,))
            stats['candles_deleted'] = cur.rowcount
    
    return stats


def get_database_stats() -> dict:
    """
    Devuelve estadísticas de la base de datos
    """
    stats = {}
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Contar registros por tabla
            for table in ['candles', 'indicators', 'scores']:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cur.fetchone()[0]
            
            # Obtener fechas extremas
            cur.execute("SELECT MIN(ts), MAX(ts) FROM candles")
            min_ts, max_ts = cur.fetchone()
            stats['oldest_ts'] = min_ts
            stats['newest_ts'] = max_ts
            stats['oldest_date'] = datetime.fromtimestamp(min_ts).isoformat() if min_ts else None
            stats['newest_date'] = datetime.fromtimestamp(max_ts).isoformat() if max_ts else None
    
    return stats


# -------------------------
# Función de migración desde SQLite
# -------------------------
def migrate_from_sqlite(sqlite_path: str, batch_size: int = 1000):
    """
    Migra datos desde SQLite a PostgreSQL
    """
    import sqlite3
    from tqdm import tqdm
    
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"Archivo SQLite no encontrado: {sqlite_path}")
    
    # Conectar a SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    
    # Migrar candles
    print("Migrando candles...")
    candles_query = "SELECT asset, interval, ts, open, high, low, close, volume FROM candles ORDER BY ts"
    
    for chunk in tqdm(pd.read_sql_query(candles_query, sqlite_conn, chunksize=batch_size)):
        save_candles(chunk)
    
    # Migrar scores
    print("Migrando scores...")
    scores_query = """
    SELECT asset, interval, ts, score, range_min, range_max, stop, target, p_ml, multiplier, created_at 
    FROM scores ORDER BY ts
    """
    
    for chunk in tqdm(pd.read_sql_query(scores_query, sqlite_conn, chunksize=batch_size)):
        save_scores(chunk)
    
    sqlite_conn.close()
    print("Migración completada")


# Para mantener compatibilidad con el código existente
def init_db(db_path: Optional[str] = None):
    """Alias para init_db (ignora db_path en PostgreSQL)"""
    init_db()


if __name__ == "__main__":
    # Ejemplo de uso
    init_db()
    stats = get_database_stats()
    print("Estadísticas de la base de datos:", stats)