"""
Módulo de storage para PostgreSQL para el proyecto Watchlist.

Características:
- Lee DATABASE_URL desde variables de entorno.
- Inserciones masivas eficientes con psycopg2.extras.execute_values.
- Funciones mantienen compatibilidad con firma (aceptan db_path pero lo ignoran).
- Manejo de JSONB para fibonacci_levels.
"""
from __future__ import annotations
import os
import json
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Iterator, List, Tuple, Dict

import pandas as pd
import psycopg2
import psycopg2.extras

DATABASE_URL = os.getenv("DATABASE_URL")

# DDL (crea tablas si no existen)
DDL = """
-- Tabla de velas
CREATE TABLE IF NOT EXISTS candles (
    id SERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    interval TEXT NOT NULL,
    ts BIGINT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset, interval, ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_asset_interval_ts ON candles (asset, interval, ts);
CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles (ts);

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candle_id)
);

CREATE INDEX IF NOT EXISTS idx_indicators_candle_id ON indicators (candle_id);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
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
    created_at BIGINT NOT NULL,
    UNIQUE(asset, interval, ts)
);

CREATE INDEX IF NOT EXISTS idx_scores_asset_interval_ts ON scores (asset, interval, ts);
CREATE INDEX IF NOT EXISTS idx_scores_ts ON scores (ts);
"""


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    """Context manager para obtener conexión; levanta error si DATABASE_URL no configurada."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL no está configurada en las variables de entorno")
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
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
    """Crear esquema si no existe (ejecuta DDL)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)


# -------------------------
# Candles
# -------------------------
def save_candles(df: pd.DataFrame, batch: int = 500, asset: Optional[str] = None,
                 interval: Optional[str] = None, db_path: Optional[str] = None) -> int:
    """
    Inserta velas en la tabla candles con ON CONFLICT DO NOTHING.
    Acepta db_path por compatibilidad (se ignora).
    Devuelve número aproximado de filas insertadas.
    """
    if df is None or df.empty:
        return 0
    df = df.copy()

    # soportar asset/interval pasados por parámetro
    if asset is not None and 'asset' not in df.columns:
        df['asset'] = asset
    if interval is not None and 'interval' not in df.columns:
        df['interval'] = interval

    # Normalizar ts
    if pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = (df['ts'].astype('int64') // 10**9).astype('int64')
    else:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        if df['ts'].isna().any():
            raise ValueError("Columna 'ts' debe contener timestamps unix (segundos)")
        if df['ts'].max() > 1e12:
            df['ts'] = (df['ts'] // 1000).astype('int64')
        else:
            df['ts'] = df['ts'].astype('int64')

    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'asset' not in df.columns or 'interval' not in df.columns:
        raise ValueError("DataFrame debe contener 'asset' e 'interval' o pasarlos como argumentos")

    rows = []
    cols = ['asset', 'interval', 'ts', 'open', 'high', 'low', 'close', 'volume']
    for _, r in df[cols].iterrows():
        rows.append((
            r['asset'],
            r['interval'],
            int(r['ts']),
            None if pd.isna(r['open']) else float(r['open']),
            None if pd.isna(r['high']) else float(r['high']),
            None if pd.isna(r['low']) else float(r['low']),
            None if pd.isna(r['close']) else float(r['close']),
            None if pd.isna(r.get('volume')) else float(r.get('volume')),
        ))

    inserted = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            template = "(" + ",".join(["%s"] * len(cols)) + ")"
            # execute_values es más rápido para muchos rows
            for i in range(0, len(rows), batch):
                batch_rows = rows[i:i+batch]
                psycopg2.extras.execute_values(
                    cur,
                    f"""
                    INSERT INTO candles (asset, interval, ts, open, high, low, close, volume)
                    VALUES %s
                    ON CONFLICT (asset, interval, ts) DO NOTHING
                    """,
                    batch_rows,
                    template=template
                )
                # rowcount puede ser -1, por eso contamos manualmente aproximado:
                inserted += len(batch_rows)
    return inserted


def load_candles(asset: str, interval: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Carga velas desde PostgreSQL, ordenadas por ts ascendente."""
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
            df['ts'] = pd.to_numeric(df['ts']).astype('int64')
            df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        return df


# -------------------------
# Indicators
# -------------------------
def save_indicators(df_ind: pd.DataFrame, batch: int = 200, db_path: Optional[str] = None) -> int:
    """
    Guarda indicadores en PostgreSQL.
    - Resuelve candle_id a partir de (asset, interval, ts) con una subquery VALUES para eficiencia.
    - Inserta indicadores y hace upsert por candle_id.
    """
    if df_ind is None or df_ind.empty:
        return 0
    if 'ts' not in df_ind.columns:
        raise ValueError("df_ind debe contener columna 'ts'")

    df = df_ind.copy()
    # Normalizar ts
    if pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = (df['ts'].astype('int64') // 10**9).astype('int64')
    else:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype('int64')

    # Build unique keys set
    keys = []
    for _, r in df.iterrows():
        asset_k = str(r.get('asset', ''))
        int_k = str(r.get('interval', ''))
        ts_k = int(r['ts'])
        keys.append((asset_k, int_k, ts_k))

    inserted = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            if keys:
                # Create subquery from VALUES and join to candles to obtain candle_id
                values_sql = ",".join(["(%s,%s,%s)"] * len(keys))
                flat_params = []
                for k in keys:
                    flat_params.extend([k[0], k[1], k[2]])
                query = f"""
                SELECT v.column1 AS asset, v.column2 AS interval, v.column3 AS ts, c.id
                FROM (VALUES {values_sql}) AS v(column1, column2, column3)
                LEFT JOIN candles c ON (c.asset = v.column1 AND c.interval = v.column2 AND c.ts = v.column3)
                """
                cur.execute(query, flat_params)
                rows = cur.fetchall()
                # map to dict: (asset, interval, ts) -> candle_id
                candle_map = {}
                for row in rows:
                    asset_r, interval_r, ts_r, cid = row[0], row[1], row[2], row[3]
                    candle_map[(asset_r, interval_r, ts_r)] = cid

                # Prepare indicator rows
                to_insert = []
                for _, r in df.iterrows():
                    asset_k = str(r.get('asset', ''))
                    int_k = str(r.get('interval', ''))
                    ts_k = int(r['ts'])
                    cid = candle_map.get((asset_k, int_k, ts_k))
                    if cid is None:
                        # no candle present -> skip
                        continue
                    fib = r.get('fibonacci_levels')
                    if isinstance(fib, dict):
                        fib = json.dumps(fib)
                    to_insert.append((
                        int(cid),
                        None if pd.isna(r.get('ema9')) else float(r.get('ema9')),
                        None if pd.isna(r.get('ema40')) else float(r.get('ema40')),
                        None if pd.isna(r.get('atr')) else float(r.get('atr')),
                        None if pd.isna(r.get('macd')) else float(r.get('macd')),
                        None if pd.isna(r.get('macd_signal')) else float(r.get('macd_signal')),
                        None if pd.isna(r.get('rsi')) else float(r.get('rsi')),
                        None if pd.isna(r.get('support')) else float(r.get('support')),
                        None if pd.isna(r.get('resistance')) else float(r.get('resistance')),
                        fib
                    ))

                if to_insert:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO indicators
                        (candle_id, ema9, ema40, atr, macd, macd_signal, rsi, support, resistance, fibonacci_levels)
                        VALUES %s
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
                        """,
                        to_insert,
                        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    )
                    inserted += len(to_insert)
    return inserted


def load_indicators(asset: str, interval: str) -> pd.DataFrame:
    """Carga indicadores por asset/interval ordenados por ts."""
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
            df['ts'] = pd.to_numeric(df['ts']).astype('int64')
            if 'fibonacci_levels' in df.columns:
                df['fibonacci_levels'] = df['fibonacci_levels'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else x
                )
        return df


# -------------------------
# Scores
# -------------------------
def save_scores(df_scores: pd.DataFrame, batch: int = 500, db_path: Optional[str] = None) -> int:
    """
    Inserta/actualiza scores masivamente. Acepta db_path por compatibilidad.
    Devuelve número de filas procesadas (aprox).
    """
    if df_scores is None or df_scores.empty:
        return 0
    df = df_scores.copy()
    if 'ts' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = (df['ts'].astype('int64') // 10**9).astype('int64')
    elif 'ts' in df.columns:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype('int64')
    else:
        raise ValueError("df_scores debe contener columna 'ts'")

    now_ts = int(time.time())
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get('asset', '')),
            str(r.get('interval', '')),
            int(r['ts']),
            None if pd.isna(r.get('score')) else float(r.get('score')),
            None if pd.isna(r.get('range_min')) else float(r.get('range_min')),
            None if pd.isna(r.get('range_max')) else float(r.get('range_max')),
            None if pd.isna(r.get('stop')) else float(r.get('stop')),
            None if pd.isna(r.get('target')) else float(r.get('target')),
            None if pd.isna(r.get('p_ml')) else float(r.get('p_ml')),
            None if pd.isna(r.get('multiplier')) else float(r.get('multiplier')),
            int(r.get('created_at', now_ts))
        ))

    inserted = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO scores
                (asset, interval, ts, score, range_min, range_max, stop, target, p_ml, multiplier, created_at)
                VALUES %s
                ON CONFLICT (asset, interval, ts) DO UPDATE SET
                    score = EXCLUDED.score,
                    range_min = EXCLUDED.range_min,
                    range_max = EXCLUDED.range_max,
                    stop = EXCLUDED.stop,
                    target = EXCLUDED.target,
                    p_ml = EXCLUDED.p_ml,
                    multiplier = EXCLUDED.multiplier,
                    created_at = EXCLUDED.created_at
                """,
                rows,
                template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            )
            inserted += len(rows)
    return inserted


def load_scores(asset: str, interval: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Carga scores (orden asc por ts)."""
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
            df['ts'] = pd.to_numeric(df['ts']).astype('int64')
        return df


# -------------------------
# Maintenance
# -------------------------
def prune_old_data(max_days: int = 30) -> dict:
    cutoff_ts = int(time.time()) - (max_days * 24 * 3600)
    stats = {'candles_deleted': 0, 'scores_deleted': 0}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM scores WHERE ts < %s", (cutoff_ts,))
            stats['scores_deleted'] = cur.rowcount
            cur.execute("DELETE FROM candles WHERE ts < %s", (cutoff_ts,))
            stats['candles_deleted'] = cur.rowcount
    return stats


def get_database_stats() -> dict:
    stats = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            for table in ['candles', 'indicators', 'scores']:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cur.fetchone()[0]
            cur.execute("SELECT MIN(ts), MAX(ts) FROM candles")
            min_ts, max_ts = cur.fetchone()
            stats['oldest_ts'] = min_ts
            stats['newest_ts'] = max_ts
            stats['oldest_date'] = datetime.fromtimestamp(min_ts).isoformat() if min_ts else None
            stats['newest_date'] = datetime.fromtimestamp(max_ts).isoformat() if max_ts else None
    return stats


# -------------------------
# Migration helper
# -------------------------
def migrate_from_sqlite(sqlite_path: str, batch_size: int = 1000):
    """Migra datos desde SQLite a PostgreSQL (aproximado)."""
    import sqlite3
    from tqdm import tqdm
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"Archivo SQLite no encontrado: {sqlite_path}")
    sqlite_conn = sqlite3.connect(sqlite_path)
    candles_query = "SELECT asset, interval, ts, open, high, low, close, volume FROM candles ORDER BY ts"
    for chunk in tqdm(pd.read_sql_query(candles_query, sqlite_conn, chunksize=batch_size)):
        save_candles(chunk)
    scores_query = """
    SELECT asset, interval, ts, score, range_min, range_max, stop, target, p_ml, multiplier, created_at
    FROM scores ORDER BY ts
    """
    for chunk in tqdm(pd.read_sql_query(scores_query, sqlite_conn, chunksize=batch_size)):
        save_scores(chunk)
    sqlite_conn.close()


# alias init_db compatible
def init_db_alias(db_path: Optional[str] = None):
    init_db()


# Backwards compatibility names (por si otros módulos importan storage.save_candles directamente)
save_candles_with_dbpath = save_candles
save_indicators_with_dbpath = save_indicators
save_scores_with_dbpath = save_scores
load_candles_with_dbpath = load_candles
load_scores_with_dbpath = load_scores
load_indicators_with_dbpath = load_indicators


if __name__ == "__main__":
    init_db()
    print(get_database_stats())
