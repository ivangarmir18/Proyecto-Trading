# core/storage_postgres.py
"""
Módulo de almacenamiento en PostgreSQL para el proyecto Watchlist.
Centraliza toda la interacción con la base de datos: inicialización,
inserción y consulta de datos (assets, prices, scores, indicadores, etc.).
"""

import os
import json
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import pandas as pd
from typing import Optional, Dict, Any


class PostgresStorage:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/watchlist")

    def get_conn(self):
        return psycopg2.connect(self.dsn)

    # -------------------------------
    # Inicialización de tablas
    # -------------------------------
    def init_db(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS assets (
            id SERIAL PRIMARY KEY,
            symbol TEXT UNIQUE NOT NULL,
            type TEXT CHECK (type IN ('stock','crypto')) NOT NULL,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS prices (
            asset TEXT NOT NULL REFERENCES assets(symbol) ON DELETE CASCADE,
            interval TEXT NOT NULL,
            ts BIGINT NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            PRIMARY KEY (asset, interval, ts)
        );

        CREATE TABLE IF NOT EXISTS indicators (
            asset TEXT NOT NULL REFERENCES assets(symbol) ON DELETE CASCADE,
            interval TEXT NOT NULL,
            ts BIGINT NOT NULL,
            name TEXT NOT NULL,
            value DOUBLE PRECISION,
            params JSONB,
            PRIMARY KEY (asset, interval, ts, name)
        );

        CREATE TABLE IF NOT EXISTS scores (
            asset TEXT NOT NULL REFERENCES assets(symbol) ON DELETE CASCADE,
            ts BIGINT NOT NULL,
            score DOUBLE PRECISION,
            components JSONB,
            method TEXT,
            PRIMARY KEY (asset, ts)
        );

        CREATE TABLE IF NOT EXISTS models (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT now(),
            config JSONB,
            metrics JSONB
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT CHECK (status IN ('pending','running','success','failed')) NOT NULL,
            started_at TIMESTAMP DEFAULT now(),
            finished_at TIMESTAMP,
            details JSONB
        );
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    # -------------------------------
    # Gestión de assets
    # -------------------------------
    def upsert_asset(self, symbol: str, type_: str, name: Optional[str] = None):
        sql = """
        INSERT INTO assets (symbol, type, name)
        VALUES (%s, %s, %s)
        ON CONFLICT (symbol) DO UPDATE SET type = EXCLUDED.type, name = EXCLUDED.name
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol, type_, name))
            conn.commit()

    # -------------------------------
    # Precios
    # -------------------------------
    def upsert_prices(self, asset: str, interval: str, df: pd.DataFrame) -> int:
        """
        Inserta precios OHLCV desde un DataFrame.
        Requiere columnas: ts, open, high, low, close, volume
        """
        records = [
            (asset, interval, int(row["ts"]), float(row["open"]), float(row["high"]),
             float(row["low"]), float(row["close"]), float(row["volume"]))
            for _, row in df.iterrows()
        ]
        sql = """
        INSERT INTO prices (asset, interval, ts, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (asset, interval, ts) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, records, page_size=500)
            conn.commit()
        return len(records)

    def load_prices(self, asset: str, interval: str) -> pd.DataFrame:
        sql = "SELECT ts, open, high, low, close, volume FROM prices WHERE asset=%s AND interval=%s ORDER BY ts"
        with self.get_conn() as conn:
            return pd.read_sql(sql, conn, params=(asset, interval))

    # -------------------------------
    # Indicadores
    # -------------------------------
    def upsert_indicators(self, asset: str, interval: str, df: pd.DataFrame, cfg: Dict[str, Any]) -> int:
        """
        Inserta indicadores calculados en la tabla `indicators`.
        Requiere columna ts y columnas de indicadores en df.
        """
        records = []
        for _, row in df.iterrows():
            ts = int(row["ts"])
            for col in df.columns:
                if col in ("ts","open","high","low","close","volume"):
                    continue
                val = row[col]
                params = json.dumps(cfg.get(col, {}))
                records.append((asset, interval, ts, col, float(val) if pd.notna(val) else None, params))

        sql = """
        INSERT INTO indicators (asset, interval, ts, name, value, params)
        VALUES %s
        ON CONFLICT (asset, interval, ts, name) DO UPDATE
            SET value = EXCLUDED.value,
                params = EXCLUDED.params
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, records, page_size=500)
            conn.commit()
        return len(records)

    # -------------------------------
    # Scores
    # -------------------------------
    def upsert_score(self, asset: str, ts: int, score: float, components: dict, method: str):
        sql = """
        INSERT INTO scores (asset, ts, score, components, method)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (asset, ts) DO UPDATE
            SET score = EXCLUDED.score,
                components = EXCLUDED.components,
                method = EXCLUDED.method
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (asset, ts, score, json.dumps(components), method))
            conn.commit()

    def load_scores(self, asset: str) -> pd.DataFrame:
        sql = "SELECT ts, score, components, method FROM scores WHERE asset=%s ORDER BY ts"
        with self.get_conn() as conn:
            return pd.read_sql(sql, conn, params=(asset,))

    # -------------------------------
    # Jobs
    # -------------------------------
    def create_job(self, name: str, status: str = "pending", details: Optional[dict] = None) -> int:
        sql = "INSERT INTO jobs (name, status, details) VALUES (%s, %s, %s) RETURNING id"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (name, status, json.dumps(details or {})))
                job_id = cur.fetchone()[0]
            conn.commit()
            return job_id

    def update_job(self, job_id: int, status: str, details: Optional[dict] = None):
        sql = """
        UPDATE jobs
        SET status=%s,
            details=COALESCE(%s, details),
            finished_at = CASE WHEN %s IN ('success','failed') THEN now() ELSE finished_at END
        WHERE id=%s
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status, json.dumps(details) if details else None, status, job_id))
            conn.commit()
