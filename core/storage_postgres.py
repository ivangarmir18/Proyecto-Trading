"""
Reestructuración completa de core/storage_postgres.py
- Implementa PostgresStorage usando psycopg2 SimpleConnectionPool
- Provee todas las funciones usadas por la UI, worker y tests:
    * init_db()
    * save_candles / load_candles
    * save_scores / load_scores / load_latest_scores
    * add_watchlist_symbol / list_watchlist / get_watchlist / remove_watchlist_symbol
    * add_backfill_request / update_backfill_status / get_backfill_status
    * save_model_record / get_latest_model_record
    * list_assets / list_intervals_for_asset / list_models_for_asset_interval
    * health(), close()
- Convenciones:
    * timestamps: ms since epoch stored in BIGINT (ts)
    * pandas DataFrames used for candle/score bulk IO
    * JSON fields use json.dumps / psycopg2.extras.Json

Nota: este fichero está pensado para ser "drop-in" con el resto del repo. Si el entorno
usa DATABASE_URL (preferred) se respetará, si no, se leerán host/port/dbname/user/password.

"""
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

DEFAULT_POOL_MIN = 1
DEFAULT_POOL_MAX = 5


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class PostgresStorage:
    """PostgresStorage: almacenamiento central usando Postgres.

    Parámetros (pueden pasarse por kwargs o por env DATABASE_URL):
        host, port, dbname, user, password, pool_min, pool_max
    """

    def __init__(self, **kwargs):
        # Leer configuración desde env si no viene en kwargs
        self._cfg = {
            "host": kwargs.get("host") or os.getenv("PGHOST") or os.getenv("POSTGRES_HOST"),
            "port": int(kwargs.get("port") or os.getenv("PGPORT") or 5432),
            "dbname": kwargs.get("dbname") or os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB"),
            "user": kwargs.get("user") or os.getenv("PGUSER") or os.getenv("POSTGRES_USER"),
            "password": kwargs.get("password") or os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD"),
            "database_url": kwargs.get("database_url") or os.getenv("DATABASE_URL"),
            "pool_min": int(kwargs.get("pool_min") or os.getenv("PG_POOL_MIN") or DEFAULT_POOL_MIN),
            "pool_max": int(kwargs.get("pool_max") or os.getenv("PG_POOL_MAX") or DEFAULT_POOL_MAX),
        }
        # Connection pool lazy init
        self._pool: Optional[SimpleConnectionPool] = None
        # expose convenience property for tests
        self._dsn = self._build_dsn()

    def _build_dsn(self) -> Optional[str]:
        if self._cfg.get("database_url"):
            return self._cfg.get("database_url")
        host = self._cfg.get("host")
        dbname = self._cfg.get("dbname")
        user = self._cfg.get("user")
        password = self._cfg.get("password")
        port = self._cfg.get("port")
        if host and dbname and user:
            return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        return None

    def _ensure_pool(self):
        if self._pool is None:
            dsn = self._dsn
            if dsn:
                # allow psycopg2 to parse full DATABASE_URL
                logger.debug("Creating connection pool to %s", dsn)
                self._pool = SimpleConnectionPool(self._cfg["pool_min"], self._cfg["pool_max"], dsn)
            else:
                # fallback: connect using individual params
                conn_params = dict(host=self._cfg.get("host"), port=self._cfg.get("port"), dbname=self._cfg.get("dbname"), user=self._cfg.get("user"), password=self._cfg.get("password"))
                logger.debug("Creating connection pool with params %s", {k: v for k, v in conn_params.items() if k != "password"})
                self._pool = SimpleConnectionPool(self._cfg["pool_min"], self._cfg["pool_max"], **conn_params)

    @contextmanager
    def _get_conn(self):
        self._ensure_pool()
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn:
                self._pool.putconn(conn)

    def close(self):
        if self._pool:
            try:
                self._pool.closeall()
            except Exception:
                logger.exception("Error closing pool")
            self._pool = None

    # -----------------
    # Schema / Init
    # -----------------
    def init_db(self):
        """Crea las tablas necesarias si no existen."""
        ddl = """
        CREATE TABLE IF NOT EXISTS candles (
            ts BIGINT NOT NULL,
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            PRIMARY KEY (asset, interval, ts)
        );

        CREATE INDEX IF NOT EXISTS idx_candles_asset_interval_ts ON candles (asset, interval, ts);

        CREATE TABLE IF NOT EXISTS scores (
            id BIGSERIAL PRIMARY KEY,
            ts BIGINT NOT NULL,
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            score JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_scores_asset_interval_ts ON scores (asset, interval, ts);

        CREATE TABLE IF NOT EXISTS indicators (
            id BIGSERIAL PRIMARY KEY,
            ts BIGINT NOT NULL,
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            indicators JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_indicators_asset_interval_ts ON indicators (asset, interval, ts);

        CREATE TABLE IF NOT EXISTS watchlist (
            asset TEXT PRIMARY KEY,
            meta JSONB
        );

        CREATE TABLE IF NOT EXISTS backfill_status (
            asset TEXT PRIMARY KEY,
            interval TEXT,
            last_ts BIGINT,
            updated_at TIMESTAMPTZ DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS models (
            id BIGSERIAL PRIMARY KEY,
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            model_name TEXT,
            metadata JSONB,
            path TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(ddl)
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception("init_db failed")
                raise
            finally:
                cur.close()

    # -----------------
    # Candles
    # -----------------
    def save_candles(self, df: pd.DataFrame, asset: str, interval: str, batch_size: int = 500):
        """Guarda velas en bloque. Se espera df con columnas ts (datetime or ms), open/high/low/close/volume.
        Normaliza y realiza upsert por (asset,interval,ts).
        """
        if df.empty:
            return
        df2 = df.copy()
        # normalize ts to ms int
        if pd.api.types.is_datetime64_any_dtype(df2["ts"]):
            df2["ts"] = (df2["ts"].astype('datetime64[ms]').astype('int64'))
        else:
            df2["ts"] = df2["ts"].astype('int64')
        records = df2[["ts", "open", "high", "low", "close", "volume"]].to_dict("records")

        insert_sql = """
        INSERT INTO candles (ts, asset, interval, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (asset, interval, ts) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
        """
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                params = [(r["ts"], asset, interval, r.get("open"), r.get("high"), r.get("low"), r.get("close"), r.get("volume")) for r in records]
                psycopg2.extras.execute_batch(cur, insert_sql, params, page_size=batch_size)
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception("save_candles failed for %s %s", asset, interval)
                raise
            finally:
                cur.close()

    def load_candles(self, asset: str, interval: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Carga velas y devuelve DataFrame con columnas ts (as datetime), open, high, low, close, volume
        ts filters are ms since epoch.
        """
        q = ["SELECT ts, open, high, low, close, volume FROM candles WHERE asset = %s AND interval = %s"]
        params: List[Any] = [asset, interval]
        if start_ts is not None:
            q.append("AND ts >= %s")
            params.append(start_ts)
        if end_ts is not None:
            q.append("AND ts <= %s")
            params.append(end_ts)
        q.append("ORDER BY ts ASC")
        if limit:
            q.append("LIMIT %s")
            params.append(limit)
        sql_q = " ".join(q) + ";"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, params)
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]) if rows else pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                return df
            finally:
                cur.close()

    # -----------------
    # Scores
    # -----------------
    def save_scores(self, df_scores: pd.DataFrame, asset: str, interval: str, batch_size: int = 500):
        """Guarda scores. df_scores expected to have ts (datetime or ms) and a column 'score' which is a dict/JSON serializable.
        """
        if df_scores.empty:
            return
        df2 = df_scores.copy()
        if pd.api.types.is_datetime64_any_dtype(df2["ts"]):
            df2["ts"] = df2["ts"].astype('datetime64[ms]').astype('int64')
        else:
            df2["ts"] = df2["ts"].astype('int64')
        records = df2[["ts", "score"]].to_dict("records")

        insert_sql = """
        INSERT INTO scores (ts, asset, interval, score)
        VALUES (%s, %s, %s, %s)
        """
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                params = [(r["ts"], asset, interval, psycopg2.extras.Json(r["score"])) for r in records]
                psycopg2.extras.execute_batch(cur, insert_sql, params, page_size=batch_size)
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception("save_scores failed for %s %s", asset, interval)
                raise
            finally:
                cur.close()

    def load_scores(self, asset: Optional[str] = None, interval: Optional[str] = None, start_ts: Optional[int] = None, end_ts: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Carga scores, posible filtrar por asset/interval. Devuelve ts (datetime UTC), asset, interval, score(dict)
        """
        q = ["SELECT ts, asset, interval, score FROM scores WHERE 1=1"]
        params: List[Any] = []
        if asset:
            q.append("AND asset = %s")
            params.append(asset)
        if interval:
            q.append("AND interval = %s")
            params.append(interval)
        if start_ts is not None:
            q.append("AND ts >= %s")
            params.append(start_ts)
        if end_ts is not None:
            q.append("AND ts <= %s")
            params.append(end_ts)
        q.append("ORDER BY ts DESC")
        if limit:
            q.append("LIMIT %s")
            params.append(limit)

        sql_q = " ".join(q) + ";"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, params)
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=["ts", "asset", "interval", "score"]) if rows else pd.DataFrame(columns=["ts", "asset", "interval", "score"])
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                return df
            finally:
                cur.close()

    def load_latest_scores(self, limit_per_asset: int = 1) -> pd.DataFrame:
        """Trae los últimos `limit_per_asset` scores por (asset, interval).
        Usa ROW_NUMBER() OVER (PARTITION BY asset, interval ORDER BY ts DESC)
        """
        sql_q = f"""
        SELECT ts, asset, interval, score FROM (
            SELECT ts, asset, interval, score,
                   ROW_NUMBER() OVER (PARTITION BY asset, interval ORDER BY ts DESC) as rn
            FROM scores
        ) t WHERE t.rn <= %s
        ORDER BY asset, interval, ts DESC;
        """
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (limit_per_asset,))
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=["ts", "asset", "interval", "score"]) if rows else pd.DataFrame(columns=["ts", "asset", "interval", "score"])
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                return df
            finally:
                cur.close()

    # -----------------
    # Watchlist / Backfill status
    # -----------------
    def add_watchlist_symbol(self, asset: str, asset_type: Optional[str] = None, added_by: Optional[str] = None) -> Dict[str, Any]:
        meta = {"asset_type": asset_type, "added_by": added_by, "created_at": datetime.now(timezone.utc).isoformat()}
        sql_q = "INSERT INTO watchlist (asset, meta) VALUES (%s, %s) ON CONFLICT (asset) DO UPDATE SET meta = EXCLUDED.meta RETURNING asset, meta;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (asset, psycopg2.extras.Json(meta)))
                row = cur.fetchone()
                conn.commit()
                return {"asset": row[0], "meta": row[1]}
            except Exception:
                conn.rollback()
                logger.exception("add_watchlist_symbol failed: %s", asset)
                raise
            finally:
                cur.close()

    def list_watchlist(self) -> List[Dict[str, Any]]:
        sql_q = "SELECT asset, meta FROM watchlist ORDER BY asset;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q)
                rows = cur.fetchall()
                return [{"asset": r[0], **({"meta": r[1]} if r[1] is not None else {})} for r in rows]
            finally:
                cur.close()

    def get_watchlist(self) -> List[Dict[str, Any]]:
        return self.list_watchlist()

    def remove_watchlist_symbol(self, asset: str) -> bool:
        sql_q = "DELETE FROM watchlist WHERE asset = %s;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (asset,))
                affected = cur.rowcount
                conn.commit()
                return affected > 0
            except Exception:
                conn.rollback()
                logger.exception("remove_watchlist_symbol failed: %s", asset)
                raise
            finally:
                cur.close()

    # Backfill status helpers
    def add_backfill_request(self, asset: str, interval: str, requested_by: Optional[str] = None) -> Dict[str, Any]:
        """Crea o actualiza backfill_status para indicar que se debe backfillear.
        La UI espera poder llamar a esto para crear solicitudes.
        """
        now = now_ms()
        meta = {"requested_by": requested_by}
        sql_q = "INSERT INTO backfill_status (asset, interval, last_ts, updated_at) VALUES (%s, %s, %s, now()) ON CONFLICT (asset) DO UPDATE SET interval = EXCLUDED.interval, updated_at = now() RETURNING asset, interval, last_ts, updated_at;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (asset, interval, None))
                row = cur.fetchone()
                conn.commit()
                return {"asset": row[0], "interval": row[1], "last_ts": row[2], "updated_at": row[3].isoformat() if row[3] else None}
            except Exception:
                conn.rollback()
                logger.exception("add_backfill_request failed: %s", asset)
                raise
            finally:
                cur.close()

    def update_backfill_status(self, asset: str, interval: Optional[str] = None, last_ts: Optional[int] = None):
        q = ["UPDATE backfill_status SET "]
        params: List[Any] = []
        sets = []
        if interval is not None:
            sets.append("interval = %s")
            params.append(interval)
        if last_ts is not None:
            sets.append("last_ts = %s")
            params.append(last_ts)
        sets.append("updated_at = now()")
        q.append(", ".join(sets))
        q.append("WHERE asset = %s RETURNING asset, interval, last_ts, updated_at;")
        params.append(asset)
        sql_q = " ".join(q)
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, params)
                row = cur.fetchone()
                conn.commit()
                if row:
                    return {"asset": row[0], "interval": row[1], "last_ts": row[2], "updated_at": row[3].isoformat() if row[3] else None}
                return None
            except Exception:
                conn.rollback()
                logger.exception("update_backfill_status failed: %s", asset)
                raise
            finally:
                cur.close()

    def get_backfill_status(self, asset: str) -> Optional[Dict[str, Any]]:
        sql_q = "SELECT asset, interval, last_ts, updated_at FROM backfill_status WHERE asset = %s;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (asset,))
                row = cur.fetchone()
                if not row:
                    return None
                return {"asset": row[0], "interval": row[1], "last_ts": row[2], "updated_at": row[3].isoformat() if row[3] else None}
            finally:
                cur.close()

    # -----------------
    # Models
    # -----------------
    def save_model_record(self, asset: str, interval: str, model_name: str, metadata: Dict[str, Any], path: str):
        sql_q = "INSERT INTO models (asset, interval, model_name, metadata, path) VALUES (%s, %s, %s, %s, %s) RETURNING id, created_at;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (asset, interval, model_name, psycopg2.extras.Json(metadata), path))
                row = cur.fetchone()
                conn.commit()
                return {"id": row[0], "created_at": row[1].isoformat()}
            except Exception:
                conn.rollback()
                logger.exception("save_model_record failed: %s %s", asset, interval)
                raise
            finally:
                cur.close()

    def get_latest_model_record(self, asset: str, interval: str) -> Optional[Dict[str, Any]]:
        sql_q = "SELECT id, model_name, metadata, path, created_at FROM models WHERE asset = %s AND interval = %s ORDER BY created_at DESC LIMIT 1;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (asset, interval))
                row = cur.fetchone()
                if not row:
                    return None
                return {"id": row[0], "model_name": row[1], "metadata": row[2], "path": row[3], "created_at": row[4].isoformat()}
            finally:
                cur.close()

    # -----------------
    # Listing helpers
    # -----------------
    def list_assets(self) -> List[str]:
        sql_q = "SELECT DISTINCT asset FROM candles ORDER BY asset;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q)
                rows = cur.fetchall()
                return [r[0] for r in rows]
            finally:
                cur.close()

    def list_intervals_for_asset(self, asset: str) -> List[str]:
        sql_q = "SELECT DISTINCT interval FROM candles WHERE asset = %s ORDER BY interval;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (asset,))
                rows = cur.fetchall()
                return [r[0] for r in rows]
            finally:
                cur.close()

    def list_models_for_asset_interval(self, asset: str, interval: str) -> List[Dict[str, Any]]:
        sql_q = "SELECT id, model_name, metadata, path, created_at FROM models WHERE asset = %s AND interval = %s ORDER BY created_at DESC;"
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_q, (asset, interval))
                rows = cur.fetchall()
                return [{"id": r[0], "model_name": r[1], "metadata": r[2], "path": r[3], "created_at": r[4].isoformat()} for r in rows]
            finally:
                cur.close()

    # -----------------
    # Health
    # -----------------
    def health(self) -> Dict[str, Any]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            return {"ok": True}
        except Exception:
            logger.exception("health check failed")
            return {"ok": False}


# Convenience factory
def make_storage_from_env(**kwargs) -> PostgresStorage:
    """Helper que crea la instancia leyendo DATABASE_URL o env vars.
    Kwargs override env vars.
    """
    cfg = {}
    cfg.update(kwargs)
    return PostgresStorage(**cfg)
