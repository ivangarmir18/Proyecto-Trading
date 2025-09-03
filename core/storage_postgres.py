# core/storage_postgres.py
import os
import json
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import pandas as pd
import psycopg2
from psycopg2 import pool
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PostgresStorage:
    """
    Postgres-backed storage (supports DATABASE_URL or separate env vars).
    Env vars: DATABASE_URL OR POSTGRES_HOST/POSTGRES_PORT/POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD
    """

    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 dbname: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 minconn: int = 1,
                 maxconn: int = 5):
        self._minconn = minconn
        self._maxconn = maxconn
        # Prefer DATABASE_URL if present (works with Supabase)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # Use DSN string
            try:
                self._pool = pool.SimpleConnectionPool(self._minconn, self._maxconn, dsn=database_url)
                logger.info("PostgresStorage initialized via DATABASE_URL.")
                return
            except Exception as e:
                logger.exception("Fallo al crear pool con DATABASE_URL: %s", e)
                # fallback to separate vars

        # Fallback: individual env vars (or provided args)
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.dbname = dbname or os.getenv("POSTGRES_DB", "trading")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")

        try:
            self._pool = pool.SimpleConnectionPool(
                self._minconn,
                self._maxconn,
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            logger.info("PostgresStorage initialized (host=%s db=%s user=%s)", self.host, self.dbname, self.user)
        except Exception as e:
            logger.exception("Error inicializando Postgres connection pool: %s", e)
            raise

        self._lock = threading.Lock()

    @contextmanager
    def _get_conn(self):
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn:
                try:
                    self._pool.putconn(conn)
                except Exception:
                    pass

    def init_db(self):
        """Create minimal schema if not exists."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS candles (
              ts BIGINT NOT NULL,
              timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
              open DOUBLE PRECISION,
              high DOUBLE PRECISION,
              low DOUBLE PRECISION,
              close DOUBLE PRECISION,
              volume DOUBLE PRECISION,
              asset TEXT NOT NULL,
              interval TEXT NOT NULL,
              PRIMARY KEY (ts, asset, interval)
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
              id SERIAL PRIMARY KEY,
              ts BIGINT NOT NULL,
              asset TEXT NOT NULL,
              interval TEXT NOT NULL,
              indicators JSONB,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS scores (
              id SERIAL PRIMARY KEY,
              ts BIGINT NOT NULL,
              asset TEXT NOT NULL,
              interval TEXT NOT NULL,
              model_id INTEGER,
              score JSONB,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS models (
              id SERIAL PRIMARY KEY,
              asset TEXT NOT NULL,
              interval TEXT NOT NULL,
              supabase_path TEXT NOT NULL,
              filename TEXT,
              meta JSONB,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_candles_asset_interval ON candles(asset, interval);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_models_asset_interval ON models(asset, interval);")
            conn.commit()
            cur.close()
        logger.info("Database schema ensured.")

    # ---------------------
    # Candles IO
    # ---------------------
    def save_candles(self, df: pd.DataFrame, batch: int = 500):
        required_cols = {"ts", "timestamp", "open", "high", "low", "close", "volume", "asset", "interval"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must contain columns {required_cols}")

        rows = [
            (
                int(row["ts"]),
                pd.to_datetime(row["timestamp"]).to_pydatetime(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row.get("volume", 0.0)),
                str(row["asset"]),
                str(row["interval"])
            )
            for _, row in df.iterrows()
        ]

        with self._get_conn() as conn:
            cur = conn.cursor()
            sql_insert = """
            INSERT INTO candles (ts, timestamp, open, high, low, close, volume, asset, interval)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ts, asset, interval) DO UPDATE
              SET open = EXCLUDED.open,
                  high = EXCLUDED.high,
                  low = EXCLUDED.low,
                  close = EXCLUDED.close,
                  volume = EXCLUDED.volume,
                  timestamp = EXCLUDED.timestamp;
            """
            for i in range(0, len(rows), batch):
                batch_rows = rows[i:i+batch]
                cur.executemany(sql_insert, batch_rows)
                conn.commit()
            cur.close()
        logger.info("Saved %d candle rows", len(rows))

    def load_candles(self, asset: str, interval: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        with self._get_conn() as conn:
            cur = conn.cursor()
            base_q = "SELECT ts, timestamp, open, high, low, close, volume, asset, interval FROM candles WHERE asset=%s AND interval=%s"
            params = [asset, interval]
            if start_ts is not None:
                base_q += " AND ts >= %s"
                params.append(int(start_ts))
            if end_ts is not None:
                base_q += " AND ts <= %s"
                params.append(int(end_ts))
            base_q += " ORDER BY ts ASC"
            if limit:
                base_q += f" LIMIT {int(limit)}"
            cur.execute(base_q, params)
            rows = cur.fetchall()
            cur.close()

        if not rows:
            return pd.DataFrame(columns=["ts", "timestamp", "open", "high", "low", "close", "volume", "asset", "interval"])

        df = pd.DataFrame(rows, columns=["ts", "timestamp", "open", "high", "low", "close", "volume", "asset", "interval"])
        df["ts"] = df["ts"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ---------------------
    # Scores IO
    # ---------------------
    def save_scores(self, df_scores: pd.DataFrame, batch: int = 500):
        required_cols = {"ts", "asset", "interval", "score"}
        if not required_cols.issubset(set(df_scores.columns)):
            raise ValueError(f"df_scores must contain {required_cols}")

        rows = []
        for _, row in df_scores.iterrows():
            rows.append((int(row["ts"]), str(row["asset"]), str(row["interval"]), row.get("model_id"), json.dumps(row["score"])))

        with self._get_conn() as conn:
            cur = conn.cursor()
            sql_insert = """
            INSERT INTO scores (ts, asset, interval, model_id, score)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING;
            """
            for i in range(0, len(rows), batch):
                cur.executemany(sql_insert, rows[i:i+batch])
                conn.commit()
            cur.close()
        logger.info("Saved %d score rows", len(rows))

    def load_scores(self, asset: str, interval: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None, model_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            base_q = "SELECT ts, asset, interval, model_id, score, created_at FROM scores WHERE asset=%s AND interval=%s"
            params = [asset, interval]
            if start_ts is not None:
                base_q += " AND ts >= %s"
                params.append(int(start_ts))
            if end_ts is not None:
                base_q += " AND ts <= %s"
                params.append(int(end_ts))
            if model_id:
                base_q += " AND model_id = %s"
                params.append(int(model_id))
            base_q += " ORDER BY ts ASC"
            cur.execute(base_q, params)
            rows = cur.fetchall()
            cur.close()
        results = []
        for r in rows:
            results.append({
                "ts": int(r[0]),
                "asset": r[1],
                "interval": r[2],
                "model_id": r[3],
                "score": r[4],
                "created_at": r[5].astimezone(timezone.utc).isoformat() if r[5] else None
            })
        return results

    # ---------------------
    # Model metadata
    # ---------------------
    def save_model_record(self, asset: str, interval: str, supabase_path: str, filename: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            sql_q = """
            INSERT INTO models (asset, interval, supabase_path, filename, meta)
            VALUES (%s,%s,%s,%s,%s)
            RETURNING id, created_at
            """
            cur.execute(sql_q, (asset, interval, supabase_path, filename, json.dumps(meta)))
            row = cur.fetchone()
            conn.commit()
            cur.close()
        if row:
            return {"id": row[0], "created_at": row[1].isoformat()}
        return {}

    def get_latest_model_record(self, asset: str, interval: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT id, asset, interval, supabase_path, filename, meta, created_at
            FROM models
            WHERE asset=%s AND interval=%s
            ORDER BY created_at DESC
            LIMIT 1
            """, (asset, interval))
            row = cur.fetchone()
            cur.close()
        if not row:
            return None
        return {
            "id": row[0],
            "asset": row[1],
            "interval": row[2],
            "supabase_path": row[3],
            "filename": row[4],
            "meta": row[5],
            "created_at": row[6].isoformat() if row[6] else None
        }

    # ---------------------
    # Helpers for dashboard
    # ---------------------
    def list_assets(self) -> List[str]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT asset FROM candles ORDER BY asset;")
            rows = cur.fetchall()
            cur.close()
        return [r[0] for r in rows] if rows else []

    def list_intervals_for_asset(self, asset: str) -> List[str]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT interval FROM candles WHERE asset=%s ORDER BY interval;", (asset,))
            rows = cur.fetchall()
            cur.close()
        return [r[0] for r in rows] if rows else []

    def list_models_for_asset_interval(self, asset: str, interval: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT id, filename, meta, created_at FROM models
            WHERE asset=%s AND interval=%s
            ORDER BY created_at DESC
            LIMIT %s
            """, (asset, interval, limit))
            rows = cur.fetchall()
            cur.close()
        res = []
        for r in rows:
            res.append({
                "id": r[0],
                "filename": r[1],
                "meta": r[2],
                "created_at": r[3].isoformat() if r[3] else None
            })
        return res

    # ---------------------
    # Health
    # ---------------------
    def health(self) -> Dict[str, Any]:
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT now();")
                _ = cur.fetchone()
                cur.close()
            return {"ok": True}
        except Exception as e:
            logger.exception("DB health check failed: %s", e)
            return {"ok": False, "error": str(e)}
