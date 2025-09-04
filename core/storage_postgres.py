# core/storage_postgres.py
"""
Postgres-backed storage for Proyecto-Trading (implementación completa y compatible).

Características:
- Pool de conexiones (psycopg2 SimpleConnectionPool)
- init_db() crea tablas: candles, scores, indicators, watchlist, backfill_status, models
- save_candles / load_candles (batch + ON CONFLICT upsert)
- save_scores / load_scores
- save_model_record / get_latest_model_record
- list_assets / list_intervals_for_asset / list_models_for_asset_interval
- watchlist helpers: add/remove/list/get
- health() para checks de conexión
- make_storage_from_env() helper para creación conveniente

Requiere:
- psycopg2, pandas
- DATABASE_URL o variables POSTGRES_HOST/PORT/DB/USER/PASSWORD

Notas:
- ts almacenado como BIGINT (epoch seconds). Si tus datos vienen en ms, la función convierte automáticamente.
- Los DataFrames usados para save_* deben tener columnas mínimas indicadas en cada función.
"""

import os
import logging
import glob
from typing import Optional, List, Dict, Any

import pandas as pd
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values, Json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)


class PostgresStorage:
    def __init__(self,
                 database_url: Optional[str] = None,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 dbname: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 minconn: int = 1,
                 maxconn: int = 8):
        """
        Inicializa pool. Prioriza DATABASE_URL. Si no existe, usa POSTGRES_* env.
        """
        database_url = database_url or os.getenv("DATABASE_URL")
        self._minconn = minconn
        self._maxconn = maxconn
        self._pool = None
        self._database_url = database_url

        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password

        # If no DATABASE_URL, build it from env / params later when needed
        self._ensure_pool()

    def _build_dsn_from_env(self) -> str:
        host = self.host or os.getenv("POSTGRES_HOST", "localhost")
        port = str(self.port or int(os.getenv("POSTGRES_PORT", "5432")))
        db = self.dbname or os.getenv("POSTGRES_DB", "trading")
        user = self.user or os.getenv("POSTGRES_USER", "postgres")
        pwd = self.password or os.getenv("POSTGRES_PASSWORD", "")
        # simple DSN for psycopg2
        return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"

    def _ensure_pool(self):
        if self._pool:
            return
        try:
            dsn = self._database_url or self._build_dsn_from_env()
            self._pool = pool.SimpleConnectionPool(self._minconn, self._maxconn, dsn)
            logger.info("Postgres pool created (min=%s max=%s).", self._minconn, self._maxconn)
        except Exception:
            logger.exception("Failed creating Postgres pool with DSN.")
            raise

    def close(self):
        if self._pool:
            try:
                self._pool.closeall()
                logger.info("Closed Postgres pool")
            except Exception:
                logger.exception("Error closing pool")

    def _get_conn(self):
        self._ensure_pool()
        try:
            return self._pool.getconn()
        except Exception:
            logger.exception("Failed to get conn from pool")
            raise

    def _put_conn(self, conn):
        if not conn:
            return
        try:
            self._pool.putconn(conn)
        except Exception:
            logger.exception("Failed to return conn to pool")

    # -------------------------
    # Schema / Init
    # -------------------------
    def init_db(self):
        """Create all tables + essential indexes if they don't exist."""
        create_statements = [
            """
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
            """,
            "CREATE INDEX IF NOT EXISTS idx_candles_asset_interval_ts ON candles (asset, interval, ts);",
            """
            CREATE TABLE IF NOT EXISTS scores (
              id BIGSERIAL PRIMARY KEY,
              ts BIGINT NOT NULL,
              asset TEXT NOT NULL,
              interval TEXT NOT NULL,
              score JSONB,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_scores_asset_interval_ts ON scores (asset, interval, ts);",
            """
            CREATE TABLE IF NOT EXISTS indicators (
              id BIGSERIAL PRIMARY KEY,
              ts BIGINT NOT NULL,
              asset TEXT NOT NULL,
              interval TEXT NOT NULL,
              indicators JSONB,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_indicators_asset_interval_ts ON indicators (asset, interval, ts);",
            """
            CREATE TABLE IF NOT EXISTS watchlist (
              asset TEXT PRIMARY KEY,
              meta JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS backfill_status (
              asset TEXT PRIMARY KEY,
              interval TEXT,
              last_ts BIGINT,
              updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS models (
              id BIGSERIAL PRIMARY KEY,
              name TEXT,
              asset TEXT,
              interval TEXT,
              metadata JSONB,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_models_asset_interval ON models (asset, interval);",
        ]
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            for s in create_statements:
                cur.execute(s)
            conn.commit()
            cur.close()
            logger.info("init_db: ensured tables and indexes")
        except Exception:
            logger.exception("init_db failed")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)

    # -------------------------
    # Candles
    # -------------------------
    @staticmethod
    def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ts column as epoch seconds ints and timestamp present."""
        if df is None or df.empty:
            return df
        df = df.copy()
        if 'ts' not in df.columns and 'timestamp' in df.columns:
            df['ts'] = df['timestamp'].apply(lambda x: int(pd.to_datetime(x).timestamp()))
        if 'timestamp' not in df.columns and 'ts' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
        # convert ms to s if necessary
        def _to_seconds(x):
            try:
                v = int(x)
                if v > 1_000_000_000_000:
                    return int(v / 1000)
                return v
            except Exception:
                return int(pd.to_datetime(x).timestamp())
        df['ts'] = df['ts'].apply(_to_seconds)
        return df

    def save_candles(self, df: pd.DataFrame, asset: Optional[str] = None, interval: Optional[str] = None, batch_size: int = 1000) -> int:
        """
        Upsert candles. Returns number of rows processed.
        Expects (or will create) columns: ts, timestamp, open, high, low, close, volume, asset, interval
        """
        if df is None or df.empty:
            logger.debug("save_candles: empty df")
            return 0
        df = df.copy()
        # Ensure ts/timestamp
        df = self._ensure_ts(df)
        # Fill missing columns
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c not in df.columns:
                df[c] = None
        if 'asset' not in df.columns:
            if not asset:
                raise ValueError("asset must be provided")
            df['asset'] = asset
        if 'interval' not in df.columns:
            if not interval:
                raise ValueError("interval must be provided")
            df['interval'] = interval
        # Drop rows without ts
        df = df.dropna(subset=['ts'])
        # deduplicate by ts/asset/interval keep latest
        df = df.sort_values('timestamp').drop_duplicates(subset=['ts','asset','interval'], keep='last')
        rows = []
        for _, r in df.iterrows():
            rows.append((int(r['ts']),
                         pd.to_datetime(r['timestamp']),
                         None if pd.isna(r.get('open')) else float(r.get('open')),
                         None if pd.isna(r.get('high')) else float(r.get('high')),
                         None if pd.isna(r.get('low')) else float(r.get('low')),
                         None if pd.isna(r.get('close')) else float(r.get('close')),
                         None if pd.isna(r.get('volume')) else float(r.get('volume')),
                         str(r['asset']),
                         str(r['interval'])))
        insert_q = """
        INSERT INTO candles (ts, timestamp, open, high, low, close, volume, asset, interval)
        VALUES %s
        ON CONFLICT (ts, asset, interval) DO UPDATE SET
          open = EXCLUDED.open,
          high = EXCLUDED.high,
          low = EXCLUDED.low,
          close = EXCLUDED.close,
          volume = EXCLUDED.volume,
          timestamp = EXCLUDED.timestamp
        """
        conn = None
        total = 0
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                execute_values(cur, insert_q, batch, page_size=batch_size)
                total += len(batch)
            conn.commit()
            cur.close()
            logger.info("save_candles: upserted %d rows", total)
            return total
        except Exception:
            logger.exception("save_candles failed")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)

    def load_candles(self, asset: str, interval: str,
                     start_ts: Optional[int] = None, end_ts: Optional[int] = None,
                     limit: Optional[int] = None, ascending: bool = True) -> pd.DataFrame:
        """
        Load candles as DataFrame. Returns columns: ts,timestamp,open,high,low,close,volume,asset,interval
        """
        if not asset or not interval:
            raise ValueError("asset & interval required")
        conditions = ["asset = %s", "interval = %s"]
        params = [asset, interval]
        if start_ts is not None:
            conditions.append("ts >= %s"); params.append(int(start_ts))
        if end_ts is not None:
            conditions.append("ts <= %s"); params.append(int(end_ts))
        where = " AND ".join(conditions)
        order = "ASC" if ascending else "DESC"
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        q = f"SELECT ts, timestamp, open, high, low, close, volume, asset, interval FROM candles WHERE {where} ORDER BY ts {order} {limit_clause};"
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(q, tuple(params))
            rows = cur.fetchall()
            cur.close()
            if not rows:
                return pd.DataFrame(columns=['ts','timestamp','open','high','low','close','volume','asset','interval'])
            df = pd.DataFrame(rows, columns=['ts','timestamp','open','high','low','close','volume','asset','interval'])
            df['ts'] = df['ts'].astype('int64')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception:
            logger.exception("load_candles failed")
            raise
        finally:
            if conn:
                self._put_conn(conn)

    # -------------------------
    # Scores
    # -------------------------
    def save_scores(self, df: pd.DataFrame) -> int:
        """
        Save scoring results.
        Expected columns: ts, asset, interval, score (dict)
        """
        if df is None or df.empty:
            return 0
        df = df.copy()
        if 'ts' not in df.columns:
            raise ValueError("save_scores expects ts")
        if 'asset' not in df.columns or 'interval' not in df.columns:
            raise ValueError("save_scores expects asset and interval")
        rows = []
        for _, r in df.iterrows():
            rows.append((int(r['ts']), str(r['asset']), str(r['interval']), Json(r.get('score') if 'score' in r else {})))
        insert_q = "INSERT INTO scores (ts, asset, interval, score) VALUES %s"
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            execute_values(cur, insert_q, rows)
            conn.commit()
            cur.close()
            logger.info("save_scores: inserted %d rows", len(rows))
            return len(rows)
        except Exception:
            logger.exception("save_scores failed")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)

    def load_scores(self, asset: Optional[str] = None, interval: Optional[str] = None,
                    start_ts: Optional[int] = None, end_ts: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        conds = []
        params = []
        if asset:
            conds.append("asset = %s"); params.append(asset)
        if interval:
            conds.append("interval = %s"); params.append(interval)
        if start_ts is not None:
            conds.append("ts >= %s"); params.append(int(start_ts))
        if end_ts is not None:
            conds.append("ts <= %s"); params.append(int(end_ts))
        where = " AND ".join(conds) if conds else "TRUE"
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        q = f"SELECT ts, asset, interval, score, created_at FROM scores WHERE {where} ORDER BY ts DESC {limit_clause};"
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(q, tuple(params))
            rows = cur.fetchall()
            cur.close()
            if not rows:
                return pd.DataFrame(columns=['ts','asset','interval','score','created_at'])
            df = pd.DataFrame(rows, columns=['ts','asset','interval','score','created_at'])
            df['ts'] = df['ts'].astype('int64')
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df
        except Exception:
            logger.exception("load_scores failed")
            raise
        finally:
            if conn:
                self._put_conn(conn)

    # -------------------------
    # Models metadata
    # -------------------------
    def save_model_record(self, name: str, asset: str, interval: str, metadata: Dict[str, Any]) -> int:
        """
        Save a model record (metadata JSON). Returns id.
        """
        q = "INSERT INTO models (name, asset, interval, metadata) VALUES (%s,%s,%s,%s) RETURNING id;"
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(q, (name, asset, interval, Json(metadata)))
            new_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            logger.info("save_model_record: id=%s", new_id)
            return int(new_id)
        except Exception:
            logger.exception("save_model_record failed")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)

    def get_latest_model_record(self, asset: str, interval: str) -> Optional[Dict[str, Any]]:
        q = "SELECT id, name, metadata, created_at FROM models WHERE asset = %s AND interval = %s ORDER BY created_at DESC LIMIT 1;"
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(q, (asset, interval))
            row = cur.fetchone()
            cur.close()
            if not row:
                return None
            return {"id": row[0], "name": row[1], "metadata": row[2], "created_at": row[3]}
        except Exception:
            logger.exception("get_latest_model_record failed")
            raise
        finally:
            if conn:
                self._put_conn(conn)

    # -------------------------
    # Listing helpers
    # -------------------------
    def list_assets(self) -> List[str]:
        """Return distinct assets from candles; fallback to watchlist / data/config/*.csv"""
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT asset FROM candles;")
            rows = cur.fetchall()
            cur.close()
            if rows:
                return sorted([r[0] for r in rows])
        except Exception:
            logger.debug("list_assets query failed, falling back to watchlist/csv")
        finally:
            if conn:
                self._put_conn(conn)
        # fallback: watchlist
        wl = self.list_watchlist()
        if wl:
            return wl
        # fallback: data/config CSVs
        assets = set()
        try:
            for p in glob.glob("data/config/*.csv"):
                try:
                    df = pd.read_csv(p)
                    for c in ("asset","symbol","ticker"):
                        if c in df.columns:
                            assets.update(df[c].astype(str).tolist())
                            break
                except Exception:
                    logger.debug("reading config csv %s failed", p)
        except Exception:
            logger.exception("list_assets fallback failed")
        return sorted(list(assets))

    def list_intervals_for_asset(self, asset: str) -> List[str]:
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT interval FROM candles WHERE asset = %s;", (asset,))
            rows = cur.fetchall()
            cur.close()
            if rows:
                return sorted([r[0] for r in rows])
            return []
        except Exception:
            logger.exception("list_intervals_for_asset failed")
            raise
        finally:
            if conn:
                self._put_conn(conn)

    def list_models_for_asset_interval(self, asset: str, interval: str) -> List[Dict[str, Any]]:
        q = "SELECT id, name, metadata, created_at FROM models WHERE asset = %s AND interval = %s ORDER BY created_at DESC;"
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(q, (asset, interval))
            rows = cur.fetchall()
            cur.close()
            return [{"id": r[0], "name": r[1], "metadata": r[2], "created_at": r[3]} for r in rows]
        except Exception:
            logger.exception("list_models_for_asset_interval failed")
            raise
        finally:
            if conn:
                self._put_conn(conn)

    # -------------------------
    # Watchlist helpers
    # -------------------------
    def _ensure_watchlist_table(self):
        # init_db should have created it, but be safe
        try:
            self.init_db()
        except Exception:
            logger.exception("Ensuring watchlist table via init_db failed")

    def add_watchlist_symbol(self, asset: str, meta: Optional[Dict[str, Any]] = None):
        self._ensure_watchlist_table()
        conn = None
        q = "INSERT INTO watchlist (asset, meta) VALUES (%s,%s) ON CONFLICT (asset) DO UPDATE SET meta = EXCLUDED.meta;"
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(q, (asset, Json(meta or {})))
            conn.commit()
            cur.close()
            logger.info("add_watchlist_symbol: %s", asset)
        except Exception:
            logger.exception("add_watchlist_symbol failed")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)

    def list_watchlist(self) -> List[str]:
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT asset FROM watchlist;")
            rows = cur.fetchall()
            cur.close()
            if rows:
                return [r[0] for r in rows]
        except Exception:
            logger.debug("watchlist table not available or empty")
        finally:
            if conn:
                self._put_conn(conn)
        # fallback to config CSVs
        assets = set()
        try:
            for p in glob.glob("data/config/*.csv"):
                try:
                    df = pd.read_csv(p)
                    for c in ("asset","symbol","ticker"):
                        if c in df.columns:
                            assets.update(df[c].astype(str).tolist())
                            break
                except Exception:
                    logger.debug("reading config csv %s failed", p)
        except Exception:
            logger.exception("list_watchlist fallback failed")
        return sorted(list(assets))

    def get_watchlist(self) -> List[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT asset, meta FROM watchlist;")
            rows = cur.fetchall()
            cur.close()
            return [{"asset": r[0], "meta": r[1]} for r in rows]
        except Exception:
            logger.exception("get_watchlist failed")
            raise
        finally:
            if conn:
                self._put_conn(conn)

    def remove_watchlist_symbol(self, asset: str):
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("DELETE FROM watchlist WHERE asset = %s;", (asset,))
            conn.commit()
            cur.close()
            logger.info("remove_watchlist_symbol: %s", asset)
        except Exception:
            logger.exception("remove_watchlist_symbol failed")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)

    # -------------------------
    # Backfill status helpers
    # -------------------------
    def update_backfill_status(self, asset: str, interval: str, last_ts: int):
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO backfill_status (asset, interval, last_ts, updated_at) VALUES (%s,%s,%s,now()) "
                "ON CONFLICT (asset) DO UPDATE SET last_ts = EXCLUDED.last_ts, interval = EXCLUDED.interval, updated_at = now();",
                (asset, interval, int(last_ts))
            )
            conn.commit()
            cur.close()
        except Exception:
            logger.exception("update_backfill_status failed")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)

    def get_backfill_status(self, asset: str) -> Optional[int]:
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT last_ts FROM backfill_status WHERE asset = %s;", (asset,))
            row = cur.fetchone()
            cur.close()
            if row:
                return int(row[0])
            return None
        except Exception:
            logger.exception("get_backfill_status failed")
            raise
        finally:
            if conn:
                self._put_conn(conn)

    # -------------------------
    # Health / utils
    # -------------------------
    def health(self) -> Dict[str, Any]:
        """
        Return a dict with simple health info (can be extended)
        """
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            cur.fetchone()
            cur.close()
            return {"ok": True}
        except Exception:
            logger.exception("health check failed")
            return {"ok": False}
        finally:
            if conn:
                self._put_conn(conn)


# Convenience factory
def make_storage_from_env(**kwargs) -> PostgresStorage:
    """
    Helper that creates PostgresStorage reading DATABASE_URL or env variables.
    Any kwargs override env.
    """
    return PostgresStorage(**kwargs)
