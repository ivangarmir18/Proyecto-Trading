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
# ----------------------------
# Wrappers de compatibilidad (NO DESTRUIDOS) para UI / fetch
# Añadir al final de core/storage_postgres.py
# ----------------------------
import pandas as _pd
import json as _json

# map to existing upsert/load functions if present
def save_candles(symbol: str, df: _pd.DataFrame) -> bool:
    """
    Compat wrapper para guardar velas.
    Internamente usa upsert_prices (si existe) o guarda CSV fallback.
    """
    try:
        if hasattr(PostgresStorage, "upsert_prices"):
            # use instance method if available
            try:
                return PostgresStorage().upsert_prices(symbol, df)
            except Exception:
                pass
        # fallback to CSV
        path = os.path.join(os.path.dirname(__file__), "..", "data", "cache", f"{symbol.replace('/','_')}.csv")
        df.to_csv(path, index=False)
        return True
    except Exception:
        return False

def load_candles(symbol: str, limit: int = 1000) -> _pd.DataFrame:
    """
    Compat wrapper para cargar velas.
    Internamente usa load_prices (si existe) o lee CSV de data/cache/.
    """
    try:
        if hasattr(PostgresStorage, "load_prices"):
            try:
                df = PostgresStorage().load_prices(symbol, limit=limit)
                if isinstance(df, _pd.DataFrame):
                    return df
            except Exception:
                pass
    except Exception:
        pass
    # fallback CSV
    path = os.path.join(os.path.dirname(__file__), "..", "data", "cache", f"{symbol.replace('/','_')}.csv")
    if os.path.exists(path):
        try:
            df = _pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
            if limit and len(df) > limit:
                return df.sort_values("timestamp").iloc[-limit:].reset_index(drop=True)
            return df
        except Exception:
            return _pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    return _pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

def list_assets() -> list:
    """
    Intentar leer assets desde la tabla si existe, sino fallback a CSVs en data/config.
    """
    try:
        if hasattr(PostgresStorage, "list_assets"):
            try:
                return PostgresStorage().list_assets()
            except Exception:
                pass
    except Exception:
        pass
    # fallback: read config CSVs
    cfgs = []
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "config"))
    for name in ("actions.csv","cryptos.csv","crypto.csv"):
        p = os.path.join(cfg_dir, name)
        if os.path.exists(p):
            try:
                df = _pd.read_csv(p, dtype=str, keep_default_na=False)
                if "symbol" in df.columns:
                    cfgs.extend(df["symbol"].astype(str).tolist())
                else:
                    cfgs.extend(df.iloc[:,0].astype(str).tolist())
            except Exception:
                continue
    seen = set(); uniq=[]
    for s in cfgs:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

# Simple settings persistence using a small json fallback if storage doesn't provide it
def save_setting(key: str, value) -> bool:
    try:
        if hasattr(PostgresStorage, "upsert_setting"):
            try:
                return PostgresStorage().upsert_setting(key, value)
            except Exception:
                pass
    except Exception:
        pass
    # fallback to JSON file
    try:
        settings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "config", "settings.json"))
        data = {}
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf8") as fh:
                    data = _json.load(fh)
            except Exception:
                data = {}
        data[key] = value
        with open(settings_path, "w", encoding="utf8") as fh:
            _json.dump(data, fh, indent=2, default=str)
        return True
    except Exception:
        return False

def load_setting(key: str, default=None):
    try:
        if hasattr(PostgresStorage, "load_setting"):
            try:
                return PostgresStorage().load_setting(key, default)
            except Exception:
                pass
    except Exception:
        pass
    settings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "config", "settings.json"))
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf8") as fh:
                data = _json.load(fh)
            return data.get(key, default)
        except Exception:
            return default
    return default

# expose short aliases for compatibility imports like `from core.storage_postgres import save_candles`
__all__ = list(globals().keys())

# --- Inicio parche storage_postgres.py: backfill_status + prune helpers ---
import os, time, logging

logger = logging.getLogger(__name__)

def _connect_psycopg2(dsn=None):
    try:
        import psycopg2
    except Exception as e:
        raise RuntimeError("psycopg2 no disponible: %s" % e)
    dsn = dsn or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL no encontrada para conectar a Postgres")
    return psycopg2.connect(dsn)

def ensure_backfill_status_table(conn=None):
    """
    Crea tabla 'backfill_status' si no existe.
    Campos: asset TEXT, interval TEXT, last_ts BIGINT, status TEXT, updated_at TIMESTAMP
    """
    close_conn = False
    if conn is None:
        conn = _connect_psycopg2()
        close_conn = True
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS backfill_status (
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            last_ts BIGINT,
            status TEXT,
            updated_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (asset, interval)
        )
    """)
    conn.commit()
    if close_conn:
        conn.close()

def get_backfill_status(asset, interval, conn=None):
    """
    Retorna last_ts (int) o None si no hay registro.
    """
    close_conn = False
    if conn is None:
        conn = _connect_psycopg2()
        close_conn = True
    ensure_backfill_status_table(conn)
    cur = conn.cursor()
    cur.execute("SELECT last_ts, status, updated_at FROM backfill_status WHERE asset=%s AND interval=%s", (asset, interval))
    row = cur.fetchone()
    if close_conn:
        conn.close()
    return row[0] if row else None

def update_backfill_status(asset, interval, last_ts, status="done", conn=None):
    """
    Inserta/actualiza el registro de backfill con last_ts (int).
    """
    close_conn = False
    if conn is None:
        conn = _connect_psycopg2()
        close_conn = True
    ensure_backfill_status_table(conn)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO backfill_status (asset, interval, last_ts, status, updated_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (asset, interval) DO UPDATE SET last_ts = EXCLUDED.last_ts, status = EXCLUDED.status, updated_at = now()
    """, (asset, interval, int(last_ts) if last_ts is not None else None, status))
    conn.commit()
    if close_conn:
        conn.close()

def prune_old_candles_postgres(retention_map=None, conn=None, dry_run=True):
    """
    Ejecuta pruning en tabla 'prices' asumiendo columnas: asset, interval, ts (en segundos).
    retention_map: dict interval -> days
    dry_run=True solo cuenta filas.
    """
    if retention_map is None:
        retention_map = {
            "5m": 7,
            "15m": 14,
            "30m": 28,
            "1h": 40,
            "4h": 70,
            "12h": 105,
            "1d": 200
        }
    close_conn = False
    if conn is None:
        conn = _connect_psycopg2()
        close_conn = True
    cur = conn.cursor()
    now_s = int(time.time())
    results = {}
    for interval, days in retention_map.items():
        cutoff = now_s - int(days) * 86400  # segundos
        if dry_run:
            cur.execute("SELECT COUNT(*) FROM prices WHERE interval = %s AND ts < %s", (interval, cutoff))
            cnt = cur.fetchone()[0]
            results[interval] = cnt
            logger.info("prune dry-run interval=%s days=%s -> rows=%s", interval, days, cnt)
        else:
            cur.execute("DELETE FROM prices WHERE interval = %s AND ts < %s", (interval, cutoff))
            logger.info("prune executed interval=%s days=%s", interval, days)
    if not dry_run:
        conn.commit()
    if close_conn:
        conn.close()
    return results

# Opcional: si existe la clase PostgresStorage, adjuntamos métodos convenience
try:
    PostgresStorage  # type: ignore
except Exception:
    PostgresStorage = None

if PostgresStorage is not None:
    def _ps_get_backfill_status(self, asset, interval):
        # intenta usar la conexión interna si existe, si no fallback a funciones arriba
        conn = getattr(self, "conn", None)
        return get_backfill_status(asset, interval, conn=conn)
    def _ps_update_backfill_status(self, asset, interval, last_ts, status="done"):
        conn = getattr(self, "conn", None)
        return update_backfill_status(asset, interval, last_ts, status=status, conn=conn)
    def _ps_prune_old_candles(self, retention_map=None, dry_run=True):
        conn = getattr(self, "conn", None)
        return prune_old_candles_postgres(retention_map=retention_map, conn=conn, dry_run=dry_run)
    # Attach:
    setattr(PostgresStorage, "get_backfill_status", _ps_get_backfill_status)
    setattr(PostgresStorage, "update_backfill_status", _ps_update_backfill_status)
    setattr(PostgresStorage, "prune_old_candles", _ps_prune_old_candles)

# --- Fin parche storage_postgres.py ---

# --- Inicio parche storage_postgres: watchlist, backfill request y upsert_score helpers ---
import json
import logging

logger = logging.getLogger(__name__)

def _ensure_watchlist_table(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS watchlist (
        asset TEXT PRIMARY KEY,
        meta JSONB,
        added_by TEXT,
        added_at TIMESTAMP DEFAULT now()
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

def _ensure_scores_table(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS scores (
        asset TEXT NOT NULL,
        ts BIGINT NOT NULL,
        score DOUBLE PRECISION,
        method TEXT,
        components JSONB,
        created_at TIMESTAMP DEFAULT now(),
        PRIMARY KEY (asset, ts)
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

# ----- Watchlist helpers as methods for PostgresStorage -----
def _ps_add_watchlist_symbol(self, asset: str, asset_type: Optional[str]=None, added_by: Optional[str]=None):
    """
    Añade asset a la watchlist; crea registro en assets si hace falta.
    """
    with self.get_conn() as conn:
        cur = conn.cursor()
        _ensure_watchlist_table(conn)
        # upsert into assets (best-effort)
        try:
            cur.execute("INSERT INTO assets (symbol, type, name) VALUES (%s, %s, %s) ON CONFLICT (symbol) DO NOTHING",
                        (asset, asset_type or "crypto", None))
        except Exception:
            conn.rollback()
        # insert/update watchlist
        try:
            cur.execute("INSERT INTO watchlist (asset, meta, added_by) VALUES (%s, %s, %s) ON CONFLICT (asset) DO UPDATE SET meta = EXCLUDED.meta, added_by = EXCLUDED.added_by",
                        (asset, json.dumps({}), added_by))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.exception("add_watchlist_symbol failed: %s", e)
            return False

def _ps_list_watchlist(self):
    with self.get_conn() as conn:
        _ensure_watchlist_table(conn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT asset, meta, added_by, added_at FROM watchlist ORDER BY added_at DESC")
            rows = cur.fetchall()
            return rows

def _ps_remove_watchlist_symbol(self, asset: str):
    with self.get_conn() as conn:
        _ensure_watchlist_table(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM watchlist WHERE asset = %s", (asset,))
            conn.commit()
            return cur.rowcount > 0

def _ps_update_watchlist_meta(self, asset: str, meta: dict):
    with self.get_conn() as conn:
        _ensure_watchlist_table(conn)
        with conn.cursor() as cur:
            cur.execute("UPDATE watchlist SET meta = %s WHERE asset = %s", (json.dumps(meta), asset))
            conn.commit()
            return cur.rowcount > 0

# ----- Backfill request (jobs) helper -----
def _ps_add_backfill_request(self, asset: str, interval: Optional[str]=None, requested_by: Optional[str]=None):
    """
    Inserta una fila en la tabla jobs para que el worker la recoja.
    Guarda: name='backfill', status='pending', details={'asset':..,'interval':..,'requested_by':..}
    """
    details = {"asset": asset, "interval": interval, "requested_by": requested_by}
    sql = "INSERT INTO jobs (name, status, details) VALUES (%s, %s, %s) RETURNING id"
    try:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, ('backfill', 'pending', json.dumps(details)))
                job_id = cur.fetchone()[0]
            conn.commit()
        return int(job_id)
    except Exception as e:
        logger.exception("add_backfill_request failed: %s", e)
        return None

# ----- upsert_score -----
def _ps_upsert_score(self, asset: str, ts: int, score: float, components: dict, method: Optional[str] = None):
    """
    Inserta/actualiza un score en tabla 'scores'.
    """
    try:
        with self.get_conn() as conn:
            _ensure_scores_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO scores (asset, ts, score, method, components)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (asset, ts) DO UPDATE SET
                        score = EXCLUDED.score,
                        method = EXCLUDED.method,
                        components = EXCLUDED.components,
                        created_at = now()
                    """,
                    (asset, int(ts), float(score) if score is not None else None, method, json.dumps(components) if components else None)
                )
            conn.commit()
        return True
    except Exception as e:
        logger.exception("upsert_score failed: %s", e)
        return False

# Attach methods to PostgresStorage class if present
try:
    PostgresStorage  # type: ignore
except NameError:
    PostgresStorage = None

if PostgresStorage is not None:
    # attach if missing (do not override existing implementations)
    if not hasattr(PostgresStorage, "add_watchlist_symbol"):
        setattr(PostgresStorage, "add_watchlist_symbol", _ps_add_watchlist_symbol)
    if not hasattr(PostgresStorage, "list_watchlist"):
        setattr(PostgresStorage, "list_watchlist", _ps_list_watchlist)
    if not hasattr(PostgresStorage, "remove_watchlist_symbol"):
        setattr(PostgresStorage, "remove_watchlist_symbol", _ps_remove_watchlist_symbol)
    if not hasattr(PostgresStorage, "update_watchlist_meta"):
        setattr(PostgresStorage, "update_watchlist_meta", _ps_update_watchlist_meta)
    if not hasattr(PostgresStorage, "add_backfill_request"):
        setattr(PostgresStorage, "add_backfill_request", _ps_add_backfill_request)
    if not hasattr(PostgresStorage, "upsert_score"):
        setattr(PostgresStorage, "upsert_score", _ps_upsert_score)

# --- Fin parche storage_postgres ---
