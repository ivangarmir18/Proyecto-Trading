# core/storage_postgres.py
"""
Postgres storage backend for Proyecto Trading.

- Proporciona: PostgresStorage class con pooling, init_db, save_candles (bulk upsert),
  get_ohlcv, purge_old_data, get_last_ts, close.
- Interfaz compatible con fetcher: puedes pasar storage.save_callback (o make_save_callback(storage))
  como `save_callback` a core.fetch.Fetcher.fetch_ohlcv(...) y backfill_range(...).
- Lee configuración desde parámetros o variables de entorno:
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
- Requiere: psycopg2-binary, pandas
"""

from __future__ import annotations
import os
import time
import math
import logging
from typing import Optional, Dict, Any, Iterable, List, Tuple
from contextlib import contextmanager

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger("core.storage_postgres")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.environ.get("STORAGE_LOG_LEVEL", "INFO"))


DEFAULT_POOL_MIN = 1
DEFAULT_POOL_MAX = int(os.getenv("POSTGRES_POOL_MAX", "5"))


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is None:
        logger.debug("ENV variable %s not set, default %s", name, default)
    return v


class PostgresStorage:
    """
    Clase principal para persistencia en Postgres.

    Usage:
        storage = PostgresStorage()  # leerá creds de env si no pasas params
        storage.init_db()
        storage.save_candles(df, 'BTCUSDT', '5m')
        df = storage.get_ohlcv('BTCUSDT', '5m', start_ms, end_ms)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        minconn: int = DEFAULT_POOL_MIN,
        maxconn: int = DEFAULT_POOL_MAX,
        connect_timeout: int = 10,
        application_name: str = "proyecto-trading-storage",
    ):
        # leer env si no vienen parámetros
        # Si existe DATABASE_URL en el .env, usarlo directamente (Render)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            self._dsn = database_url
            self._use_url = True
        else:
            host = host or _env("POSTGRES_HOST", "localhost")
            port = port or int(_env("POSTGRES_PORT", "5432"))
            dbname = dbname or _env("POSTGRES_DB", "trading")
            user = user or _env("POSTGRES_USER", "postgres")
            password = password or _env("POSTGRES_PASSWORD", "")
            self._dsn = {
                "host": host,
                "port": port,
                "dbname": dbname,
                "user": user,
                "password": password,
                "connect_timeout": connect_timeout,
                "application_name": application_name,
            }
            self._use_url = False


        self._minconn = minconn
        self._maxconn = maxconn
        self._pool: Optional[ThreadedConnectionPool] = None
        self._ensure_pool()

        # configuración para operaciones
        self._upsert_batch_size = int(os.getenv("STORAGE_UPSERT_BATCH", "500"))  # filas por batch
        self._max_retry = int(os.getenv("STORAGE_MAX_RETRY", "3"))
        logger.info("PostgresStorage inicializado dsn=%s minconn=%d maxconn=%d", {k: self._dsn[k] for k in ("host", "port", "dbname")}, minconn, maxconn)

    def _ensure_pool(self):
        if self._pool:
            return
        dsn_conn = " ".join(f"{k}={v}" for k, v in self._dsn.items() if v is not None and k != "password")
        # Use actual DSN dict to build connection string including password:
        try:
            if self._use_url:
                # Conexión directa usando DATABASE_URL
                self._pool = ThreadedConnectionPool(self._minconn, self._maxconn, self._dsn)
            else:
                # Conexión usando parámetros individuales
                conn_str = " ".join(f"{k}='{v}'" for k, v in self._dsn.items() if v is not None)
                self._pool = ThreadedConnectionPool(self._minconn, self._maxconn, conn_str)
            logger.debug("ThreadedConnectionPool creado min=%d max=%d", self._minconn, self._maxconn)
        except Exception:
            logger.exception("No se pudo crear ThreadedConnectionPool")
            raise

            logger.debug("ThreadedConnectionPool creado min=%d max=%d", self._minconn, self._maxconn)
        except Exception:
            # último recurso: intentar conexión simple para retornar error legible
            logger.exception("No se pudo crear ThreadedConnectionPool con dsn=%s", dsn_conn)
            raise

    @contextmanager
    def get_conn(self):
        """
        Context manager que coge/retorna conexión del pool y hace commit/rollback automáticamente.
        Yields psycopg2 connection.
        """
        if not self._pool:
            self._ensure_pool()
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            # commit aquí solo si no hubo excepción
            conn.commit()
        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    logger.exception("Rollback fallido")
            logger.exception("Error dentro de get_conn")
            raise
        finally:
            if conn:
                try:
                    self._pool.putconn(conn)
                except Exception:
                    logger.exception("No se pudo devolver conexión al pool")

    # ---------- DDL / Inicialización ----------
    def init_db(self):
        """
        Crea las tablas necesarias si no existen.
        Diseñado para ser idempotente.
        """
        ddl_candles = """
        CREATE TABLE IF NOT EXISTS candles (
            id BIGSERIAL PRIMARY KEY,
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts BIGINT NOT NULL, -- unix ms
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION,
            UNIQUE (asset, interval, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_candles_asset_interval_ts ON candles (asset, interval, ts);
        """
        # Opcional: tabla para indicadores/score (puedes extender)
        ddl_indicators = """
        CREATE TABLE IF NOT EXISTS indicators (
            id BIGSERIAL PRIMARY KEY,
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts BIGINT NOT NULL,
            ema9 DOUBLE PRECISION,
            ema40 DOUBLE PRECISION,
            atr DOUBLE PRECISION,
            macd DOUBLE PRECISION,
            macd_signal DOUBLE PRECISION,
            rsi DOUBLE PRECISION,
            support DOUBLE PRECISION,
            resistance DOUBLE PRECISION,
            fibonacci_levels JSONB,
            UNIQUE (asset, interval, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_indicators_asset_interval_ts ON indicators (asset, interval, ts);
        """
        ddl_scores = """
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
            created_at BIGINT NOT NULL,
            UNIQUE (asset, interval, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_scores_asset_time ON scores (asset, interval, created_at);
        """

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                logger.info("Creando tablas (si no existen)...")
                cur.execute(ddl_candles)
                cur.execute(ddl_indicators)
                cur.execute(ddl_scores)
                logger.info("DDL ejecutado correctamente.")

    # ---------- Upsert / Insert batch ----------
    def save_candles(self, df: pd.DataFrame, asset: str, interval: str, meta: Optional[Dict[str, Any]] = None):
        """
        Guarda DataFrame de velas (columns: ts, open, high, low, close, volume)
        Espera ts como pd.Timestamp (timezone-aware) o como entero ms; convierte a int ms.
        Hace upsert por (asset, interval, ts).
        """
        if df is None or df.empty:
            logger.debug("save_candles: df vacío para %s %s, no hay nada que guardar", asset, interval)
            return 0

        # Normalizar columnas y extraer datos
        df = df.copy()
        if "ts" not in df.columns:
            raise ValueError("DataFrame debe contener columna 'ts' (pd.Timestamp o ms int)")
        # Convertir ts a ms int
        if pd.api.types.is_datetime64_any_dtype(df["ts"]):
            df["ts_ms"] = (df["ts"].astype("int64") // 1_000_000).astype("int64")
        else:
            # si ya son ints asumimos ms
            df["ts_ms"] = df["ts"].astype("int64")

        # seleccionar y ordenar (evita duplicados)
        df = df[["ts_ms", "open", "high", "low", "close", "volume"]].drop_duplicates(subset=["ts_ms"])
        df = df.sort_values("ts_ms")

        rows = []
        for _, r in df.iterrows():
            rows.append((asset, interval, int(r["ts_ms"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]) if not pd.isna(r["volume"]) else None))

        total = 0
        # Ejecutar en batches para no exceder memoria/params
        chunks = [rows[i:i + self._upsert_batch_size] for i in range(0, len(rows), self._upsert_batch_size)]

        upsert_sql = """
        INSERT INTO candles (asset, interval, ts, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (asset, interval, ts) DO UPDATE
          SET open = EXCLUDED.open,
              high = EXCLUDED.high,
              low = EXCLUDED.low,
              close = EXCLUDED.close,
              volume = EXCLUDED.volume;
        """

        for idx, chunk in enumerate(chunks):
            attempts = 0
            while True:
                try:
                    with self.get_conn() as conn:
                        with conn.cursor() as cur:
                            psycopg2.extras.execute_values(cur, upsert_sql, chunk, template=None, page_size=100)
                            total += len(chunk)
                    break
                except Exception as e:
                    attempts += 1
                    logger.exception("Error upserting batch %d/%d for %s %s: %s", idx+1, len(chunks), asset, interval, e)
                    if attempts >= self._max_retry:
                        logger.error("Max retries alcanzado para batch %d; saltando", idx+1)
                        break
                    sleep_for = 2 ** attempts
                    logger.info("Reintentando batch %d en %.1fs...", idx+1, sleep_for)
                    time.sleep(sleep_for)
        logger.info("save_candles completado %d filas para %s %s", total, asset, interval)
        return total

    # ---------- Queries ----------
    def get_ohlcv(self, asset: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Recupera velas para asset+interval entre start_ms y end_ms (ambos inclusive).
        Devuelve DataFrame con columnas ts (pd.Timestamp UTC), open, high, low, close, volume ordenado por ts asc.
        """
        q = "SELECT ts, open, high, low, close, volume FROM candles WHERE asset = %s AND interval = %s"
        params: List[Any] = [asset, interval]
        if start_ms is not None:
            q += " AND ts >= %s"
            params.append(int(start_ms))
        if end_ms is not None:
            q += " AND ts <= %s"
            params.append(int(end_ms))
        q += " ORDER BY ts ASC"
        if limit:
            q += " LIMIT %s"
            params.append(int(limit))

        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(q, tuple(params))
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
                df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
                # convertir ts ms -> pd.Timestamp UTC
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                df = df[["ts", "open", "high", "low", "close", "volume"]]
                return df

    def get_last_ts(self, asset: str, interval: str) -> Optional[int]:
        """
        Devuelve el timestamp ms más reciente guardado para asset+interval, o None si no hay datos.
        """
        q = "SELECT ts FROM candles WHERE asset = %s AND interval = %s ORDER BY ts DESC LIMIT 1"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(q, (asset, interval))
                r = cur.fetchone()
                return int(r[0]) if r else None

    # ---------- Purge / Retention ----------
    def purge_old_data(self, asset: Optional[str] = None, interval: Optional[str] = None, keep_last_n: Optional[int] = None, before_ts_ms: Optional[int] = None):
        """
        Purga filas antiguas según parámetros:
        - Si before_ts_ms dado: elimina todas con ts < before_ts_ms (puede usarse para retención por días).
        - Si keep_last_n dado: mantiene las últimas N velas por asset+interval y borra el resto.
        Las combinaciones de filtros se aplican conjuntamente (AND).
        """
        conditions = []
        params: List[Any] = []
        if asset:
            conditions.append("asset = %s"); params.append(asset)
        if interval:
            conditions.append("interval = %s"); params.append(interval)
        where_clause = (" AND ".join(conditions) + " AND ") if conditions else ""
        if before_ts_ms is not None:
            # elimina por timestamp
            q = f"DELETE FROM candles WHERE {where_clause} ts < %s"
            params2 = params + [int(before_ts_ms)]
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(q, tuple(params2))
                    deleted = cur.rowcount
                    logger.info("purge_old_data: borradas %d filas antes de %d", deleted, before_ts_ms)
                    return deleted
        elif keep_last_n is not None:
            # Mantener últimas N por asset+interval. Implementación: borrar donde ts < (select min ts from last N)
            # Requiere recorrer por cada combinación asset/interval si asset/interval no son None
            assets_intervals = []
            if asset and interval:
                assets_intervals = [(asset, interval)]
            else:
                # obtener combinaciones existentes
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        if asset and not interval:
                            cur.execute("SELECT DISTINCT interval FROM candles WHERE asset = %s", (asset,))
                            intervals = [r[0] for r in cur.fetchall()]
                            assets_intervals = [(asset, it) for it in intervals]
                        elif interval and not asset:
                            cur.execute("SELECT DISTINCT asset FROM candles WHERE interval = %s", (interval,))
                            assets = [r[0] for r in cur.fetchall()]
                            assets_intervals = [(a, interval) for a in assets]
                        else:
                            cur.execute("SELECT DISTINCT asset, interval FROM candles")
                            assets_intervals = [(r[0], r[1]) for r in cur.fetchall()]

            total_deleted = 0
            for a, it in assets_intervals:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            DELETE FROM candles
                            WHERE ctid IN (
                              SELECT ctid FROM (
                                SELECT ctid, ts, ROW_NUMBER() OVER (ORDER BY ts DESC) rn
                                FROM candles
                                WHERE asset = %s AND interval = %s
                              ) x WHERE x.rn > %s
                            )
                        """, (a, it, int(keep_last_n)))
                        deleted = cur.rowcount
                        total_deleted += deleted
            logger.info("purge_old_data: total borradas %d filas por keep_last_n=%s", total_deleted, keep_last_n)
            return total_deleted
        else:
            logger.warning("purge_old_data llamado sin parametros before_ts_ms ni keep_last_n -> no se hace nada")
            return 0

    # ---------- Utility ----------
    def make_save_callback(self):
        """
        Devuelve una función con la firma (df, asset, interval, meta) que llama a self.save_candles.
        Útil para pasar directamente como callback al fetcher.
        """
        def _cb(df, asset, interval, meta=None):
            try:
                return self.save_candles(df, asset, interval, meta=meta)
            except Exception:
                logger.exception("save_callback interno falló para %s %s", asset, interval)
                raise
        return _cb

    def close(self):
        """
        Cierra el pool y libera conexiones.
        """
        if self._pool:
            try:
                self._pool.closeall()
                logger.info("Pool cerrado")
            except Exception:
                logger.exception("Error cerrando pool")
            finally:
                self._pool = None


# ---------- Convenience top-level helper ----------
def make_storage_from_env() -> PostgresStorage:
    """
    Fabrica PostgresStorage leyendo variables de entorno.
    """
    return PostgresStorage(
        host=_env("POSTGRES_HOST", "localhost"),
        port=int(_env("POSTGRES_PORT", "5432")),
        dbname=_env("POSTGRES_DB", "trading"),
        user=_env("POSTGRES_USER", "postgres"),
        password=_env("POSTGRES_PASSWORD", ""),
        minconn=int(_env("POSTGRES_POOL_MIN", str(DEFAULT_POOL_MIN))),
        maxconn=int(_env("POSTGRES_POOL_MAX", str(DEFAULT_POOL_MAX))),
    )


# If executed as script: simple smoke test (no secretos en logs)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    storage = make_storage_from_env()
    storage.init_db()

    # ejemplo de inserción pequeña
    import pandas as pd
    now = int(time.time() * 1000)
    df = pd.DataFrame([{
        "ts": pd.to_datetime(now, unit="ms", utc=True),
        "open": 100.0,
        "high": 101.0,
        "low": 99.5,
        "close": 100.8,
        "volume": 123.4,
    }])
    rows = storage.save_candles(df, "TESTASSET", "1m")
    logger.info("Inserted rows: %s", rows)
    last = storage.get_last_ts("TESTASSET", "1m")
    logger.info("Last ts: %s", last)
    df2 = storage.get_ohlcv("TESTASSET", "1m")
    logger.info("Fetched df rows: %d", len(df2))
    storage.purge_old_data(asset="TESTASSET", interval="1m", keep_last_n=0)
    storage.close()
