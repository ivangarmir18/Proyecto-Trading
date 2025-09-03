"""
dashboard/utils.py
------------------
Utilities optimizados para Postgres (Supabase) orientados al dashboard.

Objetivos:
 - Usar exclusivamente Postgres cuando DATABASE_URL esté presente
 - Conexiones seguras y parametrizadas
 - Retornar pandas.DataFrame listos para el front-end (streamlit)
 - TTL cache en memoria para consultas frecuentes (configurable)
 - Helpers para comprobar/escalar esquema (migraciones simples)
 - Logging claro y manejo de errores informativo

Notas:
 - Usa psycopg2 si está disponible; si no, intenta reutilizar PostgresStorage
   (core.storage_postgres.PostgresStorage) cuando exista.
 - Todas las funciones aceptan timestamps como strings ISO / ints; se normalizan
   internamente a pandas.Timestamp.

"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional, List, Dict, Any, Tuple

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas es requerido para dashboard.utils") from e

# Prefer psycopg2 for raw SQL access
try:
    import psycopg2
    import psycopg2.extras as pex
    PSYCOPG2_AVAILABLE = True
except Exception:
    PSYCOPG2_AVAILABLE = False

# Fallback: if the project exposes PostgresStorage, we'll use it
_POSTGRES_STORAGE_AVAILABLE = False
PostgresStorage = None
try:
    from core.storage_postgres import PostgresStorage as _PS
except Exception:
    try:
        # intenta import relativo si estructura de paquetes distinta
        from .core.storage_postgres import PostgresStorage as _PS  # type: ignore
    except Exception:
        _PS = None


if _PS is not None:
    PostgresStorage = _PS
    _POSTGRES_STORAGE_AVAILABLE = True

# Logger
logger = logging.getLogger("dashboard.utils")
logger.setLevel(os.getenv("DASHBOARD_UTILS_LOG_LEVEL", "INFO"))
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# Default cache TTL in seconds (configurable via env)
DEFAULT_TTL = int(os.getenv("DASHBOARD_CACHE_TTL", "10"))

# Helper: small TTL cache for DataFrames / results
class TTLCache:
    """Cache key->(timestamp, value) with TTL in seconds."""

    def __init__(self, ttl: int = DEFAULT_TTL):
        self.ttl = ttl
        self._store: Dict[Any, Tuple[float, Any]] = {}

    def get(self, key: Any):
        entry = self._store.get(key)
        if not entry:
            return None
        ts, value = entry
        if (time.time() - ts) > self.ttl:
            # expired
            try:
                del self._store[key]
            except KeyError:
                pass
            return None
        return value

    def set(self, key: Any, value: Any):
        self._store[key] = (time.time(), value)

    def clear(self):
        self._store.clear()

# module-level cache
_cache = TTLCache()


# Connection helper class using psycopg2
class PostgresDB:
    """Lightweight Postgres helper for SELECT queries returning pandas.DataFrame.

    Usage:
        db = PostgresDB()  # lee DATABASE_URL
        df = db.fetch_df(sql, params=(...))

    Elimina la dependencia de una implementación concreta del proyecto y permite
    inspeccionar / migrar esquema.
    """

    def __init__(self, dsn: Optional[str] = None, connect_timeout: int = 5):
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError("DATABASE_URL no definido en el entorno. No puedo conectar a Postgres.")
        self.connect_timeout = connect_timeout

    def _connect(self):
        if PSYCOPG2_AVAILABLE:
            # usa RealDictCursor para facilitar conversion a df
            conn = psycopg2.connect(self.dsn, connect_timeout=self.connect_timeout)
            return conn
        else:
            # si no hay psycopg2, intenta usar PostgresStorage si está disponible
            if _POSTGRES_STORAGE_AVAILABLE:
                raise RuntimeError("psycopg2 no está instalado en el entorno; usa PostgresStorage wrapper en su lugar.")
            raise RuntimeError("psycopg2 no disponible y PostgresStorage no encontrado.")

    def fetch_df(self, sql: str, params: Optional[tuple] = None, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """Ejecuta SQL y devuelve pandas.DataFrame. Parametrizado (seguro).

        - sql: query con placeholders %s
        - params: tuple de parámetros
        - parse_dates: lista de columnas que deben convertirse a datetime
        """
        logger.debug("fetch_df SQL: %s | params=%s", sql, params)
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=pex.RealDictCursor)
            cur.execute(sql, params or ())
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            if df.empty:
                # asegurar columnas vacías conocidas no fallen en front
                return df
            # Normalizar timestamps si existen
            if parse_dates:
                for c in parse_dates:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], utc=True)
            else:
                # si existe una columna ts -> parséala por defecto
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], utc=True)
            return df
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()

    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        """Ejecuta una sentencia no-query (DDL/ALTER/INSERT/UPDATE)."""
        logger.debug("execute SQL: %s | params=%s", sql, params)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params or ())
            conn.commit()
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()


# High-level helpers que el dashboard necesitará

def _normalize_ts(value: Optional[Any]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        # si es float/int -> timestamp unix
        try:
            return pd.to_datetime(int(value), unit="s", utc=True)
        except Exception:
            raise ValueError(f"No se pudo convertir a timestamp: {value}")


def ensure_db_connection() -> bool:
    """Chequea que se puede conectar a la DB. Retorna True si OK, False si no."""
    try:
        db = PostgresDB()
        # pequeña query rápida
        db.fetch_df("SELECT 1 as ok LIMIT 1")
        logger.info("Conexión a Postgres OK")
        return True
    except Exception as e:
        logger.error("No se pudo conectar a Postgres: %s", e)
        return False


def list_assets() -> List[str]:
    """Lista assets únicos en la tabla scores.

    Resultado cacheado (TTL configurable).
    """
    cache_key = ("list_assets",)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    db = PostgresDB()
    sql = "SELECT DISTINCT asset FROM public.scores ORDER BY asset"
    try:
        df = db.fetch_df(sql)
        assets = df["asset"].astype(str).tolist() if not df.empty else []
        _cache.set(cache_key, assets)
        return assets
    except Exception as e:
        logger.exception("Error listando assets: %s", e)
        return []


def get_scores_df(asset: str, interval: str, start_ts: Optional[Any] = None, end_ts: Optional[Any] = None, limit: Optional[int] = None, use_cache: bool = True) -> pd.DataFrame:
    """Obtiene un DataFrame con filas de scores para el asset+interval en el rango [start_ts, end_ts].

    - Asset/interval son exactos (no LIKE)
    - start_ts/end_ts pueden ser datetimes, iso-strings o unix timestamps
    - El resultado se cachea por defecto (TTL)

    Retorna DataFrame con columna ts parseada como datetime UTC.
    """
    if not asset or not interval:
        raise ValueError("asset e interval son obligatorios")

    s_ts = _normalize_ts(start_ts)
    e_ts = _normalize_ts(end_ts)

    cache_key = ("scores_df", asset, interval, str(s_ts), str(e_ts), limit)
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit scores_df %s %s", asset, interval)
            return cached

    # Construir query parametrizada
    sql = ["SELECT ts, score, range_min, range_max, stop, target, p_ml, signal_quality, ai_confidence, signal FROM public.scores WHERE asset = %s AND interval = %s"]
    params: List[Any] = [asset, interval]

    if s_ts is not None:
        sql.append("AND ts >= %s")
        params.append(s_ts)
    if e_ts is not None:
        sql.append("AND ts <= %s")
        params.append(e_ts)

    sql.append("ORDER BY ts ASC")
    if limit is not None and isinstance(limit, int) and limit > 0:
        sql.append("LIMIT %s")
        params.append(limit)

    full_sql = " ".join(sql)

    db = PostgresDB()
    try:
        df = db.fetch_df(full_sql, tuple(params), parse_dates=["ts"])
        # Normalizaciones prácticas para el dashboard (evitar NaN problemáticos)
        if not df.empty:
            # Asegurar columnas esperadas existan
            expected_cols = ["ts", "score", "range_min", "range_max", "stop", "target", "p_ml", "signal_quality", "ai_confidence", "signal"]
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            # ordenar y reset index
            df = df.sort_values("ts").reset_index(drop=True)
        if use_cache:
            _cache.set(cache_key, df)
        return df
    except Exception:
        logger.exception("Error al cargar scores para %s %s", asset, interval)
        # devolver DataFrame vacío con esquema mínimo para evitar errores de front
        cols = ["ts", "score", "range_min", "range_max", "stop", "target", "p_ml", "signal_quality", "ai_confidence", "signal"]
        return pd.DataFrame(columns=cols)


def get_latest_score(asset: str, interval: str) -> Optional[Dict[str, Any]]:
    """Devuelve la fila más reciente (como dict) para asset/interval.

    Útil para badges / valores rápidos en el dashboard.
    """
    sql = "SELECT * FROM public.scores WHERE asset = %s AND interval = %s ORDER BY ts DESC LIMIT 1"
    db = PostgresDB()
    try:
        df = db.fetch_df(sql, (asset, interval))
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        # convertir ts a ISO string con timezone
        if "ts" in row and pd.notna(row["ts"]):
            row["ts"] = pd.to_datetime(row["ts"]).isoformat()
        return row
    except Exception:
        logger.exception("Error obteniendo latest score para %s %s", asset, interval)
        return None


def ensure_signal_columns() -> None:
    """Asegura que las columnas signal_quality y ai_confidence existen (migración idempotente).

    Ejecuta ALTER TABLE IF NOT EXISTS ... (Postgres soporta IF NOT EXISTS para columnas
    a partir de versiones modernas; si no, se hace chequeo previo).
    """
    db = PostgresDB()
    # Postgres soporta 'ADD COLUMN IF NOT EXISTS' desde 9.6+ (y versiones modernas de supabase)
    sql = (
        "ALTER TABLE public.scores ADD COLUMN IF NOT EXISTS signal_quality double precision,"
        " ADD COLUMN IF NOT EXISTS ai_confidence double precision;"
    )
    try:
        db.execute(sql)
        logger.info("Aseguradas columnas signal_quality y ai_confidence (si faltaban)")
    except Exception:
        logger.exception("Fallo al asegurar columnas signal_quality/ai_confidence")
        raise


def inspect_scores_schema() -> Dict[str, str]:
    """Devuelve un dict column_name->data_type de la tabla public.scores. Útil para debug.

    No cacheado deliberadamente (queremos siempre info fresca si se acaba de migrar).
    """
    sql = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'scores' AND table_schema = 'public'"
    db = PostgresDB()
    try:
        df = db.fetch_df(sql)
        return {r["column_name"]: r["data_type"] for _, r in df.iterrows()} if not df.empty else {}
    except Exception:
        logger.exception("Error inspeccionando esquema de scores")
        return {}


# Helper utility para invalidar cache (por ejemplo tras insertar filas)
def invalidate_cache() -> None:
    _cache.clear()


# Exportar interfaz pública limpia para el dashboard
__all__ = [
    "ensure_db_connection",
    "ensure_signal_columns",
    "inspect_scores_schema",
    "list_assets",
    "get_scores_df",
    "get_latest_score",
    "invalidate_cache",
]

# Fin del fichero
