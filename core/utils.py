# core/utils.py
"""
Utilities compartidas: carga de config, logger, helpers temporales y wrapper PG.
Reemplaza o pega al final del core/utils.py existente si prefieres conservar otras utilidades.
"""

from __future__ import annotations
import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from typing import Optional

# ---------------------------
# Logging / config helpers
# ---------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Carga config JSON desde path (relative o absolute)."""
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    l = logging.getLogger(name)
    if not l.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        l.addHandler(h)
    l.setLevel(level)
    return l

def now_ms() -> int:
    return int(time.time() * 1000)

def iso_to_ms(iso: str) -> int:
    """
    Convierte ISO 8601 (con o sin TZ) a ms epoch UTC.
    """
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

# ---------------------------
# Postgres pool helper (PG_DB)
# ---------------------------
logger = get_logger("core.utils")

def _env_or(key, default):
    return os.environ.get(key, default)

class PG:
    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 10):
        self._dsn = dsn
        self._minconn = minconn
        self._maxconn = maxconn
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._init_pool()

    @classmethod
    def from_env(cls):
        host = _env_or("PG_HOST", "localhost")
        port = _env_or("PG_PORT", "5432")
        user = _env_or("PG_USER", "postgres")
        password = _env_or("PG_PASSWORD", "")
        dbname = _env_or("PG_DB", "trading")
        minc = int(_env_or("PG_MINCONN", "1"))
        maxc = int(_env_or("PG_MAXCONN", "10"))
        dsn = f"host={host} port={port} user={user} password={password} dbname={dbname}"
        return cls(dsn, minc, maxc)

    def _init_pool(self):
        if self._pool is None:
            logger.info("Inicializando pool Postgres %s-%s", self._minconn, self._maxconn)
            self._pool = psycopg2.pool.ThreadedConnectionPool(self._minconn, self._maxconn, dsn=self._dsn)

    @contextmanager
    def get_conn(self):
        if self._pool is None:
            self._init_pool()
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            self._pool.putconn(conn)

    def transaction(self):
        class _Tx:
            def __init__(self, parent):
                self.parent = parent
                self.conn = None
            def __enter__(self):
                if self.parent._pool is None:
                    self.parent._init_pool()
                self.conn = self.parent._pool.getconn()
                self.conn.autocommit = False
                return self.conn
            def __exit__(self, exc_type, exc, tb):
                try:
                    if exc_type is None:
                        self.conn.commit()
                    else:
                        self.conn.rollback()
                finally:
                    self.conn.autocommit = True
                    self.parent._pool.putconn(self.conn)
        return _Tx(self)

# Exporta instancia global reutilizable
PG_DB = PG.from_env()
