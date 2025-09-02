# dashboard/utils.py
"""
Helpers for the dashboard: loading config, DB path, wrappers to core.storage and core.fetch,
and small utilities.
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple, Any

import pandas as pd

# Attempt to import core.storage and core.fetch; fallback to sqlite direct access
try:
    from core import storage as core_storage  # type: ignore
    _HAS_CORE_STORAGE = True
except Exception:
    core_storage = None
    _HAS_CORE_STORAGE = False

try:
    from core import fetch as core_fetch  # type: ignore
    _HAS_CORE_FETCH = True
except Exception:
    core_fetch = None
    _HAS_CORE_FETCH = False


def load_config(path: str = "config.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_db_path(config: dict) -> Optional[str]:
    # priority: env DB_PATH -> config app.db_path -> default data/db/data.db
    envp = os.getenv("DB_PATH")
    if envp:
        return envp
    try:
        app = config.get("app", {})
        dbp = app.get("db_path")
        if dbp:
            return dbp
    except Exception:
        pass
    # default path
    return "data/db/data.db"


# ---- wrappers to load data for dashboard usage ----
def load_scores_df(asset: str, interval: str, db_path: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    if _HAS_CORE_STORAGE:
        return core_storage.load_scores(asset, interval, db_path=db_path, limit=limit)
    # fallback: read SQLite directly
    if not db_path:
        raise RuntimeError("No core.storage and no db_path provided")
    con = sqlite3.connect(db_path)
    q = "SELECT ts, score, range_min, range_max, stop, target, multiplier FROM scores WHERE asset=? AND interval=? ORDER BY ts DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(q, con, params=(asset, interval))
    con.close()
    if not df.empty:
        df['ts'] = pd.to_numeric(df['ts']).astype(int)
    return df


def load_candles_df(asset: str, interval: str, db_path: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    if _HAS_CORE_STORAGE:
        return core_storage.load_candles(asset, interval, db_path=db_path, limit=limit)
    if not db_path:
        raise RuntimeError("No core.storage and no db_path provided")
    con = sqlite3.connect(db_path)
    q = "SELECT ts, open, high, low, close, volume FROM candles WHERE asset=? AND interval=? ORDER BY ts"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(q, con, params=(asset, interval))
    con.close()
    if not df.empty:
        df['ts'] = pd.to_numeric(df['ts']).astype(int)
    return df


# ---- call fetch/backfill helpers if available ----
def call_refresh_watchlist(crypto_list: list, stock_list: list, crypto_interval: str = "5m", stock_resolution: str = "5", save_to_db: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Calls core.fetch.refresh_watchlist if available. Returns (ok, message).
    crypto_list / stock_list can be empty lists -> fetch will treat accordingly.
    """
    if not _HAS_CORE_FETCH:
        return False, "core.fetch no disponible en el entorno"
    try:
        # refresh_watchlist expected to have signature similar to: (crypto_symbols, stock_symbols, crypto_interval, stock_resolution, save_to_db)
        if hasattr(core_fetch, "refresh_watchlist"):
            core_fetch.refresh_watchlist(crypto_list, stock_list, crypto_interval=crypto_interval, stock_resolution=stock_resolution, save_to_db=save_to_db)
            return True, "refresh_watchlist ejecutado"
        else:
            return False, "core.fetch no implementa refresh_watchlist"
    except Exception as e:
        return False, f"Error llamando refresh_watchlist: {e}"


def call_backfill_historical(crypto_interval: str = "1h", stock_interval: str = "1h") -> Tuple[bool, Optional[str]]:
    if not _HAS_CORE_FETCH:
        return False, "core.fetch no disponible"
    try:
        if hasattr(core_fetch, "backfill_historical"):
            core_fetch.backfill_historical(crypto_interval=crypto_interval, stock_interval=stock_interval)
            return True, "backfill_historical ejecutado"
        else:
            return False, "core.fetch no implementa backfill_historical"
    except Exception as e:
        return False, f"Error en backfill_historical: {e}"


# ---- small helpers ----
def ts_to_iso(ts: int) -> str:
    try:
        return pd.to_datetime(int(ts), unit='s').strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(ts)
