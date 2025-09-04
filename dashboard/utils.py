"""
dashboard/utils.py

Funciones utilitarias pensadas para Streamlit / UI:
 - conexión a storage Postgres (PostgresStorage)
 - listado paginado de ficheros de backfill (local o Supabase)
 - listado paginado de estados de backfill (tabla backfill_status)
 - wrappers safe para obtener assets, candles, indicadores y scores
 - trigger_backfill: iniciar backfill en background (thread) y obtener handle
 - upload_backfill_file_to_supabase: helper para subir archivos de backfill

IMPORTANTE:
 - Si defines SUPABASE_URL, SUPABASE_KEY y SUPABASE_BUCKET en env, usará Supabase Storage.
 - Si no, usa archivos locales en data/backfill_sources.
 - La paginación usa 'cursor' como offset entero (0, page_size, 2*page_size ...).
"""

import os
import logging
import threading
import math
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd

# optional supabase client
try:
    from supabase import create_client  # type: ignore
    HAS_SUPABASE = True
except Exception:
    create_client = None  # type: ignore
    HAS_SUPABASE = False

from core.storage_postgres import PostgresStorage
import core.init_and_backfill as init_and_backfill
from core.orchestrator import Orchestrator

logger = logging.getLogger("dashboard.utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)

# singleton storage
_storage: Optional[PostgresStorage] = None


def storage() -> PostgresStorage:
    global _storage
    if _storage is None:
        _storage = PostgresStorage()
    return _storage


# ----------------------
# Supabase client factory
# ----------------------
_supabase_client = None


def supabase_client():
    """
    Return a supabase client if env vars present; else None.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key or not HAS_SUPABASE:
        logger.debug("Supabase not configured or client not installed")
        return None
    _supabase_client = create_client(url, key)
    return _supabase_client


def supabase_bucket_name() -> Optional[str]:
    return os.getenv("SUPABASE_BUCKET")


# ----------------------
# Assets / Candles / Indicators / Scores
# ----------------------
def get_assets() -> List[str]:
    """
    Return list of assets: prefer DB candles distinct, then watchlist, then config CSVs.
    """
    try:
        return storage().list_assets()
    except Exception:
        logger.exception("get_assets failed")
        return []


def get_intervals_for_asset(asset: str) -> List[str]:
    try:
        return storage().list_intervals_for_asset(asset)
    except Exception:
        logger.exception("get_intervals_for_asset failed for %s", asset)
        return []


def get_latest_candles(asset: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Load latest `limit` candles for asset/interval, return ascending by ts.
    """
    try:
        df = storage().load_candles(asset, interval, limit=limit, ascending=False)
        if df.empty:
            return df
        return df.sort_values("ts").reset_index(drop=True)
    except Exception:
        logger.exception("get_latest_candles error")
        return pd.DataFrame(columns=["ts", "timestamp", "open", "high", "low", "close", "volume", "asset", "interval"])


def get_indicators_for_ui(asset: str, interval: str) -> Dict[str, Any]:
    """
    Lightweight wrapper that computes/returns indicators using Orchestrator.
    """
    try:
        orch = Orchestrator(storage())
        return orch.compute_indicators_for(asset, interval)
    except Exception:
        logger.exception("get_indicators_for_ui failed")
        return {}


def get_scores_for_ui(asset: str, interval: str, limit: int = 100) -> pd.DataFrame:
    try:
        return storage().load_scores(asset=asset, interval=interval, limit=limit)
    except Exception:
        logger.exception("get_scores_for_ui failed")
        return pd.DataFrame(columns=["ts", "asset", "interval", "score", "created_at"])


# ----------------------
# Backfill files listing (paginated) - supports local and Supabase
# ----------------------
def _list_local_backfill_files(folder: str = "data/backfill_sources") -> List[Dict[str, Any]]:
    """
    Return list of files metadata sorted by mtime descending.
    Each item: {'name','path','size','updated_at'}
    """
    out = []
    if not os.path.isdir(folder):
        return out
    entries = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            full = os.path.join(root, fn)
            try:
                st = os.stat(full)
                entries.append((st.st_mtime, full, st.st_size))
            except Exception:
                continue
    # sort newest first
    entries.sort(reverse=True)
    for mtime, full, size in entries:
        out.append({"name": os.path.basename(full), "path": full, "size": size, "updated_at": float(mtime)})
    return out


def _list_supabase_backfill_files(bucket: str, prefix: str = "") -> List[Dict[str, Any]]:
    """
    List files in supabase storage bucket (flat listing). 
    Note: supabase-py storage.list may return dicts; adapt accordingly.
    """
    client = supabase_client()
    if not client:
        return []
    try:
        # supabase storage API: client.storage.from_(bucket).list(path, limit, offset)
        # We'll request large list and rely on server-side pagination if needed.
        items = client.storage.from_(bucket).list(prefix, limit=10000)  # try large limit
        out = []
        for it in items:
            # item likely has 'name','updated_at','size' depending on supabase version
            name = it.get("name") if isinstance(it, dict) else getattr(it, "name", None)
            size = it.get("size") if isinstance(it, dict) else getattr(it, "size", None)
            updated_at = it.get("updated_at") if isinstance(it, dict) else getattr(it, "updated_at", None)
            out.append({"name": name, "path": name, "size": size, "updated_at": updated_at})
        # sort newest first by updated_at if present
        out.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
        return out
    except Exception:
        logger.exception("Supabase listing failed")
        return []


def list_backfill_files(page_size: int = 50, cursor: Optional[int] = 0, source: str = "auto", prefix: str = "") -> Dict[str, Any]:
    """
    Paginated listing of backfill source files.
    - page_size: number of items per page
    - cursor: offset integer (0 = start). Returned next_cursor = None when no more pages.
    - source: 'auto'|'local'|'supabase'
    Returns: {'files': [...], 'next_cursor': int|None, 'total': int}
    """
    # decide source
    if source == "auto":
        cli = supabase_client()
        if cli and supabase_bucket_name():
            source = "supabase"
        else:
            source = "local"
    items = []
    if source == "supabase":
        bucket = supabase_bucket_name()
        if not bucket:
            source = "local"
        else:
            items = _list_supabase_backfill_files(bucket, prefix=prefix)
    if source == "local":
        items = _list_local_backfill_files()
    total = len(items)
    # handle cursor slicing
    try:
        start = int(cursor or 0)
    except Exception:
        start = 0
    end = start + int(page_size)
    page = items[start:end]
    next_cursor = end if end < total else None
    return {"files": page, "next_cursor": next_cursor, "total": total, "source": source}


# ----------------------
# Backfill status (table pagination)
# ----------------------
def list_backfill_status(page_size: int = 50, cursor: Optional[int] = 0) -> Dict[str, Any]:
    """
    Paginate over backfill_status table rows. Returns list of dicts and next_cursor (offset).
    """
    conn = None
    res = {"rows": [], "next_cursor": None, "total": 0}
    try:
        s = storage()
        # do count and select with offset-limit
        conn = s._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM backfill_status;")
        total = cur.fetchone()[0] or 0
        res["total"] = int(total)
        # select rows ordered by updated_at desc
        start = int(cursor or 0)
        cur.execute("SELECT asset, interval, last_ts, updated_at FROM backfill_status ORDER BY updated_at DESC OFFSET %s LIMIT %s;", (start, page_size))
        rows = cur.fetchall()
        cur.close()
        out = []
        for r in rows:
            out.append({"asset": r[0], "interval": r[1], "last_ts": int(r[2]) if r[2] is not None else None, "updated_at": r[3]})
        res["rows"] = out
        res["next_cursor"] = start + len(out) if (start + len(out) < total) else None
    except Exception:
        logger.exception("list_backfill_status failed")
    finally:
        if conn:
            try:
                storage()._put_conn(conn)
            except Exception:
                pass
    return res


# ----------------------
# Trigger backfill (backgroundable)
# ----------------------
class BackgroundTaskHandle:
    def __init__(self, thread: threading.Thread):
        self.thread = thread

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout: Optional[float] = None):
        return self.thread.join(timeout)


def trigger_backfill_background(concurrency: int = 4, chunk_hours: int = 168, local_folder: str = "data/backfill_sources",
                                default_days: int = 365, assets_override: Optional[List[Dict[str, str]]] = None) -> BackgroundTaskHandle:
    """
    Start backfill in a background thread and return a handle.
    Useful in Streamlit to avoid blocking main thread.
    """
    def _runner():
        try:
            init_and_backfill.run_backfill(concurrency=concurrency, chunk_hours=chunk_hours, local_folder=local_folder, default_days=default_days, assets_override=assets_override)
        except Exception:
            logger.exception("Background backfill crashed")

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return BackgroundTaskHandle(t)


# ----------------------
# Upload utility (optional)
# ----------------------
def upload_backfill_file_to_supabase(local_path: str, dest_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload a local file to Supabase Storage under SUPABASE_BUCKET.
    Returns status dict; raises on failure.
    """
    client = supabase_client()
    bucket = supabase_bucket_name()
    if not client or not bucket:
        raise RuntimeError("Supabase not configured")
    dest = dest_path or os.path.basename(local_path)
    try:
        with open(local_path, "rb") as f:
            res = client.storage.from_(bucket).upload(dest, f)
        logger.info("Uploaded %s to supabase bucket %s as %s", local_path, bucket, dest)
        return {"ok": True, "path": dest}
    except Exception:
        logger.exception("Supabase upload failed")
        raise


# ----------------------
# Misc helpers
# ----------------------
def get_backfill_progress_summary() -> Dict[str, Any]:
    """
    Quick summary: number of assets in backfill_status, total candles count.
    """
    s = storage()
    try:
        conn = s._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM backfill_status;")
        assets = cur.fetchone()[0] or 0
        cur.execute("SELECT count(*) FROM candles;")
        candles = cur.fetchone()[0] or 0
        cur.close()
        s._put_conn(conn)
        return {"backfill_assets": int(assets), "candles_total": int(candles)}
    except Exception:
        logger.exception("get_backfill_progress_summary failed")
        try:
            s._put_conn(conn)
        except Exception:
            pass
        return {"backfill_assets": 0, "candles_total": 0}
