"""
core/init_and_backfill.py

Backfill robusto y resumable para Proyecto-Trading.

Características principales:
- Lee lista de assets+intervals desde data/config/*.csv o desde la tabla watchlist.
- Para cada asset:
  - Consulta el estado de backfill en la DB (tabla backfill_status).
  - Usa archivos locales en data/backfill_sources/ si existen.
  - Si no hay archivos locales, intenta llamar a core.fetcher.fetch_historical (si existe).
  - Divide rangos largos en ventanas manejables y hace fetch/insert por trozos.
  - Retries con backoff, control de concurrencia, registro por asset y resumen final.
- Proporciona hooks (callback) opcionales para progresos por asset (útil para UI).

Uso (CLI):
  python -m core.init_and_backfill --concurrency 6 --chunk-hours 168 --local-folder data/backfill_sources

Devuelve un resumen con filas procesadas por asset.
"""
import os
import glob
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, Dict, Any, List, Tuple

import pandas as pd

from core.storage_postgres import PostgresStorage

logger = logging.getLogger("init_and_backfill")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)

# Try to import a project fetcher if present
try:
    from core.fetcher import fetch_historical  # type: ignore
    HAS_FETCHER = True
    logger.info("Found core.fetcher.fetch_historical — remote fetch enabled")
except Exception:
    fetch_historical = None  # type: ignore
    HAS_FETCHER = False
    logger.info("No core.fetcher found — relying on local CSVs or remote fetcher unavailable")


# ---------- Helpers to read configs ----------
def read_config_assets(config_glob: str = "data/config/*.csv") -> List[Dict[str, str]]:
    """
    Read CSV files under data/config to get list of dicts {'asset':..., 'interval':...}.
    Heurística tolerante a nombres de columna (asset/symbol/ticker ; interval/timeframe).
    """
    assets: List[Dict[str, str]] = []
    paths = glob.glob(config_glob)
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            logger.exception("Failed to read config CSV %s", p)
            continue
        # Try explicit columns
        a_col = None
        i_col = None
        for c in df.columns:
            cl = c.lower()
            if cl in ("asset", "symbol", "ticker"):
                a_col = c
            if cl in ("interval", "timeframe", "tf"):
                i_col = c
        if a_col:
            for _, r in df.iterrows():
                asset = str(r[a_col])
                interval = str(r[i_col]) if i_col and pd.notna(r.get(i_col)) else "1m"
                assets.append({"asset": asset, "interval": interval})
    # Deduplicate and return
    seen = set()
    uniq = []
    for x in assets:
        k = (x["asset"], x["interval"])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


# ---------- local file helpers ----------
def find_local_backfill_files(asset: str, interval: str, folder: str = "data/backfill_sources") -> List[str]:
    """
    Busca ficheros en folder que parezcan corresponder a asset+interval.
    """
    patt1 = os.path.join(folder, f"{asset}*{interval}*.csv")
    files = glob.glob(patt1)
    if not files:
        patt2 = os.path.join(folder, f"{asset}*.csv")
        files = glob.glob(patt2)
    return sorted(files)


def load_csv_to_df(path: str, asset: Optional[str] = None, interval: Optional[str] = None) -> pd.DataFrame:
    """
    Carga CSV, normaliza columnas (ts/timestamp/open/high/low/close/volume/asset/interval).
    Convierte timestamp->ts (segundos).
    Devuelve DataFrame ordenado por ts.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed reading backfill CSV %s", path)
        return pd.DataFrame(columns=["ts", "timestamp", "open", "high", "low", "close", "volume", "asset", "interval"])
    # normalize
    colmap = {}
    for ec in df.columns:
        lc = ec.lower()
        if lc in ("ts", "timestamp", "open", "high", "low", "close", "volume", "asset", "symbol", "interval", "timeframe", "volume"):
            if lc == "symbol":
                colmap[ec] = "asset"
            elif lc == "timeframe":
                colmap[ec] = "interval"
            else:
                colmap[ec] = lc
    df = df.rename(columns=colmap)
    if "timestamp" in df.columns and "ts" not in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10 ** 9
    if asset and "asset" not in df.columns:
        df["asset"] = asset
    if interval and "interval" not in df.columns:
        df["interval"] = interval
    # ensure cols
    for c in ["ts", "timestamp", "open", "high", "low", "close", "volume", "asset", "interval"]:
        if c not in df.columns:
            df[c] = None
    # drop dupes and sort
    df = df.drop_duplicates(subset=["ts", "asset", "interval"], keep="last").sort_values("ts").reset_index(drop=True)
    return df


# ---------- backfill core ----------
def _chunk_range(start_ts: int, end_ts: int, chunk_seconds: int) -> List[Tuple[int, int]]:
    """
    Divide [start_ts, end_ts] en ventanas de chunk_seconds (inclusive).
    """
    ranges = []
    cur = start_ts
    while cur <= end_ts:
        end = min(end_ts, cur + chunk_seconds - 1)
        ranges.append((cur, end))
        cur = end + 1
    return ranges


def backfill_asset(storage: PostgresStorage,
                   asset: str,
                   interval: str,
                   *,
                   local_folder: str = "data/backfill_sources",
                   chunk_hours: int = 168,
                   retries: int = 3,
                   retry_backoff: float = 2.0,
                   default_days: int = 365,
                   progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """
    Backfill resumable para un único asset/interval.
    - Usa estado en DB (backfill_status).
    - Primero intenta cargar CSVs locales.
    - Si no hay archivos locales, intenta fetch_historical si está disponible.
    - chunk_hours controla el tamaño de piezas para fetch remoto.
    - progress_callback(asset, info) se llama tras cada chunk/archivo procesado.
    Retorna un dict resumen: {'asset':..., 'interval':..., 'rows': n, 'errors': [...]}.
    """
    logger.info("Backfill start for %s %s", asset, interval)
    summary = {"asset": asset, "interval": interval, "rows": 0, "errors": []}
    try:
        # determine last ts from backfill_status or candles
        last_ts = storage.get_backfill_status(asset)
        if last_ts is None:
            try:
                last_rad = storage.load_candles(asset, interval, limit=1, ascending=False)
                if not last_rad.empty:
                    last_ts = int(last_rad.iloc[0]["ts"])
            except Exception:
                logger.debug("No candles in DB for %s %s", asset, interval)
        # gather local files
        files = find_local_backfill_files(asset, interval, local_folder)
        if files:
            logger.info("Found %d local backfill file(s) for %s", len(files), asset)
            for fpath in files:
                try:
                    df = load_csv_to_df(fpath, asset=asset, interval=interval)
                    if last_ts is not None:
                        df = df[df["ts"] > last_ts]
                    if df.empty:
                        logger.info("Local file %s has no new rows for %s", fpath, asset)
                        continue
                    n = storage.save_candles(df, asset=asset, interval=interval)
                    summary["rows"] += n
                    last_ts = int(df["ts"].max())
                    storage.update_backfill_status(asset, interval, last_ts)
                    if progress_callback:
                        progress_callback(asset, {"type": "local_file", "file": fpath, "rows": n})
                except Exception:
                    logger.exception("Failed processing local backfill file %s", fpath)
                    summary["errors"].append(f"local:{fpath}")
        else:
            # No local files -> use remote fetcher (if available)
            if not HAS_FETCHER:
                msg = f"No local files and no remote fetcher for {asset}"
                logger.warning(msg)
                summary["errors"].append("no_source")
                return summary
            # determine start_ts for fetching
            if last_ts is None:
                start_ts = int((pd.Timestamp.utcnow() - pd.Timedelta(days=default_days)).timestamp())
            else:
                start_ts = int(last_ts + 1)
            end_ts = int(pd.Timestamp.utcnow().timestamp())
            chunk_seconds = max(60, chunk_hours * 3600)
            ranges = _chunk_range(start_ts, end_ts, chunk_seconds)
            logger.info("Remote fetcher for %s -> %d ranges", asset, len(ranges))
            for (s_ts, e_ts) in ranges:
                attempt = 0
                while attempt <= retries:
                    try:
                        # fetch_historical should return a dataframe with ts/timestamp/open/high/low/close/volume
                        df = fetch_historical(asset, interval, s_ts, e_ts)
                        if df is None or df.empty:
                            logger.info("Fetcher empty for %s %s -> %s-%s", asset, interval, s_ts, e_ts)
                        else:
                            # normalize and save
                            df = load_csv_to_df(df.to_csv(index=False)) if isinstance(df, pd.DataFrame) else load_csv_to_df(df)  # safe normalization
                            # Note: if fetch_historical returns a DataFrame already with correct cols, load_csv_to_df handles string path fallback wrongly.
                            # Better: if df is DataFrame, normalize directly:
                            if isinstance(df, pd.DataFrame):
                                # ensure columns and types
                                # (load_csv_to_df expects a path; so instead apply transformation inline)
                                dfn = df.copy()
                                # unify column names
                                dfn.columns = [c.lower() for c in dfn.columns]
                                if "timestamp" in dfn.columns and "ts" not in dfn.columns:
                                    dfn["ts"] = pd.to_datetime(dfn["timestamp"]).astype("int64") // 10 ** 9
                                for c in ["open","high","low","close","volume","asset","interval"]:
                                    if c not in dfn.columns:
                                        dfn[c] = None
                                dfn["asset"] = asset
                                dfn["interval"] = interval
                                dfn = dfn[["ts","timestamp","open","high","low","close","volume","asset","interval"]]
                                df = dfn.drop_duplicates(subset=["ts","asset","interval"]).sort_values("ts")
                            # apply last_ts filter
                            if last_ts is not None:
                                df = df[df["ts"] > last_ts]
                            if df.empty:
                                logger.info("No new rows in fetched chunk %s-%s for %s", s_ts, e_ts, asset)
                            else:
                                n = storage.save_candles(df, asset=asset, interval=interval)
                                summary["rows"] += n
                                last_ts = int(df["ts"].max())
                                storage.update_backfill_status(asset, interval, last_ts)
                                if progress_callback:
                                    progress_callback(asset, {"type": "remote_chunk", "range": (s_ts, e_ts), "rows": n})
                        break  # success or empty -> next chunk
                    except Exception:
                        logger.exception("Fetcher failed for %s range %s-%s (attempt %d)", asset, s_ts, e_ts, attempt)
                        attempt += 1
                        time.sleep(retry_backoff ** attempt)
                        if attempt > retries:
                            summary["errors"].append(f"remote:{s_ts}-{e_ts}")
    except Exception:
        logger.exception("backfill_asset unexpected failure for %s", asset)
        summary["errors"].append("unexpected")
    logger.info("Backfill finished for %s rows=%s errors=%s", asset, summary["rows"], len(summary["errors"]))
    return summary


def run_backfill(concurrency: int = 4,
                 chunk_hours: int = 168,
                 local_folder: str = "data/backfill_sources",
                 default_days: int = 365,
                 assets_override: Optional[List[Dict[str, str]]] = None,
                 progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """
    Orquesta backfill para todos los assets encontrados en config o watchlist.
    - assets_override: lista [{'asset':.., 'interval':..}] para forzar conjunto.
    - progress_callback se pasa hacia cada worker.
    Devuelve resumen global.
    """
    storage = PostgresStorage()
    storage.init_db()
    if assets_override is None:
        assets = read_config_assets()
        if not assets:
            # try watchlist
            try:
                wl = storage.list_watchlist()
                assets = [{"asset": a, "interval": "1m"} for a in wl]
            except Exception:
                assets = []
    else:
        assets = assets_override

    logger.info("run_backfill: assets count=%d", len(assets))
    results = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {}
        for a in assets:
            asset = a["asset"]
            interval = a.get("interval", "1m")
            fut = ex.submit(backfill_asset,
                             storage,
                             asset,
                             interval,
                             local_folder=local_folder,
                             chunk_hours=chunk_hours,
                             default_days=default_days,
                             progress_callback=progress_callback)
            futures[fut] = (asset, interval)
        for fut in as_completed(futures):
            asset, interval = futures[fut]
            try:
                res = fut.result()
                results[f"{asset}:{interval}"] = res
            except Exception:
                logger.exception("Backfill worker crashed for %s", asset)
                results[f"{asset}:{interval}"] = {"asset": asset, "interval": interval, "rows": 0, "errors": ["worker_crash"]}
    storage.close()
    # summary
    total_rows = sum(r.get("rows", 0) for r in results.values())
    total_errors = sum(len(r.get("errors", [])) for r in results.values())
    return {"assets": len(results), "total_rows": total_rows, "total_errors": total_errors, "per_asset": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--chunk-hours", type=int, default=168)
    p.add_argument("--local-folder", type=str, default="data/backfill_sources")
    p.add_argument("--default-days", type=int, default=365)
    args = p.parse_args()
    summary = run_backfill(concurrency=args.concurrency, chunk_hours=args.chunk_hours, local_folder=args.local_folder, default_days=args.default_days)
    logger.info("Backfill summary: %s", summary)
