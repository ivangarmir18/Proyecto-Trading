#!/usr/bin/env python3
# init_and_backfill.py
"""
Init DB + Backfill CLI (completo, resumable y concurrente).

Características principales:
- init DB idempotente (--init-db).
- Soporta entrada de assets por: --asset, --csv, --watchlist (DB).
- Soporta --incremental (usa backfill_status.last_ts para reanudar).
- Rango (--start/--end) en ISO o ms; utilidades de parseo robustas.
- Concurrencia configurable (--concurrency) con ThreadPoolExecutor.
- Retries y backoff configurables (--max-retries, --backoff-factor).
- Dry-run mode (no escribe en DB).
- Logs por asset (folder logs/backfill) y summary final.
- Integración con Orchestrator.run_backfill_for (usa storage.save_candles a través del orchestrator).
- Actualiza backfill_status (update_backfill_status) tras cada asset (ok / error).
- Seguro frente a interrupciones; cada asset es independiente y reanudable.

Uso:
    python init_and_backfill.py --watchlist --incremental --concurrency 4
    python init_and_backfill.py --asset BTCUSDT --interval 1h --start 2024-01-01 --end 2024-01-03
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
import time
import math
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Intentar importar Orchestrator y storage factory (fallar con mensaje claro si no existen)
try:
    from core.orchestrator import Orchestrator
    from core.storage_postgres import make_storage_from_env
except Exception as e:
    raise RuntimeError("No se pudieron importar core.orchestrator o core.storage_postgres. Asegúrate de haberlos implementado.") from e

# Logging
LOG_DIR = Path(os.getenv("PROJECT_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
BACKFILL_LOG_DIR = LOG_DIR / "backfill"
BACKFILL_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("init_and_backfill")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
ch = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(fmt)
logger.addHandler(ch)

# Helper: parse time arg
def parse_time_arg(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    # digits -> assume ms
    if s.isdigit():
        return int(s)
    # parse iso using pandas
    try:
        ts = pd.to_datetime(s, utc=True)
        return int(ts.value // 10 ** 6)
    except Exception:
        raise argparse.ArgumentTypeError(f"Formato de tiempo no reconocido: {val}")

def load_assets_from_csv(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV no encontrado: {path}")
    df = pd.read_csv(p)
    if "asset" not in df.columns:
        raise ValueError("CSV debe contener columna 'asset'")
    out = []
    for _, row in df.iterrows():
        asset = str(row["asset"]).strip()
        interval = str(row.get("interval", "")).strip() or None
        out.append({"asset": asset, "interval": interval})
    return out

def build_asset_list(args: argparse.Namespace, storage) -> List[Dict[str, Any]]:
    if args.asset:
        return [{"asset": args.asset.strip(), "interval": args.interval}]
    if args.csv:
        return load_assets_from_csv(args.csv)
    if args.watchlist:
        if storage is None:
            raise RuntimeError("Storage requerido para --watchlist")
        wl = storage.list_watchlist()
        out = []
        for e in wl:
            if isinstance(e, dict):
                asset = e.get("asset")
                meta = e.get("meta") or {}
                interval = meta.get("interval") if isinstance(meta, dict) else None
            else:
                asset = e
                interval = None
            out.append({"asset": asset, "interval": interval or args.interval})
        return out
    raise RuntimeError("No assets specified. Usa --asset, --csv o --watchlist.")

def _asset_logger(asset: str) -> logging.Logger:
    # logger por asset que escribe en logs/backfill/{asset}.log
    name = f"backfill.{asset}"
    l = logging.getLogger(name)
    if not l.handlers:
        fh = logging.FileHandler(BACKFILL_LOG_DIR / f"{asset}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        l.addHandler(fh)
        l.setLevel(logger.level)
    return l

def _safe_update_backfill_status(storage, asset: str, interval: str, last_ts: Optional[int]):
    try:
        if hasattr(storage, "update_backfill_status"):
            storage.update_backfill_status(asset, interval=interval, last_ts=last_ts)
    except Exception:
        logger.exception("update_backfill_status fallo para %s", asset)

def _run_backfill_single(orch: Orchestrator, storage, asset: str, interval: str,
                         start_ms: Optional[int], end_ms: Optional[int],
                         max_retries: int, backoff_factor: float, dry_run: bool,
                         per_asset_log: bool = True) -> Dict[str, Any]:
    """
    Ejecuta backfill para un solo asset; maneja reintentos/backoff; actualiza backfill_status.
    Retorna dict resumen.
    """
    asset_log = _asset_logger(asset) if per_asset_log else logger
    attempt = 0
    last_exception = None

    # deduce start/end with incremental if requested by caller previously
    while attempt <= max_retries:
        attempt += 1
        try:
            if dry_run:
                asset_log.info("[DRY-RUN] Would run backfill for %s %s start=%s end=%s", asset, interval, start_ms, end_ms)
                # do not update DB
                return {"asset": asset, "interval": interval, "status": "dry-run", "attempt": attempt}
            asset_log.info("Backfill attempt %d for %s %s (range %s - %s)", attempt, asset, interval, start_ms, end_ms)
            # Orchestrator.run_backfill_for deduce rango si start_ms/end_ms is None
            summary = orch.run_backfill_for(asset, interval, start_ms=start_ms, end_ms=end_ms, batch_window_ms=args.batch_window_ms)
            asset_log.info("Backfill OK for %s %s -> %s", asset, interval, summary)
            # update backfill_status: set last_ts = end of range (use summary or end_ms)
            last_ts_val = None
            # prefer summary end if present
            if isinstance(summary, dict):
                last_ts_val = summary.get("end_ms") or summary.get("end_ts") or summary.get("last_ts")
            if last_ts_val is None and end_ms:
                last_ts_val = end_ms
            _safe_update_backfill_status(storage, asset, interval, last_ts_val)
            return {"asset": asset, "interval": interval, "status": "ok", "summary": summary, "attempt": attempt}
        except KeyboardInterrupt:
            asset_log.warning("Interrupción por teclado en backfill de %s", asset)
            return {"asset": asset, "interval": interval, "status": "interrupted", "attempt": attempt}
        except Exception as e:
            last_exception = e
            asset_log.exception("Backfill attempt %d failed for %s: %s", attempt, asset, e)
            # update backfill_status with null last_ts to mark attempted
            try:
                _safe_update_backfill_status(storage, asset, interval, last_ts=None)
            except Exception:
                asset_log.exception("No se pudo actualizar backfill_status tras fallo")
            if attempt > max_retries:
                break
            # backoff sleep
            sleep_for = backoff_factor * (2 ** (attempt - 1))
            # cap the sleep to a reasonable amount (e.g. 1 hour)
            sleep_for = min(sleep_for, 3600)
            asset_log.info("Sleeping %.1fs before retry for %s", sleep_for, asset)
            time.sleep(sleep_for)
    # final failure
    logger.exception("Backfill final failure for %s after %d attempts: %s", asset, max_retries, last_exception)
    return {"asset": asset, "interval": interval, "status": "error", "error": str(last_exception), "attempt": attempt}

# --- CLI ---
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Init DB and run resumable backfills (concurrency, retries, resume).")
    p.add_argument("--init-db", action="store_true", help="Inicializa la DB (create tables).")
    p.add_argument("--watchlist", action="store_true", help="Leer assets desde storage.watchlist")
    p.add_argument("--csv", help="CSV con columna asset (y opcional interval)")
    p.add_argument("--asset", help="Asset único para backfill (overrides watchlist/csv)")
    p.add_argument("--interval", default="1h", help="Interval por defecto si no viene en watchlist/CSV")
    p.add_argument("--start", type=parse_time_arg, help="Start ts (ISO o ms)")
    p.add_argument("--end", type=parse_time_arg, help="End ts (ISO o ms)")
    p.add_argument("--incremental", action="store_true", help="Hacer incremental (usar backfill_status.last_ts como start)")
    p.add_argument("--concurrency", type=int, default=2, help="Número de threads en paralelo")
    p.add_argument("--max-retries", type=int, default=3, help="Reintentos por asset")
    p.add_argument("--backoff-factor", type=float, default=1.0, help="Factor base para backoff exponencial (segundos)")
    p.add_argument("--batch-window-ms", type=int, default=None, help="Tamaño de ventana para backfill_range (ms). Dejar None para heurística")
    p.add_argument("--sleep-between", type=float, default=0.5, help="sleep entre assets durante ejecución secuencial")
    p.add_argument("--dry-run", action="store_true", help="No persiste nada; solo muestra qué haría")
    p.add_argument("--limit", type=int, default=None, help="Limitar número de assets a procesar (para pruebas)")
    p.add_argument("--log-file", default=None, help="Escribir log principal a fichero")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    if args.log_file:
        fh_main = logging.FileHandler(args.log_file, encoding="utf-8")
        fh_main.setFormatter(fmt)
        logger.addHandler(fh_main)

    # crear storage + orchestrator
    storage = make_storage_from_env()
    orch = Orchestrator(storage=storage)

    if args.init_db:
        logger.info("Inicializando DB (init_db) ...")
        storage.init_db()
        logger.info("DB inicializada exitosamente.")

    # construir lista de assets
    try:
        assets = build_asset_list(args, storage)
    except Exception as e:
        logger.exception("No se pudo construir lista de assets: %s", e)
        sys.exit(2)

    if args.limit:
        assets = assets[: args.limit]

    if not assets:
        logger.info("No hay assets a procesar. Salida.")
        sys.exit(0)

    logger.info("Procesando %d assets (concurrency=%d)", len(assets), args.concurrency)

    # If incremental and start not provided, deduce per-asset from backfill_status
    tasks = []
    for a in assets:
        asset = a.get("asset")
        interval = a.get("interval") or args.interval
        start_ms = args.start
        end_ms = args.end
        if args.incremental and start_ms is None:
            try:
                st = storage.get_backfill_status(asset) if hasattr(storage, "get_backfill_status") else None
                if st and st.get("last_ts"):
                    start_ms = int(st.get("last_ts")) + 1
                    logger.debug("Incremental start for %s deduced as %s", asset, start_ms)
            except Exception:
                logger.exception("No pudo leerse backfill_status para asset %s", asset)
        tasks.append({"asset": asset, "interval": interval, "start_ms": start_ms, "end_ms": end_ms})

    results = []
    # Use ThreadPoolExecutor to run backfills in parallel safely
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        future_to_task = {}
        for t in tasks:
            future = ex.submit(_run_backfill_single, orch, storage, t["asset"], t["interval"],
                               t["start_ms"], t["end_ms"], args.max_retries, args.backoff_factor, args.dry_run)
            future_to_task[future] = t
        for fut in as_completed(future_to_task):
            t = future_to_task[fut]
            try:
                res = fut.result()
            except Exception as e:
                logger.exception("Backfill task raised exception for %s: %s", t, e)
                res = {"asset": t["asset"], "interval": t["interval"], "status": "error", "error": str(e)}
            results.append(res)
            # small delay between finishing tasks to avoid DB spikes
            time.sleep(args.sleep_between)

    # Summary
    oks = [r for r in results if r.get("status") == "ok" or r.get("status") == "dry-run"]
    errs = [r for r in results if r.get("status") not in ("ok", "dry-run")]
    logger.info("Backfill finished: %d OK, %d ERR", len(oks), len(errs))
    if errs:
        logger.info("Errors details: %s", errs)
        sys.exit(1)
    sys.exit(0)
