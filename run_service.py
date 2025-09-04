#!/usr/bin/env python3
# run_service.py
"""
Worker service robusto — ciclo principal que itera watchlist y lanza backfills incrementales.

Características:
- Bloqueo de instancia (PID file) para evitar múltiples workers corriendo al mismo tiempo.
- Manejo de señales (SIGINT/SIGTERM) para parada limpia.
- Ciclo configurable por env var SERVICE_SLEEP_SECONDS y otros parámetros.
- Respeta backoff y evita que un fallo mate el worker.
- Opcional modo one-shot (--once) para debugging.
- Logs rotativos/archivo si se configura LOG_FILE.
- Usa Orchestrator.run_backfill_for para cada asset, y actualiza backfill_status.
"""
from __future__ import annotations

import os
import sys
import time
import signal
import logging
import argparse
from typing import Any
from pathlib import Path

# imports core
try:
    from core.orchestrator import Orchestrator
    from core.storage_postgres import make_storage_from_env
except Exception as e:
    raise RuntimeError("No se pudieron importar core.orchestrator o core.storage_postgres. Asegúrate de haberlos implementado.") from e

# Config defaults (env override)
SERVICE_SLEEP_SECONDS = int(os.getenv("SERVICE_SLEEP_SECONDS", "300"))
SERVICE_SLEEP_BETWEEN_ASSETS = float(os.getenv("SERVICE_SLEEP_BETWEEN_ASSETS", "0.5"))
SERVICE_MAX_CYCLES = int(os.getenv("SERVICE_MAX_CYCLES", "0"))  # 0 = infinite
PIDFILE = Path(os.getenv("SERVICE_PID_FILE", "/tmp/run_service.pid"))
LOG_FILE = os.getenv("SERVICE_LOG_FILE", None)
LOG_DIR = Path(os.getenv("PROJECT_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger("run_service")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)
if LOG_FILE:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# graceful shutdown flag
_should_stop = False

def _signal_handler(signum, frame):
    global _should_stop
    logger.info("Signal %s received; stopping gracefully after current work.", signum)
    _should_stop = True

# PID file helpers (create exclusive PID file to avoid duplicates)
def write_pidfile(pidfile: Path):
    try:
        if pidfile.exists():
            # check if process is running
            try:
                with open(pidfile, "r") as f:
                    pid = int(f.read().strip())
                # signal 0 to check existence
                os.kill(pid, 0)
                raise RuntimeError(f"PID file {pidfile} exists and process {pid} is running. Refusing to start duplicate worker.")
            except ProcessLookupError:
                # stale pidfile: remove it
                pidfile.unlink()
            except ValueError:
                pidfile.unlink()
            except PermissionError:
                pass
        pidfile.write_text(str(os.getpid()))
        pidfile.chmod(0o644)
    except Exception as e:
        logger.exception("No se pudo crear pidfile %s: %s", pidfile, e)
        raise

def remove_pidfile(pidfile: Path):
    try:
        if pidfile.exists():
            pidfile.unlink()
    except Exception:
        logger.exception("No se pudo borrar pidfile %s", pidfile)

def worker_cycle(orch: Orchestrator, storage: Any, sleep_between_assets: float):
    """
    Una iteración del worker:
      - lee watchlist
      - itera assets y llama run_backfill_for (incremental)
      - actualiza backfill_status y logs
    """
    try:
        watchlist = storage.list_watchlist() if hasattr(storage, "list_watchlist") else []
    except Exception:
        logger.exception("Error leyendo watchlist desde storage")
        watchlist = []

    if not watchlist:
        logger.info("Watchlist vacía.")
        return

    for entry in watchlist:
        if _should_stop:
            logger.info("Stop requested; breaking asset loop.")
            break
        try:
            if isinstance(entry, dict):
                asset = entry.get("asset")
                meta = entry.get("meta") or {}
                interval = meta.get("interval") or os.getenv("DEFAULT_INTERVAL", "1h")
            else:
                asset = entry
                interval = os.getenv("DEFAULT_INTERVAL", "1h")
            logger.info("Processing asset %s interval %s", asset, interval)

            # compute start_ms using backfill_status if available
            start_ms = None
            try:
                if hasattr(storage, "get_backfill_status"):
                    st = storage.get_backfill_status(asset)
                    if st and st.get("last_ts"):
                        start_ms = int(st.get("last_ts")) + 1
            except Exception:
                logger.exception("No se pudo leer backfill_status para %s", asset)

            # call orchestrator
            try:
                summary = orch.run_backfill_for(asset, interval, start_ms=start_ms)
                logger.info("Backfill summary for %s: %s", asset, summary)
                # update DB status
                if hasattr(storage, "update_backfill_status"):
                    last_ts_val = None
                    if isinstance(summary, dict):
                        last_ts_val = summary.get("end_ms") or summary.get("end_ts") or summary.get("last_ts")
                    storage.update_backfill_status(asset, interval=interval, last_ts=last_ts_val)
            except Exception:
                logger.exception("Backfill failed for asset %s; continuing to next asset", asset)
        except Exception:
            logger.exception("Failure processing watchlist entry: %s", entry)
        time.sleep(sleep_between_assets)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run background service worker for backfills (daemon).")
    p.add_argument("--once", action="store_true", help="Run a single cycle and exit (useful for cron/testing).")
    p.add_argument("--sleep", type=int, default=SERVICE_SLEEP_SECONDS, help="Seconds to sleep between cycles.")
    p.add_argument("--sleep-between-assets", type=float, default=SERVICE_SLEEP_BETWEEN_ASSETS, help="Sleep between assets.")
    p.add_argument("--max-cycles", type=int, default=SERVICE_MAX_CYCLES, help="Max cycles before exit (0=infinite).")
    p.add_argument("--no-pidfile", action="store_true", help="Don't write pidfile (for containerized runs).")
    p.add_argument("--log-file", default=None, help="Optional log file to append logs.")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.log_file:
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # write pidfile to prevent duplicate workers (unless suppressed)
    if not args.no_pidfile:
        try:
            write_pidfile(PIDFILE)
        except Exception as e:
            logger.error("PID file creation failed: %s", e)
            sys.exit(2)

    # signal handling for graceful stop
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # storage + orchestrator
    storage = make_storage_from_env()
    orch = Orchestrator(storage=storage)

    # try init db (idempotent)
    try:
        storage.init_db()
    except Exception:
        logger.exception("init_db failed (continuing)")

    cycle = 0
    try:
        while not _should_stop:
            cycle += 1
            logger.info("Worker cycle %d start", cycle)
            try:
                worker_cycle(orch, storage, args.sleep_between_assets)
            except Exception:
                logger.exception("Unhandled error in worker cycle")
            if args.once:
                logger.info("--once specified; exiting after one cycle")
                break
            if args.max_cycles and cycle >= args.max_cycles:
                logger.info("Reached max cycles %d; exiting", args.max_cycles)
                break
            # sleep with early exit if stop requested
            slept = 0
            while slept < args.sleep and not _should_stop:
                time.sleep(1)
                slept += 1
            if _should_stop:
                break
    finally:
        # cleanup
        try:
            if not args.no_pidfile:
                remove_pidfile(PIDFILE)
        except Exception:
            logger.exception("Error removing pidfile")
        try:
            if hasattr(storage, "close"):
                storage.close()
        except Exception:
            logger.exception("Error closing storage")
        logger.info("Worker stopped cleanly.")
