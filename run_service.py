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
    from core.run_service import start_retention_job  # si pegaste el parche al final de run_service.py; o usa: from scripts.start_retention import main as start_retention_job

    # arrancar job de retención en modo seguro (solo cuenta lo que borraría)
    start_retention_job(storage, dry_run=True, interval_hours=24)

    orch = Orchestrator(storage=storage)
    # arrancar processor de jobs para consumir requests 'backfill' insertadas por la UI
    from run_service import start_job_processor_thread
    _job_thread = start_job_processor_thread(storage, orch, poll_interval_secs=8, batch_limit=4)


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

# --- Inicio parche run_service/scheduler: start_retention_job ---
import threading, time, logging

logger = logging.getLogger(__name__)

def start_retention_job(storage, retention_map=None, interval_hours=24, dry_run=False):
    """
    Lanza un hilo que ejecuta prune_old_candles diariamente.
    Llamar a start_retention_job(storage) desde el main del servicio.
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

    def _job():
        logger.info("Retention job started (interval_hours=%s)", interval_hours)
        while True:
            try:
                # si storage tiene método prune_old_candles lo usamos
                if hasattr(storage, "prune_old_candles"):
                    storage.prune_old_candles(retention_map=retention_map, dry_run=dry_run)
                else:
                    # fallback a helper de storage_postgres si está presente
                    try:
                        from core.storage_postgres import prune_old_candles_postgres
                        prune_old_candles_postgres(retention_map=retention_map, dry_run=dry_run)
                    except Exception:
                        logger.warning("No prune method available on storage and fallback failed.")
            except Exception:
                logger.exception("Error en retention job")
            # dormir N horas
            time.sleep(interval_hours * 3600)

    t = threading.Thread(target=_job, daemon=True)
    t.start()
    return t

# --- Fin parche run_service/scheduler ---

# --- Inicio parche run_service.py: processor de jobs (backfill) ---
import json
import threading
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def _fetch_pending_jobs_from_db(storage, limit: int = 5) -> list:
    """
    Lee filas con status='pending' de la tabla jobs.
    Devuelve lista de dicts {id, name, details(json), started_at, ...}
    Usa storage.get_conn() si existe, o intenta usar psycopg2 con DATABASE_URL.
    """
    rows = []
    try:
        if hasattr(storage, "get_conn"):
            with storage.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, name, status, details FROM jobs WHERE status = %s ORDER BY id ASC LIMIT %s", ('pending', limit))
                    fetched = cur.fetchall()
                    for r in fetched:
                        # r may be tuple (id,name,status,details)
                        jid, name, status, details = r
                        try:
                            details_obj = json.loads(details) if details else {}
                        except Exception:
                            details_obj = details
                        rows.append({"id": jid, "name": name, "details": details_obj})
            return rows
    except Exception:
        logger.exception("Error leyendo jobs via storage.get_conn")

    # Fallback: try psycopg2 + DATABASE_URL
    try:
        import os, psycopg2
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            return []
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, status, details FROM jobs WHERE status = %s ORDER BY id ASC LIMIT %s", ('pending', limit))
                fetched = cur.fetchall()
                for jid, name, status, details in fetched:
                    try:
                        details_obj = json.loads(details) if details else {}
                    except Exception:
                        details_obj = details
                    rows.append({"id": jid, "name": name, "details": details_obj})
        return rows
    except Exception:
        logger.exception("Error leyendo jobs via DATABASE_URL fallback")
        return []

def _update_job_status(storage, job_id: int, status: str, details: Optional[Dict[str, Any]] = None):
    """
    Actualiza job.status en la tabla jobs.
    Usa storage.update_job(job_id, status, details) si existe, si no, intenta SQL directo.
    """
    try:
        if hasattr(storage, "update_job"):
            # storage.update_job espera (job_id, status, details_dict)
            return storage.update_job(job_id, status, details or {})
    except Exception:
        logger.exception("storage.update_job falló para job %s", job_id)

    # fallback SQL
    try:
        if hasattr(storage, "get_conn"):
            with storage.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE jobs SET status=%s, details=COALESCE(%s, details), finished_at = CASE WHEN %s IN ('success','failed') THEN now() ELSE finished_at END WHERE id=%s",
                                (status, json.dumps(details) if details else None, status, job_id))
                conn.commit()
            return True
    except Exception:
        logger.exception("Fallback update_job SQL falló para job %s", job_id)
    return False

def _process_single_job(storage, orchestrator, job: dict):
    """
    Procesa un job tipo 'backfill':
      - marca job como 'running'
      - extrae details {'asset','interval'}
      - llama orchestrator.run_backfill_for(asset, interval, ...)
      - marca job 'success' o 'failed' y guarda detalles
    """
    job_id = job.get("id")
    name = job.get("name")
    details = job.get("details") or {}

    logger.info("Procesando job id=%s name=%s details=%s", job_id, name, details)
    # marcar running
    try:
        _update_job_status(storage, job_id, "running", {"started_by": "worker"})
    except Exception:
        logger.exception("No se pudo marcar job running")

    try:
        if name == "backfill":
            asset = details.get("asset")
            interval = details.get("interval")
            # determinar start/end: si el details contiene start_ms/end_ms usamos, si no pasamos None para reanudar incremental en orchestrator
            start_ms = details.get("start_ms")
            end_ms = details.get("end_ms")
            # ejecutar backfill mediante orchestrator (si existe)
            if hasattr(orchestrator, "run_backfill_for"):
                res = orchestrator.run_backfill_for(asset, interval, start_ms=start_ms, end_ms=end_ms)
            else:
                # fallback: si fetcher existe directamente (sin orchestrator) intentar simple fetcher.backfill_range
                fetcher = getattr(orchestrator, "fetcher", None) or getattr(storage, "fetcher", None)
                if fetcher and hasattr(fetcher, "backfill_range"):
                    # guardamos usando storage.save_candles si existe
                    def _cb(batch):
                        try:
                            if hasattr(storage, "save_candles"):
                                storage.save_candles(asset, batch)
                        except Exception:
                            logger.exception("Error guardando batch en backup flow")
                    fetcher.backfill_range(asset, interval, start_ms or 0, end_ms or fetcher.now_ms(), callback=_cb)
                    res = {"status": "done", "note": "fetched via fetcher"}
                else:
                    raise RuntimeError("Ni orchestrator.run_backfill_for ni fetcher.backfill_range disponibles")
            # marcar success y guardar resultado summary
            _update_job_status(storage, job_id, "success", {"result": res})
            logger.info("Job %s processed successfully", job_id)
            return True
        else:
            # soportar otros tipos de jobs en el futuro
            logger.warning("Job type no soportado por worker: %s", name)
            _update_job_status(storage, job_id, "failed", {"error": "unsupported_job_type"})
            return False
    except Exception as e:
        logger.exception("Error processing job id=%s: %s", job_id, e)
        _update_job_status(storage, job_id, "failed", {"error": str(e)})
        return False

def start_job_processor_thread(storage, orchestrator, poll_interval_secs: int = 10, batch_limit: int = 5, run_in_thread: bool = True):
    """
    Lanza thread que consulta jobs pendings cada poll_interval_secs y los procesa.
    - storage: instancia PostgresStorage u otra que implemente get_conn/update_job/save_candles
    - orchestrator: instancia de Orchestrator con run_backfill_for
    Devuelve el objeto Thread (si run_in_thread=True) o None si ejecuta en mismo hilo.
    """
    def _loop():
        logger.info("Job processor arrancado (poll_interval=%ss)", poll_interval_secs)
        while True:
            try:
                jobs = _fetch_pending_jobs_from_db(storage, limit=batch_limit)
                if not jobs:
                    time.sleep(poll_interval_secs)
                    continue
                for job in jobs:
                    try:
                        _process_single_job(storage, orchestrator, job)
                    except Exception:
                        logger.exception("Error procesando job %s", job)
                # pequeña pausa entre batches para no saturar DB
                time.sleep(0.5)
            except Exception:
                logger.exception("Job processor loop fallo, durmiendo 5s")
                time.sleep(5)

    if run_in_thread:
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return t
    else:
        _loop()
        return None

# --- Fin parche run_service.py: processor de jobs (backfill) ---
