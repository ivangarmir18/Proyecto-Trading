# main.py
"""
Entrada principal para Proyecto Trading.
Uso:
  python main.py init-db         # inicializa tablas en Postgres
  python main.py backfill        # backfill para todos los assets definidos en config (desde config.backfill_start_iso o default)
  python main.py fetch-once      # ejecuta un solo ciclo (usa scripts.scheduler.run_cycle)
  python main.py run             # arranca scheduler (APScheduler) y se queda a la escucha
  python main.py purge           # purga datos antiguos según config.retention_days

Nota: este main integra una pequeña capa CLI para el subcomando `backfill`:
  --asset, --from, --to, --timeframe, --source, --csv
"""
from __future__ import annotations
import os
import signal
import sys
import time
import logging
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler

# imports desde core (asegúrate de tener core/utils.py y core/storage_postgres.py actualizados)
from core.utils import load_config, get_logger, now_ms
from core.storage_postgres import make_storage_from_env, PostgresStorage
from core.fetch import Fetcher
from scripts.scheduler import run_cycle

logger = get_logger("main")

CONFIG_PATH = os.getenv("PROJECT_CONFIG_PATH", "config.json")


def build_storage_and_fetcher(cfg):
    """
    Construye objetos Storage y Fetcher a partir de config.
    Se deja PostgresStorage como la implementación por defecto.
    """
    storage = PostgresStorage(
        retention_policy=cfg.get("storage", {}).get("retention_policy")
    )
    # Fetcher: leer rate limit del config/api
    bin_cfg = cfg.get("api", {}).get("binance", {})
    rate_limit = bin_cfg.get("rate_limit_per_min") if isinstance(bin_cfg, dict) else None
    fetcher = Fetcher(
        binance_api_key=os.getenv("ENV_BINANCE_KEY"),
        binance_secret=os.getenv("ENV_BINANCE_SECRET"),
        rate_limit_per_min=int(rate_limit) if rate_limit else None,
        default_limit=int(cfg.get("app", {}).get("default_limit", 500)),
    )
    return storage, fetcher


def cmd_init_db(cfg):
    storage, _ = build_storage_and_fetcher(cfg)
    storage.init_db()
    logger.info("Init DB completado.")
    storage.close()


def cmd_backfill(cfg):
    storage, fetcher = build_storage_and_fetcher(cfg)
    save_cb = storage.make_save_callback()
    # backfill window: intenta leer desde config backfill_start_iso o calculado desde retention
    backfill_start_iso = cfg.get("backfill_start_iso")
    if backfill_start_iso:
        start_ms = int(getattr(__import__("core.utils"), "iso_to_ms")(backfill_start_iso))
    else:
        # por defecto: backfill de los últimos 365 días
        start_ms = now_ms() - 365 * 24 * 3600 * 1000
    end_ms = now_ms()
    assets = cfg.get("assets", {})
    all_assets = (assets.get("cripto", []) or []) + (assets.get("acciones", []) or [])
    logger.info("Iniciando backfill para %d activos desde %s hasta ahora", len(all_assets), start_ms)
    for asset in all_assets:
        interval = "5m" if asset in (assets.get("cripto") or []) else "1h"
        try:
            df = fetcher.backfill_range(asset, interval, start_ms, end_ms,
                                        per_call_limit=int(cfg.get("app", {}).get("default_limit", 500)),
                                        save_callback=save_cb, progress=True)
            logger.info("Backfill %s %s -> %d rows", asset, interval, len(df))
        except Exception:
            logger.exception("Backfill falló para %s", asset)
    storage.close()


def cmd_fetch_once(cfg):
    storage, fetcher = build_storage_and_fetcher(cfg)
    try:
        res = run_cycle(storage, fetcher, cfg)
        logger.info("fetch-once result: %s", res)
    finally:
        storage.close()


def cmd_purge(cfg):
    storage, _ = build_storage_and_fetcher(cfg)
    retention = cfg.get("retention_days", {})
    # por cada interval, convertir días a ms y purge with before_ts_ms
    for interval, days in retention.items():
        before = now_ms() - int(days) * 24 * 86400 * 1000
        deleted = storage.purge_old_data(before_ts_ms=before)
        logger.info("Purgado interval %s antes de %s -> deleted %d", interval, before, deleted)
    storage.close()


def cmd_run(cfg):
    storage, fetcher = build_storage_and_fetcher(cfg)
    scheduler_cfg = cfg.get("scheduler", {})
    # frecuencias
    crypto_minutes = int(scheduler_cfg.get("crypto_cycle_minutes", 5))
    actions_minutes = int(scheduler_cfg.get("actions_cycle_minutes", 5))
    # Scheduler
    sched = BackgroundScheduler(timezone="UTC")
    # job cada X minutos: run_cycle
    sched.add_job(lambda: run_cycle(storage, fetcher, cfg), 'interval', minutes=crypto_minutes, id='cycle_job')
    logger.info("Scheduler iniciado: ciclo cada %d minutos", crypto_minutes)
    sched.start()

    # Manejo de señal para cerrar bien
    def _shutdown(signum, frame):
        logger.info("Recibida señal %s, cerrando...", signum)
        try:
            sched.shutdown(wait=False)
        except Exception:
            logger.exception("Error al apagar scheduler")
        try:
            storage.close()
        except Exception:
            logger.exception("Error al cerrar storage")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Entrando en loop principal. Usa CTRL+C para salir.")
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        _shutdown("SIGINT", None)


# -------------------------
# Small CLI helper: parse minimal flags for backfill
# -------------------------
def _parse_options_from_argv(argv):
    """
    Parse minimal flags after the command, supports:
      --asset <symbol>
      --from <ISO or ms>
      --to <ISO or ms>
      --timeframe <1m|5m|1h|...>
      --source <str>
      --csv <path>
    Returns dict with keys if present.
    """
    opts = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--asset", "--from", "--to", "--timeframe", "--source", "--csv"):
            if i + 1 < len(argv):
                k = a.lstrip('-')
                opts[k] = argv[i+1]
                i += 2
            else:
                raise SystemExit(f"Flag {a} requiere un argumento")
        else:
            # ignore unknown tokens (or you can treat as error)
            i += 1
    return opts


def main():
    if not os.path.isfile(CONFIG_PATH):
        logger.error("No se encontró config.json en %s", CONFIG_PATH)
        sys.exit(1)
    cfg = load_config(CONFIG_PATH)
    if len(sys.argv) < 2:
        logger.error("Uso: python main.py [init-db|backfill|fetch-once|run|purge]")
        sys.exit(2)
    cmd = sys.argv[1]

    # parse options for subcommands (only needed for backfill)
    options = _parse_options_from_argv(sys.argv[2:]) if len(sys.argv) > 2 else {}

    # -------------------------
    # Patch para init-db y backfill (compatible con storage_postgres)
    # -------------------------
    from core import storage_postgres as storage
    from core.utils import PG_DB

    if cmd == "init-db":
        # inicializa esquema
        print("Inicializando esquema en Postgres...")
        try:
            storage.init_db()
            print("Init DB OK")
        except Exception as e:
            print("Init DB fallo:", e)
            raise

    elif cmd == "backfill":
        # Se espera flags o CSV o uso de fetcher
        asset = options.get("asset") or os.environ.get("BACKFILL_ASSET")
        from_iso = options.get("from") or options.get("from_iso")
        to_iso = options.get("to") or options.get("to_iso")
        timeframe = options.get("timeframe") or "5m"
        source = options.get("source") or "user"
        csv_path = options.get("csv")  # opcional

        if not asset or not from_iso or not to_iso:
            raise SystemExit("backfill: faltan parámetros --asset --from --to")

        # convertir y preparar rows:
        rows = []
        if csv_path:
            import csv
            from datetime import datetime, timezone
            with open(csv_path, newline='') as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    ts_raw = r.get('ts') or r.get('timestamp') or r.get('time')
                    try:
                        if '-' in ts_raw:
                            ts = int(datetime.fromisoformat(ts_raw).timestamp()*1000)
                        else:
                            ts = int(ts_raw)
                    except Exception:
                        ts = int(float(ts_raw))
                    rows.append({
                        "ts": ts,
                        "open": r.get('Open') or r.get('open'),
                        "high": r.get('High') or r.get('high'),
                        "low": r.get('Low') or r.get('low'),
                        "close": r.get('Close') or r.get('close'),
                        "volume": r.get('Volume') or r.get('volume'),
                    })
        else:
            # try to use your existing fetcher (core.fetch.fetch_asset)
            try:
                from core.fetch import fetch_asset
                rows = list(fetch_asset(asset, timeframe, from_iso, to_iso, source))
            except Exception as e:
                raise SystemExit(f"backfill: no hay fetcher disponible y no se proporcionó CSV ({e})")

        # upsert dentro de transacción y registrar job
        with PG_DB.get_conn() as conn:
            job_id = storage.create_job(conn, "backfill", None, timeframe, {"asset": asset, "from": from_iso, "to": to_iso, "source": source})
            storage.update_job_started(conn, job_id)
            try:
                n = storage.upsert_prices(conn, asset, timeframe, rows, source=source)
                # commit explícito (main decide)
                conn.commit()
                storage.update_job_success(conn, job_id)
                print(f"Backfill OK: {n} filas (job_id={job_id})")
            except Exception as e:
                # rollback y marcar fallo
                conn.rollback()
                storage.update_job_failed(conn, job_id, str(e))
                raise

    elif cmd == "fetch-once":
        cmd_fetch_once(cfg)
    elif cmd == "run":
        cmd_run(cfg)
    elif cmd == "purge":
        cmd_purge(cfg)
    else:
        logger.error("Comando desconocido: %s", cmd)
        sys.exit(3)


if __name__ == "__main__":
    main()
