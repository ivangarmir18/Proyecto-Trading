# run_service.py
from __future__ import annotations
import os
import time
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import threading
from core.fetch import Fetcher
from core.storage_postgres import make_storage_from_env  # si exists; run_service ya usa esta

# carga .env.production solo para testing local; en Render usarás ENV vars
load_dotenv(".env.production")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("watchlist-service")

# importa tu orquestador (main.run_pipeline)
from main import run_pipeline, load_assets_from_config, load_json, run_cmd_backfill

def backfill_all_assets():
    """Hace backfill automático de todos los símbolos de la watchlist."""
    storage = PostgresStorage()
    fetcher = Fetcher(storage)

    print("🔄 Iniciando backfill automático...")

    watchlist = storage.get_watchlist()
    if not watchlist:
        print("⚠️ No hay símbolos en la watchlist. Añade alguno desde el dashboard.")
        return

    for asset in watchlist:
        symbol = asset["symbol"]
        interval = asset["interval"]

        print(f"📥 Backfilling {symbol} ({interval})...")
        last_candle = storage.get_last_candle(symbol, interval)

        if last_candle:
            start_time = last_candle["timestamp"] + timedelta(minutes=1)
        else:
            # Si no hay datos previos, descarga últimos 30 días
            start_time = datetime.utcnow() - timedelta(days=30)

        try:
            fetcher.backfill_range(symbol, interval, start_time, datetime.utcnow())
            print(f"✅ Backfill de {symbol} completo.")
        except Exception as e:
            print(f"❌ Error haciendo backfill de {symbol}: {e}")


def send_telegram(message: str):
    # usa core.utils si ya existe; fallback básico
    try:
        from core.utils import send_telegram_message
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            send_telegram_message(message, token, chat_id)
    except Exception:
        logger.debug("send_telegram failed or core.utils not present")

# Backfill poller: procesa filas en backfill_requests
def backfill_poller(storage, config, poll_interval: int = 5):
    """
    Loop que consulta backfill_requests y ejecuta backfill para cada petición.
    Usa Fetcher.backfill_range(...) y storage.save_candles(...) para persistir.
    """
    logger.info("Backfill poller iniciado (poll_interval=%s)", poll_interval)
    # Instanciar fetcher con parámetros desde config o env
    fetcher_kwargs = config.get("api", {}).get("fetcher", {}) if config else {}
    # fallbacks: lee keys desde env ya que Fetcher hace eso por defecto
    fetcher = Fetcher(
        rate_limit_per_min=int(os.getenv("BINANCE_RATE_LIMIT_PER_MIN", fetcher_kwargs.get("rate_limit_per_min", 1200))),
        default_limit=int(os.getenv("FETCH_DEFAULT_LIMIT", fetcher_kwargs.get("default_limit", 500))),
        max_attempts=int(os.getenv("FETCH_MAX_ATTEMPTS", fetcher_kwargs.get("max_attempts", 5))),
        backoff_base=float(os.getenv("FETCH_BACKOFF_BASE", fetcher_kwargs.get("backoff_base", 1.0))),
        backoff_cap=float(os.getenv("FETCH_BACKOFF_CAP", fetcher_kwargs.get("backoff_cap", 60.0))),
    )

    while True:
        try:
            pending = storage.fetch_pending_backfill_requests(limit=5)
            if not pending:
                time.sleep(poll_interval)
                continue
            for req in pending:
                req_id = req["id"]
                asset = req["asset"]
                interval = req.get("interval") or config.get("intervals", ["1h"])[0]
                logger.info("Procesando backfill request id=%s asset=%s interval=%s", req_id, asset, interval)
                try:
                    # Calcular rangos: toma start ms del config.backfill_start_iso si existe
                    backfill_start_iso = config.get("backfill_start_iso")
                    if backfill_start_iso:
                        from core.utils import iso_to_ms
                        start_ms = int(iso_to_ms(backfill_start_iso))
                    else:
                        # por defecto, retrocedemos X dias según retention si está
                        retention = config.get("retention_policy", {})
                        days = retention.get(interval, 365) if isinstance(retention, dict) else 365
                        start_ms = int((datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp() * 1000)
                    end_ms = int(datetime.datetime.utcnow().timestamp() * 1000)

                    # run backfill_range and save via callback
                    df = fetcher.backfill_range(asset, interval, start_ms, end_ms, save_callback=storage.save_candles, progress=False)
                    logger.info("Backfill id=%s asset=%s -> %d rows", req_id, asset, 0 if df is None else len(df))
                    storage.mark_backfill_done(req_id, success=True)
                except Exception as e:
                    logger.exception("Backfill failed for req %s asset %s: %s", req_id, asset, e)
                    storage.mark_backfill_done(req_id, success=False, error=str(e))
        except Exception as e:
            logger.exception("Error en backfill_poller: %s", e)
            # si falla poller entero, duerme un poco y reintenta
            time.sleep(max(5, poll_interval))


def main_loop(config_path: str = "config.json", sleep_seconds: int = 300):
    config = load_json(config_path)
    assets = load_assets_from_config(config)
    intervals = config.get("intervals", ["1h"])
    storage = make_storage_from_env()
    # arrancar backfill poller en hilo (no bloquea el loop principal)
    try:
        poll_interval = int(os.getenv("BACKFILL_POLL_INTERVAL", "5"))
        t_poller = threading.Thread(target=backfill_poller, args=(storage, config, poll_interval), daemon=True)
        t_poller.start()
        logger.info("Backfill poller thread started.")
    except Exception as e:
        logger.exception("No se pudo iniciar backfill poller: %s", e)

    db_path = os.getenv("DB_PATH", config.get("app", {}).get("db_path", "data/db/data.db"))

    logger.info("Service started: assets=%d intervals=%s db=%s", len(assets), intervals, db_path)
    while True:
        try:
            logger.info("Running pipeline cycle...")
            # run_pipeline(assets, intervals, db_path, config, do_refresh=True, do_backfill=False)
            # call with backfill False to avoid huge writes; backfill manual / scheduled less often
            run_pipeline(assets, intervals, db_path, config, do_backfill=False, do_refresh=True)
            logger.info("Pipeline cycle finished successfully")
        except Exception as e:
            msg = f"Service error: {e}"
            logger.exception(msg)
            send_telegram(f"⚠️ Watchlist service error:\n{e}")
        time.sleep(sleep_seconds)
        # --- INICIO: LÓGICA DE BACKFILL AUTOMÁTICO ---
    # Verificar si la base de datos ya tiene datos.
    # Usamos el primer activo y primer intervalo como chequeo.
    if assets: # Agregamos una verificación para evitar errores si la lista de activos está vacía
        first_asset = assets[0]['symbol']
        first_interval = intervals[0]
        
        # Llama al nuevo método del objeto 'storage'
        if not storage.has_data(first_asset, first_interval):
            logger.info("Base de datos vacía. Iniciando backfill inicial...")
            run_cmd_backfill(config)
            logger.info("Backfill inicial completado.")
        else:
            logger.info("Datos históricos encontrados. Saltando backfill inicial.")
    else:
        logger.warning("No se encontraron activos en la configuración. No se realizará el backfill.")
    # --- FIN: LÓGICA DE BACKFILL AUTOMÁTICO ---
    

if __name__ == "__main__":
    try:
        from core.storage_postgres import PostgresStorage
        storage = PostgresStorage()
        storage.init_db()
        backfill_all_assets()
        
        logger.info("Database schema ensured by storage.init_db()")
    except Exception as e:
        logger.exception("Failed to init DB on worker startup: %s", e)
    
    # sleep_seconds configurable via env
    s = int(os.getenv("SERVICE_SLEEP_SECONDS", "300"))
    main_loop(sleep_seconds=s)
