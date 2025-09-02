# run_service.py
from __future__ import annotations
import os
import time
import logging
from dotenv import load_dotenv

# carga .env.production solo para testing local; en Render usarás ENV vars
load_dotenv(".env.production")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("watchlist-service")

# importa tu orquestador (main.run_pipeline)
from main import run_pipeline, load_assets_from_config, load_json

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

def main_loop(config_path: str = "config.json", sleep_seconds: int = 300):
    config = load_json(config_path)
    assets = load_assets_from_config(config)
    intervals = config.get("intervals", ["1h"])
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

if __name__ == "__main__":
    # sleep_seconds configurable via env
    s = int(os.getenv("SERVICE_SLEEP_SECONDS", "300"))
    main_loop(sleep_seconds=s)
