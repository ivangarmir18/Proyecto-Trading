#!/usr/bin/env python3
"""
scripts/ensure_db_on_start.py

Inicializa DB (Postgres o sqlite), ejecuta backfill automático y backtests iniciales.
Control por variables de entorno:
  - SKIP_BACKFILL (1 = saltar backfill)
  - SKIP_INITIAL_BACKTEST (1 = saltar backtests iniciales)
  - BACKFILL_LIMIT (velas por símbolo, default 1000)
  - INITIAL_BACKTEST_LIMIT (número de activos a backtestear en arranque, default 5)
No aborta el arranque si algo falla; registra en logs.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ensure_db_on_start")

def main():
    logger.info("ensure_db_on_start: iniciando")

    # Init DB: prefer postgres (storage_postgres) si disponible, fallback sqlite init_db
    try:
        try:
            from core.storage_postgres import PostgresStorage
            pg = PostgresStorage()
            if hasattr(pg, "init_db"):
                pg.init_db()
                logger.info("Postgres init_db ejecutado.")
        except Exception:
            try:
                from core.storage import init_db as sqlite_init
                sqlite_init()
                logger.info("SQLite init_db ejecutado.")
            except Exception as e:
                logger.exception("No se pudo inicializar DB con los módulos storage disponibles: %s", e)
    except Exception as e:
        logger.exception("Error al init DB: %s", e)

    # Adapter ops: backfill and initial backtests
    try:
        from core.adapter import adapter
    except Exception as e:
        logger.exception("No se pudo importar core.adapter: %s", e)
        adapter = None

    SKIP_BACKFILL = os.getenv("SKIP_BACKFILL", "0").strip() == "1"
    SKIP_INITIAL_BACKTEST = os.getenv("SKIP_INITIAL_BACKTEST", "0").strip() == "1"
    BACKFILL_LIMIT = int(os.getenv("BACKFILL_LIMIT", "1000"))
    INITIAL_BACKTEST_LIMIT = int(os.getenv("INITIAL_BACKTEST_LIMIT", "5"))

    if adapter and not SKIP_BACKFILL:
        try:
            logger.info("Lanzando backfill automático (limit=%s)", BACKFILL_LIMIT)
            res = adapter.run_full_backfill(per_symbol_limit=BACKFILL_LIMIT)
            logger.info("Backfill finalizado. símbolos procesados: %s", len(res.get("results", {})))
        except Exception as e:
            logger.exception("Backfill automatico falló: %s", e)
    else:
        logger.info("Backfill automático saltado (SKIP or no adapter).")

    if adapter and not SKIP_INITIAL_BACKTEST:
        try:
            logger.info("Lanzando backtests iniciales (limit=%s)", INITIAL_BACKTEST_LIMIT)
            res_bt = adapter.run_initial_backtests(limit=INITIAL_BACKTEST_LIMIT)
            logger.info("Backtests iniciales finalizados. keys: %s", list(res_bt.keys())[:10])
        except Exception as e:
            logger.exception("Initial backtests failed: %s", e)
    else:
        logger.info("Initial backtests skipped.")

    logger.info("ensure_db_on_start completed at %s", datetime.utcnow().isoformat())

if __name__ == "__main__":
    main()
