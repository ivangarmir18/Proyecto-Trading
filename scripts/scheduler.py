# scripts/scheduler.py
"""
Scheduler / Orquestador de recolección.
Expone run_cycle(storage, fetcher, config) que hace:
 - agrupar tickers según config['scheduler']['group_size']
 - llamar a fetcher.fetch_ohlcv para cada ticker, con save_callback del storage
 - respetar pausas y límites
 - registrar métricas mínimas en logs
"""
from __future__ import annotations
import time
import math
import logging
from typing import Any, Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import get_logger, now_ms

logger = get_logger("scheduler")

def _chunk_list(lst: List[Any], n: int) -> List[List[Any]]:
    """Divide lst en chunks de tamaño n."""
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def _safe_fetch(fetcher, asset: str, interval: str, since: Optional[int], limit: Optional[int], save_cb: Callable, meta: Optional[Dict]=None):
    try:
        df = fetcher.fetch_ohlcv(asset, interval=interval, since=since, limit=limit, save_callback=save_cb, meta=meta)
        return {"asset": asset, "rows": 0 if df is None else len(df), "ok": True}
    except Exception as e:
        logger.exception("Error fetch %s %s: %s", asset, interval, e)
        return {"asset": asset, "rows": 0, "ok": False, "error": str(e)}


def run_cycle(storage, fetcher, config: Dict[str, Any]):
    """
    Ejecuta un ciclo completo: primero cripto, luego acciones (según config).
    Este ciclo es idempotente: si hay errores se registran y se continúa.
    """
    save_cb = storage.make_save_callback()
    scheduler_cfg = config.get("scheduler", {})
    group_size = int(scheduler_cfg.get("group_size", 10))
    group_interval_seconds = float(scheduler_cfg.get("group_interval_seconds", 12))
    default_limit = int(config.get("app", {}).get("default_limit", 500))

    results = []
    # 1) CRIPTO
    cryptos = config.get("assets", {}).get("cripto", []) or []
    crypto_interval = "5m"  # puedes leer intervalos desde config si lo prefieres
    crypto_cycle_min = scheduler_cfg.get("crypto_cycle_minutes", 5)
    if cryptos:
        logger.info("Starting crypto cycle: %d tickers", len(cryptos))
        groups = _chunk_list(cryptos, group_size)
        for gi, group in enumerate(groups):
            logger.debug("Crypto group %d/%d: %s", gi+1, len(groups), group)
            # paralelizar por grupo con hilos para mejorar throughput sin romper rate limits del fetcher
            with ThreadPoolExecutor(max_workers=min(len(group), 8)) as ex:
                futures = []
                for asset in group:
                    # since: si existe last_ts se puede usar para pedir solo lo nuevo
                    last_ts = storage.get_last_ts(asset, crypto_interval)
                    since = last_ts + 1 if last_ts is not None else None
                    futures.append(ex.submit(_safe_fetch, fetcher, asset, crypto_interval, since, default_limit, save_cb, {"cycle":"crypto"}))
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
            # esperar entre grupos
            logger.debug("Esperando %ds entre grupos de cripto...", group_interval_seconds)
            time.sleep(group_interval_seconds)

    # 2) ACCIONES
    acciones = config.get("assets", {}).get("acciones", []) or []
    actions_interval = "1h"  # ajustar según config si lo deseas
    if acciones:
        logger.info("Starting acciones cycle: %d tickers", len(acciones))
        groups = _chunk_list(acciones, group_size)
        for gi, group in enumerate(groups):
            logger.debug("Acciones group %d/%d: %s", gi+1, len(groups), group)
            with ThreadPoolExecutor(max_workers=min(len(group), 6)) as ex:
                futures = []
                for asset in group:
                    last_ts = storage.get_last_ts(asset, actions_interval)
                    since = last_ts + 1 if last_ts is not None else None
                    futures.append(ex.submit(_safe_fetch, fetcher, asset, actions_interval, since, default_limit, save_cb, {"cycle":"acciones"}))
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
            logger.debug("Esperando %ds entre grupos de acciones...", group_interval_seconds)
            time.sleep(group_interval_seconds)

    # resumen
    total_rows = sum(r.get("rows",0) for r in results)
    successes = sum(1 for r in results if r.get("ok"))
    failures = len(results) - successes
    logger.info("Cycle finished: total tickers processed=%d, rows=%d, successes=%d, failures=%d", len(results), total_rows, successes, failures)
    return {"processed": len(results), "rows": total_rows, "successes": successes, "failures": failures}
