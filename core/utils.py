# core/utils.py
"""
Utilities: carga de config, logger centralizado y helpers temporales.
Diseñado para ser usado por main.py, scheduler.py y otros módulos.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import pytz

DEFAULT_CONFIG_PATH = os.getenv("PROJECT_CONFIG_PATH", "config.json")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def get_logger(name: str = "proyecto-trading", level: Optional[str] = None) -> logging.Logger:
    level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


# Time helpers (consistent: use UTC everywhere)
def now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def iso_to_ms(iso_ts: str) -> int:
    # acepta ISO8601 con o sin zona
    dt = datetime.fromisoformat(iso_ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
