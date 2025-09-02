# core/utils.py
"""
Utilities shared across the project: config loader, timestamps, retry helper,
interval parsing, logging setup, CSV symbol reader, small numeric helpers.
"""
from __future__ import annotations
import json
import logging
import logging.config
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests
import pandas as pd

# -----------------------
# Config helpers
# -----------------------
def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------
# Time helpers
# -----------------------
def now_ts() -> int:
    return int(time.time())


def ts_to_iso(ts: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))


def iso_to_ts(s: str) -> int:
    # accepts ISO with or without timezone Z
    try:
        # many environments have no dateutil; fallback simple parser
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        # final fallback: try parsing ignoring timezone
        import dateutil.parser
        dt = dateutil.parser.isoparse(s)
        return int(dt.timestamp())


# -----------------------
# Interval helpers
# -----------------------
_INTERVAL_MAP = {
    'm': 60,
    'min': 60,
    'h': 3600,
    'd': 86400,
    'w': 86400 * 7
}

def parse_interval_to_seconds(interval: str) -> int:
    """
    Simple parser: '5m' -> 300, '1h' -> 3600, '1d' -> 86400, '2h' -> 7200
    """
    if not interval:
        raise ValueError("interval is empty")
    s = str(interval).strip().lower()
    # handle formats like '5m', '15min', '1h', '4h', '1d', '1w'
    num = ''
    unit = ''
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            unit += ch
    if num == '':
        num = '1'
    unit = unit or 'm'
    # normalize some units
    if unit in ('min', 'm'):
        sec_unit = 60
    elif unit in ('h',):
        sec_unit = 3600
    elif unit in ('d',):
        sec_unit = 86400
    elif unit in ('w',):
        sec_unit = 86400 * 7
    else:
        raise ValueError(f"Unknown interval unit: {unit}")
    return int(num) * sec_unit


# -----------------------
# Retry decorator & request helper
# -----------------------
def retry(exceptions=Exception, tries: int = 4, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying function on exceptions.
    Usage:
      @retry(Exception, tries=5, delay=1, backoff=2)
      def f(...):
          ...
    """
    def deco(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"{func.__name__} failed with {e}, retrying in {_delay}s..."
                    logging.getLogger(__name__).warning(msg)
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            # last attempt
            return func(*args, **kwargs)
        return wrapper
    return deco


@retry((requests.exceptions.RequestException,), tries=5, delay=1.0, backoff=2.0)
def request_with_retries(method: str, url: str, **kwargs):
    """
    Thin wrapper around requests to apply retries (via decorator).
    """
    r = requests.request(method, url, timeout=15, **kwargs)
    r.raise_for_status()
    return r


# -----------------------
# Logging
# -----------------------
DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default", "level": "INFO"}
    },
    "root": {"handlers": ["console"], "level": "INFO"}
}

def setup_logging(config: Optional[dict] = None):
    if config is None:
        config = DEFAULT_LOGGING
    logging.config.dictConfig(config)


# -----------------------
# Small numeric helpers
# -----------------------
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b
    except Exception:
        return default


def normalize_0_1(x: float, xmin: float, xmax: float) -> float:
    if xmax == xmin:
        return 0.0
    return max(0.0, min(1.0, (x - xmin) / (xmax - xmin)))


# -----------------------
# CSV helpers
# -----------------------
def read_symbols_csv(path: str, col: str = 'symbol') -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if col not in df.columns:
        # try first column
        return df.iloc[:, 0].dropna().astype(str).tolist()
    return df[col].dropna().astype(str).tolist()


# -----------------------
# Utility: chunking
# -----------------------
def chunked_iterable(it: Iterable[Any], size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
