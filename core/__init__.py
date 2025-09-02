# core/__init__.py
"""
Core package: expone funciones de storage para que otros módulos importen.
"""
from .storage import (
    init_db,
    get_connection,
    save_candles,
    load_candles,
    save_indicators,
    load_indicators,
    save_scores,
    load_scores,
)

__all__ = [
    "init_db",
    "get_connection",
    "save_candles",
    "load_candles",
    "save_indicators",
    "load_indicators",
    "save_scores",
    "load_scores",
]
