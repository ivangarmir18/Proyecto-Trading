#__init__
# indicators/__init__.py
"""
Paquete de indicadores: wrapper que expone cálculos comunes y una función
compute_all_indicators(df) que añade columnas útiles a un DataFrame de velas.

Requiere los módulos:
- indicators.ema
- indicators.rsi
- indicators.atr
- indicators.macd
- indicators.fibonacci

df expected columns: ['ts','open','high','low','close','volume'] (ts unix seconds or datetime)
Returns a DataFrame copy with new columns:
  ema9, ema40, macd_line, macd_signal, macd_hist, atr, rsi,
  support, resistance, fibonacci_levels (dict)
"""
from typing import Dict, Any
import pandas as pd

from .ema import ema as _ema
from .rsi import rsi as _rsi
from .atr import atr as _atr
from .macd import macd as _macd
from .fibonacci import compute_fibonacci_levels as _compute_fib, save_levels_cache as _save_fib_cache

__all__ = ["compute_all_indicators", "compute_support_resistance", "compute_fibonacci_levels"]


def compute_support_resistance(df: pd.DataFrame, lookback: int = 144) -> Dict[str, float]:
    """
    Simple support/resistance: returns recent low and high over `lookback` candles.
    """
    recent = df.tail(lookback)
    if recent.empty:
        return {"support": float("nan"), "resistance": float("nan")}
    support = float(recent['low'].min())
    resistance = float(recent['high'].max())
    return {"support": support, "resistance": resistance}


def compute_fibonacci_levels(df: pd.DataFrame, symbol: str = "UNK", lookback: int = 144):
    """
    Wrapper to compute fibonacci levels; also saves a cache json (optional).
    Returns dict of levels.
    """
    levels = _compute_fib(df, lookback=lookback)
    try:
        _save_fib_cache(symbol, levels)
    except Exception:
        # non-fatal: caching optional
        pass
    return levels


def compute_all_indicators(df: pd.DataFrame, symbol: str = "UNK", fib_lookback: int = 144) -> pd.DataFrame:
    """
    Given candle DataFrame, returns a new DataFrame with indicator columns appended.

    Important: does NOT modify input df in-place (returns a copy).
    """
    if df is None:
        raise ValueError("df is required")
    if df.empty:
        return df.copy()

    d = df.copy()

    # ensure close column numeric
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
    d['high'] = pd.to_numeric(d['high'], errors='coerce')
    d['low'] = pd.to_numeric(d['low'], errors='coerce')
    d['open'] = pd.to_numeric(d['open'], errors='coerce')
    # EMA
    d['ema9'] = _ema(d['close'], period=9)
    d['ema40'] = _ema(d['close'], period=40)

    # MACD
    macd_line, macd_signal, macd_hist = _macd(d['close'])
    d['macd_line'] = macd_line
    d['macd_signal'] = macd_signal
    d['macd_hist'] = macd_hist

    # ATR (needs high, low, close)
    d['atr'] = _atr(d[['high', 'low', 'close']])

    # RSI
    d['rsi'] = _rsi(d['close'])

    # Support/Resistance (simple)
    sr = compute_support_resistance(d, lookback=fib_lookback)
    d['support'] = sr['support']
    d['resistance'] = sr['resistance']

    # Fibonacci levels dict (same for all rows, computed from tail)
    fib_levels = compute_fibonacci_levels(d, symbol=symbol, lookback=fib_lookback)
    # store json/dict in a column — same value repeated (useful for caching)
    d['fibonacci_levels'] = [fib_levels] * len(d)

    # keep columns order tidy
    cols = list(d.columns)
    # move indicators to the end (optional)
    return d


# small convenience alias
compute_support_resistance.__doc__ = "Return support/resistance dict over lookback candles."
