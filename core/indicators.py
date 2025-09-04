# core/indicators.py
"""
core/indicators.py
-------------------
Funciones para calcular indicadores técnicos y un wrapper `apply_indicators`
que devuelve un DataFrame con columnas canónicas y una fila por vela con:
 - ts (unix seconds)
 - open, high, low, close, volume (si vienen)
 - ema, rsi, macd (hist), atr, fibonacci (dict), support (dict)

La salida está preparada para serializarse a JSON y guardarse en la tabla `indicators`
como `value` (ej. value: { "ema": ..., "rsi": ..., "macd": ..., "atr": ..., "fibonacci": {...}, "support": {...} })

Design goals:
 - vectorizado con pandas / numpy
 - tolerante a datos faltantes
 - parámetros configurables
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------- low-level indicator implementations ---------------------- #
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (pandas ewm)."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (classic Wilder smoothing). Returns 0..100."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder's smoothing
    roll_up = up.ewm(alpha=1.0/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50.0)  # neutral for early values


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, histogram (macd - signal)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range for ATR calculation. Expects columns high, low, close."""
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1).fillna(df['close'])
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder). Returns same index as df."""
    tr = true_range(df)
    # Wilder smoothing
    atr_series = tr.ewm(alpha=1.0/period, adjust=False).mean()
    return atr_series.fillna(method='backfill')  # fill initial NaNs


# ---------------------- support / fibonacci helpers ---------------------- #
def find_recent_swing_high_low(df: pd.DataFrame, lookback: int = 100) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    Busca el swing high y swing low más relevantes en un lookback (simple heuristic):
    - swing_high: máximo de highs en window lookback
    - swing_low: mínimo de lows en window lookback
    Devuelve (swing_high_price, swing_low_price, index_high, index_low)
    """
    if df.shape[0] < 2:
        return None, None, None, None
    sub = df.tail(lookback)
    idx_high = int(sub['high'].idxmax())
    idx_low = int(sub['low'].idxmin())
    return float(sub.loc[idx_high, 'high']), float(sub.loc[idx_low, 'low']), idx_high, idx_low


def fibonacci_levels_from_range(high: float, low: float) -> Dict[str, float]:
    """
    Devuelve niveles fibonacci básicos (0.0..1.0) traducidos a precios:
    common retracements: 0.236, 0.382, 0.5, 0.618, 0.786
    """
    span = high - low if high is not None and low is not None else None
    if span is None or span == 0:
        return {}
    levels = {}
    for r in (0.236, 0.382, 0.5, 0.618, 0.786):
        levels[f"fib_{r}"] = high - r * span
    levels["fib_0"] = high
    levels["fib_1"] = low
    return levels


def support_score_from_levels(close: float, support_price: float, span: float) -> float:
    """
    Calcula un soporte simple: cuanto más cercano esté el precio al soporte (por debajo o encima),
    mejor puntuación. Retorna 0..1.
    """
    if support_price is None or span is None or span == 0:
        return 0.5
    # distancia relativa
    dist = abs(close - support_price) / span
    score = max(0.0, 1.0 - dist)  # si close==support -> 1.0 ; si muy lejos -> cercano a 0
    return float(min(max(score, 0.0), 1.0))


# ---------------------- main wrapper: apply_indicators ---------------------- #
def apply_indicators(df: pd.DataFrame, asset: Optional[str] = None, interval: Optional[str] = None,
                     cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Calcula indicadores sobre df de velas.

    Entradas:
      df: DataFrame que debe contener columnas: ts (unix seconds), open, high, low, close, volume (volume opcional).
          Puede tener un índice arbitrario; retornará ordenado por ts asc.
      asset, interval: opcionales, para meta logging/guardado.
      cfg: dict con parámetros opcionales:
          - ema_fast, ema_slow (int)
          - rsi_period (int)
          - macd_fast, macd_slow, macd_signal (int)
          - atr_period (int)
          - fib_lookback (int)
          - support_lookback (int)

    Salida:
      DataFrame ordenado por ts asc, con columnas:
        ts, open, high, low, close, volume,
        ema_short, ema_long, ema (float normalized approx -1..1),
        rsi (0..100),
        macd_line, macd_signal, macd_hist,
        atr,
        fibonacci (dict),
        support (dict)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cfg = cfg or {}
    ema_fast = int(cfg.get("ema_fast", 9))
    ema_slow = int(cfg.get("ema_slow", 21))
    rsi_period = int(cfg.get("rsi_period", 14))
    macd_fast = int(cfg.get("macd_fast", 12))
    macd_slow = int(cfg.get("macd_slow", 26))
    macd_signal = int(cfg.get("macd_signal", 9))
    atr_period = int(cfg.get("atr_period", 14))
    fib_lookback = int(cfg.get("fib_lookback", 200))
    support_lookback = int(cfg.get("support_lookback", 200))

    # copy and ensure columns
    df = df.copy()
    required = ['ts', 'open', 'high', 'low', 'close']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"apply_indicators: dataframe missing required column '{c}'")

    # order by ts asc
    df = df.sort_values("ts").reset_index(drop=True)
    close = df['close'].astype(float)

    # EMA
    df['ema_short'] = ema(close, ema_fast)
    df['ema_long'] = ema(close, ema_slow)
    # normalized ema signal in -1..1: use relative difference
    df['ema'] = ((df['ema_short'] - df['ema_long']) / df['ema_long'].replace(0, np.nan)).fillna(0.0)
    # clip to prevent extreme outliers
    df['ema'] = np.tanh(df['ema'] * 5)  # heuristic scaling

    # RSI
    df['rsi'] = rsi(close, period=rsi_period).clip(0, 100)

    # MACD
    macd_line, macd_signal_s, macd_hist = macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal_s
    df['macd_hist'] = macd_hist
    # normalize macd_hist to -1..1 using tanh
    df['macd'] = np.tanh(df['macd_hist'] / (df['close'].rolling(window=20).mean().replace(0, np.nan)))  # avoids scale issues
    df['macd'] = df['macd'].fillna(0.0)

    # ATR
    df['atr'] = atr(df[['high', 'low', 'close']].rename(columns={'high':'high','low':'low','close':'close'}), period=atr_period).fillna(method='bfill')

    # Fibonacci & support (per-row computed using lookback windows)
    fib_levels_list = []
    support_info_list = []

    # We'll compute swing high/low using rolling windows for each row (but optimized by reusing tail slices)
    # Simpler approach: compute single recent swing using the last fib_lookback rows (appropriate for indicators stored per ts)
    for idx in range(df.shape[0]):
        # slice up to current idx inclusive, but keep at least 2 rows
        start_idx = max(0, idx - fib_lookback + 1)
        window = df.iloc[start_idx:idx+1]
        if window.shape[0] < 2:
            fib_levels_list.append({})
            support_info_list.append({})
            continue
        high, low, hi_idx, lo_idx = find_recent_swing_high_low(window, lookback=window.shape[0])
        if high is None or low is None or high == low:
            fib_levels_list.append({})
            support_info_list.append({})
            continue
        fib_levels = fibonacci_levels_from_range(high, low)
        # compute support_score candidate: use lowest low as support
        span = high - low
        support_price = low  # pragmatic support = recent low
        close_price = float(df.iloc[idx]['close'])
        support_score = support_score_from_levels(close_price, support_price, span)
        support_info = {"support_price": support_price, "span": span, "support_score": support_score}
        fib_levels_list.append(fib_levels)
        support_info_list.append(support_info)

    df['fibonacci'] = fib_levels_list
    df['support'] = support_info_list

    # prepare final compact "value" column with canonical keys for storage
    # For each row, create a dict with keys: ema, rsi, macd, atr, fibonacci, support
    values = []
    for _, row in df.iterrows():
        val = {
            "ema": float(row['ema']) if not pd.isna(row['ema']) else 0.0,
            "rsi": float(row['rsi']) if not pd.isna(row['rsi']) else 50.0,
            "macd": float(row['macd']) if not pd.isna(row['macd']) else 0.0,
            "atr": float(row['atr']) if not pd.isna(row['atr']) else 0.0,
            "fibonacci": (row['fibonacci'] if isinstance(row['fibonacci'], dict) else {}),
            "support": (row['support'] if isinstance(row['support'], dict) else {})
        }
        values.append(val)

    df_out = df.copy()
    # attach canonical value dict and keep minimal columns
    df_out['value'] = values

    # Recommended returned columns: ts, open, high, low, close, volume, value
    keep_cols = [c for c in ['ts', 'open', 'high', 'low', 'close', 'volume', 'value'] if c in df_out.columns]
    return df_out[keep_cols]
