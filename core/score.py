# core/score.py
"""
core/score.py
--------------
Cálculo del "score" por fila (timestamp) a partir de los indicadores ya calculados
(en la forma que produce apply_indicators: value dict con keys 'ema','rsi','macd','atr','fibonacci','support').

Funciones principales:
 - indicator_to_score_* : mapeos de cada indicador a valor 0..1
 - compute_score_for_row(value_dict, weights, cfg) -> dict { score, details }
 - compute_scores_from_df(df, weights, method='weighted', cfg=None) -> list[dict]

Salida por fila:
{
  "ts": <int>,
  "score": <0..1 float>,
  "details": {
      "ema": {"raw":..., "score":...},
      "rsi": {...},
      "macd": {...},
      "atr": {...},
      "fibonacci": {...},
      "support": {...},
      "entry": <float>,
      "stop": <float>,
      "target": <float>
  }
}

Stop/target:
 - stop_distance = 1.3 * ATR
 - target_distance = 2.6 * ATR
 - For long: stop = entry - stop_distance ; target = entry + target_distance
 - For short: reversed (not implemented auto-detection of short vs long; current implementation assumes LONG bias)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import math
import numpy as np
import logging
import pandas
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# -------------------- mapping functions (raw -> 0..1) -------------------- #
def ema_to_score(raw_ema: float) -> float:
    """
    raw_ema: value computed by indicators.apply_indicators (tanh scaled relative diff, roughly -1..1)
    Interpreting: positive => bullish -> higher score.
    """
    try:
        s = float(raw_ema)
    except Exception:
        s = 0.0
    # map -1..1 -> 0..1
    s = max(-1.0, min(1.0, s))
    return (s + 1.0) / 2.0


def rsi_to_score(raw_rsi: float, lower: float = 30.0, upper: float = 70.0) -> float:
    """
    raw_rsi: 0..100. Best zone typically above 50, penalize overbought extreme > upper and oversold < lower.
    We map:
      - rsi <= lower -> 0.25
      - rsi == 50 -> 0.6
      - rsi between lower..upper -> linear mapping favoring middle
      - rsi > upper -> decreasing score (overbought)
    Adjust mapping to prefer slightly bullish (>50).
    """
    try:
        r = float(raw_rsi)
    except Exception:
        return 0.5
    # clamp
    r = max(0.0, min(100.0, r))
    # normalize 0..1
    mid = 50.0
    if r <= lower:
        return 0.2
    if r >= upper:
        # penalize heavy overbought slightly
        val = 0.8 - (r - upper) / (100.0 - upper) * 0.6
        return float(max(0.0, min(0.95, val)))
    # between lower and upper: map to 0.2..0.9 favoring mid->higher
    norm = (r - lower) / (upper - lower)
    # use ease curve
    val = 0.2 + 0.7 * (norm ** 0.8)
    return float(min(max(val, 0.0), 1.0))


def macd_to_score(raw_macd: float) -> float:
    """
    raw_macd: roughly -1..1 after tanh scaling. Positive -> bullish.
    Map to 0..1.
    """
    try:
        m = float(raw_macd)
    except Exception:
        m = 0.0
    m = max(-1.0, min(1.0, m))
    return (m + 1.0) / 2.0


def atr_to_score(raw_atr: float, reference_price: Optional[float] = None) -> float:
    """
    ATR is absolute volatility. Lower ATR relative to price is preferred for 'cleaner' entries.
    We compute coefficient = atr / reference_price (if reference present), then map invert: lower -> higher score.
    If reference_price missing or zero, fallback to an empirical mapping that normalizes within [0..1] using a soft curve.
    """
    try:
        a = float(raw_atr)
    except Exception:
        return 0.5
    if a <= 0:
        return 0.9
    if reference_price and reference_price > 0:
        rel = a / reference_price
        # typical rel ranges: 0.0001 .. 0.05 ; map via logistic-like function
        val = 1.0 / (1.0 + rel * 200.0)  # scale factor 200 chosen empirically
        return float(max(0.0, min(1.0, val)))
    # fallback: small atr -> good
    val = 1.0 / (1.0 + a)
    return float(max(0.0, min(1.0, val)))


def fibonacci_to_score(fib_dict: Dict[str, float], close: Optional[float]) -> float:
    """
    fib_dict: mapping fib levels to prices (from indicators)
    close: current price
    We reward price near a lower fib support (i.e., close to fib_0..fib_1). If info missing, return 0.5.
    """
    if not fib_dict or close is None:
        return 0.5
    # find fib levels below or equal to close, choose the nearest one
    try:
        levels = [(k, float(v)) for k, v in fib_dict.items() if isinstance(v, (int, float))]
        if not levels:
            return 0.5
        # compute distances
        dist = [(k, abs(close - price), price) for k, price in levels]
        # choose minimal distance
        kmin, dmin, price_min = min(dist, key=lambda x: x[1])
        # normalize distance relative to span between fib_0 and fib_1 if available
        if 'fib_0' in fib_dict and 'fib_1' in fib_dict:
            span = abs(float(fib_dict['fib_0']) - float(fib_dict['fib_1'])) or 1.0
            score = 1.0 - min(1.0, dmin / (0.5 * span))
            return float(max(0.0, min(1.0, score)))
        else:
            # fallback: small distance -> good
            score = 1.0 / (1.0 + dmin)
            return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.5


def support_to_score(support_info: Dict[str, Any], close: Optional[float]) -> float:
    """
    support_info expected as {"support_price": float, "span": float, "support_score": float}
    If present, return support_score; otherwise 0.5
    """
    if not support_info or close is None:
        return 0.5
    try:
        return float(max(0.0, min(1.0, float(support_info.get('support_score', 0.5)))))
    except Exception:
        return 0.5


# -------------------- compute single-row score -------------------- #
def compute_score_for_row(value: Dict[str, Any], weights: Dict[str, float], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    value: dict produced by apply_indicators (keys: ema, rsi, macd, atr, fibonacci, support)
    weights: normalized dict (sum -> 1.0) with keys matching: ema, support, atr, macd, rsi, fibonacci
    cfg: optional dict: { 'entry_bias': 'close'|'fib_support', 'side': 'long'|'short' }
    Returns:
      { "score": 0..1, "details": { per-indicator raw & mapped, entry, stop, target } }
    """
    cfg = cfg or {}
    entry_bias = cfg.get("entry_bias", "close")  # currently only close supported
    side = cfg.get("side", "long")  # 'long' or 'short' (used to invert stop/target)

    # read raw values with safe defaults
    raw_ema = float(value.get('ema', 0.0))
    raw_rsi = float(value.get('rsi', 50.0))
    raw_macd = float(value.get('macd', 0.0))
    raw_atr = float(value.get('atr', 0.0))
    raw_fib = value.get('fibonacci', {}) or {}
    raw_support = value.get('support', {}) or {}
    # reference price for ATR normalization - prefer close
    ref_price = None
    # If value contains a 'close' key deliver it; otherwise external caller should pass reference_price in cfg
    ref_price = float(value.get('close')) if value.get('close') is not None else cfg.get('reference_price')

    # per-indicator mapped scores 0..1
    s_ema = ema_to_score(raw_ema)
    s_rsi = rsi_to_score(raw_rsi, lower=cfg.get('rsi_lower', 30.0), upper=cfg.get('rsi_upper', 70.0))
    s_macd = macd_to_score(raw_macd)
    s_atr = atr_to_score(raw_atr, reference_price=ref_price)
    s_fib = fibonacci_to_score(raw_fib, ref_price)
    s_support = support_to_score(raw_support, ref_price)

    # ensure canonical keys in weights
    w_ema = float(weights.get('ema', 0.0))
    w_support = float(weights.get('support', 0.0))
    w_atr = float(weights.get('atr', 0.0))
    w_macd = float(weights.get('macd', 0.0))
    w_rsi = float(weights.get('rsi', 0.0))
    w_fib = float(weights.get('fibonacci', 0.0))

    # weighted sum
    score = (s_ema * w_ema + s_support * w_support + s_atr * w_atr + s_macd * w_macd + s_rsi * w_rsi + s_fib * w_fib)
    # clamp to 0..1
    score = max(0.0, min(1.0, float(score)))

    # compute entry/stop/target using ATR (if ref_price available)
    entry_price = None
    stop_price = None
    target_price = None
    if ref_price:
        entry_price = float(ref_price)
        stop_distance = 1.3 * raw_atr
        target_distance = 2.6 * raw_atr
        if side == 'long':
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance

    details = {
        "ema": {"raw": raw_ema, "score": s_ema, "weight": w_ema},
        "rsi": {"raw": raw_rsi, "score": s_rsi, "weight": w_rsi},
        "macd": {"raw": raw_macd, "score": s_macd, "weight": w_macd},
        "atr": {"raw": raw_atr, "score": s_atr, "weight": w_atr},
        "fibonacci": {"raw": raw_fib, "score": s_fib, "weight": w_fib},
        "support": {"raw": raw_support, "score": s_support, "weight": w_support},
        "entry": entry_price,
        "stop": stop_price,
        "target": target_price
    }

    return {"score": score, "details": details}


# -------------------- batch compute for DataFrame -------------------- #
def compute_scores_from_df(df: "pandas.DataFrame", weights: Dict[str, float], method: str = "weighted", cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    df: pandas.DataFrame where every row corresponds to a timestamp and contains indicator columns
        OR is the result of pd.json_normalize on indicators.value (i.e., has columns ema,rsi,macd,atr,fibonacci,support)
        Important: If 'ts' column present, it will be used for output.
        If 'close' present, it will be used as reference_price for ATR normalization and entry.
    weights: dict normalized (sum -> 1.0). Keys: ema, support, atr, macd, rsi, fibonacci
    method: reserved for future variants; currently only 'weighted' supported.
    cfg: optional dict passed to compute_score_for_row
    Returns: list of dicts: [{ "ts": <int>, "score": <0..1>, "details": {...} }, ...] ordered by df order.
    """
    import pandas as pd
    if df is None or df.empty:
        return []
    cfg = cfg or {}
    # ensure keys exist in weights
    expected = ['ema', 'support', 'atr', 'macd', 'rsi', 'fibonacci']
    # safe-get weights with 0 defaults
    w = {k: float(weights.get(k, 0.0)) for k in expected}

    out = []
    # iterate rows vectorized-friendly (pandas apply is acceptable here)
    def _row_to_val(row):
        # build value dict consistent with compute_score_for_row expectations
        val = {}
        for k in expected:
            val[k] = row.get(k, None)
        # include close if present
        if 'close' in row.index:
            val['close'] = row['close']
        if 'ts' in row.index:
            val['ts'] = int(row['ts'])
        return val

    # Use DataFrame.itertuples for speed
    for i, row in df.iterrows():
        row_map = row.to_dict()
        val = _row_to_val(row)
        scored = compute_score_for_row(val, w, cfg=cfg)
        ts = int(row_map.get('ts')) if 'ts' in row_map and not pd.isna(row_map.get('ts')) else None
        out_item = {"ts": ts, "score": float(scored['score']), "details": scored['details']}
        out.append(out_item)
    return out
