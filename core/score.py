# core/score.py
"""
Score engine for crypto & stocks.
- Computes indicators (assumes indicators modules exist).
- Computes multi-dim score (EMA, support proximity, ATR, MACD, RSI, Fibonacci).
- Computes range_min, range_max, stop, target.
- Saves scores via core.storage.save_scores (rows list).
"""

from pathlib import Path
import json
import math
import time
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

# local imports (assume these files exist and are as previously provided)
from core.storage import load_candles, save_scores  # load_candles(symbol, interval) -> df
from indicators.ema import ema
from indicators.rsi import rsi
from indicators.atr import atr
from indicators.macd import macd
from indicators.fibonacci import compute_fibonacci_levels, nearest_level, proximity_score, save_levels_cache

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

# timeframe hierarchy to find adjacent lower/higher resolutions
# Note: keys are in your project notation ("5m","1h","4h","1d","1w")
TF_HIERARCHY = ["5m", "1h", "2h", "4h", "1d", "1w"]


# --------------------------
# Helpers
# --------------------------
def load_config(path: Path = None) -> dict:
    if path is None:
        path = CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_adjacent_higher(tf: str) -> Optional[str]:
    """Return the next *higher resolution* (coarser) in TF_HIERARCHY, e.g. 4h -> 1d.
       If not found, return None."""
    try:
        idx = TF_HIERARCHY.index(tf)
    except ValueError:
        return None
    # higher = next index to the right (coarser)
    if idx + 1 < len(TF_HIERARCHY):
        return TF_HIERARCHY[idx + 1]
    return None


def normalize_between_0_1(value: float, vmin: float, vmax: float) -> float:
    if vmin == vmax:
        return 0.5
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def sigmoid(x: float) -> float:
    # stable sigmoid mapped to (0,1)
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# --------------------------
# Indicator-derived scoring functions
# --------------------------
def score_trend(close: float, ema9: float, ema40: float) -> float:
    """
    Trend score:
    - 1.0 if close > ema40
    - 0.0 if close < ema9
    - linear interpolation between ema9..ema40 otherwise
    """
    if ema40 is None or math.isnan(ema40) or ema9 is None or math.isnan(ema9):
        # fallback: price above ema9 -> 0.6, else 0.4 (weak signal)
        return 0.6 if ema9 is not None and close > ema9 else 0.4 if ema9 is not None else 0.5
    if close > ema40:
        return 1.0
    if close < ema9:
        return 0.0
    denom = (ema40 - ema9)
    if denom == 0:
        return 0.5
    return max(0.0, min(1.0, (close - ema9) / denom))


def score_atr_rel(atr_val: float, price: float, atr_norm_scale: float = 0.02) -> float:
    """
    Normalize ATR relative to price: smaller ATR => higher score.
    atr_norm_scale: e.g. 0.02 means 2% of price is reference scale.
    """
    if price is None or price == 0:
        return 0.5
    atr_rel = (atr_val / price) if price else 0.0
    # map: 0 -> 1, atr_norm_scale -> 0, linear below atr_norm_scale
    val = 1.0 - (atr_rel / atr_norm_scale)
    return max(0.0, min(1.0, val))


def score_macd_hist(macd_hist: float, atr_val: float) -> float:
    """
    Normalize MACD histogram relative to ATR:
    - compute macd_signal = hist / (atr + eps)
    - pass through sigmoid shifted to map negative-><0.5, positive->>0.5
    """
    eps = 1e-9
    denom = (atr_val + eps)
    x = (macd_hist) / denom
    # scale factor to reduce extreme values
    x_scaled = x * 0.5
    return sigmoid(x_scaled)


def score_rsi_val(rsi_val: float) -> float:
    if rsi_val is None or math.isnan(rsi_val):
        return 0.5
    return max(0.0, min(1.0, rsi_val / 100.0))


def score_fibonacci_and_support(close: float, atr_val: float, fib_levels: Dict[str, float]) -> Tuple[float, Optional[float]]:
    """
    Returns (fibonacci_score, nearest_level_value)
    - Uses proximity_score (which uses ATR as scale).
    """
    if not fib_levels:
        return 0.5, None
    _, level = nearest_level(close, fib_levels)
    if level is None:
        return 0.5, None
    sc = proximity_score(close, level, atr_val)
    return sc, float(level)


# --------------------------
# Range / stop / target helpers
# --------------------------
def compute_stop_target(last_price: float, atr_val: float, multipliers: dict, support_level: Optional[float], resistance_level: Optional[float]) -> Tuple[float, float]:
    stop_mult = multipliers.get("stop", 1.3)
    target_mult = multipliers.get("target", 1.3)
    # If we have a realistic support/resistance use them, otherwise fall back to price +/- ATR*multipliers
    if support_level is not None:
        stop = support_level - stop_mult * atr_val
    else:
        stop = last_price - stop_mult * atr_val
    if resistance_level is not None:
        target = resistance_level + target_mult * atr_val
    else:
        target = last_price + target_mult * atr_val
    return float(stop), float(target)


# --------------------------
# Main processing per symbol / interval
# --------------------------
def process_symbol_interval(symbol: str, interval: str = "1h", config: dict = None, save_rows: bool = True) -> Optional[dict]:
    """
    Process a single symbol at `interval`.
    - loads candles via core.storage.load_candles(symbol, interval)
    - computes indicators, score, range, stop, target per candle
    - persists scores via core.storage.save_scores if save_rows True
    - returns dict {"last": last_row_dict, "rows": rows_list} for inspection
    """
    if config is None:
        config = load_config()
    weights = config.get("weights", {"ema": 0.25, "support": 0.20, "atr": 0.15, "macd": 0.15, "rsi": 0.10, "fibonacci": 0.15})
    multipliers = config.get("atr_multipliers", {"stop": 1.3, "target": 1.3})
    atr_norm_scale = config.get("atr_norm_scale", 0.02)  # 2% price = normalization

    # load main candles for this symbol & interval
    df = load_candles(symbol, interval=interval)
    if df is None or df.empty:
        print(f"[score] no candles for {symbol} {interval}")
        return None

    df = df.sort_values("ts").reset_index(drop=True)

    # compute indicators (vectorized)
    df["EMA9"] = ema(df["close"], period=9)
    df["EMA40"] = ema(df["close"], period=40)
    df["RSI"] = rsi(df["close"], period=14)
    df["ATR"] = atr(df, period=14)  # uses df high/low/close
    macd_line, macd_signal, macd_hist = macd(df["close"])
    df["MACD_hist"] = macd_hist

    # Compute Fibonacci levels for this interval
    fib_lookback = config.get("fibonacci_lookback", 144)
    fib_levels_main = compute_fibonacci_levels(df, lookback=fib_lookback)

    # Also compute Fibonacci for adjacent higher timeframe (if available) to "validate" levels
    adj_higher = find_adjacent_higher(interval)
    fib_levels_higher = {}
    if adj_higher:
        # attempt load candles from higher timeframe to compute levels; if not available skip
        try:
            df_higher = load_candles(symbol, interval=adj_higher)
            if df_higher is not None and not df_higher.empty:
                fib_lookback_higher = config.get("fibonacci_lookback_higher", fib_lookback)
                fib_levels_higher = compute_fibonacci_levels(df_higher, lookback=fib_lookback_higher)
        except Exception:
            fib_levels_higher = {}

    # combine levels: give priority to those that exist in both sets (within small tolerance)
    combined_levels = {}
    # add main levels
    for r, val in fib_levels_main.items():
        combined_levels[f"m_{r}"] = val
    # add higher levels but tag differently; if a higher level is very close (within 0.2% price) to a main one, increase its significance
    for r, val in fib_levels_higher.items():
        # look for near duplicates
        found = False
        for k_main, v_main in list(combined_levels.items()):
            if abs(v_main - val) <= max(1e-9, 0.002 * v_main):  # within 0.2% -> consider same
                # average them (strengthen)
                combined_levels[k_main] = float((v_main + val) / 2.0)
                found = True
                break
        if not found:
            combined_levels[f"h_{r}"] = val

    # cache combined levels for inspection (optional)
    try:
        save_levels_cache(symbol, combined_levels)
    except Exception:
        pass

    rows_to_save: List[Dict[str, Any]] = []

    # iterate each candle and compute score
    for _, row in df.iterrows():
        close = float(row["close"])
        ema9 = float(row.get("EMA9", np.nan)) if not pd.isna(row.get("EMA9", np.nan)) else None
        ema40 = float(row.get("EMA40", np.nan)) if not pd.isna(row.get("EMA40", np.nan)) else None
        rsi_v = float(row.get("RSI", 50.0)) if not pd.isna(row.get("RSI", 50.0)) else 50.0
        atr_v = float(row.get("ATR", 0.0))
        macd_h = float(row.get("MACD_hist", 0.0))

        # components
        comp = {}
        comp["ema"] = score_trend(close, ema9, ema40)
        comp["rsi"] = score_rsi_val(rsi_v)
        comp["atr"] = score_atr_rel(atr_v, close, atr_norm_scale)
        comp["macd"] = score_macd_hist(macd_h, atr_v)
        fib_sc, nearest = score_fibonacci_and_support(close, atr_v, combined_levels)
        comp["fibonacci"] = fib_sc

        # support proximity: use the same nearest level (we treat fib as proxy for support/resistance)
        comp["support"] = comp["fibonacci"]  # for now; could be extended with own support detection

        # weighted sum
        score_val = 0.0
        for k, w in weights.items():
            v = float(comp.get(k, 0.0))
            score_val += v * float(w)
        score_val = max(0.0, min(1.0, score_val))

        # --- START: IA adjustment ---
        try:
            from core.ai_inference import predict_prob
            # Preparar diccionario de features desde la vela actual
            feat = {
                "EMA9_EMA40_gap": (row["EMA9"] - row["EMA40"]) if row.get("EMA9") is not None and row.get("EMA40") is not None else 0.0,
                "RSI": row["RSI"] if row.get("RSI") is not None else 50.0,
                "ATR_rel": (row["ATR"] / row["close"]) if row.get("close") != 0 else 0.0,
                "MACD_hist": row["MACD_hist"] if row.get("MACD_hist") is not None else 0.0,
                "vol_over_ma5": (row["volume"] / (row.get("volume_ma5", 0.0)+1e-9)) if "volume_ma5" in row else 1.0,
                "ret_1": row.get("ret_1", 0.0),
                "ret_3": row.get("ret_3", 0.0)
            }
            model_feature_names = ["EMA9_EMA40_gap","RSI","ATR_rel","MACD_hist","vol_over_ma5","ret_1","ret_3"]
            p_ml = predict_prob(feat, model_feature_names)
        except Exception:
            p_ml = None

        if p_ml is not None:
            # multiplicador configurable
            min_mult = config.get("ai_multiplier_min", 0.85)
            max_mult = config.get("ai_multiplier_max", 1.0)
            multiplier = min_mult + (max_mult - min_mult) * p_ml
        else:
            multiplier = 1.0

        # Score ajustado por IA
        score_val = score_val * multiplier
        score_val = max(0.0, min(1.0, score_val))
        # --- END: IA adjustment ---

        # compute ranges, stop, target - using nearest as support/resistance if present
        # Determine resistance: next higher level above price (if any)
        resistance_level = None
        support_level = None
        if combined_levels:
            levels_vals = sorted(list(set([float(v) for v in combined_levels.values()])))
            # nearest below = support, nearest above = resistance
            below = [lv for lv in levels_vals if lv <= close]
            above = [lv for lv in levels_vals if lv > close]
            support_level = max(below) if below else None
            resistance_level = min(above) if above else None

        range_min = float(close - 0.5 * atr_v)
        range_max = float(close + 0.5 * atr_v)
        stop, target = compute_stop_target(close, atr_v, multipliers, support_level, resistance_level)

        ts_val = int(row["ts"]) if "ts" in row else int(pd.to_datetime(row["timestamp"]).astype("int64") // 10**6)
        ts_iso = pd.to_datetime(row["timestamp"]).isoformat() if "timestamp" in row else pd.to_datetime(ts_val, unit="ms").isoformat()

        rows_to_save.append({
            "ts": int(ts_val),
            "ts_iso": ts_iso,
            "score": float(score_val),
            "range_min": float(range_min),
            "range_max": float(range_max),
            "stop": float(stop),
            "target": float(target),
            "p_ml": float(p_ml) if p_ml is not None else None,
            "multiplier": float(multiplier) if multiplier is not None else None
        })

    # persist
    if save_rows and rows_to_save:
        try:
            # create DataFrame in the format expected by core.storage.save_scores
            df_scores = pd.DataFrame(rows_to_save)
            df_scores["asset"] = symbol
            df_scores["interval"] = interval
            # reorder columns to match expectations (optional)
            cols_preferred = ["asset", "interval", "ts", "score", "range_min", "range_max", "stop", "target", "p_ml", "multiplier", "ts_iso"]
            df_scores = df_scores[[c for c in cols_preferred if c in df_scores.columns]]
            # call storage.save_scores
            save_scores(df_scores)
        except Exception as e:
            print(f"[score] save_scores error for {symbol} {interval}: {e}")

    # return last for quick inspection and also full rows
    last = rows_to_save[-1] if rows_to_save else None
    return {"last": last, "rows": rows_to_save}


def compute_all_scores(*args, db_path: Optional[str] = None, config: dict = None):
    """
    Compute scores.

    Backwards-compatible behavior:
    - Called with no positional args: original behavior — compute scores for all symbols
      from data/config/cryptos.csv and actions.csv (uses config file).
    - Called as compute_all_scores(asset, interval, db_path=..., config=...):
      compute scores for that single asset/interval and return a DataFrame suitable
      for storage.save_scores(...) (so main.py can call storage.save_scores on it).
    """
    if config is None:
        config = load_config()

    # If called with two positional args, assume single-asset mode.
    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
        asset = args[0]
        interval = args[1]
        # Use process_symbol_interval but avoid double-saving here — let main.py decide.
        res = process_symbol_interval(asset, interval, config=config, save_rows=False)
        if res is None:
            return pd.DataFrame()  # empty DataFrame
        rows = res.get("rows", [])
        if not rows:
            return pd.DataFrame()
        df_scores = pd.DataFrame(rows)
        df_scores["asset"] = asset
        df_scores["interval"] = interval
        # reorder columns
        cols_preferred = ["asset", "interval", "ts", "score", "range_min", "range_max", "stop", "target", "p_ml", "multiplier", "ts_iso"]
        df_scores = df_scores[[c for c in cols_preferred if c in df_scores.columns]]
        return df_scores

    # ---------- original behavior (no positional args) ----------
    cfg_dir = ROOT / "data" / "config"
    cryptos = []
    stocks = []
    try:
        p = cfg_dir / "cryptos.csv"
        if p.exists():
            cryptos = pd.read_csv(p)["symbol"].dropna().astype(str).tolist()
    except Exception:
        cryptos = []
    try:
        p = cfg_dir / "actions.csv"
        if p.exists():
            stocks = pd.read_csv(p)["symbol"].dropna().astype(str).tolist()
    except Exception:
        stocks = []

    results = {}
    for s in cryptos:
        try:
            results[s] = process_symbol_interval(s, interval=config.get("crypto_interval", "1h"), config=config, save_rows=True)
            print(f"[score] processed {s} {config.get('crypto_interval', '1h')} -> {results[s]}")
        except Exception as e:
            print(f"[score] error processing {s}: {e}")

    for s in stocks:
        try:
            results[s] = process_symbol_interval(s, interval=config.get("stock_interval", "1h"), config=config, save_rows=True)
            print(f"[score] processed {s} {config.get('stock_interval', '1h')} -> {results[s]}")
        except Exception as e:
            print(f"[score] error processing {s}: {e}")

    return results
