# core/score.py
"""
Score engine for crypto & stocks (adaptado para storage_postgres).

Características:
- Usa core.storage (si existe) o core.storage_postgres.
- process_symbol_interval(...) calcula indicadores + score por vela y devuelve filas.
- compute_all_scores(...) soporta modo single-asset (devuelve DataFrame)
  y modo "todos" (devuelve dict con resultados por símbolo y persiste).
"""
from __future__ import annotations
import json
import math
import time
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

# Intentar usar core.storage (si existe) para abstracción; fallback a storage_postgres
try:
    from core import storage as storage_mod  # type: ignore
except Exception:
    try:
        import core.storage_postgres as storage_mod  # type: ignore
    except Exception:
        storage_mod = None

if storage_mod is None:
    raise RuntimeError("No se encuentra módulo de storage (ni core.storage ni core.storage_postgres).")

# Importar funciones de storage (asumimos que existen)
# storage_mod debe exponer: load_candles, load_indicators (opcional), save_scores
load_candles = getattr(storage_mod, "load_candles")
load_indicators = getattr(storage_mod, "load_indicators", lambda a, i: pd.DataFrame())
save_scores = getattr(storage_mod, "save_scores")

# Indicadores locales
from indicators.ema import ema
from indicators.rsi import rsi
from indicators.atr import atr
from indicators.macd import macd
from indicators.fibonacci import compute_fibonacci_levels, nearest_level, proximity_score, save_levels_cache

# Config / constantes
ROOT = None  # no lo usamos directamente aquí
CONFIG_PATH_DEFAULT = "config.json"

TF_HIERARCHY = ["5m", "1h", "2h", "4h", "1d", "1w"]


def load_config(path: str = None) -> dict:
    p = path or CONFIG_PATH_DEFAULT
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_adjacent_higher(tf: str) -> Optional[str]:
    try:
        idx = TF_HIERARCHY.index(tf)
    except ValueError:
        return None
    if idx + 1 < len(TF_HIERARCHY):
        return TF_HIERARCHY[idx + 1]
    return None


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# scoring helpers (copiados/adaptados de tu versión)
def score_trend(close: float, ema9: Optional[float], ema40: Optional[float]) -> float:
    if ema40 is None or ema9 is None or math.isnan(ema40) or math.isnan(ema9):
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
    if price is None or price == 0:
        return 0.5
    atr_rel = (atr_val / price) if price else 0.0
    val = 1.0 - (atr_rel / atr_norm_scale)
    return max(0.0, min(1.0, val))


def score_macd_hist(macd_hist: float, atr_val: float) -> float:
    eps = 1e-9
    denom = (atr_val + eps)
    x = (macd_hist) / denom
    x_scaled = x * 0.5
    return sigmoid(x_scaled)


def score_rsi_val(rsi_val: float) -> float:
    if rsi_val is None or math.isnan(rsi_val):
        return 0.5
    return max(0.0, min(1.0, rsi_val / 100.0))


def score_fibonacci_and_support(close: float, atr_val: float, fib_levels: Dict[str, float]) -> Tuple[float, Optional[float]]:
    if not fib_levels:
        return 0.5, None
    _, level = nearest_level(close, fib_levels)
    if level is None:
        return 0.5, None
    sc = proximity_score(close, level, atr_val)
    return sc, float(level)


def compute_stop_target(last_price: float, atr_val: float, multipliers: dict, support_level: Optional[float], resistance_level: Optional[float]) -> Tuple[float, float]:
    stop_mult = multipliers.get("stop", 1.3)
    target_mult = multipliers.get("target", 1.3)
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
# Main per-symbol processing
# --------------------------
def process_symbol_interval(symbol: str, interval: str = "1h", config: dict = None, save_rows: bool = True, db_path: Optional[str] = None) -> Optional[dict]:
    """
    Procesa un símbolo/intervalo, calcula indicadores y scores.
    Si save_rows=True persiste usando storage.save_scores(..., db_path=db_path)
    Retorna {"last": last_row, "rows": rows_list}
    """
    if config is None:
        config = load_config()

    weights = config.get("weights", {"ema": 0.25, "support": 0.20, "atr": 0.15, "macd": 0.15, "rsi": 0.10, "fibonacci": 0.15})
    multipliers = config.get("atr_multipliers", {"stop": 1.3, "target": 1.3})
    atr_norm_scale = config.get("atr_norm_scale", 0.02)

    # load candles
    try:
        df = load_candles(symbol, interval)
    except Exception as e:
        print(f"[score] error loading candles for {symbol} {interval}: {e}")
        return None

    if df is None or df.empty:
        print(f"[score] no candles for {symbol} {interval}")
        return None

    df = df.sort_values("ts").reset_index(drop=True)

    # compute indicators (vectorized)
    try:
        df["EMA9"] = ema(df["close"], period=9)
    except Exception:
        df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    try:
        df["EMA40"] = ema(df["close"], period=40)
    except Exception:
        df["EMA40"] = df["close"].ewm(span=40, adjust=False).mean()

    try:
        df["RSI"] = rsi(df["close"], period=14)
    except Exception:
        df["RSI"] = 50.0

    try:
        df["ATR"] = atr(df, period=14)
    except Exception:
        # fallback simple ATR
        high_low = df["high"] - df["low"]
        df["ATR"] = high_low.rolling(window=14, min_periods=1).mean().fillna(0.0)

    try:
        macd_line, macd_signal, macd_hist = macd(df["close"])
        df["MACD_hist"] = macd_hist
    except Exception:
        df["MACD_hist"] = 0.0

    # Fibonacci (main timeframe)
    fib_lookback = config.get("fibonacci_lookback", 144)
    try:
        fib_levels_main = compute_fibonacci_levels(df, lookback=fib_lookback)
    except Exception:
        fib_levels_main = {}

    # Fibonacci from adjacent higher timeframe (if available)
    adj_higher = find_adjacent_higher(interval)
    fib_levels_higher = {}
    if adj_higher:
        try:
            df_higher = load_candles(symbol, adj_higher)
            if df_higher is not None and not df_higher.empty:
                fib_levels_higher = compute_fibonacci_levels(df_higher, lookback=config.get("fibonacci_lookback_higher", fib_lookback))
        except Exception:
            fib_levels_higher = {}

    # combine fib levels (prioritize/merge)
    combined_levels = {}
    for r, val in fib_levels_main.items():
        combined_levels[f"m_{r}"] = val
    for r, val in fib_levels_higher.items():
        found = False
        for k_main, v_main in list(combined_levels.items()):
            if abs(v_main - val) <= max(1e-9, 0.002 * v_main):
                combined_levels[k_main] = float((v_main + val) / 2.0)
                found = True
                break
        if not found:
            combined_levels[f"h_{r}"] = val

    # optional cache levels
    try:
        save_levels_cache(symbol, combined_levels)
    except Exception:
        pass

    rows_to_save: List[Dict[str, Any]] = []

    # iterate rows
    for _, row in df.iterrows():
        try:
            close = float(row["close"])
        except Exception:
            continue
        ema9 = float(row.get("EMA9")) if not pd.isna(row.get("EMA9")) else None
        ema40 = float(row.get("EMA40")) if not pd.isna(row.get("EMA40")) else None
        rsi_v = float(row.get("RSI")) if not pd.isna(row.get("RSI")) else 50.0
        atr_v = float(row.get("ATR")) if not pd.isna(row.get("ATR")) else 0.0
        macd_h = float(row.get("MACD_hist")) if not pd.isna(row.get("MACD_hist")) else 0.0

        comp = {}
        comp["ema"] = score_trend(close, ema9, ema40)
        comp["rsi"] = score_rsi_val(rsi_v)
        comp["atr"] = score_atr_rel(atr_v, close, atr_norm_scale)
        comp["macd"] = score_macd_hist(macd_h, atr_v)
        fib_sc, nearest = score_fibonacci_and_support(close, atr_v, combined_levels)
        comp["fibonacci"] = fib_sc
        comp["support"] = comp["fibonacci"]

        score_val = 0.0
        for k, w in weights.items():
            v = float(comp.get(k, 0.0))
            score_val += v * float(w)
        score_val = max(0.0, min(1.0, score_val))

        # AI adjustment (optional)
        try:
            from core.ai_inference import predict_prob
            feat = {
                "EMA9_EMA40_gap": (row.get("EMA9", 0.0) - row.get("EMA40", 0.0)) if row.get("EMA9") is not None and row.get("EMA40") is not None else 0.0,
                "RSI": row.get("RSI", 50.0) if row.get("RSI") is not None else 50.0,
                "ATR_rel": (row.get("ATR", 0.0) / (row.get("close", 1.0))) if row.get("close") else 0.0,
                "MACD_hist": row.get("MACD_hist", 0.0) if row.get("MACD_hist") is not None else 0.0,
                "vol_over_ma5": (row.get("volume", 1.0) / (row.get("volume_ma5", 1.0) + 1e-9)) if "volume_ma5" in row else 1.0,
                "ret_1": row.get("ret_1", 0.0) if row.get("ret_1") is not None else 0.0,
                "ret_3": row.get("ret_3", 0.0) if row.get("ret_3") is not None else 0.0
            }
            p_ml = predict_prob(feat)
        except Exception:
            p_ml = None

        if p_ml is not None:
            min_mult = config.get("ai_multiplier_min", 0.85)
            max_mult = config.get("ai_multiplier_max", 1.0)
            multiplier = min_mult + (max_mult - min_mult) * float(p_ml)
        else:
            multiplier = 1.0

        score_val = score_val * multiplier
        score_val = max(0.0, min(1.0, score_val))

        # support/resistance detection from combined_levels
        resistance_level = None
        support_level = None
        if combined_levels:
            levels_vals = sorted(list(set([float(v) for v in combined_levels.values()])))
            below = [lv for lv in levels_vals if lv <= close]
            above = [lv for lv in levels_vals if lv > close]
            support_level = max(below) if below else None
            resistance_level = min(above) if above else None

        range_min = float(close - 0.5 * atr_v)
        range_max = float(close + 0.5 * atr_v)
        stop, target = compute_stop_target(close, atr_v, multipliers, support_level, resistance_level)

        ts_val = int(row["ts"]) if "ts" in row else int(time.time())
        ts_iso = pd.to_datetime(int(ts_val), unit='s').isoformat()

        rows_to_save.append({
            "asset": symbol,
            "interval": interval,
            "ts": int(ts_val),
            "score": float(score_val),
            "range_min": float(range_min),
            "range_max": float(range_max),
            "stop": float(stop),
            "target": float(target),
            "p_ml": float(p_ml) if p_ml is not None else None,
            "multiplier": float(multiplier) if multiplier is not None else None,
            "ts_iso": ts_iso,
            "created_at": int(time.time())
        })

    if save_rows and rows_to_save:
        try:
            df_scores = pd.DataFrame(rows_to_save)
            # ensure types
            if 'created_at' not in df_scores.columns:
                df_scores['created_at'] = int(time.time())
            # call storage.save_scores (db_path optional)
            try:
                save_scores(df_scores, db_path=db_path)
            except TypeError:
                # some implementations may not accept db_path kw
                save_scores(df_scores)
        except Exception as e:
            print(f"[score] save_scores error for {symbol} {interval}: {e}")

    last = rows_to_save[-1] if rows_to_save else None
    return {"last": last, "rows": rows_to_save}


# --------------------------
# compute_all_scores (backwards compat)
# --------------------------
def compute_all_scores(*args, db_path: Optional[str] = None, config: dict = None):
    """
    Backwards-compatible compute_all_scores:
    - If called with (asset, interval, ...) returns a pd.DataFrame (single asset mode).
    - If called with no positional args, computes+persists for assets read from data/config CSVs (returns dict).
    """
    if config is None:
        config = load_config()

    # single-asset mode
    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
        asset = args[0]
        interval = args[1]
        res = process_symbol_interval(asset, interval, config=config, save_rows=False, db_path=db_path)
        if res is None:
            return pd.DataFrame()
        rows = res.get("rows", [])
        if not rows:
            return pd.DataFrame()
        df_scores = pd.DataFrame(rows)
        # ensure created_at exists
        if 'created_at' not in df_scores.columns:
            df_scores['created_at'] = int(time.time())
        # reorder columns
        cols_preferred = ["asset", "interval", "ts", "score", "range_min", "range_max", "stop", "target", "p_ml", "multiplier", "created_at", "ts_iso"]
        df_scores = df_scores[[c for c in cols_preferred if c in df_scores.columns]]
        return df_scores

    # original behavior: compute for all assets in data/config
    cfg_dir = Path("data") / "config"
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
    # cryptos
    for s in cryptos:
        try:
            res = process_symbol_interval(s, interval=config.get("crypto_interval", "1h"), config=config, save_rows=True, db_path=db_path)
            results[s] = res
            print(f"[score] processed {s}")
        except Exception as e:
            print(f"[score] error processing {s}: {e}")

    # stocks
    for s in stocks:
        try:
            res = process_symbol_interval(s, interval=config.get("stock_interval", "1h"), config=config, save_rows=True, db_path=db_path)
            results[s] = res
            print(f"[score] processed {s}")
        except Exception as e:
            print(f"[score] error processing {s}: {e}")

    return results
