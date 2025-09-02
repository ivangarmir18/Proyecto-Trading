# core/score.py
"""
Score module

- compute_scores(df, weights) -> DataFrame with normalized indicator columns + 'score' (0..1)
- compute_and_save_scores_for_asset(storage, asset, interval, cfg) -> calcula últimos scores y los guarda en la tabla `scores`
- utilities: normalization helpers, atr, ema fallback si no existen indicadores

Assumptions:
- Input candles DataFrame has columns: ts (pd.Timestamp UTC) open high low close volume
- If indicator columns (ema9, ema40, atr, macd, rsi, support, resistance) do NOT exist, the module computes basic versions:
    - ema9, ema40 using pandas ewm
    - atr simple using True Range & SMA
    - rsi basic implementation
- Scores are written to `scores` table in Postgres via storage.get_conn()
"""

from __future__ import annotations
import logging
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("core.score")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.getenv("SCORE_LOG_LEVEL", "INFO"))

# Default weights (same as tu memoria técnica, configurables en config.json)
DEFAULT_WEIGHTS = {
    "ema": 0.25,
    "support": 0.20,
    "atr": 0.15,
    "macd": 0.15,
    "rsi": 0.10,
    "fibonacci": 0.15,  # if fibonacci not present it's ignored
}

# ---------- Indicator fallback computations ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_ema_columns(df: pd.DataFrame):
    if "ema9" not in df.columns:
        df["ema9"] = ema(df["close"], 9)
    if "ema40" not in df.columns:
        df["ema40"] = ema(df["close"], 40)
    return df

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(period, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_ema_columns(df)
    if "atr" not in df.columns:
        df["atr"] = atr(df)
    if "rsi" not in df.columns:
        df["rsi"] = rsi(df["close"])
    # MACD (simple): ema12 - ema26
    if "macd" not in df.columns:
        ema12 = ema(df["close"], 12)
        ema26 = ema(df["close"], 26)
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    # supports/resistances: simple local min/max over window if not present
    if "support" not in df.columns:
        df["support"] = df["low"].rolling(window=20, min_periods=1).min()
    if "resistance" not in df.columns:
        df["resistance"] = df["high"].rolling(window=20, min_periods=1).max()

    return df

# ---------- Normalizers ----------
def normalize_between_0_1(series: pd.Series) -> pd.Series:
    # robust scaler to 0..1
    minv = series.min()
    maxv = series.max()
    if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
        return pd.Series(0.5, index=series.index)
    return (series - minv) / (maxv - minv)

def normalize_rsi_to_01(rsi_series: pd.Series) -> pd.Series:
    # rsi 0..100 -> 0..1
    return rsi_series.clip(0,100) / 100.0

# ---------- Score computation ----------
def compute_scores(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Returns a copy of df with new columns:
      - score (0..1)
      - score_components (dict-like columns optional)
    """
    weights = weights or DEFAULT_WEIGHTS.copy()
    df = df.copy()
    df = compute_basic_indicators(df)

    # Component normalizations
    # 1) Trend: price relative to ema9/ema40 (prefer price > ema40)
    df["trend_raw"] = 0.0
    price = df["close"]
    ema9 = df["ema9"]
    ema40 = df["ema40"]
    # simple rule: if price > ema40 => 1; if price < ema9 => 0; else linear between
    cond_up = price > ema40
    cond_down = price < ema9
    mid_mask = ~(cond_up | cond_down)
    df.loc[cond_up, "trend_raw"] = 1.0
    df.loc[cond_down, "trend_raw"] = 0.0
    # linear interpolation between ema9 and ema40 for mid
    denom = (ema40 - ema9).replace(0, np.nan)
    df.loc[mid_mask, "trend_raw"] = ((price - ema9) / denom).clip(0,1).fillna(0.5)

    # 2) proximity to support: distance / (3*ATR)
    df["dist_support"] = (price - df["support"]).abs()
    df["support_score_raw"] = (1 - (df["dist_support"] / (3 * df["atr"] + 1e-9))).clip(0,1)

    # 3) atr score (less atr = more stable) -> invert normalized atr (lower atr => higher score)
    df["atr_norm"] = normalize_between_0_1(df["atr"].fillna(0))
    df["atr_score_raw"] = 1.0 - df["atr_norm"]

    # 4) macd: normalize macd by absolute range
    df["macd_norm"] = normalize_between_0_1(df["macd"].fillna(0))
    df["macd_score_raw"] = df["macd_norm"]  # more macd -> stronger momentum

    # 5) rsi
    df["rsi_score_raw"] = normalize_rsi_to_01(df["rsi"].fillna(50))

    # 6) fibonacci: placeholder (if you have levels column you can implement)
    if "fibonacci" in df.columns:
        # assume fibonacci already encoded 0..1 proximity score
        df["fib_score_raw"] = normalize_between_0_1(df["fibonacci"].astype(float).fillna(0))
    else:
        df["fib_score_raw"] = 0.5

    # Compose weighted score (only keys present in weights)
    # map weights keys to component columns
    comp_map = {
        "ema": "trend_raw",
        "support": "support_score_raw",
        "atr": "atr_score_raw",
        "macd": "macd_score_raw",
        "rsi": "rsi_score_raw",
        "fibonacci": "fib_score_raw",
    }

    # Sum weighted components; make sure total weight sums to 1 (normalize if not)
    total_w = sum(weights.values()) if weights else 1.0
    if total_w <= 0:
        total_w = 1.0

    df["score"] = 0.0
    for k, w in weights.items():
        comp_col = comp_map.get(k)
        if comp_col and comp_col in df.columns:
            df["score"] += df[comp_col].fillna(0) * (w / total_w)
        else:
            # missing comp: ignore but log once
            logger.debug("Weight key %s has no corresponding component column", k)

    # clip and ensure numeric
    df["score"] = df["score"].clip(0,1).fillna(0.0)

    # Optionally compute range_min, range_max, stop, target using ATR and support/resistance
    atr_mult_stop = float(os.getenv("ATR_STOP_MULT", "1.3"))
    atr_mult_target = float(os.getenv("ATR_TARGET_MULT", "1.3"))
    df["stop"] = df["support"] - atr_mult_stop * df["atr"]
    df["target"] = df["resistance"] + atr_mult_target * df["atr"]
    # range_min = support or open; range_max = price + atr*0.5
    df["range_min"] = df["support"]
    df["range_max"] = df["close"] + 0.5 * df["atr"]

    return df

# ---------- Save scores to Postgres ----------

def save_scores_to_db(storage, df_scores: pd.DataFrame, asset: str, interval: str) -> int:
    """
    Guarda las filas relevantes de df_scores en la tabla `scores`.
    df_scores debe contener columnas: ts (pd.Timestamp), score, range_min, range_max, stop, target
    Devuelve número de filas insertadas/upserted.
    """
    if df_scores is None or df_scores.empty:
        logger.debug("save_scores_to_db: vacío, nada que guardar")
        return 0

    df = df_scores.copy()
    # ensure ts_ms int
    if pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts_ms"] = (df["ts"].astype("int64") // 1_000_000).astype("int64")
    else:
        df["ts_ms"] = df["ts"].astype("int64")
    rows = []
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    for _, r in df.iterrows():
        rows.append((
            asset,
            interval,
            int(r["ts_ms"]),
            float(r.get("score", 0.0)),
            float(r.get("range_min")) if not pd.isna(r.get("range_min")) else None,
            float(r.get("range_max")) if not pd.isna(r.get("range_max")) else None,
            float(r.get("stop")) if not pd.isna(r.get("stop")) else None,
            float(r.get("target")) if not pd.isna(r.get("target")) else None,
            int(now_ms),
        ))

    upsert_sql = """
    INSERT INTO scores (asset, interval, ts, score, range_min, range_max, stop, target, created_at)
    VALUES %s
    ON CONFLICT (asset, interval, ts) DO UPDATE
      SET score = EXCLUDED.score,
          range_min = EXCLUDED.range_min,
          range_max = EXCLUDED.range_max,
          stop = EXCLUDED.stop,
          target = EXCLUDED.target,
          created_at = EXCLUDED.created_at;
    """

    inserted = 0
    try:
        with storage.get_conn() as conn:
            with conn.cursor() as cur:
                import psycopg2.extras as pe
                pe.execute_values(cur, upsert_sql, rows, template=None, page_size=200)
                inserted = cur.rowcount if cur.rowcount is not None else len(rows)
        logger.info("save_scores_to_db: insertadas %d filas para %s %s", inserted, asset, interval)
    except Exception:
        logger.exception("Error guardando scores para %s %s", asset, interval)
    return inserted

# ---------- High-level helper -----------
def compute_and_save_scores_for_asset(storage, asset: str, interval: str, lookback_bars: int = 500, weights: Optional[Dict[str,float]] = None):
    """
    High level: obtiene candles desde storage, calcula indicadores y score, guarda en DB.
    - lookback_bars: cuántas velas leer para calcular indicadores y scores
    """
    # obtener datos
    # End using storage.get_ohlcv (returns ts pd.Timestamp)
    df = storage.get_ohlcv(asset, interval, limit=lookback_bars)
    if df is None or df.empty:
        logger.warning("No hay velas para %s %s", asset, interval)
        return 0
    df_scored = compute_scores(df, weights=weights)
    # guardamos solo las últimas N filas que contienen score calculado
    inserted = save_scores_to_db(storage, df_scored[["ts", "score", "range_min", "range_max", "stop", "target"]], asset, interval)
    return inserted
