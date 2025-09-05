# core/score.py
"""
Módulo de scoring técnico y utilidades relacionadas.

Contiene:
- funciones de normalización y helpers estadísticos
- creación de una configuración por defecto para el score
- cálculo de series de score (compute_score_timeseries)
- funciones para persistir scores de forma segura (compute_and_persist_scores)
- utilidades ATR / entry/stop/target
- AIScoringSystem: wrapper que usa compute_score_safe
"""

from typing import Optional, Dict, Any, List, Union, Tuple
import math
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger("core.score")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


# -------------------------
# Low-level helpers
# -------------------------
def _sigmoid(x: pd.Series) -> pd.Series:
    """Sigmoid numerically stable for pandas Series."""
    # clip to avoid overflow
    x_clipped = x.clip(-50, 50)
    return 1 / (1 + np.exp(-x_clipped))


def _zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Z-score: (x - mean) / std over rolling window if provided, else global.
    Returns zeros for constant series.
    """
    if series is None or series.empty:
        return series
    if window and window > 1:
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std(ddof=0).replace(0, np.nan).fillna(1.0)
    else:
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            std = 1.0
        mean = pd.Series(mean, index=series.index)
        std = pd.Series(std, index=series.index)
    return (series - mean) / std


def _minmax(series: pd.Series, vmin: float, vmax: float) -> pd.Series:
    if series is None or series.empty:
        return series
    denom = vmax - vmin
    if denom == 0:
        return pd.Series(0.5, index=series.index)
    return (series - vmin) / denom


def _clip01(series: pd.Series, lo: float = 0.0, hi: float = 1.0) -> pd.Series:
    if series is None or series.empty:
        return series
    return series.clip(lower=lo, upper=hi)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Safe division: returns 0 where denominator is zero/NaN.
    """
    a = a.fillna(0)
    b = b.fillna(0)
    out = pd.Series(index=a.index, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = a / b
    out = out.replace([np.inf, -np.inf], 0).fillna(0.0)
    return out


# -------------------------
# Configuration dataclasses / helpers
# -------------------------
def make_default_score_config() -> Dict[str, Any]:
    """
    Devuelve una configuración de score por defecto. La estructura es simple y extensible.
    """
    return {
        "components": {
            "trend": {"type": "ema_diff", "params": {"short": 9, "long": 40}, "weight": 0.25},
            "support": {"type": "support_resistance", "params": {}, "weight": 0.20},
            "momentum": {"type": "rsi", "params": {"window": 14}, "weight": 0.15},
            "volatility": {"type": "atr", "params": {"window": 14}, "weight": 0.10},
            "volume": {"type": "volume_spike", "params": {"window": 20}, "weight": 0.10},
            "ai_confidence": {"type": "ai", "params": {}, "weight": 0.20}
        },
        "normalize": True,
        "score_scale": {"min": 0.0, "max": 1.0},
        "atr_multiplier_stop": 1.3,
        "atr_multiplier_target": 2.6
    }


# -------------------------
# Component extraction helpers
# -------------------------
def _get_series_from_cfg(df: pd.DataFrame, comp: Dict[str, Any]) -> pd.Series:
    """
    Extrae una serie desde df según la configuración de componente.
    comp.type puede ser:
      - 'ema_diff' -> requiere columnas close, calcula (ema_short - ema_long)/close
      - 'rsi' -> requiere columna 'rsi' si ya calculada; fallbacks a NaN
      - 'atr' -> requiere columnas high/low/close
      - 'support_resistance' -> requiere columnas support/resistance si existen
      - 'volume_spike' -> usa volume
      - 'ai' -> uses column 'ai_score' if present
    """
    t = comp.get("type")
    params = comp.get("params", {}) or {}

    if df is None or df.empty:
        return pd.Series(dtype=float)

    if t == "ema_diff":
        s = comp.get("params", {})
        short = s.get("short", 9)
        long = s.get("long", 40)
        # fallback compute if not present
        ema_short = df.get(f"ema_{short}") if f"ema_{short}" in df.columns else df["close"].ewm(span=short, adjust=False).mean()
        ema_long = df.get(f"ema_{long}") if f"ema_{long}" in df.columns else df["close"].ewm(span=long, adjust=False).mean()
        series = _safe_div((ema_short - ema_long), df["close"].replace(0, np.nan))
        return series.fillna(0.0)

    if t == "rsi":
        col = comp.get("params", {}).get("column", "rsi")
        if col in df.columns:
            # Higher RSI normally means overbought -> depending on mapping we might invert
            return df[col]
        return pd.Series(np.nan, index=df.index)

    if t == "atr":
        # compute ATR if not present
        if "atr" in df.columns:
            return df["atr"]
        # fallback compute: simple high-low rolling mean
        return (df["high"] - df["low"]).rolling(params.get("window", 14), min_periods=1).mean().fillna(0.0)

    if t == "volume_spike":
        v = df["volume"].fillna(0)
        ma = v.rolling(params.get("window", 20), min_periods=1).mean().replace(0, 1)
        return _safe_div(v, ma)

    if t == "ai":
        # Expect column ai_score between 0..1
        if "ai_score" in df.columns:
            return df["ai_score"]
        return pd.Series(np.nan, index=df.index)

    # default: try to find column with same name
    key = comp.get("name")
    if key and key in df.columns:
        return df[key]
    return pd.Series(np.nan, index=df.index)


def _normalize_series(s: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    """
    Normaliza una serie según cfg.norm (minmax/zscore/sigmoid) y parámetros.
    """
    norm_type = cfg.get("norm", "minmax")
    if s is None or s.empty:
        return s
    if norm_type == "zscore":
        window = cfg.get("window")
        return _clip01((_zscore(s, window) + 3) / 6.0)  # map typical zscore range to 0..1
    if norm_type == "sigmoid":
        return _sigmoid(s)
    # default minmax
    vmin = cfg.get("vmin", float(s.min()) if not s.empty else 0.0)
    vmax = cfg.get("vmax", float(s.max()) if not s.empty else 1.0)
    if vmax == vmin:
        return pd.Series(0.5, index=s.index)
    return _clip01(_minmax(s, vmin, vmax))


def _aggregate_weighted(components_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Combina columnas de components_df (column = component name) usando weights (ya normalizados).
    Retorna serie 0..1
    """
    if components_df is None or components_df.empty:
        return pd.Series(dtype=float)
    wsum = sum(weights.values()) if weights else 0.0
    if wsum == 0:
        # average equally
        return components_df.mean(axis=1).fillna(0.0)
    # ensure order and missing columns replaced by 0
    cols = list(weights.keys())
    missing = [c for c in cols if c not in components_df.columns]
    for m in missing:
        components_df[m] = 0.0
    # multiply and sum
    out = pd.Series(0.0, index=components_df.index)
    for k, wt in weights.items():
        out = out + components_df[k].fillna(0.0) * wt
    return _clip01(out)


# -------------------------
# Score calculation entrypoints
# -------------------------
def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normaliza dict de pesos para que sumen 1. Si suma 0 o negativos -> distribuye igualmente.
    """
    if not weights:
        return {}
    # ensure non-negative
    cleaned = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        n = len(cleaned)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def compute_score_timeseries(df: pd.DataFrame, score_config: Optional[Dict[str, Any]] = None, weights_override: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Compute a timeseries of scores from a dataframe of candles+indicators.

    Returns DataFrame with columns: ts, score, and component columns for details.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "score"])
    cfg = score_config or make_default_score_config()
    components_cfg = cfg.get("components", {})
    # assemble components into dataframe
    comp_vals = {}
    # compute each component series
    for name, comp in components_cfg.items():
        comp_local = dict(comp)  # copy
        comp_local['name'] = name
        series = _get_series_from_cfg(df, comp_local)
        # normalization per component (cfg can include norm details)
        norm_cfg = comp_local.get("norm_cfg", {"norm": "minmax"})
        s_norm = _normalize_series(series, norm_cfg)
        comp_vals[name] = s_norm.fillna(0.0)

    components_df = pd.DataFrame(comp_vals, index=df.index)
    # weights: take weights_override if provided else from cfg
    base_weights = {name: float(components_cfg[name].get("weight", 0.0)) for name in components_cfg.keys()}
    if weights_override:
        # overlay overrides
        merged = base_weights.copy()
        for k, v in weights_override.items():
            if k in merged:
                merged[k] = float(v)
            else:
                # allow adding new weight keys (but they will map to zero comps if absent)
                merged[k] = float(v)
        weights = normalize_weights(merged)
    else:
        weights = normalize_weights(base_weights)
    # aggregate weighted
    score_series = _aggregate_weighted(components_df, weights)
    # apply global scaling if needed
    smin = cfg.get("score_scale", {}).get("min", 0.0)
    smax = cfg.get("score_scale", {}).get("max", 1.0)
    if smin != 0.0 or smax != 1.0:
        score_series = smin + (smax - smin) * score_series
    out = pd.DataFrame({"ts": df["ts"].astype(int), "score": score_series})
    # attach component columns for details
    for col in components_df.columns:
        out[col] = components_df[col].values
    return out


def compute_and_persist_scores(df: pd.DataFrame, score_config: Optional[Dict[str, Any]] = None, weights_override: Optional[Dict[str, float]] = None, storage: Any = None, asset: Optional[str] = None, interval: Optional[str] = None):
    """
    Computes scores and persists them to storage if provided.
    storage is expected to have either `save_scores(asset, interval, rows)` or `make_save_callback()`.
    rows format: list[dict(ts, score, components...)]
    """
    series_df = compute_score_timeseries(df, score_config, weights_override)
    rows = series_df.to_dict("records")
    if storage is not None:
        # attempt sensible persist methods
        try:
            if hasattr(storage, "save_scores"):
                storage.save_scores(asset, interval, rows)
            elif hasattr(storage, "make_save_callback"):
                cb = storage.make_save_callback()
                cb(asset, interval, rows)
            elif hasattr(storage, "upsert_scores"):
                storage.upsert_scores(asset, interval, rows)
            else:
                # fallback: try generic method 'upsert_candles' is NOT appropriate
                logger.info("No persistence method found in storage for scores")
        except Exception as e:
            logger.exception("Error persisting scores: %s", e)
    return series_df


def compute_latest_score(df: pd.DataFrame, score_config: Optional[Dict[str, Any]] = None, weights_override: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """
    Compute and return the last score entry as dict {ts, score, components...} or None.
    """
    out_df = compute_score_timeseries(df, score_config, weights_override)
    if out_df is None or out_df.empty:
        return None
    last = out_df.iloc[-1].to_dict()
    # ensure ts int, score float
    last["ts"] = int(last.get("ts", 0))
    last["score"] = float(last.get("score", 0.0))
    return last


# -------------------------
# ATR, entry/stop/target helpers
# -------------------------
def compute_atr(df: pd.DataFrame, window: int = 14, high_col="high", low_col="low", close_col="close") -> Optional[float]:
    """
    Compute ATR over given dataframe. Returns single float (last ATR) or None.
    """
    if df is None or df.empty:
        return None
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else None


def infer_direction_by_ema(df: pd.DataFrame, short_span: int = 9, long_span: int = 40, close_col="close") -> str:
    """
    Infer direction: 'long' if short EMA > long EMA, 'short' if opposite, else 'neutral'
    """
    if df is None or df.empty:
        return "neutral"
    close = df[close_col].astype(float)
    ema_s = close.ewm(span=short_span, adjust=False).mean().iloc[-1]
    ema_l = close.ewm(span=long_span, adjust=False).mean().iloc[-1]
    if ema_s > ema_l:
        return "long"
    if ema_s < ema_l:
        return "short"
    return "neutral"


def compute_entry_stop_target(df: pd.DataFrame, atr: Optional[float] = None, atr_multiplier_stop: float = 1.3, atr_multiplier_target: float = 2.6, side: str = "long"):
    """
    Computes entry, stop and target prices given the last close and ATR.
    Returns (entry, stop, target) — if atr is None returns (None,None,None).
    Entry is last close, stop = entry - atr*mult_stop for long (opposite for short).
    """
    if df is None or df.empty:
        return (None, None, None)
    last_close = float(df["close"].iloc[-1])
    if atr is None:
        atr = compute_atr(df)
        if atr is None:
            return (last_close, None, None)
    if side == "long":
        stop = last_close - atr * atr_multiplier_stop
        target = last_close + atr * atr_multiplier_target
    else:
        stop = last_close + atr * atr_multiplier_stop
        target = last_close - atr * atr_multiplier_target
    return (last_close, float(stop), float(target))


def compute_stop_target_for_asset(adapter_obj, asset: str, interval: str = "1h", lookback: int = 200, atr_window: int = 14, atr_multiplier_stop: float = 1.3, atr_multiplier_target: float = 2.6):
    """
    High-level helper to compute entry/stop/target using the storage adapter.
    adapter_obj should implement `get_ohlcv` or `load_candles`.
    """
    try:
        if hasattr(adapter_obj, "get_ohlcv"):
            df = adapter_obj.get_ohlcv(asset, interval, 0, int(pd.Timestamp.now().timestamp() * 1000))
        elif hasattr(adapter_obj, "load_candles"):
            rows = adapter_obj.load_candles(asset, interval, None, None)
            df = pd.DataFrame(rows) if rows else pd.DataFrame()
        else:
            raise RuntimeError("adapter_obj no expone get_ohlcv/load_candles")
        if df is None or df.empty:
            return None
        # ensure types and last N
        if "ts" in df.columns:
            df = df.sort_values("ts").reset_index(drop=True)
        if lookback:
            df = df.tail(lookback)
        atr = compute_atr(df, window=atr_window)
        side = infer_direction_by_ema(df)
        return compute_entry_stop_target(df, atr, atr_multiplier_stop, atr_multiplier_target, side)
    except Exception:
        logger.exception("compute_stop_target_for_asset failed for %s %s", asset, interval)
        return None


# -------------------------
# Persistence fallback
# -------------------------
def _persist_score_fallback(storage, asset: str, ts: int, score_val: float, components: dict, method: str = None):
    """
    Fallback to persist a single score point. Tries multiple storage API names.
    """
    try:
        row = {"ts": int(ts), "score": float(score_val), "components": components}
        if hasattr(storage, "save_score"):
            storage.save_score(asset, row)
            return True
        if hasattr(storage, "save_scores"):
            storage.save_scores(asset, [row])
            return True
        if hasattr(storage, "make_save_callback"):
            cb = storage.make_save_callback()
            cb(asset, None, [row])
            return True
        # else try generic
        return False
    except Exception:
        logger.exception("persist score fallback failed")
        return False


# -------------------------
# High level compute_score_safe & AIScoringSystem
# -------------------------
def compute_score_safe(df, score_config: Optional[Dict[str, Any]] = None, weights_override: Optional[Dict[str, float]] = None):
    """
    Safe wrapper to compute scores given a dataframe of candles and indicators.
    Returns a list of dicts: [{'ts':..., 'score':..., 'component1':..., ...}, ...]
    """
    try:
        df_in = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception:
        df_in = pd.DataFrame()

    if df_in is None or df_in.empty:
        return []

    cfg = score_config or make_default_score_config()
    out_df = compute_score_timeseries(df_in, cfg, weights_override)
    # convert to list of dicts
    records = out_df.to_dict("records")
    # normalize types
    for r in records:
        r["ts"] = int(r.get("ts", 0))
        r["score"] = float(r.get("score", 0.0))
    return records


class AIScoringSystem:
    """
    Wrapper simple para la parte IA que produce scores/predicciones.
    Interfaz:
      s = AIScoringSystem(storage, config)
      predictions = s.predict(df)
    """
    def __init__(self, storage=None, config: Optional[Dict[str, Any]] = None):
        self.storage = storage
        self.config = config or {}
        # if there's compute_score_safe in module, use it (it is)
        self._compute = compute_score_safe

    def predict(self, df: Union[pd.DataFrame, List[dict], None], weights_override: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        if df is None:
            return []
        try:
            # If df is a DataFrame, pass through compute_score_safe
            return self._compute(df, self.config.get("score_config", None), weights_override)
        except Exception:
            logger.exception("AIScoringSystem.predict failed; returning empty")
            return []


# -------------------------
# Convenience: compute_and_persist_last_score
# -------------------------
def compute_and_persist_last_score(asset: str, df, score_config: Optional[Dict[str, Any]] = None, storage_module=None, weights_override: Optional[Dict[str, float]] = None):
    """
    Compute latest score and persist via storage_module (if available).
    Returns the persisted last score dict or None.
    """
    recs = compute_score_safe(df, score_config, weights_override)
    if not recs:
        return None
    last = recs[-1]
    ts = last.get("ts")
    score_val = last.get("score")
    components = {k: v for k, v in last.items() if k not in ("ts", "score")}
    if storage_module:
        try:
            ok = _persist_score_fallback(storage_module, asset, ts, score_val, components)
            if not ok:
                logger.info("No persistence method available for scores")
        except Exception:
            logger.exception("Error persisting last score")
    return last
