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
import os
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
# Helper to load external config if present
# -------------------------
def load_score_config_file() -> Optional[Dict[str, Any]]:
    """
    Intenta cargar 'config/score_config.json' relativo al repo. Devuelve None si no existe.
    """
    try:
        base = os.path.dirname(__file__)
        cfg_path = os.path.join(base, "..", "config", "score_config.json")
        cfg_path = os.path.normpath(cfg_path)
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path, "r", encoding="utf8") as fh:
                return json.load(fh)
    except Exception:
        logger.exception("load_score_config_file failed")
    return None


# -------------------------
# Default configuration
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
# Low-level helpers
# -------------------------
def _sigmoid(x: pd.Series) -> pd.Series:
    """Sigmoid numerically stable for pandas Series."""
    x_clipped = x.clip(-50, 50)
    return 1 / (1 + np.exp(-x_clipped))


def _zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    if window is None:
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            std = 1.0
        return (series - mean) / std
    else:
        roll_mean = series.rolling(window, min_periods=1).mean()
        roll_std = series.rolling(window, min_periods=1).std(ddof=0).replace(0, 1.0)
        return (series - roll_mean) / roll_std


def _minmax(series: pd.Series, vmin: float, vmax: float) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    denom = vmax - vmin
    if denom == 0:
        return pd.Series(0.5, index=series.index)
    return (series - vmin) / denom


def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=1.0).fillna(0.0)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.replace(0, np.nan)
    out = a / b2
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _aggregate_weighted(components_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Combina columnas de components_df (column = component name) usando weights (ya normalizados).
    Retorna serie 0..1
    """
    if components_df is None or components_df.empty:
        return pd.Series(dtype=float)
    wsum = sum(weights.values()) if weights else 0.0
    if wsum == 0:
        return components_df.mean(axis=1).fillna(0.0)
    # ensure columns exist for all keys
    cols = list(weights.keys())
    missing = [c for c in cols if c not in components_df.columns]
    for m in missing:
        components_df[m] = 0.0
    out = pd.Series(0.0, index=components_df.index)
    for k, wt in weights.items():
        out = out + components_df[k].fillna(0.0) * wt
    return _clip01(out)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normaliza dict de pesos para que sumen 1. Si suma 0 o negativos -> distribuye igualmente.
    """
    if not weights:
        return {}
    cleaned = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        n = len(cleaned)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


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
        ema_short = df.get(f"ema_{short}") if f"ema_{short}" in df.columns else df["close"].ewm(span=short, adjust=False).mean()
        ema_long = df.get(f"ema_{long}") if f"ema_{long}" in df.columns else df["close"].ewm(span=long, adjust=False).mean()
        series = _safe_div((ema_short - ema_long), df["close"].replace(0, np.nan))
        return series.fillna(0.0)

    if t == "rsi":
        period = params.get("window", 14)
        if f"rsi_{period}" in df.columns:
            return df[f"rsi_{period}"]
        # fallback: compute approx RSI if close exists
        if "close" in df.columns:
            delta = df["close"].diff()
            up = delta.clip(lower=0).rolling(period, min_periods=1).mean()
            down = -delta.clip(upper=0).rolling(period, min_periods=1).mean().replace(0, np.nan)
            rs = up / down.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50.0)
        return pd.Series(np.nan, index=df.index)

    if t == "atr":
        window = int(params.get("window", 14))
        high = df.get("high")
        low = df.get("low")
        close = df.get("close")
        if high is None or low is None or close is None:
            return pd.Series(np.nan, index=df.index)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=window, adjust=False).mean()
        return atr.fillna(0.0)

    if t == "support_resistance":
        # expects columns support, resistance or computed by indicators module
        if "support" in df.columns and "resistance" in df.columns:
            # signal: distance to nearest level normalized by close
            close = df["close"].replace(0, np.nan)
            dist = pd.concat([(close - df["support"]).abs(), (close - df["resistance"]).abs()], axis=1).min(axis=1)
            out = _safe_div(1.0, dist.replace(0, np.nan))
            return out.fillna(0.0)
        return pd.Series(np.nan, index=df.index)

    if t == "volume_spike":
        w = int(params.get("window", 20))
        if "volume" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        v = df["volume"].astype(float)
        ma = v.rolling(w, min_periods=1).mean().replace(0, 1)
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


def _normalize_series(s: pd.Series, cfg_norm: Optional[Dict[str, Any]]) -> pd.Series:
    """
    Normaliza una serie segun cfg_norm:
     - tipo 'minmax' -> requiere min/max
     - tipo 'zscore' -> ventana
     - si cfg_norm es None -> se aplica _clip01 al resultado si posible
    """
    if cfg_norm is None:
        return _clip01(s.fillna(0.0))
    t = cfg_norm.get("type", "minmax")
    if t == "minmax":
        vmin = float(cfg_norm.get("min", float(s.min()) if not s.empty else 0.0))
        vmax = float(cfg_norm.get("max", float(s.max()) if not s.empty else 1.0))
        return _clip01(_minmax(s, vmin, vmax))
    if t == "zscore":
        w = int(cfg_norm.get("window", 200))
        z = _zscore(s, w)
        clip = cfg_norm.get("clip", None)
        if clip and isinstance(clip, (list, tuple)) and len(clip) == 2:
            return _clip01((z - clip[0]) / max(1e-9, (clip[1] - clip[0])))
        return _clip01(z)
    # fallback
    return _clip01(s.fillna(0.0))


# -------------------------
# Score calculation entrypoints
# -------------------------
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
        # normalization for component (cfg can include norm details)
        norm_cfg = comp_local.get("norm") or comp_local.get("norm_cfg") or comp_local.get("normalize")
        if isinstance(norm_cfg, dict):
            s_norm = _normalize_series(series, norm_cfg)
        else:
            # default minmax normalization if possible
            s_norm = _normalize_series(series, comp_local.get("norm", None))
        comp_vals[name] = s_norm.fillna(0.0)

    components_df = pd.DataFrame(comp_vals, index=df.index)

    # weights: take weights_override if provided else from cfg
    base_weights = {name: float(components_cfg[name].get("weight", 0.0)) for name in components_cfg.keys()}

    # allow loading score_config from config/score_config.json if score_config param was None
    if score_config is None:
        file_cfg = load_score_config_file()
        if file_cfg and isinstance(file_cfg, dict):
            # merge file config into cfg for components (but keep explicit score_config param priority)
            file_comps = file_cfg.get("components", {})
            # do not overwrite existing component definitions, only fill missing weight or params
            for k, v in file_comps.items():
                if k in components_cfg:
                    if "weight" in v and ("weight" not in components_cfg[k] or components_cfg[k].get("weight", 0) == 0):
                        components_cfg[k]["weight"] = float(v.get("weight", components_cfg[k].get("weight", 0)))
                else:
                    components_cfg[k] = v

    # Prepare alias mapping for overrides: map possible external keys to component names
    alias_map = {}
    for cname, cdef in components_cfg.items():
        alias_map[cname] = cname
        if isinstance(cdef, dict):
            for field in ("source", "expr", "name", "type"):
                v = cdef.get(field) if isinstance(cdef.get(field), str) else None
                if v:
                    alias_map[v] = cname

    if weights_override:
        merged = base_weights.copy()
        for k, v in weights_override.items():
            # accept only overrides that map to known components (via alias_map)
            if k in merged:
                merged[k] = float(v)
            elif k in alias_map:
                mapped = alias_map[k]
                merged[mapped] = float(v)
            else:
                # unknown override key: ignore and log for debugging
                logger.debug("Ignoring unknown weight override key: %s", k)
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

    out = pd.DataFrame({"ts": df["ts"].astype(int) if "ts" in df.columns else np.arange(len(df)), "score": score_series})
    # attach component columns for details
    for col in components_df.columns:
        out[col] = components_df[col].values

    # attach weights metadata (readonly)
    out.attrs = {"weights": weights}
    return out


def compute_latest_score(df: pd.DataFrame, score_config: Optional[Dict[str, Any]] = None, weights_override: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """
    Compute and return the last score entry as dict {ts, score, components...} or None.
    """
    out_df = compute_score_timeseries(df, score_config, weights_override)
    if out_df is None or out_df.empty:
        return None
    last = out_df.iloc[-1].to_dict()
    # merge components into flat dict (exclude ts)
    res = {"ts": last.get("ts"), "score": float(last.get("score", 0.0))}
    comps = {k: float(v) for k, v in last.items() if k not in ("ts", "score")}
    res.update(comps)
    return res


# -------------------------
# Persistence helpers (safe, fallback)
# -------------------------
def _persist_score_fallback(storage_module: Any, asset: str, ts: int, score: float, components: Dict[str, Any], method: str = "weighted"):
    """
    Guarda score usando las APIs disponibles del storage_module.
    Intenta: save_score / save_scores / upsert_scores / save_setting como fallback.
    """
    try:
        payload = {"asset": asset, "ts": int(ts), "score": float(score), "components": components, "method": method}
    except Exception:
        payload = {"asset": asset, "ts": int(ts), "score": float(score), "components": components, "method": method}

    try:
        if hasattr(storage_module, "save_score"):
            try:
                return storage_module.save_score(payload)
            except Exception:
                logger.debug("storage_module.save_score failed", exc_info=True)
        if hasattr(storage_module, "save_scores"):
            try:
                return storage_module.save_scores([payload])
            except Exception:
                logger.debug("storage_module.save_scores failed", exc_info=True)
        if hasattr(storage_module, "upsert_scores"):
            try:
                return storage_module.upsert_scores([payload])
            except Exception:
                logger.debug("storage_module.upsert_scores failed", exc_info=True)
        # fallback to generic save_setting
        if hasattr(storage_module, "save_setting"):
            try:
                return storage_module.save_setting(f"score_last_{asset}", payload)
            except Exception:
                logger.debug("storage_module.save_setting failed", exc_info=True)
    except Exception:
        logger.exception("Error persisting score fallback")
    return False


def compute_and_persist_scores(df: pd.DataFrame, asset: str, interval: str, storage_module: Any = None, score_config: Optional[Dict[str, Any]] = None, weights_override: Optional[Dict[str, float]] = None):
    """
    Calcula scores y persiste los resultados (si storage_module está disponible).
    Devuelve el DataFrame de scores.
    """
    out_df = compute_score_timeseries(df, score_config, weights_override)
    try:
        if storage_module:
            # intenta métodos comunes de persistencia
            if hasattr(storage_module, "save_scores"):
                try:
                    storage_module.save_scores(out_df.to_dict("records"))
                except Exception:
                    logger.exception("storage_module.save_scores failed")
            elif hasattr(storage_module, "persist_scores"):
                try:
                    storage_module.persist_scores(out_df, asset=asset, interval=interval)
                except Exception:
                    logger.exception("storage_module.persist_scores failed")
            else:
                # fallback: use helper
                last = out_df.iloc[-1].to_dict() if not out_df.empty else None
                if last:
                    _persist_score_fallback(storage_module, asset, last.get("ts"), last.get("score"), {k: last[k] for k in out_df.columns if k not in ("ts", "score")})
    except Exception:
        logger.exception("persist score fallback failed")
    return out_df


# -------------------------
# ATR / entry/stop/target helpers
# -------------------------
def compute_atr_last(df: pd.DataFrame, window: int = 14) -> Optional[float]:
    if df is None or df.empty:
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
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
    if f"ema_{short_span}" in df.columns and f"ema_{long_span}" in df.columns:
        s = df[f"ema_{short_span}"].iloc[-1]
        l = df[f"ema_{long_span}"].iloc[-1]
    else:
        close = df[close_col].astype(float)
        s = close.ewm(span=short_span, adjust=False).mean().iloc[-1]
        l = close.ewm(span=long_span, adjust=False).mean().iloc[-1]
    if s > l:
        return "long"
    if s < l:
        return "short"
    return "neutral"


# -------------------------
# High level compute_score_safe & AIScoringSystem
# -------------------------
class AIScoringSystem:
    """
    Wrapper to call AI train/infer modules if available.
    Expect a module core.ai_inference or core.ai_interference exposing `predict_scores` or `infer`.
    """

    def __init__(self, storage=None, config: Optional[Dict[str, Any]] = None, ai_module: Any = None):
        self.storage = storage
        self.config = config or {}
        import importlib
        self.ai_module = ai_module
        if self.ai_module is None:
            try:
                self.ai_module = importlib.import_module("core.ai_inference")
            except Exception:
                try:
                    self.ai_module = importlib.import_module("core.ai_interference")
                except Exception:
                    self.ai_module = None

    def predict(self, df: pd.DataFrame, asset: Optional[str] = None) -> pd.DataFrame:
        """
        Devuelve DataFrame con columna 'ai_score' alineada al índice del df.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["ai_score"])
        if not self.ai_module:
            return pd.DataFrame({"ai_score": pd.Series(np.nan, index=df.index)})
        try:
            if hasattr(self.ai_module, "predict_scores"):
                out = self.ai_module.predict_scores(df, asset=asset, config=self.config)
                # allow series or dataframe
                if isinstance(out, pd.Series):
                    return pd.DataFrame({"ai_score": out})
                if isinstance(out, pd.DataFrame):
                    if "ai_score" in out.columns:
                        return out[["ai_score"]]
                    # try first numeric column
                    for c in out.columns:
                        if np.issubdtype(out[c].dtype, np.number):
                            return pd.DataFrame({"ai_score": out[c]})
                # fallback attempt: convert to series
                ser = pd.Series(out, index=df.index)
                return pd.DataFrame({"ai_score": ser})
            elif hasattr(self.ai_module, "infer"):
                out = self.ai_module.infer(df, asset=asset, config=self.config)
                ser = pd.Series(out, index=df.index)
                return pd.DataFrame({"ai_score": ser})
        except Exception:
            logger.exception("AIScoringSystem.predict failed; returning empty")
        return pd.DataFrame({"ai_score": pd.Series(np.nan, index=df.index)})


# -------------------------
# Utility to compute stop/target by ATR
# -------------------------
def compute_stop_target_for_asset(df: pd.DataFrame, atr_window: int = 14, stop_mult: float = 1.3, target_mult: float = 2.6) -> Optional[Dict[str, float]]:
    """
    Given a dataframe with standard OHLCV, compute TR/ATR and propose stop/target (last candle).
    Returns {atr, stop, target, entry}
    """
    try:
        atr = compute_atr_last(df, atr_window)
        if atr is None:
            return None
        last_close = float(df["close"].iloc[-1])
        stop = last_close - stop_mult * atr
        target = last_close + target_mult * atr
        return {"atr": atr, "stop": float(stop), "target": float(target), "entry": float(last_close)}
    except Exception:
        logger.exception("compute_stop_target_for_asset failed for given df")
        return None
