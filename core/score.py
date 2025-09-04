# core/score.py
"""
Cálculo y persistencia del SCORE de un activo.

Diseño:
- Entrada: DataFrame OHLCV + columnas de indicadores (producidas por core.indicators.apply_indicators).
- Configurable por componentes: cada componente mapea 1 columna o expresión a [0,1].
- Normalizaciones disponibles: zscore->sigmoid, minmax, clip, escalado e inversión.
- Agregación: weighted (por defecto). Preparado para añadir otras (rank, vote).
- *Blend* opcional con un score alternativo (IA) por columna o serie externa.
- Persistencia: guarda un histórico de scores en Postgres con upsert por ts.

Ejemplo mínimo:
    from core.score import compute_and_persist_scores, make_default_score_config
    cfg = make_default_score_config()
    compute_and_persist_scores("BTCUSDT", df_ind, cfg, storage=storage, method="weighted")

Estructura de config (ejemplo):
{
  "method": "weighted",
  "blend": {"enabled": True, "alpha": 0.3, "alt_col": "ai_score"},  # opcional
  "components": {
    "rsi": {
      "source": "rsi_14",               # columna o 'expr' (pandas eval)
      "norm": {"type": "minmax", "min": 0, "max": 100},  # mapea 0..100 -> 0..1
      "weight": 0.3
    },
    "trend": {
      "expr": "(close / ema_50) - 1",   # expresión con columnas del DF
      "norm": {"type": "zscore", "window": 200, "clip": [0,1]}, # zscore->sigmoid->clip
      "weight": 0.4
    },
    "macd": {
      "source": "macd_hist",
      "norm": {"type": "zscore", "window": 200, "clip": [0,1]},
      "weight": 0.3
    }
  }
}
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger("score")


# ==========================
# Helpers matemáticos
# ==========================

def _sigmoid(x: pd.Series) -> pd.Series:
    # numéricamente estable
    return 1.0 / (1.0 + pd.Series.map(-x, lambda v: math.exp(v) if pd.notna(v) else float("nan")))

def _zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    if window and window > 1:
        mean = series.rolling(window=window, min_periods=max(5, window // 5)).mean()
        std = series.rolling(window=window, min_periods=max(5, window // 5)).std(ddof=0)
    else:
        mean = series.expanding(min_periods=10).mean()
        std = series.expanding(min_periods=10).std(ddof=0)
    z = (series - mean) / std.replace(0, pd.NA)
    return z

def _minmax(series: pd.Series, vmin: float, vmax: float) -> pd.Series:
    rng = (vmax - vmin) if (vmax is not None and vmin is not None) else None
    if rng is None or rng == 0:
        return pd.Series(index=series.index, dtype=float)
    return (series - vmin) / rng

def _clip01(series: pd.Series, lo: float = 0.0, hi: float = 1.0) -> pd.Series:
    return series.clip(lower=lo, upper=hi)

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, pd.NA)


# ==========================
# Config dataclasses
# ==========================

@dataclass
class NormCfg:
    type: str = "zscore"             # "zscore" | "minmax" | "none"
    window: Optional[int] = None     # zscore rolling window
    min: Optional[float] = None      # minmax
    max: Optional[float] = None      # minmax
    clip: Optional[Tuple[float, float]] = (0.0, 1.0)  # clip final
    invert: bool = False             # invierte (1 - x) al final

@dataclass
class ComponentCfg:
    source: Optional[str] = None     # nombre de columna
    expr: Optional[str] = None       # pandas.eval expression
    norm: NormCfg = NormCfg()
    weight: float = 1.0              # peso en agregación

@dataclass
class BlendCfg:
    enabled: bool = False
    alpha: float = 0.25              # 0..1 (peso del alternativo)
    alt_col: Optional[str] = None    # columna con score alternativo (ej. IA)


# ==========================
# Extracción/normalización de componentes
# ==========================

def _get_series_from_cfg(df: pd.DataFrame, comp: ComponentCfg) -> pd.Series:
    """
    Obtiene la serie fuente para el componente:
    - Si comp.source, usa esa columna.
    - Si comp.expr, evalúa expresión con pandas.eval (columnas del DF).
    """
    if comp.source:
        if comp.source not in df.columns:
            raise KeyError(f"Columna no encontrada para componente: {comp.source}")
        s = df[comp.source].astype(float)
    elif comp.expr:
        # pandas.eval con columnas del df (más seguro que eval)
        s = pd.eval(comp.expr, local_dict={c: df[c] for c in df.columns if c in comp.expr})
        s = pd.Series(s, index=df.index, dtype=float)
    else:
        raise ValueError("ComponentCfg requiere 'source' o 'expr'")
    return s

def _normalize_series(s: pd.Series, cfg: NormCfg) -> pd.Series:
    if cfg.type == "zscore":
        z = _zscore(s, window=cfg.window)
        out = _sigmoid(z)  # mapear a (0,1)
    elif cfg.type == "minmax":
        if cfg.min is None or cfg.max is None:
            raise ValueError("minmax requiere 'min' y 'max'")
        out = _minmax(s, cfg.min, cfg.max)
    elif cfg.type == "none":
        out = s.astype(float)
    else:
        raise ValueError(f"Normalización no soportada: {cfg.type}")

    if cfg.clip:
        lo, hi = cfg.clip
        out = _clip01(out, lo, hi)
    if cfg.invert:
        out = 1.0 - out
    return out


# ==========================
# Agregación de componentes
# ==========================

def _aggregate_weighted(components_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    # asegura que pesos existan para todas las columnas
    w = pd.Series({k: weights.get(k, 1.0) for k in components_df.columns}, dtype=float)
    denom = w.sum()
    if denom == 0 or pd.isna(denom):
        return pd.Series(index=components_df.index, dtype=float)
    # broadcasting por columnas
    return (components_df * w.reindex(components_df.columns)).sum(axis=1) / denom


# ==========================
# API pública
# ==========================

def make_default_score_config() -> Dict[str, Any]:
    """
    Config por defecto (razonable si tienes EMA50, RSI14 y MACD_hist):
    - RSI (0..100) -> minmax
    - Trend: (close/ema_50 - 1) -> zscore->sigmoid
    - MACD hist -> zscore->sigmoid
    """
    return {
        "method": "weighted",
        "blend": {"enabled": False, "alpha": 0.25, "alt_col": None},
        "components": {
            "rsi": {
                "source": "rsi_14",
                "norm": {"type": "minmax", "min": 0, "max": 100, "clip": [0,1], "invert": False},
                "weight": 0.3
            },
            "trend": {
                "expr": "(close / ema_50) - 1",
                "norm": {"type": "zscore", "window": 200, "clip": [0,1], "invert": False},
                "weight": 0.4
            },
            "macd": {
                "source": "macd_hist",
                "norm": {"type": "zscore", "window": 200, "clip": [0,1], "invert": False},
                "weight": 0.3
            }
        }
    }


def compute_score_timeseries(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calcula componentes normalizados y el score para todo el histórico del DataFrame.
    Requiere 'ts' como columna (ms epoch) y columnas de indicadores referenciadas en la config.

    Returns:
        DataFrame con columnas:
          ['ts', 'score', 'score_base', 'score_alt'(si aplica), 'comp.<name>' ...]
    """
    if "ts" not in df.columns:
        raise ValueError("El DataFrame debe contener columna 'ts' en ms")

    comps_cfg_raw = config.get("components", {})
    if not comps_cfg_raw:
        raise ValueError("config.components requerido")

    # Construir componentes normalizados
    comp_cols = {}
    weights = {}
    for name, raw in comps_cfg_raw.items():
        comp_cfg = ComponentCfg(
            source=raw.get("source"),
            expr=raw.get("expr"),
            norm=NormCfg(
                type=raw.get("norm", {}).get("type", "zscore"),
                window=raw.get("norm", {}).get("window"),
                min=raw.get("norm", {}).get("min"),
                max=raw.get("norm", {}).get("max"),
                clip=tuple(raw.get("norm", {}).get("clip", (0.0, 1.0))) if raw.get("norm", {}).get("clip") else None,
                invert=bool(raw.get("norm", {}).get("invert", False)),
            ),
            weight=float(raw.get("weight", 1.0)),
        )
        s_raw = _get_series_from_cfg(df, comp_cfg)
        s_norm = _normalize_series(s_raw, comp_cfg.norm)
        comp_cols[name] = s_norm
        weights[name] = comp_cfg.weight

    comps_df = pd.DataFrame(comp_cols, index=df.index)

    # Agregación
    method = (config.get("method") or "weighted").lower()
    if method == "weighted":
        base_score = _aggregate_weighted(comps_df, weights)
    else:
        raise ValueError(f"Método de score no soportado: {method}")

    # Blend con alternativo
    out = pd.DataFrame(index=df.index)
    out["ts"] = df["ts"].astype("int64")
    out["score_base"] = _clip01(base_score, 0.0, 1.0)

    blend_cfg_raw = config.get("blend", {}) or {}
    blend = BlendCfg(
        enabled=bool(blend_cfg_raw.get("enabled", False)),
        alpha=float(blend_cfg_raw.get("alpha", 0.25)),
        alt_col=blend_cfg_raw.get("alt_col"),
    )
    if blend.enabled and blend.alt_col and blend.alt_col in df.columns:
        alt = df[blend.alt_col].astype(float)
        score = (1.0 - blend.alpha) * out["score_base"] + blend.alpha * alt
        out["score_alt"] = alt
        out["score"] = _clip01(score, 0.0, 1.0)
    else:
        out["score"] = out["score_base"]

    # Adjuntar componentes normalizados para trazabilidad
    for c in comps_df.columns:
        out[f"comp.{c}"] = comps_df[c]

    return out.reset_index(drop=True)


def compute_and_persist_scores(
    asset: str,
    df: pd.DataFrame,
    config: Dict[str, Any],
    storage: Optional[Any] = None,
    method: Optional[str] = None,
    persist: bool = True,
) -> pd.DataFrame:
    """
    Calcula el score (histórico) y, si se indica, lo guarda en BD.

    - asset: símbolo del activo (ej. "BTCUSDT")
    - df: DataFrame con 'ts' y columnas de indicadores (p.ej. tras apply_indicators)
    - config: ver docstring arriba
    - storage: instancia que expone upsert_score(asset, ts, score, components, method)
    - method: fuerza un método distinto a config['method'] (opcional)
    - persist: guarda en BD si True y hay storage

    Devuelve el DataFrame con columnas ['ts','score','score_base','score_alt'(si aplica),'comp.*']
    """
    if not isinstance(df, pd.DataFrame) or "ts" not in df.columns:
        raise ValueError("df debe ser un DataFrame con columna 'ts'")

    _config = dict(config or {})
    if method:
        _config["method"] = method

    scored = compute_score_timeseries(df, _config)

    if persist and storage is not None:
        meth = (_config.get("method") or "weighted").lower()
        # guardamos fila a fila para mantener histórico consistente y permitir upsert
        # (optimizable más adelante a batch si lo necesitas)
        for _, row in scored.iterrows():
            ts = int(row["ts"])
            score_val = float(row["score"]) if pd.notna(row["score"]) else None
            # componemos dict de componentes (solo comp.*)
            comps = {k.replace("comp.", ""): (float(row[k]) if pd.notna(row[k]) else None)
                     for k in scored.columns if k.startswith("comp.")}
            try:
                storage.upsert_score(asset, ts, score_val, comps, meth)
            except Exception as e:
                logger.exception("Error guardando score (asset=%s ts=%s): %s", asset, ts, e)

    return scored


def compute_latest_score(
    asset: str,
    df: pd.DataFrame,
    config: Dict[str, Any],
    storage: Optional[Any] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Calcula el score SOLO para la última fila de df.
    Ideal para ciclos en tiempo real.

    Devuelve:
      {"ts": int, "score": float, "score_base": float, "components": {..}, "method": str}
    """
    scored = compute_score_timeseries(df.tail(1000), config)
    last = scored.iloc[-1].to_dict()
    out = {
        "ts": int(last["ts"]),
        "score": float(last["score"]) if pd.notna(last["score"]) else None,
        "score_base": float(last["score_base"]) if pd.notna(last["score_base"]) else None,
        "components": {k.replace("comp.", ""): (float(last[k]) if pd.notna(last[k]) else None)
                       for k in scored.columns if k.startswith("comp.")},
        "method": (config.get("method") or "weighted").lower(),
    }
    if persist and storage is not None:
        try:
            storage.upsert_score(asset, out["ts"], out["score"], out["components"], out["method"])
        except Exception as e:
            logger.exception("Error guardando último score (asset=%s ts=%s): %s", asset, out["ts"], e)
    return out
# ----------------------------
# Stop / Target helpers (ATR-based)
# Pegar al final de core/score.py
# ----------------------------
import math
from typing import Dict, Optional
import numpy as np

# Nota: 'logger' ya existe en este módulo; si no, definimos uno suave
try:
    logger
except NameError:
    import logging
    logger = logging.getLogger("score")

def compute_atr(df, window: int = 14, high_col="high", low_col="low", close_col="close") -> Optional[float]:
    """
    Calcula ATR simple (Wilder) sobre df ordenado ascendentemente por tiempo.
    Devuelve el último ATR (float) o None si no hay datos suficientes.
    """
    if df is None or len(df) < max(2, window):
        return None
    if not all(c in df.columns for c in (high_col, low_col, close_col)):
        logger.debug("compute_atr: columnas OHLC faltan en df")
        return None
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing: ATR_t = (ATR_{t-1} * (n-1) + TR_t) / n
    # Implementado con .ewm(adjust=False) aproximado o con rolling mean for warmup
    try:
        atr = tr.rolling(window=window, min_periods=window).mean()
        # For more Wilder exact: use ewm on full TR with alpha=1/window after warmup; keep simple for now
        atr = atr.fillna(method="ffill")
        last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else None
        return last_atr
    except Exception as e:
        logger.exception("compute_atr error: %s", e)
        return None

def infer_direction_by_ema(df, short_span: int = 9, long_span: int = 40, close_col="close") -> str:
    """
    Heurística sencilla para inferir si la tendencia actual es 'long' o 'short'
    usando dos EMAs sobre el close. Devuelve 'long' si EMA_short > EMA_long, sino 'short'.
    """
    if df is None or close_col not in df.columns or len(df) < max(short_span, long_span):
        return "long"  # fallback neutro
    close = df[close_col].astype(float)
    ema_short = close.ewm(span=short_span, adjust=False).mean()
    ema_long = close.ewm(span=long_span, adjust=False).mean()
    return "long" if ema_short.iloc[-1] > ema_long.iloc[-1] else "short"

def compute_entry_stop_target(
    df,
    atr_window: int = 14,
    stop_mult: float = 1.3,
    target_mult: float = 2.6,
    entry_price: Optional[float] = None,
    direction: Optional[str] = None,
    close_col="close"
) -> Dict:
    """
    Dado un DataFrame de velas (orden asc), calcula:
      - atr (último)
      - entry_price (último close si no se pasa)
      - stop (entry ± 1.3 * atr)
      - target (entry ± 2.6 * atr)
      - direction: 'long' o 'short' (si no se pasa, se infiere por EMA)
    Devuelve dict con keys: atr, entry, stop, target, direction, pct_stop, pct_target
    """
    out = {"atr": None, "entry": None, "stop": None, "target": None, "direction": None,
           "pct_stop": None, "pct_target": None}
    if df is None or len(df) == 0:
        return out

    try:
        atr = compute_atr(df, window=atr_window, close_col=close_col)
        out["atr"] = atr
        entry = float(entry_price) if entry_price is not None else float(df[close_col].iloc[-1])
        out["entry"] = entry

        if direction is None:
            direction = infer_direction_by_ema(df, close_col=close_col)
        out["direction"] = direction

        if atr is None or atr <= 0:
            # No ATR available -> no stop/target
            return out

        if direction == "long":
            stop = entry - stop_mult * atr
            target = entry + target_mult * atr
        else:
            # short
            stop = entry + stop_mult * atr
            target = entry - target_mult * atr

        out.update({
            "stop": float(stop),
            "target": float(target),
            "pct_stop": (stop - entry) / entry if entry != 0 else None,
            "pct_target": (target - entry) / entry if entry != 0 else None
        })
        return out
    except Exception as e:
        logger.exception("compute_entry_stop_target error: %s", e)
        return out

# Convenience helper that uses the project adapter to load candles for an asset
def compute_stop_target_for_asset(adapter_obj, asset: str, interval: str = "1h", lookback: int = 200,
                                  atr_window: int = 14, stop_mult: float = 1.3, target_mult: float = 2.6) -> Dict:
    """
    Usa `adapter_obj.load_candles(asset, limit=lookback)` y calcula stop/target.
    adapter_obj es el singleton `core.adapter.adapter` o equivalente.
    """
    try:
        df = adapter_obj.load_candles(asset, limit=lookback)
        # adapter retorna DataFrame o None
        return compute_entry_stop_target(df, atr_window=atr_window, stop_mult=stop_mult,
                                         target_mult=target_mult, entry_price=None)
    except Exception as e:
        logger.exception("compute_stop_target_for_asset failed for %s %s: %s", asset, interval, e)
        return {"error": str(e)}
