# core/ai_inference.py
"""
AI inference + explainability module for Watchlist

Este módulo combina:
 - utilidades de explain / explainability (basado en tu antiguo ai_interference.py)
 - funciones para guardar/cargar modelos (joblib/pickle)
 - predict_scores(df, asset, interval, ...) -> devuelve DataFrame con columna 'ai_score'
 - helpers de lista y metadatos de modelos

Diseño:
 - Models folder: <repo_root>/models
 - Cada modelo tiene .bin (joblib/pickle) y .meta.json con metadata (feature_names, created_at, notes...)
 - predict_scores intenta elegir las columnas apropiadas según meta.feature_names; si faltan las rellena a 0
 - explain_scores mantiene la funcionalidad original basada en OpenAI si está configurado,
   con caching simple y TTL controlable por env var.
"""

from __future__ import annotations
import os
import time
import json
import glob
import math
import traceback
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# prefer joblib if available
try:
    import joblib as _joblib
    _USE_JOBLIB = True
except Exception:
    import pickle as _pickle
    _USE_JOBLIB = False

# Optional OpenAI support for explainability
try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

logger = logging.getLogger("core.ai_inference")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Paths
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
AI_CACHE_DIR = os.path.join(BASE_DIR, "data", "ai_cache")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(AI_CACHE_DIR, exist_ok=True)

# Env-configurable params
AI_EXPLAIN_CACHE_TTL = int(os.environ.get("AI_EXPLAIN_CACHE_TTL", 60 * 60 * 4))  # 4h default
AI_CALL_MIN_INTERVAL = float(os.environ.get("AI_CALL_MIN_INTERVAL", 0.5))  # seconds
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")  # fallbacks

# If openai available and key, set it
if _OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        pass


# ------------------------
# Low-level dump/load
# ------------------------
def _safe_dump(obj: object, path: str) -> bool:
    try:
        if _USE_JOBLIB:
            _joblib.dump(obj, path)
        else:
            with open(path, "wb") as fh:
                _pickle.dump(obj, fh)
        return True
    except Exception:
        logger.exception("Failed to dump model to %s", path)
        return False


def _safe_load(path: str):
    try:
        if _USE_JOBLIB:
            return _joblib.load(path)
        else:
            with open(path, "rb") as fh:
                return _pickle.load(fh)
    except Exception:
        logger.exception("Failed to load model from %s", path)
        return None


# ------------------------
# Model file helpers
# ------------------------
def _model_filename(asset: Optional[str], interval: Optional[str], ts: Optional[int] = None) -> str:
    stamp = ts if ts is not None else int(time.time())
    asset_s = asset.replace("/", "_") if isinstance(asset, str) else "generic"
    interval_s = str(interval) if interval is not None else "any"
    return f"{asset_s}__{interval_s}__{stamp}.bin"


def _meta_filename(model_fname: str) -> str:
    return model_fname + ".meta.json"


def list_models(asset: Optional[str] = None, interval: Optional[str] = None) -> List[str]:
    """
    Lista rutas de fichero .bin ordenadas (más recientes primero).
    """
    asset_pat = "*" if asset is None else asset.replace("/", "_")
    int_pat = "*" if interval is None else str(interval)
    pat = os.path.join(MODELS_DIR, f"{asset_pat}__{int_pat}__*.bin")
    files = sorted(glob.glob(pat), reverse=True)
    return files


def get_latest_model_path(asset: Optional[str], interval: Optional[str]) -> Optional[str]:
    lst = list_models(asset, interval)
    return lst[0] if lst else None


def save_model(model: object, meta: Optional[Dict[str, Any]] = None, asset: Optional[str] = None, interval: Optional[str] = None, model_name: Optional[str] = None) -> Optional[str]:
    """
    Guarda un modelo binario y su metadata. Devuelve la ruta del fichero .bin o None.
    meta: dict recomendada con 'feature_names', 'notes', etc.
    """
    try:
        ts = int(time.time())
        fname = model_name if model_name else _model_filename(asset, interval, ts)
        path = os.path.join(MODELS_DIR, fname)
        ok = _safe_dump(model, path)
        if not ok:
            return None
        meta_out = dict(meta) if isinstance(meta, dict) else {}
        meta_out.update({"asset": asset, "interval": interval, "saved_at": int(ts), "model_file": os.path.basename(path)})
        meta_path = os.path.join(MODELS_DIR, _meta_filename(os.path.basename(path)))
        try:
            with open(meta_path, "w", encoding="utf8") as fh:
                json.dump(meta_out, fh, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to write model meta to %s", meta_path)
        logger.info("Model saved to %s (meta: %s)", path, meta_out.get("model_file"))
        return path
    except Exception:
        logger.exception("save_model failed")
        return None


def load_model(asset: Optional[str] = None, interval: Optional[str] = None, model_path: Optional[str] = None) -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    """
    Carga modelo y meta. Si model_path es None, busca el último modelo para asset/interval.
    Devuelve (model, meta_dict) o (None, None).
    """
    try:
        if model_path:
            model_file = model_path
        else:
            model_file = get_latest_model_path(asset, interval)
            if not model_file:
                return None, None
        model = _safe_load(model_file)
        meta = {}
        try:
            meta_path = os.path.join(MODELS_DIR, _meta_filename(os.path.basename(model_file)))
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf8") as fh:
                    meta = json.load(fh)
        except Exception:
            logger.exception("Failed to read meta for model %s", model_file)
            meta = {}
        return model, meta
    except Exception:
        logger.exception("load_model failed")
        return None, None


# ------------------------
# Model predict helpers
# ------------------------
def _model_predict(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Normalize different model interfaces into numpy array (0..1 scale not guaranteed).
    """
    try:
        # sklearn-like: predict_proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return np.asarray(probs[:, 1], dtype=float)
            return np.asarray(np.max(probs, axis=1), dtype=float)

        if hasattr(model, "predict"):
            out = model.predict(X)
            arr = np.asarray(out, dtype=float)
            # if 0/1 labels -> return as-is
            uniques = np.unique(arr)
            if set(uniques.tolist()) <= {0.0, 1.0}:
                return arr
            # otherwise sigmoid/minmax fallback
            try:
                from scipy.special import expit
                return expit(arr)
            except Exception:
                amin, amax = float(np.nanmin(arr)), float(np.nanmax(arr))
                if amax - amin == 0:
                    return np.clip(arr, 0.0, 1.0)
                return (arr - amin) / (amax - amin)

        # LightGBM booster (.predict)
        clsname = type(model).__name__.lower()
        if "booster" in clsname or "lgb" in clsname:
            out = model.predict(X)
            return np.asarray(out, dtype=float)

        # Callable model
        try:
            out = model(X)
            arr = np.asarray(out, dtype=float)
            if arr.ndim == 1:
                amin = float(np.nanmin(arr))
                amax = float(np.nanmax(arr))
                if amax - amin > 1e-12:
                    return (arr - amin) / (amax - amin)
                return np.clip(arr, 0.0, 1.0)
        except Exception:
            pass
    except Exception:
        logger.exception("Model predict failed")
    return None


def predict_scores(df: pd.DataFrame, asset: Optional[str] = None, interval: Optional[str] = None, model_path: Optional[str] = None, model_meta: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Devuelve DataFrame con columna 'ai_score' alineada al índice de df.
    - intenta cargar un modelo (asset/interval) si model_path no dado
    - si meta indica feature_names, se usa para seleccionar las columnas (faltantes rellenadas a 0.0)
    - si no hay modelo devuelve serie NaN
    """
    if df is None:
        return pd.DataFrame({"ai_score": []})
    if df.empty:
        return pd.DataFrame({"ai_score": pd.Series(dtype=float, index=df.index)})

    model, meta = load_model(asset, interval, model_path) if (model_path is None) else load_model(None, None, model_path)
    if model is None:
        # no model -> nan series
        return pd.DataFrame({"ai_score": pd.Series([float("nan")] * len(df), index=df.index)})

    # features selection
    feature_names = None
    if isinstance(model_meta, dict):
        feature_names = model_meta.get("feature_names") or model_meta.get("features")
    if not feature_names and isinstance(meta, dict):
        feature_names = meta.get("feature_names") or meta.get("features")

    if feature_names:
        missing = [c for c in feature_names if c not in df.columns]
        X = df.reindex(columns=feature_names).copy()
        if missing:
            for c in missing:
                X[c] = 0.0
    else:
        # fallback to reasonable defaults
        preferred = ["close", "ema_9", "ema_40", "ema_diff", "rsi_like", "atr"]
        present = [c for c in preferred if c in df.columns]
        if present:
            X = df[present].copy()
        else:
            X = df.select_dtypes(include=[np.number]).fillna(0.0)
            if X.empty:
                return pd.DataFrame({"ai_score": pd.Series([float("nan")] * len(df), index=df.index)})

    try:
        out = _model_predict(model, X)
        if out is None:
            return pd.DataFrame({"ai_score": pd.Series([float("nan")] * len(df), index=df.index)})
        arr = np.asarray(out, dtype=float)
        # align length
        if arr.shape[0] != len(df):
            if arr.shape[0] > len(df):
                arr = arr[-len(df):]
            else:
                pad = np.full(len(df) - arr.shape[0], arr[-1] if arr.size > 0 else 0.0)
                arr = np.concatenate([arr, pad])
        # normalize to 0..1
        amin = float(np.nanmin(arr)) if arr.size > 0 else 0.0
        amax = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        if math.isfinite(amin) and math.isfinite(amax) and (amax - amin) > 1e-12:
            arr_scaled = (arr - amin) / (amax - amin)
        else:
            arr_scaled = np.clip(arr, 0.0, 1.0)
        return pd.DataFrame({"ai_score": pd.Series(arr_scaled, index=df.index)})
    except Exception:
        logger.exception("predict_scores failed")
        return pd.DataFrame({"ai_score": pd.Series([float("nan")] * len(df), index=df.index)})


# aliases for compatibility
def infer(df: pd.DataFrame, asset: Optional[str] = None, interval: Optional[str] = None, **kwargs):
    return predict_scores(df, asset=asset, interval=interval, **kwargs)


def predict(df: pd.DataFrame, asset: Optional[str] = None, interval: Optional[str] = None, **kwargs):
    return predict_scores(df, asset=asset, interval=interval, **kwargs)


# ------------------------
# Explainability functions (ported / adapted from ai_interference)
# ------------------------
_last_ai_call_ts = 0.0


def _ensure_seconds_between_calls(min_interval: float):
    global _last_ai_call_ts
    now = time.time()
    if now - _last_ai_call_ts < min_interval:
        wait = min_interval - (now - _last_ai_call_ts)
        time.sleep(wait)
    _last_ai_call_ts = time.time()


def _cache_path_for(asset: str) -> str:
    safe = asset.replace("/", "_")
    return os.path.join(AI_CACHE_DIR, f"{safe}.json")


def _write_ai_cache(asset: str, payload: Dict[str, Any]):
    try:
        path = _cache_path_for(asset)
        with open(path, "w", encoding="utf8") as fh:
            json.dump({"ts": int(time.time()), "payload": payload}, fh, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to write AI cache for %s", asset)


def _read_ai_cache(asset: str, max_age: Optional[int] = None) -> Optional[Dict[str, Any]]:
    try:
        path = _cache_path_for(asset)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf8") as fh:
            data = json.load(fh)
        ts = int(data.get("ts", 0))
        if max_age is not None and (int(time.time()) - ts) > max_age:
            return None
        return data.get("payload")
    except Exception:
        logger.exception("Failed to read AI cache for %s", asset)
    return None


def describe_scores_df(scores_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extrae estadísticas y resumen útil de scores_df.
    scores_df: DataFrame with column 'score' or numeric series-like objects.
    """
    if scores_df is None or scores_df.empty:
        return {"count": 0}
    try:
        # try to extract numeric values or flatten if necessary
        if "score" in scores_df.columns:
            s = pd.to_numeric(scores_df["score"], errors="coerce").dropna()
        else:
            # assume it's a series-like
            s = pd.Series(scores_df).astype(float).dropna()
        if s.empty:
            return {"count": 0}
        desc = s.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        top = s.nlargest(5).tolist()
        bot = s.nsmallest(5).tolist()
        return {
            "count": int(desc.get("count", 0)),
            "mean": float(desc.get("mean", float("nan"))),
            "std": float(desc.get("std", float("nan"))),
            "min": float(desc.get("min", float("nan"))),
            "max": float(desc.get("max", float("nan"))),
            "quantiles": { "p01": desc.get("1%"), "p05": desc.get("5%"), "p10": desc.get("10%"), "p50": desc.get("50%"), "p90": desc.get("90%"), "p99": desc.get("99%") },
            "top": top,
            "bottom": bot
        }
    except Exception:
        logger.exception("describe_scores_df failed")
        return {"count": 0}


def _local_explain_text(asset: str, scores_df: pd.DataFrame) -> str:
    """
    Genera un texto breve explicativo a partir de estadísticas (fallback si no hay OpenAI).
    """
    try:
        desc = describe_scores_df(scores_df)
        if desc.get("count", 0) == 0:
            return f"No hay suficientes datos para explicar {asset}."
        return (f"Resumen de scores para {asset}: {desc.get('count')} muestras, media {desc.get('mean'):.3f}, "
                f"desviación {desc.get('std'):.3f}, min {desc.get('min'):.3f}, max {desc.get('max'):.3f}. "
                f"Top ejemplos: {desc.get('top')} .")
    except Exception:
        logger.exception("_local_explain_text failed")
        return "No explanation available."


def _call_openai(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 512) -> Optional[str]:
    """
    Realiza llamada a OpenAI (chat completions) si está configurado.
    """
    if not _OPENAI_AVAILABLE or not OPENAI_API_KEY:
        logger.debug("OpenAI not available or API key missing")
        return None
    try:
        _ensure_seconds_between_calls(AI_CALL_MIN_INTERVAL)
        # Chat completions style
        if hasattr(openai, "ChatCompletion"):
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            # extract text
            if resp and "choices" in resp and len(resp["choices"]) > 0:
                return resp["choices"][0]["message"]["content"].strip()
        # fallback older completions
        if hasattr(openai, "Completion"):
            resp = openai.Completion.create(engine=model, prompt=prompt, max_tokens=max_tokens, temperature=0.2)
            if resp and "choices" in resp and len(resp["choices"]) > 0:
                return resp["choices"][0]["text"].strip()
    except Exception:
        logger.exception("OpenAI call failed")
    return None


def explain_scores(asset: str, scores_df: pd.DataFrame, use_openai: bool = True, model: str = "gpt-3.5-turbo", force_refresh: bool = False) -> str:
    """
    Devuelve un texto explicativo (string) para scores_df.
    - usa cache en data/ai_cache/{asset}.json con TTL AI_EXPLAIN_CACHE_TTL
    - si use_openai True y OpenAI configurado, intentará generar texto con la API (con rate-limit)
    - si falla, devuelve un resumen local
    """
    try:
        if not force_refresh:
            cached = _read_ai_cache(asset, max_age=AI_EXPLAIN_CACHE_TTL)
            if cached:
                return cached.get("explain_text", _local_explain_text(asset, scores_df))

        # build prompt
        prompt = f"Explique brevemente (3-5 líneas) del siguiente resumen de scores para el activo {asset}.\n"
        prompt += "Proporcione puntos de interés y posibles razones de movimientos extremos.\n\n"
        try:
            desc = describe_scores_df(scores_df)
            prompt += "Resumen estadístico (json):\n" + json.dumps(desc, default=str) + "\n\n"
        except Exception:
            prompt += "No pudimos generar resumen estadístico.\n\n"

        # prefer local fallback
        if use_openai and _OPENAI_AVAILABLE and OPENAI_API_KEY:
            out = _call_openai(prompt, model=model)
            if out:
                payload = {"explain_text": out, "via": "openai", "model": model, "ts": int(time.time())}
                _write_ai_cache(asset, payload)
                return out

        # fallback local explanation
        out = _local_explain_text(asset, scores_df)
        payload = {"explain_text": out, "via": "local", "model": None, "ts": int(time.time())}
        _write_ai_cache(asset, payload)
        return out
    except Exception:
        logger.exception("explain_scores failed")
        return _local_explain_text(asset, scores_df)


# Expose public API names
__all__ = [
    "save_model", "load_model", "list_models", "get_latest_model_path",
    "predict_scores", "infer", "predict",
    "describe_scores_df", "explain_scores"
]
