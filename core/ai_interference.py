# core/ai_interference.py
"""
AI Interference / Explainability utilities

Funciones principales:
 - explain_scores(asset, storage=None, n_samples=200, use_openai=True, model=None)
   -> devuelve texto explicativo (string). Usa OpenAI si está disponible y configurado,
      si no ofrece un resumen local estadístico.
 - describe_scores_df(scores_df) -> dict con estadísticas útiles.

Implementación:
 - Caching simple por fichero: data/ai_cache/{asset}.json con TTL controlado por env AI_EXPLAIN_CACHE_TTL.
 - Rate-limit simple entre llamadas a OpenAI controlado por AI_CALL_MIN_INTERVAL (segundos).
 - Manejo robusto de errores: si falla OpenAI, cae al resumen local sin lanzar.
"""
from __future__ import annotations

import os
import time
import json
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

# optional import of openai
try:
    import openai
except Exception:
    openai = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# cache dir
CACHE_DIR = Path(os.getenv("AI_CACHE_DIR", "data/ai_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = int(os.getenv("AI_EXPLAIN_CACHE_TTL", "3600"))  # seconds

# minimum seconds between OpenAI calls (simple rate limiter)
MIN_INTERVAL = float(os.getenv("AI_CALL_MIN_INTERVAL", "0.5"))
_LAST_AI_CALL_TS = 0.0


def _ensure_seconds_between_calls():
    """Simple wait to ensure MIN_INTERVAL between calls."""
    global _LAST_AI_CALL_TS
    now = time.time()
    elapsed = now - _LAST_AI_CALL_TS
    if elapsed < MIN_INTERVAL:
        to_sleep = MIN_INTERVAL - elapsed
        logger.debug("Sleeping %.3fs to respect AI_CALL_MIN_INTERVAL", to_sleep)
        time.sleep(to_sleep)
    _LAST_AI_CALL_TS = time.time()


def _cache_path_for(asset: str) -> Path:
    safe = str(asset).replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe}.json"


def _load_cached(asset: str) -> Optional[Dict[str, Any]]:
    p = _cache_path_for(asset)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        ts = obj.get("_generated_at", 0)
        if time.time() - float(ts) > CACHE_TTL:
            return None
        return obj
    except Exception:
        return None


def _save_cache(asset: str, payload: Dict[str, Any]) -> None:
    p = _cache_path_for(asset)
    try:
        payload["_generated_at"] = time.time()
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write AI cache for %s", asset)


def describe_scores_df(scores_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extrae estadísticas y resumen útil de scores_df.
    scores_df: DataFrame with columns ['ts','score'] where 'score' is dict-like or numeric.
    Devuelve dict con count, mean, std, min, max, quantiles y small-sample examples.
    """
    if scores_df is None or scores_df.empty:
        return {"count": 0}

    def _extract_numeric(val):
        # tries to get numeric pred from whatever representation
        try:
            if isinstance(val, dict):
                if "pred" in val:
                    return float(val["pred"])
                # try first numeric
                for v in val.values():
                    try:
                        return float(v)
                    except Exception:
                        continue
            return float(val)
        except Exception:
            return float("nan")

    s = scores_df["score"].apply(_extract_numeric).dropna()
    if s.empty:
        return {"count": 0}

    desc = s.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    top = s.nlargest(5).tolist()
    bot = s.nsmallest(5).tolist()
    out = {
        "count": int(desc.get("count", 0)),
        "mean": float(desc.get("mean", math.nan)),
        "std": float(desc.get("std", math.nan)),
        "min": float(desc.get("min", math.nan)),
        "max": float(desc.get("max", math.nan)),
        "25%": float(desc.get("25%", math.nan)),
        "50%": float(desc.get("50%", math.nan)),
        "75%": float(desc.get("75%", math.nan)),
        "1%": float(desc.get("1%", math.nan)) if "1%" in desc else None,
        "95%": float(desc.get("95%", math.nan)) if "95%" in desc else None,
        "top5": top,
        "bottom5": bot,
    }
    return out


def _local_explain_text(asset: str, stats: Dict[str, Any]) -> str:
    """
    Genera texto explicativo a partir de stats dict (output de describe_scores_df).
    """
    if not stats or stats.get("count", 0) == 0:
        return f"No hay scores numéricos suficientes para el activo {asset}."
    lines = []
    lines.append(f"Resumen automatizado de scores para {asset}:")
    lines.append(f"- Número de muestras: {stats.get('count')}")
    lines.append(f"- Media: {stats.get('mean'):.6f}, desviación típica: {stats.get('std'):.6f}")
    lines.append(f"- Rango: min={stats.get('min'):.6f} / max={stats.get('max'):.6f}")
    q25 = stats.get("25%")
    q75 = stats.get("75%")
    if q25 is not None and q75 is not None:
        lines.append(f"- IQR (25%-75%): {q25:.6f} - {q75:.6f}")
    top = stats.get("top5") or []
    bot = stats.get("bottom5") or []
    if top:
        lines.append("- Top 5 preds: " + ", ".join([f"{v:.6f}" for v in top]))
    if bot:
        lines.append("- Bottom 5 preds: " + ", ".join([f"{v:.6f}" for v in bot]))
    # suggestions
    lines.append("")
    lines.append("Sugerencias automáticas:")
    if stats.get("std", 0) > 1e-3 * max(1.0, abs(stats.get("mean", 0))):
        lines.append("- La varianza es relativamente alta: considera normalizar features o reducir la complejidad del modelo.")
    else:
        lines.append("- La varianza es baja; el modelo podría estar sobreajustado o producir predicciones conservadoras.")
    lines.append("- Revisa que los datos de entrada no contengan leaks temporales y que el backfill de velas esté completo.")
    return "\n".join(lines)


def _call_openai(prompt: str, model: Optional[str] = None, max_tokens: int = 512) -> Optional[str]:
    """
    Llama a OpenAI ChatCompletion (síncrono). Retorna texto o None en caso de fallo.
    """
    if openai is None:
        logger.debug("openai package no disponible")
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.debug("OPENAI_API_KEY no configurada")
        return None

    # comply with rate/min interval
    _ensure_seconds_between_calls()

    try:
        openai.api_key = key
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Build a compact system + user prompt
        messages = [
            {"role": "system", "content": "Eres un asistente que resume métricas de modelos y ofrece recomendaciones prácticas para mejorar modelos de trading."},
            {"role": "user", "content": prompt},
        ]
        # Try ChatCompletion (works with older clients); if not, fallback to openai.ChatCompletion.create
        resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.2)
        # The API returns choices -> message for ChatCompletion
        if resp and getattr(resp, "choices", None):
            choice = resp.choices[0]
            # support both .message.content and .text
            text = None
            if getattr(choice, "message", None) and getattr(choice.message, "content", None):
                text = choice.message.content
            elif getattr(choice, "text", None):
                text = choice.text
            if text:
                return str(text).strip()
        # fallback to text if shape differs
        return None
    except Exception:
        logger.exception("OpenAI call failed")
        return None


def explain_scores(asset: str, storage=None, n_samples: int = 200, use_openai: bool = True, model: Optional[str] = None) -> str:
    """
    Main function:
      - intenta cargar n_samples scores desde storage (si se proporciona)
      - si hay cache válida devuelve cache
      - si OPENAI y use_openai -> llama OpenAI (con rate limit + retries breve)
      - si falla, genera resumen local y lo devuelve
    """
    # 1) try cache
    cache = _load_cached(asset)
    if cache is not None:
        logger.debug("Returning cached AI explanation for %s", asset)
        return cache.get("text", "")

    # 2) load scores from storage if available
    scores_df = None
    if storage is not None and hasattr(storage, "load_scores"):
        try:
            scores_df = storage.load_scores(asset=asset, limit=n_samples)
        except Exception:
            logger.exception("storage.load_scores failed for explain_scores")
            scores_df = None

    # 3) compute local stats
    stats = describe_scores_df(scores_df) if scores_df is not None else {"count": 0}

    # 4) attempt OpenAI if requested
    text = None
    if use_openai and openai is not None and os.getenv("OPENAI_API_KEY"):
        # build compact prompt
        if stats.get("count", 0) == 0:
            prompt = f"Provee una breve explicación sobre posibles problemas y checks para el asset {asset}. No hay suficientes scores numéricos; sugiere pasos de depuración."
        else:
            # include stats summary in the prompt
            stats_json = json.dumps(stats, ensure_ascii=False, indent=2)
            prompt = (
                f"Here are summary statistics for model scores for asset {asset}:\n\n{stats_json}\n\n"
                "Provide a clear explanation of what these statistics mean, likely causes (data, model), "
                "and 3 practical suggestions to investigate or improve the model. Be concise and actionable."
            )
        # try with small retries
        retries = int(os.getenv("AI_OPENAI_MAX_RETRIES", "2"))
        for attempt in range(1, retries + 1):
            text = _call_openai(prompt, model=model)
            if text:
                break
            sleep = 0.5 * attempt
            logger.debug("OpenAI attempt %d failed, sleeping %.2fs", attempt, sleep)
            time.sleep(sleep)

    # 5) fallback to local explanation
    if not text:
        text = _local_explain_text(asset, stats)

    # 6) cache and return
    try:
        _save_cache(asset, {"text": text, "stats": stats})
    except Exception:
        logger.exception("Failed to save AI cache (continuing)")

    return text
