# dashboard/app.py
"""
Dashboard Watchlist - versión integrada y robusta
Reemplaza completamente dashboard/app.py por este archivo.

Características clave:
- Detección automática de core modules (storage_postgres, fetch, indicators, score, orchestrator, ai_train, ai_inference)
- Carga watchlist desde data/config/actions.csv + cryptos.csv (detecta ~93 activos si existen)
- Panel lateral dinámico: sliders para TODOS los componentes del score, toggle IA, parámetros de indicadores
- Tabla principal ordenable/filtrable por columnas (precio, score, volatilidad, etc.)
- Detalle de activo: velas, EMAs, RSI, MACD, ATR, Bollinger, score timeseries
- Acciones desde la UI: actualizar datos, backfill, backtest, entrenar IA, infer IA, export CSV/JSON
- Backfill / backtest / train / infer ejecutados en background (threads) con estado en st.session_state
- Persistencia de settings usando storage_postgres.save_setting/load_setting si existe, fallback JSON en data/config/settings.json
- Guarda resultados de backtests en data/db/backtests/
- Mínimas dependencias: streamlit, pandas, numpy, plotly (si no existe fetch/postgres usa CSVs)

Instrucciones:
- Haz backup: cp dashboard/app.py dashboard/app.py.bak
- Pega este archivo y ejecuta: streamlit run dashboard/app.py
- En Render: startCommand -> python scripts/ensure_db_on_start.py && streamlit run dashboard/app.py --server.port ${PORT:-10000} --server.address 0.0.0.0
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Setup logging (pegado al inicio, junto a imports)
# ----------------------------
import logging
from logging.handlers import RotatingFileHandler
from collections import deque

LOG_PATH = "logs/app.log"
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

# deque en memoria para mostrar logs in-process
INMEM_LOG = deque(maxlen=1000)

class StreamlitMemoryHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            INMEM_LOG.append(msg)
        except Exception:
            pass

# configure root logger once (evita duplicados)
root_logger = logging.getLogger()
if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    # --- Safe log handler creation (creates dir, fallback to stdout) ---
    import os, tempfile
    from logging import StreamHandler
    
    log_dir = os.path.dirname(LOG_PATH) or "."
    
    # Intentar crear la carpeta logs
    try:
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        # Si no se puede crear, usar /tmp
        temp_log = os.path.join(tempfile.gettempdir(), "watchlist_app.log")
        root_logger.warning("No se pudo crear %s (err=%s). Usando fallback %s", log_dir, e, temp_log)
        LOG_PATH = temp_log
        log_dir = os.path.dirname(LOG_PATH)
    
    # Intentar crear el RotatingFileHandler; si falla usar StreamHandler
    try:
        fh = RotatingFileHandler(LOG_PATH, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)
    except Exception as e:
        root_logger.warning(
            "RotatingFileHandler no disponible para %s (err=%s). Usando StreamHandler (stdout).",
            LOG_PATH, e
        )
    sh = StreamHandler()
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)

    memh = StreamlitMemoryHandler()
    memh.setFormatter(fmt)
    root_logger.addHandler(memh)
    root_logger.setLevel(logging.INFO)

# -------------------------
# Basic paths & env
# -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
CONF_DIR = os.path.join(DATA_DIR, "config")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DB_DIR = os.path.join(DATA_DIR, "db")
BACKTESTS_DIR = os.path.join(DB_DIR, "backtests")
SETTINGS_FILE = os.path.join(CONF_DIR, "settings.json")

os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(BACKTESTS_DIR, exist_ok=True)

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("watchlist.dashboard")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# -------------------------
# Safe importer / resolver helpers
# -------------------------
def safe_import(module_name: str, attr: Optional[str] = None):
    try:
        m = __import__(module_name, fromlist=[attr] if attr else [])
        if attr:
            return getattr(m, attr)
        return m
    except Exception:
        return None

# Detect core modules (try many variants / names)
fetch_mod = safe_import("core.fetch")
indicators_mod = safe_import("core.indicators")
score_mod = safe_import("core.score")
orchestrator_mod = safe_import("core.orchestrator")
ai_train_mod = safe_import("core.ai_train")
ai_inf_mod = safe_import("core.ai_inference")
storage_mod = safe_import("core.storage")  # fallback storage (sqlite)
storage_pg_mod = safe_import("core.storage_postgres")

# If storage_postgres exposes PostgresStorage class instantiate if possible
_pg_storage = None
if storage_pg_mod and hasattr(storage_pg_mod, "PostgresStorage"):
    try:
        PgCls = getattr(storage_pg_mod, "PostgresStorage")
        _pg_storage = PgCls()
        logger.info("PostgresStorage instantiated for dashboard integration.")
    except Exception as e:
        logger.warning("PostgresStorage instantiation failed: %s", e)
        _pg_storage = None

# -------------------------
# Wrappers for expected core functions (compat layer)
# -------------------------
# Watchlist loader (reads both actions.csv and cryptos.csv by default)
def load_watchlist_from_config() -> List[str]:
    files = ["actions.csv", "cryptos.csv", "crypto.csv", "actions.csv"]  # accept variants
    out = []
    for fname in files:
        path = os.path.join(CONF_DIR, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
                if "symbol" in df.columns:
                    syms = df["symbol"].dropna().astype(str).str.strip().tolist()
                else:
                    syms = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                out.extend([s for s in syms if s])
            except Exception:
                logger.exception("Failed reading watchlist file %s", path)
    # dedupe preserving order
    seen = set(); unique = []
    for s in out:
        if s not in seen:
            seen.add(s); unique.append(s)
    return unique

def list_assets_source() -> List[str]:
    # 1) try fetch module function variants
    try:
        if fetch_mod:
            for fn in ("list_watchlist_assets", "list_assets", "get_watchlist", "get_assets"):
                if hasattr(fetch_mod, fn):
                    try:
                        res = getattr(fetch_mod, fn)()
                        if isinstance(res, (list, tuple)):
                            return list(res)
                        if isinstance(res, pd.DataFrame):
                            if "symbol" in res.columns:
                                return res["symbol"].astype(str).tolist()
                            return res.iloc[:, 0].astype(str).tolist()
                    except Exception:
                        logger.exception("fetch_mod.%s failed", fn)
        # 2) try postgres storage listing
        if _pg_storage and hasattr(_pg_storage, "list_assets"):
            try:
                res = _pg_storage.list_assets()
                if isinstance(res, list):
                    return res
            except Exception:
                logger.exception("pg.list_assets failed")
        # 3) try storage module
        if storage_mod and hasattr(storage_mod, "list_assets"):
            try:
                res = storage_mod.list_assets()
                if isinstance(res, list):
                    return res
            except Exception:
                logger.exception("storage.list_assets failed")
    except Exception:
        logger.exception("list_assets_source unexpected error")
    # 4) fallback to config csvs
    cfg = load_watchlist_from_config()
    if cfg:
        return cfg
    return ["BTCUSDT", "ETHUSDT", "AAPL", "TSLA"]

# Candle loader wrapper
def load_candles_source(symbol: str, limit: int = 1000, timeframe: Optional[str] = None, force_network: bool = False) -> pd.DataFrame:
    """
    Strategy:
      - If not force_network: try Postgres storage (_pg_storage.load_candles) or storage_mod.load_candles
      - Then try fetch_mod functions (fetch_ohlcv, get_candles, get_latest_candles, fetch_candles)
      - Persist network results to storage or CSV for subsequent loads
      - Fallback: read CSV in data/cache/{symbol}.csv
    """
    # local import to avoid top-level heavy imports
    import pandas as _pd

    # 1) try storage
    if not force_network:
        try:
            if _pg_storage and hasattr(_pg_storage, "load_candles"):
                df = _pg_storage.load_candles(symbol, limit=limit)
                if isinstance(df, _pd.DataFrame) and not df.empty:
                    return df
            if storage_mod and hasattr(storage_mod, "load_candles"):
                try:
                    df = storage_mod.load_candles(symbol, limit=limit)
                    if isinstance(df, _pd.DataFrame) and not df.empty:
                        return df
                except Exception:
                    logger.debug("storage_mod.load_candles failed for %s", symbol)
        except Exception:
            logger.exception("storage load attempt failed")

    # 2) try fetch functions
    tried_network = False
    try:
        if fetch_mod:
            # try many common names
            for fn in ("get_candles", "fetch_candles", "get_latest_candles", "fetch"):
                if hasattr(fetch_mod, fn):
                    try:
                        f = getattr(fetch_mod, fn)
                        # some implementations expect (symbol, limit) others (symbol, limit, timeframe)
                        try:
                            df = f(symbol, limit=limit) if "limit" in f.__code__.co_varnames else f(symbol)
                        except TypeError:
                            try:
                                df = f(symbol, limit, timeframe)
                            except Exception:
                                df = f(symbol)
                        if isinstance(df, _pd.DataFrame) and not df.empty:
                            tried_network = True
                            # attempt to persist
                            try:
                                if _pg_storage and hasattr(_pg_storage, "save_candles"):
                                    _pg_storage.save_candles(symbol, df)
                                elif storage_mod and hasattr(storage_mod, "save_candles"):
                                    # note: some storage.save_candles signature may be (symbol, df) or (df, symbol)
                                    try:
                                        storage_mod.save_candles(symbol, df)
                                    except TypeError:
                                        try:
                                            storage_mod.save_candles(df, asset=symbol)
                                        except Exception:
                                            pass
                                else:
                                    # CSV fallback
                                    p = os.path.join(CACHE_DIR, f"{symbol.replace('/','_')}.csv")
                                    try:
                                        df.to_csv(p, index=False)
                                    except Exception:
                                        pass
                            except Exception:
                                logger.exception("persisting network result failed")
                            return df
                    except Exception:
                        logger.debug("fetch_mod.%s failed for %s", fn, symbol)
    except Exception:
        logger.exception("fetch module network attempt failed for %s", symbol)

    # 3) if network tried but returned empty, try storage again (maybe persisted)
    if tried_network:
        try:
            if _pg_storage and hasattr(_pg_storage, "load_candles"):
                df = _pg_storage.load_candles(symbol, limit=limit)
                if isinstance(df, _pd.DataFrame) and not df.empty:
                    return df
            if storage_mod and hasattr(storage_mod, "load_candles"):
                df = storage_mod.load_candles(symbol, limit=limit)
                if isinstance(df, _pd.DataFrame) and not df.empty:
                    return df
        except Exception:
            pass

    # 4) CSV fallback
    try:
        csv_path = os.path.join(CACHE_DIR, f"{symbol.replace('/','_')}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["timestamp"], infer_datetime_format=True)
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            if limit and len(df) > limit:
                df = df.iloc[-limit:].reset_index(drop=True)
            return df
    except Exception:
        logger.exception("CSV fallback load failed for %s", symbol)

    # 5) final: empty DF structure
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

# Settings persistence wrapper
def save_setting_source(key: str, value: Any) -> bool:
    # prefer Postgres storage if available
    try:
        if _pg_storage and hasattr(_pg_storage, "save_setting"):
            try:
                return _pg_storage.save_setting(key, value)
            except Exception:
                logger.exception("pg.save_setting failed")
        if storage_mod and hasattr(storage_mod, "save_setting"):
            try:
                return storage_mod.save_setting(key, value)
            except Exception:
                logger.exception("storage.save_setting failed")
    except Exception:
        logger.exception("save_setting wrapper failed")
    # fallback JSON file in data/config/settings.json
    try:
        data = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf8") as fh:
                data = json.load(fh)
    except Exception:
        data = {}
    data[key] = value
    try:
        with open(SETTINGS_FILE, "w", encoding="utf8") as fh:
            json.dump(data, fh, indent=2, default=str)
        return True
    except Exception:
        logger.exception("fallback save_setting failed")
        return False

def load_setting_source(key: str, default: Any = None) -> Any:
    try:
        if _pg_storage and hasattr(_pg_storage, "load_setting"):
            try:
                return _pg_storage.load_setting(key, default)
            except Exception:
                logger.exception("pg.load_setting failed")
        if storage_mod and hasattr(storage_mod, "load_setting"):
            try:
                return storage_mod.load_setting(key, default)
            except Exception:
                logger.exception("storage.load_setting failed")
    except Exception:
        logger.exception("load_setting wrapper failed")
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf8") as fh:
                data = json.load(fh)
            return data.get(key, default)
    except Exception:
        logger.exception("fallback load_setting failed")
    return default

# -------------------------
# Indicator and score helpers (use core modules if available)
# -------------------------
# detect indicator functions
_enrich_fn = None
_apply_indicators_fn = None
_compute_score_fn = None
_enrich_name = None

if indicators_mod:
    # common function names
    for name in ("enrich_with_indicators_and_score", "apply_indicators", "enrich", "compute_indicators"):
        if hasattr(indicators_mod, name):
            _apply_indicators_fn = getattr(indicators_mod, name)
            _enrich_name = name
            break
    # if module has helper to compute both indicators and score
    for name in ("enrich_with_indicators_and_score",):
        if hasattr(indicators_mod, name):
            _enrich_fn = getattr(indicators_mod, name)
            break

if score_mod:
    # try to detect compute_score and default config
    for name in ("compute_score", "compute_latest_score", "compute_score_timeseries"):
        if hasattr(score_mod, name):
            _compute_score_fn = getattr(score_mod, name)
            break
    if not _compute_score_fn:
        # maybe indicators module includes compute_score too
        if indicators_mod and hasattr(indicators_mod, "compute_score"):
            _compute_score_fn = getattr(indicators_mod, "compute_score")

# safe wrapper to compute indicators + score for a df
def enrich_indicators_and_score(df: pd.DataFrame, indicators: Optional[List[str]] = None, params: Optional[Dict] = None, score_weights: Optional[Dict] = None, lookback: int = 20) -> pd.DataFrame:
    # prefer enrich function if present
    try:
        if _enrich_fn:
            try:
                return _enrich_fn(df, indicators=indicators, params=params, score_weights=score_weights, lookback=lookback)
            except TypeError:
                # older signature
                return _enrich_fn(df)
        if _apply_indicators_fn:
            try:
                # apply indicators then compute score if compute function present
                out = _apply_indicators_fn(df, indicators=indicators, params=params)
            except TypeError:
                out = _apply_indicators_fn(df)
            if _compute_score_fn:
                try:
                    # compute_score may expect df+weights or other signature
                    sc = _compute_score_fn(out, weights=score_weights) if "weights" in _compute_score_fn.__code__.co_varnames else _compute_score_fn(out)
                    out["score"] = sc
                except Exception:
                    logger.exception("compute_score via score_mod failed")
            return out
    except Exception:
        logger.exception("enrich_indicators_and_score failed; falling back to inline computations")

    # Fallback inline basic indicators (safe)
    df = df.copy()
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        # simple EMA defaults
        try:
            df["ema_9"] = close.ewm(span=9, adjust=False).mean()
            df["ema_50"] = close.ewm(span=50, adjust=False).mean()
            # RSI 14
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.ewm(com=13, adjust=False).mean()
            ma_down = down.ewm(com=13, adjust=False).mean()
            rs = ma_up / ma_down.replace(0, np.nan)
            df["rsi_14"] = 100 - (100 / (1 + rs))
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            df["macd"] = macd_line
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_line - macd_signal
        except Exception:
            pass
    # naive score: normalized last returns and rsi
    try:
        score = pd.Series(0.5, index=df.index)
        if "close" in df.columns:
            chg = close.pct_change(10).fillna(0)
            score = 0.5 + 0.5 * np.tanh(chg * 5)
        if "rsi_14" in df.columns:
            score = 0.7 * score + 0.3 * (df["rsi_14"] / 100.0)
        df["score"] = score.clip(0, 1)
    except Exception:
        df["score"] = 0.5
    return df

# -------------------------
# Background runner (thread-based, stores status in st.session_state)
# -------------------------
def run_background(fn, args=(), kwargs=None, task_key=None, on_done=None):
    kwargs = kwargs or {}
    key = f"task_{task_key or str(time.time()).replace('.', '_')}"
    st.session_state.setdefault(key, {"status": "pending", "started_at": None, "finished_at": None, "result": None, "error": None})
    def _worker():
        st.session_state[key]["status"] = "running"
        st.session_state[key]["started_at"] = datetime.utcnow().isoformat()
        try:
            res = fn(*args, **kwargs)
            st.session_state[key]["status"] = "done"
            st.session_state[key]["result"] = res
            st.session_state[key]["finished_at"] = datetime.utcnow().isoformat()
            if on_done:
                try:
                    on_done(res)
                except Exception:
                    logger.exception("on_done callback error")
        except Exception as e:
            st.session_state[key]["status"] = "error"
            st.session_state[key]["error"] = str(e)
            st.session_state[key]["finished_at"] = datetime.utcnow().isoformat()
            logger.exception("Background task error")
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return key

# -------------------------
# UI: page config
# -------------------------
st.set_page_config(page_title="Watchlist", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Sidebar (dynamic and integrated)
# -------------------------
with st.sidebar:
    st.title("Watchlist — Config")
    st.markdown("Ajustes globales, pesos del score y acciones.")

    # Load watchlist count from config files (actions + cryptos) OR from source
    watchlist_config = load_watchlist_from_config()
    # If config empty, try list_assets_source
    if not watchlist_config:
        try:
            watchlist_config = list_assets_source()
        except Exception:
            watchlist_config = ["BTCUSDT", "ETHUSDT", "AAPL", "TSLA"]

    st.markdown(f"**Activos detectados (config):** {len(watchlist_config)}")

    # Score configuration dynamic based on core.score.make_default_score_config()
    score_config = {}
    try:
        if score_mod and hasattr(score_mod, "make_default_score_config"):
            score_config = score_mod.make_default_score_config() or {}
    except Exception:
        logger.exception("make_default_score_config failed")

    # fallback default config (5 components) if score_config missing
    if not score_config:
        score_config = {
            "method": "weighted",
            "components": {
                "momentum": {"weight": 0.25},
                "trend": {"weight": 0.25},
                "volatility": {"weight": 0.2},
                "volume": {"weight": 0.15},
                "liquidity": {"weight": 0.15},
            }
        }

    comp_cfgs = score_config.get("components", {}) or {}

    # Load saved weights and merge
    saved_w = load_setting_source("score_weights", None)
    if saved_w and isinstance(saved_w, dict):
        for k in comp_cfgs.keys():
            if k in saved_w:
                try:
                    comp_cfgs[k]["weight"] = float(saved_w[k])
                except Exception:
                    pass

    st.subheader("Pesos del Score")
    st.markdown("Ajusta los pesos y guarda. Se normalizan al guardar si sum > 0.")

    weights_form = {}
    i = 0
    for comp_name, comp_def in comp_cfgs.items():
        if i >= 30:
            break
        default_w = float(comp_def.get("weight", 0.0))
        w = st.slider(comp_name, 0.0, 1.0, default_w, step=0.01, key=f"wt_{comp_name}")
        weights_form[comp_name] = float(w)
        i += 1

    if st.button("Guardar pesos score"):
        ssum = sum(weights_form.values())
        if ssum > 0:
            normalized = {k: float(v) / ssum for k, v in weights_form.items()}
        else:
            normalized = weights_form
        ok = save_setting_source("score_weights", normalized)
        if ok:
            st.success("Pesos guardados")
        else:
            st.error("Error guardando pesos")
        logger.info("Saved score_weights: %s", normalized)

    st.markdown("---")
    st.subheader("Indicadores (por defecto)")
    # small list of indicator params persisted
    ema_short = int(load_setting_source("ema_short", 9) or 9)
    ema_long = int(load_setting_source("ema_long", 50) or 50)
    rsi_period = int(load_setting_source("rsi_period", 14) or 14)

    new_ema_short = st.number_input("EMA corta", min_value=3, max_value=200, value=ema_short, step=1)
    new_ema_long = st.number_input("EMA larga", min_value=5, max_value=400, value=ema_long, step=1)
    new_rsi = st.number_input("RSI periodo", min_value=2, max_value=100, value=rsi_period, step=1)

    if st.button("Guardar parámetros indicadores"):
        save_setting_source("ema_short", int(new_ema_short))
        save_setting_source("ema_long", int(new_ema_long))
        save_setting_source("rsi_period", int(new_rsi))
        st.success("Parámetros guardados")

    st.markdown("---")
    st.subheader("Operaciones")
    if st.button("Forzar backfill ahora"):
        def _do_backfill():
            # try orchestrator or fetch.run_full_backfill
            
            if orchestrator_mod and hasattr(orchestrator_mod, "run_full_backfill"):
                try:
                    return orchestrator_mod.run_full_backfill()
                except Exception as e:
                    logger.exception("orchestrator.run_full_backfill failed")
                    return {"error": str(e)}
            if fetch_mod and hasattr(fetch_mod, "run_full_backfill"):
                try:
                    return fetch_mod.run_full_backfill()
                except Exception as e:
                    logger.exception("fetch.run_full_backfill failed")
                    return {"error": str(e)}
            # fallback: run list and try load_candles_source per asset
            res = {}
            for s in (watchlist_config[:50] if len(watchlist_config) > 0 else ["BTCUSDT"]):
                try:
                    df = load_candles_source(s, limit=500, force_network=True)
                    res[s] = {"rows": len(df)}
                except Exception as e:
                    res[s] = {"error": str(e)}
            return res
        key = run_background(_do_backfill, task_key="manual_backfill")
        st.info(f"Backfill lanzado en background (task {key})")

    if st.button("Entrenar IA (background)"):
        def _do_train():
            if ai_train_mod:
                # try common signatures
                for fn in ("train", "do_train", "run_train", "train_ai_model"):
                    if hasattr(ai_train_mod, fn):
                        try:
                            return getattr(ai_train_mod, fn)({})
                        except Exception:
                            logger.exception("ai_train.%s failed", fn)
            return {"error": "no_ai_train"}
        key = run_background(_do_train, task_key="train_ai")
        st.info(f"Entrenamiento IA en background (task {key})")

    ia_default = bool(load_setting_source("ia_enabled", False))
    ia_enabled = st.checkbox("Activar IA (persistente)", value=ia_default)
    if st.button("Guardar IA toggle"):
        save_setting_source("ia_enabled", bool(ia_enabled))
        st.success("Guardado IA toggle")

# -------------------------
# Main layout: tabs
# -------------------------
tabs = st.tabs(["Watchlist", "Detalle", "Backtests", "Settings", "System"])
tab_watch, tab_detail, tab_backtests, tab_settings, tab_system = tabs

# -------------------------
# Watchlist tab: table with filters and actions
# -------------------------
with tab_watch:
    st.header("Watchlist")
    st.markdown("Listado de activos con precio, score e indicadores clave. Filtra, ordena y selecciona para ver detalle.")

    # Load assets - prefer config but also try list_assets_source
    assets = watchlist_config if watchlist_config else list_assets_source()
    # ensure dedupe
    seen = set(); assets = [a for a in assets if a not in seen and not seen.add(a)]

    # controls
    c1, c2, c3, c4 = st.columns([3,1,1,1])
    q = c1.text_input("Buscar símbolo", value="", placeholder="ej. BTC, AAPL...")
    min_price = float(c2.number_input("Precio mínimo", value=0.0, step=0.01))
    max_rows = int(c3.selectbox("Mostrar filas", [10,25,50,100], index=1))
    if c4.button("Refrescar"):
        st.rerun()

    # build summary rows by reading last candle (non-blocking loop)
    rows = []
    for s in assets:
        try:
            df2 = load_candles_source(s, limit=2)
            last_price = None
            if isinstance(df2, pd.DataFrame) and not df2.empty and "close" in df2.columns:
                last_price = float(pd.to_numeric(df2["close"].iloc[-1], errors="coerce") or 0.0)
        except Exception:
            last_price = None
        rows.append({"symbol": s, "last_price": last_price})

    df_summary = pd.DataFrame(rows)
    # filter by query and min_price
    if q:
        df_summary = df_summary[df_summary["symbol"].str.contains(q, case=False, na=False)]
    if min_price > 0:
        df_summary = df_summary[df_summary["last_price"].fillna(0) >= float(min_price)]
    df_summary = df_summary.head(max_rows).reset_index(drop=True)

    # Enrich with score quickly (attempt to load indicators for last N rows)
    score_weights = load_setting_source("score_weights", None)
    # compute a lightweight score per asset by loading small history
    enriched_rows = []
    for r in df_summary.to_dict("records"):
        s = r["symbol"]
        try:
            dfc = load_candles_source(s, limit=60)
            if isinstance(dfc, pd.DataFrame) and not dfc.empty:
                df_e = enrich_indicators_and_score(dfc.tail(60), indicators=None, params=None, score_weights=score_weights, lookback=20)
                sc = float(df_e["score"].iloc[-1]) if "score" in df_e.columns and not df_e.empty else None
                vol = None
                if "atr_14" in df_e.columns:
                    try:
                        vol = float(df_e["atr_14"].iloc[-1])
                    except Exception:
                        vol = None
                r.update({"score": sc, "volatility": vol})
            else:
                r.update({"score": None, "volatility": None})
        except Exception:
            r.update({"score": None, "volatility": None})
        enriched_rows.append(r)

    df_summary2 = pd.DataFrame(enriched_rows)
    # allow sorting by score if present
    sort_by = st.selectbox("Ordenar por", options=["score", "last_price", "volatility", "symbol"], index=0 if "score" in df_summary2.columns else 3)
    ascending = st.checkbox("Orden ascendente", value=False)
    if sort_by in df_summary2.columns:
        df_summary2 = df_summary2.sort_values(by=sort_by, ascending=ascending, na_position="last")
    st.dataframe(df_summary2.reset_index(drop=True).style.format({"last_price": lambda v: f"{v:,.6f}" if pd.notna(v) else "—",
                                                                   "score": lambda v: f"{v:.3f}" if pd.notna(v) else "—",
                                                                   "volatility": lambda v: f"{v:.6f}" if pd.notna(v) else "—"}), height=420)

    # actions per row
    st.markdown("**Acciones por activo**")
    for rec in df_summary2.to_dict("records"):
        s = rec["symbol"]
        cols = st.columns([2,2,1,2,2])
        cols[0].markdown(f"**{s}**")
        cols[1].markdown(f"Precio: {rec.get('last_price') if rec.get('last_price') is not None else '—'}")
        if cols[2].button("Ver detalle", key=f"view_{s}"):
            st.session_state["selected_asset"] = s
            st.rerun()
        if cols[3].button("Backtest", key=f"bt_{s}"):
            def _bt_work(sym=s):
                # prefer orchestrator_mod
                if orchestrator_mod:
                    for name in ("run_backtest_for", "run_backtest", "backtest"):
                        if hasattr(orchestrator_mod, name):
                            try:
                                return getattr(orchestrator_mod, name)(sym)
                            except Exception:
                                logger.exception("orchestrator %s failed", name)
                # fallback: simple EMA crossover backtest (quick)
                try:
                    df_hist = load_candles_source(sym, limit=1000)
                    if df_hist is None or df_hist.empty:
                        return {"error": "no_data"}
                    # simple strategy: buy when ema_short > ema_long, sell opposite
                    es = int(load_setting_source("ema_short", 9) or 9)
                    el = int(load_setting_source("ema_long", 50) or 50)
                    close = pd.to_numeric(df_hist["close"], errors="coerce")
                    ema_s = close.ewm(span=es, adjust=False).mean()
                    ema_l = close.ewm(span=el, adjust=False).mean()
                    pos = (ema_s > ema_l).astype(int)
                    # compute returns
                    ret = close.pct_change().fillna(0)
                    strat_ret = ret * pos.shift(1).fillna(0)
                    cum = (1 + strat_ret).cumprod() - 1
                    total_return = float(cum.iloc[-1]) if len(cum) > 0 else 0.0
                    # drawdown
                    peak = (1 + strat_ret).cumprod().cummax()
                    dd = ((1 + strat_ret).cumprod() / peak - 1).min()
                    winrate = float((strat_ret > 0).sum() / max(1, (strat_ret != 0).sum()))
                    metrics = {"return": total_return, "drawdown": float(dd), "winrate": winrate}
                    # save result
                    fname = os.path.join(BACKTESTS_DIR, f"{sym.replace('/','_')}_bt_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
                    with open(fname, "w", encoding="utf8") as fh:
                        json.dump({"metrics": metrics, "symbol": sym, "generated_at": datetime.utcnow().isoformat()}, fh, default=str, indent=2)
                    return {"metrics": metrics, "saved": fname}
                except Exception as e:
                    logger.exception("fallback backtest failed for %s", sym)
                    return {"error": str(e)}
            key = run_background(_bt_work, task_key=f"backtest_{s}")
            st.info(f"Backtest lanzado (task {key})")
        if cols[4].button("Actualizar datos", key=f"upd_{s}"):
            def _upd(sym=s):
                df = load_candles_source(sym, limit=1000, force_network=True)
                return {"rows": len(df) if isinstance(df, pd.DataFrame) else 0}
            key = run_background(_upd, task_key=f"update_{s}")
            st.info(f"Actualización en background (task {key})")

# -------------------------
# Detalle tab: show selected asset detail
# -------------------------
with tab_detail:
    st.header("Detalle del activo")
    selected = st.session_state.get("selected_asset", None)
    if not selected:
        st.info("Selecciona un activo desde la pestaña Watchlist para ver detalle.")
    else:
        st.subheader(selected)
        # load a generous range of candles
        dfc = load_candles_source(selected, limit=500)
        if dfc is None or dfc.empty:
            st.warning("No hay datos históricos para este activo. Sube CSV a data/cache/ o habilita fetch/storage.")
        else:
            # enrich with indicators + score
            score_weights = load_setting_source("score_weights", None)
            params_ind = {"ema_short": int(load_setting_source("ema_short", 9)), "ema_long": int(load_setting_source("ema_long", 50)), "rsi_period": int(load_setting_source("rsi_period", 14))}
            try:
                df_enriched = enrich_indicators_and_score(dfc, indicators=None, params=params_ind, score_weights=score_weights, lookback=30)
            except Exception:
                df_enriched = dfc.copy()
            # plot candles + indicators
            try:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df_enriched["timestamp"], open=df_enriched["open"], high=df_enriched["high"], low=df_enriched["low"], close=df_enriched["close"], name="Candles"), row=1, col=1)
                # EMA lines if present
                for col in df_enriched.columns:
                    if col.startswith("ema_"):
                        fig.add_trace(go.Scatter(x=df_enriched["timestamp"], y=df_enriched[col], name=col, line=dict(width=1.5)), row=1, col=1)
                # RSI in second panel if present
                rsi_cols = [c for c in df_enriched.columns if c.startswith("rsi")]
                if rsi_cols:
                    fig.add_trace(go.Scatter(x=df_enriched["timestamp"], y=df_enriched[rsi_cols[0]], name=rsi_cols[0]), row=2, col=1)
                else:
                    if "close" in df_enriched.columns:
                        # plot volume if exists
                        if "volume" in df_enriched.columns:
                            fig.add_trace(go.Bar(x=df_enriched["timestamp"], y=df_enriched["volume"], showlegend=False), row=2, col=1)
                fig.update_layout(height=700, margin=dict(l=8,r=8,t=30,b=8), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                logger.exception("plotting failed for %s", selected)
                st.write("Error plotting — revisa logs")

            # show recent indicators and score
            last_rows = df_enriched.tail(10).copy()
            st.subheader("Últimas filas (indicadores)")
            display_cols = [c for c in last_rows.columns if c in ["timestamp","close","volume"] or c.startswith("ema_") or c.startswith("rsi") or c.startswith("macd") or c.startswith("atr") or c=="score"]
            st.dataframe(last_rows[display_cols].reset_index(drop=True).style.format({"close": "{:,.6f}"}), height=260)

            # actions: download CSV, infer IA, run backtest
            a1, a2, a3 = st.columns([1,1,1])
            if a1.button("Descargar CSV"):
                csv = df_enriched.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar CSV", data=csv, file_name=f"{selected}_candles.csv", mime="text/csv")
            if a2.button("Infer IA (background)"):
                def _do_infer(sym=selected):
                    if ai_inf_mod:
                        for fn in ("predict", "infer", "run_predict"):
                            if hasattr(ai_inf_mod, fn):
                                try:
                                    return getattr(ai_inf_mod, fn)(sym)
                                except Exception:
                                    logger.exception("ai_inference.%s failed", fn)
                    return {"error": "no_ai_infer"}
                key = run_background(_do_infer, task_key=f"infer_{selected}")
                st.info(f"Inferencia lanzada (task {key})")
            if a3.button("Backtest (background)"):
                # reuse backtest action from watchlist (could call orchestrator)
                def _do_bt(sym=selected):
                    if orchestrator_mod:
                        for name in ("run_backtest_for", "run_backtest", "backtest"):
                            if hasattr(orchestrator_mod, name):
                                try:
                                    return getattr(orchestrator_mod, name)(sym)
                                except Exception:
                                    logger.exception("orchestrator %s failed", name)
                    # fallback simple backtest (same as above)
                    df_hist = load_candles_source(sym, limit=1000)
                    if df_hist is None or df_hist.empty:
                        return {"error": "no_data"}
                    es = int(load_setting_source("ema_short", 9))
                    el = int(load_setting_source("ema_long", 50))
                    close = pd.to_numeric(df_hist["close"], errors="coerce")
                    ema_s = close.ewm(span=es, adjust=False).mean()
                    ema_l = close.ewm(span=el, adjust=False).mean()
                    pos = (ema_s > ema_l).astype(int)
                    ret = close.pct_change().fillna(0)
                    strat_ret = ret * pos.shift(1).fillna(0)
                    cum = (1 + strat_ret).cumprod() - 1
                    total_return = float(cum.iloc[-1]) if len(cum)>0 else 0.0
                    peak = (1 + strat_ret).cumprod().cummax()
                    dd = ((1 + strat_ret).cumprod() / peak - 1).min()
                    winrate = float((strat_ret>0).sum() / max(1, (strat_ret!=0).sum()))
                    metrics = {"return": total_return, "drawdown": float(dd), "winrate": winrate}
                    fname = os.path.join(BACKTESTS_DIR, f"{sym.replace('/','_')}_bt_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
                    with open(fname, "w", encoding="utf8") as fh:
                        json.dump({"metrics": metrics, "symbol": sym, "generated_at": datetime.utcnow().isoformat()}, fh, default=str, indent=2)
                    return {"metrics": metrics, "saved": fname}
                key = run_background(_do_bt, task_key=f"backtest_{selected}")
                st.info(f"Backtest lanzado (task {key})")

# -------------------------
# Backtests tab: list saved backtests and show results
# -------------------------
with tab_backtests:
    st.header("Backtests guardados")
    files = sorted([f for f in os.listdir(BACKTESTS_DIR) if f.endswith(".json")], reverse=True)
    if not files:
        st.info("No hay backtests guardados.")
    else:
        sel = st.selectbox("Selecciona resultado", options=files)
        if sel:
            path = os.path.join(BACKTESTS_DIR, sel)
            try:
                with open(path, "r", encoding="utf8") as fh:
                    data = json.load(fh)
                st.json(data)
                if st.button("Descargar JSON"):
                    with open(path, "rb") as fh:
                        st.download_button("Descargar", fh.read(), file_name=sel, mime="application/json")
            except Exception:
                logger.exception("reading backtest file failed")

# -------------------------
# Settings tab (show settings & allow manual edits)
# -------------------------
with tab_settings:
    st.header("Ajustes y estado")
    st.subheader("Settings persistentes")
    try:
        sdata = load_setting_source("score_weights", None)
        st.write("score_weights:", sdata)
    except Exception:
        st.write("No se pudo leer score_weights")

    st.subheader("Variables de entorno y módulos detectados")
    mod_status = {
        "fetch_mod": bool(fetch_mod),
        "indicators_mod": bool(indicators_mod),
        "score_mod": bool(score_mod),
        "orchestrator_mod": bool(orchestrator_mod),
        "ai_train_mod": bool(ai_train_mod),
        "ai_inf_mod": bool(ai_inf_mod),
        "storage_mod": bool(storage_mod),
        "storage_pg": bool(storage_pg_mod),
    }
    st.json(mod_status)

# -------------------------
# System tab: health, tasks
# -------------------------
with tab_system:
    st.header("Sistema & Tareas")
    st.subheader("Health")
    try:
        health = {}
        if fetch_mod and hasattr(fetch_mod, "health_check"):
            try:
                health = fetch_mod.health_check()
            except Exception:
                logger.exception("fetch_mod.health_check failed")
        else:
            # basic health
            health = {
                "time": datetime.utcnow().isoformat(),
                "modules": {
                    "fetch": bool(fetch_mod),
                    "indicators": bool(indicators_mod),
                    "score": bool(score_mod),
                    "orchestrator": bool(orchestrator_mod),
                    "ai_train": bool(ai_train_mod),
                    "ai_infer": bool(ai_inf_mod),
                    "pg_storage": bool(_pg_storage),
                },
                "cache_files": len([f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]) if os.path.exists(CACHE_DIR) else 0
            }
        st.json(health)
    except Exception:
        logger.exception("health display failed")

    st.subheader("Tareas en esta sesión")
    tasks_keys = sorted([k for k in st.session_state.keys() if k.startswith("task_")])
    if not tasks_keys:
        st.info("No hay tareas en background en esta sesión.")
    else:
        for tk in tasks_keys:
            v = st.session_state[tk]
            st.markdown(f"**{tk}** — {v.get('status')} — started: {v.get('started_at')} finished: {v.get('finished_at')}")
            st.json(v)

# -------------------------
# End
# -------------------------
# ----------------------------
# Snippet Streamlit: mostrar Stop/Target y Visualizador de logs
# Pega dentro de dashboard/app.py en la zona de detalle de activo
# ----------------------------
import streamlit as st
from core.adapter import adapter as core_adapter  # singleton adapter
from core.score import compute_stop_target_for_asset

st.markdown("### Stop / Target (ATR-based)")

col1, col2, col3 = st.columns(3)
asset_sel = st.session_state.get("selected_asset", None)  # ajusta según la variable que uses
interval_sel = st.session_state.get("selected_interval", "1h")

if asset_sel:
    if st.button("Calcular stop/target (ATR)"):
        with st.spinner("Calculando..."):
            res = compute_stop_target_for_asset(core_adapter, asset_sel, interval_sel, lookback=200)
            if res.get("error"):
                st.error(f"Error: {res['error']}")
            else:
                col1.metric("Entry", f"{res.get('entry', 'n/a'):.6f}" if res.get("entry") else "n/a")
                col2.metric("Stop", f"{res.get('stop', 'n/a'):.6f}" if res.get("stop") else "n/a")
                col3.metric("Target", f"{res.get('target', 'n/a'):.6f}" if res.get("target") else "n/a")
                st.write(f"ATR (window 14): {res.get('atr')}")
                st.write(f"Dirección inferida: {res.get('direction')}")
else:
    st.info("Selecciona un activo para calcular stop/target")

# Simple visualizador de tail de logs (comandos y errores)
st.markdown("### Visualizador de Logs (últimas líneas)")
LOG_PATH = "logs/app.log"  # ajusta si tu app usa otro log
try:
    with open(LOG_PATH, "r", errors="ignore") as fh:
        lines = fh.readlines()[-300:]  # últimas 300 líneas
    st.code("".join(lines[-200:]))  # muestra las últimas 200 líneas
except FileNotFoundError:
    st.write("No se encontró el fichero de logs en", LOG_PATH)

# ----------------------------
# Mostrar logs en memoria + fichero
# Pegar antes de st.markdown("---")
# ----------------------------
import streamlit as st
from collections import deque

st.markdown("### Logs en memoria (lo que produce este proceso)")
show_mem = list(INMEM_LOG)[-200:]
if show_mem:
    st.code("\n".join(show_mem))
else:
    st.write("No hay logs en memoria todavía para este proceso")

st.markdown("### Logs desde fichero (workers / procesos externos)")
# Reusar la función tail_file del snippet A (cópiala arriba si no existe)
log_choice = "logs/app.log"
nlines = st.sidebar.slider("Líneas fichero", 50, 500, 300)
if st.button("Refresh fichero"):
    lines = tail_file(log_choice, nlines)
    st.code("\n".join(lines[-nlines:]))
else:
    lines = tail_file(log_choice, 200)
    st.code("\n".join(lines[-nlines:]))

# ----------------------------
# UI helpers: trigger backfill and show simple text logs
# ----------------------------
import logging
import importlib
from datetime import datetime

_logger = logging.getLogger(__name__)

def ui_trigger_backfill(symbols=None, per_symbol_limit=1000):
    """
    Called from the UI to trigger backfill. Uses orchestrator_mod or fetch_mod if available.
    Returns a dict-like status.
    """
    try:
        # prefer orchestrator if present
        try:
            orchestrator_mod = importlib.import_module("core.orchestrator")
        except Exception:
            orchestrator_mod = None
        try:
            fetch_mod = importlib.import_module("core.fetch")
        except Exception:
            fetch_mod = None

        if orchestrator_mod and hasattr(orchestrator_mod, "run_full_backfill"):
            try:
                return orchestrator_mod.run_full_backfill(symbols=symbols, per_symbol_limit=per_symbol_limit)
            except Exception:
                _logger.exception("orchestrator.run_full_backfill failed")
        if fetch_mod and hasattr(fetch_mod, "run_full_backfill"):
            try:
                return fetch_mod.run_full_backfill(symbols=symbols, per_symbol_limit=per_symbol_limit)
            except Exception:
                _logger.exception("fetch.run_full_backfill failed")
        if fetch_mod and hasattr(fetch_mod, "safe_run_full_backfill"):
            try:
                return fetch_mod.safe_run_full_backfill(symbols=symbols, per_symbol_limit=per_symbol_limit)
            except Exception:
                _logger.exception("fetch.safe_run_full_backfill failed")
    except Exception:
        _logger.exception("ui_trigger_backfill failed")
    return {"error": "no_backfill_available", "timestamp": datetime.utcnow().isoformat()}

def tail_log_file(path: str, nlines: int = 200):
    """
    Small helper to tail a text file (best-effort). Returns list of lines.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            size = 1024
            data = b""
            while len(data.splitlines()) <= nlines and end > 0:
                read_size = min(size, end)
                f.seek(end - read_size)
                chunk = f.read(read_size)
                data = chunk + data
                end -= read_size
            lines = data.splitlines()[-nlines:]
            return [l.decode("utf-8", errors="replace") for l in lines]
    except Exception:
        _logger.exception("tail_log_file failed for %s", path)
        return []

# ----------------------------
# Compat: alias tail_file -> tail_log_file (backwards compatibility)
# ----------------------------
def tail_file(path: str, nlines: int = 200):
    """
    Backwards-compatible wrapper expected by older code.
    - Si existe tail_log_file en este módulo lo reutiliza.
    - Si no, hace un tail best-effort del fichero.
    """
    try:
        # If tail_log_file exists and is callable, use it
        if "tail_log_file" in globals() and callable(globals()["tail_log_file"]):
            try:
                return globals()["tail_log_file"](path, nlines)
            except Exception:
                # fall through to local fallback
                pass
    except Exception:
        pass

    # Best-effort fallback tail implementation (binary-safe)
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            block_size = 1024
            data = b""
            while file_size > 0 and len(data.splitlines()) <= nlines:
                read_size = min(block_size, file_size)
                f.seek(file_size - read_size)
                chunk = f.read(read_size)
                data = chunk + data
                file_size -= read_size
            lines = data.splitlines()[-nlines:]
            return [l.decode("utf-8", errors="replace") for l in lines]
    except FileNotFoundError:
        # If file doesn't exist, return empty list (UI can handle)
        return []
    except Exception:
        # last-resort: avoid raising to keep UI running
        try:
            return []
        except Exception:
            return []

st.markdown("---")



