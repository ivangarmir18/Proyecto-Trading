# dashboard/app.py
"""
Dashboard Watchlist - versión organizada y robusta.
Sustituye el fichero anterior por este.

Instrucciones:
 - Haz backup: copy dashboard\\app.py dashboard\\app.py.bak
 - Guarda este fichero y ejecuta: streamlit run dashboard/app.py
"""
from __future__ import annotations
import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import deque

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Paths & basic setup
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
CONF_DIR = os.path.join(DATA_DIR, "config")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DB_DIR = os.path.join(DATA_DIR, "db")
BACKTESTS_DIR = os.path.join(DB_DIR, "backtests")
SETTINGS_FILE = os.path.join(CONF_DIR, "settings.json")
LOG_PATH = os.path.join("logs", "app.log")

os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(BACKTESTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

# ----------------------------
# Logging (file + in-memory deque for UI)
# ----------------------------
INMEM_LOG = deque(maxlen=1000)
class StreamlitMemoryHandler(logging.Handler):
    def emit(self, record):
        try:
            INMEM_LOG.append(self.format(record))
        except Exception:
            pass

root_logger = logging.getLogger()
if not root_logger.handlers:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(LOG_PATH, maxBytes=10*1024*1024, backupCount=5)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)
    except Exception:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root_logger.addHandler(sh)
    memh = StreamlitMemoryHandler()
    memh.setFormatter(fmt)
    root_logger.addHandler(memh)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger("watchlist.dashboard")

# ----------------------------
# Safe import helper
# ----------------------------
def safe_import(module_name: str, attr: Optional[str] = None):
    try:
        m = __import__(module_name, fromlist=[attr] if attr else [])
        if attr:
            return getattr(m, attr)
        return m
    except Exception:
        return None

# ----------------------------
# Detect core modules (tolerant)
# ----------------------------
fetch_mod = safe_import("core.fetch")
indicators_mod = safe_import("core.indicators")
score_mod = safe_import("core.score")
orchestrator_mod = safe_import("core.orchestrator")
ai_train_mod = safe_import("core.ai_train")
ai_inf_mod = safe_import("core.ai_inference")
storage_mod = safe_import("core.storage")
storage_pg_mod = safe_import("core.storage_postgres")
adapter_mod = safe_import("core.adapter")

# Try to obtain a singleton adapter object (if core.adapter exposes `adapter` or `StorageAdapter`)
core_adapter = None
try:
    if adapter_mod and hasattr(adapter_mod, "adapter"):
        core_adapter = getattr(adapter_mod, "adapter")
    elif adapter_mod and hasattr(adapter_mod, "StorageAdapter"):
        # maybe adapter exposes class; try to instantiate safely
        try:
            core_adapter = getattr(adapter_mod, "StorageAdapter")()
        except Exception:
            core_adapter = None
except Exception:
    core_adapter = None

# Try to instantiate PostgresStorage if provided
_pg_storage = None
if storage_pg_mod:
    try:
        if hasattr(storage_pg_mod, "PostgresStorage"):
            PgCls = getattr(storage_pg_mod, "PostgresStorage")
            try:
                _pg_storage = PgCls()
                logger.info("PostgresStorage instantiated for dashboard.")
            except Exception:
                _pg_storage = None
    except Exception:
        _pg_storage = None

# helper from score module
compute_stop_target_for_asset = None
if score_mod and hasattr(score_mod, "compute_stop_target_for_asset"):
    compute_stop_target_for_asset = getattr(score_mod, "compute_stop_target_for_asset")

# ----------------------------
# tail_file utility (safe)
# ----------------------------
def tail_file(path: str, n: int = 200) -> List[str]:
    """Return last n lines from file path (best-effort)."""
    try:
        if not path or not os.path.exists(path):
            return []
        with open(path, "rb") as f:
            f.seek(0, 2)
            filesize = f.tell()
            block_size = 1024
            data = b""
            while len(data.splitlines()) <= n and filesize > 0:
                read_from = max(0, filesize - block_size)
                f.seek(read_from)
                chunk = f.read(filesize - read_from)
                data = chunk + data
                filesize = read_from
                block_size *= 2
            lines = data.splitlines()[-n:]
            return [ln.decode(errors="replace") for ln in lines]
    except Exception:
        logger.exception("tail_file failed for %s", path)
        return []

# ----------------------------
# Watchlist loaders & storage wrappers
# ----------------------------
def load_watchlist_from_config() -> List[str]:
    files = ["actions.csv", "cryptos.csv", "crypto.csv"]
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
                logger.exception("Failed reading %s", path)
    seen = set(); unique = []
    for s in out:
        if s not in seen:
            seen.add(s); unique.append(s)
    return unique

def list_assets_source() -> List[str]:
    # try fetch module
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
                            return res.iloc[:,0].astype(str).tolist()
                    except Exception:
                        logger.exception("fetch_mod.%s failed", fn)
        # try Postgres storage
        if _pg_storage and hasattr(_pg_storage, "list_assets"):
            try:
                res = _pg_storage.list_assets()
                if isinstance(res, list):
                    return res
            except Exception:
                logger.exception("pg.list_assets failed")
        # try storage_mod
        if storage_mod and hasattr(storage_mod, "list_assets"):
            try:
                res = storage_mod.list_assets()
                if isinstance(res, list):
                    return res
            except Exception:
                logger.exception("storage.list_assets failed")
    except Exception:
        logger.exception("list_assets_source unexpected error")
    cfg = load_watchlist_from_config()
    if cfg:
        return cfg
    return ["BTCUSDT", "ETHUSDT", "AAPL", "TSLA"]

# Candle loader wrapper (tries storage -> fetch -> csv)
def load_candles_source(symbol: str, limit: int = 1000, timeframe: Optional[str] = None, force_network: bool = False) -> pd.DataFrame:
    # 1) try Postgres storage
    try:
        if not force_network and _pg_storage and hasattr(_pg_storage, "load_candles"):
            df = _pg_storage.load_candles(symbol, limit=limit)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        if not force_network and storage_mod and hasattr(storage_mod, "load_candles"):
            try:
                df = storage_mod.load_candles(symbol, limit=limit)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception:
                logger.debug("storage_mod.load_candles failed for %s", symbol)
    except Exception:
        logger.exception("storage load attempt failed")

    # 2) try fetch_mod
    tried_network = False
    try:
        if fetch_mod:
            for fn in ("get_candles","fetch_candles","get_latest_candles","fetch_multi","get_ohlcv","fetch"):
                if hasattr(fetch_mod, fn):
                    f = getattr(fetch_mod, fn)
                    try:
                        # try multiple signatures
                        try:
                            df = f(symbol, timeframe) if "timeframe" in f.__code__.co_varnames else f(symbol)
                        except TypeError:
                            try:
                                df = f(symbol, limit)
                            except Exception:
                                df = f(symbol)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            tried_network = True
                            # persist best-effort
                            try:
                                if _pg_storage and hasattr(_pg_storage, "upsert_candles"):
                                    rows = df.to_dict("records")
                                    _pg_storage.upsert_candles(symbol, timeframe or "1m", rows)
                                elif storage_mod and hasattr(storage_mod, "save_candles"):
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

    # 3) if network tried, check storage again
    if tried_network:
        try:
            if _pg_storage and hasattr(_pg_storage, "load_candles"):
                df = _pg_storage.load_candles(symbol, limit=limit)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            if storage_mod and hasattr(storage_mod, "load_candles"):
                df = storage_mod.load_candles(symbol, limit=limit)
                if isinstance(df, pd.DataFrame) and not df.empty:
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

    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# ----------------------------
# Settings persistence wrappers
# ----------------------------
def save_setting_source(key: str, value: Any) -> bool:
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

    # fallback JSON
    data = {}
    try:
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

# ----------------------------
# Indicators & score enrichment wrapper (uses core modules if available)
# ----------------------------
_apply_indicators_fn = None
_enrich_fn = None
_compute_score_fn = None

if indicators_mod:
    for name in ("apply_indicators", "enrich", "compute_indicators", "enrich_with_indicators_and_score"):
        if hasattr(indicators_mod, name):
            _apply_indicators_fn = getattr(indicators_mod, name)
            break
    if hasattr(indicators_mod, "enrich_with_indicators_and_score"):
        _enrich_fn = getattr(indicators_mod, "enrich_with_indicators_and_score")

if score_mod:
    for name in ("compute_scores_from_df","compute_score_timeseries","compute_latest_score","compute_score"):
        if hasattr(score_mod, name):
            _compute_score_fn = getattr(score_mod, name)
            break

def enrich_indicators_and_score(df: pd.DataFrame, indicators: Optional[List[str]] = None, params: Optional[Dict] = None, score_weights: Optional[Dict] = None, lookback: int = 20) -> pd.DataFrame:
    try:
        if _enrich_fn:
            try:
                return _enrich_fn(df, indicators=indicators, params=params, score_weights=score_weights, lookback=lookback)
            except TypeError:
                return _enrich_fn(df)
        if _apply_indicators_fn:
            try:
                out = _apply_indicators_fn(df, indicators=indicators, params=params)
            except TypeError:
                out = _apply_indicators_fn(df)
            if _compute_score_fn:
                try:
                    sc = _compute_score_fn(out, weights=score_weights) if "weights" in _compute_score_fn.__code__.co_varnames else _compute_score_fn(out)
                    out["score"] = sc
                except Exception:
                    logger.exception("compute_score via score_mod failed")
            return out
    except Exception:
        logger.exception("enrich_indicators_and_score failed; falling back to inline computations")

    # Inline fallback indicators + naive score
    df = df.copy()
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        try:
            df["ema_9"] = close.ewm(span=9, adjust=False).mean()
            df["ema_50"] = close.ewm(span=50, adjust=False).mean()
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.ewm(com=13, adjust=False).mean()
            ma_down = down.ewm(com=13, adjust=False).mean()
            rs = ma_up / ma_down.replace(0, np.nan)
            df["rsi_14"] = 100 - (100 / (1 + rs))
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            df["macd"] = macd_line
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_line - macd_signal
        except Exception:
            pass
    try:
        score = pd.Series(0.5, index=df.index)
        if "close" in df.columns:
            chg = close.pct_change(10).fillna(0)
            score = 0.5 + 0.5 * np.tanh(chg * 5)
        if "rsi_14" in df.columns:
            score = 0.7 * score + 0.3 * (df["rsi_14"] / 100.0)
        df["score"] = score.clip(0,1)
    except Exception:
        df["score"] = 0.5
    return df

# ----------------------------
# Background runner (threads) - stores status in st.session_state
# ----------------------------
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

# ----------------------------
# Streamlit UI configuration
# ----------------------------
st.set_page_config(page_title="Watchlist", layout="wide", initial_sidebar_state="expanded")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("Watchlist — Config")
    st.markdown("Ajustes globales, pesos del score y acciones.")

    watchlist_config = load_watchlist_from_config()
    if not watchlist_config:
        try:
            watchlist_config = list_assets_source()
        except Exception:
            watchlist_config = ["BTCUSDT", "ETHUSDT", "AAPL", "TSLA"]
    st.markdown(f"**Activos detectados (config):** {len(watchlist_config)}")

    # Score config (try core.score)
    score_config = {}
    try:
        if score_mod and hasattr(score_mod, "make_default_score_config"):
            score_config = score_mod.make_default_score_config() or {}
    except Exception:
        logger.exception("make_default_score_config failed")
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

    # ---------------------------
    # Operaciones (Backfill + Auto-update + Train)
    # reemplazar el bloque "Operaciones" existente en el sidebar por este
    # ---------------------------
    st.markdown("---")
    st.subheader("Operaciones")
    
    # helper robusto que invoca la mejor función de backfill disponible
    def invoke_backfill(symbols=None, per_symbol_limit=1000, timeout=None):
        """
        Intenta, por orden:
         1) orchestrator.run_full_backfill_safe(...) o run_full_backfill(...)
         2) instanciar core.orchestrator.Orchestrator() y llamar run_full_backfill_safe
         3) fetch_mod.run_full_backfill(...)
         4) fallback: iterar assets y load_candles_source(force_network=True)
        Devuelve dict con resumen por símbolo o error.
        """
        res = {"started_at": datetime.utcnow().isoformat(), "per_symbol_limit": per_symbol_limit}
        try:
            # 1) módulo orchestrator con función safe
            if orchestrator_mod:
                # prefer direct function run_full_backfill_safe
                for fn in ("run_full_backfill_safe","run_full_backfill","run_full_backfill_safe_sync","run_full_backfill_sync","run_full_backfill"):
                    if hasattr(orchestrator_mod, fn):
                        try:
                            f = getattr(orchestrator_mod, fn)
                            logger.info("Invocando orchestrator_mod.%s()", fn)
                            out = f(symbols=symbols, per_symbol_limit=per_symbol_limit) if "symbols" in f.__code__.co_varnames else f()
                            res["result_from"] = f"orchestrator_mod.{fn}"
                            res["result"] = out
                            return res
                        except Exception:
                            logger.exception("orchestrator_mod.%s failed", fn)
                # try instantiating class Orchestrator
                if hasattr(orchestrator_mod, "Orchestrator"):
                    try:
                        Or = getattr(orchestrator_mod, "Orchestrator")
                        inst = Or()
                        if hasattr(inst, "run_full_backfill_safe"):
                            logger.info("Invocando Orchestrator().run_full_backfill_safe()")
                            out = inst.run_full_backfill_safe(symbols=symbols, per_symbol_limit=per_symbol_limit)
                            res["result_from"] = "Orchestrator.run_full_backfill_safe"
                            res["result"] = out
                            return res
                        if hasattr(inst, "run_full_backfill"):
                            out = inst.run_full_backfill(symbols=symbols, per_symbol_limit=per_symbol_limit)
                            res["result_from"] = "Orchestrator.run_full_backfill"
                            res["result"] = out
                            return res
                    except Exception:
                        logger.exception("Instanciación/ejecución Orchestrator failed")
            # 2) fetch_mod fallback
            if fetch_mod:
                for fn in ("run_full_backfill","safe_run_full_backfill","run_full_backfill_safe"):
                    if hasattr(fetch_mod, fn):
                        try:
                            f = getattr(fetch_mod, fn)
                            logger.info("Invocando fetch_mod.%s()", fn)
                            out = f(symbols=symbols, per_symbol_limit=per_symbol_limit)
                            res["result_from"] = f"fetch_mod.{fn}"
                            res["result"] = out
                            return res
                        except Exception:
                            logger.exception("fetch_mod.%s failed", fn)
            # 3) fallback: per-symbol force network (más lento, robusto)
            logger.info("Fallback network per-symbol backfill (force_network). Symbols: %s", symbols)
            summary = {}
            use_list = symbols if symbols else (watchlist_config if watchlist_config else list_assets_source())
            for s in use_list:
                try:
                    df = load_candles_source(s, limit=per_symbol_limit, force_network=True)
                    summary[s] = {"rows": len(df) if isinstance(df, pd.DataFrame) else 0}
                except Exception as e:
                    summary[s] = {"error": str(e)}
            res["result_from"] = "per_symbol_force_network"
            res["result"] = summary
            return res
        except Exception as e:
            logger.exception("invoke_backfill unexpected error")
            return {"error": str(e)}
    
    # background wrapper that stores result and log into session_state
    def ui_backfill_background(symbols=None, per_symbol_limit=1000, task_key=None):
        def _fn():
            out = invoke_backfill(symbols=symbols, per_symbol_limit=per_symbol_limit)
            # store in session_state and local json for outside processes
            key = f"backfill_result_{int(time.time())}"
            try:
                st.session_state.setdefault("backfill_results", {})
                st.session_state["backfill_results"][key] = out
            except Exception:
                # in some Streamlit versions threads cannot mutate session_state reliably;
                # as fallback escribimos un fichero temporal con el resultado
                try:
                    tmp = os.path.join(DATA_DIR, f"backfill_result_{int(time.time())}.json")
                    with open(tmp, "w", encoding="utf8") as fh:
                        json.dump(out, fh, indent=2, default=str)
                    logger.info("Wrote fallback backfill result to %s", tmp)
                except Exception:
                    logger.exception("fallback write backfill result failed")
            return out
        return run_background(_fn, task_key=(task_key or f"ui_backfill_{int(time.time())}"))
    
    # Manual backfill button (reliable)
    if st.button("Forzar backfill ahora"):
        key = ui_backfill_background(symbols=None, per_symbol_limit=1000, task_key="manual_backfill")
        st.info(f"Backfill lanzado en background (task {key}). Revisa pestaña System -> Tareas o Logs para ver progreso.")
    
    # Auto-update toggle (runs backfill every 5 minutes while enabled)
    auto_flag = st.checkbox("Auto-update (backfill cada 5 min mientras estás en la sesión)", value=bool(st.session_state.get("auto_backfill_enabled", False)))
    st.session_state["auto_backfill_enabled"] = bool(auto_flag)
    
    # Background thread controller (start only once per session)
    def _auto_backfill_loop():
        logger.info("Auto-backfill loop started")
        while st.session_state.get("auto_backfill_enabled", False):
            try:
                # run one backfill cycle
                logger.info("Auto-backfill cycle triggered")
                out_key = ui_backfill_background(symbols=None, per_symbol_limit=500, task_key="auto_backfill")
                # sleep 300s but exit early if disabled
                sleep_seconds = 300
                for _ in range(int(sleep_seconds/5)):
                    if not st.session_state.get("auto_backfill_enabled", False):
                        break
                    time.sleep(5)
            except Exception:
                logger.exception("auto_backfill_loop unexpected error")
                time.sleep(10)
        logger.info("Auto-backfill loop finished")
    
    # start auto thread if flagged and not already running
    if st.session_state.get("auto_backfill_enabled", False) and not st.session_state.get("auto_backfill_thread_alive", False):
        # start a daemon thread
        try:
            t = threading.Thread(target=_auto_backfill_loop, daemon=True)
            t.start()
            st.session_state["auto_backfill_thread_alive"] = True
            st.success("Auto-backfill activado (ciclos cada 5 minutos).")
        except Exception:
            logger.exception("Could not start auto_backfill thread")
            st.error("No se pudo iniciar auto-backfill")
    
    # quick action: force UI to refresh cached summary after backfill completes
    if st.button("Refrescar datos en UI (releer candles)"):
        # mark timestamp used in main loop to reload data
        st.session_state["ui_force_reload_ts"] = int(time.time())
        st.experimental_rerun()
    
    # Train IA (background) preserved (no cambios en comportamiento)
    if st.button("Entrenar IA (background)"):
        def _do_train():
            if ai_train_mod:
                for fn in ("train","do_train","run_train","train_ai_model"):
                    if hasattr(ai_train_mod, fn):
                        try:
                            return getattr(ai_train_mod, fn)({})
                        except Exception:
                            logger.exception("ai_train.%s failed", fn)
            return {"error":"no_ai_train"}
        key = run_background(_do_train, task_key="train_ai")
        st.info(f"Entrenamiento IA en background (task {key})")


# ----------------------------
# Main tabs
# ----------------------------
tabs = st.tabs(["Watchlist", "Detalle", "Backtests", "Settings", "System"])
tab_watch, tab_detail, tab_backtests, tab_settings, tab_system = tabs

# ----------------------------
# Watchlist tab
# ----------------------------
with tab_watch:
    st.header("Watchlist")
    st.markdown("Listado de activos con precio, score e indicadores clave.")

    assets = watchlist_config if watchlist_config else list_assets_source()
    seen = set(); assets = [a for a in assets if a not in seen and not seen.add(a)]

    c1, c2, c3, c4 = st.columns([3,1,1,1])
    q = c1.text_input("Buscar símbolo", value="", placeholder="ej. BTC, AAPL...")
    min_price = float(c2.number_input("Precio mínimo", value=0.0, step=0.01))
    max_rows = int(c3.selectbox("Mostrar filas", [10,25,50,100], index=1))
    if c4.button("Refrescar"):
        st.experimental_rerun()

    rows = []
    for s in assets:
        last_price = None
        try:
            df2 = load_candles_source(s, limit=2)
            if isinstance(df2, pd.DataFrame) and not df2.empty and "close" in df2.columns:
                last_price = float(pd.to_numeric(df2["close"].iloc[-1], errors="coerce") or 0.0)
        except Exception:
            last_price = None
        rows.append({"symbol": s, "last_price": last_price})

    df_summary = pd.DataFrame(rows)
    if q:
        df_summary = df_summary[df_summary["symbol"].str.contains(q, case=False, na=False)]
    if min_price > 0:
        df_summary = df_summary[df_summary["last_price"].fillna(0) >= float(min_price)]
    df_summary = df_summary.head(max_rows).reset_index(drop=True)

    score_weights = load_setting_source("score_weights", None)
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
    sort_options = [c for c in ["score","last_price","volatility","symbol"] if c in df_summary2.columns] or ["symbol"]
    sort_by = st.selectbox("Ordenar por", options=sort_options, index=0)
    ascending = st.checkbox("Orden ascendente", value=False)
    if sort_by in df_summary2.columns:
        df_summary2 = df_summary2.sort_values(by=sort_by, ascending=ascending, na_position="last")
    st.dataframe(df_summary2.reset_index(drop=True).style.format({"last_price": lambda v: f"{v:,.6f}" if pd.notna(v) else "—",
                                                                   "score": lambda v: f"{v:.3f}" if pd.notna(v) else "—",
                                                                   "volatility": lambda v: f"{v:.6f}" if pd.notna(v) else "—"}), height=420)

    st.markdown("**Acciones por activo**")
    for rec in df_summary2.to_dict("records"):
        s = rec["symbol"]
        cols = st.columns([2,2,1,2,2])
        cols[0].markdown(f"**{s}**")
        cols[1].markdown(f"Precio: {rec.get('last_price') if rec.get('last_price') is not None else '—'}")
        if cols[2].button("Ver detalle", key=f"view_{s}"):
            st.session_state["selected_asset"] = s
            st.experimental_rerun()
        if cols[3].button("Backtest", key=f"bt_{s}"):
            def _bt_work(sym=s):
                # prefer orchestrator
                if orchestrator_mod:
                    for name in ("run_backtest_for","run_backtest","backtest"):
                        if hasattr(orchestrator_mod, name):
                            try:
                                return getattr(orchestrator_mod, name)(sym)
                            except Exception:
                                logger.exception("orchestrator %s failed", name)
                # fallback simple EMA backtest
                try:
                    df_hist = load_candles_source(sym, limit=1000)
                    if df_hist is None or df_hist.empty:
                        return {"error":"no_data"}
                    es = int(load_setting_source("ema_short", 9))
                    el = int(load_setting_source("ema_long", 50))
                    close = pd.to_numeric(df_hist["close"], errors="coerce")
                    ema_s = close.ewm(span=es, adjust=False).mean()
                    ema_l = close.ewm(span=el, adjust=False).mean()
                    pos = (ema_s > ema_l).astype(int)
                    ret = close.pct_change().fillna(0)
                    strat_ret = ret * pos.shift(1).fillna(0)
                    cum = (1 + strat_ret).cumprod() - 1
                    total_return = float(cum.iloc[-1]) if len(cum) > 0 else 0.0
                    peak = (1 + strat_ret).cumprod().cummax()
                    dd = ((1 + strat_ret).cumprod() / peak - 1).min()
                    winrate = float((strat_ret > 0).sum() / max(1, (strat_ret != 0).sum()))
                    metrics = {"return": total_return, "drawdown": float(dd), "winrate": winrate}
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

# ----------------------------
# Detalle tab
# ----------------------------
with tab_detail:
    st.header("Detalle del activo")
    selected = st.session_state.get("selected_asset", None)
    if not selected:
        st.info("Selecciona un activo desde la pestaña Watchlist para ver detalle.")
    else:
        st.subheader(selected)
        dfc = load_candles_source(selected, limit=500)
        if dfc is None or dfc.empty:
            st.warning("No hay datos históricos para este activo. Sube CSV a data/cache/ o habilita fetch/storage.")
        else:
            score_weights = load_setting_source("score_weights", None)
            params_ind = {"ema_short": int(load_setting_source("ema_short",9)), "ema_long": int(load_setting_source("ema_long",50)), "rsi_period": int(load_setting_source("rsi_period",14))}
            try:
                df_enriched = enrich_indicators_and_score(dfc, indicators=None, params=params_ind, score_weights=score_weights, lookback=30)
            except Exception:
                df_enriched = dfc.copy()
            try:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                xaxis = df_enriched["timestamp"] if "timestamp" in df_enriched.columns else df_enriched["ts"] if "ts" in df_enriched.columns else df_enriched.index
                fig.add_trace(go.Candlestick(x=xaxis, open=df_enriched["open"], high=df_enriched["high"], low=df_enriched["low"], close=df_enriched["close"], name="Candles"), row=1, col=1)
                for col in df_enriched.columns:
                    if col.startswith("ema_"):
                        fig.add_trace(go.Scatter(x=xaxis, y=df_enriched[col], name=col, line=dict(width=1.5)), row=1, col=1)
                rsi_cols = [c for c in df_enriched.columns if c.startswith("rsi")]
                if rsi_cols:
                    fig.add_trace(go.Scatter(x=xaxis, y=df_enriched[rsi_cols[0]], name=rsi_cols[0]), row=2, col=1)
                else:
                    if "volume" in df_enriched.columns:
                        fig.add_trace(go.Bar(x=xaxis, y=df_enriched["volume"], showlegend=False), row=2, col=1)
                fig.update_layout(height=700, margin=dict(l=8,r=8,t=30,b=8), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                logger.exception("plotting failed for %s", selected)
                st.write("Error plotting — revisa logs")

            last_rows = df_enriched.tail(10).copy()
            st.subheader("Últimas filas (indicadores)")
            display_cols = [c for c in last_rows.columns if c in ["timestamp","close","volume"] or c.startswith("ema_") or c.startswith("rsi") or c.startswith("macd") or c.startswith("atr") or c=="score"]
            st.dataframe(last_rows[display_cols].reset_index(drop=True).style.format({"close":"{:,.6f}"}), height=260)

            a1, a2, a3 = st.columns([1,1,1])
            if a1.button("Descargar CSV"):
                csv = df_enriched.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar CSV", data=csv, file_name=f"{selected}_candles.csv", mime="text/csv")
            if a2.button("Infer IA (background)"):
                def _do_infer(sym=selected):
                    if ai_inf_mod:
                        for fn in ("predict","infer","run_predict"):
                            if hasattr(ai_inf_mod, fn):
                                try:
                                    return getattr(ai_inf_mod, fn)(sym)
                                except Exception:
                                    logger.exception("ai_inference.%s failed", fn)
                    return {"error":"no_ai_infer"}
                key = run_background(_do_infer, task_key=f"infer_{selected}")
                st.info(f"Inferencia lanzada (task {key})")
            if a3.button("Backtest (background)"):
                def _do_bt(sym=selected):
                    if orchestrator_mod:
                        for name in ("run_backtest_for","run_backtest","backtest"):
                            if hasattr(orchestrator_mod, name):
                                try:
                                    return getattr(orchestrator_mod, name)(sym)
                                except Exception:
                                    logger.exception("orchestrator %s failed", name)
                    df_hist = load_candles_source(sym, limit=1000)
                    if df_hist is None or df_hist.empty:
                        return {"error":"no_data"}
                    es = int(load_setting_source("ema_short",9))
                    el = int(load_setting_source("ema_long",50))
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

# ----------------------------
# Backtests tab
# ----------------------------
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

# ----------------------------
# Settings tab
# ----------------------------
with tab_settings:
    st.header("Ajustes y estado")
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

# ----------------------------
# System tab
# ----------------------------
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

# ----------------------------
# Stop/Target (ATR) widget & Logs viewer
# ----------------------------
st.markdown("---")
st.markdown("### Stop / Target (ATR-based)")
col1, col2, col3 = st.columns(3)
asset_sel = st.session_state.get("selected_asset", None)
interval_sel = st.session_state.get("selected_interval", "1h")
if asset_sel:
    if st.button("Calcular stop/target (ATR)"):
        with st.spinner("Calculando..."):
            if compute_stop_target_for_asset and core_adapter:
                try:
                    res = compute_stop_target_for_asset(core_adapter, asset_sel, interval_sel, lookback=200)
                except Exception:
                    logger.exception("compute_stop_target_for_asset failed")
                    res = {"error": "calc_failed"}
            else:
                res = {"error": "no_calc_available"}
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

st.markdown("### Visualizador de Logs (últimas líneas)")
try:
    lines = tail_file(LOG_PATH, 300)
    st.code("\n".join(lines[-200:]) if lines else "No hay logs en fichero todavía")
except Exception:
    st.write("No se pudo leer fichero de logs")

st.markdown("### Logs en memoria (lo que produce este proceso)")
show_mem = list(INMEM_LOG)[-200:]
if show_mem:
    st.code("\n".join(show_mem))
else:
    st.write("No hay logs en memoria todavía para este proceso")

# End of file
