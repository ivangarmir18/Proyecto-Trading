# dashboard/app.py
"""
Dashboard robusto y autocontenido para Watchlist.
- Prioriza core.storage_postgres.PostgresStorage.
- Fallback a core.fetch.
- Fallback final a CSVs en data/cache y data/config.
- Backfill de arranque (no bloqueante) a menos que SKIP_BACKFILL=1.
- Auto-updater configurable (AUTO_UPDATE_ENABLED, AUTO_UPDATE_INTERVAL_MIN).
- Botones: backfill manual, correr backtest por activo, entrenar IA, inferencia IA.
- Persistencia de settings: PostgresStorage.save_setting / load_setting si existe, sino JSON fallback en data/config/settings.json.
- Guarda backtests en data/db/backtests/.
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- config paths ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CONF_DIR = os.path.join(DATA_DIR, "config")
DB_DIR = os.path.join(DATA_DIR, "db")
BACKTESTS_DIR = os.path.join(DB_DIR, "backtests")
SETTINGS_FILE = os.path.join(CONF_DIR, "settings.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(BACKTESTS_DIR, exist_ok=True)

# --- logging ---
logger = logging.getLogger("watchlist.dashboard")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# --- dynamic import helper ---
def _import_safe(module: str, attr: Optional[str] = None):
    try:
        m = __import__(module, fromlist=[attr] if attr else [])
        return getattr(m, attr) if attr else m
    except Exception:
        return None

# --- try PostgresStorage first ---
PostgresStorage = None
pg = None
try:
    mod_ps = _import_safe("core.storage_postgres")
    if mod_ps and hasattr(mod_ps, "PostgresStorage"):
        PostgresStorage = getattr(mod_ps, "PostgresStorage")
        try:
            pg = PostgresStorage()  # assume it reads DATABASE_URL from env or config
            logger.info("Usando core.storage_postgres.PostgresStorage")
        except Exception as e:
            logger.warning("No se pudo instanciar PostgresStorage: %s", e)
            pg = None
except Exception:
    pg = None

# --- try core.fetch and orchestrator/ai modules ---
fetch_mod = _import_safe("core.fetch")
orchestrator_mod = _import_safe("core.orchestrator")
ai_train_mod = _import_safe("core.ai_train")
ai_inf_mod = _import_safe("core.ai_inference")

# common function resolution (several possible names)
_list_assets_fn = None
_load_candles_fn = None
_run_full_backfill_fn = None
_run_backtest_for_fn = None
_train_ai_fn = None
_infer_ai_fn = None
_save_setting_fn = None
_load_setting_fn = None

# If pg available and exposes methods, prefer those
if pg:
    if hasattr(pg, "list_assets"):
        _list_assets_fn = getattr(pg, "list_assets")
    if hasattr(pg, "load_candles"):
        _load_candles_fn = getattr(pg, "load_candles")
    elif hasattr(pg, "get_candles"):
        _load_candles_fn = getattr(pg, "get_candles")
    if hasattr(pg, "run_full_backfill"):
        _run_full_backfill_fn = getattr(pg, "run_full_backfill")
    if hasattr(pg, "save_setting"):
        _save_setting_fn = getattr(pg, "save_setting")
    if hasattr(pg, "load_setting"):
        _load_setting_fn = getattr(pg, "load_setting")

# If not provided by pg, try fetch_mod
if not _list_assets_fn and fetch_mod:
    for name in ("list_watchlist_assets", "list_assets", "get_watchlist", "get_assets"):
        if hasattr(fetch_mod, name):
            _list_assets_fn = getattr(fetch_mod, name)
            break

if not _load_candles_fn and fetch_mod:
    for name in ("get_candles", "get_latest_candles", "fetch_candles", "get_historical"):
        if hasattr(fetch_mod, name):
            _load_candles_fn = getattr(fetch_mod, name)
            break

if not _run_full_backfill_fn and fetch_mod:
    for name in ("run_full_backfill", "backfill_all", "run_backfill"):
        if hasattr(fetch_mod, name):
            _run_full_backfill_fn = getattr(fetch_mod, name)
            break

# orchestrator backtest
if orchestrator_mod:
    for name in ("run_backtest_for", "run_backtest", "backtest"):
        if hasattr(orchestrator_mod, name):
            _run_backtest_for_fn = getattr(orchestrator_mod, name)
            break

if ai_train_mod:
    for name in ("train", "do_train", "run_train"):
        if hasattr(ai_train_mod, name):
            _train_ai_fn = getattr(ai_train_mod, name)
            break

if ai_inf_mod:
    for name in ("predict", "infer", "run_predict"):
        if hasattr(ai_inf_mod, name):
            _infer_ai_fn = getattr(ai_inf_mod, name)
            break

# --- fallback CSV loader for candles and watchlist ---
def fallback_list_assets() -> List[str]:
    candidates = [
        os.path.join(CONF_DIR, "watchlist.csv"),
        os.path.join(CONF_DIR, "watchlist.json"),
        os.path.join(CONF_DIR, "assets.csv"),
        os.path.join(CONF_DIR, "assets.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                if c.endswith(".csv"):
                    df = pd.read_csv(c)
                    if "symbol" in df.columns:
                        return df["symbol"].astype(str).tolist()
                    return df.iloc[:, 0].astype(str).tolist()
                else:
                    j = pd.read_json(c)
                    return j.iloc[:, 0].astype(str).tolist()
            except Exception:
                continue
    return ["BTCUSDT", "ETHUSDT", "AAPL", "TSLA"]

def fallback_load_candles(symbol: str, limit: int = 1000) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
            expected = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in expected:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df.sort_values("timestamp").reset_index(drop=True)
            if limit and len(df) > limit:
                return df.iloc[-limit:].reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

# --- settings persistence helpers ---
def _fallback_load_setting(key, default=None):
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf8") as f:
                j = json.load(f)
            return j.get(key, default)
    except Exception:
        pass
    return default

def _fallback_save_setting(key, value):
    data = {}
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf8") as f:
                data = json.load(f)
    except Exception:
        data = {}
    data[key] = value
    try:
        with open(SETTINGS_FILE, "w", encoding="utf8") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception:
        return False

def save_setting(key, value):
    try:
        if _save_setting_fn:
            return _save_setting_fn(key, value)
    except Exception:
        logger.exception("save_setting via primary failed")
    return _fallback_save_setting(key, value)

def load_setting(key, default=None):
    try:
        if _load_setting_fn:
            return _load_setting_fn(key, default)
    except Exception:
        logger.exception("load_setting via primary failed")
    return _fallback_load_setting(key, default)

# --- wrapper functions used by UI ---
def list_assets():
    try:
        if _list_assets_fn:
            out = _list_assets_fn()
            # if returns dataframe/list/dict coerce to list
            if isinstance(out, pd.DataFrame):
                if "symbol" in out.columns:
                    return out["symbol"].astype(str).tolist()
                return out.iloc[:, 0].astype(str).tolist()
            if isinstance(out, (list, tuple)):
                return list(out)
            if isinstance(out, dict):
                return list(out.keys())
    except Exception:
        logger.exception("list_assets primary failed")
    return fallback_list_assets()

def load_candles(symbol: str, limit: int = 1000) -> pd.DataFrame:
    try:
        if _load_candles_fn:
            df = _load_candles_fn(symbol, limit=limit)
            if isinstance(df, (list, tuple)):
                try:
                    df = pd.DataFrame(df)
                except Exception:
                    return fallback_load_candles(symbol, limit=limit)
            if isinstance(df, pd.DataFrame):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                if limit and len(df) > limit:
                    return df.sort_values("timestamp").iloc[-limit:].reset_index(drop=True)
                return df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df
    except Exception:
        logger.exception("load_candles primary failed for %s", symbol)
    return fallback_load_candles(symbol, limit=limit)

# --- auto backfill at startup (non-blocking) ---
def _run_start_backfill():
    if os.getenv("SKIP_BACKFILL", "0").strip() == "1":
        logger.info("SKIP_BACKFILL=1 -> skipping startup backfill")
        return {"skipped": True}
    if _run_full_backfill_fn:
        try:
            logger.info("Running startup backfill via core.fetch/core.storage_postgres")
            try:
                res = _run_full_backfill_fn()
            except TypeError:
                res = _run_full_backfill_fn(per_symbol_limit=int(os.getenv("BACKFILL_LIMIT", "1000")))
            logger.info("Startup backfill finished")
            return {"skipped": False, "result": res}
        except Exception as e:
            logger.exception("Startup backfill failed: %s", e)
            return {"skipped": False, "error": str(e)}
    logger.info("No run_full_backfill function available; skipping network backfill")
    return {"skipped": False, "result": "no_backfill_fn"}

_start_backfill_started = False
def maybe_start_backfill_thread():
    global _start_backfill_started
    if _start_backfill_started:
        return
    _start_backfill_started = True
    def _worker():
        try:
            res = _run_start_backfill()
            logger.info("Startup backfill result: %s", str(res))
        except Exception:
            logger.exception("Startup backfill worker crashed")
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

# start non-blocking
maybe_start_backfill_thread()

# --- background runner for UI tasks ---
def run_in_background(fn, args=(), kwargs=None, task_key=None, on_done=None):
    kwargs = kwargs or {}
    key = f"task_{task_key or str(time.time()).replace('.','_')}"
    st.session_state[key] = {"status": "running", "started_at": datetime.utcnow().isoformat(), "result": None, "error": None}
    def _worker():
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
            logger.exception("Background task error")
            st.session_state[key]["status"] = "error"
            st.session_state[key]["error"] = str(e)
            st.session_state[key]["finished_at"] = datetime.utcnow().isoformat()
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return key

# --- plotting helpers ---
def plot_candles_with_indicators(df: pd.DataFrame, ema_short: int = 9, ema_long: int = 50, show_volume: bool = True):
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    # Ensure numeric
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if f"ema_{ema_short}" not in df.columns:
        df[f"ema_{ema_short}"] = df["close"].ewm(span=ema_short, adjust=False).mean()
    if f"ema_{ema_long}" not in df.columns:
        df[f"ema_{ema_long}"] = df["close"].ewm(span=ema_long, adjust=False).mean()
    # RSI
    rsi_col = "rsi_14"
    if rsi_col not in df.columns:
        delta = df["close"].diff()
        up = delta.clip(lower=0).fillna(0)
        down = -1 * delta.clip(upper=0).fillna(0)
        ma_up = up.ewm(com=(14 - 1), adjust=False).mean()
        ma_down = down.ewm(com=(14 - 1), adjust=False).mean()
        rs = ma_up / (ma_down.replace(0, np.nan))
        df[rsi_col] = 100 - (100 / (1 + rs))
    rows = 2 if show_volume or rsi_col in df.columns else 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Candles"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[f"ema_{ema_short}"], name=f"EMA{ema_short}", line=dict(width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[f"ema_{ema_long}"], name=f"EMA{ema_long}", line=dict(width=1.0, dash="dash")), row=1, col=1)
    if show_volume and "volume" in df.columns:
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], showlegend=False), row=2, col=1)
    # add RSI to second row as line
    if rsi_col in df.columns and rows == 2:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[rsi_col], name="RSI(14)"), row=2, col=1)
    fig.update_layout(height=700, margin=dict(l=8, r=8, t=30, b=8), template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

# --- Streamlit UI ---
st.set_page_config(page_title="Watchlist", layout="wide", initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    st.title("Watchlist — Config")
    st.markdown("Ajustes globales y operaciones.")

    ia_default = load_setting("ia_enabled", False)
    ia_enabled = st.checkbox("Activar IA (persistente)", value=bool(ia_default))
    if st.button("Guardar IA toggle"):
        save_setting("ia_enabled", bool(ia_enabled))
        st.success("Guardado")

    st.markdown("---")
    st.subheader("Score weights")
    sw = load_setting("score_weights", {"momentum": 0.4, "trend": 0.4, "volatility": 0.2})
    m = st.slider("Momentum", 0.0, 1.0, float(sw.get("momentum", 0.4)), step=0.01)
    t_ = st.slider("Trend", 0.0, 1.0, float(sw.get("trend", 0.4)), step=0.01)
    v = st.slider("Volatility", 0.0, 1.0, float(sw.get("volatility", 0.2)), step=0.01)
    if st.button("Guardar pesos score"):
        save_setting("score_weights", {"momentum": m, "trend": t_, "volatility": v})
        st.success("Pesos guardados")

    st.markdown("---")
    st.subheader("Visualización")
    show_volume = st.checkbox("Mostrar volumen", value=True)
    ema_short = st.number_input("EMA corta", min_value=3, max_value=200, value=9, step=1)
    ema_long = st.number_input("EMA larga", min_value=5, max_value=400, value=50, step=1)

    st.markdown("---")
    st.subheader("Operaciones")
    if st.button("Forzar backfill ahora"):
        def _do_backfill():
            if _run_full_backfill_fn:
                try:
                    return _run_full_backfill_fn()
                except Exception as e:
                    return {"error": str(e)}
            return {"error": "no_backfill_fn"}
        key = run_in_background(_do_backfill, task_key="manual_backfill")
        st.info(f"Backfill en background (task {key})")

    if st.button("Entrenar IA (background)"):
        def _do_train():
            if _train_ai_fn:
                try:
                    return _train_ai_fn({})
                except Exception as e:
                    return {"error": str(e)}
            return {"error": "no_train_fn"}
        key = run_in_background(_do_train, task_key="train_ai")
        st.info(f"Entrenamiento IA en background (task {key})")

# Main layout: tabs
tabs = st.tabs(["Watchlist", "Settings", "System"])
tab_watch, tab_settings, tab_system = tabs

# Watchlist tab
with tab_watch:
    st.header("Watchlist")
    st.markdown("Activos monitorizados. Si no hay fetch, coloca CSVs en data/cache/{SYMBOL}.csv")

    # assets
    try:
        assets = list_assets()
    except Exception:
        logger.exception("list_assets failed")
        assets = fallback_list_assets()

    # filters
    f1, f2, f3, f4 = st.columns([3, 1, 1, 1])
    q = f1.text_input("Buscar símbolo", value="", placeholder="BTC, AAPL, etc.")
    min_price = f2.number_input("Precio mínimo", value=0.0, step=0.01)
    max_rows = f3.selectbox("Mostrar filas", [10, 25, 50, 100], index=1)
    if f4.button("Refrescar"):
        st.experimental_rerun()

    rows = []
    for s in assets:
        try:
            df2 = load_candles(s, limit=2)
            df2 = df2 if isinstance(df2, pd.DataFrame) else fallback_load_candles(s, limit=2)
            df2 = df2.dropna(subset=["close"]) if "close" in df2.columns else df2
            last_price = float(df2.iloc[-1]["close"]) if (isinstance(df2, pd.DataFrame) and not df2.empty) else None
        except Exception:
            last_price = None
        rows.append({"symbol": s, "last_price": last_price})

    df_summary = pd.DataFrame(rows)
    if q:
        df_summary = df_summary[df_summary["symbol"].str.contains(q, case=False, na=False)]
    if min_price and min_price > 0:
        df_summary = df_summary[df_summary["last_price"].fillna(0) >= float(min_price)]
    df_summary = df_summary.head(max_rows).reset_index(drop=True)

    st.dataframe(df_summary.style.format({"last_price": lambda v: f"{v:,.6f}" if v is not None else "—"}), height=320)
    st.markdown("**Acciones**")
    for r in df_summary.to_dict("records"):
        s = r["symbol"]
        cols = st.columns([2, 2, 1, 2])
        cols[0].markdown(f"**{s}**")
        cols[1].markdown(f"Precio: {format(r['last_price'], '.6f') if r['last_price'] is not None else '—'}")
        if cols[2].button("Ver detalle", key=f"view_{s}"):
            st.session_state["selected_asset"] = s
            st.experimental_rerun()
        if cols[3].button("Backtest", key=f"bt_{s}"):
            def _do_bt(sym=s):
                if _run_backtest_for_fn:
                    try:
                        return _run_backtest_for_fn(sym)
                    except Exception as e:
                        return {"error": str(e)}
                return {"error": "no_backtest_engine"}
            key = run_in_background(_do_bt, task_key=f"backtest_{s}", on_done=lambda res, sym=s: save_backtest_result_local(sym, res))
            st.info(f"Backtest para {s} lanzado (task {key})")

    st.markdown("---")
    selected = st.session_state.get("selected_asset", None)
    if not selected:
        st.info("Selecciona un activo para ver el detalle.")
    else:
        st.subheader(f"Detalle — {selected}")
        try:
            dfc = load_candles(selected, limit=1500)
            if not isinstance(dfc, pd.DataFrame):
                dfc = fallback_load_candles(selected, limit=1500)
        except Exception:
            dfc = fallback_load_candles(selected, limit=1500)
        dfc = dfc if isinstance(dfc, pd.DataFrame) else fallback_load_candles(selected, limit=1500)
        if dfc.empty:
            st.warning("No hay datos históricos. Añade CSV en data/cache/{SYMBOL}.csv o habilita fetch/storage_postgres.")
        else:
            fig = plot_candles_with_indicators(dfc, ema_short=int(ema_short), ema_long=int(ema_long), show_volume=bool(show_volume))
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns([1,1,1])
            if c1.button("Descargar CSV"):
                csv = dfc.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar CSV", data=csv, file_name=f"{selected}_candles.csv", mime="text/csv")
            if c2.button("Infer IA (background)"):
                def _do_infer(sym=selected):
                    if _infer_ai_fn:
                        try:
                            return _infer_ai_fn(sym)
                        except Exception as e:
                            return {"error": str(e)}
                    return {"error": "no_ai_infer"}
                key = run_in_background(_do_infer, task_key=f"infer_{selected}")
                st.info(f"Inferencia IA lanzada (task {key})")
            if c3.button("Backtest (background)"):
                def _do_bt2(sym=selected):
                    if _run_backtest_for_fn:
                        try:
                            return _run_backtest_for_fn(sym)
                        except Exception as e:
                            return {"error": str(e)}
                    return {"error": "no_backtest_engine"}
                key = run_in_background(_do_bt2, task_key=f"backtest_{selected}", on_done=lambda res, s=selected: save_backtest_result_local(s, res))
                st.info(f"Backtest lanzado (task {key})")

# Settings tab
with tab_settings:
    st.header("Settings")
    st.markdown("Ajustes persistentes (IA, pesos, etc.)")
    cur_w = load_setting("score_weights", {"momentum": 0.4, "trend": 0.4, "volatility": 0.2})
    st.json(cur_w)
    st.markdown("Tareas en background (session)")
    keys = [k for k in st.session_state.keys() if k.startswith("task_")]
    if not keys:
        st.info("No hay tareas.")
    else:
        for k in keys:
            st.markdown(f"**{k}**")
            st.json(st.session_state[k])

    st.subheader("Backtests guardados")
    files = sorted([f for f in os.listdir(BACKTESTS_DIR) if f.endswith(".json")], reverse=True)
    if not files:
        st.info("No hay backtests guardados.")
    else:
        sel = st.selectbox("Selecciona backtest", options=files)
        if sel:
            path = os.path.join(BACKTESTS_DIR, sel)
            with open(path, "r", encoding="utf8") as f:
                data = json.load(f)
            st.json(data)
            if st.button("Descargar JSON"):
                with open(path, "rb") as fh:
                    st.download_button("Download", fh.read(), file_name=sel, mime="application/json")

# System tab
with tab_system:
    st.header("System")
    mod_status = {
        "postgres_storage": bool(pg),
        "core.fetch": bool(fetch_mod),
        "orchestrator": bool(orchestrator_mod),
        "ai_train": bool(ai_train_mod),
        "ai_infer": bool(ai_inf_mod),
    }
    st.json(mod_status)
    st.subheader("Cache files")
    if os.path.exists(CACHE_DIR):
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
        st.write(f"{len(files)} archivos en data/cache/")
        for f in files[:200]:
            st.write(f"- {f}")
    else:
        st.info("No hay carpeta cache")

    st.markdown("---")
    st.subheader("Tasks (session)")
    tks = sorted([k for k in st.session_state.keys() if k.startswith("task_")])
    if not tks:
        st.info("No hay tasks en esta sesión")
    else:
        for tk in tks:
            st.markdown(f"**{tk}** — {st.session_state[tk].get('status')}")
            st.json(st.session_state[tk])

# end of file
