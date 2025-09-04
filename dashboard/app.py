# dashboard/app.py
"""
Dashboard profesional completo para Watchlist.
Características:
 - Watchlist interactiva con filtros, ordenar y botones por fila.
 - Panel de detalle (velas, EMAs, RSI, volumen, export).
 - Settings persistentes (score weights, toggle IA).
 - Background auto-updater que actualiza velas en cache automáticamente.
 - Forzar backfill, ejecutar backtests en background y guardar resultados.
 - System status con tasks session y cache listing.
 - Fail-safe total: si faltan módulos core, usa fallbacks y no rompe.
"""

import os
import sys
import time
import json
import threading
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# asegurar ruta repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.adapter import adapter
from core.settings import save_setting, load_setting, health_status

LOG = logging.getLogger("dashboard")
LOG.setLevel(logging.INFO)

DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DB_DIR = os.path.join(DATA_DIR, "db")
BACKTESTS_DIR = os.path.join(DB_DIR, "backtests")
os.makedirs(BACKTESTS_DIR, exist_ok=True)

# -----------------------
# Utilidades internas
# -----------------------
def format_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):,.6f}"
    except Exception:
        return str(x)

def ensure_df_ts(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def save_backtest_result(symbol: str, result: Dict[str, Any]) -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{symbol.replace('/','_')}_backtest_{now}.json"
    path = os.path.join(BACKTESTS_DIR, fname)
    try:
        with open(path, "w", encoding="utf8") as f:
            json.dump(result, f, default=str, indent=2)
    except Exception:
        LOG.exception("Error saving backtest result")
    return path

# -----------------------
# Background task runner
# -----------------------
def _run_background(task_fn, args=(), kwargs=None, task_key=None, on_done=None):
    kwargs = kwargs or {}
    key = f"task_{task_key or str(time.time())}"
    st.session_state[key] = {"status": "running", "started_at": datetime.utcnow().isoformat(), "result": None, "error": None}

    def _worker():
        try:
            res = task_fn(*args, **kwargs)
            st.session_state[key]["status"] = "done"
            st.session_state[key]["result"] = res
            st.session_state[key]["finished_at"] = datetime.utcnow().isoformat()
            if on_done:
                try:
                    on_done(res)
                except Exception:
                    LOG.exception("on_done callback failed")
        except Exception as e:
            LOG.exception("Background task error")
            st.session_state[key]["status"] = "error"
            st.session_state[key]["error"] = str(e)
            st.session_state[key]["finished_at"] = datetime.utcnow().isoformat()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return key

# -----------------------
# Auto-updater (periodic)
# -----------------------
AUTO_UPDATE_ENABLED = os.getenv("AUTO_UPDATE_ENABLED", "1").strip() != "0"
AUTO_UPDATE_INTERVAL_MIN = int(os.getenv("AUTO_UPDATE_INTERVAL_MIN", "5"))
AUTO_UPDATE_INITIAL_DELAY_SEC = int(os.getenv("AUTO_UPDATE_INITIAL_DELAY_SEC", "5"))
AUTO_UPDATE_BATCH = int(os.getenv("AUTO_UPDATE_BATCH", "10"))  # cuantos símbolos por iteración

def _auto_updater_loop():
    """
    Funcion que corre en background: itera sobre la lista de activos y llama adapter.update_symbol
    para mantener cache actualizada. No bloquea la UI.
    """
    LOG.info("Auto-updater thread started (enabled=%s interval_min=%s)", AUTO_UPDATE_ENABLED, AUTO_UPDATE_INTERVAL_MIN)
    # initial delay to give process time to start
    time.sleep(AUTO_UPDATE_INITIAL_DELAY_SEC)
    while True:
        try:
            assets = adapter.list_assets()
            if not assets:
                LOG.info("Auto-updater: no assets listed")
                time.sleep(AUTO_UPDATE_INTERVAL_MIN * 60)
                continue
            # iterar por lotes para no saturar
            for i in range(0, len(assets), AUTO_UPDATE_BATCH):
                batch = assets[i:i+AUTO_UPDATE_BATCH]
                for s in batch:
                    try:
                        adapter.update_symbol(s, limit=500)
                    except Exception:
                        LOG.exception("Auto-updater failed for %s", s)
                # pequeña pausa entre batches
                time.sleep(1)
            # esperar siguiente ciclo
            time.sleep(AUTO_UPDATE_INTERVAL_MIN * 60)
        except Exception:
            LOG.exception("Auto-updater top loop error")
            time.sleep(AUTO_UPDATE_INTERVAL_MIN * 60)

# lanzar auto-updater en background (no se reinicia en reruns)
if AUTO_UPDATE_ENABLED and ("_auto_updater_started" not in st.session_state):
    st.session_state["_auto_updater_started"] = True
    try:
        t = threading.Thread(target=_auto_updater_loop, daemon=True)
        t.start()
        LOG.info("Auto-updater launched")
    except Exception:
        LOG.exception("Failed to start auto-updater")

# -----------------------
# Score weights defaults
# -----------------------
DEFAULT_SCORE_WEIGHTS = {"momentum": 0.4, "trend": 0.4, "volatility": 0.2}
def load_score_weights():
    return load_setting("score_weights", DEFAULT_SCORE_WEIGHTS)
def save_score_weights(weights):
    return save_setting("score_weights", weights)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Watchlist — Dashboard", layout="wide", initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    st.title("Watchlist - Config")
    st.markdown("Ajustes globales y operaciones.")

    # IA toggle persistent
    ia_default = load_setting("ia_enabled", False)
    ia_enabled = st.checkbox("Activar IA (persistente)", value=bool(ia_default))
    if st.session_state.get("_ia_saved_flag") is None:
        st.session_state["_ia_saved_flag"] = True
        save_setting = save_setting if 'save_setting' in globals() else save_setting  # fallback guard
        save_setting("ia_enabled", bool(ia_enabled))
    else:
        # detect changes and save
        if st.session_state.get("ia_enabled") != ia_enabled:
            save_setting("ia_enabled", bool(ia_enabled))
    st.session_state["ia_enabled"] = ia_enabled

    st.markdown("---")
    st.subheader("Score weights")
    cur_weights = load_score_weights()
    w_momentum = st.slider("Momentum", 0.0, 1.0, float(cur_weights.get("momentum", 0.4)), step=0.01, key="w_momentum")
    w_trend = st.slider("Trend", 0.0, 1.0, float(cur_weights.get("trend", 0.4)), step=0.01, key="w_trend")
    w_vol = st.slider("Volatility", 0.0, 1.0, float(cur_weights.get("volatility", 0.2)), step=0.01, key="w_vol")
    if st.button("Guardar pesos score"):
        new_w = {"momentum": w_momentum, "trend": w_trend, "volatility": w_vol}
        save_score_weights(new_w)
        st.success("Pesos guardados")

    st.markdown("---")
    st.subheader("Visualización")
    show_volume = st.checkbox("Mostrar volumen", value=True)
    ema_short = st.number_input("EMA corta", min_value=3, max_value=200, value=9, step=1)
    ema_long = st.number_input("EMA larga", min_value=5, max_value=400, value=50, step=1)

    st.markdown("---")
    st.subheader("Operaciones rápidas")
    if st.button("Forzar backfill (todos)"):
        key = _run_background(adapter.run_full_backfill, args=(), task_key="backfill_all")
        st.info(f"Backfill lanzado (task {key})")
    if st.button("Entrenar IA global"):
        key = _run_background(adapter.train_ai, args=({},), task_key="train_ai")
        st.info(f"Entrenamiento IA lanzado (task {key})")

# Pestañas principales
tabs = st.tabs(["Watchlist", "Settings", "System"])
tab_watch, tab_settings, tab_system = tabs

# -------- WATCHLIST TAB --------
with tab_watch:
    st.header("Watchlist")
    st.markdown("Listado de activos monitorizados. Filtra, ordena y abre detalle.")

    try:
        assets = adapter.list_assets()
    except Exception:
        LOG.exception("adapter.list_assets failed")
        assets = ["BTCUSDT", "ETHUSDT", "AAPL"]

    # filtros rápidos
    f1, f2, f3, f4 = st.columns([3, 1, 1, 1])
    q = f1.text_input("Buscar símbolo", value="", placeholder="p.ej. BTC")
    min_price = f2.number_input("Precio min", value=0.0, step=0.01)
    max_rows = f3.selectbox("Mostrar filas", options=[10, 25, 50, 100], index=1)
    if f4.button("Refrescar datos"):
        st.experimental_rerun()

    # construir tabla resumen
    summary_rows = []
    for s in assets:
        try:
            df = adapter.load_candles(s, limit=3)
            df = ensure_df_ts(df)
            last_price = float(df.iloc[-1]["close"]) if (isinstance(df, pd.DataFrame) and not df.empty) else None
        except Exception:
            last_price = None
        summary_rows.append({"symbol": s, "last_price": last_price})

    df_summary = pd.DataFrame(summary_rows)
    if q:
        df_summary = df_summary[df_summary["symbol"].str.contains(q, case=False, na=False)]
    if min_price and min_price > 0:
        df_summary = df_summary[df_summary["last_price"].fillna(0) >= float(min_price)]
    df_summary = df_summary.head(max_rows).reset_index(drop=True)

    st.dataframe(df_summary.style.format({"last_price": lambda v: format_float(v)}), height=320)
    st.markdown("**Acciones por activo**")
    for row in df_summary.to_dict("records"):
        s = row["symbol"]
        cols = st.columns([2, 2, 1, 2])
        cols[0].markdown(f"**{s}**")
        cols[1].markdown(f"Precio: {format_float(row['last_price'])}")
        if cols[2].button("Ver detalle", key=f"view_{s}"):
            st.session_state["selected_asset"] = s
            st.experimental_rerun()
        if cols[3].button("Backtest", key=f"bt_{s}"):
            def _on_done_save(res, symbol=s):
                try:
                    path = save_backtest_result(symbol, res)
                    LOG.info("Backtest saved to %s", path)
                except Exception:
                    LOG.exception("Saving backtest failed")
            key = _run_background(adapter.run_backtest_for, args=(s,), task_key=f"backtest_{s}", on_done=_on_done_save)
            st.info(f"Backtest para {s} lanzado (task {key})")

    st.markdown("---")
    selected = st.session_state.get("selected_asset", None)
    if not selected:
        st.info("Selecciona un activo desde la lista para ver detalles.")
    else:
        st.subheader(f"Detalle — {selected}")
        dfc = adapter.load_candles(selected, limit=1500)
        dfc = ensure_df_ts(dfc)
        if dfc.empty:
            st.warning("No hay datos históricos. Ejecuta backfill o coloca CSV en data/cache/{SYMBOL}.csv")
        else:
            params = {"ema_short": ema_short, "ema_long": ema_long, "rsi_period": 14}
            dfc2 = adapter.apply_indicators(dfc, indicators=["ema", "rsi"], params=params)
            # prepare figure
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=dfc2["timestamp"], open=dfc2["open"], high=dfc2["high"], low=dfc2["low"], close=dfc2["close"], name="Candles"), row=1, col=1)
            ema_short_col = f"ema_{ema_short}"
            ema_long_col = f"ema_{ema_long}"
            if ema_short_col in dfc2.columns:
                fig.add_trace(go.Scatter(x=dfc2["timestamp"], y=dfc2[ema_short_col], name=f"EMA{ema_short}", line=dict(width=1.4)), row=1, col=1)
            if ema_long_col in dfc2.columns:
                fig.add_trace(go.Scatter(x=dfc2["timestamp"], y=dfc2[ema_long_col], name=f"EMA{ema_long}", line=dict(width=1.0, dash="dash")), row=1, col=1)
            if show_volume and "volume" in dfc2.columns:
                fig.add_trace(go.Bar(x=dfc2["timestamp"], y=dfc2["volume"], showlegend=False), row=2, col=1)
            if f"rsi_{14}" in dfc2.columns:
                # add RSI as line in second subplot as overlay
                fig.add_trace(go.Scatter(x=dfc2["timestamp"], y=dfc2[f"rsi_{14}"], name="RSI(14)"), row=2, col=1)
            fig.update_layout(height=700, margin=dict(l=8, r=8, t=30, b=8), template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            # actions
            a1, a2, a3 = st.columns([1,1,1])
            if a1.button("Export CSV"):
                csv = dfc2.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar CSV", data=csv, file_name=f"{selected}_candles.csv", mime="text/csv")
            if a2.button("Abrir detalle completo"):
                st.session_state["open_detail_full"] = selected
                st.experimental_rerun()
            if a3.button("Infer IA"):
                key = _run_background(adapter.infer_ai, args=(selected,), task_key=f"infer_{selected}")
                st.info(f"Inferencia IA lanzada (task {key})")

# -------- SETTINGS TAB --------
with tab_settings:
    st.header("Settings & Administración")
    st.markdown("Ajustes persistentes y herramientas administrativas.")

    st.subheader("Pesos del score (actuales)")
    cur_weights = load_score_weights()
    st.json(cur_weights)
    st.markdown("Edítalos en la barra lateral y pulsa 'Guardar pesos score'.")

    st.subheader("Tareas Background (session state)")
    task_keys = sorted([k for k in st.session_state.keys() if str(k).startswith("task_")])
    if not task_keys:
        st.info("No hay tasks en esta sesión.")
    else:
        for tk in task_keys:
            st.markdown(f"**{tk}** — status: {st.session_state[tk].get('status')}")
            st.json(st.session_state[tk])

    st.subheader("Backtests guardados")
    files = sorted([f for f in os.listdir(BACKTESTS_DIR) if f.endswith(".json")], reverse=True)
    if not files:
        st.info("No hay backtests guardados todavía.")
    else:
        sel = st.selectbox("Selecciona backtest", options=files)
        if sel:
            path = os.path.join(BACKTESTS_DIR, sel)
            with open(path, "r", encoding="utf8") as f:
                data = json.load(f)
            st.json(data)
            if st.button("Descargar JSON"):
                with open(path, "rb") as f:
                    st.download_button("Download", f.read(), file_name=sel)

# -------- SYSTEM TAB --------
with tab_system:
    st.header("System Status")
    st.markdown("Estado del sistema, cache, y tareas.")

    try:
        stat = health_status() or adapter.health_status()
    except Exception:
        LOG.exception("health_status error")
        stat = adapter.health_status()
    st.subheader("Resumen")
    st.json(stat)

    st.subheader("Cache files (data/cache)")
    if os.path.exists(CACHE_DIR):
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
        st.write(f"{len(cache_files)} archivos en cache")
        for f in cache_files[:200]:
            st.write(f"- {f}")
    else:
        st.info("No existe carpeta data/cache")

    st.markdown("---")
    st.subheader("Tasks (session)")
    task_keys = sorted([k for k in st.session_state.keys() if str(k).startswith("task_")])
    if not task_keys:
        st.info("No hay tasks en esta sesión.")
    else:
        for tk in task_keys:
            t = st.session_state[tk]
            st.markdown(f"**{tk}** — status: {t.get('status')}")
            st.json(t)

# EOF
