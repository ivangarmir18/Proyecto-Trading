# dashboard/app.py
"""
Streamlit dashboard for Proyecto Trading
- Requisitos: streamlit, plotly, pandas
- Run: streamlit run dashboard/app.py
"""
import os
import json
import time
import logging
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Import your core modules (asegúrate de que el path PYTHONPATH incluye repo root)
from core.storage_postgres import make_storage_from_env, PostgresStorage
from core.fetch import Fetcher
from core.score import compute_and_save_scores_for_asset
# ai_train optional import later when needed

logger = logging.getLogger("dashboard")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.getenv("DASHBOARD_LOG_LEVEL", "INFO"))

# Config
PROJECT_CONFIG_PATH = os.getenv("PROJECT_CONFIG_PATH", "config.json")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")  # set in env for basic protection

st.set_page_config(page_title="Proyecto Trading — Dashboard", layout="wide")

# ---------- Helpers ----------
@st.cache_resource
def get_storage():
    return make_storage_from_env()

@st.cache_resource
def get_fetcher():
    # Fetcher will read ENV keys if not provided
    return Fetcher(binance_api_key=os.getenv("ENV_BINANCE_KEY"), binance_secret=os.getenv("ENV_BINANCE_SECRET"), rate_limit_per_min=int(os.getenv("ENV_RATE_LIMIT","1200")))

def load_config():
    try:
        with open(PROJECT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"No pude leer config.json ({PROJECT_CONFIG_PATH}): {e}")
        return {}

def save_config(cfg: dict):
    with open(PROJECT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def add_symbol_to_config(symbol: str, kind: str):
    cfg = load_config()
    assets = cfg.setdefault("assets", {})
    key = "cripto" if kind == "crypto" else "acciones"
    lst = assets.setdefault(key, [])
    if symbol in lst:
        return False
    lst.append(symbol)
    save_config(cfg)
    return True

def query_latest_scores(storage: PostgresStorage, limit: Optional[int] = 1000) -> pd.DataFrame:
    """
    Select latest score per asset+interval using DISTINCT ON.
    """
    q = """
    SELECT DISTINCT ON (asset, interval) asset, interval, score, range_min, range_max, stop, target, created_at, ts
    FROM scores
    ORDER BY asset, interval, created_at DESC
    LIMIT %s
    """
    with storage.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (limit,))
            rows = cur.fetchall()
            cols = ["asset", "interval", "score", "range_min", "range_max", "stop", "target", "created_at", "ts"]
            df = pd.DataFrame(rows, columns=cols)
            if not df.empty:
                df["created_at"] = pd.to_datetime(df["created_at"], unit="ms", utc=True)
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def query_latest_candle(storage: PostgresStorage, asset: str, interval: str) -> Optional[pd.Series]:
    df = storage.get_ohlcv(asset, interval, limit=1)
    if df is None or df.empty:
        return None
    return df.iloc[-1]

def fetch_and_score(storage: PostgresStorage, fetcher: Fetcher, asset: str, interval: str):
    """Quick fetch latest and recompute score for asset/interval"""
    save_cb = storage.make_save_callback()
    df = fetcher.fetch_ohlcv(asset, interval=interval, since=None, limit=500, save_callback=save_cb, meta={"ui_refresh": True})
    # compute and save score (last lookback maybe 500)
    compute_and_save_scores_for_asset(storage, asset, interval, lookback_bars=500)
    return df

# ---------- Authentication ----------
def check_password():
    if not DASHBOARD_PASSWORD:
        return True  # open but warn
    pwd = st.session_state.get("pwd", "")
    if pwd == DASHBOARD_PASSWORD:
        st.session_state["authed"] = True
        return True
    return False

# ---------- UI ----------
st.title("Proyecto Trading — Dashboard")

if DASHBOARD_PASSWORD:
    if "pwd" not in st.session_state:
        st.session_state["pwd"] = ""
    if "authed" not in st.session_state:
        st.session_state["authed"] = False

    with st.sidebar:
        st.header("Acceso")
        if not st.session_state.get("authed", False):
            st.text_input("Password", type="password", key="pwd")
            if st.button("Entrar"):
                if check_password():
                    st.rerun()()
                else:
                    st.error("Password incorrecto")
        else:
            st.success("Autenticado")

else:
    st.sidebar.info("Dashboard sin password (setea DASHBOARD_PASSWORD en entorno para protegerlo)")

storage = get_storage()
fetcher = get_fetcher()

st.sidebar.header("Acciones rápidas")
cfg = load_config()
assets_cfg = cfg.get("assets", {})
total_assets = len((assets_cfg.get("cripto") or []) + (assets_cfg.get("acciones") or []))
st.sidebar.markdown(f"**Activos configurados:** {total_assets}")

# Add symbol form
with st.sidebar.expander("Añadir símbolo"):
    kind = st.selectbox("Tipo", ("crypto", "action"))
    symbol_input = st.text_input("Símbolo (ej. BTCUSDT o AAPL)", key="add_symbol")
    add_backfill = st.checkbox("Hacer backfill al añadir (rango por defecto)", value=False)
    if st.button("Añadir símbolo"):
        if not symbol_input:
            st.sidebar.error("Introduce símbolo válido")
        else:
            added = add_symbol_to_config(symbol_input.strip().upper(), kind)
            if added:
                st.sidebar.success(f"{symbol_input.upper()} añadido a config ({'crypto' if kind=='crypto' else 'acciones'})")
                if add_backfill:
                    st.sidebar.info("Iniciando backfill... (esto puede tardar)")
                    # trigger backfill via fetcher.backfill_range; conservative default: last 30 days
                    try:
                        end_ms = int(time.time() * 1000)
                        start_ms = end_ms - 30 * 24 * 3600 * 1000
                        df_back = fetcher.backfill_range(symbol_input.strip().upper(), "5m" if kind == "crypto" else "1h", start_ms, end_ms, save_callback=storage.make_save_callback(), progress=False)
                        st.sidebar.success(f"Backfill completado: {len(df_back)} velas")
                    except Exception as e:
                        st.sidebar.error(f"Backfill fallo: {e}")
            else:
                st.sidebar.warning("Símbolo ya presente en config.json")

# Top legend / quick metrics
st.markdown("---")
st.header("Resumen")
col1, col2, col3, col4 = st.columns([2,2,2,4])
with col1:
    st.metric("Activos totales", total_assets)
with col2:
    df_scores = query_latest_scores(storage, limit=2000)
    avg_score = float(df_scores["score"].mean()) if not df_scores.empty else 0.0
    st.metric("Score medio", f"{avg_score:.3f}")
with col3:
    last_update = df_scores["created_at"].max() if not df_scores.empty else None
    st.metric("Última actualización", last_update.strftime("%Y-%m-%d %H:%M:%S") if last_update is not None else "n/a")
with col4:
    st.write("**Leyenda / Orden**: Puedes filtrar y ordenar la tabla abajo. Selecciona filas para ver vela y ejecutar acciones (Refresh, Backfill, Train).")

st.markdown("---")

# Filters and controls
st.sidebar.header("Filtros tabla")
asset_type = st.sidebar.selectbox("Tipo de activo", ("All", "Crypto", "Actions"))
search = st.sidebar.text_input("Buscar (asset)", "")
score_min, score_max = st.sidebar.slider("Rango score", 0.0, 1.0, (0.0, 1.0), step=0.01)
date_from = st.sidebar.date_input("Desde", value=None)
date_to = st.sidebar.date_input("Hasta", value=None)
sort_by = st.sidebar.selectbox("Ordenar por", ("score", "asset", "created_at", "ts"))
sort_desc = st.sidebar.checkbox("Descendente", value=True)

# Refresh / batch actions
st.sidebar.header("Acciones")
if st.sidebar.button("Refresh tabla (releer scores)"):
    st.rerun()()

# Main table
st.subheader("Listado de activos y scores")
df = df_scores.copy() if not df_scores.empty else pd.DataFrame(columns=["asset","interval","score","range_min","range_max","stop","target","created_at","ts"])

# Filter by asset type using config
if asset_type != "All":
    cfg = load_config()
    if asset_type == "Crypto":
        allowed = set([s.upper() for s in (cfg.get("assets",{}).get("cripto") or [])])
    else:
        allowed = set([s.upper() for s in (cfg.get("assets",{}).get("acciones") or [])])
    df = df[df["asset"].str.upper().isin(allowed)]

if search:
    df = df[df["asset"].str.contains(search.strip(), case=False)]

df = df[(df["score"] >= float(score_min)) & (df["score"] <= float(score_max))]

# date filters if set
if date_from:
    df = df[df["created_at"] >= pd.to_datetime(pd.to_datetime(date_from).tz_localize("UTC"))]
if date_to:
    df = df[df["created_at"] <= pd.to_datetime(pd.to_datetime(date_to).tz_localize("UTC") + pd.Timedelta(days=1))]

if not df.empty:
    df_display = df.sort_values(by=sort_by, ascending=not sort_desc)
else:
    df_display = df

st.write(f"Mostrando {len(df_display)} filas")
# selectable table
selected = st.experimental_data_editor(df_display, num_rows="dynamic", use_container_width=True)

# If a single row is selected (or table clicked), show details
if selected is not None and len(selected) > 0:
    # We'll take the first selected row for detailed view
    sel = selected.iloc[0]
    sel_asset = sel["asset"]
    sel_interval = sel["interval"]
    st.markdown("---")
    st.subheader(f"Detalle: {sel_asset} — {sel_interval}")
    colA, colB, colC = st.columns([3,1,1])
    with colA:
        st.write(f"Score: **{sel['score']:.3f}**")
        st.write(f"Range: {sel['range_min']} — {sel['range_max']}")
        st.write(f"Stop: {sel['stop']} / Target: {sel['target']}")
        st.write(f"Última vela ts: {sel['ts']} — last update: {sel['created_at']}")
    with colB:
        if st.button("Refresh símbolo"):
            with st.spinner("Actualizando..."):
                try:
                    fetch_and_score(storage, fetcher, sel_asset, sel_interval)
                    st.success("Símbolo actualizado y score recalculado.")
                    st.rerun()()
                except Exception as e:
                    st.error(f"Error update: {e}")
    with colC:
        if st.button("Backfill símbolo (30d)"):
            with st.spinner("Haciendo backfill 30d..."):
                try:
                    end_ms = int(time.time() * 1000)
                    start_ms = end_ms - 30 * 24 * 3600 * 1000
                    df_back = fetcher.backfill_range(sel_asset, sel_interval, start_ms, end_ms, save_callback=storage.make_save_callback(), progress=False)
                    st.success(f"Backfill completado: {len(df_back)} velas")
                    st.rerun()()
                except Exception as e:
                    st.error(f"Backfill error: {e}")

    # Price chart
    df_candles = storage.get_ohlcv(sel_asset, sel_interval, limit=400)
    if df_candles is not None and not df_candles.empty:
        fig = go.Figure(data=[go.Candlestick(x=df_candles["ts"],
                                             open=df_candles["open"], high=df_candles["high"],
                                             low=df_candles["low"], close=df_candles["close"], name=sel_asset)])
        # overlay ema if present in indicators table by trying to get indicators via storage (join omitted for speed)
        try:
            # attempt to read indicators from indicators table via SQL
            with storage.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ts, ema9, ema40 FROM indicators WHERE asset=%s AND interval=%s ORDER BY ts ASC", (sel_asset, sel_interval))
                    rows = cur.fetchall()
                    if rows:
                        inds = pd.DataFrame(rows, columns=["ts", "ema9", "ema40"])
                        inds["ts"] = pd.to_datetime(inds["ts"], unit="ms", utc=True)
                        fig.add_trace(go.Scatter(x=inds["ts"], y=inds["ema9"], mode="lines", name="EMA9"))
                        fig.add_trace(go.Scatter(x=inds["ts"], y=inds["ema40"], mode="lines", name="EMA40"))
        except Exception as e:
            logger.debug("No pude leer indicators: %s", e)
        fig.update_layout(height=400, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay velas para mostrar.")

# Footer actions: CSV, train model
st.markdown("---")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Descargar CSV listado"):
        tmp = df_display.copy()
        tmp["created_at"] = tmp["created_at"].astype(str)
        csv = tmp.to_csv(index=False)
        st.download_button("Descargar CSV", csv, file_name="scores_list.csv", mime="text/csv")

with col2:
    if st.button("Recalcular scores para todo (puede tardar)"):
        with st.spinner("Recalculando scores para todos..."):
            assets_all = (cfg.get("assets",{}).get("cripto",[]) or []) + (cfg.get("assets",{}).get("acciones",[]) or [])
            for a in assets_all:
                try:
                    interval = "5m" if a in (cfg.get("assets",{}).get("cripto",[]) or []) else "1h"
                    compute_and_save_scores_for_asset(storage, a, interval, lookback_bars=500)
                except Exception as e:
                    logger.exception("Score fail for %s: %s", a, e)
            st.success("Scores recalculados.")
            st.rerun()()

with col3:
    if st.button("Entrenar IA (últimos assets seleccionados)"):
        st.warning("Entrenamiento IA lanzado desde UI. Revisa logs. Puede tardar.")
        # optional: call ai_train.train_model for selected asset(s)
        try:
            from core.ai_train import train_model
            if "sel_asset" in locals():
                res = train_model(storage, sel_asset, sel_interval, lookback=2000)
                st.success(f"Modelo entrenado: {res}")
            else:
                st.warning("Selecciona un símbolo antes de entrenar.")
        except Exception as e:
            st.error(f"Error entrenando IA: {e}")

st.write("Dashboard versión — integrado con storage_postgres, fetcher, score y ai_train")
