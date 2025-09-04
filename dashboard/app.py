# dashboard/app.py
"""
Streamlit dashboard for Watchlist
--------------------------------
Completo, profesional, optimizado para integración con core.Orchestrator / StorageAdapter (Postgres)
Características:
 - Sidebar: DB health, timeframe selector, weights sliders (normalizar / guardar), retention config
 - Main: tabla de assets (precio, last_ts, last_score, score change), filtros, sort
 - Detalle por asset: velas (descarga CSV), indicadores, evolución del score (últimos N), botones de acción
 - Botones globales: Backfill (por asset/timeframe), Compute Indicators, Compute Scores, Run Backtest, Train IA
 - Jobs: ejecución en background mediante ThreadPoolExecutor con estado en st.session_state
 - Logs en UI: handler que captura logs y muestra últimos mensajes
 - Descargas CSV para velas / scores / backtests
 - Seguridad: inputs validados, try/except con feedback
 - Intenta usar Orchestrator desde core.orchestrator.make_orchestrator
"""

from __future__ import annotations
import os
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import streamlit as st
import pandas as pd

# core modules (assume you already have these in core/)
from core.orchestrator import make_orchestrator
from core.adapter import normalize_weights

# Basic logging setup (we will also capture logs to show in UI)
logger = logging.getLogger("watchlist_dashboard")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(ch)


# ---------- Utility: Streamlit log capture ----------
class StreamlitLogHandler(logging.Handler):
    def __init__(self, capacity: int = 500):
        super().__init__()
        self.capacity = capacity
        if "dashboard_logs" not in st.session_state:
            st.session_state["dashboard_logs"] = []

    def emit(self, record):
        msg = self.format(record)
        logs = st.session_state.get("dashboard_logs", [])
        logs.append(msg)
        # keep capacity
        if len(logs) > self.capacity:
            logs = logs[-self.capacity:]
        st.session_state["dashboard_logs"] = logs


_stream_handler = StreamlitLogHandler()
_stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(_stream_handler)


# ---------- App-level constants and defaults ----------
DEFAULT_WEIGHTS = {"ema": 0.25, "support": 0.2, "atr": 0.2, "macd": 0.15, "rsi": 0.1, "fibonacci": 0.1}
DEFAULT_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "12h", "1d"]
DEFAULT_INDICATOR_LOOKBACK = 500

# Thread executor for background jobs (shared)
if "executor" not in st.session_state:
    st.session_state["executor"] = ThreadPoolExecutor(max_workers=int(os.getenv("DASH_MAX_WORKERS", "4")))

# jobs registry
if "jobs" not in st.session_state:
    st.session_state["jobs"] = {}  # job_id -> Future

# orchestrator singleton
if "orchestrator" not in st.session_state:
    try:
        cfg_path = os.getenv("WATCHLIST_CONFIG", "config.json")
        config = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                config = json.load(f)
        st.session_state["orchestrator"] = make_orchestrator(config=config, db_url=os.getenv("DATABASE_URL"))
        logger.info("Orchestrator instantiated in session_state.")
    except Exception as e:
        logger.exception("Failed to instantiate Orchestrator: %s", e)
        st.session_state["orchestrator"] = None

orch = st.session_state.get("orchestrator")

# helper to ensure session defaults
if "weights" not in st.session_state:
    st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
if "timeframe" not in st.session_state:
    st.session_state["timeframe"] = DEFAULT_TIMEFRAMES[0]
if "selected_asset" not in st.session_state:
    st.session_state["selected_asset"] = None
if "dashboard_logs" not in st.session_state:
    st.session_state["dashboard_logs"] = []


# ---------- Helpers ----------
def human_ts(ts: Optional[int]):
    if not ts:
        return "-"
    try:
        return pd.to_datetime(int(ts), unit="s")
    except Exception:
        return str(ts)


def run_job(fn, *args, job_name: str = "job", **kwargs) -> str:
    """
    Submits a job to executor and stores Future in session_state jobs.
    Returns job_id.
    """
    job_id = f"{job_name}-{int(time.time()*1000)}"
    logger.info("Submitting job %s", job_id)

    def _wrap(fn, *args, **kwargs):
        try:
            logger.info("Job %s started", job_id)
            res = fn(*args, **kwargs)
            logger.info("Job %s finished: %s", job_id, getattr(res, "__repr__", lambda: res)())
            return res
        except Exception as e:
            logger.exception("Job %s failed: %s", job_id, e)
            raise

    fut: Future = st.session_state["executor"].submit(_wrap, fn, *args, **kwargs)
    st.session_state["jobs"][job_id] = fut
    return job_id


def job_status(job_id: str) -> Dict[str, Any]:
    fut = st.session_state["jobs"].get(job_id)
    if not fut:
        return {"status": "unknown"}
    if fut.running():
        return {"status": "running"}
    if fut.done():
        try:
            res = fut.result()
            return {"status": "done", "result": res}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    return {"status": "pending"}


def get_assets_list() -> List[str]:
    try:
        if orch:
            return orch.list_assets()
    except Exception:
        logger.exception("list_assets failed")
    # fallback: use config or defaults
    cfg_path = os.getenv("WATCHLIST_CONFIG", "config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
                return cfg.get("assets", {}).get("cryptos", []) + cfg.get("assets", {}).get("stocks", [])
    except Exception:
        pass
    return ["BTCUSDT", "ETHUSDT", "AAPL", "MSFT"]


def safe_export_csv(df: pd.DataFrame, name: str = "export.csv"):
    if df is None or df.empty:
        st.warning("No hay datos para exportar.")
        return
    st.download_button(label=f"Descargar {name}", data=df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")


# ---------- Core actions (call orchestrator) ----------
def action_backfill(asset: str, interval: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None, provider: Optional[str] = None):
    if not orch:
        raise RuntimeError("Orchestrator no inicializado.")
    return orch.fetch_and_store(asset=asset, interval=interval, start_ts=start_ts, end_ts=end_ts, provider=provider)


def action_compute_indicators(asset: str, interval: str, lookback: int = DEFAULT_INDICATOR_LOOKBACK):
    if not orch:
        raise RuntimeError("Orchestrator no inicializado.")
    return orch.compute_and_store_indicators(asset=asset, interval=interval, lookback=lookback)


def action_compute_scores(asset: str, interval: str, weights: Dict[str, float]):
    if not orch:
        raise RuntimeError("Orchestrator no inicializado.")
    return orch.compute_and_store_scores(asset=asset, interval=interval, weights=weights)


def action_apply_retention():
    if not orch:
        raise RuntimeError("Orchestrator no inicializado.")
    return orch.apply_retention()


def action_health_check():
    if not orch:
        return {"ok": False, "error": "Orchestrator missing"}
    return orch.health_check()


# ---------- UI building blocks ----------
def sidebar_panel():
    st.sidebar.title("Watchlist · Configuración")
    # DB / health
    st.sidebar.subheader("Base de datos")
    db_info = action_health_check()
    if db_info.get("ok"):
        st.sidebar.success("DB disponible")
        st.sidebar.write(f"Ping: {db_info.get('db_ping', True)}")
    else:
        st.sidebar.error("DB no disponible")
        st.sidebar.write(str(db_info.get("db_err", "No details")))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Timeframe / Retención")
    tf = st.sidebar.selectbox("Timeframe", DEFAULT_TIMEFRAMES, index=DEFAULT_TIMEFRAMES.index(st.session_state["timeframe"]) if st.session_state.get("timeframe") in DEFAULT_TIMEFRAMES else 0)
    st.session_state["timeframe"] = tf

    st.sidebar.subheader("Score weights")
    # two-column layout for compactness
    weight_keys = ["ema", "support", "atr", "macd", "rsi", "fibonacci"]
    cols = st.sidebar.columns(2)
    new_weights = {}
    for i, k in enumerate(weight_keys):
        col = cols[i % 2]
        val = col.number_input(k, min_value=0.0, max_value=10.0, value=float(st.session_state["weights"].get(k, DEFAULT_WEIGHTS[k])), step=0.01, key=f"w_{k}")
        new_weights[k] = float(val)
    # actions for weights
    if st.sidebar.button("Normalizar pesos"):
        normalized = normalize_weights(new_weights, expected_keys=weight_keys)
        st.session_state["weights"] = normalized
        st.sidebar.success(f"Pesos normalizados (suma={sum(normalized.values()):.3f})")
    st.sidebar.write("Pesos actuales:")
    st.sidebar.json(st.session_state["weights"])

    if st.sidebar.button("Guardar pesos en sesión"):
        st.session_state["weights"] = normalize_weights(new_weights, expected_keys=weight_keys)
        st.sidebar.success("Pesos guardados en sesión. (Si quieres persistir en DB, implementa endpoint o configuración).")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Acciones globales")
    if st.sidebar.button("Aplicar retención (limpiar DB)"):
        try:
            jid = run_job(action_apply_retention, job_name="apply_retention")
            st.sidebar.info(f"Retention job lanzado: {jid}")
        except Exception as e:
            st.sidebar.error(f"Error lanzando retention: {e}")

    if st.sidebar.button("Ver logs recientes"):
        st.sidebar.write("\n".join(st.session_state.get("dashboard_logs", [])[-50:]))

    st.sidebar.markdown("---")
    st.sidebar.caption("Variables de entorno: asegúrate DATABASE_URL en Render para usar Postgres.")

def asset_table_panel():
    st.header("Assets · Watchlist")
    assets = get_assets_list()
    # filter input
    q = st.text_input("Filtrar activos (por símbolo)", value="")
    if q:
        assets = [a for a in assets if q.lower() in a.lower()]
    # build a table with basic info: latest close (for timeframe), last_score, last_ts
    rows = []
    for sym in assets:
        try:
            last_ts = orch.get_last_candle_ts(sym, st.session_state["timeframe"]) if orch else None
        except Exception:
            last_ts = None
        last_score = "-"
        # read last score via storage impl if available (best effort)
        try:
            impl = getattr(orch.storage, "impl", None)
            if impl and hasattr(impl, "engine"):
                with impl.engine.connect() as conn:
                    qsql = "select score from scores where asset = :asset order by ts desc limit 1"
                    r = conn.execute(qsql, {"asset": sym}).fetchone()
                    if r:
                        last_score = float(r[0])
        except Exception:
            last_score = "-"
        rows.append({"asset": sym, "timeframe": st.session_state["timeframe"], "last_ts": human_ts(last_ts), "last_score": last_score})
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No hay activos listados. Añádelos en config.json o al storage.")
        return
    # allow sorting and selection
    st.dataframe(df, use_container_width=True)
    # select asset
    sel = st.selectbox("Seleccionar asset para ver detalle", ["--"] + assets)
    if sel and sel != "--":
        st.session_state["selected_asset"] = sel
        asset_detail_panel(sel)
    else:
        st.info("Selecciona un activo para ver detalle o lanzar acciones sobre él.")


def asset_detail_panel(asset: str):
    st.subheader(f"Detalle: {asset} · {st.session_state['timeframe']}")
    col1, col2, col3 = st.columns([2, 1, 1])
    # basic stats
    try:
        last_ts = orch.get_last_candle_ts(asset, st.session_state["timeframe"])
    except Exception:
        last_ts = None
    col1.metric("Última vela (ts)", human_ts(last_ts) if last_ts else "-")
    # show backfill status
    try:
        bf = orch.storage.get_backfill_status(asset, st.session_state["timeframe"])
        if bf:
            col2.write(f"Backfill: {bf.get('status')} @ {human_ts(bf.get('last_ts'))}")
        else:
            col2.write("Backfill: -")
    except Exception:
        col2.write("Backfill: -")
    # last score quick
    try:
        impl = getattr(orch.storage, "impl", None)
        last_score = None
        if impl and hasattr(impl, "engine"):
            with impl.engine.connect() as conn:
                r = conn.execute("select score, ts from scores where asset=:asset order by ts desc limit 1", {"asset": asset}).fetchone()
                if r:
                    last_score = {"score": float(r[0]), "ts": int(r[1])}
                    col3.metric("Último score", f"{last_score['score']:.3f}", delta="-")
                else:
                    col3.metric("Último score", "-")
    except Exception:
        col3.metric("Último score", "-")
    st.markdown("----")
    # actions for this asset
    act_col1, act_col2, act_col3 = st.columns(3)
    if act_col1.button("Backfill (rápido)"):
        try:
            jid = run_job(action_backfill, asset, st.session_state["timeframe"], job_name=f"backfill-{asset}-{st.session_state['timeframe']}")
            st.success(f"Backfill lanzado: {jid}")
        except Exception as e:
            st.error(f"Error lanzando backfill: {e}")

    if act_col2.button("Compute indicators"):
        try:
            jid = run_job(action_compute_indicators, asset, st.session_state["timeframe"], DEFAULT_INDICATOR_LOOKBACK, job_name=f"indicators-{asset}")
            st.success(f"Compute indicators lanzado: {jid}")
        except Exception as e:
            st.error(f"Error lanzando compute indicators: {e}")

    if act_col3.button("Compute scores"):
        try:
            w = normalize_weights(st.session_state.get("weights", DEFAULT_WEIGHTS))
            jid = run_job(action_compute_scores, asset, st.session_state["timeframe"], w, job_name=f"scores-{asset}")
            st.success(f"Compute scores lanzado: {jid}")
        except Exception as e:
            st.error(f"Error lanzando compute scores: {e}")

    st.markdown("----")
    # show last N candles and indicators
    try:
        df_candles = pd.DataFrame(orch.storage.export_last_candles(asset, st.session_state["timeframe"], limit=500))
        if not df_candles.empty:
            st.caption("Últimas velas (descargar archivo CSV si quieres).")
            safe_export_csv(df_candles, name=f"{asset}_{st.session_state['timeframe']}_candles.csv")
            st.dataframe(df_candles.head(200), use_container_width=True)
        else:
            st.info("No hay velas para este asset/timeframe. Ejecuta Backfill.")
    except Exception as e:
        logger.exception("Error mostrando velas: %s", e)
        st.error("Error mostrando velas: " + str(e))

    # show recent indicators (from indicators table) if present
    try:
        impl = getattr(orch.storage, "impl", None)
        if impl and hasattr(impl, "engine"):
            import pandas as pd
            conn = impl.engine.connect()
            q = "select ts, value from indicators where asset = :asset and interval = :interval order by ts desc limit 200"
            df_ind = pd.read_sql(q, conn, params={"asset": asset, "interval": st.session_state["timeframe"]})
            conn.close()
            if not df_ind.empty:
                # expand value dict
                vals = pd.json_normalize(df_ind['value'])
                vals['ts'] = df_ind['ts'].values
                st.caption("Indicadores recientes (ultimo rango):")
                st.dataframe(vals.head(200), use_container_width=True)
            else:
                st.info("No hay indicadores calculados (ejecuta Compute indicators).")
    except Exception:
        st.info("No hay indicadores o error al leerlos.")

    # recent scores
    try:
        impl = getattr(orch.storage, "impl", None)
        if impl and hasattr(impl, "engine"):
            import pandas as pd
            conn = impl.engine.connect()
            q = "select ts, method, score, details from scores where asset = :asset order by ts desc limit 200"
            df_scores = pd.read_sql(q, conn, params={"asset": asset})
            conn.close()
            if not df_scores.empty:
                st.caption("Scores recientes:")
                st.dataframe(df_scores.head(200), use_container_width=True)
                safe_export_csv(df_scores, name=f"{asset}_scores.csv")
    except Exception:
        logger.exception("Error leyendo scores")
        st.info("No hay scores o error al leerlos.")


def jobs_panel():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Jobs activos")
    jobs = st.session_state.get("jobs", {})
    if not jobs:
        st.sidebar.write("No hay jobs en ejecución.")
        return
    for jid, fut in list(jobs.items())[-10:]:
        st.sidebar.write(f"{jid} : {('running' if not fut.done() else 'done')}")
        if fut.done():
            try:
                res = fut.result()
                st.sidebar.write(f"  → result: {str(res)[:200]}")
            except Exception as e:
                st.sidebar.write(f"  → error: {e}")


def logs_panel():
    st.subheader("Logs del sistema (últimas entradas)")
    logs = st.session_state.get("dashboard_logs", [])[-150:]
    if not logs:
        st.info("No hay logs todavía.")
    else:
        st.code("\n".join(logs[-150:]))


# ---------- Main layout ----------
def main():
    st.set_page_config(page_title="Watchlist Dashboard", layout="wide")
    sidebar_panel()
    # top toolbar
    top_col1, top_col2, top_col3 = st.columns([3, 1, 1])
    top_col1.title("Watchlist · Dashboard")
    # quick health
    hc = action_health_check()
    if hc.get("ok"):
        top_col2.success("DB OK")
    else:
        top_col2.error("DB ERROR")
    # quick refresh button
    if top_col3.button("Refrescar datos"):
        # small action: clear logs and re-fetch assets list
        st.experimental_rerun()

    # main panels
    asset_table_panel()
    st.markdown("---")
    logs_panel()

    # jobs panel (sidebar)
    jobs_panel()

    st.markdown("---")
    st.caption("Interfaz avanzada para gestión de backfill, indicadores y scores. Usa los botones por asset para acciones puntuales.")

if __name__ == "__main__":
    main()
