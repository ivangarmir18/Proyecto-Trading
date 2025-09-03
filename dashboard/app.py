"""
dashboard/app.py
Dashboard Postgres-first — versión robusta y compatible con PostgresStorage (psycopg2 pool)
- Compatible con tu core/storage_postgres.py actual (usa _get_conn(), health(), save_candles(), load_candles()).
- No usa SQLite. Fallback de engine sólo para diagnósticos si hace falta.
- Backfill incremental en background (solo si storage tiene save_candles).
- Mensajes claros en UI y logs útiles.
"""

# Asegurar que el proyecto raíz está en sys.path para que `import core` funcione
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import threading
import time
from typing import Tuple, List, Optional, Any
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import sqlalchemy as sa
import yaml

# -------------------------
# Helper: cargar assets desde CSV según config.json
# -------------------------
import json
import csv

def load_assets_from_csv(config_path: str = "config.json") -> list[str]:
    """
    Carga assets desde los CSV indicados en config.json bajo 'asset_files'.
    Retorna una lista de strings con los activos.
    """
    assets = []
    if not os.path.exists(config_path):
        return assets

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for key in ("cryptos", "stocks"):
        path = cfg.get("asset_files", {}).get(key)
        if path and os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and row[0].strip():
                        assets.append(row[0].strip())
    return assets


# Import del storage y fetcher (deben existir en tu repo)
from core.storage_postgres import PostgresStorage
from core.fetch import Fetcher

# Config logger
logger = logging.getLogger("dashboard")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(os.environ.get("DASHBOARD_LOG_LEVEL", "INFO"))

# -------------------------
# Helpers DB/compatibilidad
# -------------------------
def parse_db_url_short(db_url: str):
    try:
        p = urlparse(db_url)
        return {"host": p.hostname, "port": p.port, "db": p.path.lstrip("/") if p.path else "", "user": p.username}
    except Exception:
        return None

def try_execute_sql(storage: Any, sql: str, params: Optional[tuple] = None):
    """
    Ejecuta SQL usando mecanismos disponibles en storage:
      - si tiene _get_conn() (tu PostgresStorage): úsalo
      - si tiene engine: usar engine.connect()
      - si tiene get_connection(): llamarlo
    Devuelve cursor.fetchall() resultado o lanza excepción.
    """
    params = params or ()
    # 1) storage._get_conn()
    try:
        if hasattr(storage, "_get_conn"):
            with storage._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                rows = cur.fetchall()
                cur.close()
                return rows
    except Exception as e:
        logger.debug("try_execute_sql via _get_conn failed: %s", e)

    # 2) storage.engine
    try:
        engine = getattr(storage, "engine", None)
        if engine is not None:
            with engine.connect() as conn:
                res = conn.execute(sa.text(sql), params)
                return res.fetchall()
    except Exception as e:
        logger.debug("try_execute_sql via engine failed: %s", e)

    # 3) storage.get_connection() or storage.connect()
    try:
        for attr in ("get_connection", "connect", "get_conn"):
            if hasattr(storage, attr):
                fn = getattr(storage, attr)
                conn = fn() if callable(fn) else fn
                cur = conn.cursor()
                cur.execute(sql, params)
                rows = cur.fetchall()
                cur.close()
                return rows
    except Exception as e:
        logger.debug("try_execute_sql via alt conn failed: %s", e)

    raise RuntimeError("No se pudo ejecutar SQL: no hay método de conexión disponible en storage")

def safe_load_watchlist(storage: Any, config_path: str = "config.yml") -> Tuple[List[str], Optional[str]]:
    """
    Intenta cargar watchlist en este orden:
      1) Tabla 'watchlist' en Postgres (si existe)
      2) Archivo config.yml (key 'watchlist')
      3) Env WATCHLIST
    """
    # 1) intentar tabla watchlist via SQL si storage permite ejecutar SQL
    try:
        rows = try_execute_sql(storage, "SELECT asset FROM watchlist ORDER BY id LIMIT 100")
        if rows:
            assets = [r[0] for r in rows]
            return assets, "db"
    except Exception as e:
        logger.debug("No se pudo leer watchlist desde DB (continuando): %s", e)

    # 2) config.yml
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            wl = cfg.get("watchlist")
            if isinstance(wl, list):
                return wl, "config"
            elif isinstance(wl, str):
                return [x.strip() for x in wl.split(",") if x.strip()], "config"
    except Exception as e:
        logger.debug("Error leyendo config.yml: %s", e)

    # 3) ENV
    env_wl = os.getenv("WATCHLIST", "")
    if env_wl:
        assets = [x.strip() for x in env_wl.split(",") if x.strip()]
        if assets:
            return assets, "env"

    return [], None

def ensure_storage_connected() -> PostgresStorage:
    """
    Instancia PostgresStorage y valida conexión.
    Usa storage.health() si existe; si no, intenta buscar engine/conn o crea SQLAlchemy engine temporal.
    Devuelve la instancia storage (posiblemente con storage.engine añadido).
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL no definida. La aplicación requiere Postgres.")

    storage = PostgresStorage()  # usa tu implementación actual (psycopg2 pool)

    # Preferir health() si existe
    try:
        if hasattr(storage, "health") and callable(storage.health):
            h = storage.health()
            if h.get("ok", False):
                logger.info("Storage health OK (via storage.health())")
                return storage
            else:
                logger.warning("storage.health() devolvió error: %s", h.get("error"))
    except Exception as e:
        logger.debug("storage.health() fallo: %s", e)

    # Intentar encontrar engine/conn en attributes
    engine = getattr(storage, "engine", None)
    if engine is None:
        for alt in ("_pool", "get_engine", "db_engine", "connection", "conn", "get_connection"):
            if hasattr(storage, alt):
                try:
                    candidate = getattr(storage, alt)
                    engine = candidate() if callable(candidate) else candidate
                    if engine:
                        logger.info("Usando engine/conn desde attribute %s", alt)
                        break
                except Exception:
                    logger.debug("No usable engine from %s", alt)

    # Si no hay engine, crear uno temporal con SQLAlchemy a partir de DATABASE_URL
    if engine is None:
        try:
            engine = sa.create_engine(db_url, pool_pre_ping=True, future=True)
            # adjuntar para compatibilidad con el resto del código
            try:
                setattr(storage, "engine", engine)
                logger.info("Se creó un engine SQLAlchemy fallback y se adjuntó a storage")
            except Exception:
                logger.info("Se creó engine fallback pero no se pudo adjuntar a storage (se usará localmente)")
        except Exception as e:
            logger.exception("No se pudo crear engine fallback desde DATABASE_URL: %s", e)
            raise RuntimeError(f"No se pudo conectar a Postgres: {e}")

    # Verificar el engine con SELECT 1
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except Exception as e:
        logger.exception("Engine test falló: %s", e)
        raise RuntimeError(f"No se pudo conectar a Postgres (engine test failed): {e}")

    return storage

# -------------------------
# Backfill incremental (uso seguro)
# -------------------------
def start_background_backfill(storage: Any, assets: List[str], interval: str = "1h",
                              max_hours_per_asset: int = 24, run_interval_seconds: int = 60*30):
    """
    Inicia un hilo daemon que actualiza incrementalmente los assets. Solo se lanza si storage tiene save_candles.
    """
    if not assets:
        logger.info("No hay assets para backfill background")
        return

    if not hasattr(storage, "save_candles") or not callable(getattr(storage, "save_candles")):
        logger.info("Storage no soporta save_candles() — no se lanzará backfill background")
        return

    def _worker():
        logger.info("Backfill background thread arrancado")
        while True:
            now = pd.Timestamp.utcnow().tz_convert("UTC")
            for asset in assets:
                try:
                    # obtener last_ts si existe
                    last_ts = None
                    try:
                        last = try_execute_sql(storage, "SELECT MAX(ts) FROM candles WHERE asset=%s AND interval=%s", (asset, interval))
                        if last and last[0] and last[0][0] is not None:
                            last_ts = int(last[0][0])
                    except Exception:
                        logger.debug("No se pudo obtener last_ts via SQL (fallará con storage.load_candles)")

                    if last_ts:
                        since = pd.to_datetime(last_ts, unit="s", utc=True) + pd.Timedelta(seconds=1)
                    else:
                        since = now - pd.Timedelta(hours=max_hours_per_asset)

                    # llamar fetcher (se encargará de llamar a save_callback si lo hace)
                    fetcher = Fetcher(storage=storage)
                    logger.info("Backfill background: %s desde %s", asset, since)
                    try:
                        fetcher.backfill_range(asset=asset, interval=interval, start_ts_ms=int(since.timestamp()*1000),
                                               end_ts_ms=int(now.timestamp()*1000), per_call_limit=None,
                                               save_callback=lambda df, a, inter, meta: storage.save_candles(df))
                    except TypeError:
                        # si tu fetcher usa otra firma (compatibilidad), intenta call simple
                        try:
                            fetcher.backfill_range(asset=asset, interval=interval, since=int(since.timestamp()*1000), progress=False)
                        except Exception as e:
                            logger.exception("fetcher.backfill_range fallback failed for %s: %s", asset, e)
                except Exception:
                    logger.exception("Error en background backfill para %s", asset)
                time.sleep(1.0)
            logger.info("Backfill background: pasada completa — durmiendo %ds", run_interval_seconds)
            time.sleep(run_interval_seconds)

    t = threading.Thread(target=_worker, daemon=True, name="backfill-bg-thread")
    t.start()

# -------------------------
# UI principal
# -------------------------
def main():
    st.set_page_config(page_title="Portfa — Dashboard", layout="wide")
    st.title("Portfa — Dashboard")

    st.sidebar.markdown("## Estado y acciones")

    # Instanciar y validar storage (mensajes claros)
    try:
        storage = ensure_storage_connected()
    except Exception as e:
        st.sidebar.error("Fallo inicializando Postgres")
        st.error(f"No se pudo conectar a Postgres: {e}")
        st.info("Formato esperado: postgresql://<user>:<pass>@<host>:<port>/<db>?sslmode=require")
        logger.exception("Fallo inicializando Postgres: %s", e)
        return

    # Cargar watchlist (DB -> config -> env)
    assets, source = safe_load_watchlist(storage)
    # Si safe_load_watchlist no encontró nada, intentar cargar desde CSV
    if not assets:
        assets = load_assets_from_csv("config.json")  # path a tu config.json
    if assets:
        source = "csv"
        st.sidebar.success(f"Watchlist cargada desde CSV ({len(assets)} assets)")

    if not assets:
        st.warning("Watchlist deshabilitada (configuración requerida)")
        st.info("Opciones: crear tabla `watchlist` en Postgres; o poner WATCHLIST env; o config.yml con key 'watchlist'.")
        manual = st.text_input("Introducir watchlist temporal (comma-separated)", value="")
        if manual:
            assets = [x.strip() for x in manual.split(",") if x.strip()]
            source = "manual"
    else:
        st.sidebar.success(f"Watchlist cargada desde: {source} ({len(assets)} assets)")

    # Iniciar backfill en background solo si storage soporta save_candles
    if assets and hasattr(storage, "save_candles"):
        start_background_backfill(storage, assets, interval="1h", max_hours_per_asset=24, run_interval_seconds=60*30)

    # Sidebar controls
    with st.sidebar.expander("Acciones"):
        hours = st.number_input("Horas a traer (max por asset)", min_value=1, max_value=168, value=24)
        interval = st.selectbox("Intervalo", ["1h", "1d"], index=0)
        if st.button("Actualizar ahora (background)"):
            if not assets:
                st.error("No hay assets en la watchlist")
            else:
                # lanzar one-shot background updater
                threading.Thread(target=run_incremental_backfill, args=(storage, assets, interval, int(hours)), daemon=True).start()
                st.success("Backfill lanzado en background")

    # Mostrar watchlist y resumen
    st.subheader("Watchlist")
    if not assets:
        st.write("_No hay assets configurados_")
    else:
        rows = []
        for a in assets:
            last_ts = None
            try:
                last = try_execute_sql(storage, "SELECT MAX(ts) FROM candles WHERE asset=%s AND interval=%s", (a, interval))
                if last and last[0] and last[0][0] is not None:
                    last_ts = int(last[0][0])
            except Exception:
                # fallback a storage.load_candles si existe
                try:
                    if hasattr(storage, "load_candles"):
                        df_tmp = storage.load_candles(a, interval, limit=1)
                        if df_tmp is not None and not getattr(df_tmp, "empty", True):
                            last_ts = int(df_tmp["ts"].iloc[-1])
                except Exception:
                    pass
            last_dt = pd.to_datetime(last_ts, unit="s", utc=True) if last_ts else "-"
            rows.append({"asset": a, "last_ts": last_ts or "-", "last_datetime_utc": last_dt})
        df = pd.DataFrame(rows)
        st.dataframe(df)

        cols = st.columns([3, 1, 1, 1])
        with cols[0]:
            selected = st.selectbox("Selecciona un asset", options=assets)
        with cols[1]:
            if st.button("Ver últimos datos"):
                try:
                    if hasattr(storage, "load_candles"):
                        df_c = storage.load_candles(selected, interval, start_ts=None, end_ts=None)
                        if df_c is None or getattr(df_c, "empty", True):
                            st.warning("No hay datos históricos disponibles para este asset")
                        else:
                            st.write(df_c.tail(200))
                    else:
                        st.error("Storage no soporta load_candles()")
                except Exception as e:
                    logger.exception("Error mostrando candles: %s", e)
                    st.error(f"Error cargando datos: {e}")
        with cols[2]:
            if st.button("Forzar backfill 6h (background)"):
                if not assets:
                    st.error("No hay assets")
                else:
                    threading.Thread(target=run_incremental_backfill, args=(storage, [selected], "1h", 6), daemon=True).start()
                    st.success(f"Backfill 6h lanzado para {selected}")
        with cols[3]:
            if st.button("Stats"):
                try:
                    rows = try_execute_sql(storage, "SELECT COUNT(1), MIN(ts), MAX(ts) FROM candles WHERE asset=%s AND interval=%s", (selected, interval))
                    if rows and rows[0]:
                        cnt, first_ts, last_ts = rows[0]
                        st.write({
                            "count": int(cnt) if cnt is not None else 0,
                            "first_ts": pd.to_datetime(first_ts, unit="s", utc=True) if first_ts else None,
                            "last_ts": pd.to_datetime(last_ts, unit="s", utc=True) if last_ts else None
                        })
                except Exception as e:
                    logger.exception("Error obteniendo stats: %s", e)
                    st.error(f"Error obteniendo stats: {e}")

    st.markdown("---")
    st.caption("Portfa Dashboard — Postgres-only. Revisa DATABASE_URL en entorno si hay problemas.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Si la app no conecta, revisa DATABASE_URL y los logs del servicio.")

# helper para lanzar backfill one-shot (compatibilidad con run_incremental_backfill anterior)
def run_incremental_backfill(storage: Any, assets: List[str], interval: str = "1h", max_hours: int = 24):
    """
    Simple wrapper que lanza backfill uno por uno (sin bloquear). Usa Fetcher y storage.save_candles si está.
    """
    try:
        fetcher = Fetcher(storage=storage)
        now = pd.Timestamp.utcnow().tz_convert("UTC")
        for asset in assets:
            try:
                # calcular since similar a background worker
                last = try_execute_sql(storage, "SELECT MAX(ts) FROM candles WHERE asset=%s AND interval=%s", (asset, interval))
                last_ts = int(last[0][0]) if last and last[0][0] is not None else None
            except Exception:
                last_ts = None
            if last_ts:
                since = pd.to_datetime(last_ts, unit="s", utc=True) + pd.Timedelta(seconds=1)
            else:
                since = now - pd.Timedelta(hours=max_hours)
            logger.info("One-shot backfill %s since %s", asset, since)
            try:
                # intentar la firma moderna (start_ts_ms, end_ts_ms)
                fetcher.backfill_range(asset=asset, interval=interval,
                                       start_ts_ms=int(since.timestamp()*1000), end_ts_ms=int(now.timestamp()*1000),
                                       save_callback=(lambda df, a, inter, meta: storage.save_candles(df)) if hasattr(storage, "save_candles") else None,
                                       progress=False)
            except TypeError:
                # fallback a firma antigua
                try:
                    fetcher.backfill_range(asset=asset, interval=interval, since=int(since.timestamp()*1000), progress=False)
                except Exception as e:
                    logger.exception("fetcher.backfill_range fallback failed: %s", e)
    except Exception:
        logger.exception("run_incremental_backfill failed")

if __name__ == "__main__":
    main()
