"""
dashboard/app.py
Dashboard Postgres-first para Portfa

Características:
- Forzar conexión a Postgres (si no hay DATABASE_URL o falla la conexión -> error claro)
- Diagnóstico en sidebar (host/port/db sin exponer contraseña)
- Watchlist: carga desde tabla `watchlist` en Postgres, o desde config.yml, o desde env WATCHLIST
- Botón para lanzar backfill incremental en background (por asset o global)
- Mostrar últimos candles de un asset (si storage ofrece load_candles)
- No usa SQLite en ningún caso
"""

import os
import logging
import threading
import time
from typing import Tuple, List
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import sqlalchemy as sa
import yaml

# Ajusta según tu estructura de proyecto
from core.storage_postgres import PostgresStorage
from core.fetch import Fetcher

logger = logging.getLogger(__name__)

# -----------------------------
# Utilidades DB / diagnostico
# -----------------------------
def parse_db_url_short(db_url: str):
    """Parsea la URL para mostrar host/port/db/user sin exponer contraseña"""
    try:
        p = urlparse(db_url)
        return {
            "host": p.hostname,
            "port": p.port,
            "db": p.path.lstrip("/") if p.path else "",
            "user": p.username,
        }
    except Exception:
        return None

def require_postgres_storage() -> PostgresStorage:
    """
    Intenta instanciar PostgresStorage y hace un chequeo rápido (SELECT 1).
    Si falla, lanza RuntimeError — la app no continuará sin Postgres.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL no definida. La aplicación requiere Postgres (no se permite SQLite).")

    parsed = parse_db_url_short(db_url)
    if parsed:
        st.sidebar.info(f"DB: host={parsed['host']} port={parsed['port']} db={parsed['db']}")
    else:
        st.sidebar.warning("DATABASE_URL definida pero no se pudo parsear para mostrar diagnóstico")

    try:
        storage = PostgresStorage()
        engine = getattr(storage, "engine", None)
        if engine is None:
            raise RuntimeError("PostgresStorage no expone 'engine'")
        # chequeo rápido de aliveness
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        st.sidebar.success("Conectado a Postgres")
        return storage
    except Exception as e:
        logger.exception("No se pudo conectar a Postgres: %s", e)
        raise RuntimeError(f"No se pudo conectar a Postgres: {e}")

# ---- START: background incremental updater (paste after storage init) ----
import threading
import time
import pandas as pd
import logging
import sqlalchemy as sa

logger = logging.getLogger(__name__)

# PARAMETERS - ajustalos según tolerancia
MAX_HOURS_PER_ASSET_PER_RUN = 24    # máximo que traerá por asset en cada pasada
RUN_INTERVAL_SECONDS = 60 * 30     # cada cuánto corre la pasada completa (30 min)
SLEEP_BETWEEN_ASSETS = 1.0          # segundos entre assets para suavizar peticiones
EMPTY_CHUNK_BREAK = 3              # si 3 chunks vacíos seguidos, asumimos no hay más histórico

def get_last_ts_for(asset: str, interval: str, engine) -> int | None:
    q = "SELECT MAX(ts) AS last_ts FROM candles WHERE asset = :asset AND interval = :interval"
    try:
        with engine.connect() as conn:
            r = conn.execute(sa.text(q), {"asset": asset, "interval": interval}).fetchone()
        return int(r["last_ts"]) if r and r["last_ts"] is not None else None
    except Exception as e:
        logger.exception("get_last_ts_for error: %s", e)
        return None

def background_incremental_loop(storage, assets, interval="1h"):
    """
    Loop que corre indefinidamente cada RUN_INTERVAL_SECONDS y pide solo los datos faltantes,
    limitado a MAX_HOURS_PER_ASSET_PER_RUN por asset para cada pasada.
    """
    engine = getattr(storage, "engine", None)
    if engine is None:
        logger.error("Storage no expone engine. Cancelando background_incremental_loop")
        return
    fetcher = None
    while True:
        try:
            if not assets:
                logger.info("No assets en watchlist, saltando esta pasada")
                time.sleep(RUN_INTERVAL_SECONDS)
                continue

            # Instanciar fetcher cada pasada para evitar recursos colgados
            fetcher = Fetcher(storage=storage)

            now = pd.Timestamp.utcnow().tz_convert("UTC")
            for asset in assets:
                try:
                    last_ts = get_last_ts_for(asset, interval, engine)
                    if last_ts:
                        since = pd.to_datetime(last_ts, unit='s', utc=True) + pd.Timedelta(seconds=1)
                    else:
                        # si no hay datos, traemos solo MAX_HOURS_PER_ASSET_PER_RUN retroactivo
                        since = now - pd.Timedelta(hours=MAX_HOURS_PER_ASSET_PER_RUN)

                    # no pedir más de MAX_HOURS_PER_ASSET_PER_RUN
                    earliest = now - pd.Timedelta(hours=MAX_HOURS_PER_ASSET_PER_RUN)
                    if since < earliest:
                        since = earliest

                    logger.info("Background update: asset=%s interval=%s since=%s", asset, interval, since)

                    # Ejecutar backfill_range; tu fetcher debe manejar chunks vacíos y save_callback
                    fetcher.backfill_range(asset=asset, interval=interval, since=since, progress=False)

                except Exception:
                    logger.exception("Error en background update para %s", asset)

                time.sleep(SLEEP_BETWEEN_ASSETS)

        except Exception:
            logger.exception("Excepción inesperada en background incremental loop")
        # Esperar antes de la siguiente pasada completa
        time.sleep(RUN_INTERVAL_SECONDS)

def start_background_incremental(storage, assets, interval="1h"):
    t = threading.Thread(target=background_incremental_loop, args=(storage, assets, interval), daemon=True)
    t.start()
    logger.info("Background incremental thread started (daemon)")

# Example of starting it (put after you resolved 'assets' from watchlist)
# start_background_incremental(storage, assets, interval="1h")

# ---- END: background incremental updater ----

# -----------------------------
# Watchlist: DB -> config.yml -> ENV
# -----------------------------
def load_watchlist(storage: PostgresStorage | None, config_path: str = "config.yml") -> Tuple[List[str], str | None]:
    """
    Intenta cargar la watchlist en este orden:
     1) DB: tabla `watchlist` (campo asset)
     2) config.yml clave `watchlist`
     3) ENV WATCHLIST (comma-separated)
    Devuelve: (assets_list, source) donde source es 'db'|'config'|'env'|'manual' o None
    """
    # 1) DB
    if storage is not None:
        try:
            engine = getattr(storage, "engine", None)
            if engine is not None:
                q = "SELECT asset FROM watchlist ORDER BY id"
                with engine.connect() as conn:
                    rows = conn.execute(sa.text(q)).fetchall()
                if rows:
                    assets = [r["asset"] if "asset" in r.keys() else r[0] for r in rows]
                    return assets, "db"
        except Exception as e:
            logger.debug("No se pudo leer watchlist desde DB (seguir intentando): %s", e)

    # 2) config.yml
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            wl = cfg.get("watchlist")
            if isinstance(wl, str):
                assets = [x.strip() for x in wl.split(",") if x.strip()]
            elif isinstance(wl, list):
                assets = wl
            else:
                assets = []
            if assets:
                return assets, "config"
        except Exception as e:
            logger.debug("Error leyendo config.yml: %s", e)

    # 3) ENV
    env_wl = os.getenv("WATCHLIST")
    if env_wl:
        assets = [x.strip() for x in env_wl.split(",") if x.strip()]
        if assets:
            return assets, "env"

    return [], None

# -----------------------------
# Backfill incremental en background
# -----------------------------
def run_incremental_backfill_assets(storage: PostgresStorage, assets: List[str],
                                   interval: str = "1h", max_hours: int = 24, sleep_between: float = 1.0):
    """
    Lanza un job en background que para cada asset consulta MAX(ts) y pide solo lo que falta
    (hasta max_hours de ventana).
    """
    def _job():
        logger.info("Iniciando backfill incremental background para %s", assets)
        fetcher = Fetcher(storage=storage)
        engine = getattr(storage, "engine", None)
        now = pd.Timestamp.utcnow().tz_convert("UTC")
        for asset in assets:
            try:
                q = "SELECT MAX(ts) AS last_ts FROM candles WHERE asset = :asset AND interval = :interval"
                with engine.connect() as conn:
                    r = conn.execute(sa.text(q), {"asset": asset, "interval": interval}).fetchone()
                last_ts = int(r["last_ts"]) if r and r["last_ts"] is not None else None
                if last_ts:
                    since = pd.to_datetime(last_ts, unit="s", utc=True) + pd.Timedelta(seconds=1)
                else:
                    since = now - pd.Timedelta(hours=max_hours)
                earliest = now - pd.Timedelta(hours=max_hours)
                if since < earliest:
                    since = earliest
                logger.info("Backfilling %s desde %s (cap %sh)", asset, since, max_hours)
                fetcher.backfill_range(asset=asset, interval=interval, since=since, progress=False)
            except Exception:
                logger.exception("Error backfilling %s", asset)
            time.sleep(sleep_between)
        logger.info("Backfill incremental acabado")
    threading.Thread(target=_job, daemon=True).start()

# -----------------------------
# Helpers UI / DB info
# -----------------------------
def get_asset_last_ts(storage: PostgresStorage, asset: str, interval: str = "1h"):
    engine = getattr(storage, "engine", None)
    if engine is None:
        return None
    try:
        q = "SELECT MAX(ts) as last_ts FROM candles WHERE asset = :asset AND interval = :interval"
        with engine.connect() as conn:
            r = conn.execute(sa.text(q), {"asset": asset, "interval": interval}).fetchone()
        return int(r["last_ts"]) if r and r["last_ts"] is not None else None
    except Exception:
        return None

# -----------------------------
# Main UI
# -----------------------------
def main():
    st.set_page_config(page_title="Portfa — Dashboard", layout="wide")
    st.title("Portfa — Dashboard")

    # Sidebar - estado
    st.sidebar.markdown("## Estado y acciones")
    # Forzar Postgres: si no conecta, mostramos error y no continuamos
    try:
        storage = require_postgres_storage()
    except Exception as e:
        st.sidebar.error(f"Fallo inicializando Postgres: {e}")
        st.error("La aplicación requiere PostgreSQL para funcionar. Define DATABASE_URL correctamente y recarga.")
        # mostrar hint para el usuario (no exponer contraseña)
        st.info("Formato esperado: postgres://<user>:<pass>@<host>:<port>/<db>?sslmode=require")
        return

    # Cargar watchlist (DB > config > env)
    assets, wl_source = load_watchlist(storage)
    # Iniciar el thread que se actualiza en background (no bloquear la UI)
    start_background_incremental(storage, assets, interval="1h")

    if not assets:
        st.warning("Watchlist deshabilitada (configuración requerida)")
        st.info("""
        Activa la watchlist con una de las siguientes opciones:
        - Crear tabla `watchlist` en Postgres e insertar filas (columna `asset`)
        - Añadir `watchlist` en config.yml
        - Definir variable de entorno WATCHLIST con valores separados por comas
        """)
        manual = st.text_input("Introducir watchlist temporal (comma-separated)", value="")
        if manual:
            assets = [x.strip() for x in manual.split(",") if x.strip()]
            wl_source = "manual"
    else:
        st.sidebar.success(f"Watchlist cargada desde: {wl_source} ({len(assets)} assets)")

    # Sidebar actions
    with st.sidebar.expander("Acciones"):
        st.markdown("### Actualizaciones")
        hours = st.number_input("Horas a traer (max por asset)", min_value=1, max_value=168, value=24)
        interval = st.selectbox("Intervalo", ["1h", "1d"], index=0)
        sleep_between = st.number_input("Segundos entre assets", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        if st.button("Actualizar ahora (background)"):
            if not assets:
                st.error("No hay assets en la watchlist para actualizar")
            else:
                run_incremental_backfill_assets(storage, assets, interval=interval, max_hours=int(hours), sleep_between=float(sleep_between))
                st.success("Backfill lanzado en background para la watchlist")

    # Sección principal: ver/watchlist y detalles
    st.subheader("Watchlist")
    if not assets:
        st.write("_No hay assets configurados en la watchlist._")
    else:
        # Mostrar mini tabla con asset + última ts humana
        info_rows = []
        for a in assets:
            last_ts = get_asset_last_ts(storage, a, interval=interval)
            last_dt = pd.to_datetime(last_ts, unit="s", utc=True) if last_ts else None
            info_rows.append({"asset": a, "last_ts": last_ts or "-", "last_datetime_utc": last_dt or "-"})
        info_df = pd.DataFrame(info_rows)
        st.dataframe(info_df)

        cols = st.columns([2, 1, 1, 1])
        with cols[0]:
            selected = st.selectbox("Selecciona un asset", options=assets)
        with cols[1]:
            show_last = st.button("Ver últimos datos")
        with cols[2]:
            if st.button("Forzar backfill 6h (background)"):
                run_incremental_backfill_assets(storage, [selected], interval=interval, max_hours=6, sleep_between=0.5)
                st.success(f"Backfill 6h lanzado para {selected}")
        with cols[3]:
            if st.button("Mostrar stats"):
                try:
                    engine = getattr(storage, "engine")
                    q = "SELECT COUNT(1) as cnt, MIN(ts) as first_ts, MAX(ts) as last_ts FROM candles WHERE asset = :asset AND interval = :interval"
                    with engine.connect() as conn:
                        r = conn.execute(sa.text(q), {"asset": selected, "interval": interval}).fetchone()
                    st.write({
                        "count": int(r["cnt"]),
                        "first_ts": pd.to_datetime(r["first_ts"], unit="s", utc=True) if r["first_ts"] else None,
                        "last_ts": pd.to_datetime(r["last_ts"], unit="s", utc=True) if r["last_ts"] else None
                    })
                except Exception as e:
                    logger.exception("Error obteniendo stats: %s", e)
                    st.error(f"Error obteniendo stats: {e}")

        # Mostrar candles si se solicita
        if show_last:
            try:
                df = storage.load_candles(selected, interval, start_ts=None, end_ts=None)
                if df is None or getattr(df, "empty", True):
                    st.warning("No hay datos históricos disponibles para este asset")
                else:
                    st.write(df.tail(200))
            except Exception as e:
                logger.exception("Error mostrando candles: %s", e)
                st.error(f"Error cargando datos: {e}")

    st.markdown("---")
    st.caption("Portfa Dashboard — Postgres-only. Si hay problemas con la conexión revisa DATABASE_URL en el entorno y los logs.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Si la app no conecta a Postgres, revisa la variable de entorno DATABASE_URL y que tu DB acepte conexiones externas.")

if __name__ == "__main__":
    main()
