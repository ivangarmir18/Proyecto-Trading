# dashboard/utils.py
"""
Dashboard Utilities para Proyecto-Trading

Este módulo combina TODO lo que tu proyecto original usaba + mejoras:
- Funciones legacy (no rompen imports de app.py)
- Carga de assets desde CSV y DB
- Gestión de watchlist (leer/guardar CSV)
- Cache de velas con PostgresStorage
- Render de listas largas con scroll en Streamlit
- Manejo de errores robusto y logging
"""

import os
import logging
from typing import List, Optional
import pandas as pd
import streamlit as st
from functools import lru_cache

# Importa tu Storage
from core.storage_postgres import PostgresStorage

logger = logging.getLogger("dashboard.utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)


# -------------------------
# Conexión a Storage (singleton)
# -------------------------
_storage: Optional[PostgresStorage] = None

def storage() -> PostgresStorage:
    """
    Devuelve instancia única de PostgresStorage (singleton).
    """
    global _storage
    if _storage is None:
        try:
            _storage = PostgresStorage()
            _storage.init_db()
            logger.info("Storage inicializado correctamente")
        except Exception as e:
            logger.exception("Error inicializando PostgresStorage: %s", e)
            raise
    return _storage


# -------------------------
# Assets desde CSV / DB
# -------------------------
@st.cache_data
def load_assets_from_cache(csv_path: str = "data/assets.csv") -> List[str]:
    """
    Carga lista de assets desde CSV (mantiene compatibilidad con código original).
    """
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "symbol" in df.columns:
                return sorted(df["symbol"].dropna().astype(str).tolist())
        logger.warning("Archivo de assets no encontrado o sin columna 'symbol'")
        return []
    except Exception:
        logger.exception("Error al cargar assets desde CSV")
        return []


def get_all_assets_from_db() -> List[str]:
    """
    Devuelve lista de assets únicos presentes en la tabla candles.
    """
    try:
        s = storage()
        return sorted(s.get_all_assets() or [])
    except Exception:
        logger.exception("Error obteniendo assets desde DB")
        return []


# -------------------------
# Watchlist de usuario
# -------------------------
WATCHLIST_CSV = "data/user_watchlist.csv"

def load_user_watchlist_csv(path: str = WATCHLIST_CSV) -> List[str]:
    """
    Lee la watchlist desde CSV. Crea archivo vacío si no existe.
    """
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df["symbol"].dropna().astype(str).tolist()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame({"symbol": []}).to_csv(path, index=False)
        return []
    except Exception:
        logger.exception("Error cargando watchlist")
        return []


def save_user_watchlist_csv(symbols: List[str], path: str = WATCHLIST_CSV):
    """
    Guarda la watchlist del usuario en CSV.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame({"symbol": symbols}).to_csv(path, index=False)
        logger.info("Watchlist guardada correctamente en %s", path)
    except Exception:
        logger.exception("Error guardando watchlist")


# -------------------------
# Cache de velas
# -------------------------
@lru_cache(maxsize=64)
def get_cached_candles(asset: str, interval: str, limit: int = 1000, ascending: bool = False) -> pd.DataFrame:
    """
    Devuelve velas desde DB con cache en memoria.
    """
    try:
        s = storage()
        df = s.load_candles(asset, interval, limit=limit, ascending=ascending)
        return df
    except Exception:
        logger.exception("Error cargando velas en cache")
        return pd.DataFrame()


# -------------------------
# UI Helpers
# -------------------------
def render_scrollable_list(items: List[str], title: str = "Lista", height: int = 300):
    """
    Renderiza lista scrollable en Streamlit (ideal para assets/backfills grandes).
    """
    if not items:
        st.info(f"No hay elementos en {title}")
        return
    with st.container():
        st.markdown(f"#### {title}")
        st.markdown(
            f"""
            <div style="height:{height}px;overflow:auto;border:1px solid #ddd;border-radius:8px;padding:5px;">
            {"<br>".join(items)}
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------------
# Aliases Legacy (para compatibilidad)
# -------------------------
get_assets_from_csv = load_assets_from_cache
