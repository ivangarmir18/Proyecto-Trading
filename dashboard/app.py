"""
dashboard/app_full.py
=====================

Single-file Streamlit dashboard "Trading Project" — todo-en-uno.

Objetivo:
 - Dashboard profesional y robusto para gestionar la WATCHLIST, BACKFILL, SCORES y revisión de ASSETS.
 - Manejo de errores extenso: si la capa de storage no provee una función, el UI sigue funcionando y muestra mensajes claros.
 - Integraciones opcionales: AI (OpenAI) si se proporciona OPENAI_API_KEY en el entorno.
 - Funcionalidades:
    * Añadir / eliminar símbolos en watchlist
    * Filtrado flexible por cualquier columna / atributo
    * Ordenación por cualquier columna
    * Backfill (crear peticiones en DB) por selección o bulk
    * Ver peticiones pendientes (y marcarlas processed)
    * Explorar scores (filtrar por score, interval, model, asset)
    * Visualizar candles (últimas N velas) y calcular Targets/Stops por ATR (14)
    * Export CSV/download
    * Simple password gate con DASHBOARD_PASSWORD (opcional)
 - Requisitos: streamlit, pandas, numpy. Recomendado: psycopg2 / storage_postgres en core.

Uso:
  streamlit run dashboard/app_full.py

Nota importante:
  Este archivo pretende ser autónomo dentro de tu repo. Espera que exista un módulo core.storage_postgres
  con una función make_storage_from_env() o al menos una clase PostgresStorage con métodos usados abajo.
  Si tu storage tiene nombres distintos, el dashboard falla de forma controlada y te indicará qué método falta.
"""
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

import os
import traceback
import json
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from dashboard.utils import load_assets_from_cache, load_user_watchlist_csv, save_user_watchlist_csv


# Optional AI integration
OPENAI_AVAILABLE = False
if os.getenv("OPENAI_API_KEY"):
    try:
        import openai

        openai.api_key = os.getenv("OPENAI_API_KEY")
        OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False

# Attempt to import factory for storage
make_storage_from_env = None
PostgresStorage = None
make_storage_from_env = None
PostgresStorage = None
try:
    from core.storage_postgres import make_storage_from_env, PostgresStorage  # type: ignore
except Exception as e:
    # mostrar la traza en Streamlit y en logs para depuración
    import traceback, logging
    logging.getLogger(__name__).exception("Error importando core.storage_postgres")
    try:
        import streamlit as st
        st.error("Error importando core.storage_postgres. Mira el log para más detalles.")
        st.exception(traceback.format_exc())
    except Exception:
        # si no funciona Streamlit (p.ej. en proceso worker), no rompemos el arranque
        pass
    make_storage_from_env = None
    PostgresStorage = None

# ---------------------------
# Utility wrappers & helpers
# ---------------------------

def safe_call(storage: Any, fn_name: str, *args, default=None, **kwargs):
    """
    Llamada segura a métodos de storage. Si no existe o lanza excepción, devuelve default.
    Además retorna (ok, result, error) tuple when verbose needed.
    """
    try:
        if storage is None:
            return default
        fn = getattr(storage, fn_name, None)
        if not fn:
            return default
        return fn(*args, **kwargs)
    except Exception as e:
        # Log to Streamlit if in UI context, otherwise ignore
        try:
            st.error(f"Error calling storage.{fn_name}: {e}")
        except Exception:
            pass
        return default


def make_storage():
    """
    Intenta crear storage en varios modos:
      1) make_storage_from_env() si existe
      2) PostgresStorage() desde env vars (si clase exportada)
      3) raise RuntimeError con mensaje útil
    """
    # 1) factory
    if make_storage_from_env is not None:
        try:
            s = make_storage_from_env()
            return s
        except Exception as e:
            # Dejar que fallback intente
            st.warning(f"make_storage_from_env() falló: {e}")
    # 2) direct class
    if PostgresStorage is not None:
        try:
            # Intentar constructor sin args — PostgresStorage debe leer env
            s = PostgresStorage()
            return s
        except Exception as e:
            st.warning(f"PostgresStorage() fallo: {e}")
    # 3) no storage posible
    raise RuntimeError(
        "No se pudo crear storage automáticamente. Asegúrate de que `core.storage_postgres` exporta "
        "make_storage_from_env() o PostgresStorage y que DATABASE_URL está configurada."
    )


def df_from_list_of_dicts(maybe_list) -> pd.DataFrame:
    if maybe_list is None:
        return pd.DataFrame()
    if isinstance(maybe_list, pd.DataFrame):
        return maybe_list
    try:
        return pd.DataFrame(maybe_list)
    except Exception:
        return pd.DataFrame()


def safe_to_csv_download(df: pd.DataFrame, filename: str):
    """Return (label, data) for st.download_button usage"""
    try:
        csv = df.to_csv(index=False)
        return csv
    except Exception:
        return df.to_json(orient="records")


def calculate_atr_from_df(df_candles: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calcula ATR simple (True Range rolling mean).
    df_candles must tener columnas: high, low, close (puede usar floats/strings).
    Devuelve ATR del último punto o None.
    """
    try:
        df = df_candles.copy()
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["high", "low", "close"])
        if df.empty:
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean().iloc[-1]
        return float(atr)
    except Exception:
        return None


# ---------------------------
# Streamlit app layout start
# ---------------------------
st.set_page_config(page_title="Trading Project", layout="wide")
st.title("Trading Project — Dashboard")

# Authentication simple
PASSWORD = os.getenv("DASHBOARD_PASSWORD") or None
if PASSWORD:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        st.write("### Introduce la contraseña para acceder al dashboard")
        pwd = st.text_input("Contraseña", type="password")
        if st.button("Entrar"):
            if pwd == PASSWORD:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta")
        st.stop()

# Create / cache storage
@st.cache_resource
def get_storage_cached():
    try:
        return make_storage()
    except Exception as e:
        # we want a clear message in UI, but propagate for pages to handle
        raise

try:
    storage = get_storage_cached()
except Exception as e:
    st.error(f"No se pudo inicializar el storage: {e}")
    st.stop()

# Sidebar navigation and global controls
with st.sidebar:
    st.header("Navegación")
    page = st.radio("Ir a:", ["Watchlist", "Backfill Requests", "Scores Explorer", "Asset Viewer", "Health & Logs"])
    st.markdown("---")
    st.write("Acciones globales")
    if st.button("Forzar refresco (UI)"):
        st.rerun()
    if OPENAI_AVAILABLE:
        st.info("AI disponible: OpenAI API key detectada.")
    else:
        st.caption("AI deshabilitada (poner OPENAI_API_KEY para activar)")

# -------------
# Page: Watchlist
# -------------
def page_watchlist(storage):
    st.header("Watchlist — Gestión avanzada")

    # Left column: add symbol
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Añadir / Editar símbolo")
        with st.form("add_symbol", clear_on_submit=True):
            asset = st.text_input("Símbolo (ej. BTCUSDT)").strip().upper()
            asset_type = st.selectbox("Tipo", ["crypto", "stock", "forex", "other"], index=0)
            interval = st.text_input("Intervalo (ej. 1h, 15m)", value="1h")
            added_by = st.text_input("Añadido por", value="dashboard")
            metadata_text = st.text_area("Metadata JSON (opcional)", value="{}")
            try:
                metadata = json.loads(metadata_text) if metadata_text.strip() else {}
            except Exception as e:
                st.warning("Metadata no es JSON válido — se usará {}")
                metadata = {}
            submitted = st.form_submit_button("Guardar en Watchlist")
            if submitted:
                if not asset:
                    st.error("El símbolo no puede estar vacío")
                else:
                    try:
                        res = safe_call(storage, "add_watchlist_symbol", asset, asset_type, interval, added_by, metadata, default=None)
                        if res is None:
                            # Try with named args (compatibility)
                            res = safe_call(storage, "add_watchlist_symbol", asset=asset, asset_type=asset_type, interval=interval, added_by=added_by, metadata=metadata, default=None)
                        if res is None:
                            st.error("Storage no implementa add_watchlist_symbol() o falló. Revisa logs.")
                        else:
                            st.success(f"{asset} añadido/actualizado en watchlist.")
                            st.rerun()
                    except Exception as e:
                        st.exception(e)
                        
    # -----------------------------
    # Catálogo desde CSV (data/cache) y añadir seleccionados
    # -----------------------------
    st.markdown("---")
    st.subheader("Añadir desde catálogo CSV (data/cache)")
    try:
        assets_df = load_assets_from_cache()
    except Exception:
        assets_df = None
    
    if assets_df is None or assets_df.empty:
        st.info("No se han encontrado CSV en data/cache/ — si tienes candles en disco, revisa su ruta.")
        assets_list = []
    else:
        st.write(f"{len(assets_df)} activos detectados en data/cache")
        tipo_sel = st.selectbox("Tipo", options=["all", "crypto", "stock", "other"], index=0)
        if tipo_sel != "all":
            shown_df = assets_df[assets_df["asset_type"] == tipo_sel]
        else:
            shown_df = assets_df
        assets_list = shown_df["asset"].astype(str).tolist()
        selected_assets = st.multiselect("Selecciona activos para añadir a tu watchlist", options=assets_list, default=[])
    
    if st.button("Añadir seleccionados"):
        if not assets_list or not selected_assets:
            st.warning("Selecciona primero los activos que quieres añadir.")
        else:
            user_id = st.session_state.get("user_id", "default")
            successes, failures = [], []
            for a in selected_assets:
                # Intentar añadir vía storage si está disponible
                res = safe_call(storage, "add_watchlist_symbol", a, "auto", "1h", "dashboard", default=None) \
                      or safe_call(storage, "add_watchlist_symbol", asset=a, asset_type="auto", interval="1h", added_by="dashboard", default=None)
                if res is None:
                    # Fallback CSV
                    try:
                        existing = [r["asset"] for r in load_user_watchlist_csv(user_id)]
                        new_set = sorted(set(existing) | set([a.upper()]))
                        save_user_watchlist_csv(new_set, user_id=user_id)
                        successes.append(a)
                    except Exception as e:
                        failures.append((a, str(e)))
                else:
                    successes.append(a)
            if successes:
                st.success(f"Añadidos: {', '.join(successes)}")
                st.experimental_rerun()
            if failures:
                st.error(f"Fallos: {failures}")

    with right:
        st.subheader("Watchlist — Tabla y filtros")

        # Load watchlist (try get_watchlist or list_watchlist)
        # Load watchlist (DB storage first, then CSV fallback)
    raw_wl = safe_call(storage, "get_watchlist", default=None) or safe_call(storage, "list_watchlist", default=None)
    if not raw_wl:
        user_id = st.session_state.get("user_id", "default")
        try:
            raw_wl = load_user_watchlist_csv(user_id)
        except Exception:
            raw_wl = None
    
    df = df_from_list_of_dicts(raw_wl)
    if df.empty:
        st.info("La watchlist está vacía. Añade símbolos desde el catálogo a la izquierda o usa la UI para crear tu selección personal.")
        return

        # Ensure consistent columns
        if "asset" not in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"symbol": "asset"})
        if "created_at" in df.columns:
            # normalize ts
            try:
                df["created_at"] = pd.to_datetime(df["created_at"])
            except Exception:
                pass

        # Optional: attach latest score if storage supports load_latest_scores or load_scores
        scores_attached = False
        try:
            # Try storage.load_latest_scores() returning a dataframe/list/dict
            df_scores = safe_call(storage, "load_latest_scores", default=None)
            if df_scores is None:
                df_scores = safe_call(storage, "load_scores", default=None)
            if df_scores is not None:
                df_scores = df_from_list_of_dicts(df_scores)
                if not df_scores.empty and "asset" in df_scores.columns:
                    # We expect df_scores to have asset and score (or JSON score)
                    if "score" in df_scores.columns and df_scores["score"].apply(lambda s: isinstance(s, (int, float))).any():
                        # prefer numeric score
                        score_col = "score"
                    else:
                        # try to extract numeric from JSON-like
                        df_scores["score_extracted"] = df_scores["score"].apply(lambda s: (s.get("score") if isinstance(s, dict) and "score" in s else np.nan))
                        score_col = "score_extracted"
                    # Merge by asset, keep last
                    df_scores = df_scores.sort_values(by=["asset"]).drop_duplicates(subset=["asset"], keep="last")
                    df = df.merge(df_scores[["asset", score_col]], on="asset", how="left")
                    df = df.rename(columns={score_col: "score"})
                    scores_attached = True
        except Exception:
            scores_attached = False

        # Filters UI (dynamic)
        st.markdown("**Filtros**")
        c1, c2, c3, c4 = st.columns([2,2,1,1])
        with c1:
            search = st.text_input("Buscar (asset / tipo / metadata)", value="").strip().upper()
        with c2:
            interval_filter = st.text_input("Intervalo (ej. 1h) — vacío = todos", value="").strip()
        with c3:
            min_score = st.number_input("Score mínimo (si disponible)", value=float(0.0))
        with c4:
            col_options = list(df.columns)
            order_by = st.selectbox("Ordenar por", options=col_options, index=0)
            asc = st.checkbox("Ascendente", value=False)

        # Apply filters
        df_filtered = df.copy()
        if search:
            def row_matches_search(r):
                s = str(r.get("asset", "")).upper()
                if search in s:
                    return True
                t = str(r.get("asset_type", "")).upper()
                if search in t:
                    return True
                try:
                    meta = r.get("metadata", "")
                    if isinstance(meta, dict):
                        if search in json.dumps(meta).upper():
                            return True
                    elif search in str(meta).upper():
                        return True
                except Exception:
                    pass
                return False
            df_filtered = df_filtered[df_filtered.apply(row_matches_search, axis=1)]
        if interval_filter:
            df_filtered = df_filtered[df_filtered["interval"].astype(str).str.contains(interval_filter, na=False)]
        if scores_attached and "score" in df_filtered.columns:
            df_filtered["score_num"] = pd.to_numeric(df_filtered["score"], errors="coerce").fillna(0.0)
            df_filtered = df_filtered[df_filtered["score_num"] >= float(min_score)]

        # Ordering
        if order_by in df_filtered.columns:
            try:
                df_filtered = df_filtered.sort_values(by=order_by, ascending=asc)
            except Exception:
                df_filtered = df_filtered.sort_values(by=order_by, ascending=asc, key=lambda s: s.astype(str))

        st.write(f"Mostrando {len(df_filtered)} símbolos")
        # Multi-select assets for actions
        assets_list = list(df_filtered["asset"].astype(str).tolist())
        selected = st.multiselect("Selecciona símbolos para acciones", options=assets_list)

        # Data display (use data_editor if available)
        try:
            st.dataframe(df_filtered.reset_index(drop=True))
        except Exception:
            st.write(df_filtered)

        # Action buttons
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            if st.button("Eliminar seleccionados"):
                if not selected:
                    st.warning("Selecciona al menos un símbolo primero.")
                else:
                    deleted = 0
                    for sym in selected:
                        ok = safe_call(storage, "remove_watchlist_symbol", sym, default=False)
                        if ok:
                            deleted += 1
                    st.success(f"Eliminados: {deleted}")
                    st.rerun()
        with a2:
            if st.button("Pedir backfill (seleccionados)"):
                if not selected:
                    st.warning("Selecciona al menos un símbolo.")
                else:
                    created = 0
                    for sym in selected:
                        try:
                            # get interval from df
                            row = df_filtered[df_filtered["asset"] == sym].iloc[0]
                            interval = row.get("interval") or "1h"
                        except Exception:
                            interval = "1h"
                        res = safe_call(storage, "add_backfill_request", sym, interval, "dashboard_manual", {}, default=None)
                        if res is None:
                            # try named args
                            res = safe_call(storage, "add_backfill_request", asset=sym, interval=interval, requested_by="dashboard_manual", params={}, default=None)
                        if res:
                            created += 1
                    st.success(f"Peticiones creadas: {created}")
        with a3:
            if st.button("Pedir backfill (todos filtrados)"):
                created = 0
                for _, r in df_filtered.iterrows():
                    sym = r.get("asset")
                    interval = r.get("interval") or "1h"
                    res = safe_call(storage, "add_backfill_request", sym, interval, "dashboard_bulk", {}, default=None)
                    if res:
                        created += 1
                st.success(f"Peticiones creadas: {created}")
        with a4:
            if st.button("Exportar filtrado (CSV)"):
                csv = safe_to_csv_download(df_filtered)
                st.download_button("Descargar CSV", csv, file_name=f"watchlist_filtered_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")

        # Detail & ATR suggestions
        st.markdown("---")
        st.subheader("Detalle y sugerencias (Targets / Stops por ATR)")
        if selected:
            sel = selected[0]
            st.write(f"Detalle: **{sel}**")
            try:
                row = df_filtered[df_filtered["asset"] == sel].iloc[0].to_dict()
                st.json(row)
            except Exception:
                st.write("No se pudo mostrar detalle del registro.")

            # Show recent candles if available and compute ATR
            if hasattr(storage, "load_candles"):
                interval_for_sel = row.get("interval") or "1h"
                st.write(f"Cargando velas para {sel} ({interval_for_sel}) — si están disponibles en storage")
                try:
                    df_c = safe_call(storage, "load_candles", sel, interval_for_sel, None, None, default=None)
                    df_c = df_from_list_of_dicts(df_c)
                    if df_c is None or df_c.empty:
                        st.info("No hay velas históricas disponibles en storage para este asset/interval.")
                    else:
                        # Normalize
                        if "ts" in df_c.columns:
                            try:
                                df_c["timestamp"] = pd.to_datetime(df_c["ts"], unit="s")
                            except Exception:
                                pass
                        st.write("Últimas 200 velas (si hay):")
                        st.dataframe(df_c.tail(200).reset_index(drop=True))
                        atr = calculate_atr_from_df(df_c, period=14)
                        try:
                            last_close = float(df_c.sort_values("ts").iloc[-1]["close"])
                        except Exception:
                            last_close = None
                        if atr is not None and last_close is not None:
                            k = st.number_input("Multiplicador ATR (k)", value=1.5, step=0.1)
                            suggested_stop = last_close - k * atr
                            suggested_target = last_close + k * atr
                            st.metric("Último cierre", f"{last_close:.6f}")
                            st.metric("ATR(14)", f"{atr:.6f}")
                            st.metric("Sugerencia Stop", f"{suggested_stop:.6f}")
                            st.metric("Sugerencia Target", f"{suggested_target:.6f}")
                            if st.button("Guardar target/stop en metadata"):
                                try:
                                    meta = row.get("metadata") or {}
                                    if isinstance(meta, str):
                                        try:
                                            meta = json.loads(meta)
                                        except Exception:
                                            meta = {}
                                    meta.update({
                                        "suggested_stop": float(suggested_stop),
                                        "suggested_target": float(suggested_target),
                                        "atr": float(atr),
                                        "atr_k": float(k),
                                        "saved_at": datetime.utcnow().isoformat()
                                    })
                                    res = safe_call(storage, "add_watchlist_symbol", sel, row.get("asset_type"), row.get("interval"), "dashboard_targets", meta, default=None)
                                    if res is None:
                                        # fallback named
                                        res = safe_call(storage, "add_watchlist_symbol", asset=sel, asset_type=row.get("asset_type"), interval=row.get("interval"), added_by="dashboard_targets", metadata=meta, default=None)
                                    if res:
                                        st.success("Target/stop guardados en metadata de watchlist.")
                                    else:
                                        st.error("No se pudo guardar metadata: storage no implementa add_watchlist_symbol correctamente.")
                                except Exception as e:
                                    st.exception(e)
                        else:
                            st.info("No se pudo calcular ATR/último cierre con los datos disponibles.")
                except Exception as e:
                    st.exception(e)
            else:
                st.info("Storage no implementa load_candles(). No es posible calcular targets/stops automáticamente.")

        else:
            st.info("Selecciona un símbolo (de la lista) para ver detalle y sugerencias ATR.")

# -------------
# Page: Backfill Requests
# -------------
def page_backfill_requests(storage):
    st.header("Backfill Requests — Cola y acciones")
    try:
        reqs = safe_call(storage, "fetch_pending_backfill_requests", 500, default=[])
        df_reqs = df_from_list_of_dicts(reqs)
        if df_reqs.empty:
            st.info("No hay peticiones pendientes.")
            return
        st.write(f"Peticiones pendientes: {len(df_reqs)}")
        st.dataframe(df_reqs)

        # Select some requests
        sel_ids = st.multiselect("Selecciona peticiones (id) para acciones", options=list(df_reqs["id"].tolist()))

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Marcar seleccionadas como processed"):
                if not sel_ids:
                    st.warning("Selecciona al menos una petición.")
                else:
                    updated = 0
                    for rid in sel_ids:
                        ok = safe_call(storage, "update_backfill_request_status", rid, "processed", default=False)
                        if ok:
                            updated += 1
                    st.success(f"Actualizadas: {updated}")
                    st.rerun()
        with col2:
            if st.button("Reintentar seleccionadas (recrear request)"):
                recreated = 0
                for rid in sel_ids:
                    try:
                        row = df_reqs[df_reqs["id"] == rid].iloc[0]
                        asset = row.get("asset")
                        interval = row.get("interval") or "1h"
                        res = safe_call(storage, "add_backfill_request", asset, interval, "dashboard_retry", {}, default=None)
                        if res:
                            recreated += 1
                    except Exception:
                        pass
                st.success(f"Recreadas: {recreated}")
        with col3:
            if st.button("Exportar lista CSV"):
                csv = safe_to_csv_download(df_reqs)
                st.download_button("Descargar CSV", csv, file_name=f"backfill_requests_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")

    except Exception as e:
        st.exception(e)

# -------------
# Page: Scores Explorer
# -------------
def page_scores_explorer(storage):
    st.header("Scores Explorer — filtra, ordena, exporta")

    # Try various methods to obtain scores
    df_scores = None
    try:
        df_scores = safe_call(storage, "load_scores", default=None)
        if df_scores is None:
            df_scores = safe_call(storage, "load_latest_scores", default=None)
    except Exception:
        df_scores = None

    df_scores = df_from_list_of_dicts(df_scores)
    if df_scores.empty:
        st.info("No se pudieron cargar scores desde storage. Asegúrate de que storage implementa load_scores().")
        return

    # Normalize common columns
    if "ts" in df_scores.columns:
        try:
            df_scores["timestamp"] = pd.to_datetime(df_scores["ts"], unit="s")
        except Exception:
            try:
                df_scores["timestamp"] = pd.to_datetime(df_scores["ts"])
            except Exception:
                pass

    st.write(f"Records: {len(df_scores)}")
    # Filters
    c1, c2, c3, c4 = st.columns([2,2,1,1])
    with c1:
        asset_q = st.text_input("Asset contains")
    with c2:
        interval_q = st.text_input("Interval equals")
    with c3:
        model_q = st.text_input("Model id equals")
    with c4:
        min_score = st.number_input("Score >= (if numeric)", value=0.0)

    df_filtered = df_scores.copy()
    if asset_q:
        if "asset" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["asset"].astype(str).str.contains(asset_q, case=False)]
    if interval_q and "interval" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["interval"] == interval_q]
    if model_q and "model_id" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["model_id"].astype(str) == model_q]
    if "score" in df_filtered.columns:
        try:
            df_filtered["score_num"] = pd.to_numeric(df_filtered["score"], errors="coerce").fillna(0.0)
            df_filtered = df_filtered[df_filtered["score_num"] >= float(min_score)]
        except Exception:
            pass

    st.dataframe(df_filtered)
    if st.button("Exportar Scores CSV"):
        csv = safe_to_csv_download(df_filtered)
        st.download_button("Descargar CSV", csv, file_name=f"scores_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")

# -------------
# Page: Asset Viewer
# -------------
def page_asset_viewer(storage):
    st.header("Asset Viewer — ver velas y pistas")
    # Choose asset from watchlist if available
    raw_wl = safe_call(storage, "get_watchlist", default=None) or safe_call(storage, "list_watchlist", default=None)
    df_wl = df_from_list_of_dicts(raw_wl)
    assets = []
    if not df_wl.empty and "asset" in df_wl.columns:
        assets = df_wl["asset"].astype(str).tolist()

    asset = st.selectbox("Selecciona asset", options=([""] + assets))
    interval = st.selectbox("Intervalo", options=["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    n_candles = st.number_input("Velas a mostrar", min_value=10, max_value=10000, value=500, step=10)

    if not asset:
        st.info("Selecciona un asset para ver velas.")
        return

    if not hasattr(storage, "load_candles"):
        st.error("Storage no implementa load_candles(asset, interval, start_ts, end_ts).")
        return

    try:
        df_c = safe_call(storage, "load_candles", asset, interval, None, None, default=None)
        df_c = df_from_list_of_dicts(df_c)
        if df_c.empty:
            st.info("No hay velas disponibles en storage para este asset/interval.")
            return
        # Try to normalize
        if "ts" in df_c.columns:
            try:
                df_c["timestamp"] = pd.to_datetime(df_c["ts"], unit="s")
            except Exception:
                try:
                    df_c["timestamp"] = pd.to_datetime(df_c["ts"])
                except Exception:
                    pass
        df_small = df_c.sort_values("ts").tail(int(n_candles)).reset_index(drop=True)
        st.line_chart(df_small.set_index("timestamp")["close"])
        st.dataframe(df_small.tail(200))
    except Exception as e:
        st.exception(e)

# -------------
# Page: Health & Logs
# -------------
def page_health_logs(storage):
    st.header("Health & Logs")
    st.write("Estado del storage / Base de datos")

    try:
        health = safe_call(storage, "health", default=None)
        if health is None:
            st.warning("Storage no implementa health(). Intentando operación simple...")
            # Try a simple action: fetch one watchlist row via get_watchlist
            try:
                wl = safe_call(storage, "get_watchlist", default=None) or safe_call(storage, "list_watchlist", default=None)
                if wl is None:
                    st.error("No fue posible realizar operaciones básicas con el storage.")
                else:
                    st.success("Storage responde (get_watchlist ok).")
                    st.write(f"Watchlist length (sample): {len(wl)}")
            except Exception as e:
                st.exception(e)
        else:
            st.json(health)
    except Exception as e:
        st.exception(e)

    st.markdown("---")
    st.write("Acciones administrativas rápidas")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Inicializar tablas watchlist/backfill (si storage lo soporta)"):
            try:
                # call a helper that might create tables: init_db or similar if exists
                res = safe_call(storage, "init_db", default=None)
                if res is not None:
                    st.success("init_db() ejecutado (ver logs del storage).")
                else:
                    st.info("storage no implementa init_db(). Si usas Postgres, revisa core/storage_postgres._ensure_watchlist_tables().")
            except Exception as e:
                st.exception(e)
    with col2:
        if st.button("Test DB query (SELECT 1)"):
            try:
                res = safe_call(storage, "health", default=None)
                if res is not None:
                    st.success("health() OK")
                else:
                    st.warning("health() no disponible en storage, intenta ejecutar un script de comprobación en el server.")
            except Exception as e:
                st.exception(e)

# ---------------------------
# AI helper (opcional)
# ---------------------------
def ai_explain_score(asset: str, score_obj: Any) -> Optional[str]:
    """
    Si OPENAI_API_KEY está presente, enviar prompt para generar explicación breve.
    Devuelve texto o None si falla.
    """
    if not OPENAI_AVAILABLE:
        return None
    try:
        prompt = (
            f"Explica brevemente el score para el asset {asset}.\n"
            f"Score raw: {json.dumps(score_obj, default=str)}\n"
            "Dame un resumen corto en lenguaje claro con posibles niveles objetivo basados en el score."
        )
        resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=200, temperature=0.0)
        txt = resp.choices[0].text.strip()
        return txt
    except Exception:
        return None

# ---------------------------
# Dispatch pages
# ---------------------------
try:
    if page == "Watchlist":
        page_watchlist(storage)
    elif page == "Backfill Requests":
        page_backfill_requests(storage)
    elif page == "Scores Explorer":
        page_scores_explorer(storage)
    elif page == "Asset Viewer":
        page_asset_viewer(storage)
    elif page == "Health & Logs":
        page_health_logs(storage)
    else:
        st.info("Página no encontrada.")
except Exception as e:
    st.exception(e)
    st.stop()
