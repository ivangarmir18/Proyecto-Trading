"""
Dashboard completo y exhaustivo para Proyecto-Trading (VERSIÓN CORREGIDA)
=================================

Esta versión corrige la pestaña Watchlist: ahora lee la watchlist desde archivos
locales del proyecto (p. ej. data/config, data/actions.csv, data/cryptos.csv,
data/cache/assets_*.json) y ofrece sincronización con storage (DB).

Pega este archivo en dashboard/app_full.py (sobrescribe) y reinicia Streamlit.
"""
from __future__ import annotations

import os
import json
import logging
import tempfile
import glob
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime

import streamlit as st
import pandas as pd

# Intentar importar módulos core (cada import es opcional — el UI seguirá funcionando sin ellos)
_core_imports = {}
try:
    from core.storage_postgres import make_storage_from_env, PostgresStorage
    _core_imports['storage'] = True
except Exception:
    PostgresStorage = None
    make_storage_from_env = None
    _core_imports['storage'] = False

try:
    from core.orchestrator import Orchestrator
    _core_imports['orchestrator'] = True
except Exception:
    Orchestrator = None
    _core_imports['orchestrator'] = False

try:
    import core.fetch as fetch_module
    _core_imports['fetch'] = True
except Exception:
    fetch_module = None
    _core_imports['fetch'] = False

try:
    import core.ai_train as ai_train_module
    _core_imports['ai_train'] = True
except Exception:
    ai_train_module = None
    _core_imports['ai_train'] = False

try:
    import core.ai_interference as ai_interf_module
    _core_imports['ai_interference'] = True
except Exception:
    ai_interf_module = None
    _core_imports['ai_interference'] = False

try:
    import core.score as score_module
    _core_imports['score'] = True
except Exception:
    score_module = None
    _core_imports['score'] = False

try:
    import core.backtest as backtest_module
    _core_imports['backtest'] = True
except Exception:
    backtest_module = None
    _core_imports['backtest'] = False

# logger
logger = logging.getLogger('dashboard.app_full')
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

st.set_page_config(page_title='Trading Intelligence — Full Dashboard', layout='wide')

# ---------------------------
# Helpers: storage factory
# ---------------------------
@st.cache_resource
def get_storage() -> Optional[PostgresStorage]:
    """
    Devuelve un singleton de storage (cacheado por Streamlit).
    No recibe argumentos (evita problemas de hashing).
    """
    if make_storage_from_env is None:
        st.warning('Fábrica de storage no disponible (make_storage_from_env). Algunas funciones estarán deshabilitadas.')
        return None
    try:
        s = make_storage_from_env()
        try:
            # Init DB es idempotente; lo intentamos para facilitar despliegues.
            s.init_db()
        except Exception:
            logger.exception('init_db fallo (continuando)')
        return s
    except Exception as e:
        logger.exception('No se pudo crear storage: %s', e)
        st.error(f'No se pudo conectar a la DB: {e}')
        return None

# IMPORTANT: Streamlit attempts to hash/cache function arguments.
# To avoid UnhashableParamError when passing objects like PostgresStorage or Orchestrator,
# we name parameters with a leading underscore so Streamlit will not hash them.
@st.cache_resource
def get_orchestrator(_storage: Optional[PostgresStorage] = None, _config: Optional[Dict[str, Any]] = None) -> Optional[Orchestrator]:
    """
    Devuelve un Orchestrator instanciado y cacheado por Streamlit.

    Nota: parámetros con _ leading underscore NO son hasheados por Streamlit,
    lo que evita errores al pasar objetos no-hasheables (p. ej. conexiones DB).
    """
    if Orchestrator is None:
        return None
    try:
        storage = _storage
        # si no se pasó storage explícitamente, intentar construir el singleton
        if storage is None:
            try:
                storage = get_storage()
            except Exception:
                storage = None
        config = _config or {}
        return Orchestrator(storage, config=config)
    except Exception:
        logger.exception('No se pudo crear orchestrator')
        return None

# ---------------------------
# Utilities específicas para watchlist desde archivos
# ---------------------------

def _read_json_file(path: Path):
    try:
        content = path.read_text(encoding='utf-8')
        data = json.loads(content)
        return data
    except Exception:
        return None

def _read_csv_file(path: Path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

def load_watchlist_from_repo_files(base_dirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Busca y carga la watchlist desde varios ficheros comunes en el repo:
      - data/actions.csv
      - data/cryptos.csv
      - data/watchlist.csv
      - data/config.json
      - data/cache/assets_*.json
      - data/config/*.json
      - data/actions/*.csv
    Devuelve lista de dicts: [{ "asset": "BTCUSDT", "meta": {...} }, ...]
    """
    base_dirs = base_dirs or ['data', 'data/config', 'data/actions']
    candidates = []
    # explicit candidate filenames
    explicit = [
        "data/actions.csv",
        "data/cryptos.csv",
        "data/watchlist.csv",
        "data/config.json",
        "data/watchlist.json",
    ]
    for p in explicit:
        candidates.append(Path(p))
    # glob patterns
    candidates += [Path(p) for p in glob.glob("data/cache/assets_*.json")]
    candidates += [Path(p) for p in glob.glob("data/config/*.json")]
    candidates += [Path(p) for p in glob.glob("data/actions/*.csv")]
    candidates += [Path(p) for p in glob.glob("data/*.csv")]

    seen_assets = set()
    out: List[Dict[str, Any]] = []

    for p in candidates:
        if not p.exists():
            continue
        # JSON handling
        if p.suffix.lower() in ('.json',):
            data = _read_json_file(p)
            if data is None:
                continue
            # accept various shapes
            if isinstance(data, dict):
                # maybe {"watchlist": [...]} or {"assets": [...]}
                possible = []
                for key in ('watchlist', 'assets', 'data'):
                    if key in data and isinstance(data[key], list):
                        possible = data[key]
                        break
                if not possible:
                    # if dict with string keys mapping to meta, convert
                    # e.g. {"BTCUSDT": {...}, "ETHUSDT": {...}}
                    arr = []
                    for k, v in data.items():
                        if isinstance(k, str):
                            arr.append({"asset": k, "meta": v})
                    possible = arr
                for item in possible:
                    if isinstance(item, str):
                        a = item.strip()
                        if a and a not in seen_assets:
                            seen_assets.add(a)
                            out.append({"asset": a, "meta": {"source": str(p)}})
                    elif isinstance(item, dict) and item.get("asset"):
                        a = str(item.get("asset")).strip()
                        if a and a not in seen_assets:
                            seen_assets.add(a)
                            out.append({"asset": a, "meta": item.get("meta") or {"source": str(p)}})
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        a = item.strip()
                        if a and a not in seen_assets:
                            seen_assets.add(a)
                            out.append({"asset": a, "meta": {"source": str(p)}})
                    elif isinstance(item, dict) and item.get("asset"):
                        a = str(item.get("asset")).strip()
                        if a and a not in seen_assets:
                            seen_assets.add(a)
                            out.append({"asset": a, "meta": item.get("meta") or {"source": str(p)}})
            continue

        # CSV handling
        if p.suffix.lower() in ('.csv',):
            df = _read_csv_file(p)
            if df is None:
                continue
            # look for column names
            if 'asset' in df.columns:
                for v in df['asset'].dropna().astype(str).str.strip().unique():
                    if v and v not in seen_assets:
                        seen_assets.add(v)
                        # gather meta columns if exist
                        meta = {}
                        for c in df.columns:
                            if c != 'asset' and df[c].notna().any():
                                meta[c] = df.loc[df['asset'].astype(str).str.strip() == v, c].iloc[0] if not df.loc[df['asset'].astype(str).str.strip() == v, c].empty else None
                        out.append({"asset": v, "meta": {"source": str(p), **meta}})
            else:
                # fallback: assume first column contains asset codes
                try:
                    first_col = df.columns[0]
                    for v in df[first_col].dropna().astype(str).str.strip().unique():
                        if v and v not in seen_assets:
                            seen_assets.add(v)
                            out.append({"asset": v, "meta": {"source": str(p)}})
                except Exception:
                    continue
            continue

    # Final: if no files found, try to read data/cache/assets_*.json again as last resort
    if not out:
        for p in glob.glob("data/cache/assets_*.json"):
            try:
                j = _read_json_file(Path(p))
                if isinstance(j, list):
                    for item in j:
                        if isinstance(item, dict) and item.get('asset'):
                            a = str(item.get('asset')).strip()
                            if a and a not in seen_assets:
                                seen_assets.add(a)
                                out.append({"asset": a, "meta": {"source": p}})
                        elif isinstance(item, str):
                            a = item.strip()
                            if a and a not in seen_assets:
                                seen_assets.add(a)
                                out.append({"asset": a, "meta": {"source": p}})
            except Exception:
                continue

    return out

def sync_watchlist_to_storage(storage: Any, watchlist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sincroniza la lista `watchlist` (lista de dicts {asset,meta}) con storage.
    Añade cada símbolo con storage.add_watchlist_symbol si existe.
    Devuelve resumen.
    """
    results = {"added": [], "failed": []}
    if storage is None:
        return {"error": "storage not available"}
    for item in watchlist:
        asset = item.get("asset")
        meta = item.get("meta") or {}
        try:
            storage.add_watchlist_symbol(asset, asset_type=meta.get("type") if isinstance(meta, dict) else None, added_by=meta.get("added_by") if isinstance(meta, dict) else None)
            # optionally update meta if storage supports update_watchlist_meta
            if hasattr(storage, "update_watchlist_meta"):
                try:
                    storage.update_watchlist_meta(asset, meta or {})
                except Exception:
                    # not critical
                    logger.exception("update_watchlist_meta failed for %s", asset)
            results["added"].append(asset)
        except Exception as e:
            results["failed"].append({"asset": asset, "error": str(e)})
    return results

# ---------------------------
# UI small components
# ---------------------------

def download_df_as_csv(df: pd.DataFrame, filename: str = 'export.csv'):
    return df.to_csv(index=False).encode('utf-8')

# JSON prettifier
def pretty_json(j):
    try:
        return json.dumps(j, indent=2, ensure_ascii=False)
    except Exception:
        return str(j)

# ---------------------------
# Tabs
# ---------------------------

def overview_tab(storage: Optional[PostgresStorage]):
    st.title('Overview')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Storage')
        st.write('Disponible' if storage else 'NO disponible')
        if storage:
            try:
                h = storage.health()
                st.json(h)
            except Exception as e:
                st.error(f'Health check fallo: {e}')
    with col2:
        st.subheader('Core modules')
        st.write({k: v for k, v in _core_imports.items()})
    with col3:
        st.subheader('Quick actions')
        if st.button('Reinicializar DB (init_db)'):
            if storage:
                try:
                    storage.init_db()
                    st.success('DB inicializada (init_db)')
                except Exception as e:
                    st.error(f'init_db falló: {e}')
            else:
                st.error('Storage no disponible')


def watchlist_tab(storage: Optional[PostgresStorage]):
    st.title('Watchlist (desde archivos del repo)')
    st.markdown('Esta watchlist **no** se importa por uploader. Se lee desde los ficheros del repositorio (p. ej. `data/actions.csv`, `data/cryptos.csv`, `data/config/*.json`, `data/cache/assets_*.json`). Si quieres sincronizar con la DB, usa el botón "Sincronizar con DB".')

    col1, col2 = st.columns([2,1])
    with col1:
        # mostrar qué archivos se encontraron y su contenido
        st.subheader('Ficheros detectados y activos')
        # localizar y mostrar las rutas encontradas
        found_files = []
        candidates = [
            Path("data/actions.csv"),
            Path("data/cryptos.csv"),
            Path("data/watchlist.csv"),
            Path("data/config.json"),
        ]
        candidates += [Path(p) for p in glob.glob("data/cache/assets_*.json")]
        candidates += [Path(p) for p in glob.glob("data/config/*.json")]
        candidates += [Path(p) for p in glob.glob("data/actions/*.csv")]
        candidates += [Path(p) for p in glob.glob("data/*.csv")]

        for p in candidates:
            if p.exists():
                found_files.append(str(p))

        if not found_files:
            st.info("No se detectaron ficheros de watchlist en rutas esperadas (data/*). Revisa tu repo.")
        else:
            st.write("Ficheros encontrados:")
            for fpath in found_files:
                st.text(f"- {fpath}")

        st.markdown("---")
        st.subheader("Watchlist combinada desde archivos")
        wl = load_watchlist_from_repo_files()
        if not wl:
            st.info("No se encontraron assets en los ficheros. Asegúrate de que existen archivos en data/ (actions.csv, cryptos.csv, data/config/*.json, data/cache/assets_*.json, etc.)")
        else:
            df = pd.DataFrame(wl)
            # mostrar asset + source (si existe)
            if 'meta' in df.columns:
                df['source'] = df['meta'].apply(lambda m: (m.get('source') if isinstance(m, dict) else str(m)) if m else '')
            st.dataframe(df[['asset','source']].rename(columns={'asset':'Asset','source':'Source'}), use_container_width=True)

        st.markdown("---")
        st.subheader("Sincronizar con DB (añadir símbolos a la watchlist en storage)")
        if st.button("Sincronizar watchlist desde archivos → DB"):
            if not storage:
                st.error("Storage no disponible — no puedo sincronizar con la base de datos.")
            else:
                if not wl:
                    st.warning("No hay assets detectados para sincronizar.")
                else:
                    with st.spinner("Sincronizando..."):
                        res = sync_watchlist_to_storage(storage, wl)
                    if res.get("error"):
                        st.error(f"Error: {res['error']}")
                    else:
                        st.success(f"Sincronizados: {len(res.get('added',[]))}. Fallos: {len(res.get('failed',[]))}")
                        if res.get('failed'):
                            st.json(res.get('failed'))

    with col2:
        st.subheader('Acciones locales (solo lectura de archivos)')
        st.markdown('Si necesitas editar la watchlist localmente, edita los ficheros en `data/` en tu repo y vuelve a desplegar / refrescar.')
        st.markdown('---')
        st.write('Acciones rápidas:')
        if st.button('Recargar vista de archivos'):
            st.experimental_rerun()
        st.markdown('---')
        st.info('La importación por uploader se ha desactivado en favor de la lectura directa desde los ficheros del proyecto.')

def backfill_tab(storage: Optional[PostgresStorage], orchestrator: Optional[Orchestrator]):
    st.title('Backfill')
    st.markdown('Gestiona solicitudes de backfill y ejecuta backfills manuales')
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader('Solicitudes pendientes (tabla backfill_status)')
        try:
            if storage:
                # cargar toda la tabla backfill_status: no API directa; usar list_assets + get_backfill_status por asset
                assets = storage.list_watchlist() if hasattr(storage, 'list_watchlist') else []
                rows = []
                # if backfill_status table exists, try to query a simple helper method; else iterate
                try:
                    # if storage has method to fetch all backfill statuses, use it
                    if hasattr(storage, 'get_all_backfill_status'):
                        rows = storage.get_all_backfill_status()
                    else:
                        # fallback: iterate known assets and call get_backfill_status
                        for a in storage.list_watchlist():
                            asset = a.get('asset')
                            stat = storage.get_backfill_status(asset)
                            if stat:
                                rows.append(stat)
                except Exception:
                    logger.exception('get backfill status loop')
                df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['asset','interval','last_ts','updated_at'])
                st.dataframe(df)
            else:
                st.info('Storage no disponible')
        except Exception as e:
            st.error(f'Error mostrando backfill_status: {e}')
    with col2:
        st.subheader('Pedir un backfill')
        with st.form('req_backfill'):
            asset = st.text_input('Asset')
            interval = st.text_input('Interval (e.g. 5m,1h)')
            requester = st.text_input('Requested by', value=os.getenv('USER','dashboard'))
            submit = st.form_submit_button('Crear request')
            if submit:
                if not asset or not interval:
                    st.error('Completa asset y interval')
                elif not storage:
                    st.error('Storage no disponible')
                else:
                    try:
                        res = storage.add_backfill_request(asset.strip(), interval.strip(), requested_by=requester)
                        st.success(f'Request creado para {asset} {interval}')
                    except Exception as e:
                        st.error(f'Error creando request: {e}')
        st.markdown('---')
        st.subheader('Backfill inmediato (intento síncrono)')
        asset2 = st.text_input('Asset para backfill inmediato')
        interval2 = st.text_input('Interval para backfill inmediato')
        if st.button('Ejecutar backfill ahora'):
            if not orchestrator:
                st.error('Orchestrator no disponible — no se puede ejecutar backfill programado desde UI')
            elif not asset2 or not interval2:
                st.error('Completa asset y interval')
            else:
                try:
                    with st.spinner('Ejecutando backfill...'):
                        # El orchestrator puede exponer init_and_backfill o run_backfill_for
                        if hasattr(orchestrator, 'run_backfill_for'):
                            orchestrator.run_backfill_for(asset2.strip(), interval2.strip())
                        elif hasattr(orchestrator, 'compute_indicators_for'):
                            # fallback: compute indicators as proxy for fetching
                            orchestrator.compute_indicators_for(asset2.strip(), interval2.strip(), lookback=1000)
                        else:
                            raise RuntimeError('Orchestrator no expone run_backfill_for ni compute_indicators_for')
                    st.success('Backfill completado (o iniciado)')
                except Exception as e:
                    st.error(f'Backfill falló: {e}')

def scores_tab(storage: Optional[PostgresStorage]):
    st.title('Scores & Indicators')
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader('Últimos scores por asset')
        n = st.number_input('Últimos por asset', min_value=1, max_value=10, value=1)
        if st.button('Cargar últimos scores'):
            if not storage:
                st.error('Storage no disponible')
            else:
                try:
                    df = storage.load_latest_scores(limit_per_asset=int(n))
                    if df.empty:
                        st.info('No hay scores')
                    else:
                        st.dataframe(df)
                except Exception as e:
                    st.error(f'Error cargando scores: {e}')
    with col2:
        st.subheader('Inferencia (ejecutar modelo y persistir scores)')
        asset = st.text_input('Asset para inferir')
        interval = st.text_input('Interval')
        if st.button('Ejecutar inferencia (infer_and_persist)'):
            if not storage:
                st.error('Storage no disponible')
            elif not asset or not interval:
                st.error('Rellena asset/interval')
            else:
                if score_module and hasattr(score_module, 'infer_and_persist'):
                    try:
                        with st.spinner('Ejecutando inferencia...'):
                            score_module.infer_and_persist(storage, asset.strip(), interval.strip(), model_name=None)
                        st.success('Inferencia ejecutada y guardada en scores')
                    except Exception as e:
                        st.error(f'Inferencia fallo: {e}')
                else:
                    st.error('Modulo score/infer_and_persist no disponible — implementa core.score.infer_and_persist')

def assets_tab(storage: Optional[PostgresStorage]):
    st.title('Assets Inspector')
    try:
        assets = storage.list_assets() if storage else []
    except Exception:
        assets = []
    asset = st.selectbox('Asset', options=[''] + sorted(assets))
    if not asset:
        return
    intervals = storage.list_intervals_for_asset(asset) if storage else []
    interval = st.selectbox('Interval', options=[''] + intervals)
    start = st.text_input('start ts (ms, optional)')
    end = st.text_input('end ts (ms, optional)')
    limit = st.number_input('limit', min_value=1, max_value=5000, value=500)
    if st.button('Cargar velas'):
        if not storage:
            st.error('Storage no disponible')
        elif not interval:
            st.error('Selecciona interval')
        else:
            try:
                s = int(start) if start else None
                e = int(end) if end else None
                df = storage.load_candles(asset, interval, start_ts=s, end_ts=e, limit=limit)
                if df.empty:
                    st.info('No hay velas')
                else:
                    # Preferimos mostrar gráfico con timestamps legibles
                    try:
                        df_plot = df.copy()
                        df_plot['ts_dt'] = pd.to_datetime(df_plot['ts'], unit='ms', utc=True)
                        df_plot = df_plot.set_index('ts_dt')
                        st.line_chart(df_plot['close'])
                    except Exception:
                        st.line_chart(df.set_index('ts')['close'])
                    st.dataframe(df.tail(500))
                    # si core.score tiene funciones de indicadores, mostrar ejemplo
                    if score_module and hasattr(score_module, 'features_from_candles'):
                        try:
                            feats = score_module.features_from_candles(df)
                            st.subheader('Features sample')
                            st.dataframe(feats.tail(50))
                        except Exception:
                            logger.exception('features_from_candles')
            except Exception as e:
                st.error(f'Error cargando velas: {e}')

def models_tab(storage: Optional[PostgresStorage]):
    st.title('Models Registry')
    st.markdown('Lista y gestión de modelos guardados en storage.models')
    asset = st.text_input('Asset (opcional)')
    interval = st.text_input('Interval (opcional)')
    if st.button('Listar modelos'):
        if not storage:
            st.error('Storage no disponible')
        else:
            rows = []
            try:
                if asset and interval:
                    rows = storage.list_models_for_asset_interval(asset.strip(), interval.strip())
                else:
                    # intentar listar todos — si no hay helper, leer por asset known
                    if hasattr(storage, 'list_models'):
                        rows = storage.list_models()
                    else:
                        # fallback: provemos assets y list_models_for_asset_interval
                        for a in storage.list_assets():
                            for iv in storage.list_intervals_for_asset(a):
                                rows += storage.list_models_for_asset_interval(a, iv)
                if not rows:
                    st.info('No hay modelos')
                else:
                    st.dataframe(pd.DataFrame(rows))
            except Exception as e:
                st.error(f'Error listando modelos: {e}')

def train_infer_tab(storage: Optional[PostgresStorage]):
    st.title('Train / Infer')
    st.markdown('Entrenamiento de modelos y ejecución de inferencia desde UI')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Entrenar modelo (train_ai_model)')
        asset = st.text_input('Asset para entrenar')
        interval = st.text_input('Interval')
        model_name = st.text_input('Model name', value='model_auto')
        epochs = st.number_input('Epochs (si aplica)', min_value=1, max_value=1000, value=10)
        if st.button('Entrenar modelo'):
            if not ai_train_module or not hasattr(ai_train_module, 'train_ai_model'):
                st.error('core.ai_train.train_ai_model no disponible — implementa core/ai_train.train_ai_model')
            elif not storage:
                st.error('Storage no disponible')
            elif not asset or not interval:
                st.error('Rellena asset e interval')
            else:
                try:
                    with st.spinner('Entrenando (puede tardar)...'):
                        ai_train_module.train_ai_model(storage, asset.strip(), interval.strip(), model_name=model_name, epochs=int(epochs))
                    st.success('Entrenamiento finalizado y modelo persistido')
                except Exception as e:
                    st.error(f'Entrenamiento falló: {e}')
    with col2:
        st.subheader('Inferir (batch)')
        asset_i = st.text_input('Asset para inferencia', key='i_asset')
        interval_i = st.text_input('Interval para inferencia', key='i_interval')
        if st.button('Ejecutar inferencia y persistir'):
            if not score_module or not hasattr(score_module, 'infer_and_persist'):
                st.error('core.score.infer_and_persist no disponible')
            elif not storage:
                st.error('Storage no disponible')
            elif not asset_i or not interval_i:
                st.error('Rellena asset/interval')
            else:
                try:
                    with st.spinner('Ejecutando inferencia...'):
                        score_module.infer_and_persist(storage, asset_i.strip(), interval_i.strip(), model_name=None)
                    st.success('Inferencia guardada')
                except Exception as e:
                    st.error(f'Inferencia falló: {e}')

def backtest_tab(storage: Optional[PostgresStorage]):
    st.title('Backtest')
    st.markdown('Ejecutar backtests desde UI y visualizar métricas y equity curve')
    asset = st.text_input('Asset para backtest')
    interval = st.text_input('Interval')
    start = st.text_input('Start ts ms (optional)')
    end = st.text_input('End ts ms (optional)')
    if st.button('Ejecutar backtest'):
        if not backtest_module or not hasattr(backtest_module, 'run_backtest_for'):
            st.error('core.backtest.run_backtest_for no disponible — implementa motor de backtest')
        elif not storage:
            st.error('Storage no disponible')
        elif not asset or not interval:
            st.error('Rellena asset/interval')
        else:
            try:
                s = int(start) if start else None
                e = int(end) if end else None
                with st.spinner('Ejecutando backtest...'):
                    res = backtest_module.run_backtest_for(storage, asset.strip(), interval.strip(), start_ts=s, end_ts=e)
                st.subheader('Resultados')
                # compatibilidad: algunos motores devuelven 'metrics' y 'equity_curve'; otros devuelven summary
                if isinstance(res, dict):
                    # try common keys first
                    metrics = {}
                    metrics_keys = ['total_return', 'cagr', 'annual_volatility', 'sharpe', 'max_drawdown']
                    for k in metrics_keys:
                        if k in res:
                            metrics[k] = res[k]
                    if metrics:
                        st.json(metrics)
                    elif 'metrics' in res:
                        st.json(res['metrics'])
                    else:
                        st.json(res)
                    if 'equity_curve' in res:
                        df = pd.DataFrame(res['equity_curve'])
                        if 'ts' in df.columns:
                            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                            df = df.set_index('ts')
                        if 'equity' in df.columns:
                            st.line_chart(df['equity'])
                else:
                    st.json(res)
            except Exception as e:
                st.error(f'Backtest falló: {e}')

def ai_tab(storage: Optional[PostgresStorage]):
    st.title('IA — OpenAI / Interference')
    if not os.getenv('OPENAI_API_KEY'):
        st.info('OPENAI_API_KEY no configurada. Configúrala en el entorno para habilitar la IA.')
    else:
        st.success('OpenAI configurado')
    st.markdown('Funciones IA opcionales: resumen de activos, generación de insights, explicaciones de scores')
    if ai_interf_module and hasattr(ai_interf_module, 'explain_scores'):
        asset = st.text_input('Asset a explicar (IA)')
        if st.button('Explicar scores con IA'):
            if not asset:
                st.error('Especifica asset')
            else:
                try:
                    with st.spinner('Llamando IA...'):
                        text = ai_interf_module.explain_scores(asset)
                    st.text_area('Explicación IA', value=text, height=300)
                except Exception as e:
                    st.error(f'IA fallo: {e}')
    else:
        st.info('core.ai_interference.explain_scores no disponible — implementa para habilitar')

def settings_tab(storage: Optional[PostgresStorage]):
    st.title('Settings')
    st.markdown('Variables de entorno y configuración de despliegue')
    envs = {k: bool(os.getenv(k)) for k in ['DATABASE_URL','OPENAI_API_KEY']}
    st.json(envs)
    st.markdown('Instrucciones para ambiente: export DATABASE_URL="postgresql://user:pass@host:5432/db"')

def logs_tab():
    st.title('Logs')
    log_dir = '/tmp/watchlist_logs'
    if not os.path.exists(log_dir):
        st.info('No existe /tmp/watchlist_logs')
        return
    files = sorted(os.listdir(log_dir), reverse=True)
    sel = st.selectbox('Fichero de logs', options=[''] + files)
    if sel:
        path = os.path.join(log_dir, sel)
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                data = f.read()[-20000:]
            st.code(data)
        except Exception as e:
            st.error(f'Error leyendo log: {e}')

# ---------------------------
# Main
# ---------------------------

def main():
    # Obtenemos storage y orchestrator (get_orchestrator acepta storage posicionalmente)
    storage = get_storage()
    orchestrator = get_orchestrator(storage)

    tabs = ['Overview','Watchlist','Backfill','Scores','Assets','Models','Train/Infer','Backtest','IA','Settings','Logs']
    choice = st.sidebar.selectbox('Pestañas', options=tabs)

    if choice == 'Overview':
        overview_tab(storage)
    elif choice == 'Watchlist':
        watchlist_tab(storage)
    elif choice == 'Backfill':
        backfill_tab(storage, orchestrator)
    elif choice == 'Scores':
        scores_tab(storage)
    elif choice == 'Assets':
        assets_tab(storage)
    elif choice == 'Models':
        models_tab(storage)
    elif choice == 'Train/Infer':
        train_infer_tab(storage)
    elif choice == 'Backtest':
        backtest_tab(storage)
    elif choice == 'IA':
        ai_tab(storage)
    elif choice == 'Settings':
        settings_tab(storage)
    elif choice == 'Logs':
        logs_tab()
    else:
        st.info('Seleccione una pestaña')

if __name__ == '__main__':
    main()
