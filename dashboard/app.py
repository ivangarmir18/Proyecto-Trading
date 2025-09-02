"""
Streamlit dashboard mejorado para Watchlist (Cripto & Acciones).

Cambios principales:
- Corrige indentaciones y estructura general.
- Top-bar (header) con elementos de control principales (filtro global, búsqueda/añadir asset, refresh, backfill).
- Añadir asset desde UI: seleccionar tipo (crypto/stock) y símbolo; guarda en CSV local (data/config/) y lanza backfill.
- Tabla superior compacta (summary) con la última fila de score por asset, ordenable por columnas (usando st.dataframe).
- Filtros en una barra horizontal encima de la tabla (intervalo, fecha, score range, ordenación).
- Panel de detalle y gráfico de velas para la fila seleccionada.
- Uso de variables de entorno (para Render) y lectura de config.json cuando esté disponible.

Notas:
- Algunas funciones auxiliares (load_config, get_db_path, load_scores_df, load_candles_df,
  call_backfill_historical, call_refresh_watchlist, ts_to_iso) se importan de dashboard.utils o utils.
  Si sus firmas son diferentes, se intentan llamadas alternativas con comprobaciones.
- No se incluye ninguna clave o secret en este fichero.
"""

from __future__ import annotations
import os
from pathlib import Path
import time
import datetime
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Intentar imports robustos para ejecutar desde la raíz o desde el folder dashboard/
try:
    from dashboard.utils import (
        load_config,
        get_db_path,
        load_scores_df,
        load_candles_df,
        call_refresh_watchlist,
        call_backfill_historical,
        ts_to_iso,
    )
except Exception:
    from utils import (
        load_config,
        get_db_path,
        load_scores_df,
        load_candles_df,
        call_refresh_watchlist,
        call_backfill_historical,
        ts_to_iso,
    )

# UI constants
APP_TITLE = "Watchlist Dashboard — Crypto & Stocks"
DEFAULT_CONFIG_PATH = "config.json"
CSV_CONFIG_DIR = Path("data/config")

st.set_page_config(APP_TITLE, layout="wide")

# ---- Authentication (very simple) ----
PASSWORD = os.getenv("DASHBOARD_PASSWORD", None)


def auth_check() -> bool:
    """Basic password gate stored in DASHBOARD_PASSWORD env var. Uses session_state."""
    if PASSWORD is None:
        return True
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        return True

    # Simple login form (modal-like in sidebar)
    with st.sidebar:
        st.write("**Dashboard — Autenticación**")
        pw = st.text_input("Contraseña", type="password")
        if st.button("Entrar"):
            if pw == PASSWORD:
                st.session_state.auth_ok = True
                st.experimental_rerun()
            else:
                st.error("Contraseña incorrecta")
    return False


# ---- Cached loaders ----
@st.cache_data(ttl=30)
def _load_scores(asset: str, interval: str, db: Optional[str], limit: int = 5000) -> pd.DataFrame:
    try:
        return load_scores_df(asset, interval, db_path=db, limit=limit)
    except TypeError:
        # fallback if signature differs
        return load_scores_df(asset, interval, db, limit)


@st.cache_data(ttl=30)
def _load_candles(asset: str, interval: str, db: Optional[str], limit: int = 2000) -> pd.DataFrame:
    try:
        return load_candles_df(asset, interval, db_path=db, limit=limit)
    except TypeError:
        return load_candles_df(asset, interval, db, limit)


def parse_interval_seconds(itv: str) -> int:
    try:
        import re
        m = re.match(r"(\d+)([mhdw]+)$", itv)
        if not m:
            return 0
        n = int(m.group(1)); unit = m.group(2)
        if unit.startswith('m'):
            return n * 60
        if unit.startswith('h'):
            return n * 3600
        if unit.startswith('d'):
            return n * 86400
        if unit.startswith('w'):
            return n * 604800
    except Exception:
        return 0
    return 0


# ---- Utilities for assets list management ----

def read_assets_from_config(cfg: dict) -> Dict[str, List[str]]:
    """Return dict with keys 'crypto' and 'stock' containing lists."""
    res = {'crypto': [], 'stock': []}
    try:
        assets = cfg.get('assets', {})
        res['crypto'] = assets.get('cripto', []) or assets.get('crypto', []) or []
        res['stock'] = assets.get('acciones', []) or assets.get('stocks', []) or assets.get('acciones', []) or []
    except Exception:
        pass
    return res


def read_assets_from_csvs() -> Dict[str, List[str]]:
    res = {'crypto': [], 'stock': []}
    try:
        CSV_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        c1 = CSV_CONFIG_DIR / 'cryptos.csv'
        c2 = CSV_CONFIG_DIR / 'actions.csv'
        if c1.exists():
            try:
                dfc = pd.read_csv(c1)
                if 'symbol' in [c.lower() for c in dfc.columns]:
                    key = next(c for c in dfc.columns if c.lower() == 'symbol')
                    res['crypto'] = dfc[key].astype(str).str.strip().tolist()
                else:
                    res['crypto'] = dfc.iloc[:, 0].astype(str).str.strip().tolist()
            except Exception:
                res['crypto'] = []
        if c2.exists():
            try:
                dfa = pd.read_csv(c2)
                if 'symbol' in [c.lower() for c in dfa.columns]:
                    key = next(c for c in dfa.columns if c.lower() == 'symbol')
                    res['stock'] = dfa[key].astype(str).str.strip().tolist()
                else:
                    res['stock'] = dfa.iloc[:, 0].astype(str).str.strip().tolist()
            except Exception:
                res['stock'] = []
    except Exception:
        pass
    return res


def append_asset_to_csv(symbol: str, asset_type: str) -> None:
    csv_path = CSV_CONFIG_DIR / ('cryptos.csv' if asset_type == 'crypto' else 'actions.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        # Avoid duplicates
        try:
            df = pd.read_csv(csv_path)
            cols_low = [c.lower() for c in df.columns]
            if 'symbol' in cols_low:
                key = next(c for c in df.columns if c.lower() == 'symbol')
                existing = df[key].astype(str).str.strip().tolist()
            else:
                existing = df.iloc[:, 0].astype(str).str.strip().tolist()
        except Exception:
            existing = []
        if symbol in existing:
            return
        with csv_path.open('a', encoding='utf-8', newline='') as fh:
            fh.write(f"{symbol}\n")
    else:
        with csv_path.open('w', encoding='utf-8', newline='') as fh:
            fh.write('symbol\n')
            fh.write(f"{symbol}\n")


# ---- Main ----

def main():
    if not auth_check():
        return

    # Load config if exists
    cfg = {}
    try:
        cfg = load_config(DEFAULT_CONFIG_PATH)
    except Exception:
        try:
            import json
            p = Path(DEFAULT_CONFIG_PATH)
            if p.exists():
                cfg = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}

    db_path = None
    try:
        db_path = get_db_path(cfg)
    except Exception:
        db_path = os.environ.get('DATABASE_URL') or cfg.get('app', {}).get('db_path')

    # Build initial assets list from config + CSVs
    assets_cfg = read_assets_from_config(cfg)
    assets_csv = read_assets_from_csvs()
    all_crypto = sorted(list(dict.fromkeys(assets_cfg.get('crypto', []) + assets_csv.get('crypto', []))))
    all_stock = sorted(list(dict.fromkeys(assets_cfg.get('stock', []) + assets_csv.get('stock', []))))
    all_assets = all_crypto + all_stock

    # Top bar: header + quick summary and add asset
    header_col, controls_col = st.columns([3, 1])
    with header_col:
        st.title(APP_TITLE)
        st.caption("Monitorización técnica — Cripto & Acciones — actualizado")

    with controls_col:
        # Quick actions
        if st.button('Refresh watchlist'):
            try:
                call_refresh_watchlist()
                st.success('Refresh lanzado')
            except Exception as e:
                st.error(f'Error lanzando refresh: {e}')
        if st.button('Backfill (emergency)'):
            try:
                call_backfill_historical()
                st.success('Backfill lanzado (global)')
            except Exception as e:
                st.error(f'Error lanzando backfill: {e}')

    st.markdown('---')

    # Filters row
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
    with c1:
        chosen_type = st.selectbox('Tipo', ['all', 'crypto', 'stock'], index=0)
        assets_options = all_assets if chosen_type == 'all' else (all_crypto if chosen_type == 'crypto' else all_stock)
        chosen_assets = st.multiselect('Assets', options=assets_options, default=assets_options[:10])
    with c2:
        interval = st.selectbox('Intervalo', options=['5m', '30m', '1h', '4h', '1d'], index=0)
        # derived seconds
        _ = parse_interval_seconds(interval)
    with c3:
        today = datetime.date.today()
        date_range = st.date_input('Rango fecha', value=(today - datetime.timedelta(days=30), today))
    with c4:
        score_min, score_max = st.slider('Score range', 0.0, 1.0, (0.0, 1.0), step=0.01)
    with c5:
        sort_by = st.selectbox('Ordenar por', options=['ts', 'score', 'asset'], index=0)
        sort_order = st.selectbox('Orden', options=['desc', 'asc'], index=0)

    # Add new asset UI (compact)
    add_col1, add_col2 = st.columns([3, 2])
    with add_col1:
        st.subheader('Añadir nuevo asset')
        new_sym = st.text_input('Símbolo (p.e. BTCUSDT o AAPL)', value='')
    with add_col2:
        type_choice = st.selectbox('Tipo asset', ['crypto', 'stock'], index=0)
        if st.button('Añadir y lanzar backfill'):
            sym = new_sym.strip().upper()
            if not sym:
                st.error('Introduce un símbolo válido')
            else:
                try:
                    append_asset_to_csv(sym, 'crypto' if type_choice == 'crypto' else 'stock')
                    st.success(f'{sym} añadido a la lista de {type_choice}s')
                    # Try best-effort to call backfill for the symbol
                    try:
                        # Try common signature first
                        call_backfill_historical(asset=sym, asset_type=type_choice, interval=interval)
                        st.success('Backfill para símbolo lanzado')
                    except TypeError:
                        # fallback to generic call
                        try:
                            call_backfill_historical()
                            st.info('Backfill global lanzado como fallback')
                        except Exception as e:
                            st.warning(f'No se pudo lanzar backfill automáticamente: {e}')
                    except Exception as e:
                        st.warning(f'Backfill: {e}')
                except Exception as e:
                    st.error(f'Error guardando el símbolo: {e}')

    st.markdown('---')

    # If no assets selected -> show info
    if not chosen_assets:
        st.info('Selecciona uno o varios assets en el filtro superior.')
        st.stop()

    # Build summary table (latest score per asset)
    summary_rows = []
    for sym in chosen_assets:
        try:
            s = _load_scores(sym, interval, db=db_path, limit=5)
            if s is None or s.empty:
                continue
            latest = s.sort_values('ts', ascending=False).iloc[0]
            summary_rows.append({
                'asset': sym,
                'ts': int(latest.get('ts', 0)),
                'score': float(latest.get('score', 0.0)),
                'range_min': latest.get('range_min'),
                'range_max': latest.get('range_max'),
                'stop': latest.get('stop'),
                'target': latest.get('target'),
            })
        except Exception as e:
            st.warning(f'No se pudo cargar score para {sym}: {e}')

    if not summary_rows:
        st.warning('No hay datos de scores para los assets seleccionados.')
        st.stop()

    summary_df = pd.DataFrame(summary_rows)

    # Apply score/date filters to summary (coarse)
    try:
        start_dt, end_dt = date_range
        start_ts = int(datetime.datetime.combine(start_dt, datetime.time.min).timestamp())
        end_ts = int(datetime.datetime.combine(end_dt, datetime.time.max).timestamp())
        summary_df = summary_df[(summary_df['ts'] >= start_ts) & (summary_df['ts'] <= end_ts)]
    except Exception:
        pass
    summary_df = summary_df[(summary_df['score'] >= score_min) & (summary_df['score'] <= score_max)]

    # Sorting summary
    asc = True if sort_order == 'asc' else False
    if sort_by in summary_df.columns:
        summary_df = summary_df.sort_values(sort_by, ascending=asc)
    else:
        summary_df = summary_df.sort_values('ts', ascending=False)

    st.subheader('Resumen rápido (último score por asset)')
    st.dataframe(summary_df.reset_index(drop=True))

    # Select a row to deep-dive
    st.markdown('---')
    st.subheader('Detalle y vela')
    sel_asset = st.selectbox('Elige asset para detalle', options=summary_df['asset'].tolist(), index=0)
    # load latest score row for selected
    sel_scores = _load_scores(sel_asset, interval, db=db_path, limit=200)
    if sel_scores is None or sel_scores.empty:
        st.warning('No hay scores para el asset seleccionado')
        st.stop()
    sel_row = sel_scores.sort_values('ts', ascending=False).iloc[0]

    # Show JSON + metrics
    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        st.json({
            'ts': int(sel_row['ts']),
            'dt': ts_to_iso(int(sel_row['ts'])),
            'asset': sel_asset,
            'score': float(sel_row.get('score', 0.0)),
            'range_min': sel_row.get('range_min'),
            'range_max': sel_row.get('range_max'),
            'stop': sel_row.get('stop'),
            'target': sel_row.get('target'),
        })
    with cols[1]:
        st.metric('Score', f"{float(sel_row.get('score',0.0)):.3f}")
    with cols[2]:
        if sel_row.get('stop') is not None:
            st.metric('Stop', f"{float(sel_row.get('stop')):.6f}")
    with cols[3]:
        if sel_row.get('target') is not None:
            st.metric('Target', f"{float(sel_row.get('target')):.6f}")

    # Plot candles around selected ts
    try:
        candles_df = _load_candles(sel_asset, interval, db=db_path, limit=2000)
    except Exception as e:
        st.error(f'Error cargando velas: {e}')
        candles_df = pd.DataFrame()

    if candles_df is None or candles_df.empty:
        st.warning('No hay velas disponibles para el asset seleccionado')
        return

    candles_df['ts'] = candles_df['ts'].astype(int)
    sel_ts = int(sel_row['ts'])
    if sel_ts in candles_df['ts'].values:
        center_idx = candles_df.index[candles_df['ts'] == sel_ts][0]
    else:
        center_idx = (candles_df['ts'] - sel_ts).abs().idxmin()

    window = 150
    lo = max(0, center_idx - window)
    hi = min(len(candles_df) - 1, center_idx + window)
    view = candles_df.iloc[lo:hi+1].copy()

    fig = go.Figure(data=[
        go.Candlestick(x=pd.to_datetime(view['ts'], unit='s'),
                       open=view['open'],
                       high=view['high'],
                       low=view['low'],
                       close=view['close'])
    ])
    if pd.notna(sel_row.get('stop')):
        fig.add_hline(y=float(sel_row['stop']), line_dash='dot', annotation_text='Stop')
    if pd.notna(sel_row.get('target')):
        fig.add_hline(y=float(sel_row['target']), line_dash='dot', annotation_text='Target')

    st.plotly_chart(fig, use_container_width=True)

    st.caption('Dashboard — diseñado para MVP intermedio. Protege la instancia en producción y desarrollo')


if __name__ == '__main__':
    main()
