# dashboard/app.py
"""
Streamlit dashboard para Watchlist (cripto & acciones).

Mejoras aplicadas en esta versión:
- Corrige llamadas a _load_scores/_load_candles (pasan `db=` en vez de `db_path=`).
- Añade barra de filtros horizontal (asset multiselect, interval, score range, fecha, sort, stop/target, tipo).
- Inicializa la lista de assets combinando config + data/config CSVs (si existen) para "empezar con todas las monedas y acciones".
- Mantiene botones de Refresh/Backfill en la barra lateral.
- Mejora uso de imports: intenta `from dashboard.utils` y cae a `from utils` para ejecutar desde la carpeta `dashboard`.

Nota: no se cambian las funciones utilitarias en dashboard.utils; este script usa sus funciones cuando están disponibles.
"""
from __future__ import annotations
import os
import time
from typing import Optional, List, Tuple
from pathlib import Path
import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go

# Try robust imports for utils: allow running both desde la raiz (as package) y desde dashboard/ (direct)
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

st.set_page_config(APP_TITLE, layout="wide")

# ---- Authentication (simple) ----
PASSWORD = os.getenv("DASHBOARD_PASSWORD", None)


def auth_check() -> bool:
    """Very basic password protection."""
    if PASSWORD is None:
        return True
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        return True
    with st.sidebar:
    st.header("Operaciones")

    # --- Añadir nuevo asset desde el dashboard ---
    st.subheader("Añadir asset (nuevo)")
    new_sym = st.text_input("Nuevo símbolo (p.e. BTCUSDT o AAPL)", value="")
    type_choice = st.radio("Tipo de asset", ["crypto", "stock"], index=0)
    if st.button("Añadir y lanzar backfill"):
        sym = new_sym.strip()
        if not sym:
            st.error("Introduce un símbolo válido.")
        else:
            # decide csv destino
            csv_path = CSV_CONFIG_DIR / ("cryptos.csv" if type_choice == "crypto" else "actions.csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            # leer existentes (si hay)
            existing = []
            try:
                if csv_path.exists():
                    dfc = pd.read_csv(csv_path)
                    cols_low = [c.lower() for c in dfc.columns]
                    if "symbol" in cols_low:
                        key = next(c for c in dfc.columns if c.lower() == "symbol")
                        existing = dfc[key].astype(str).str.strip().tolist()
                    else:
                        existing = dfc.iloc[:, 0].astype(str).str.strip().tolist()
            except Exception:
                existing = []

            if sym in existing:
                st.warning(f"{sym} ya existe en {csv_path.name}")
            else:
                try:
                    # append row preserving header if present
                    if csv_path.exists():
                        # append simple line
                        with csv_path.open("a", encoding="utf-8", newline="") as fh:
                            # ensure newline
                            fh.write(f"{sym}\n")
                    else:
                        # crear con cabecera 'symbol'
                        with csv_path.open("w", encoding="utf-8", newline="") as fh:
                            fh.write("symbol\n")
                            fh.write(f"{sym}\n")
                    st.success(f"{sym} añadido a {csv_path.name}")

                    # Lanzar backfill (completo) para que se descargue histórico y se guarde
                    with st.spinner("Lanzando backfill para el nuevo símbolo (puede tardar)..."):
                        try:
                            ok, msg = call_backfill_historical(crypto_interval=interval, stock_interval=interval)
                            if ok:
                                st.success(msg or "Backfill lanzado correctamente")
                            else:
                                st.error(msg or "Backfill falló al iniciarse")
                        except Exception as e:
                            st.error(f"Error al lanzar backfill: {e}")
                except Exception as e:
                    st.error(f"Error guardando CSV: {e}")

    st.markdown("---")
    st.caption("Variables importantes (env): DB_PATH, DASHBOARD_PASSWORD")
    st.write("Última carga:", ts_to_iso(int(time.time())))

# ---- Load data helpers (cached) ----
@st.cache_data(ttl=30)
def _load_scores(asset: str, interval: str, db: Optional[str], limit: int = 5000) -> pd.DataFrame:
    return load_scores_df(asset, interval, db_path=db, limit=limit)


@st.cache_data(ttl=30)
def _load_candles(asset: str, interval: str, db: Optional[str], limit: int = 2000) -> pd.DataFrame:
    return load_candles_df(asset, interval, db_path=db, limit=limit)

# Validate chosen assets
if not chosen_assets:
    st.info("Selecciona uno o varios assets en la barra de filtros de arriba (multi-select). Si quieres todos, elige todos.")
    st.stop()

# Compile full scores dataframe by concatenating selected assets (careful with limits)
all_scores_list = []
for sym in chosen_assets:
    try:
        df = _load_scores(sym, interval, db=db_path, limit=10000)
        if df is not None and not df.empty:
            df['asset'] = sym
            all_scores_list.append(df)
    except Exception as e:
        st.warning(f"No se pudieron cargar scores para {sym}: {e}")

if not all_scores_list:
    st.warning("No hay scores cargados para los assets seleccionados.")
    st.stop()

scores_df = pd.concat(all_scores_list, ignore_index=True)

# Apply top filters: date range, score range
try:
    # ts is unix seconds; convert date inputs to timestamps
    start_dt, end_dt = date_range
    start_ts = int(datetime.datetime.combine(start_dt, datetime.time.min).timestamp())
    end_ts = int(datetime.datetime.combine(end_dt, datetime.time.max).timestamp())
    scores_df = scores_df[(scores_df['ts'] >= start_ts) & (scores_df['ts'] <= end_ts)]
except Exception:
    pass

scores_df = scores_df[(scores_df['score'] >= score_min) & (scores_df['score'] <= score_max)]

# Additional quick filters ui (inline)
with st.expander("Filtros avanzados (stop/target/rango)"):
    min_stop = st.number_input("Stop >=", value=float(scores_df['stop'].min()) if 'stop' in scores_df.columns and not scores_df['stop'].isna().all() else 0.0)
    max_target = st.number_input("Target <=", value=float(scores_df['target'].max()) if 'target' in scores_df.columns and not scores_df['target'].isna().all() else 1e12)
    # apply
    if 'stop' in scores_df.columns:
        scores_df = scores_df[scores_df['stop'] >= min_stop]
    if 'target' in scores_df.columns:
        scores_df = scores_df[scores_df['target'] <= max_target]

# Sorting
ascending = True if sort_order == 'asc' else False
if sort_by in scores_df.columns:
    scores_df = scores_df.sort_values(sort_by, ascending=ascending)
else:
    # default sort by ts desc
    scores_df = scores_df.sort_values('ts', ascending=False)

# Show top N (max_rows) — we put control in an expander
with st.expander("Vista y descarga"):
    st.write(f"Total filas tras filtros: {len(scores_df)}")
    max_rows = st.number_input("Max filas a mostrar", min_value=10, max_value=20000, value=500, step=10)
    out_df = scores_df.head(max_rows).reset_index(drop=True)
    st.dataframe(out_df)
    csv = out_df.to_csv(index=False)
    st.download_button("Descargar CSV (scores)", data=csv, file_name=f"scores_{'_'.join(chosen_assets)}_{interval}.csv", mime="text/csv")

# Select a row for details
sel_idx = st.number_input("Selecciona index (fila) para ver vela y rango", min_value=0, max_value=max(0, len(out_df)-1), value=0)
sel_row = out_df.iloc[sel_idx]

# Right detail pane
right_col = st.columns([1])[0]
with right_col:
    st.subheader("Detalle seleccionado")
    st.json({
        'ts': int(sel_row['ts']),
        'dt': ts_to_iso(int(sel_row['ts'])),
        'asset': sel_row.get('asset'),
        'score': float(sel_row.get('score', 0.0)),
        'range_min': sel_row.get('range_min'),
        'range_max': sel_row.get('range_max'),
        'stop': sel_row.get('stop'),
        'target': sel_row.get('target'),
        'multiplier': sel_row.get('multiplier', None),
    })
    st.metric("Score", f"{float(sel_row.get('score',0.0)):.3f}")
    if sel_row.get('stop') and sel_row.get('target'):
        stop, target = float(sel_row['stop']), float(sel_row['target'])
        st.metric("Stop", f"{stop:.6f}")
        st.metric("Target", f"{target:.6f}")

# ---- Candles and plot for selected time window ----
with st.expander("Ver velas alrededor de la señal"):
    try:
        candles_df = _load_candles(sel_row.get('asset'), interval, db=db_path, limit=2000)
    except Exception as e:
        st.error(f"Error cargando velas: {e}")
        candles_df = pd.DataFrame()

    if candles_df is None or candles_df.empty:
        st.warning("No hay velas disponibles.")
    else:
        # find index of the selected ts
        sel_ts = int(sel_row['ts'])
        candles_df['ts'] = candles_df['ts'].astype(int)
        try:
            idx_list = candles_df.index[candles_df['ts'] == sel_ts].tolist()
            if idx_list:
                center = idx_list[0]
            else:
                center = (candles_df['ts'] - sel_ts).abs().idxmin()
        except Exception:
            center = 0
        window = 100
        lo = max(0, center - window)
        hi = min(len(candles_df) - 1, center + window)
        view = candles_df.iloc[lo:hi+1].copy()

        fig = go.Figure(data=[go.Candlestick(x=pd.to_datetime(view['ts'], unit='s'),
                                             open=view['open'],
                                             high=view['high'],
                                             low=view['low'],
                                             close=view['close'])])
        if pd.notna(sel_row.get('stop')):
            fig.add_hline(y=float(sel_row['stop']), line_dash="dot", annotation_text="Stop")
        if pd.notna(sel_row.get('target')):
            fig.add_hline(y=float(sel_row['target']), line_dash="dot", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)

# ---- helper to parse interval seconds ----
def parse_interval_seconds(itv: str) -> int:
    try:
        import re
        m = re.match(r'(\d+)([mhdw]+)', itv)
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

st.caption("Dashboard — diseñado para el MVP intermedio. Protege la instancia en producción y desarrollo")
