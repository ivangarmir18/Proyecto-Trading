# dashboard/watchlist_ui.py
"""
UI de Watchlist para Streamlit.

Función exportada:
- render_watchlist_ui(storage)

Interacciones:
- Lista actual (desde storage.list_watchlist())
- Añadir asset (llama storage.add_watchlist_symbol)
- Eliminar asset (llama storage.remove_watchlist_symbol)
- Importar CSV (usa dashboard.utils.save_user_watchlist_csv y persiste en DB)
- Exportar CSV
- Botón para solicitar Backfill por asset (llama storage.add_backfill_request)
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import json
from typing import Optional, Any

from dashboard import utils

def _to_df(wl):
    if not wl:
        return pd.DataFrame(columns=["asset", "meta"])
    return pd.DataFrame([{"asset": r.get("asset"), "meta": json.dumps(r.get("meta") or {}, ensure_ascii=False)} if isinstance(r, dict) else {"asset": r, "meta": ""} for r in wl])

def render_watchlist_ui(storage: Optional[Any]):
    st.subheader("Watchlist")
    col_a, col_b = st.columns([3, 1])

    # left: list
    with col_a:
        if storage:
            try:
                wl = storage.list_watchlist()
            except Exception:
                st.error("Error leyendo watchlist desde storage")
                wl = utils.load_user_watchlist_csv()
        else:
            wl = utils.load_user_watchlist_csv()
        df = _to_df(wl)
        st.dataframe(df, use_container_width=True)

        # Export button
        if not df.empty:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Exportar CSV", csv, file_name="watchlist_export.csv")

    # right: actions
    with col_b:
        st.markdown("### Acciones")
        with st.form("add_watchlist_form", clear_on_submit=True):
            new_asset = st.text_input("Nuevo asset", "")
            new_interval = st.text_input("Interval (opcional)", "")
            added_by = st.text_input("Added by", value=st.session_state.get("user", "dashboard"))
            submit = st.form_submit_button("Añadir")
            if submit:
                if not new_asset:
                    st.error("Escribe un asset")
                else:
                    if not storage:
                        st.warning("Storage no disponible — añadiendo a CSV local")
                        prev = utils.load_user_watchlist_csv()
                        prev.append({"asset": new_asset.strip(), "meta": {"interval": new_interval or None, "added_by": added_by}})
                        utils.save_user_watchlist_csv(prev)
                        st.success("Añadido al CSV local")
                    else:
                        try:
                            res = storage.add_watchlist_symbol(new_asset.strip(), asset_type=None, added_by=added_by)
                            # optionally set interval in meta via update (if PostgresStorage supports)
                            try:
                                if hasattr(storage, "update_watchlist_meta"):
                                    storage.update_watchlist_meta(new_asset.strip(), {"interval": new_interval or None, "added_by": added_by})
                            except Exception:
                                # ignore optional update error
                                pass
                            st.success(f"Añadido a DB: {res.get('asset')}")
                        except Exception as e:
                            st.error(f"Error añadiendo a storage: {e}")

        st.markdown("---")
        st.markdown("Eliminar / Backfill")
        with st.form("rm_backfill"):
            sel = st.selectbox("Selecciona asset", options=[""] + list(df["asset"].dropna().astype(str).tolist()))
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Eliminar"):
                    if not sel:
                        st.error("Selecciona asset")
                    else:
                        if storage:
                            try:
                                ok = storage.remove_watchlist_symbol(sel)
                                if ok:
                                    st.success(f"Eliminado {sel}")
                                else:
                                    st.info("No estaba en la watchlist")
                            except Exception as e:
                                st.error(f"Error eliminando: {e}")
                        else:
                            # remove from CSV
                            prev = utils.load_user_watchlist_csv()
                            prev = [x for x in prev if x.get("asset") != sel]
                            utils.save_user_watchlist_csv(prev)
                            st.success(f"Eliminado {sel} (CSV local)")
            with col2:
                if st.form_submit_button("Solicitar backfill"):
                    if not sel:
                        st.error("Selecciona asset")
                    else:
                        interval = st.text_input("Interval para backfill", value="")
                        if not interval:
                            st.error("Especifica interval")
                        else:
                            if storage:
                                try:
                                    storage.add_backfill_request(sel, interval, requested_by=st.session_state.get("user", "dashboard"))
                                    st.success(f"Backfill solicitado para {sel} {interval}")
                                except Exception as e:
                                    st.error(f"Error solicitando backfill: {e}")
                            else:
                                st.warning("Storage no disponible: no se puede solicitar backfill")

    # Import CSV (bottom)
    st.markdown("---")
    st.subheader("Importar CSV a watchlist")
    uploaded = st.file_uploader("CSV con columna 'asset' (opcional 'interval','meta')", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            if "asset" not in df_in.columns:
                st.error("CSV debe tener columna 'asset'")
            else:
                loaded = []
                for _, r in df_in.iterrows():
                    asset = str(r["asset"]).strip()
                    meta = None
                    if "meta" in r and not pd.isna(r["meta"]):
                        try:
                            meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else r["meta"]
                        except Exception:
                            meta = {"raw": r["meta"]}
                    interval = r.get("interval") if "interval" in r else None
                    try:
                        if storage:
                            storage.add_watchlist_symbol(asset, asset_type=None, added_by=st.session_state.get("user","csv"))
                            # optional meta update
                            if hasattr(storage, "update_watchlist_meta"):
                                storage.update_watchlist_meta(asset, {"interval": interval})
                        else:
                            prev = utils.load_user_watchlist_csv()
                            prev.append({"asset": asset, "meta": {"interval": interval, **(meta or {})}})
                            utils.save_user_watchlist_csv(prev)
                        loaded.append(asset)
                    except Exception:
                        st.warning(f"Fallo añadiendo {asset}")
                st.success(f"Añadidos: {len(loaded)}")
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")
