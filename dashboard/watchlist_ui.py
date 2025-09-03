# dashboard/watchlist_ui.py
import streamlit as st
from typing import Optional
import datetime

def render_watchlist_ui(storage):
    """
    UI mínima para añadir símbolos a la watchlist y pedir backfill.
    `storage` debe exponer:
      - add_watchlist_symbol(asset, asset_type, added_by) -> dict
      - add_backfill_request(asset, interval, requested_by) -> dict
      - list_watchlist() -> list
      - remove_watchlist_symbol(asset) -> bool
    Si no dispone, caerá con excepción (el caller debe manejarlo).
    """
    st.sidebar.header("Añadir símbolo (Watchlist)")
    asset_type = st.sidebar.selectbox("Tipo", options=["crypto", "stock"])
    symbol_input = st.sidebar.text_input("Símbolo (ej. SOLUSDT, BTCUSDT, AAPL)", "")
    interval_default = st.sidebar.selectbox("Interval por defecto para backfill", options=["1m","5m","15m","30m","1h","4h","1d"], index=1)
    added_by = st.sidebar.text_input("Tu nombre (opcional)", "")

    if st.sidebar.button("Añadir y pedir histórico"):
        symbol = symbol_input.strip().upper()
        if not symbol:
            st.sidebar.error("Introduce un símbolo válido.")
            return
        try:
            res = storage.add_watchlist_symbol(symbol, asset_type, added_by or None)
            bf = storage.add_backfill_request(symbol, interval_default, requested_by=(added_by or "dashboard"))
            st.sidebar.success(f"{symbol} añadido. Backfill id={bf.get('id')}")
        except Exception as e:
            st.sidebar.error(f"Error al añadir símbolo: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Watchlist")
    try:
        wl = storage.list_watchlist()
        if not wl:
            st.sidebar.write("Vacía")
        else:
            for item in wl:
                st.sidebar.write(f"{item['asset']} ({item['asset_type']}) — añadido: {item['created_at']}")
                if st.sidebar.button(f"Eliminar {item['asset']}", key=f"del_{item['id']}"):
                    try:
                        ok = storage.remove_watchlist_symbol(item["asset"])
                        if ok:
                            st.sidebar.success(f"{item['asset']} eliminado.")
                            st.experimental_rerun()
                    except Exception as e:
                        st.sidebar.error(f"No se pudo eliminar: {e}")
    except Exception as e:
        st.sidebar.info("Watchlist no disponible: " + str(e))
        # No romper el resto del dashboard si storage no está disponible

