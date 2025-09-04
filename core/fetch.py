# core/fetch.py
"""
core/fetch.py — fetchers unificados y fetcher-adapter para backfill.

Proporciona:
- get_candles(asset, interval, limit=None, start_ts=None, end_ts=None)
- binance_fetcher(asset, interval, start_ts, end_ts)  # returns list[dict]
- yfinance_fetcher(asset, interval, start_ts, end_ts)
- run_backfill(asset, interval, storage_adapter, start_ts=None, end_ts=None, batch_save=200)

Diseño:
- No auto-guardar; storage_adapter.backfill_symbol usa un fetcher que devuelve listas de velas.
- Timestamps en segundos (int).
- Intenta usar ccxt para Binance y yfinance para acciones, pero degrade con mensajes claros.
"""
from __future__ import annotations
import os
import time
import logging
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

# imports opcionales
try:
    import ccxt
except Exception:
    ccxt = None

try:
    import yfinance as yf
    import pandas as pd
except Exception:
    yf = None
    pd = None

BINANCE_TF_MAP = {
    "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "12h": "12h", "1d": "1d"
}

def _to_seconds(ts_ms_or_s: int) -> int:
    ts = int(ts_ms_or_s)
    if ts > 10_000_000_000:
        return int(ts // 1000)
    return ts

def _ensure_binance():
    if not ccxt:
        raise RuntimeError("ccxt no está instalado. Instálalo con `pip install ccxt` para usar Binance.")
    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET")
    exchange = ccxt.binance({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
    return exchange

def binance_fetcher(asset: str, interval: str, start_ts: Optional[int], end_ts: Optional[int], limit_per_call: int = 1000) -> List[Dict]:
    exchange = _ensure_binance()
    tf = BINANCE_TF_MAP.get(interval, interval)
    symbol = asset if "/" in asset else asset + "/USDT"
    since_ms = int(start_ts)*1000 if start_ts else None
    end_ms = int(end_ts)*1000 if end_ts else None

    out = []
    last_ts_ms = since_ms
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, since=last_ts_ms, limit=limit_per_call)
        except Exception as e:
            log.exception("Error fetch_ohlcv binance: %s", e)
            raise
        if not ohlcv:
            break
        for row in ohlcv:
            ts_ms, o, h, l, c, vol = row[0], row[1], row[2], row[3], row[4], row[5]
            ts_s = int(ts_ms // 1000)
            if start_ts and ts_s < start_ts:
                continue
            if end_ts and ts_s > end_ts:
                return out
            out.append({
                "asset": asset,
                "interval": interval,
                "ts": ts_s,
                "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(vol)
            })
            last_ts_ms = ts_ms + 1
        if len(ohlcv) < limit_per_call:
            break
        time.sleep(0.2)
    log.info("binance_fetcher fetched %d candles for %s %s", len(out), asset, interval)
    return out

def yfinance_fetcher(asset: str, interval: str, start_ts: Optional[int], end_ts: Optional[int]) -> List[Dict]:
    if not yf or not pd:
        raise RuntimeError("yfinance/pandas no instalados. Instala `pip install yfinance pandas` para datos de acciones.")
    yf_interval = {"5m":"5m","15m":"15m","30m":"30m","1h":"60m","1d":"1d"}.get(interval, "1d")
    start = None
    end = None
    if start_ts:
        start = pd.to_datetime(int(start_ts), unit='s', utc=True)
    if end_ts:
        end = pd.to_datetime(int(end_ts), unit='s', utc=True)
    ticker = yf.Ticker(asset)
    try:
        if start is not None:
            df = ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=False, prepost=False)
        else:
            df = ticker.history(period="2y", interval=yf_interval, auto_adjust=False, prepost=False)
    except Exception as e:
        log.exception("Error yfinance history: %s", e)
        raise
    out = []
    if df is None or df.empty:
        return out
    df = df.reset_index()
    for _, row in df.iterrows():
        # pandas name may be 'Datetime' or 'Date'
        if 'Datetime' in row and not pd.isna(row['Datetime']):
            ts = int(pd.Timestamp(row['Datetime']).timestamp())
        elif 'Date' in row and not pd.isna(row['Date']):
            ts = int(pd.Timestamp(row['Date']).timestamp())
        else:
            continue
        out.append({
            "asset": asset,
            "interval": interval,
            "ts": int(ts),
            "open": float(row['Open']), "high": float(row['High']),
            "low": float(row['Low']), "close": float(row['Close']),
            "volume": float(row.get('Volume', 0.0))
        })
    log.info("yfinance_fetcher fetched %d candles for %s %s", len(out), asset, interval)
    return out

def get_candles(asset: str, interval: str, start_ts: Optional[int]=None, end_ts: Optional[int]=None, provider_preference: Optional[str]=None) -> List[Dict]:
    is_crypto = asset.upper().endswith("USDT") or "/" in asset
    if provider_preference == "binance" or (is_crypto and ccxt):
        return binance_fetcher(asset, interval, start_ts, end_ts)
    if provider_preference == "yfinance" or (not is_crypto and yf):
        return yfinance_fetcher(asset, interval, start_ts, end_ts)
    errors = []
    if ccxt:
        try:
            return binance_fetcher(asset, interval, start_ts, end_ts)
        except Exception as e:
            errors.append(str(e))
    if yf:
        try:
            return yfinance_fetcher(asset, interval, start_ts, end_ts)
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError("No hay proveedor disponible para get_candles. Errores: " + " || ".join(errors))

def run_backfill(asset: str, interval: str, storage_adapter, start_ts: Optional[int]=None, end_ts: Optional[int]=None, provider_preference: Optional[str]=None):
    def fetcher(a, i, s, e):
        return get_candles(asset, interval, start_ts=s or start_ts, end_ts=e or end_ts, provider_preference=provider_preference)
    return storage_adapter.backfill_symbol(asset, interval, fetcher, start_ts=start_ts, end_ts=end_ts)
