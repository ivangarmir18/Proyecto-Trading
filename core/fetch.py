# core/fetch.py
"""
Fetcher robusto para Proyecto-Trading.

Objetivos:
- Proveer `fetch_ohlcv(symbol, interval, start_ms=None, end_ms=None)` -> pd.DataFrame
  con columnas: ts (ms, int), open, high, low, close, volume
- `backfill_range(symbol, interval, start_ms, end_ms, callback=None)` que itera por ventanas y llama a save/ callback.
- Soporte multi-fuente: Binance (ccxt) para cripto, Finnhub (API) o yfinance para acciones.
- Rotación de keys para Finnhub, control de rate limit simple, retries y backoff.
- Normalización de símbolos (ej: BTCUSDT -> BTC/USDT para ccxt).
- Utilidades: ms_from_iso, now_ms.
- Diseño: configurable por dict `config` o leyendo config.json si existe.

Nota: el código hace uso de ccxt, yfinance y requests. Asegúrate de instalar:
    pip install ccxt yfinance requests pandas numpy

"""
from __future__ import annotations

import json
import logging
import os
import time
import math
from typing import Optional, Dict, Any, List, Callable, Tuple

import pandas as pd
import numpy as np
import requests

# opcionales: ccxt y yfinance
try:
    import ccxt
except Exception:
    ccxt = None

try:
    import yfinance as yf
except Exception:
    yf = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Default interval mapping for ccxt/timeframes (común)
_CCXT_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}

# Helper util
def now_ms() -> int:
    return int(time.time() * 1000)


def ms_from_iso(iso: str) -> int:
    """Convierte 'YYYY-mm-dd' o timestamp iso a ms"""
    try:
        return int(pd.to_datetime(iso, utc=True).value // 10 ** 6)
    except Exception:
        raise


class RateLimiter:
    """
    Rate limiter muy simple: permite N requests por ventana de segundos.
    No es extremadamente preciso, pero evita picos brutales.
    """

    def __init__(self, max_requests: int = 100, per_seconds: int = 60):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._timestamps: List[float] = []

    def wait(self):
        now = time.time()
        window_start = now - self.per_seconds
        # limpiar timestamps viejos
        self._timestamps = [t for t in self._timestamps if t >= window_start]
        if len(self._timestamps) >= self.max_requests:
            earliest = self._timestamps[0]
            sleep_for = (earliest + self.per_seconds) - now
            if sleep_for > 0:
                logger.debug("RateLimiter sleeping %.2fs", sleep_for)
                time.sleep(sleep_for)
        self._timestamps.append(time.time())


class FinnhubKeyManager:
    """
    Rotación simple de keys para Finnhub (lista). Devuelve key round-robin.
    """

    def __init__(self, keys: Optional[List[str]] = None):
        self.keys = keys or []
        self._i = 0

    def next_key(self) -> Optional[str]:
        if not self.keys:
            return None
        key = self.keys[self._i % len(self.keys)]
        self._i += 1
        return key


class Fetcher:
    """
    Clase principal.

    Constructor:
        Fetcher(storage=None, config=None)

    config: dict opcional con estructura muy simple:
        {
            "api": {
                "binance": {"api_key": "", "api_secret": "", "rate_limit_per_min": 1200},
                "finnhub": {"keys": ["k1", "k2"], "rate_limit_per_min": 60}
            },
            "default_data_source": "binance"  # o "finnhub" o "yfinance"
        }

    Métodos principales:
        fetch_ohlcv(symbol, interval, start_ms=None, end_ms=None) -> pd.DataFrame
        backfill_range(symbol, interval, start_ms, end_ms, callback=None)
        now_ms(), ms_from_iso()
    """

    def __init__(self, storage: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        self.storage = storage
        self.config = config or {}
        # try load config.json from repo root if present and config is empty
        if not config:
            cfg_path = os.getenv("PROJECT_CONFIG_PATH", "config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg_json = json.load(f)
                        self.config.update(cfg_json)
                except Exception:
                    logger.exception("Error leyendo config.json, ignorando")

        # Setup finnhub key manager
        fin_cfg = self.config.get("api", {}).get("finnhub", {})
        fin_keys = fin_cfg.get("keys") if isinstance(fin_cfg, dict) else None
        self.finnhub_keys = FinnhubKeyManager(fin_keys)

        # Rate limiters
        bin_rate = self.config.get("api", {}).get("binance", {}).get("rate_limit_per_min", 1200)
        fin_rate = self.config.get("api", {}).get("finnhub", {}).get("rate_limit_per_min", 60)
        self.binance_limiter = RateLimiter(max_requests=max(1, bin_rate), per_seconds=60)
        self.finnhub_limiter = RateLimiter(max_requests=max(1, fin_rate), per_seconds=60)

        # Init ccxt binance exchange lazily
        self._ccxt_exchange = None
        self._maybe_init_ccxt()

    # -------------------------
    # Helpers
    # -------------------------
    def _maybe_init_ccxt(self):
        if ccxt is None:
            return
        if self._ccxt_exchange is not None:
            return
        bin_cfg = self.config.get("api", {}).get("binance", {})
        api_key = bin_cfg.get("api_key") if isinstance(bin_cfg, dict) else None
        api_secret = bin_cfg.get("api_secret") if isinstance(bin_cfg, dict) else None
        try:
            exchange = ccxt.binance({
                "enableRateLimit": True,
                "apiKey": api_key,
                "secret": api_secret,
                "options": {"defaultType": "spot"},
            })
            self._ccxt_exchange = exchange
        except Exception:
            logger.exception("No se pudo inicializar ccxt.binance")

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normaliza símbolos simples (quita espacios, mayúsculas)."""
        return symbol.strip().upper()

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        """Heurística: si acaba en USDT o contiene 'USD' o mezcla mayúsculas sin punto -> crypto."""
        s = symbol.upper()
        if s.endswith("USDT") or s.endswith("BTC") or s.endswith("ETH"):
            return True
        # si contiene slash (p.e. 'BTC/USDT') lo consideramos crypto
        if "/" in s:
            return True
        # simple fallback: si contiene '.' o '-' lo consideramos stock (p.e. IBE.MC)
        if "." in s or "-" in s:
            return False
        # otherwise, assume stock if shorter than 5? (heurística)
        if len(s) <= 5:
            # pero si incluye 'USD' treat as crypto
            if "USD" in s:
                return True
            return True
        return False

    @staticmethod
    def _ccxt_symbol(symbol: str) -> str:
        """Convierte 'BTCUSDT' -> 'BTC/USDT' para ccxt si hace falta."""
        s = symbol.strip().upper()
        if "/" in s:
            return s
        # heurística simple: split last 3 or 4 chars as quote (USDT->4, USD->3)
        # prefer common suffixes
        common_quotes = ["USDT", "BUSD", "USDC", "USD"]
        for q in common_quotes:
            if s.endswith(q):
                base = s[: -len(q)]
                return f"{base}/{q}"
        # fallback: last 3 chars:
        if len(s) > 6:
            return f"{s[:-3]}/{s[-3:]}"
        # as last fallback, return s
        return s

    # -------------------------
    # Fetchers por fuente
    # -------------------------
    def _fetch_binance_ohlcv(self, symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> pd.DataFrame:
        """Usa ccxt para obtener OHLCV desde Binance. Devuelve df con ts (ms) y columnas OHLCV."""
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")

        self._maybe_init_ccxt()
        if self._ccxt_exchange is None:
            raise RuntimeError("ccxt exchange no inicializado (ver config)")

        tf = _CCXT_INTERVALS.get(interval, interval)
        symbol_ccxt = self._ccxt_symbol(symbol)
        limit_per_fetch = 1000  # ccxt límite por llamada (aprox)
        all_rows = []

        # ccxt uses milliseconds since epoch for since param.
        since = int(start_ms) if start_ms else None
        stop_at = int(end_ms) if end_ms else None
        self.binance_limiter.wait()

        # fetch iteratively
        while True:
            try:
                ohlcv = self._ccxt_exchange.fetch_ohlcv(symbol_ccxt, timeframe=tf, since=since, limit=limit_per_fetch)
            except ccxt.NetworkError as e:
                logger.warning("ccxt network error: %s (retry)", e)
                time.sleep(1)
                continue
            except ccxt.ExchangeError as e:
                logger.error("ccxt exchange error: %s", e)
                raise

            if not ohlcv:
                break
            # ccxt OHLCV format: [ts, open, high, low, close, volume]
            all_rows.extend(ohlcv)
            # advance since for next page
            last_ts = ohlcv[-1][0]
            # break if we've reached stop_at
            if stop_at and last_ts >= stop_at:
                break
            # next since = last_ts + 1
            since = last_ts + 1
            # safety: if len < limit -> finished
            if len(ohlcv) < limit_per_fetch:
                break
            # small sleep to respect rate limit
            time.sleep(0.2)

        if not all_rows:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
        # ensure numeric and int ms
        df["ts"] = df["ts"].astype("int64")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _fetch_yfinance_ohlcv(self, symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> pd.DataFrame:
        """Usa yfinance para acciones. interval debe ser streaming compatible (1m,5m,1h,1d)."""
        if yf is None:
            raise RuntimeError("yfinance no está instalado")
        # yfinance interval mapping: '1m','2m','5m','15m','30m','60m','90m','1h' etc
        yf_interval = interval
        # yfinance uses period or start/end in datetime strings
        start = pd.to_datetime(start_ms, unit="ms") if start_ms else None
        end = pd.to_datetime(end_ms, unit="ms") if end_ms else None
        try:
            # yfinance returns timezone-aware index
            df = yf.download(tickers=symbol, interval=yf_interval, start=start, end=end, progress=False, threads=False)
            if df is None or df.empty:
                return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.reset_index()
            # index name may be Datetime
            ts = pd.to_datetime(df.iloc[:, 0], utc=True)
            df2 = pd.DataFrame({
                "ts": (ts.view("int64") // 10 ** 6).astype("int64"),
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "volume": df["Volume"].astype(float),
            })
            return df2
        except Exception as e:
            logger.exception("yfinance fetch failed for %s: %s", symbol, e)
            raise

    def _fetch_finnhub_ohlcv(self, symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> pd.DataFrame:
        """
        Usa Finnhub REST API. Documentación: https... (no se consulta aquí)
        Endpoint esperado: /stock/candle?symbol=...&resolution=...&from=...&to=...
        resolution: 1,5,15,30,60,D,W,M
        """
        key = self.finnhub_keys.next_key()
        if not key:
            raise RuntimeError("No Finnhub API key configured")
        # resolution mapping
        res_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
            "1d": "D", "1w": "W", "1M": "M"
        }
        resolution = res_map.get(interval, interval)
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": int(math.floor((start_ms or (now_ms() - 86400 * 1000)) / 1000)),
            "to": int(math.floor((end_ms or now_ms()) / 1000)),
            "token": key,
        }
        self.finnhub_limiter.wait()
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.error("Finnhub error %s: %s", resp.status_code, resp.text)
            raise RuntimeError(f"Finnhub error {resp.status_code}")
        j = resp.json()
        # Finnhub returns: s (status), c (close array), h, l, o, v, t (array of timestamps seconds)
        if j.get("s") != "ok":
            # possible values: no_data
            logger.warning("Finnhub returned non-ok status: %s", j.get("s"))
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        ts = [int(t * 1000) for t in j.get("t", [])]
        df = pd.DataFrame({
            "ts": ts,
            "open": j.get("o", []),
            "high": j.get("h", []),
            "low": j.get("l", []),
            "close": j.get("c", []),
            "volume": j.get("v", []),
        })
        # ensure types
        df["ts"] = df["ts"].astype("int64")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # -------------------------
    # Public API
    # -------------------------
    def _choose_source(self, symbol: str) -> str:
        """
        Lógica para elegir fuente de datos:
         - si es crypto -> binance (ccxt)
         - else -> finnhub (si keys) o yfinance
         - permite override en self.config['default_data_source']
        """
        override = self.config.get("default_data_source")
        if override:
            return override
        s = self.normalize_symbol(symbol)
        if self._is_crypto(s):
            return "binance"
        if self.finnhub_keys.keys:
            return "finnhub"
        # fallback
        return "yfinance" if yf is not None else "binance"

    def fetch_ohlcv(self, symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> pd.DataFrame:
        """
        Devuelve DataFrame con columnas: ts (ms int), open, high, low, close, volume.
        Elige fuente automáticamente si no se especifica. Lanza excepciones claras si falla.
        """
        symbol = self.normalize_symbol(symbol)
        source = self._choose_source(symbol)
        logger.debug("fetch_ohlcv: %s %s from %s", symbol, interval, source)

        # implement retries with exponential backoff
        attempts = 0
        max_attempts = 4
        backoff_base = 0.8

        while True:
            try:
                if source == "binance":
                    return self._fetch_binance_ohlcv(symbol, interval, start_ms=start_ms, end_ms=end_ms)
                elif source == "finnhub":
                    return self._fetch_finnhub_ohlcv(symbol, interval, start_ms=start_ms, end_ms=end_ms)
                elif source == "yfinance":
                    return self._fetch_yfinance_ohlcv(symbol, interval, start_ms=start_ms, end_ms=end_ms)
                else:
                    raise RuntimeError(f"No data source configured for {source}")
            except Exception as e:
                attempts += 1
                logger.warning("fetch_ohlcv failed attempt %d for %s: %s", attempts, symbol, e)
                if attempts >= max_attempts:
                    logger.exception("fetch_ohlcv finally failed for %s %s", symbol, interval)
                    raise
                sleep = backoff_base * (2 ** (attempts - 1)) + (0.1 * attempts)
                time.sleep(sleep)

    def backfill_range(self, symbol: str, interval: str, start_ms: int, end_ms: int, batch_window_ms: Optional[int] = None, callback: Optional[Callable[[pd.DataFrame], None]] = None) -> None:
        """
        Realiza backfill de un rango extenso dividiéndolo en ventanas manejables.
        - batch_window_ms: si None, se calcula en función del interval (p.e. 1000 velas).
        - callback: función que recibe DataFrame con velas y debe guardarlas en storage (p.e. storage.save_candles)
        """
        symbol = self.normalize_symbol(symbol)
        if batch_window_ms is None:
            # heurística (n velas por fetch)
            candles_per_batch = 1000
            # estimar ms por vela según interval
            unit = interval.lower()
            if unit.endswith("m"):
                mins = int(unit[:-1]) if unit[:-1].isdigit() else 1
                ms_per = mins * 60 * 1000
            elif unit.endswith("h"):
                hours = int(unit[:-1]) if unit[:-1].isdigit() else 1
                ms_per = hours * 60 * 60 * 1000
            elif unit.endswith("d"):
                days = int(unit[:-1]) if unit[:-1].isdigit() else 1
                ms_per = days * 24 * 60 * 60 * 1000
            else:
                ms_per = 60 * 1000
            batch_window_ms = candles_per_batch * ms_per

        cursor = int(start_ms)
        end = int(end_ms)
        while cursor < end:
            window_end = min(cursor + batch_window_ms - 1, end)
            try:
                df = self.fetch_ohlcv(symbol, interval, start_ms=cursor, end_ms=window_end)
                if df is None or df.empty:
                    logger.info("No data in window %d-%d for %s", cursor, window_end, symbol)
                else:
                    if callback:
                        callback(df)
                    else:
                        # si storage disponible, intentar guardar con la API save_candles
                        if self.storage and hasattr(self.storage, "save_candles"):
                            try:
                                self.storage.save_candles(df, symbol, interval)
                            except Exception:
                                logger.exception("Error saving candles for %s", symbol)
                # avanzar cursor al final de la ventana + 1 ms
                if df is not None and not df.empty:
                    last_ts = int(df["ts"].max())
                    cursor = last_ts + 1
                else:
                    cursor = window_end + 1
            except Exception:
                logger.exception("Error en backfill window %d-%d for %s", cursor, window_end, symbol)
                # avanzar para evitar bucle infinito (pero con backoff)
                cursor = window_end + 1
                time.sleep(1)

    # alias util
    def ms_from_iso(self, iso: str) -> int:
        return ms_from_iso(iso)

    def now_ms(self) -> int:
        return now_ms()
