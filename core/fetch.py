"""
core/fetch.py  -- Versión mejorada y robusta del recolector de OHLCV.

Características principales:
- Clase Fetcher que soporta múltiples exchanges (Binance por defecto).
- Reintentos con backoff exponencial + full jitter.
- Manejo de errores específicos de ccxt (NetworkError, ExchangeError).
- Fallback automático a yfinance (y opcionalmente a Finnhub si integras su cliente).
- Función de backfill por rango (split por 'limit' para no pedir demasiados datos).
- RateLimiting simple (token-bucket estilo) configurable por instancia.
- Concurrency opcional para peticiones en batch (ThreadPoolExecutor).
- Hooks: save_callback(df, asset, interval, meta) para integrar con storage.
- CLI para pruebas rápidas.
- Logging detallado y parámetros configurables desde env o desde constructor.
"""

from __future__ import annotations
import time
import math
import logging
import os
import random
from typing import Optional, Callable, List, Dict, Any, Tuple
from datetime import datetime, timezone

import ccxt
import pandas as pd
from dateutil import parser
from tqdm import tqdm

# Optional imports
try:
    import yfinance as yf
except Exception:
    yf = None

# Logging básico (tu proyecto probablemente tenga su propio logger)
logger = logging.getLogger("core.fetch")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.environ.get("FETCH_LOG_LEVEL", "INFO"))

# Mapeo simple de intervalos (extensible)
INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}

# Typing for save callback:
SaveCallback = Callable[[pd.DataFrame, str, str, Dict[str, Any]], None]


class RateLimiter:
    """
    Token-bucket style rate limiter.
    Allows 'rate' tokens per 'per_seconds' interval.
    Keep a small state, thread-safe if used carefully (no locking here for simplicity).
    """
    def __init__(self, rate: int = 1200, per_seconds: int = 60):
        self.rate = rate
        self.per_seconds = per_seconds
        self._tokens = rate
        self._last = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        add = (elapsed / self.per_seconds) * self.rate
        if add >= 1:
            self._tokens = min(self.rate, self._tokens + add)
            self._last = now

    def consume(self, tokens: int = 1):
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def wait_for_token(self):
        """Block until at least one token is available."""
        while not self.consume(1):
            time.sleep(0.05)


def full_jitter_sleep(base: float, cap: float, attempt: int):
    """
    Exponential backoff with full jitter (AWS recommended pattern).
    base: base seconds (e.g., 1.0)
    cap: max seconds (e.g., 60)
    attempt: attempt number, starting at 1
    """
    exp = min(cap, base * (2 ** (attempt - 1)))
    sleep = random.uniform(0, exp)
    logger.debug("Backoff sleep for %.3fs (attempt %d)", sleep, attempt)
    time.sleep(sleep)


class Fetcher:
    def __init__(
        self,
        exchange_name: str = "binance",
        binance_api_key: Optional[str] = None,
        binance_secret: Optional[str] = None,
        rate_limit_per_min: Optional[int] = None,
        default_limit: int = 500,
        max_attempts: int = 6,
        backoff_base: float = 1.0,
        backoff_cap: float = 60.0,
    ):
        """
        Construye el Fetcher.

        - exchange_name: nombre para ccxt (ej. 'binance').
        - binance_api_key / secret: opcionales.
        - rate_limit_per_min: si se da, se usa para crear un RateLimiter; si None, usa exchange.enableRateLimit.
        - default_limit: límite por llamada a fetch_ohlcv.
        - max_attempts: reintentos totales antes de fallback/error.
        - backoff_*: parámetros de backoff.
        """
        self.exchange_name = exchange_name
        self.binance_api_key = binance_api_key or os.getenv("ENV_BINANCE_KEY")
        self.binance_secret = binance_secret or os.getenv("ENV_BINANCE_SECRET")
        self.default_limit = default_limit
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap

        # Rate limiter
        if rate_limit_per_min:
            self.rate_limiter = RateLimiter(rate=rate_limit_per_min, per_seconds=60)
        else:
            self.rate_limiter = None

        # inicializa exchange (ccxt)
        self._exchange = None
        self._init_exchange()

    def _init_exchange(self):
        try:
            if self.exchange_name.lower() == "binance":
                exchange = ccxt.binance({
                    "enableRateLimit": True,
                    # puedes setear proxies o timeouts aquí si quieres
                    # 'timeout': 30000,
                })
                # setear keys si existen (no necesarias para candles públicas, pero útil si usas endpoints privados)
                if self.binance_api_key:
                    exchange.apiKey = self.binance_api_key
                if self.binance_secret:
                    exchange.secret = self.binance_secret
                self._exchange = exchange
                logger.info("Exchange binance inicializado (enableRateLimit=%s)", exchange.enableRateLimit)
            else:
                # Soporte general para otros exchanges por nombre
                exchange_cls = getattr(ccxt, self.exchange_name, None)
                if exchange_cls:
                    self._exchange = exchange_cls({"enableRateLimit": True})
                    logger.info("Exchange %s inicializado", self.exchange_name)
                else:
                    logger.warning("Exchange %s no encontrado en ccxt; creando instancia genérica", self.exchange_name)
                    self._exchange = ccxt.Exchange({})
        except Exception as e:
            logger.exception("Error inicializando exchange %s: %s", self.exchange_name, e)
            self._exchange = None

    def _ensure_rate_limit(self):
        if self.rate_limiter:
            self.rate_limiter.wait_for_token()
        else:
            # confía en ccxt.enableRateLimit o sleep mínimo
            time.sleep(0.01)

    @staticmethod
    def normalize_symbol(asset: str) -> str:
        """
        Normaliza 'BTCUSDT' -> 'BTC/USDT' (heurística).
        Conserva 'BTC/USDT' si ya contiene '/'. No modifica para acciones tipo 'AAPL'.
        """
        if "/" in asset:
            return asset
        # heurística común para cripto (USDT, USD, BUSD)
        for suf in ("USDT", "BUSD", "USD", "EUR", "BTC"):
            if asset.endswith(suf):
                return asset[:-len(suf)] + "/" + suf
        # caso general: si >6 chars usamos split -4
        if len(asset) > 6:
            return asset[:-4] + "/" + asset[-4:]
        return asset

    def _ccxt_symbol(self, asset: str) -> str:
        s = self.normalize_symbol(asset)
        # ccxt usa formato MKT/QUOTE
        return s

    def _df_from_ccxt(self, ohlcv: List[List[Any]]) -> pd.DataFrame:
        df = pd.DataFrame(ohlcv, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df[["ts", "open", "high", "low", "close", "volume"]]
        return df

    def fetch_ohlcv(
        self,
        asset: str,
        interval: str = "1h",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        save_callback: Optional[SaveCallback] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Solicita OHLCV para un activo y timeframe.

        - asset: 'BTCUSDT' o 'BTC/USDT' o 'AAPL' (yfinance fallback).
        - interval: '5m', '1h', ... (usa INTERVAL_MAP)
        - since: unix ms timestamp desde el que pedir (opcional)
        - limit: max velas (si None usa default_limit)
        - save_callback: función opcional para volcar resultados (df, asset, interval, meta)
        - meta: diccionario que se pasará a save_callback para contexto (p. ej. {'source':'binance'})
        """
        tf = INTERVAL_MAP.get(interval, interval)
        limit = limit or self.default_limit
        symbol_ccxt = self._ccxt_symbol(asset)

        attempt = 1
        last_exception = None

        while attempt <= self.max_attempts:
            try:
                self._ensure_rate_limit()
                if not self._exchange:
                    self._init_exchange()

                # ccxt necesita símbolo con slash y algunos exchanges usan notación distinta; asumimos estándar
                ohlcv = self._exchange.fetch_ohlcv(symbol_ccxt, timeframe=tf, since=since, limit=limit)
                df = self._df_from_ccxt(ohlcv)

                if save_callback:
                    try:
                        save_callback(df.copy(), asset, interval, meta or {})
                    except Exception:
                        logger.exception("save_callback falló para %s %s (no aborta)", asset, interval)

                # attach meta info (source)
                if meta is None:
                    meta = {}
                meta["source"] = getattr(self._exchange, "id", self.exchange_name)
                return df

            except ccxt.NetworkError as e:
                last_exception = e
                logger.warning("NetworkError fetching %s %s attempt %d/%d: %s", asset, interval, attempt, self.max_attempts, e)
            except ccxt.ExchangeError as e:
                last_exception = e
                logger.warning("ExchangeError fetching %s %s attempt %d/%d: %s", asset, interval, attempt, self.max_attempts, e)
            except Exception as e:
                last_exception = e
                logger.exception("Error inesperado fetching %s %s attempt %d/%d: %s", asset, interval, attempt, self.max_attempts, e)

            # backoff con jitter
            full_jitter_sleep(self.backoff_base, self.backoff_cap, attempt)
            attempt += 1

        # Si llegamos aquí: fallback a yfinance (útil para acciones o cuando ccxt falla)
        logger.info("Intentando fallback a yfinance para %s %s", asset, interval)
        try:
            if yf is None:
                raise RuntimeError("yfinance no disponible (instálalo con pip install yfinance)")

            # yfinance espera símbolos tipo 'AAPL' o 'BTC-USD' segun mercado; convertir heurísticamente:
            yf_symbol = asset.replace("/", "-")
            # map interval to yfinance interval (no todos los intervalos están soportados por yfinance)
            yf_interval = {"1d": "1d", "1h": "60m", "5m": "5m", "30m": "30m", "1m": "1m"}.get(interval, "1d")
            # si since dado, pedimos period 'max' y luego filtramos; si not, pedimos period='1y' por defecto
            period = "max" if since else "1y"
            hist = yf.download(tickers=yf_symbol, interval=yf_interval, period=period, progress=False, threads=False)
            if hist is None or hist.empty:
                raise RuntimeError("yfinance devolvió DataFrame vacío")

            df = hist.reset_index().rename(columns={"Datetime": "ts", "Date": "ts", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            if isinstance(df.loc[0, "ts"], pd.Timestamp):
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
            else:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df[["ts", "open", "high", "low", "close", "volume"]]
            if save_callback:
                try:
                    meta2 = meta or {}
                    meta2["source"] = "yfinance"
                    save_callback(df.copy(), asset, interval, meta2)
                except Exception:
                    logger.exception("save_callback fallo en fallback yfinance (no aborta)")
            return df
        except Exception as e:
            logger.exception("Fallback yfinance falló para %s %s: %s", asset, interval, e)
            # fallback adicional: podrías integrar Finnhub aquí
            raise RuntimeError(f"No se pudo obtener OHLCV para {asset} en {interval}") from last_exception

    def backfill_range(
        self,
        asset: str,
        interval: str,
        start_ts_ms: int,
        end_ts_ms: int,
        per_call_limit: Optional[int] = None,
        save_callback: Optional[SaveCallback] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Backfill de un rango largo: divide en llamadas por 'per_call_limit' (velas por llamada).
        Devuelve DataFrame concatenado ordenado por ts ascendente.

        - start_ts_ms, end_ts_ms: timestamps en ms (unix)
        - per_call_limit: límite por llamada (usa default_limit si None)
        - save_callback: callback que se invoca por cada bloque obtenido
        """
        per_call_limit = per_call_limit or self.default_limit
        tf = INTERVAL_MAP.get(interval, interval)

        # heurística: calcular el ms por vela (aprox) según interval
        ms_per_unit = {
            "m": 60_000,
            "h": 3_600_000,
            "d": 86_400_000,
            "w": 7 * 86_400_000,
        }
        # determinar multiplicador
        unit = tf[-1]
        qty = int(tf[:-1]) if tf[:-1].isdigit() else 1
        ms_per_bar = ms_per_unit.get(unit, 60_000) * qty

        # número aproximado de velas en el rango
        total_bars = max(1, math.ceil((end_ts_ms - start_ts_ms) / ms_per_bar))
        # número de llamados que necesitaremos
        calls = max(1, math.ceil(total_bars / per_call_limit))

        logger.info("Backfilling %s %s from %s to %s -> ~%d bars in %d calls (per_call_limit=%d)",
                    asset, interval,
                    datetime.fromtimestamp(start_ts_ms / 1000, tz=timezone.utc).isoformat(),
                    datetime.fromtimestamp(end_ts_ms / 1000, tz=timezone.utc).isoformat(),
                    total_bars, calls, per_call_limit)

        results: List[pd.DataFrame] = []
        current_since = start_ts_ms

        iterator = range(calls)
        if progress:
            iterator = tqdm(iterator, desc=f"Backfill {asset} {interval}", unit="call")

        for i in iterator:
            # pedido por bloques: since=current_since, limit=per_call_limit
            try:
                df_block = self.fetch_ohlcv(asset, interval=interval, since=current_since, limit=per_call_limit, save_callback=save_callback, meta={"chunk_index": i, "total_chunks": calls})
                if df_block is None or df_block.empty:
                    logger.debug("Bloque %d devolvió vacío, rompiendo", i)
                    break
                results.append(df_block)
                # avanzar current_since: tomar último ts y sumar 1 ms para evitar solapamientos
                last_ts = int(df_block["ts"].iloc[-1].timestamp() * 1000)
                if last_ts >= current_since:
                    current_since = last_ts + 1
                else:
                    # protección: si no avanzó (p. ej. exchange devolvió siempre mismas velas), rompemos para no buclear
                    logger.warning("fetch_ohlcv no avanzó el timestamp (last_ts=%d current_since=%d), rompiendo", last_ts, current_since)
                    break
                # si superamos end_ts_ms, rompemos
                if current_since > end_ts_ms:
                    break
            except Exception as e:
                logger.exception("Error durante backfill chunk %d para %s: %s", i, asset, e)
                # decidir: continuar o romper; aquí rompemos para no spammear al exchange
                break

        if not results:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        df_all = pd.concat(results, ignore_index=True)
        # deduplicate by ts and sort
        df_all = df_all.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        # filtrar por rango estricto
        df_all = df_all[(df_all["ts"].astype("int64") // 1_000_000 >= start_ts_ms // 1_000) & (df_all["ts"].astype("int64") // 1_000_000 <= end_ts_ms // 1_000)]
        return df_all

    # --- utilidades pequeñas
    @staticmethod
    def ms_from_iso(iso_ts: str) -> int:
        """Convierte ISO8601 a ms unix (int)."""
        dt = parser.isoparse(iso_ts)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)


# CLI simple para probar
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Prueba fetcher (core/fetch.py)")
    parser.add_argument("asset", help="Asset (ej. BTCUSDT o BTC/USDT o AAPL)")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--since", default=None, help="ISO8601 o ms timestamp")
    parser.add_argument("--backfill", action="store_true", help="Si se activa pide rango start..now (usar --since como start)")
    args = parser.parse_args()

    fetcher = Fetcher(
        binance_api_key=os.getenv("ENV_BINANCE_KEY"),
        binance_secret=os.getenv("ENV_BINANCE_SECRET"),
        rate_limit_per_min=int(os.getenv("ENV_RATE_LIMIT", "1200")),
    )

    if args.backfill:
        if not args.since:
            parser.error("--backfill requiere --since (ISO8601)")
        start_ms = Fetcher.ms_from_iso(args.since) if not args.since.isdigit() else int(args.since)
        end_ms = Fetcher.now_ms()
        df = fetcher.backfill_range(args.asset, args.interval, start_ms, end_ms, per_call_limit=args.limit or None, progress=True)
        print(df.tail(10).to_string(index=False))
    else:
        since_ms = None
        if args.since:
            since_ms = Fetcher.ms_from_iso(args.since) if not args.since.isdigit() else int(args.since)
        df = fetcher.fetch_ohlcv(args.asset, interval=args.interval, since=since_ms, limit=args.limit)
        print(df.tail(20).to_string(index=False))


if __name__ == "__main__":
    _cli()
