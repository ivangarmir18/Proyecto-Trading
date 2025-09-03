"""
core/fetch.py - Versión mejorada con soporte completo para Finnhub
Características:
- Soporte para Binance (criptos) y Finnhub (acciones)
- Rotación automática de API keys para Finnhub
- Mecanismos de respaldo (fallback) robustos
- Compatibilidad total con el sistema existente
"""

from __future__ import annotations
import time
import math
import logging
import os
import random
from typing import Optional, Callable, List, Dict, Any, Tuple
from datetime import datetime, timezone
from enum import Enum

import ccxt
import pandas as pd
import requests
from dateutil import parser
from tqdm import tqdm

# Optional imports
try:
    import yfinance as yf
except Exception:
    yf = None

logger = logging.getLogger("core.fetch")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.environ.get("FETCH_LOG_LEVEL", "INFO"))

# Mapeo de intervalos
INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d", "1w": "1w"
}

# Typing for save callback
SaveCallback = Callable[[pd.DataFrame, str, str, Dict[str, Any]], None]

class DataSource(Enum):
    BINANCE = "binance"
    FINNHUB = "finnhub"
    YFINANCE = "yfinance"
    UNKNOWN = "unknown"

class RateLimiter:
    """Token-bucket style rate limiter"""
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
        """Block until at least one token is available"""
        while not self.consume(1):
            time.sleep(0.05)

def full_jitter_sleep(base: float, cap: float, attempt: int):
    """Exponential backoff with full jitter"""
    exp = min(cap, base * (2 ** (attempt - 1)))
    sleep = random.uniform(0, exp)
    logger.debug("Backoff sleep for %.3fs (attempt %d)", sleep, attempt)
    time.sleep(sleep)

class FinnhubKeyManager:
    """Gestiona rotación de API keys para Finnhub"""
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("Se requieren API keys para Finnhub")
        self.keys = keys
        self.key_usage = {key: 0 for key in keys}
        self.key_limits = {key: 60 for key in keys}  # Límite por minuto por key
        self.last_reset = time.time()
        
    def get_key(self):
        """Obtiene la mejor key disponible basado en uso reciente"""
        current_time = time.time()
        # Resetear contadores cada minuto
        if current_time - self.last_reset > 60:
            self.key_usage = {key: 0 for key in self.keys}
            self.last_reset = current_time
            
        # Encontrar key con menor uso
        available_keys = [key for key in self.keys 
                         if self.key_usage[key] < self.key_limits[key]]
        if not available_keys:
            # Todas las keys alcanzaron su límite, esperar reset
            sleep_time = 60 - (current_time - self.last_reset)
            if sleep_time > 0:
                time.sleep(sleep_time + 1)
                return self.get_key()
            
        best_key = min(available_keys, key=lambda k: self.key_usage[k])
        self.key_usage[best_key] += 1
        return best_key

class Fetcher:
    def __init__(
        self,
        exchange_name: str = "binance",
        binance_api_key: Optional[str] = None,
        binance_secret: Optional[str] = None,
        finnhub_keys: Optional[List[str]] = None,
        rate_limit_per_min: Optional[int] = None,
        default_limit: int = 500,
        max_attempts: int = 6,
        backoff_base: float = 1.0,
        backoff_cap: float = 60.0,
    ):
        self.exchange_name = exchange_name
        self.binance_api_key = binance_api_key or os.getenv("ENV_BINANCE_KEY")
        self.binance_secret = binance_secret or os.getenv("ENV_BINANCE_SECRET")
        self.default_limit = default_limit
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap

        # Configurar Finnhub
        self.finnhub_keys = finnhub_keys or []
        if not self.finnhub_keys:
            # Intentar cargar keys de environment variables
            finnhub_keys_env = os.getenv("FINNHUB_KEYS", "")
            if finnhub_keys_env:
                self.finnhub_keys = [k.strip() for k in finnhub_keys_env.split(",") if k.strip()]
            else:
                # Buscar keys individuales
                for i in range(1, 6):
                    key = os.getenv(f"FINNHUB_KEY_{i}")
                    if key:
                        self.finnhub_keys.append(key)
        
        if self.finnhub_keys:
            self.finnhub_key_manager = FinnhubKeyManager(self.finnhub_keys)
        else:
            self.finnhub_key_manager = None

        # Rate limiter
        if rate_limit_per_min:
            self.rate_limiter = RateLimiter(rate=rate_limit_per_min, per_seconds=60)
        else:
            self.rate_limiter = None

        # Inicializar exchange
        self._exchange = None
        self._init_exchange()

    def _init_exchange(self):
        try:
            if self.exchange_name.lower() == "binance":
                exchange = ccxt.binance({
                    "enableRateLimit": True,
                    "apiKey": self.binance_api_key,
                    "secret": self.binance_secret,
                })
                self._exchange = exchange
                logger.info("Exchange binance inicializado")
            else:
                exchange_cls = getattr(ccxt, self.exchange_name, None)
                if exchange_cls:
                    self._exchange = exchange_cls({"enableRateLimit": True})
                    logger.info("Exchange %s inicializado", self.exchange_name)
                else:
                    logger.warning("Exchange %s no encontrado", self.exchange_name)
                    self._exchange = ccxt.Exchange({})
        except Exception as e:
            logger.exception("Error inicializando exchange: %s", e)
            self._exchange = None

    def _ensure_rate_limit(self):
        if self.rate_limiter:
            self.rate_limiter.wait_for_token()
        else:
            time.sleep(0.01)

    @staticmethod
    def normalize_symbol(asset: str) -> str:
        if "/" in asset:
            return asset
        for suf in ("USDT", "BUSD", "USD", "EUR", "BTC"):
            if asset.endswith(suf):
                return asset[:-len(suf)] + "/" + suf
        if len(asset) > 6:
            return asset[:-4] + "/" + asset[-4:]
        return asset

    def _ccxt_symbol(self, asset: str) -> str:
        return self.normalize_symbol(asset)

    def _df_from_ccxt(self, ohlcv: List[List[Any]]) -> pd.DataFrame:
        df = pd.DataFrame(ohlcv, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df[["ts", "open", "high", "low", "close", "volume"]]

    def _fetch_finnhub_ohlcv(self, symbol: str, resolution: str = "5", 
                           from_ts: Optional[int] = None, to_ts: Optional[int] = None,
                           count: Optional[int] = None) -> pd.DataFrame:
        """Fetch OHLCV data from Finnhub API"""
        if not self.finnhub_key_manager:
            raise RuntimeError("No Finnhub API keys configured")
        
        api_key = self.finnhub_key_manager.get_key()
        url = "https://finnhub.io/api/v1/stock/candle"
        
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "token": api_key
        }
        
        if from_ts and to_ts:
            params["from"] = from_ts
            params["to"] = to_ts
        elif count:
            params["count"] = count
        else:
            # Por defecto, últimos 100 candles
            params["count"] = 100
        
        attempt = 0
        while attempt < self.max_attempts:
            try:
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 429:
                    # Rate limit exceeded
                    attempt += 1
                    sleep_time = 2 ** attempt
                    logger.warning("Finnhub rate limit exceeded, sleeping %s seconds", sleep_time)
                    time.sleep(sleep_time)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                if data["s"] != "ok":
                    raise ValueError(f"Finnhub API error: {data.get('s', 'unknown')}")
                
                # Convertir a DataFrame
                df = pd.DataFrame({
                    "ts": data["t"],
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                    "volume": data["v"]
                })
                
                df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
                return df[["ts", "open", "high", "low", "close", "volume"]]
                
            except requests.exceptions.RequestException as e:
                attempt += 1
                logger.warning("Error fetching from Finnhub (attempt %d/%d): %s", 
                             attempt, self.max_attempts, e)
                if attempt >= self.max_attempts:
                    raise
                full_jitter_sleep(self.backoff_base, self.backoff_cap, attempt)
        
        raise RuntimeError(f"Failed to fetch data from Finnhub after {self.max_attempts} attempts")

    def _determine_data_source(self, asset: str) -> DataSource:
        """Determina la mejor fuente de datos para un asset"""
        # Verificar si es cripto (termina con USDT, BTC, etc.)
        if any(asset.upper().endswith(x) for x in ("USDT", "BUSD", "BTC", "ETH", "USD")):
            return DataSource.BINANCE
        
        # Verificar si tenemos keys de Finnhub
        if self.finnhub_key_manager:
            return DataSource.FINNHUB
        
        # Fallback a yfinance
        if yf is not None:
            return DataSource.YFINANCE
            
        return DataSource.UNKNOWN

    def fetch_ohlcv(
        self,
        asset: str,
        interval: str = "1h",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        save_callback: Optional[SaveCallback] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        tf = INTERVAL_MAP.get(interval, interval)
        limit = limit or self.default_limit
        source = self._determine_data_source(asset)

        attempt = 0
        last_exception = None

        while attempt < self.max_attempts:
            try:
                self._ensure_rate_limit()
                df = None
                
                if source == DataSource.BINANCE:
                    symbol_ccxt = self._ccxt_symbol(asset)
                    if not self._exchange:
                        self._init_exchange()
                    
                    ohlcv = self._exchange.fetch_ohlcv(symbol_ccxt, timeframe=tf, 
                                                     since=since, limit=limit)
                    df = self._df_from_ccxt(ohlcv)
                    meta = meta or {}
                    meta["source"] = "binance"
                    
                elif source == DataSource.FINNHUB:
                    # Finnhub usa resoluciones diferentes
                    resolution_map = {
                        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
                        "1h": "60", "1d": "D", "1w": "W"
                    }
                    resolution = resolution_map.get(tf, "5")
                    
                    # Convertir timestamps
                    from_ts = since // 1000 if since else None
                    to_ts = int(time.time()) if since else None
                    
                    df = self._fetch_finnhub_ohlcv(asset, resolution, from_ts, to_ts, limit)
                    meta = meta or {}
                    meta["source"] = "finnhub"
                    
                elif source == DataSource.YFINANCE and yf is not None:
                    yf_symbol = asset.replace("/", "-")
                    yf_interval = {"1d": "1d", "1h": "60m", "5m": "5m", 
                                 "30m": "30m", "1m": "1m"}.get(interval, "1d")
                    
                    period = "max" if since else "1y"
                    hist = yf.download(tickers=yf_symbol, interval=yf_interval, 
                                     period=period, progress=False, threads=False)
                    
                    if hist is None or hist.empty:
                        raise RuntimeError("yfinance devolvió DataFrame vacío")
                    
                    df = hist.reset_index().rename(columns={
                        "Datetime": "ts", "Date": "ts", "Open": "open", 
                        "High": "high", "Low": "low", "Close": "close", 
                        "Volume": "volume"
                    })
                    
                    if isinstance(df.loc[0, "ts"], pd.Timestamp):
                        df["ts"] = pd.to_datetime(df["ts"], utc=True)
                    else:
                        df["ts"] = pd.to_datetime(df["ts"], utc=True)
                    
                    df = df[["ts", "open", "high", "low", "close", "volume"]]
                    meta = meta or {}
                    meta["source"] = "yfinance"
                
                else:
                    raise RuntimeError(f"No hay fuente de datos disponible para {asset}")
                
                if save_callback and df is not None:
                    try:
                        save_callback(df.copy(), asset, interval, meta or {})
                    except Exception:
                        logger.exception("Error en save_callback para %s %s", asset, interval)
                
                return df

            except Exception as e:
                last_exception = e
                logger.warning("Error fetching %s %s (attempt %d/%d): %s", 
                             asset, interval, attempt + 1, self.max_attempts, e)
                attempt += 1
                full_jitter_sleep(self.backoff_base, self.backoff_cap, attempt)
        
        raise RuntimeError(f"Failed to fetch OHLCV for {asset} {interval}: {last_exception}")

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
        per_call_limit = per_call_limit or self.default_limit
        source = self._determine_data_source(asset)

        # Para Finnhub, podemos pedir más datos de una vez
        if source == DataSource.FINNHUB:
            per_call_limit = min(per_call_limit * 2, 1000)  # Finnhub permite hasta 1000 candles

        # Calcular número aproximado de velas
        interval_seconds = parse_interval_to_seconds(interval)
        ms_per_bar = interval_seconds * 1000
        total_bars = max(1, math.ceil((end_ts_ms - start_ts_ms) / ms_per_bar))
        calls = max(1, math.ceil(total_bars / per_call_limit))

        logger.info("Backfilling %s %s from %s to %s -> ~%d bars in %d calls",
                    asset, interval,
                    datetime.fromtimestamp(start_ts_ms / 1000, tz=timezone.utc).isoformat(),
                    datetime.fromtimestamp(end_ts_ms / 1000, tz=timezone.utc).isoformat(),
                    total_bars, calls)

        results = []
        current_since = start_ts_ms

        iterator = range(calls)
        if progress:
            iterator = tqdm(iterator, desc=f"Backfill {asset} {interval}", unit="call")

        for i in iterator:
            try:
                df_block = self.fetch_ohlcv(asset, interval=interval, since=current_since,
                                          limit=per_call_limit, save_callback=save_callback,
                                          meta={"chunk_index": i, "total_chunks": calls})
                
                if df_block is None or df_block.empty:
                    logger.debug("Bloque %d vacío, terminando", i)
                    break
                    
                results.append(df_block)
                
                # Avanzar al siguiente timestamp
                last_ts = df_block["ts"].iloc[-1]
                if isinstance(last_ts, pd.Timestamp):
                    last_ts_ms = int(last_ts.timestamp() * 1000)
                else:
                    last_ts_ms = int(last_ts)
                    
                if last_ts_ms >= current_since:
                    current_since = last_ts_ms + 1
                else:
                    logger.warning("Timestamp no avanzó, terminando")
                    break
                    
                if current_since > end_ts_ms:
                    break
                    
            except Exception as e:
                logger.exception("Error en chunk %d para %s: %s", i, asset, e)
                break

        if not results:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            
        df_all = pd.concat(results, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        
        # Filtrar por rango exacto
        df_all["ts_ms"] = (df_all["ts"].astype("int64") // 1_000_000).astype("int64")
        df_all = df_all[(df_all["ts_ms"] >= start_ts_ms) & (df_all["ts_ms"] <= end_ts_ms)]
        return df_all.drop(columns=["ts_ms"])

    # Métodos utilitarios
    @staticmethod
    def ms_from_iso(iso_ts: str) -> int:
        dt = parser.isoparse(iso_ts)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)

# Función de utilidad para parsear intervalos (debe estar definida en utils)
def parse_interval_to_seconds(interval: str) -> int:
    """Convierte intervalos como '5m' a segundos"""
    if not interval:
        raise ValueError("Intervalo vacío")
    
    s = str(interval).strip().lower()
    num = ''.join(filter(str.isdigit, s))
    unit = ''.join(filter(str.isalpha, s))
    
    if not num:
        num = "1"
    
    unit_seconds = {
        "m": 60, "min": 60, "h": 3600, "d": 86400, "w": 604800
    }.get(unit, 60)
    
    return int(num) * unit_seconds