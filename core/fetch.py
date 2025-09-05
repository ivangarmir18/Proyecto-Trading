# core/fetch.py
"""
Fetcher centralizado y robusto para el proyecto.

Estrategia:
 - Binance (ccxt) para criptos (prioridad)
 - Finnhub (rotación de claves + límites) para acciones/crypto como fallback primario si Binance no tiene symbol
 - yfinance como último recurso
 - Persiste resultados usando core.adapter.adapter.save_candles o storage modules si existen
 - Exports: Fetcher class, get_candles, fetch_multi, run_full_backfill, safe_run_full_backfill
"""
from __future__ import annotations
import os
import time
import math
import logging
import threading
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger("core.fetch")
logger.setLevel(logging.INFO)

# Optional imports (graceful fallback)
try:
    import ccxt
except Exception:
    ccxt = None

try:
    import requests
except Exception:
    requests = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import finnhub
except Exception:
    finnhub = None

# Attempt to import adapter/storage modules for persistence
try:
    from core import adapter as core_adapter_module
    core_adapter = getattr(core_adapter_module, "adapter", None)
except Exception:
    core_adapter = None

try:
    storage_postgres = __import__("core.storage_postgres", fromlist=["*"])
except Exception:
    storage_postgres = None

try:
    storage_mod = __import__("core.storage", fromlist=["*"])
except Exception:
    storage_mod = None

# Rate limiter (token bucket)
class RateLimiter:
    """
    Token-bucket rate limiter. rate = tokens per interval_seconds.
    Example: 1200 per 60s -> RateLimiter(1200, 60)
    """
    def __init__(self, rate: int, per_seconds: int = 60):
        self.rate = int(rate)
        self.per_seconds = int(per_seconds)
        self._capacity = float(self.rate)
        self._tokens = float(self._capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        add = (elapsed / self.per_seconds) * self.rate
        self._tokens = min(self._capacity, self._tokens + add)
        self._last = now

    def consume(self, tokens: int = 1, block: bool = True, max_wait: Optional[float] = None) -> bool:
        """
        Try to consume `tokens`. If block=True, waits until available or max_wait expires.
        Returns True if tokens were consumed, False otherwise.
        """
        end = None if max_wait is None else time.monotonic() + max_wait
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            if not block:
                return False
            if end and time.monotonic() > end:
                return False
            # sleep small amount proportional to refill time
            time.sleep(min(0.1, max(0.01, self.per_seconds / max(1, self.rate) )))

# Finnhub key manager (rotate keys, per-key limiter)
class FinnhubKeyManager:
    def __init__(self, keys: Optional[List[str]] = None, per_key_limit: int = 60):
        """
        keys: list of API keys
        per_key_limit: requests per minute per key (default 60)
        """
        self.keys = [k.strip() for k in (keys or []) if k and k.strip()]
        self.per_key_limit = int(per_key_limit) or 60
        self._idx = 0
        self._lock = threading.Lock()
        self._limiters = {k: RateLimiter(self.per_key_limit, 60) for k in self.keys}

    def has_keys(self) -> bool:
        return len(self.keys) > 0

    def next_key(self, block: bool = True, max_wait: Optional[float] = None) -> Optional[str]:
        """
        Return the next available key (consumes 1 token). If none, returns None.
        If block True, will wait rotating among keys until one becomes available or max_wait hits.
        """
        if not self.keys:
            return None
        start = time.monotonic()
        while True:
            with self._lock:
                k = self.keys[self._idx]
                self._idx = (self._idx + 1) % len(self.keys)
            limiter = self._limiters.get(k)
            if limiter and limiter.consume(1, block=False):
                return k
            # if not available, try next; if none available and block True then wait
            all_blocked = all(not lim.consume(0, block=False) and lim._tokens < 1 for lim in self._limiters.values())
            if not block:
                return None
            # Wait small time before retrying
            if max_wait is not None and (time.monotonic() - start) > max_wait:
                return None
            time.sleep(0.05)

# Global rate limiters
_BINANCE_RATE = RateLimiter(int(os.getenv("BINANCE_MAX_PER_MIN", "1200")), 60)  # default 1200/min
# Finnhub: per-key limit default 60, but global expected 240/min (60*4 keys)
finnhub_keys_env = os.getenv("FINNHUB_KEYS") or os.getenv("FINNHUB_KEY") or os.getenv("FINNHUB_KEYS")
if finnhub_keys_env:
    # accept comma-separated
    fk_list = [k.strip() for k in finnhub_keys_env.split(",") if k.strip()]
else:
    fk_list = []
FinnhubManager = FinnhubKeyManager(fk_list, per_key_limit=int(os.getenv("FINNHUB_PER_KEY_LIMIT", "60")))

# Binance helper (ccxt)
class BinanceFetcher:
    def __init__(self):
        self._exchange = None
        self._enabled = False
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_SECRET")
        if ccxt is None:
            logger.info("ccxt not installed; Binance fetch disabled.")
            return
        try:
            ex_cls = ccxt.binance
            kwargs = {"enableRateLimit": True}
            if api_key and secret:
                kwargs.update({"apiKey": api_key, "secret": secret})
            self._exchange = ex_cls(kwargs)
            # ccxt has its own rate limiting but we still use our limiter
            self._enabled = True
            logger.info("BinanceFetcher initialized (ccxt).")
        except Exception as e:
            logger.exception("BinanceFetcher init failed: %s", e)
            self._exchange = None
            self._enabled = False

    def symbol_to_binance(self, symbol: str) -> str:
        # Accept 'BTCUSDT' or 'BTC/USDT' or 'BTC-USD'; produce 'BTC/USDT'
        s = symbol.replace("-", "/").replace("_", "/")
        if "/" in s:
            base, quote = s.split("/")
            return f"{base.strip().upper()}/{quote.strip().upper()}"
        # Heuristic: split into base and quote: if endswith USDT or USD or BTC etc.
        for q in ("USDT", "USD", "BTC", "ETH"):
            if s.upper().endswith(q):
                base = s[:-len(q)]
                return f"{base.upper()}/{q}"
        # fallback: return symbol as-is but with slash removed -> try add /USDT
        return f"{s.upper()}/USDT"

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 500) -> pd.DataFrame:
        """
        Returns DataFrame with columns ['ts','timestamp','open','high','low','close','volume'] (ts in ms)
        """
        if not self._enabled or not self._exchange:
            raise RuntimeError("Binance fetcher not available")
        # consume local rate limiter
        if not _BINANCE_RATE.consume(1, block=True, max_wait=10):
            raise RuntimeError("Binance rate limiter: cannot consume token now")
        bn_sym = self.symbol_to_binance(symbol)
        try:
            # ccxt expects timeframe like '1m'
            ohlcv = self._exchange.fetch_ohlcv(bn_sym, timeframe=timeframe, limit=int(limit))
            # ohlcv entries: [timestamp_ms, open, high, low, close, volume]
            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
            return df[["ts","timestamp","open","high","low","close","volume"]]
        except ccxt.ExchangeError as e:
            logger.warning("Binance exchange error for %s: %s", symbol, e)
            raise
        except Exception as e:
            logger.exception("Binance fetch failed for %s: %s", symbol, e)
            raise

# Finnhub fetcher via HTTP (requests) + rotation
def fetch_with_finnhub(symbol: str, resolution: str = "1", limit: int = 500) -> pd.DataFrame:
    """
    Uses finnhub.io API endpoint /crypto/candle (for crypto) and /stock/candle for stocks.
    resolution: '1' | '5' | '15' | '60' | 'D'
    limit: approximate number of rows requested (we convert to time range)
    Returns same DataFrame shape as others.
    """
    if requests is None:
        raise RuntimeError("requests not available for Finnhub fetch")

    # choose timeframe in seconds
    res_map_seconds = {"1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600, "D": 86400}
    step = res_map_seconds.get(str(resolution), 60)
    # calculate from/to window to get ~limit rows
    now = int(time.time())
    _from = now - (limit * step) - 5
    _to = now
    # decide endpoint: crypto or stock
    # if symbol looks like crypto (contains USDT/BTC/ETH and no dot), use crypto endpoint
    symbol_upper = symbol.upper()
    is_crypto = any(symbol_upper.endswith(s) for s in ("USDT","BTC","ETH")) and ("/" not in symbol and ":" not in symbol and "." not in symbol)
    # prepare request params
    # pick a key via FinnhubManager
    key = FinnhubManager.next_key(block=True, max_wait=10) if FinnhubManager.has_keys() else os.getenv("FINNHUB_KEY")
    if not key:
        raise RuntimeError("No Finnhub API key available")
    base_url = "https://finnhub.io/api/v1"
    if is_crypto:
        # Finnhub expects "BINANCE:BTCUSDT" style for crypto; try BINANCE: symbol
        fh_symbol = f"BINANCE:{symbol_upper}" if ":" not in symbol_upper else symbol_upper
        endpoint = f"{base_url}/crypto/candle"
        params = {"symbol": fh_symbol, "resolution": str(resolution), "from": _from, "to": _to, "token": key}
    else:
        # for stocks use /stock/candle with symbol like AAPL
        endpoint = f"{base_url}/stock/candle"
        params = {"symbol": symbol_upper, "resolution": str(resolution), "from": _from, "to": _to, "token": key}

    # apply per-key rate limiter if present
    if FinnhubManager.has_keys():
        # we consumed key via next_key (which already uses limiter.consume)
        pass

    try:
        r = requests.get(endpoint, params=params, timeout=20)
        if r.status_code != 200:
            logger.warning("Finnhub returned status %s for %s: %s", r.status_code, symbol, r.text[:200])
            return pd.DataFrame(columns=["ts","timestamp","open","high","low","close","volume"])
        data = r.json()
        # expected data keys: c, h, l, o, s, t, v
        if data.get("s") != "ok":
            logger.info("Finnhub no data for %s (s=%s)", symbol, data.get("s"))
            return pd.DataFrame(columns=["ts","timestamp","open","high","low","close","volume"])
        times = data.get("t", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        vols = data.get("v", [])
        rows = []
        for i, ts in enumerate(times):
            ms = int(ts) * 1000
            rows.append({
                "ts": ms,
                "timestamp": datetime.utcfromtimestamp(ts),
                "open": opens[i] if i < len(opens) else None,
                "high": highs[i] if i < len(highs) else None,
                "low": lows[i] if i < len(lows) else None,
                "close": closes[i] if i < len(closes) else None,
                "volume": vols[i] if i < len(vols) else None,
            })
        df = pd.DataFrame(rows)
        return df[["ts","timestamp","open","high","low","close","volume"]]
    except Exception as e:
        logger.exception("Finnhub fetch error for %s: %s", symbol, e)
        return pd.DataFrame(columns=["ts","timestamp","open","high","low","close","volume"])

# yfinance fallback
def fetch_with_yfinance(symbol: str, period: str = "1y", interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed")
    try:
        # yfinance expects tickers like 'BTC-USD' or 'BTC-USD' for crypto; if symbol endswith USDT convert to -USD heuristics
        yf_symbol = symbol
        # Try common conversions
        if symbol.upper().endswith("USDT"):
            base = symbol[:-4]
            yf_symbol = f"{base}-USD"  # heuristic
        df = yf.download(tickers=yf_symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["ts","timestamp","open","high","low","close","volume"])
        # yfinance returns DateTimeIndex
        df = df.reset_index()
        # columns: Datetime, Open, High, Low, Close, Volume
        # normalize column names
        colmap = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("open","open.1"): colmap[c] = "open"
            if lc in ("high",): colmap[c] = "high"
            if lc in ("low",): colmap[c] = "low"
            if lc in ("close",): colmap[c] = "close"
            if lc in ("volume",): colmap[c] = "volume"
            if 'datetime' in lc or 'date' in lc or lc == 'index':
                colmap[c] = "timestamp"
        df = df.rename(columns=colmap)
        if "timestamp" not in df.columns:
            # try index as timestamp
            df["timestamp"] = pd.to_datetime(df.iloc[:,0])
        # Create 'ts' milliseconds
        df["ts"] = (pd.to_datetime(df["timestamp"]).astype('int64') // 10**6).astype(int)
        # Keep required columns
        out = df[["ts","timestamp","open","high","low","close","volume"]].copy()
        # limit rows by last N
        if limit and len(out) > limit:
            out = out.tail(limit).reset_index(drop=True)
        return out
    except Exception as e:
        logger.exception("yfinance fetch failed for %s: %s", symbol, e)
        return pd.DataFrame(columns=["ts","timestamp","open","high","low","close","volume"])

# High-level get_candles that tries binance -> finnhub -> yfinance
_binance_fetcher_singleton: Optional[BinanceFetcher] = None
def _get_binance_fetcher() -> Optional[BinanceFetcher]:
    global _binance_fetcher_singleton
    if _binance_fetcher_singleton is None:
        _binance_fetcher_singleton = BinanceFetcher()
    return _binance_fetcher_singleton

def get_candles(symbol: str, limit: int = 1000, timeframe: str = "1m", force_provider: Optional[str] = None) -> pd.DataFrame:
    """
    Top-level function to retrieve candles.
    - symbol: e.g. 'BTCUSDT', 'AAPL'...
    - limit: number of rows desired
    - timeframe: '1m', '5m', '1h', etc. (maps to provider expectations)
    - force_provider: 'binance' | 'finnhub' | 'yfinance' to force a provider
    Returns pandas.DataFrame with columns ['ts','timestamp','open','high','low','close','volume']
    """
    symbol = str(symbol).strip()
    tried = []
    # If caller forces provider
    providers = []
    if force_provider:
        providers = [force_provider.lower()]
    else:
        # Prefer Binance for cryptos, otherwise try Finnhub then Yahoo
        upper = symbol.upper()
        if any(upper.endswith(s) for s in ("USDT","BTC","ETH","BNB")):
            providers = ["binance","finnhub","yfinance"]
        else:
            providers = ["finnhub","binance","yfinance"]

    last_exc = None
    for p in providers:
        try:
            if p == "binance":
                bf = _get_binance_fetcher()
                if bf and bf._enabled:
                    df = bf.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        logger.info("Fetched %d rows from Binance for %s", len(df), symbol)
                        _persist_candles_best_effort(symbol, timeframe, df)
                        return df
                    tried.append(("binance", "empty"))
            elif p == "finnhub":
                # resolution mapping: '1m' -> '1', '5m' -> '5', '1h' -> '60', '1d'->'D'
                res_map = {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1d":"D"}
                res = res_map.get(timeframe, "1")
                df = fetch_with_finnhub(symbol, resolution=res, limit=limit)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info("Fetched %d rows from Finnhub for %s", len(df), symbol)
                    _persist_candles_best_effort(symbol, timeframe, df)
                    return df
                tried.append(("finnhub","empty"))
            elif p == "yfinance":
                # map timeframe to yfinance interval param
                yf_map = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"60m","1d":"1d"}
                yi = yf_map.get(timeframe, "1m")
                df = fetch_with_yfinance(symbol, period="1y", interval=yi, limit=limit)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info("Fetched %d rows from yfinance for %s", len(df), symbol)
                    _persist_candles_best_effort(symbol, timeframe, df)
                    return df
                tried.append(("yfinance","empty"))
        except Exception as e:
            last_exc = e
            logger.exception("Provider %s failed for %s: %s", p, symbol, e)
    # nothing returned
    logger.info("No provider returned data for %s. tried: %s", symbol, tried)
    # return empty df with expected columns
    return pd.DataFrame(columns=["ts","timestamp","open","high","low","close","volume"])

# persistence helper (best-effort)
def _persist_candles_best_effort(symbol: str, timeframe: str, df: pd.DataFrame):
    """
    Try several known persistence entrypoints:
     - core.adapter.adapter.save_candles(symbol, df)
     - storage_postgres.save_candles(symbol, df) or upsert variants
     - storage.save_candles(...)
     - fallback: write CSV to data/cache/
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    # normalize columns: ensure ts, timestamp present
    if "ts" not in df.columns and "timestamp" in df.columns:
        df["ts"] = (pd.to_datetime(df["timestamp"]).astype('int64') // 10**6).astype(int)
    # attempt core.adapter
    try:
        if core_adapter and hasattr(core_adapter, "save_candles"):
            try:
                core_adapter.save_candles(symbol, df)
                return True
            except Exception:
                logger.exception("core_adapter.save_candles failed")
        # maybe adapter exposes _storage
        if core_adapter and hasattr(core_adapter, "_storage") and hasattr(core_adapter._storage, "save_candles"):
            try:
                core_adapter._storage.save_candles(symbol, df)
                return True
            except Exception:
                logger.exception("core_adapter._storage.save_candles failed")
    except Exception:
        logger.exception("adapter persistence attempts failed")
    # try storage_postgres functions
    try:
        if storage_postgres:
            # some implementations call it save_candles or upsert_candles
            if hasattr(storage_postgres, "save_candles"):
                try:
                    storage_postgres.save_candles(symbol, df)
                    return True
                except Exception:
                    logger.exception("storage_postgres.save_candles failed")
            if hasattr(storage_postgres, "upsert_candles"):
                try:
                    rows = df.to_dict("records")
                    storage_postgres.upsert_candles(symbol, timeframe or "1m", rows)
                    return True
                except Exception:
                    logger.exception("storage_postgres.upsert_candles failed")
    except Exception:
        logger.exception("storage_postgres persistence failed")
    # try generic storage_mod
    try:
        if storage_mod and hasattr(storage_mod, "save_candles"):
            try:
                storage_mod.save_candles(symbol, df)
                return True
            except Exception:
                logger.exception("storage_mod.save_candles failed")
    except Exception:
        logger.exception("storage_mod persistence attempts failed")
    # fallback CSV
    try:
        cache_dir = os.path.join(os.getcwd(), "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{symbol.replace('/','_')}.csv")
        df.to_csv(path, index=False)
        logger.info("Wrote fallback CSV to %s", path)
        return True
    except Exception:
        logger.exception("Fallback CSV write failed")
        return False

# Fetcher class (for tests and structured use)
class Fetcher:
    def __init__(self, binance_api_key: Optional[str] = None, binance_secret: Optional[str] = None,
                 finnhub_keys: Optional[List[str]] = None, rate_limit_per_min: int = 1200):
        # allow passing keys at runtime (overrides env)
        if binance_api_key:
            os.environ["BINANCE_API_KEY"] = binance_api_key
        if binance_secret:
            os.environ["BINANCE_SECRET"] = binance_secret
        if finnhub_keys:
            # override FinnhubManager
            global FinnhubManager
            FinnhubManager = FinnhubKeyManager(finnhub_keys, per_key_limit=int(os.getenv("FINNHUB_PER_KEY_LIMIT","60")))
        # set global rate
        global _BINANCE_RATE
        _BINANCE_RATE = RateLimiter(int(rate_limit_per_min), 60)
        # instantiate any internal fetchers
        self._binance = _get_binance_fetcher()

    def get_candles(self, symbol: str, limit: int = 1000, timeframe: str = "1m", force_provider: Optional[str] = None) -> pd.DataFrame:
        return get_candles(symbol, limit=limit, timeframe=timeframe, force_provider=force_provider)

    def fetch_multi(self, symbols: List[str], limit_per_symbol: int = 500, timeframe: str = "1m", parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch many symbols. If parallel=True spawn threads (bounded).
        Returns dict symbol->DataFrame
        """
        out: Dict[str, pd.DataFrame] = {}
        if not parallel or len(symbols) <= 1:
            for s in symbols:
                out[s] = self.get_candles(s, limit=limit_per_symbol, timeframe=timeframe)
            return out

        threads = []
        lock = threading.Lock()
        def worker(sym):
            try:
                df = self.get_candles(sym, limit=limit_per_symbol, timeframe=timeframe)
            except Exception as e:
                logger.exception("fetch_multi worker failed for %s", sym)
                df = pd.DataFrame(columns=["ts","timestamp","open","high","low","close","volume"])
            with lock:
                out[sym] = df

        # limit thread concurrency to e.g. 6
        max_workers = min(8, max(2, len(symbols)))
        sem = threading.Semaphore(max_workers)
        def thread_spawn(sym):
            sem.acquire()
            try:
                worker(sym)
            finally:
                sem.release()

        tlist = []
        for s in symbols:
            t = threading.Thread(target=thread_spawn, args=(s,), daemon=True)
            t.start()
            tlist.append(t)
        for t in tlist:
            t.join(timeout=60)
        return out

# Backfill helpers
def run_full_backfill(symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000) -> Dict[str, Any]:
    """
    High-level backfill entrypoint used by dashboard/orchestrator fallback.
    Attempts to fetch data for each symbol using get_candles and persist results.
    Returns summary dict.
    """
    if symbols is None:
        try:
            # try adapter or storage listing
            if core_adapter and hasattr(core_adapter, "list_assets"):
                symbols = core_adapter.list_assets()
            elif storage_postgres and hasattr(storage_postgres, "list_assets"):
                symbols = storage_postgres.list_assets()
            elif storage_mod and hasattr(storage_mod, "list_assets"):
                symbols = storage_mod.list_assets()
        except Exception:
            pass
    symbols = symbols or []
    summary = {}
    f = Fetcher()
    for s in symbols:
        try:
            df = f.get_candles(s, limit=per_symbol_limit, timeframe="1m")
            summary[s] = {"rows": len(df) if isinstance(df, pd.DataFrame) else 0}
        except Exception as e:
            logger.exception("Backfill fetch failed for %s", s)
            summary[s] = {"error": str(e)}
    return summary

def safe_run_full_backfill(symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000) -> Dict[str, Any]:
    try:
        return run_full_backfill(symbols=symbols, per_symbol_limit=per_symbol_limit)
    except Exception as e:
        logger.exception("safe_run_full_backfill unexpected error")
        return {"error": str(e)}

# Exports
__all__ = [
    "RateLimiter", "FinnhubKeyManager", "Fetcher",
    "get_candles", "fetch_with_binance", "fetch_with_finnhub", "fetch_with_yfinance",
    "run_full_backfill", "safe_run_full_backfill", "fetch_multi"
]
