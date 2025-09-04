# core/fetch.py
"""
core/fetch.py — implementación robusta y compatible del módulo de adquisición.
Soporta:
 - Finnhub (si FINNHUB_API_KEY presente y cliente instalado)
 - Binance (via ccxt, si instalado y claves en env)
 - yfinance (fallback para acciones, si instalado)
 - Postgres storage via core.storage_postgres.PostgresStorage (si existe)
 - CSV fallback en data/cache/
 - Rate limiting (token-bucket) por proveedor con backoff/retries
 - Funciones públicas:
    - list_watchlist_assets()
    - get_candles(symbol, limit=..., timeframe=...)
    - fetch_candles(...)  (alias)
    - get_latest_candles(...) (alias)
    - run_full_backfill(symbols=None, per_symbol_limit=1000)
    - fetch_multi(symbols, ...)
    - set_rate_limits(dict)
    - health_check()
Design goals: backward-compatible, fail-soft, logs útiles para Render.
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

# Optional libs
try:
    import finnhub  # type: ignore
except Exception:
    finnhub = None
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

# Try to import Postgres storage if exists
PgStorage = None
pg = None
try:
    mod = __import__("core.storage_postgres", fromlist=["PostgresStorage"])
    if hasattr(mod, "PostgresStorage"):
        PgStorage = getattr(mod, "PostgresStorage")
        try:
            pg = PgStorage()
        except Exception:
            pg = None
except Exception:
    pg = None

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CONF_DIR = os.path.join(DATA_DIR, "config")
DB_DIR = os.path.join(DATA_DIR, "db")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Logging
logger = logging.getLogger("core.fetch")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# ENV keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip() or None
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip() or None
BINANCE_API_SECRET = os.getenv("BINANCE_SECRET", "").strip() or None

# Default rate limits (per minute)
DEFAULT_RATE_LIMITS = {
    "finnhub": 240,   # per minute (as you mentioned)
    "binance": 1200,  # per minute for fetches
    "yfinance": 3000, # effectively unconstrained locally, but keep high
    "requests": 6000  # generic HTTP requests per minute allowance
}

# Token-bucket rate limiter implementation (thread-safe)
class TokenBucket:
    def __init__(self, rate_per_minute: int, capacity: Optional[int] = None):
        self.rate_per_minute = max(1, int(rate_per_minute))
        # tokens per second
        self._rate_per_sec = self.rate_per_minute / 60.0
        self.capacity = capacity or max(1, self.rate_per_minute)
        self._tokens = float(self.capacity)
        self._last = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self._rate_per_sec)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_for(self, tokens: float = 1.0, timeout: float = 10.0):
        """Block (polling) until tokens available or timeout. Returns True if consumed."""
        deadline = time.time() + timeout
        while time.time() <= deadline:
            if self.consume(tokens):
                return True
            time.sleep(0.05)
        return False

# Global rate buckets per-provider
_rate_buckets_lock = threading.Lock()
_rate_buckets: Dict[str, TokenBucket] = {}

def init_rate_buckets(custom_limits: Optional[Dict[str, int]] = None):
    with _rate_buckets_lock:
        lims = dict(DEFAULT_RATE_LIMITS)
        if custom_limits:
            lims.update(custom_limits)
        for k, v in lims.items():
            _rate_buckets[k] = TokenBucket(rate_per_minute=int(v), capacity=int(max(1, v)))

# initialize defaults
init_rate_buckets()

def set_rate_limits(limits: Dict[str, int]):
    """Update rate limits at runtime. limits: {'finnhub':240, 'binance':1200}"""
    init_rate_buckets(limits or {})

# Retry/backoff helper
def with_retries(fn, retries=4, backoff_base=0.8, allowed_exceptions=(Exception,), rate_bucket_key: Optional[str]=None):
    """
    Returns a wrapper that applies retries/backoff and respects a token bucket if provided.
    Usage: call wrapper(*args, **kwargs)
    """
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, retries + 1):
            # rate limit wait
            if rate_bucket_key:
                bucket = _rate_buckets.get(rate_bucket_key)
                if bucket:
                    ok = bucket.wait_for(timeout=10.0)
                    if not ok:
                        logger.warning("Rate bucket timeout for %s", rate_bucket_key)
            try:
                return fn(*args, **kwargs)
            except allowed_exceptions as e:
                last_exc = e
                sleep_time = backoff_base * (2 ** (attempt - 1)) + (0.1 * attempt)
                logger.debug("Retry %d/%d for %s after %s s due to %s", attempt, retries, getattr(fn, "__name__", "fn"), sleep_time, repr(e))
                time.sleep(sleep_time)
        # all retries failed
        logger.exception("Function %s failed after %d retries: %s", getattr(fn, "__name__", "fn"), retries, last_exc)
        raise last_exc
    return wrapper

# Helper to ensure df structure
def _ensure_df_structure(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    # normalize timestamp column
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            df["timestamp"] = df["timestamp"]
    # keep column order
    return df[expected]

# CSV persistence
def _csv_path(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")

def _save_csv(symbol: str, df: pd.DataFrame) -> bool:
    try:
        p = _csv_path(symbol)
        # ensure timestamp serializable
        if "timestamp" in df.columns:
            d = df.copy()
            d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            d.to_csv(p, index=False)
        else:
            df.to_csv(p, index=False)
        logger.debug("Saved CSV cache for %s -> %s", symbol, p)
        return True
    except Exception:
        logger.exception("Failed to save CSV for %s", symbol)
        return False

def _load_csv(symbol: str, limit: Optional[int]=1000) -> pd.DataFrame:
    p = _csv_path(symbol)
    if not os.path.exists(p):
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    try:
        df = pd.read_csv(p, parse_dates=["timestamp"], infer_datetime_format=True)
        df = _ensure_df_structure(df)
        if limit and len(df) > limit:
            return df.sort_values("timestamp").iloc[-limit:].reset_index(drop=True)
        return df
    except Exception:
        logger.exception("Failed to load CSV for %s", symbol)
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# Storage helpers (Postgres preferred)
def _load_from_storage(symbol: str, limit: int=1000) -> pd.DataFrame:
    if pg and hasattr(pg, "load_candles"):
        try:
            df = pg.load_candles(symbol, limit=limit)
            if isinstance(df, pd.DataFrame):
                return _ensure_df_structure(df)
        except Exception:
            logger.exception("pg.load_candles failed for %s", symbol)
    # fallback to CSV
    return _load_csv(symbol, limit=limit)

def _save_to_storage(symbol: str, df: pd.DataFrame) -> bool:
    if pg and hasattr(pg, "save_candles"):
        try:
            ok = pg.save_candles(symbol, df)
            if ok:
                return True
        except Exception:
            logger.exception("pg.save_candles failed for %s", symbol)
    return _save_csv(symbol, df)

# -------------------------
# Network fetchers
# -------------------------

# FINNHUB fetcher (stock / crypto / forex depending on plan)
def _fetch_with_finnhub(symbol: str, resolution: str = "1", count: int = 1000) -> pd.DataFrame:
    """
    resolution: '1','5','15','60','D' (Finnhub uses string resolutions)
    count: approx number of candles -- we'll compute a from/until using now
    """
    if finnhub is None or FINNHUB_API_KEY is None:
        logger.debug("Finnhub not available or FINNHUB_API_KEY missing")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    # Respect rate limits via token bucket
    bucket_key = "finnhub"
    bucket = _rate_buckets.get(bucket_key)
    if bucket and not bucket.wait_for(timeout=10.0):
        logger.warning("Finnhub rate bucket wait failed")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    try:
        client = finnhub.Client(api_key=FINNHUB_API_KEY)  # type: ignore
        # Compute from/until based on resolution and count (best-effort)
        now = int(time.time())
        # resolution to seconds heuristic
        if resolution.upper() in ("D", "D1", "DAYS"):
            span = 24*3600
        else:
            try:
                span = int(resolution) * 60
            except Exception:
                span = 60
        from_ts = now - int(span * max(1, count))
        to_ts = now
        # Finnhub: client.stock_candles(symbol, resolution, _from, to)
        # The symbol format depends on exchange (user must provide appropriate symbol)
        resp = client.stock_candles(symbol, resolution, from_ts, to_ts)  # may return dict with 'c','t','o','h','l','v'
        if not resp or not isinstance(resp, dict) or "c" not in resp:
            logger.debug("Finnhub returned no candles for %s", symbol)
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        ts = resp.get("t", [])
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(ts, unit="s"),
            "open": resp.get("o", []),
            "high": resp.get("h", []),
            "low": resp.get("l", []),
            "close": resp.get("c", []),
            "volume": resp.get("v", []),
        })
        return _ensure_df_structure(df)
    except Exception:
        logger.exception("Finnhub fetch failed for %s", symbol)
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# Binance via ccxt
def _fetch_with_ccxt_binance(symbol: str, timeframe: str = "1m", limit: int = 1000) -> pd.DataFrame:
    if ccxt is None:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    # rate bucket
    bucket_key = "binance"
    bucket = _rate_buckets.get(bucket_key)
    if bucket and not bucket.wait_for(timeout=10.0):
        logger.warning("Binance rate bucket wait failed")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    try:
        params = {"enableRateLimit": True}
        if BINANCE_API_KEY:
            params.update({"apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET})
        exchange = ccxt.binance(params)
        # symbol must be ccxt format like 'BTC/USDT'
        s = symbol
        if "/" not in s:
            # try to sanitize: 'BTCUSDT' -> 'BTC/USDT'
            if s.endswith("USDT"):
                s = f"{s[:-4]}/USDT"
            elif s.endswith("BTC"):
                s = f"{s[:-3]}/BTC"
        ohlcv = exchange.fetch_ohlcv(s, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return _ensure_df_structure(df)
    except Exception:
        logger.exception("ccxt/binance fetch failed for %s", symbol)
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# yfinance fallback
def _fetch_with_yfinance(symbol: str, period: str = "1y", interval: str = "1d", limit: Optional[int] = None) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    try:
        tk = symbol
        # sanitize: remove slash if present (yfinance uses 'AAPL' or 'AAPL.MX' etc)
        tk = tk.replace("/", "-")
        hist = yf.Ticker(tk).history(period=period, interval=interval, auto_adjust=False)
        if hist is None or hist.empty:
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        df = hist.reset_index().rename(columns={"Date": "timestamp", "Open": "open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if limit and len(df) > limit:
            df = df.iloc[-limit:]
        return _ensure_df_structure(df)
    except Exception:
        logger.exception("yfinance fetch failed for %s", symbol)
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# Generic network attempt respecting priority and rate-limits
def _network_get_candles(symbol: str, limit: int = 1000, timeframe: Optional[str] = None) -> pd.DataFrame:
    """
    Try in order:
     - Finnhub (if FINNHUB_API_KEY and finnhub available)
     - Binance via ccxt heuristics (if ccxt available)
     - yfinance (fallback for equities)
    """
    sym_up = symbol.upper()
    # Try Finnhub for tickers if configured (best for many stocks)
    if FINNHUB_API_KEY and finnhub:
        # choose resolution: if timeframe provided map to finnhub resolution (best-effort)
        resolution = None
        if timeframe:
            # map '1m'->'1', '5m'->'5', '1d'->'D'
            if timeframe.endswith("m"):
                resolution = timeframe[:-1]
            elif timeframe.endswith("h"):
                resolution = str(int(timeframe[:-1]) * 60)
            elif timeframe.lower() in ("1d","1D","D"):
                resolution = "D"
        else:
            resolution = "1"  # default 1-min
        try:
            df = _fetch_with_finnhub(sym_up, resolution, count=limit)
            if df is not None and not df.empty:
                return df
        except Exception:
            logger.exception("Finnhub attempt failed for %s", symbol)

    # If looks like crypto (endswith USDT/BTC/ETH) try ccxt/binance
    if ccxt:
        crypto_markers = ("USDT", "BTC", "ETH", "BNB", "USDC")
        if any(sym_up.endswith(m) for m in crypto_markers):
            try:
                tf = timeframe or "1m"
                df = _fetch_with_ccxt_binance(sym_up, timeframe=tf, limit=limit)
                if df is not None and not df.empty:
                    return df
            except Exception:
                logger.exception("ccxt attempt failed for %s", symbol)

    # fallback to yfinance for equities
    if yf:
        try:
            interval = "1d"
            if timeframe and timeframe.endswith("m"):
                # yfinance does not support <1m trivial mapping; use 1d default
                interval = "1d"
            df = _fetch_with_yfinance(symbol, period="1y", interval=interval, limit=limit)
            if df is not None and not df.empty:
                return df
        except Exception:
            logger.exception("yfinance attempt failed for %s", symbol)

    # last resort: empty df
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# -------------------------
# Public API
# -------------------------
def list_watchlist_assets() -> List[str]:
    # Prefer Postgres storage listing if available
    try:
        if pg and hasattr(pg, "list_assets"):
            out = pg.list_assets()
            if isinstance(out, list):
                return out
            if isinstance(out, pd.DataFrame):
                if "symbol" in out.columns:
                    return out["symbol"].astype(str).tolist()
                return out.iloc[:,0].astype(str).tolist()
    except Exception:
        logger.exception("pg.list_assets failed; falling back")

    # Try config files
    candidates = [os.path.join(CONF_DIR, "watchlist.csv"), os.path.join(CONF_DIR, "watchlist.json")]
    for c in candidates:
        if os.path.exists(c):
            try:
                if c.endswith(".csv"):
                    df = pd.read_csv(c)
                    if "symbol" in df.columns:
                        return df["symbol"].astype(str).tolist()
                    return df.iloc[:,0].astype(str).tolist()
                else:
                    j = json.load(open(c,"r",encoding="utf8"))
                    if isinstance(j, list):
                        return j
                    if isinstance(j, dict):
                        return list(j.keys())
            except Exception:
                logger.exception("Failed read watchlist file %s", c)
                continue

    # fallback
    return ["BTCUSDT", "ETHUSDT", "AAPL", "TSLA"]

def get_candles(symbol: str, limit: int = 1000, timeframe: Optional[str] = None, force_network: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame with columns timestamp, open, high, low, close, volume.
    Strategy:
      1) If not force_network: try storage (pg) via pg.load_candles
      2) Try network get (Finnhub/ccxt/yfinance)
      3) Save network result to storage (pg or csv)
      4) If network fails, load from CSV/storage fallback
    timeframe: string like '1m','5m','1h','1d' — used where supported
    """
    # 1) storage
    if not force_network:
        try:
            df = _load_from_storage(symbol, limit=limit)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            logger.exception("load from storage failed for %s", symbol)

    # 2) network
    try:
        df_net = _network_get_candles(symbol, limit=limit, timeframe=timeframe)
        if isinstance(df_net, pd.DataFrame) and not df_net.empty:
            # 3) save to storage
            try:
                _save_to_storage(symbol, df_net)
            except Exception:
                logger.exception("saving network result failed for %s", symbol)
            return df_net
    except Exception:
        logger.exception("network_get_candles failed for %s", symbol)

    # 4) fallback storage/csv
    try:
        return _load_from_storage(symbol, limit=limit)
    except Exception:
        logger.exception("final fallback load failed for %s", symbol)
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# Aliases expected by other modules
fetch_candles = get_candles
get_latest_candles = get_candles
fetch = get_candles

def fetch_multi(symbols: List[str], limit_per_symbol:int = 1000, timeframe: Optional[str] = None, parallel: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple symbols (either from storage or network). Returns dict symbol->DataFrame.
    If parallel True uses threads (bounded).
    """
    results: Dict[str, pd.DataFrame] = {}
    if not symbols:
        return results

    def _worker(sym):
        try:
            d = get_candles(sym, limit=limit_per_symbol, timeframe=timeframe)
            results[sym] = d
        except Exception:
            logger.exception("fetch_multi failed for %s", sym)
            results[sym] = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    if parallel:
        threads = []
        for s in symbols:
            t = threading.Thread(target=_worker, args=(s,), daemon=True)
            threads.append(t)
            t.start()
            # avoid starting unlimited threads: small throttle
            time.sleep(0.01)
        for t in threads:
            t.join(timeout=30)
    else:
        for s in symbols:
            _worker(s)
    return results

def run_full_backfill(symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000) -> Dict[str, Any]:
    """
    Full backfill: iterate over symbols and fetch network data, saving to storage.
    Returns summary dict.
    NOTE: This can be slow; call from background thread/process.
    """
    out: Dict[str, Any] = {"started_at": datetime.utcnow().isoformat(), "results": {}}
    syms = symbols or list_watchlist_assets()
    for s in syms:
        try:
            df = _network_get_candles(s, limit=per_symbol_limit)
            if isinstance(df, pd.DataFrame) and not df.empty:
                saved = _save_to_storage(s, df)
                out["results"][s] = {"rows": len(df), "saved": bool(saved)}
            else:
                # try to load fallback and report
                df2 = _load_from_storage(s, limit=per_symbol_limit)
                out["results"][s] = {"rows": len(df2) if isinstance(df2, pd.DataFrame) else 0, "saved": False}
        except Exception as e:
            logger.exception("Backfill error for %s", s)
            out["results"][s] = {"error": str(e)}
    out["finished_at"] = datetime.utcnow().isoformat()
    return out

def health_check() -> Dict[str, Any]:
    """Return a dict summarizing available modules and status (useful for UI)."""
    return {
        "time": datetime.utcnow().isoformat(),
        "modules": {
            "finnhub": bool(finnhub and FINNHUB_API_KEY),
            "ccxt": bool(ccxt),
            "yfinance": bool(yf),
            "pg_storage": bool(pg),
        },
        "rate_buckets": {k: getattr(v, "rate_per_minute", None) for k,v in _rate_buckets.items()},
        "cache_files": len([f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]) if os.path.exists(CACHE_DIR) else 0
    }

# compatibility exports
list_assets = list_watchlist_assets
run_backfill = run_full_backfill
# End of file
# ----------------------------
# Wrappers públicos compat (añadir al final de core/fetch.py)
# ----------------------------
import pandas as _pd
import os as _os
from datetime import datetime as _dt

# list_watchlist_assets -> intenta fetch internals o config
def list_watchlist_assets():
    try:
        # si el módulo define una función específica
        for name in ("list_watchlist_assets", "list_assets", "get_watchlist", "get_assets"):
            if name in globals() and callable(globals()[name]):
                try:
                    out = globals()[name]()
                    if isinstance(out, (list,tuple)):
                        return list(out)
                except Exception:
                    pass
    except Exception:
        pass
    # fallback: leer archivos data/config
    cfg_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "data", "config"))
    out=[]
    for fn in ("actions.csv","cryptos.csv","crypto.csv"):
        p = _os.path.join(cfg_dir, fn)
        if _os.path.exists(p):
            try:
                df = _pd.read_csv(p, dtype=str, keep_default_na=False)
                if "symbol" in df.columns:
                    out.extend(df["symbol"].tolist())
                else:
                    out.extend(df.iloc[:,0].astype(str).tolist())
            except Exception:
                continue
    # dedupe
    seen=set(); final=[]
    for s in out:
        if s not in seen:
            seen.add(s); final.append(s)
    return final

# get_candles alias: intenta storage -> network -> csv fallback
def get_candles(symbol: str, limit: int = 1000, timeframe: str=None, force_network: bool=False) -> _pd.DataFrame:
    # try storage loader if implemented elsewhere (storage_postgres wrappers)
    try:
        import core.storage_postgres as _sp
        if hasattr(_sp, "load_candles") and not force_network:
            try:
                df = _sp.load_candles(symbol, limit=limit)
                if isinstance(df, _pd.DataFrame) and not df.empty:
                    return df
            except Exception:
                pass
    except Exception:
        pass

    # try existing fetch implementations
    if "fetch_ohlcv" in globals():
        try:
            # try to call with common signature
            try:
                df = fetch_ohlcv(symbol, timeframe or "1m", start_ms=None, end_ms=None, limit=limit)
            except TypeError:
                try:
                    df = fetch_ohlcv(symbol, timeframe or "1m")
                except Exception:
                    df = fetch_ohlcv(symbol)
            if isinstance(df, _pd.DataFrame) and not df.empty:
                # persist via storage_postgres if possible
                try:
                    import core.storage_postgres as _sp2
                    if hasattr(_sp2, "save_candles"):
                        _sp2.save_candles(symbol, df)
                except Exception:
                    pass
                return df
        except Exception:
            pass

    # if there is a network backfill function, call it then try storage
    if "backfill_range" in globals():
        try:
            try:
                backfill_range(symbol, timeframe or "1m", None, None)
            except TypeError:
                backfill_range(symbol, timeframe or "1m")
            # try load again from storage
            try:
                import core.storage_postgres as _sp3
                if hasattr(_sp3, "load_candles"):
                    df = _sp3.load_candles(symbol, limit=limit)
                    if isinstance(df, _pd.DataFrame) and not df.empty:
                        return df
            except Exception:
                pass
        except Exception:
            pass

    # csv fallback
    try:
        p = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "data", "cache", f"{symbol.replace('/','_')}.csv"))
        if _os.path.exists(p):
            df = _pd.read_csv(p, parse_dates=["timestamp"], infer_datetime_format=True)
            if limit and len(df) > limit:
                return df.sort_values("timestamp").iloc[-limit:].reset_index(drop=True)
            return df
    except Exception:
        pass

    return _pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# aliases
fetch_candles = get_candles
get_latest_candles = get_candles

def run_full_backfill(symbols=None, per_symbol_limit:int=1000):
    out = {"started_at": _dt.utcnow().isoformat(), "results": {}}
    try:
        if symbols is None:
            symbols = list_watchlist_assets()
        for s in symbols:
            try:
                # try existing backfill fn
                if "backfill_range" in globals():
                    backfill_range(s, None, None, None)
                    # try to load saved rows
                    try:
                        import core.storage_postgres as _sp4
                        if hasattr(_sp4, "load_candles"):
                            df = _sp4.load_candles(s, limit=per_symbol_limit)
                            out["results"][s] = {"rows": len(df) if df is not None else 0}
                            continue
                    except Exception:
                        pass
                # fallback: call fetch_ohlcv and persist
                if "fetch_ohlcv" in globals():
                    df = fetch_ohlcv(s, "1m")
                    try:
                        import core.storage_postgres as _sp5
                        if hasattr(_sp5, "save_candles"):
                            _sp5.save_candles(s, df)
                    except Exception:
                        pass
                    out["results"][s] = {"rows": len(df) if hasattr(df,'__len__') else 0}
            except Exception as e:
                out["results"][s] = {"error": str(e)}
    except Exception as e:
        out["error"] = str(e)
    out["finished_at"] = _dt.utcnow().isoformat()
    return out

# --- Inicio parche fetch.py: backfill_range genérico ---
import time, logging

logger = logging.getLogger(__name__)

def _default_backfill_range(self, asset, interval, start_ms, end_ms, batch_window_ms=6*3600*1000, callback=None):
    """
    Backfill por ventanas: intenta usar varios métodos de fetch disponibles:
    - self.fetch_candles(asset, interval, start, end)
    - self.get_candles(asset, interval, start, end)
    - self._network_get_candles(asset, interval, start, end)
    Los tiempos manejados son en ms.
    callback(batch_df) será llamado con cada batch (pandas DataFrame o lista).
    """
    # preferir batch_window_ms en ms
    cur = int(start_ms)
    end_ms = int(end_ms)
    # detect name of fetch function
    fetch_fn = None
    for name in ("fetch_candles", "get_candles", "_network_get_candles", "fetch_ohlcv", "fetch_range"):
        if hasattr(self, name):
            fetch_fn = getattr(self, name)
            break
    if fetch_fn is None:
        raise RuntimeError("No fetch function found on fetcher (required for backfill_range)")

    while cur < end_ms:
        to_ms = min(cur + int(batch_window_ms), end_ms)
        try:
            # muchos fetchers usan start/end en ms, algunos usan seconds -> el fetch_fn debe aceptar ms en este proyecto
            batch = fetch_fn(asset, interval, start=cur, end=to_ms)
        except TypeError:
            # intentar con otra firma: (asset, interval, cur, to_ms)
            batch = fetch_fn(asset, interval, cur, to_ms)
        except Exception as e:
            logger.exception("Error fetching window %s-%s for %s: %s", cur, to_ms, asset, e)
            raise
        if callback:
            callback(batch)
        else:
            # si no hay callback, intentar guardar en storage si existe
            if hasattr(self, "storage") and hasattr(self.storage, "save_candles"):
                try:
                    self.storage.save_candles(asset, interval, batch)
                except Exception:
                    logger.exception("No se pudo guardar batch en storage")
        # avanzar cursor
        cur = to_ms
        # respetar rate-limit básico
        time.sleep(0.05)

# attach to Fetcher class if exists
try:
    Fetcher  # type: ignore
except Exception:
    Fetcher = None

if Fetcher is not None and not hasattr(Fetcher, "backfill_range"):
    setattr(Fetcher, "backfill_range", _default_backfill_range)

# --- Fin parche fetch.py ---

