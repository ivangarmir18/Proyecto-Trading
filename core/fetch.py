"""
Unified fetch module ajustado para integrarse con core.storage_postgres.save_candles.

Cambios principales:
- Detecta las variables de entorno BINANCE_KEY / BINANCE_SECRET y ENV_BINANCE_KEY / ENV_BINANCE_SECRET.
- Usa save_candles(...) compatible con PostgreSQL (acepta db_path opcional).
- Mejora robustez en firmas y reintentos.
- Mantiene compatibilidad con CSV-driven backfill.
"""
from __future__ import annotations
from pathlib import Path
import os
import time
import csv
from typing import List, Dict, Optional, Literal

import pandas as pd
import requests

# dotenv opcional para local
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Binance client
from binance.client import Client as BinanceClient

# Import save_candles compatible (acepta db_path opcional)
from core.storage_postgres import save_candles

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
CONFIG_DIR = ROOT / "data" / "config"
DB_DIR = ROOT / "data" / "db"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_PATH = str(DB_DIR / "data.db")  # kept for compatibility, ignored by storage_postgres

# env keys: compatible con varias convenciones
BINANCE_API_KEY = os.getenv("BINANCE_KEY") or os.getenv("ENV_BINANCE_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_SECRET") or os.getenv("ENV_BINANCE_SECRET")

_FINNHUB_KEYS_RAW = os.getenv("FINNHUB_KEYS") or os.getenv("ENV_FINNHUB_KEYS", "")
FINNHUB_KEYS = [k.strip() for k in _FINNHUB_KEYS_RAW.split(",") if k.strip()]
if not FINNHUB_KEYS:
    for i in range(1, 6):
        k = os.getenv(f"FINNHUB_KEY{i}") or os.getenv(f"ENV_FINNHUB_KEY{i}")
        if k:
            FINNHUB_KEYS.append(k)

FINNHUB_PER_KEY_MIN_INTERVAL_S = 1.05

VALID_BINANCE_INTERVALS = {"5m", "30m", "1h", "2h", "4h", "8h", "1d", "1w"}

INTERVALS_MAP_CRYPTO = {
    "5m": "5min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "8h": "8h",
    "1d": "1d",
    "1w": "1W",
}


# -------------------------------
# Utilities
# -------------------------------
def read_symbols_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        if "symbol" in cols:
            key = df.columns[cols.index("symbol")]
            return df[key].dropna().astype(str).str.strip().tolist()
        return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    except Exception:
        out = []
        with path.open(newline="", encoding="utf-8") as fh:
            rdr = csv.reader(fh)
            for r in rdr:
                if r and r[0].strip():
                    out.append(r[0].strip())
        return out


def find_config_csv(basename: str) -> Path:
    candidates = [
        CONFIG_DIR / f"{basename}.csv",
        CONFIG_DIR / f"{basename[:-1]}.csv" if basename.endswith('s') else CONFIG_DIR / f"{basename}s.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return CONFIG_DIR / f"{basename}.csv"


def _cache_to_csv(df: pd.DataFrame, name: str):
    path = CACHE_DIR / f"{name}.csv"
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass


def _normalize_candles_df(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    if df is None:
        raise ValueError("DataFrame is None")
    if df.empty:
        raise ValueError("DataFrame empty")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([str(i) for i in col if i is not None and str(i) != ""]) for col in df.columns]
    df = df.copy()

    if 'timestamp' not in [c.lower() for c in df.columns]:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

    lower_cols = {c: c.lower() for c in df.columns}

    def find_col(possible_names):
        for p in possible_names:
            for orig, low in lower_cols.items():
                if low == p or low.endswith('_' + p) or low.startswith(p + '_') or p in low:
                    return orig
        return None

    mapping = {}
    ts_col = find_col(['timestamp', 'date', 'datetime', 'index'])
    if ts_col:
        mapping[ts_col] = 'timestamp'
    for target, candidates in {
        'open': ['open'],
        'high': ['high'],
        'low': ['low'],
        'close': ['close'],
        'volume': ['volume', 'vol']
    }.items():
        c = find_col(candidates)
        if c:
            mapping[c] = target
    if mapping:
        df = df.rename(columns=mapping)

    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        except Exception:
            raise ValueError("No se pudo parsear columna 'timestamp' a datetime")
        df['ts'] = (df['timestamp'].astype('int64') // 10**9).astype('int64')
    else:
        if 'ts' in df.columns:
            if df['ts'].dtype.kind in 'fi':
                if df['ts'].max() > 1e12:
                    df['ts'] = (df['ts'] // 1000).astype('int64')
                df['ts'] = df['ts'].astype('int64')
                df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
            else:
                raise ValueError("ts column exists but not numeric")
        else:
            raise ValueError("No se detectó columna 'timestamp' ni 'ts' en el DataFrame")

    required = {'ts', 'open', 'high', 'low', 'close'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame de candles debe contener columnas: {required}; faltan: {missing}")

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    cols_out = ['ts', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    cols_out = [c for c in cols_out if c in df.columns]
    return df[cols_out]


# -------------------------------
# Binance fetch
# -------------------------------
def _ensure_binance_client() -> BinanceClient:
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError("Binance API key/secret not found in environment variables.")
    return BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)


def _df_from_binance_klines(raw_klines) -> pd.DataFrame:
    df = pd.DataFrame(raw_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["ts"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    return df[["ts", "timestamp", "open", "high", "low", "close", "volume"]]


def fetch_binance_ohlcv(symbol: str, interval: str = "5m", limit: int = 1000,
                        max_retries: int = 3, sleep_between: float = 0.3) -> pd.DataFrame:
    if interval not in VALID_BINANCE_INTERVALS:
        raise ValueError(f"Interval {interval} not supported. Choose among {VALID_BINANCE_INTERVALS}")
    client = _ensure_binance_client()
    attempt = 0
    while attempt < max_retries:
        try:
            raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = _df_from_binance_klines(raw)
            _cache_to_csv(df, f"{symbol}_{interval}")
            time.sleep(sleep_between)
            return df
        except Exception as e:
            attempt += 1
            wait = 2 ** attempt
            print(f"[binance] error fetching {symbol} attempt {attempt}/{max_retries}: {e}. waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch Binance OHLCV for {symbol} after {max_retries} attempts.")


# -------------------------------
# yfinance fetch
# -------------------------------
def fetch_yfinance_historical(symbol: str, period: str = "6mo", interval: str = "1h",
                              max_retries: int = 3, sleep_between: float = 0.2) -> pd.DataFrame:
    import yfinance as yf
    attempt = 0
    while attempt < max_retries:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
            if df is None or df.empty:
                raise RuntimeError("yfinance returned empty dataframe.")
            df = _normalize_candles_df(df, symbol=symbol)
            _cache_to_csv(df, f"{symbol}_{interval}_yfinance")
            time.sleep(sleep_between)
            return df
        except Exception as e:
            attempt += 1
            wait = 2 ** attempt
            print(f"[yfinance] error fetching {symbol} attempt {attempt}/{max_retries}: {e}. waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch yfinance for {symbol} after {max_retries} attempts.")


# -------------------------------
# Finnhub fetch
# -------------------------------
def _finnhub_request(symbol: str, resolution: str, _from: int, to: int, api_key: str):
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {"symbol": symbol, "resolution": resolution, "from": _from, "to": to, "token": api_key}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Finnhub error {r.status_code}: {r.text}")
    return r.json()


def fetch_finnhub_ohlcv(symbol: str, resolution: str = "5", minutes_lookback: int = 60, api_key: Optional[str] = None) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("Finnhub API key required")
    to_ts = int(time.time())
    from_ts = to_ts - minutes_lookback * 60
    data = _finnhub_request(symbol, resolution, from_ts, to_ts, api_key)
    if data.get("s") != "ok":
        raise RuntimeError(f"Finnhub response status not ok for {symbol}: {data}")
    df = pd.DataFrame({
        "ts_s": data["t"],
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data.get("v", [0]*len(data["t"]))
    })
    df["timestamp"] = pd.to_datetime(df["ts_s"], unit="s", utc=True)
    df["ts"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    df = df[["ts", "timestamp", "open", "high", "low", "close", "volume"]]
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df


class FinnhubKeyRing:
    def __init__(self, keys: List[str]):
        if not keys:
            raise RuntimeError("No Finnhub keys provided")
        self.keys = keys
        self.n = len(keys)
        self.last_used = {i: 0.0 for i in range(self.n)}

    def get_available_key_index(self) -> int:
        now = time.time()
        for i in range(self.n):
            if now - self.last_used[i] >= FINNHUB_PER_KEY_MIN_INTERVAL_S:
                return i
        earliest_next = min(self.last_used[i] + FINNHUB_PER_KEY_MIN_INTERVAL_S for i in range(self.n))
        wait = max(earliest_next - now, 0.01)
        time.sleep(wait)
        return self.get_available_key_index()

    def acquire_key(self) -> (int, str):
        idx = self.get_available_key_index()
        self.last_used[idx] = time.time()
        return idx, self.keys[idx]


# -------------------------------
# Resampling / generation
# -------------------------------
def _generate_and_save_resamples(base_df: pd.DataFrame, symbol: str, base_interval: str = "5m", db_path: Optional[str] = None):
    """
    Genera intervalos derivados (resample) a partir de base_df y guarda con save_candles.
    db_path se acepta por compatibilidad y se ignora en storage_postgres.
    """
    if base_df is None or base_df.empty:
        return
    df = _normalize_candles_df(base_df, symbol=symbol)
    df = df.sort_values('ts')
    df = df.set_index('timestamp')

    for target_interval, rule in INTERVALS_MAP_CRYPTO.items():
        if target_interval == base_interval:
            out = df.reset_index()
        else:
            agg = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            out = df.resample(rule).agg(agg).dropna()
            if out.empty:
                continue
            out = out.reset_index()
            out['ts'] = (out['timestamp'].astype('int64') // 10**9).astype('int64')

        out['asset'] = symbol
        out['interval'] = target_interval
        cols = ['ts', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'asset', 'interval']
        out = out[[c for c in cols if c in out.columns]]
        try:
            # save_candles acepta db_path opcional por compatibilidad
            save_candles(out, db_path=db_path)
        except Exception as e:
            print(f"[resample] failed saving {symbol} {target_interval}: {e}")


# -------------------------------
# High level flows
# -------------------------------
def backfill_historical(crypto_interval: str = "5m", stock_interval: str = "1h",
                        crypto_limit: int = 1000, stock_period: str = "6mo", db_path: Optional[str] = None):
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    cryptos = read_symbols_csv(find_config_csv("cryptos"))
    stocks = read_symbols_csv(find_config_csv("actions"))

    for s in cryptos:
        try:
            base_interval = crypto_interval
            print(f"[backfill] fetching crypto {s} base_interval={base_interval}")
            base_df = fetch_binance_ohlcv(s, interval=base_interval, limit=crypto_limit)
            _generate_and_save_resamples(base_df, s, base_interval=base_interval, db_path=db_path)
            print(f"[backfill] crypto {s} processed (base {base_interval})")
        except Exception as e:
            print(f"[backfill] error crypto {s}: {e}")

    for s in stocks:
        try:
            print(f"[backfill] fetching stock {s} recent 5m via Finnhub/yfinance")
            base_df = None
            if FINNHUB_KEYS:
                ring = FinnhubKeyRing(FINNHUB_KEYS)
                idx, key = ring.acquire_key()
                try:
                    base_df = fetch_finnhub_ohlcv(s, resolution="5", minutes_lookback=60*24*7, api_key=key)
                except Exception:
                    base_df = None
            if base_df is None:
                try:
                    base_df = fetch_yfinance_historical(s, period="7d", interval="5m")
                except Exception:
                    base_df = None
            if base_df is not None and not base_df.empty:
                _generate_and_save_resamples(base_df, s, base_interval="5m", db_path=db_path)
            try:
                long_df = fetch_yfinance_historical(s, period=stock_period, interval=stock_interval)
                long_df['asset'] = s
                long_df['interval'] = stock_interval
                save_candles(long_df, db_path=db_path)
            except Exception as e:
                print(f"[backfill] could not fetch long history for {s}: {e}")

            print(f"[backfill] stock {s} processed")
        except Exception as e:
            print(f"[backfill] error stock {s}: {e}")

    print("[backfill] done.")


def refresh_watchlist(cryptos: List[str], stocks: List[str],
                      crypto_interval: str = "5m", stock_resolution: str = "5",
                      save_to_db: bool = False, db_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    results: Dict[str, pd.DataFrame] = {}

    for s in cryptos:
        try:
            df = fetch_binance_ohlcv(s, interval=crypto_interval, limit=200)
            results[s] = df
            if save_to_db:
                _generate_and_save_resamples(df, s, base_interval=crypto_interval, db_path=db_path)
        except Exception as e:
            print(f"[refresh] crypto {s} error: {e}")

    if FINNHUB_KEYS:
        ring = FinnhubKeyRing(FINNHUB_KEYS)
        for s in stocks:
            try:
                idx, key = ring.acquire_key()
                df = fetch_finnhub_ohlcv(s, resolution=stock_resolution, minutes_lookback=60, api_key=key)
                results[s] = df
                if save_to_db:
                    _generate_and_save_resamples(df, s, base_interval=f"{stock_resolution}m", db_path=db_path)
            except Exception as e:
                print(f"[refresh] finnhub {s} failed: {e} - falling back to yfinance")
                try:
                    df = fetch_yfinance_historical(s, period="7d", interval="5m")
                    results[s] = df
                    if save_to_db:
                        _generate_and_save_resamples(df, s, base_interval="5m", db_path=db_path)
                except Exception as e2:
                    print(f"[refresh] fallback also failed for {s}: {e2}")
    else:
        for s in stocks:
            try:
                df = fetch_yfinance_historical(s, period="7d", interval="5m")
                results[s] = df
                if save_to_db:
                    _generate_and_save_resamples(df, s, base_interval="5m", db_path=db_path)
            except Exception as e:
                print(f"[refresh] yfinance {s} failed: {e}")

    return results


# CSV driven backfill
def backfill_from_csv(csv_path: str, interval: str = "5m", asset_type: Optional[Literal['auto','crypto','stock']] = 'auto',
                      save_to_db: bool = True, pause_between: float = 1.0, db_path: Optional[str] = None):
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    symbols = []
    with p.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and any(h.strip().lower() == 'symbol' for h in reader.fieldnames):
            key = next(h for h in reader.fieldnames if h.strip().lower() == 'symbol')
            for r in reader:
                s = (r.get(key) or '').strip()
                if s:
                    symbols.append(s)
        else:
            fh.seek(0)
            simple = csv.reader(fh)
            rows = [r for r in simple if r and any(c.strip() for c in r)]
            if not rows:
                return []
            first = rows[0]
            # treat as not header by default if ambiguous
            start_i = 1 if any(not cell.replace('.','',1).isalnum() for cell in first[:1]) else 0
            for row in rows[start_i:]:
                if row and row[0].strip():
                    symbols.append(row[0].strip())
    if not symbols:
        print("No se detectaron símbolos en el CSV.")
        return []
    print(f"[backfill_from_csv] symbols detected: {symbols[:20]}{'...' if len(symbols)>20 else ''}")
    processed = []
    for s in symbols:
        atype = asset_type
        if atype == 'auto':
            usdt_like = any(s.upper().endswith(x) for x in ('USDT','USDC','BUSD','BTC','ETH')) or ('/' in s and s.split('/')[-1].upper() in ('USDT','USD'))
            atype = 'crypto' if usdt_like else 'stock'
        try:
            if atype == 'crypto':
                print(f"[backfill_from_csv] fetching crypto {s} base_interval={interval}")
                base_df = fetch_binance_ohlcv(s, interval=interval, limit=1000)
                if save_to_db:
                    _generate_and_save_resamples(base_df, s, base_interval=interval, db_path=db_path)
                print(f"[backfill_from_csv] saved crypto {s} ({len(base_df)} rows)")
            else:
                print(f"[backfill_from_csv] fetching stock {s} recent 5m via finn/yf")
                base_df = None
                if FINNHUB_KEYS:
                    ring = FinnhubKeyRing(FINNHUB_KEYS)
                    idx, key = ring.acquire_key()
                    try:
                        base_df = fetch_finnhub_ohlcv(s, resolution="5", minutes_lookback=60*24*7, api_key=key)
                    except Exception:
                        base_df = None
                if base_df is None:
                    try:
                        base_df = fetch_yfinance_historical(s, period="7d", interval="5m")
                    except Exception:
                        base_df = None
                if base_df is not None and save_to_db:
                    _generate_and_save_resamples(base_df, s, base_interval="5m", db_path=db_path)
                try:
                    long_df = fetch_yfinance_historical(s, period="6mo", interval="1h")
                    long_df['asset'] = s
                    long_df['interval'] = '1h'
                    save_candles(long_df, db_path=db_path)
                except Exception as e:
                    print(f"[backfill_from_csv] could not fetch long history for {s}: {e}")
            processed.append(s)
        except Exception as e:
            print(f"[backfill_from_csv] error fetching {s}: {e}")
        time.sleep(pause_between)
    print("[backfill_from_csv] done.")
    return processed


if __name__ == "__main__":
    print("Quick local test: backfill_historical (careful with API limits)")
    backfill_historical(crypto_interval="5m", stock_interval="1h", crypto_limit=500, stock_period="6mo")
