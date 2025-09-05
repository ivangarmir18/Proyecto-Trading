# core/storage.py
"""
Módulo de storage para el proyecto Watchlist.

Funciones principales:
- init_db(db_path): crear las tablas si no existen.
- get_connection(db_path): contexto con sqlite3 connection (WAL, foreign_keys ON).
- save_candles(df, db_path, asset=None, interval=None): inserta velas (ts en segundos).
- load_candles(asset, interval, db_path): devuelve DataFrame con columnas ts,open,high,low,close,volume,asset,interval
- save_indicators(df_ind, db_path): guarda indicadores asociados (requiere que la vela exista).
- load_indicators(asset, interval, db_path): devuelve DataFrame con indicadores unidos a candles (ts referencia).
- save_scores(df_scores, db_path)
- load_scores(asset, interval, db_path)

Notas:
- No dependemos de pandas.to_sql para tener control sobre upserts.
- Por defecto usa DB en data/db/watchlist.db o la ruta indicada via DB_PATH env var.
"""

from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import os
from typing import Optional
import pandas as pd
import time

# Default DB path aligned with fetch DEFAULT_DB_PATH
DEFAULT_DB = Path(os.getenv("DB_PATH", "data/db/data.db"))

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS candles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asset TEXT NOT NULL,
  interval TEXT NOT NULL,
  ts INTEGER NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL,
  UNIQUE(asset, interval, ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_asset_interval_ts ON candles (asset, interval, ts);

CREATE TABLE IF NOT EXISTS indicators (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candle_id INTEGER NOT NULL,
  ema9 REAL,
  ema40 REAL,
  atr REAL,
  macd REAL,
  macd_signal REAL,
  rsi REAL,
  support REAL,
  resistance REAL,
  fibonacci_levels TEXT,
  FOREIGN KEY (candle_id) REFERENCES candles(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_indicators_candle ON indicators (candle_id);

CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asset TEXT NOT NULL,
  interval TEXT NOT NULL,
  ts INTEGER NOT NULL,
  score REAL NOT NULL,
  range_min REAL,
  range_max REAL,
  stop REAL,
  target REAL,
  p_ml REAL,
  multiplier REAL,
  created_at INTEGER NOT NULL,
  UNIQUE(asset, interval, ts)
);
CREATE INDEX IF NOT EXISTS idx_scores_asset_time ON scores (asset, interval, ts);
"""


@contextmanager
def get_connection(db_path: Optional[str] = None):
    """
    Context manager that yields a sqlite3 connection configured:
    - WAL journal mode
    - FOREIGN KEYS ON
    - reasonable busy timeout
    """
    db_file = Path(db_path) if db_path else DEFAULT_DB
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file), timeout=30, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        conn.commit()
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None):
    """Create schema if it does not exist."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        # execute DDL statements safely
        cur.executescript(DDL)
        conn.commit()


# -------------------------
# Candles
# -------------------------
def save_candles(df: pd.DataFrame, db_path: Optional[str] = None, batch: int = 500, asset: Optional[str] = None, interval: Optional[str] = None):
    """
    Inserta o ignora velas en la tabla candles.
    df debe tener columnas: ['ts','open','high','low','close'] y opcionalmente 'volume'.
    Si asset/interval se pasan por separado se asignarán a las filas (si df no las contiene).
    ts debe ser integer seconds (unix).

    Compatibilidad: soporta llamadas anteriores como save_candles(df, "BTCUSDT", interval="1h") donde
    el segundo argumento era interpretado como db_path en firmas antiguas — por eso recomendamos usar siempre kwargs.
    """
    if df is None or df.empty:
        return

    df = df.copy()

    # allow callers to pass asset/interval as separate columns or args
    if asset is not None and 'asset' not in df.columns:
        df['asset'] = asset
    if interval is not None and 'interval' not in df.columns:
        df['interval'] = interval

    required = {'ts', 'open', 'high', 'low', 'close'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame de candles debe contener columnas: {required}")

    # normalize ts to integer seconds
    if pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = (df['ts'].astype('int64') // 10**9).astype(int)
    else:
        # try numeric coercion
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        if df['ts'].isna().any():
            raise ValueError("Columna 'ts' debe contener timestamps unix (segundos) o datetimes convertibles")
        # if values look like ms (very large), convert to seconds
        if df['ts'].max() > 1e12:
            df['ts'] = (df['ts'] // 1000).astype(int)
        else:
            df['ts'] = df['ts'].astype(int)

    # ensure types for numeric cols
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # ensure asset/interval columns exist (fill blanks with empty string)
    if 'asset' not in df.columns:
        df['asset'] = ''
    if 'interval' not in df.columns:
        df['interval'] = ''

    # prepare rows
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get('asset','')),
            str(r.get('interval','')),
            int(r['ts']),
            float(r['open']),
            float(r['high']),
            float(r['low']),
            float(r['close']),
            float(r.get('volume')) if pd.notna(r.get('volume')) else None
        ))

    with get_connection(db_path) as conn:
        cur = conn.cursor()
        sql = ("INSERT OR IGNORE INTO candles (asset, interval, ts, open, high, low, close, volume) "
               "VALUES (?,?,?,?,?,?,?,?)")
        for i in range(0, len(rows), batch):
            cur.executemany(sql, rows[i:i+batch])
            conn.commit()


def load_candles(asset: str, interval: str, db_path: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Carga velas desde la BBDD y devuelve DataFrame con columnas:
    ts (int unix seconds), timestamp (datetime UTC), open, high, low, close, volume, asset, interval
    """
    with get_connection(db_path) as conn:
        sql = "SELECT ts, open, high, low, close, volume, asset, interval FROM candles WHERE asset=? AND interval=? ORDER BY ts"
        params = (asset, interval)
        if limit:
            sql += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(sql, conn, params=params)
        if not df.empty:
            df['ts'] = pd.to_numeric(df['ts']).astype(int)
            df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        return df


# -------------------------
# Indicators
# -------------------------
def save_indicators(df_ind: pd.DataFrame, db_path: Optional[str] = None, batch: int = 200):
    """
    Guarda indicadores. df_ind debe contener: ['asset','interval','ts','ema9','ema40','atr',...]
    Para cada fila se busca el candle_id y se inserta en indicators.
    Si candle no existe la fila se ignora (se recomienda guardar candles antes).
    """
    if df_ind is None or df_ind.empty:
        return

    if 'ts' not in df_ind.columns:
        raise ValueError("df_ind debe contener columna 'ts'")

    df_ind = df_ind.copy()
    # normalize ts
    if pd.api.types.is_datetime64_any_dtype(df_ind['ts']):
        df_ind['ts'] = (df_ind['ts'].astype('int64') // 10**9).astype(int)
    else:
        df_ind['ts'] = pd.to_numeric(df_ind['ts'], errors='coerce').astype('Int64')

    with get_connection(db_path) as conn:
        cur = conn.cursor()
        # build unique lookup of (asset,interval,ts) needed
        unique_keys = {}
        for _, r in df_ind.iterrows():
            asset_k = str(r.get('asset',''))
            int_k = str(r.get('interval',''))
            ts_k = int(r['ts'])
            unique_keys[(asset_k,int_k,ts_k)] = None
        if unique_keys:
            placeholders = ','.join('?' for _ in range(len(unique_keys)*3))
            # fetch ids in loop for reliability (could be optimized further)
            for (asset_k,int_k,ts_k) in list(unique_keys.keys()):
                cur.execute("SELECT id FROM candles WHERE asset=? AND interval=? AND ts=?", (asset_k,int_k,ts_k))
                row = cur.fetchone()
                if row:
                    unique_keys[(asset_k,int_k,ts_k)] = int(row[0])

        rows = []
        for _, r in df_ind.iterrows():
            asset_k = str(r.get('asset',''))
            int_k = str(r.get('interval',''))
            ts_k = int(r['ts'])
            candle_id = unique_keys.get((asset_k,int_k,ts_k))
            if candle_id is None:
                continue
            rows.append((
                int(candle_id),
                float(r.get('ema9')) if pd.notna(r.get('ema9')) else None,
                float(r.get('ema40')) if pd.notna(r.get('ema40')) else None,
                float(r.get('atr')) if pd.notna(r.get('atr')) else None,
                float(r.get('macd')) if pd.notna(r.get('macd')) else None,
                float(r.get('macd_signal')) if pd.notna(r.get('macd_signal')) else None,
                float(r.get('rsi')) if pd.notna(r.get('rsi')) else None,
                float(r.get('support')) if pd.notna(r.get('support')) else None,
                float(r.get('resistance')) if pd.notna(r.get('resistance')) else None,
                str(r.get('fibonacci_levels')) if pd.notna(r.get('fibonacci_levels')) else None
            ))

        if rows:
            sql = ("INSERT INTO indicators (candle_id, ema9, ema40, atr, macd, macd_signal, rsi, support, resistance, fibonacci_levels) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?)")
            for i in range(0, len(rows), batch):
                cur.executemany(sql, rows[i:i+batch])
                conn.commit()


def load_indicators(asset: str, interval: str, db_path: Optional[str] = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        q = ("SELECT c.ts as ts, i.ema9, i.ema40, i.atr, i.macd, i.macd_signal, i.rsi, i.support, i.resistance, i.fibonacci_levels "
             "FROM candles c JOIN indicators i ON c.id=i.candle_id WHERE c.asset=? AND c.interval=? ORDER BY c.ts")
        df = pd.read_sql_query(q, conn, params=(asset, interval))
        if not df.empty:
            df['ts'] = pd.to_numeric(df['ts']).astype(int)
        return df


# -------------------------
# Scores
# -------------------------
def save_scores(df_scores: pd.DataFrame, db_path: Optional[str] = None, batch: int = 300):
    """
    Guarda scores en la tabla scores.
    df_scores debe contener: ['asset','interval','ts','score','range_min','range_max','stop','target','p_ml','multiplier']
    created_at será el momento actual si no se provee.
    Realiza INSERT OR REPLACE sobre UNIQUE(asset,interval,ts) para actualizar si ya existe.
    """
    if df_scores is None or df_scores.empty:
        return

    df = df_scores.copy()
    # normalize ts
    if 'ts' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = (df['ts'].astype('int64') // 10**9).astype(int)
    elif 'ts' in df.columns:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype(int)

    rows = []
    now_ts = int(time.time())
    for _, r in df.iterrows():
        rows.append((
            str(r.get('asset','')),
            str(r.get('interval','')),
            int(r['ts']),
            float(r.get('score', 0.0)),
            float(r.get('range_min')) if pd.notna(r.get('range_min')) else None,
            float(r.get('range_max')) if pd.notna(r.get('range_max')) else None,
            float(r.get('stop')) if pd.notna(r.get('stop')) else None,
            float(r.get('target')) if pd.notna(r.get('target')) else None,
            float(r.get('p_ml')) if pd.notna(r.get('p_ml')) else None,
            float(r.get('multiplier')) if pd.notna(r.get('multiplier')) else None,
            int(r.get('created_at', now_ts))
        ))

    with get_connection(db_path) as conn:
        cur = conn.cursor()
        sql = ("INSERT OR REPLACE INTO scores (asset, interval, ts, score, range_min, range_max, stop, target, p_ml, multiplier, created_at) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?)")
        for i in range(0, len(rows), batch):
            cur.executemany(sql, rows[i:i+batch])
            conn.commit()


def load_scores(asset: str, interval: str, db_path: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        sql = ("SELECT ts, score, range_min, range_max, stop, target, p_ml, multiplier, created_at "
               "FROM scores WHERE asset=? AND interval=? ORDER BY ts")
        params = (asset, interval)
        if limit:
            sql += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(sql, conn, params=params)
        if not df.empty:
            df['ts'] = pd.to_numeric(df['ts']).astype(int)
        return df
# core/fetch.py
"""
Unified fetch module (updated):

- Downloads base candles (5m) for cryptos (Binance) and generates resampled intervals.
- For stocks: uses Finnhub (rotating keys) for intraday 5m updates and yfinance for historical backfill.
- Functions:
  - backfill_historical(...): backfill historics and generate resamples from base data.
  - refresh_watchlist(...): get latest 5m candles (Binance for crypto, Finnhub for stocks) and optionally save and resample.
  - backfill_from_csv(...): convenience to backfill a list of symbols from CSV.

Notes:
- This file expects core/storage.save_candles(df, db_path=...) to accept a DataFrame containing columns
  ['ts','timestamp','open','high','low','close','volume','asset','interval'] and to write them into the candles table.
- DB path defaults to data/db/watchlist.db but you can pass db_path to functions.
"""
from pathlib import Path
import os
import time
from typing import List, Dict, Optional, Literal
import pandas as pd
import requests
import csv

# optional: load .env automatically if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Binance client (python-binance)
from binance.client import Client as BinanceClient

# storage helper (expects save_candles(df, db_path=...))
from core.storage import save_candles

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
CONFIG_DIR = ROOT / "data" / "config"
DB_DIR = ROOT / "data" / "db"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# default DB path (string expected by save_candles)
DEFAULT_DB_PATH = str(DB_DIR / "watchlist.db")

# env keys
BINANCE_API_KEY = os.getenv("ENV_BINANCE_KEY")
BINANCE_API_SECRET = os.getenv("ENV_BINANCE_SECRET")
_FINNHUB_KEYS_RAW = os.getenv("ENV_FINNHUB_KEYS", "")
FINNHUB_KEYS = [k.strip() for k in _FINNHUB_KEYS_RAW.split(",") if k.strip()]
if not FINNHUB_KEYS:
    for i in range(1, 6):
        k = os.getenv(f"ENV_FINNHUB_KEY{i}")
        if k:
            FINNHUB_KEYS.append(k)

FINNHUB_PER_KEY_MIN_INTERVAL_S = 1.05

VALID_BINANCE_INTERVALS = {"5m", "1h", "2h", "4h", "1d", "1w"}

# resample map (target intervals and pandas rule)
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
    except Exception:
        out = []
        with path.open(newline="", encoding="utf-8") as fh:
            rdr = csv.reader(fh)
            for r in rdr:
                if r and r[0].strip():
                    out.append(r[0].strip())
        return out
    cols = [c.lower() for c in df.columns]
    if "symbol" in cols:
        key = df.columns[cols.index("symbol")]
        return df[key].dropna().astype(str).str.strip().tolist()
    return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()


def find_config_csv(basename: str) -> Path:
    candidates = [
        CONFIG_DIR / f"{basename}.csv",
        CONFIG_DIR / f"{basename[:-1]}.csv" if basename.endswith('s') else CONFIG_DIR / f"{basename}s.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return CONFIG_DIR / f"{basename}.csv"


# -------------------------------
# Normalizers
# -------------------------------

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
    # Flatten multiindex
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([str(i) for i in col if i is not None and str(i) != ""]) for col in df.columns]
    df = df.copy()
    # if timestamp is index
    if not any(c.lower() == 'timestamp' for c in df.columns):
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


# -------------------------------
# Finnhub key ring
# -------------------------------
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
    base_df must be normalized (ts seconds, timestamp datetime, open/high/low/close/volume)
    Generates all intervals defined in INTERVALS_MAP_CRYPTO and saves them using save_candles.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if base_df is None or base_df.empty:
        return
    # ensure normalized
    df = _normalize_candles_df(base_df, symbol=symbol)
    df = df.sort_values('ts')
    # set timestamp index for resample
    df = df.set_index('timestamp')

    for target_interval, rule in INTERVALS_MAP_CRYPTO.items():
        if target_interval == base_interval:
            out = df.reset_index()
        else:
            # OHLCV resample
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
            # recompute ts in seconds
            out['ts'] = (out['timestamp'].astype('int64') // 10**9).astype('int64')
        # add meta
        out['asset'] = symbol
        out['interval'] = target_interval
        # reorder expected columns
        cols = ['ts', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'asset', 'interval']
        out = out[[c for c in cols if c in out.columns]]
        try:
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

    # CRYPTOS: download base 5m (or crypto_interval if specified) and generate resamples
    for s in cryptos:
        try:
            base_interval = crypto_interval
            print(f"[backfill] fetching crypto {s} base_interval={base_interval}")
            base_df = fetch_binance_ohlcv(s, interval=base_interval, limit=crypto_limit)
            # save base and derived intervals
            _generate_and_save_resamples(base_df, s, base_interval=base_interval, db_path=db_path)
            print(f"[backfill] crypto {s} processed (base {base_interval})")
        except Exception as e:
            print(f"[backfill] error crypto {s}: {e}")

    # STOCKS: attempt to get recent 5m via Finnhub or yfinance and longer history via yfinance
    for s in stocks:
        try:
            print(f"[backfill] fetching stock {s} recent 5m via Finnhub/yfinance")
            base_df = None
            # try Finnhub recent 5m (minutes_lookback large so we can resample)
            if FINNHUB_KEYS:
                ring = FinnhubKeyRing(FINNHUB_KEYS)
                idx, key = ring.acquire_key()
                try:
                    base_df = fetch_finnhub_ohlcv(s, resolution="5", minutes_lookback=60*24*7, api_key=key)
                except Exception:
                    base_df = None
            if base_df is None:
                # fallback: try yfinance 5m for recent 7 days
                try:
                    base_df = fetch_yfinance_historical(s, period="7d", interval="5m")
                except Exception:
                    base_df = None
            # if we have base (5m), resample and save
            if base_df is not None and not base_df.empty:
                _generate_and_save_resamples(base_df, s, base_interval="5m", db_path=db_path)
            # for longer history, fetch at stock_interval and ensure saved (use that as coarser interval)
            try:
                long_df = fetch_yfinance_historical(s, period=stock_period, interval=stock_interval)
                # annotate and save long_df
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

    # cryptos: fetch latest base interval (5m)
    for s in cryptos:
        try:
            df = fetch_binance_ohlcv(s, interval=crypto_interval, limit=200)
            results[s] = df
            if save_to_db:
                _generate_and_save_resamples(df, s, base_interval=crypto_interval, db_path=db_path)
        except Exception as e:
            print(f"[refresh] crypto {s} error: {e}")

    # stocks: use finn keys for 5m updates; fallback to yfinance
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
            first_is_header = any(not cell.replace('.','',1).isalnum() and not cell.isupper() for cell in first[:1]) or (len(first)==1 and not first[0].strip().isalnum())
            start_i = 1 if first_is_header else 0
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
                # also save longer history
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

def prune_old_candles(db_path: Optional[str] = None, retention_days_map: dict = None):
    """
    retention_days_map e.g. {'5m': 70, '30m': 180, '1h': 300, '1d': 2000}
    Borra rows más antiguas que retention para cada intervalo.
    """
    if retention_days_map is None:
        retention_days_map = {"5m": 70, "30m": 180, "1h": 300, "1d": 2000}
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        for interval, days in retention_days_map.items():
            cutoff = int(time.time()) - int(days) * 86400
            try:
                cur.execute("DELETE FROM candles WHERE interval=? AND ts < ?", (interval, cutoff))
                conn.commit()
            except Exception as e:
                print(f"[prune] error pruning {interval}: {e}")


if __name__ == "__main__":
    print("Quick local test: backfill_historical (careful with API limits)")
    backfill_historical(crypto_interval="5m", stock_interval="1h", crypto_limit=1000, stock_period="6mo")
