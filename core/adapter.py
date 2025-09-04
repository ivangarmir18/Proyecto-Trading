# core/adapter.py
"""
Adapter seguro entre dashboard y el core del repo.
Incluye:
 - load/save de velas (intenta core.storage o core.fetch; fallback a CSV)
 - apply_indicators (intenta core.indicators; fallback EMA/RSI)
 - run_full_backfill / update_symbol / run_initial_backtests
 - run_backtest_for / train_ai / infer_ai (llama a core.orchestrator / core.ai_* si existen)
 - save/load settings (intenta core.storage, luego sqlite fallback, luego json)
 - health_status
Diseñado para no romper nada si faltan módulos.
"""

import os
import sys
import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

LOG = logging.getLogger("core.adapter")
LOG.setLevel(logging.INFO)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CONF_DIR = os.path.join(DATA_DIR, "config")
DB_DIR = os.path.join(DATA_DIR, "db")
SETTINGS_DB = os.path.join(DB_DIR, "settings.db")
SETTINGS_JSON = os.path.join(CONF_DIR, "settings.json")
BACKTESTS_DIR = os.path.join(DB_DIR, "backtests")

# ensure dirs exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(BACKTESTS_DIR, exist_ok=True)

def _import_safe(module_path: str, attr: Optional[str] = None):
    """Importa dinámicamente sin lanzar excepción (devuelve None si no existe)."""
    try:
        module = __import__(module_path, fromlist=[attr] if attr else [])
        return getattr(module, attr) if attr else module
    except Exception:
        return None

# intentamos ligar con implementaciones reales si están
_storage = _import_safe("core.storage")
_fetch = _import_safe("core.fetch")
_indicators = _import_safe("core.indicators")
_orchestrator = _import_safe("core.orchestrator")
_ai_train = _import_safe("core.ai_train")
_ai_inf = _import_safe("core.ai_inference")

class Adapter:
    def __init__(self):
        LOG.info("Adapter inicializado. storage=%s fetch=%s indicators=%s orchestrator=%s ai_train=%s ai_infer=%s",
                 bool(_storage), bool(_fetch), bool(_indicators), bool(_orchestrator), bool(_ai_train), bool(_ai_inf))

    # ---------------------------
    # Assets / candles interface
    # ---------------------------
    def list_assets(self) -> List[str]:
        """Devuelve lista de símbolos (watchlist)."""
        try:
            if _storage and hasattr(_storage, "list_assets"):
                return _storage.list_assets()
            if _fetch and hasattr(_fetch, "list_watchlist_assets"):
                return _fetch.list_watchlist_assets()
        except Exception:
            LOG.exception("list_assets core failed")
        return self._fallback_list_assets()

    def _fallback_list_assets(self) -> List[str]:
        candidates = [
            os.path.join(CONF_DIR, "watchlist.csv"),
            os.path.join(CONF_DIR, "watchlist.json"),
            os.path.join(CONF_DIR, "assets.csv"),
            os.path.join(CONF_DIR, "assets.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    if c.endswith(".csv"):
                        df = pd.read_csv(c)
                        if "symbol" in df.columns:
                            return df["symbol"].astype(str).tolist()
                        return df.iloc[:, 0].astype(str).tolist()
                    else:
                        j = pd.read_json(c)
                        return j.iloc[:, 0].astype(str).tolist()
                except Exception:
                    continue
        # fallback default
        return ["BTCUSDT", "ETHUSDT", "AAPL", "TSLA"]

    def load_candles(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Carga velas desde storage/fetch/csv fallback."""
        try:
            if _storage and hasattr(_storage, "load_candles"):
                return _storage.load_candles(symbol, limit=limit)
            if _fetch:
                # aceptar varios nombres posibles
                for fn in ("get_candles", "get_latest_candles", "get_historical", "fetch_candles"):
                    f = getattr(_fetch, fn, None)
                    if callable(f):
                        try:
                            df = f(symbol, limit=limit)
                            if isinstance(df, pd.DataFrame):
                                return df
                        except Exception:
                            LOG.exception("fetch.%s failed for %s", fn, symbol)
        except Exception:
            LOG.exception("load_candles core failed")

        # CSV fallback
        path_csv = os.path.join(CACHE_DIR, f"{symbol}.csv")
        if os.path.exists(path_csv):
            try:
                df = pd.read_csv(path_csv, parse_dates=["timestamp"])
                return df.sort_values("timestamp").reset_index(drop=True)
            except Exception:
                LOG.exception("CSV fallback read failed for %s", path_csv)
        # empty df standard
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def save_candles(self, symbol: str, df: pd.DataFrame) -> bool:
        """Guarda velas en storage o CSV fallback."""
        try:
            if _storage and hasattr(_storage, "save_candles"):
                return _storage.save_candles(symbol, df)
        except Exception:
            LOG.exception("save_candles using core.storage failed")
        try:
            path = os.path.join(CACHE_DIR, f"{symbol}.csv")
            df.to_csv(path, index=False)
            return True
        except Exception:
            LOG.exception("save_candles csv fallback failed for %s", symbol)
        return False

    # ---------------------------
    # Indicators
    # ---------------------------
    def apply_indicators(self, df: pd.DataFrame, indicators: List[str] = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Intenta core.indicators.apply_indicators, si no calcula EMA/RSI minimal."""
        indicators = indicators or ["ema", "rsi"]
        params = params or {}
        try:
            if _indicators and hasattr(_indicators, "apply_indicators"):
                return _indicators.apply_indicators(df.copy(), indicators=indicators, params=params)
        except Exception:
            LOG.exception("apply_indicators via core.indicators failed")
        # fallback minimal
        df = df.copy()
        if "close" in df.columns:
            close = df["close"].astype(float)
            ema_short = int(params.get("ema_short", 9))
            ema_long = int(params.get("ema_long", 50))
            df[f"ema_{ema_short}"] = close.ewm(span=ema_short, adjust=False).mean()
            df[f"ema_{ema_long}"] = close.ewm(span=ema_long, adjust=False).mean()
            period = int(params.get("rsi_period", 14))
            df[f"rsi_{period}"] = self._rsi(close, period)
        return df

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0).fillna(0)
        down = -1 * delta.clip(upper=0).fillna(0)
        ma_up = up.ewm(com=(period - 1), adjust=False).mean()
        ma_down = down.ewm(com=(period - 1), adjust=False).mean()
        rs = ma_up / (ma_down.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    # ---------------------------
    # Backfill / update helpers
    # ---------------------------
    def update_symbol(self, symbol: str, limit: int = 200) -> Dict[str, Any]:
        """
        Recupera las últimas velas para un símbolo y las guarda.
        Busca funciones en core.fetch o core.storage; si falla retorna info de cache.
        """
        try:
            if _fetch:
                for fn_name in ("get_latest_candles", "get_candles", "fetch_candles", "get_historical"):
                    fn = getattr(_fetch, fn_name, None)
                    if callable(fn):
                        try:
                            df = fn(symbol, limit=limit)
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                self.save_candles(symbol, df)
                                return {"symbol": symbol, "rows": len(df), "source": f"fetch::{fn_name}"}
                        except Exception:
                            LOG.exception("fetch fn %s failed for %s", fn_name, symbol)
            df = self.load_candles(symbol, limit=limit)
            return {"symbol": symbol, "rows": len(df) if isinstance(df, pd.DataFrame) else 0, "source": "cache/fallback"}
        except Exception as e:
            LOG.exception("update_symbol error: %s", e)
            return {"symbol": symbol, "error": str(e)}

    def run_full_backfill(self, symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000) -> Dict[str, Any]:
        """
        Ejecuta backfill para todos los símbolos o la lista dada.
        Si core.fetch tiene run_full_backfill, lo utiliza.
        """
        res = {"started_at": datetime.utcnow().isoformat(), "per_symbol_limit": per_symbol_limit, "results": {}}
        try:
            if _fetch and hasattr(_fetch, "run_full_backfill"):
                try:
                    out = _fetch.run_full_backfill(symbols=symbols, limit=per_symbol_limit)
                    res["results"] = {"core_fetch": out}
                    res["finished_at"] = datetime.utcnow().isoformat()
                    return res
                except Exception:
                    LOG.exception("core.fetch.run_full_backfill failed")

            syms = symbols or self.list_assets()
            for s in syms:
                try:
                    r = self.update_symbol(s, limit=per_symbol_limit)
                    res["results"][s] = r
                except Exception as e:
                    LOG.exception("backfill for %s failed: %s", s, e)
                    res["results"][s] = {"error": str(e)}
            res["finished_at"] = datetime.utcnow().isoformat()
        except Exception as e:
            LOG.exception("run_full_backfill top error: %s", e)
            res["error"] = str(e)
        return res

    # ---------------------------
    # Backtest / AI helpers
    # ---------------------------
    def run_backtest_for(self, symbol: str) -> Dict[str, Any]:
        """Llama a core.orchestrator.run_backtest_for si existe."""
        try:
            if _orchestrator and hasattr(_orchestrator, "run_backtest_for"):
                return _orchestrator.run_backtest_for(symbol)
        except Exception as e:
            LOG.exception("run_backtest_for error: %s", e)
            return {"error": "backtest_failed", "message": str(e)}
        return {"error": "not_implemented", "message": "Backtest engine no disponible."}

    def run_initial_backtests(self, symbols: Optional[List[str]] = None, limit: int = 5) -> Dict[str, Any]:
        """Ejecuta backtests para primeros 'limit' símbolos y devuelve resumen."""
        syms = symbols or self.list_assets()
        res = {}
        count = 0
        for s in syms:
            if count >= limit:
                break
            try:
                r = self.run_backtest_for(s)
                res[s] = r
            except Exception as e:
                LOG.exception("initial backtest failed for %s: %s", s, e)
                res[s] = {"error": str(e)}
            count += 1
        return res

    def train_ai(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            if _ai_train and hasattr(_ai_train, "train"):
                return _ai_train.train(params or {})
        except Exception as e:
            LOG.exception("train_ai error: %s", e)
            return {"error": "train_failed", "message": str(e)}
        return {"error": "not_implemented", "message": "Modulo IA no disponible."}

    def infer_ai(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            if _ai_inf and hasattr(_ai_inf, "predict"):
                return _ai_inf.predict(symbol)
            if _ai_inf and hasattr(_ai_inf, "infer"):
                return _ai_inf.infer(symbol)
        except Exception as e:
            LOG.exception("infer_ai error: %s", e)
            return {"error": "infer_failed", "message": str(e)}
        return None

    # ---------------------------
    # Settings persistence
    # ---------------------------
    def save_setting(self, key: str, value: Any) -> bool:
        """Guarda setting intentando storage.real -> sqlite -> json."""
        try:
            if _storage and hasattr(_storage, "save_setting"):
                return _storage.save_setting(key, value)
        except Exception:
            LOG.exception("save_setting via core.storage failed")

        # sqlite fallback
        try:
            conn = sqlite3.connect(SETTINGS_DB)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""")
            cur.execute("INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                        (key, json.dumps(value, default=str), datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
            return True
        except Exception:
            LOG.exception("save_setting sqlite fallback failed")

        # json fallback
        try:
            data = {}
            if os.path.exists(SETTINGS_JSON):
                with open(SETTINGS_JSON, "r", encoding="utf8") as f:
                    data = json.load(f)
            data[key] = value
            with open(SETTINGS_JSON, "w", encoding="utf8") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception:
            LOG.exception("save_setting json fallback failed")
        return False

    def load_setting(self, key: str, default: Any = None) -> Any:
        """Carga setting: intenta storage.real -> sqlite -> json -> default."""
        try:
            if _storage and hasattr(_storage, "load_setting"):
                return _storage.load_setting(key, default)
        except Exception:
            pass
        try:
            conn = sqlite3.connect(SETTINGS_DB)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""")
            cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cur.fetchone()
            conn.close()
            if row:
                try:
                    return json.loads(row[0])
                except Exception:
                    return row[0]
        except Exception:
            pass
        try:
            if os.path.exists(SETTINGS_JSON):
                with open(SETTINGS_JSON, "r", encoding="utf8") as f:
                    j = json.load(f)
                return j.get(key, default)
        except Exception:
            pass
        return default

    # ---------------------------
    # Health / status
    # ---------------------------
    def health_status(self) -> Dict[str, Any]:
        status = {
            "time": datetime.utcnow().isoformat(),
            "modules": {
                "storage": bool(_storage),
                "fetch": bool(_fetch),
                "indicators": bool(_indicators),
                "orchestrator": bool(_orchestrator),
                "ai_train": bool(_ai_train),
                "ai_infer": bool(_ai_inf),
            },
            "data_cache": {
                "cache_dir_exists": os.path.exists(CACHE_DIR),
                "cache_files": len([f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]) if os.path.exists(CACHE_DIR) else 0
            }
        }
        return status

# singleton
adapter = Adapter()
