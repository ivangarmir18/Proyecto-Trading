# core/adapter.py
"""
Adapter seguro entre dashboard y el core del repo.

Responsabilidades principales:
 - load/save de velas (intenta core.storage o core.fetch; fallback a CSV)
 - apply_indicators (intenta core.indicators; fallback EMA/RSI)
 - run_full_backfill / update_symbol / run_initial_backtests
 - run_backtest_for / train_ai / infer_ai (llama a core.orchestrator / core.ai_* si existen)
 - save/load settings (intenta core.storage, luego sqlite fallback, luego json)
 - health_status
 - gestión de jobs en background con persistencia (para poder reanudar)
"""

from typing import Any, Dict, Optional, List, Callable
import importlib
import inspect
import os
import json
import threading
import time
import logging
import csv
import traceback

import pandas as pd

logger = logging.getLogger("core.adapter")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# -----------------------------
# Safe loader helper
# -----------------------------
def _import_safe(module_path: str, attr: Optional[str] = None):
    try:
        mod = importlib.import_module(module_path)
        if attr:
            return getattr(mod, attr)
        return mod
    except Exception:
        return None

# try to bind to core modules if exist
_fetch = _import_safe("core.fetch")
_storage = _import_safe("core.storage_postgres") or _import_safe("core.storage") or _import_safe("core.storage_adapter")
_indicators = _import_safe("core.indicators")
_orchestrator = _import_safe("core.orchestrator")
_score = _import_safe("core.score")
_ai_train = _import_safe("core.ai_train")
_ai_inference = _import_safe("core.ai_inference") or _import_safe("core.ai_interference")

# Settings file fallback location (if storage not present)
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CONF_DIR = os.path.join(BASE_DIR, "config")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
SETTINGS_FILE = os.path.join(CONF_DIR, "settings.local.json")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)

# Job persistence key name
_JOBS_SETTING_KEY = "core_adapter_jobs"

# -----------------------------
# Adapter class
# -----------------------------
class CoreAdapter:
    def __init__(self):
        # module bindings
        self._fetch = _fetch
        self._storage = _storage
        self._indicators = _indicators
        self._orchestrator = _orchestrator
        self._score = _score
        self._ai_train = _ai_train
        self._ai_inference = _ai_inference

        # jobs stored in memory; will try to load persisted jobs on init
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._job_lock = threading.RLock()
        self._load_persisted_jobs()

        logger.info("CoreAdapter initialized. fetch=%s storage=%s indicators=%s score=%s ai_train=%s ai_inf=%s",
                    bool(self._fetch), bool(self._storage), bool(self._indicators), bool(self._score),
                    bool(self._ai_train), bool(self._ai_inference))

    # -----------------------------
    # Settings persistence wrappers
    # -----------------------------
    def save_setting_source(self, key: str, value: Any) -> bool:
        """
        Save setting using storage module if possible, else JSON file fallback.
        """
        try:
            if self._storage and hasattr(self._storage, "save_setting"):
                try:
                    return self._storage.save_setting(key, value)
                except Exception:
                    logger.exception("pg.save_setting failed")
            # fallback to local file (merge)
            data = {}
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE, "r", encoding="utf8") as fh:
                        data = json.load(fh)
                except Exception:
                    logger.exception("Failed reading local settings file; will overwrite")
            data[key] = value
            try:
                with open(SETTINGS_FILE, "w", encoding="utf8") as fh:
                    json.dump(data, fh, indent=2, default=str)
                return True
            except Exception:
                logger.exception("fallback save_setting failed")
                return False
        except Exception:
            logger.exception("save_setting_source top-level failed")
            return False

    def load_setting_source(self, key: str, default: Any = None) -> Any:
        """
        Load setting using storage module if possible, else JSON file fallback.
        """
        try:
            if self._storage and hasattr(self._storage, "load_setting"):
                try:
                    return self._storage.load_setting(key, default)
                except Exception:
                    logger.exception("pg.load_setting failed")
            # fallback local file
            try:
                if os.path.exists(SETTINGS_FILE):
                    with open(SETTINGS_FILE, "r", encoding="utf8") as fh:
                        data = json.load(fh)
                    return data.get(key, default)
            except Exception:
                logger.exception("fallback load_setting failed")
            return default
        except Exception:
            logger.exception("load_setting_source top-level failed")
            return default

    # -----------------------------
    # Jobs persistence & management
    # -----------------------------
    def _persist_jobs(self):
        """Persiste self.jobs usando save_setting_source (atomic)."""
        try:
            with self._job_lock:
                self.save_setting_source(_JOBS_SETTING_KEY, self.jobs)
        except Exception:
            logger.exception("Failed to persist jobs")

    def _load_persisted_jobs(self):
        """Carga jobs persistidos desde storage/settings file y actualiza self.jobs."""
        try:
            loaded = self.load_setting_source(_JOBS_SETTING_KEY, {})
            if isinstance(loaded, dict):
                self.jobs = loaded
            else:
                self.jobs = {}
        except Exception:
            logger.exception("Failed to load persisted jobs; starting empty")
            self.jobs = {}

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        with self._job_lock:
            return dict(self.jobs)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._job_lock:
            return self.jobs.get(job_id)

    def _create_job(self, kind: str, meta: Optional[Dict[str, Any]] = None) -> str:
        ts = int(time.time())
        job_id = f"{kind}__{ts}__{len(self.jobs)+1}"
        job = {"id": job_id, "kind": kind, "meta": meta or {}, "status": "running", "started_at": ts, "ended_at": None, "result": None, "error": None}
        with self._job_lock:
            self.jobs[job_id] = job
            self._persist_jobs()
        return job_id

    def _finish_job(self, job_id: str, result: Any = None, error: Optional[str] = None):
        with self._job_lock:
            j = self.jobs.get(job_id)
            if not j:
                return
            j["ended_at"] = int(time.time())
            if error:
                j["status"] = "error"
                j["error"] = error
            else:
                j["status"] = "finished"
                j["result"] = result
            self._persist_jobs()

    def _run_in_background(self, kind: str, fn: Callable, *args, **kwargs) -> str:
        job_id = self._create_job(kind, {"args_repr": str(args)[:200], "kwargs_repr": str(kwargs)[:200]})
        def _runner():
            try:
                res = fn(*args, **kwargs)
                self._finish_job(job_id, result=res)
            except Exception as e:
                logger.exception("Background job %s failed", job_id)
                self._finish_job(job_id, error=str(e))
        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return job_id

    # -----------------------------
    # Fetch helpers (compat calls)
    # -----------------------------
    def _call_fetch_get_candles(self, symbol: str, timeframe: str = "1m", limit: int = 1000, force_provider: Optional[str] = None):
        """Llama a fetcher.get_candles/fetch_ohlcv adaptándose a la firma del fetcher."""
        if not self._fetch:
            raise RuntimeError("No fetch module available")
        f = None
        if hasattr(self._fetch, "get_candles") and callable(getattr(self._fetch, "get_candles")):
            f = getattr(self._fetch, "get_candles")
        elif hasattr(self._fetch, "fetch_ohlcv") and callable(getattr(self._fetch, "fetch_ohlcv")):
            f = getattr(self._fetch, "fetch_ohlcv")
        else:
            raise RuntimeError("Fetcher does not expose get_candles nor fetch_ohlcv")

        sig = inspect.signature(f)
        kwargs = {}
        for p in sig.parameters.values():
            if p.name in ("timeframe", "interval", "tf"):
                kwargs[p.name] = timeframe
            elif p.name in ("limit", "count", "n"):
                kwargs[p.name] = limit
            elif p.name in ("force_provider", "provider"):
                if force_provider:
                    kwargs[p.name] = force_provider
            # ignore others
        try:
            if kwargs:
                return f(symbol, **kwargs)
            # try common positional combination
            try:
                return f(symbol, timeframe, limit)
            except Exception:
                return f(symbol)
        except Exception:
            logger.exception("Fetcher call failed")
            raise

    # -----------------------------
    # Load candles (used by UI)
    # -----------------------------
    def load_candles_source(self, symbol: str, limit: int = 1000, timeframe: Optional[str] = None, force_provider: Optional[str] = None) -> pd.DataFrame:
        """
        Intenta cargar candles de la mejor fuente disponible:
         1) storage.get_ohlcv(asset, interval, start_ms, end_ms)
         2) storage.load_candles(asset, interval)
         3) fetcher.get_candles(...)
         4) cache CSV fallback
        Devuelve DataFrame (ordenado por timestamp asc) con columnas: ts,timestamp,open,high,low,close,volume
        """
        tf = timeframe or "1m"
        # 1) storage.get_ohlcv
        try:
            if self._storage:
                if hasattr(self._storage, "get_ohlcv"):
                    try:
                        now_ms = int(time.time() * 1000)
                        start_ms = max(0, now_ms - limit * 60 * 1000)
                        df = self._storage.get_ohlcv(symbol, tf, start_ms, now_ms)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            return df.sort_values("ts").reset_index(drop=True)
                    except Exception:
                        logger.debug("storage.get_ohlcv attempt failed", exc_info=True)
                # try load_candles
                if hasattr(self._storage, "load_candles"):
                    try:
                        rows = self._storage.load_candles(symbol, tf)
                        if rows:
                            df = pd.DataFrame(rows)
                            return df.sort_values("ts").reset_index(drop=True)
                    except Exception:
                        logger.debug("storage.load_candles failed", exc_info=True)
        except Exception:
            logger.exception("storage read failed")

        # 2) fetcher fallback
        try:
            if self._fetch:
                df = self._call_fetch_get_candles(symbol, timeframe=tf, limit=limit, force_provider=force_provider)
                # if fetch returns DataFrame-like list/dict convert
                if isinstance(df, pd.DataFrame):
                    return df.sort_values("ts").reset_index(drop=True)
                try:
                    return pd.DataFrame(list(df)).sort_values("ts").reset_index(drop=True)
                except Exception:
                    pass
        except Exception:
            logger.exception("fetch fallback failed")

        # 3) CSV cache fallback
        try:
            path_csv = os.path.join(CACHE_DIR, f"{symbol.replace('/','_')}.csv")
            if os.path.exists(path_csv):
                try:
                    df = pd.read_csv(path_csv, parse_dates=["timestamp"])
                    return df.sort_values("timestamp").reset_index(drop=True)
                except Exception:
                    logger.exception("CSV fallback read failed for %s", path_csv)
        except Exception:
            logger.exception("csv fallback failed")
        # empty DF
        return pd.DataFrame(columns=["ts", "timestamp", "open", "high", "low", "close", "volume"])

    # -----------------------------
    # Save candles compat (tries multiple storage APIs)
    # -----------------------------
    def save_candles(self, symbol: str, df: pd.DataFrame, interval: Optional[str] = None) -> bool:
        """
        Guarda velas usando las APIs disponibles:
          - storage.upsert_prices(asset, interval, df)
          - storage.upsert_candles(asset, interval, df)
          - storage.save_candles(asset, df) or storage.save_candles(df, dbpath, asset, interval)
          - fallback CSV cache
        Devuelve True/False.
        """
        if df is None:
            return False
        # ensure DataFrame
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                logger.exception("Provided df not convertible to DataFrame")
                return False
        try:
            if self._storage:
                # upsert_prices (common in storage_postgres)
                if hasattr(self._storage, "upsert_prices"):
                    try:
                        # prefer (asset, interval, df)
                        if interval:
                            out = self._storage.upsert_prices(symbol, interval, df)
                        else:
                            out = self._storage.upsert_prices(symbol, df)
                        logger.debug("upsert_prices returned: %s", out)
                        return True
                    except Exception:
                        logger.debug("upsert_prices failed", exc_info=True)
                # upsert_candles
                if hasattr(self._storage, "upsert_candles"):
                    try:
                        out = self._storage.upsert_candles(symbol, interval or "1m", df)
                        logger.debug("upsert_candles returned: %s", out)
                        return True
                    except Exception:
                        logger.debug("upsert_candles failed", exc_info=True)
                # save_candles (compat wrappers)
                if hasattr(self._storage, "save_candles"):
                    try:
                        # try signature save_candles(symbol, df)
                        out = self._storage.save_candles(symbol, df)
                        logger.debug("save_candles(symbol, df) returned: %s", out)
                        return True
                    except TypeError:
                        # try other signature save_candles(df, db_path, asset, interval)
                        try:
                            out = self._storage.save_candles(df, None, symbol, interval or "1m")
                            logger.debug("save_candles(df, None, symbol, interval) returned: %s", out)
                            return True
                        except Exception:
                            logger.debug("save_candles fallback signature failed", exc_info=True)
                    except Exception:
                        logger.debug("save_candles failed", exc_info=True)
                # make_save_callback
                if hasattr(self._storage, "make_save_callback"):
                    try:
                        cb = self._storage.make_save_callback()
                        if callable(cb):
                            cb(symbol, interval or "1m", df.to_dict("records"))
                            return True
                    except Exception:
                        logger.debug("make_save_callback failed", exc_info=True)
        except Exception:
            logger.exception("storage save attempts failed")

        # fallback: csv cache
        try:
            path = os.path.join(CACHE_DIR, f"{symbol.replace('/','_')}_{interval or '1m'}.csv")
            df.to_csv(path, index=False)
            logger.info("Wrote CSV fallback to %s", path)
            return True
        except Exception:
            logger.exception("CSV fallback write failed")
            return False

    # -----------------------------
    # Indicators application
    # -----------------------------
    def apply_indicators(self, df: pd.DataFrame, indicators: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Aplica indicadores delegando a core.indicators.apply_indicators si existe,
        o usa un fallback mínimo (ema, rsi).
        """
        if df is None or df.empty:
            return df
        try:
            if self._indicators and hasattr(self._indicators, "apply_indicators"):
                try:
                    if indicators is None:
                        return self._indicators.apply_indicators(df.copy(), params=params or {})
                    else:
                        return self._indicators.apply_indicators(df.copy(), indicators=indicators, params=params or {})
                except TypeError:
                    return self._indicators.apply_indicators(df.copy())
        except Exception:
            logger.exception("apply_indicators via core.indicators failed")

        # fallback
        try:
            out = df.copy()
            if "close" in out.columns:
                close = out["close"].astype(float)
                ema_short = int((params or {}).get("ema_short", 9))
                ema_long = int((params or {}).get("ema_long", 50))
                out[f"ema_{ema_short}"] = close.ewm(span=ema_short, adjust=False).mean()
                out[f"ema_{ema_long}"] = close.ewm(span=ema_long, adjust=False).mean()
                period = int((params or {}).get("rsi_period", 14))
                # rsi basic
                delta = close.diff()
                up = delta.clip(lower=0).rolling(period, min_periods=1).mean()
                down = -delta.clip(upper=0).rolling(period, min_periods=1).mean().replace(0, 1e-9)
                rs = up / down
                out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
            return out
        except Exception:
            logger.exception("Fallback indicators failed")
            return df

    # -----------------------------
    # Backfill helpers delegating to Orchestrator if available
    # -----------------------------
    def run_full_backfill(self, symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Orquesta un backfill completo. If an orchestrator module exists, delegate; else loop symbols.
        """
        if self._orchestrator and hasattr(self._orchestrator, "Orchestrator"):
            try:
                Or = getattr(self._orchestrator, "Orchestrator")
                inst = Or()
                if hasattr(inst, "run_full_backfill"):
                    return inst.run_full_backfill(symbols=symbols, per_symbol_limit=per_symbol_limit, timeframe=timeframe)
            except Exception:
                logger.exception("Orchestrator.run_full_backfill call failed; falling back to internal loop")

        # fallback loop
        res = {"started_at": int(time.time()), "results": {}, "finished_at": None}
        syms = symbols or (self.list_assets() if self.list_assets() else [])
        for s in syms:
            try:
                # use fetcher direct if available
                df = self.load_candles_source(s, limit=per_symbol_limit, timeframe=timeframe or "1m")
                saved = False
                if not df.empty:
                    saved = self.save_candles(s, df, interval=timeframe or "1m")
                res["results"][s] = {"rows": len(df), "saved": bool(saved)}
            except Exception as e:
                logger.exception("backfill symbol failed: %s", e)
                res["results"][s] = {"error": str(e)}
        res["finished_at"] = int(time.time())
        return res

    def run_full_backfill_background(self, symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000, timeframe: Optional[str] = None) -> str:
        return self._run_in_background("full_backfill", self.run_full_backfill, symbols, per_symbol_limit, timeframe)

    # -----------------------------
    # Backtest / Train AI
    # -----------------------------
    def run_backtest_for(self, symbol: str, interval: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Delegate to core.backtest or orchestrator if available.
        """
        bt_mod = _import_safe("core.backtest")
        if bt_mod:
            if hasattr(bt_mod, "run_backtest_for"):
                try:
                    return bt_mod.run_backtest_for(symbol, interval, params or {})
                except Exception:
                    logger.exception("backtest.run_backtest_for failed")
        # fallback: no backtest module
        return {"error": "no backtest module available"}

    def train_ai(self, asset: str, interval: str, train_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Entrena IA y guarda modelo usando ai_inference.save_model si existe.
        Devuelve metrics dict from AITrainer.train_model_from_storage or train_model.
        """
        if not self._ai_train:
            raise RuntimeError("No AI training module available")
        trainer_cls = getattr(self._ai_train, "AITrainer", None)
        if trainer_cls is None:
            raise RuntimeError("AI training class not found in core.ai_train")
        trainer = trainer_cls(storage=self._storage, config=train_params or {})
        # train and save via callback if ai_inference exists
        try:
            if hasattr(trainer, "train_model_and_save"):
                if self._ai_inference and hasattr(self._ai_inference, "save_model"):
                    def _save_cb(model, meta):
                        try:
                            # add feature_names from trainer.model_meta if present
                            meta_out = meta if isinstance(meta, dict) else {}
                            if hasattr(trainer, "model_meta") and isinstance(trainer.model_meta, dict):
                                meta_out = {**trainer.model_meta, **meta_out}
                            self._ai_inference.save_model(model, meta_out, asset=asset, interval=interval)
                        except Exception:
                            logger.exception("ai_inference.save_model callback failed")
                    res = trainer.train_model_and_save(asset, interval, save_model_cb=_save_cb)
                    return res
                else:
                    # train but no save callback available
                    res = trainer.train_model_and_save(asset, interval, save_model_cb=None)
                    return res
            elif hasattr(trainer, "train_model_from_storage"):
                res = trainer.train_model_from_storage(asset, interval)
                return res
            else:
                raise RuntimeError("AITrainer missing expected train methods")
        except Exception:
            logger.exception("train_ai failed")
            raise

    def train_ai_background(self, asset: str, interval: str, train_params: Optional[Dict[str, Any]] = None) -> str:
        return self._run_in_background("train_ai", self.train_ai, asset, interval, train_params or {})

    # -----------------------------
    # Assets helpers
    # -----------------------------
    def list_assets(self) -> List[str]:
        try:
            if self._storage and hasattr(self._storage, "list_assets"):
                try:
                    return self._storage.list_assets()
                except Exception:
                    logger.exception("storage.list_assets failed")
            if self._fetch and hasattr(self._fetch, "list_watchlist_assets"):
                try:
                    return self._fetch.list_watchlist_assets()
                except Exception:
                    logger.exception("fetch.list_watchlist_assets failed")
        except Exception:
            logger.exception("list_assets top-level failed")
        return []

    # -----------------------------
    # Health
    # -----------------------------
    def health_status(self) -> Dict[str, Any]:
        return {
            "time": int(time.time()),
            "modules": {
                "fetch": bool(self._fetch),
                "storage": bool(self._storage),
                "indicators": bool(self._indicators),
                "score": bool(self._score),
                "ai_train": bool(self._ai_train),
                "ai_inference": bool(self._ai_inference)
            },
            "jobs_count": len(self.jobs)
        }


# singleton adapter
_adapter_singleton: Optional[CoreAdapter] = None

def get_adapter(singleton: bool = True) -> CoreAdapter:
    global _adapter_singleton
    if singleton and _adapter_singleton:
        return _adapter_singleton
    _adapter_singleton = CoreAdapter()
    return _adapter_singleton


# convenience top-level functions for scripts that import core.adapter directly
_adapter = get_adapter(True)

def load_candles_source(symbol: str, limit: int = 1000, timeframe: Optional[str] = None, force_provider: Optional[str] = None) -> pd.DataFrame:
    return _adapter.load_candles_source(symbol, limit=limit, timeframe=timeframe, force_provider=force_provider)

def save_candles(symbol: str, df: pd.DataFrame, interval: Optional[str] = None) -> bool:
    return _adapter.save_candles(symbol, df, interval=interval)

def run_full_backfill_background(symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000, timeframe: Optional[str] = None) -> str:
    return _adapter.run_full_backfill_background(symbols, per_symbol_limit, timeframe)

def train_ai_background(asset: str, interval: str, train_params: Optional[Dict[str, Any]] = None) -> str:
    return _adapter.train_ai_background(asset, interval, train_params)
