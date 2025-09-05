# core/orchestrator.py
"""
Orchestrator: coordina Fetcher, Storage, Indicators, Score, Backtest y AI.
Objetivo: ser tolerante a ausencias de módulos y exponer una API estable que use
las funciones disponibles en los módulos core.*.

Principales responsabilidades:
- backfill_symbol(asset, interval, start_ms=None, end_ms=None, persist=True)
- run_full_backfill(symbols, per_symbol_limit, timeframe)
- compute_indicators_for(asset, interval, lookback=None)
- compute_scores(asset, interval, lookback=None)
- run_backtest_for(asset, interval)
- train_ai(asset, interval, train_params)
- helpers para jobs en background
"""

from typing import Any, Dict, Optional, List, Callable
import importlib
import inspect
import json
import threading
import time
import logging
import os

logger = logging.getLogger("core.orchestrator")
logger.addHandler(logging.NullHandler())

# Safe import helper: devuelve None si no existe
def _import_safe(module_path: str, attr: Optional[str] = None):
    try:
        mod = importlib.import_module(module_path)
        if attr:
            return getattr(mod, attr)
        return mod
    except Exception:
        return None


# Intentamos enlazar con implementaciones reales si están
fetcher = _import_safe("core.fetch")
storage = _import_safe("core.storage_postgres") or _import_safe("core.storage") or _import_safe("core.storage_adapter")
indicators = _import_safe("core.indicators")
score_mod = _import_safe("core.score")
backtest_mod = _import_safe("core.backtest")
ai_train = _import_safe("core.ai_train")
ai_inference = _import_safe("core.ai_inference") or _import_safe("core.ai_interference")  # fallback al typo existente

# Cargar configuración mínima si existe config.json al root del repo
DEFAULT_CONFIG = {}
try:
    root = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(root, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf8") as fh:
            DEFAULT_CONFIG = json.load(fh)
except Exception:
    DEFAULT_CONFIG = {}

class Orchestrator:
    def __init__(
        self,
        fetcher_obj: Any = None,
        storage_obj: Any = None,
        indicators_obj: Any = None,
        score_obj: Any = None,
        ai_obj: Any = None,
        backtest_obj: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.fetcher = fetcher_obj or fetcher
        self.storage = storage_obj or storage
        self.indicators = indicators_obj or indicators
        self.score = score_obj or score_mod
        self.ai = ai_obj or ai_train  # trainer module expected; inference separate
        self.backtest = backtest_obj or backtest_mod
        self.config = config or DEFAULT_CONFIG or {}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._job_counter = 0

        logger.info("Orchestrator inicializado. storage=%s fetcher=%s indicators=%s score=%s ai=%s backtest=%s",
                    bool(self.storage), bool(self.fetcher), bool(self.indicators),
                    bool(self.score), bool(self.ai), bool(self.backtest))

    # ----------------------------
    # Job helpers
    # ----------------------------
    def _next_job_id(self, prefix: str) -> str:
        self._job_counter += 1
        return f"{prefix}-{int(time.time())}-{self._job_counter}"

    def _start_job(self, job_type: str, fn: Callable, *args, **kwargs) -> str:
        job_id = self._next_job_id(job_type)
        self.jobs[job_id] = {"status": "running", "started_at": time.time(), "ended_at": None, "error": None, "result": None}

        def _runner():
            try:
                res = fn(*args, **kwargs)
                self.jobs[job_id]["result"] = res
                self.jobs[job_id]["status"] = "finished"
            except Exception as e:
                logger.exception("Job %s (%s) error: %s", job_id, job_type, e)
                self.jobs[job_id]["error"] = str(e)
                self.jobs[job_id]["status"] = "error"
            finally:
                self.jobs[job_id]["ended_at"] = time.time()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        return self.jobs

    # ----------------------------
    # Fetcher helper: llamar a get_candles adaptándose a su firma
    # ----------------------------
    def _call_get_candles(self, symbol: str, timeframe: str = "1m", limit: int = 1000, force_provider: Optional[str] = None):
        """
        Llama a fetcher.get_candles adaptándose a la firma que ofrezca.
        Evitamos pasar 'force_network' porque hay implementaciones sin ese arg.
        """
        if not self.fetcher:
            raise RuntimeError("No fetcher disponible")

        # buscar función get_candles
        f = None
        if hasattr(self.fetcher, "get_candles") and callable(getattr(self.fetcher, "get_candles")):
            f = getattr(self.fetcher, "get_candles")
        elif hasattr(self.fetcher, "fetch_ohlcv") and callable(getattr(self.fetcher, "fetch_ohlcv")):
            # provider style: fetch_ohlcv(symbol, timeframe, limit)
            f = getattr(self.fetcher, "fetch_ohlcv")

        if f is None:
            raise RuntimeError("Fetcher no expone get_candles ni fetch_ohlcv")

        sig = inspect.signature(f)
        kwargs = {}
        # try common param names
        for p in sig.parameters.values():
            if p.name in ("symbol", "asset") and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                # symbol mapped via positional below
                pass
            elif p.name in ("timeframe", "time_frame", "tf", "interval"):
                kwargs[p.name] = timeframe
            elif p.name in ("limit", "count", "n"):
                kwargs[p.name] = limit
            elif p.name in ("force_provider", "provider"):
                if force_provider:
                    kwargs[p.name] = force_provider
            elif p.name in ("start_ms", "from_ts", "since"):
                # do not populate by default
                pass
            # ignore unknown params
        try:
            # prefer calling with keyword args when possible
            if kwargs:
                return f(symbol, **kwargs)
            else:
                # fallback to simple calls
                return f(symbol, timeframe, limit) if len(sig.parameters) >= 3 else f(symbol)
        except TypeError as te:
            # último recurso: intentar las llamadas habituales sin kwargs
            logger.debug("get_candles fallback TypeError: %s", te)
            try:
                return f(symbol, timeframe, limit)
            except Exception:
                return f(symbol)

    # ----------------------------
    # Storage helper: escribir velas con varios nombres posibles
    # ----------------------------
    def _persist_candles(self, asset: str, interval: str, rows: List[Dict[str, Any]]) -> int:
        """
        Persistir rows (lista de dicts) usando la API que encuentre en storage:
        intenta upsert_prices/upsert_candles/save_candles o callback fallback.
        Devuelve número de filas escritas (estimado) o lanza excepción.
        """
        if not self.storage:
            raise RuntimeError("No storage disponible para persistir candles")

        # transformar rows a DataFrame local si storage no soporta DataFrames
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
        except Exception:
            df = None

        # intentos por nombre
        try:
            # 1) upsert_prices(asset, interval, df) es preferible en PostgresStorage
            if hasattr(self.storage, "upsert_prices"):
                try:
                    n = self.storage.upsert_prices(asset, interval, df)
                    return n if isinstance(n, int) else (len(rows) if rows else 0)
                except TypeError:
                    # intentar con (asset, df) por compat (no pasa interval)
                    try:
                        n = self.storage.upsert_prices(asset, df)
                        return n if isinstance(n, int) else (len(rows) if rows else 0)
                    except Exception:
                        logger.debug("upsert_prices fallback failed", exc_info=True)

            # 2) upsert_candles
            if hasattr(self.storage, "upsert_candles"):
                try:
                    n = self.storage.upsert_candles(asset, interval, df)
                    return n if isinstance(n, int) else (len(rows) if rows else 0)
                except TypeError:
                    try:
                        n = self.storage.upsert_candles(df)  # some legacy shape
                        return n if isinstance(n, int) else (len(rows) if rows else 0)
                    except Exception:
                        logger.debug("upsert_candles fallback failed", exc_info=True)

            # 3) save_candles(symbol, df) compat wrappers
            if hasattr(self.storage, "save_candles"):
                try:
                    ok = self.storage.save_candles(asset, df)
                    if isinstance(ok, bool):
                        return len(rows) if ok else 0
                    if isinstance(ok, int):
                        return ok
                except TypeError:
                    # maybe signature (df, db_path, asset, interval) -> try call in that way
                    try:
                        ok = self.storage.save_candles(df, None, asset, interval)
                        if isinstance(ok, bool):
                            return len(rows) if ok else 0
                        if isinstance(ok, int):
                            return ok
                    except Exception:
                        logger.debug("save_candles fallback failed", exc_info=True)

            # 4) make_save_callback provided by storage
            if hasattr(self.storage, "make_save_callback"):
                cb = self.storage.make_save_callback()
                if callable(cb):
                    cb(asset, interval, rows)
                    return len(rows)
        except Exception as e:
            logger.exception("Persistencia de candles falló: %s", e)
            raise

        # última opción: guardar CSV en data/cache si existe carpeta (no es ideal)
        try:
            import csv, os
            root = os.path.dirname(os.path.dirname(__file__))
            cache_dir = os.path.join(root, "data", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"{asset.replace('/','_')}_{interval}.csv")
            # write header from keys of first row
            if rows:
                with open(path, "w", newline="", encoding="utf8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            return len(rows)
        except Exception:
            logger.exception("CSV fallback persist failed")
            raise RuntimeError("No persistence method available for candles")

    # ----------------------------
    # Backfill single symbol
    # ----------------------------
    def backfill_symbol(self, asset: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, persist: bool = True) -> Dict[str, Any]:
        """
        Backfill simple: pide candles al fetcher y los persiste con storage.
        start_ms/end_ms se pasan al fetcher si soporta backfill_range, pero aquí intentamos
        una llamada universal: preferimos fetcher.backfill_range si existe, sino get_candles.
        Devuelve dict con conteo y errores.
        """
        if not self.fetcher:
            raise RuntimeError("No fetcher disponible para backfill_symbol")
        if persist and not self.storage:
            raise RuntimeError("No storage disponible para persistir candles")

        try:
            df = None
            # prefer backfill_range if lo implementa (muchos fetchers lo hacen)
            if hasattr(self.fetcher, "backfill_range") and callable(getattr(self.fetcher, "backfill_range")):
                try:
                    # la función puede aceptar (asset, interval, start_ms, end_ms) o (symbol, start, end)
                    br = getattr(self.fetcher, "backfill_range")
                    sig = inspect.signature(br)
                    kwargs = {}
                    if "interval" in sig.parameters:
                        kwargs["interval"] = interval
                    if "start_ms" in sig.parameters:
                        kwargs["start_ms"] = start_ms
                    if "end_ms" in sig.parameters:
                        kwargs["end_ms"] = end_ms
                    # llamar con lo que acepte
                    df = br(asset, **kwargs) if kwargs else br(asset)
                except TypeError:
                    # fallback a get_candles
                    df = None
                except Exception:
                    logger.exception("fetcher.backfill_range falló, fallback a get_candles")
                    df = None

            # fallback: single fetch con get_candles (uso de nuestra capa de compat)
            if df is None:
                limit = int(self.config.get("backfill_limit", 1000))
                # force_provider si config lo sugiere
                force_provider = self.config.get("force_provider", None)
                df = self._call_get_candles(asset, timeframe=interval, limit=limit, force_provider=force_provider)

            # normalizar a lista de rows (dicts)
            rows = []
            if df is None:
                rows = []
            else:
                try:
                    # si es DataFrame
                    import pandas as pd
                    if isinstance(df, pd.DataFrame):
                        # asegurar columnas ts o timestamp
                        if "ts" not in df.columns and "timestamp" in df.columns:
                            # convertir timestamp a ts millis si es datetime
                            try:
                                df = df.copy()
                                if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                                    df["ts"] = (df["timestamp"].astype("int64") // 10**6)
                                else:
                                    # intentar parse
                                    df["ts"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 10**6
                            except Exception:
                                pass
                        # normalizar columnas esperadas
                        # keep only expected set if present
                        expected = ["ts", "timestamp", "open", "high", "low", "close", "volume"]
                        present = [c for c in expected if c in df.columns]
                        rows = df[present].to_dict("records")
                    else:
                        # iterable of dicts or list
                        rows = list(df)
                except Exception:
                    logger.exception("Error procesando resultado de fetcher into rows; attempting naive conversion")
                    try:
                        rows = list(df)
                    except Exception:
                        rows = []

            # persistir si procede
            if persist and rows:
                written = self._persist_candles(asset, interval, rows)
            else:
                written = len(rows)

            return {"asset": asset, "interval": interval, "rows": written, "error": None}
        except Exception as e:
            logger.exception("backfill_symbol failed for %s %s", asset, interval)
            return {"asset": asset, "interval": interval, "rows": 0, "error": str(e)}

    # ----------------------------
    # Run full backfill (multiple symbols)
    # ----------------------------
    def run_full_backfill(self, symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Orquesta un backfill completo. Si el fetcher expone run_full_backfill, lo delega.
        Sino itera por symbols y llama backfill_symbol.
        """
        res = {"started_at": time.time(), "results": {}, "finished_at": None, "error": None}
        try:
            # prefer fetcher-level run_full_backfill if expone uno
            if hasattr(self.fetcher, "run_full_backfill") and callable(self.fetcher.run_full_backfill):
                try:
                    out = self.fetcher.run_full_backfill(symbols=symbols, limit=per_symbol_limit, timeframe=timeframe)
                    res["results"] = out
                    res["finished_at"] = time.time()
                    return res
                except Exception:
                    logger.exception("fetcher.run_full_backfill failed; fallback to orchestrator loop")

            syms = symbols or (self.storage.list_assets() if (self.storage and hasattr(self.storage, "list_assets")) else [])
            if not syms:
                logger.info("No symbols provided for run_full_backfill")
                res["error"] = "no symbols"
                return res

            for s in syms:
                try:
                    r = self.backfill_symbol(s, timeframe or "1m", persist=True)
                    res["results"][s] = r
                except Exception as e:
                    logger.exception("backfill for %s failed: %s", s, e)
                    res["results"][s] = {"error": str(e)}
            res["finished_at"] = time.time()
            return res
        except Exception as e:
            logger.exception("run_full_backfill failed: %s", e)
            res["error"] = str(e)
            return res

    def run_full_backfill_background(self, symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000, timeframe: Optional[str] = None) -> str:
        return self._start_job("full_backfill", self.run_full_backfill, symbols, per_symbol_limit, timeframe)

    # ----------------------------
    # Indicators & score
    # ----------------------------
    def compute_indicators_for(self, asset: str, interval: str, lookback: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load candles from storage and compute indicators (delegar a core.indicators si existe).
        Devuelve resumen con filas y nombres de columnas añadidas.
        """
        if not self.storage:
            raise RuntimeError("No storage disponible para compute_indicators_for")

        try:
            # cargar candles from storage: prefer storage.get_ohlcv or load_candles
            df = None
            if hasattr(self.storage, "get_ohlcv"):
                try:
                    now_ms = int(time.time() * 1000)
                    start_ms = 0 if lookback is None else max(0, now_ms - lookback)
                    df = self.storage.get_ohlcv(asset, interval, start_ms, now_ms)
                except Exception:
                    df = None
            if (df is None or (hasattr(df, "empty") and df.empty)) and hasattr(self.storage, "load_candles"):
                rows = self.storage.load_candles(asset, interval, None, None)
                import pandas as pd
                df = pd.DataFrame(rows) if rows else pd.DataFrame()

            if df is None:
                return {"asset": asset, "interval": interval, "rows": 0, "cols": [], "error": "no candles loaded"}

            # delegate to indicators module if present
            if self.indicators and hasattr(self.indicators, "apply_indicators"):
                try:
                    out = self.indicators.apply_indicators(df.copy(), params=params or {})
                except TypeError:
                    out = self.indicators.apply_indicators(df.copy())
            else:
                # fallback: add minimal EMA/RSI
                import pandas as pd
                df = df.copy()
                if "close" in df.columns:
                    close = df["close"].astype(float)
                    ema_short = int((params or {}).get("ema_short", 9))
                    ema_long = int((params or {}).get("ema_long", 50))
                    df[f"ema_{ema_short}"] = close.ewm(span=ema_short, adjust=False).mean()
                    df[f"ema_{ema_long}"] = close.ewm(span=ema_long, adjust=False).mean()
                    # basic rsi
                    period = int((params or {}).get("rsi_period", 14))
                    delta = close.diff()
                    up = delta.clip(lower=0).rolling(period).mean()
                    down = -delta.clip(upper=0).rolling(period).mean()
                    rs = up / (down.replace(0, 1e-9))
                    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
                out = df

            # try to persist indicators if storage offers method
            try:
                if hasattr(self.storage, "save_indicators"):
                    # storage.save_indicators expects DataFrame or rows
                    try:
                        self.storage.save_indicators(out if not hasattr(out, "to_dict") else out.to_dict("records"))
                    except Exception:
                        try:
                            self.storage.save_indicators(out)
                        except Exception:
                            logger.debug("storage.save_indicators fallback failed", exc_info=True)
            except Exception:
                logger.exception("persist indicators failed")

            cols = list(out.columns) if hasattr(out, "columns") else []
            rows = len(out) if hasattr(out, "__len__") else None
            return {"asset": asset, "interval": interval, "rows": rows, "cols": cols, "error": None}
        except Exception as e:
            logger.exception("compute_indicators_for failed: %s", e)
            return {"asset": asset, "interval": interval, "rows": 0, "cols": [], "error": str(e)}

    def compute_scores(self, asset: str, interval: str, lookback: Optional[int] = None, score_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute scores for asset/interval based on candles+indicators. Delegar a core.score si existe.
        Devuelve summary con última puntuación.
        """
        if not self.storage:
            raise RuntimeError("No storage disponible para compute_scores")

        try:
            # load candles+indicators similar a compute_indicators_for
            df = None
            if hasattr(self.storage, "get_ohlcv"):
                try:
                    now_ms = int(time.time() * 1000)
                    start_ms = 0 if lookback is None else max(0, now_ms - lookback)
                    df = self.storage.get_ohlcv(asset, interval, start_ms, now_ms)
                except Exception:
                    df = None
            if (df is None or (hasattr(df, "empty") and df.empty)) and hasattr(self.storage, "load_candles"):
                rows = self.storage.load_candles(asset, interval, None, None)
                import pandas as pd
                df = pd.DataFrame(rows) if rows else pd.DataFrame()

            if df is None or (hasattr(df, "empty") and df.empty):
                return {"asset": asset, "interval": interval, "score": None, "error": "no candles"}

            if self.score and hasattr(self.score, "compute_score_timeseries"):
                out_df = self.score.compute_score_timeseries(df, score_config or self.config.get("score_config"))
                # optionally persist using score module's helpers
                if hasattr(self.score, "persist_scores") and self.storage:
                    try:
                        self.score.persist_scores(out_df, storage_module=self.storage, asset=asset, interval=interval)
                    except Exception:
                        logger.debug("score.persist_scores failed", exc_info=True)
                # return latest
                if out_df is None or out_df.empty:
                    return {"asset": asset, "interval": interval, "score": None, "error": "score computation empty"}
                last = out_df.iloc[-1]
                return {"asset": asset, "interval": interval, "score": float(last.get("score", None)), "components": last.to_dict(), "error": None}
            else:
                # fallback minimal score: normalized close change
                import pandas as pd
                df = df.copy()
                if "close" in df.columns:
                    last_close = float(df["close"].iloc[-1])
                    prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else last_close
                    score = max(0.0, min(1.0, (last_close - prev_close) / max(1e-9, prev_close) / 0.01 + 0.5))
                else:
                    score = None
                return {"asset": asset, "interval": interval, "score": score, "error": None}
        except Exception as e:
            logger.exception("compute_scores failed: %s", e)
            return {"asset": asset, "interval": interval, "score": None, "error": str(e)}

    # ----------------------------
    # Backtest / AI
    # ----------------------------
    def run_backtest_for(self, asset: str, interval: str, strategy_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.backtest:
            raise RuntimeError("No backtest module available")
        try:
            if hasattr(self.backtest, "run_backtest_for"):
                return self.backtest.run_backtest_for(asset, interval, strategy_params or {})
            # fallback if backtest exposes a different function name
            if hasattr(self.backtest, "run_backtest"):
                return self.backtest.run_backtest(asset, interval, strategy_params or {})
            raise RuntimeError("backtest module no expone run_backtest_for ni run_backtest")
        except Exception as e:
            logger.exception("run_backtest_for failed: %s", e)
            return {"error": str(e)}

    def train_ai(self, asset: str, interval: str, train_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Entrena la IA: intenta preferir una clase AITrainer con método train_model_from_storage,
        o funciones expuestas en core.ai_train.
        """
        if not self.ai:
            raise RuntimeError("No AI training module available")
        try:
            # prefer AITrainer class if present
            if hasattr(self.ai, "AITrainer"):
                trainer_cls = getattr(self.ai, "AITrainer")
                trainer = trainer_cls(storage=self.storage, config=train_params or {})
                if hasattr(trainer, "train_model_from_storage"):
                    return trainer.train_model_from_storage(asset, interval, train_params or {})
                if hasattr(trainer, "train_model"):
                    return trainer.train_model(asset, interval, train_params or {})
                raise RuntimeError("AITrainer no expone train_model ni train_model_from_storage")
            elif hasattr(self.ai, "train_model"):
                # function style
                return self.ai.train_model(self.storage, asset, interval, train_params or {})
            else:
                raise RuntimeError("ai module no expone AITrainer ni train_model")
        except Exception as e:
            logger.exception("train_ai failed: %s", e)
            raise

    def train_ai_background(self, asset: str, interval: str, train_params: Optional[Dict[str, Any]] = None) -> str:
        return self._start_job("train_ai", self.train_ai, asset, interval, train_params or {})

    # ----------------------------
    # Health / status
    # ----------------------------
    def health_status(self) -> Dict[str, Any]:
        return {
            "time": time.time(),
            "modules": {
                "fetch": bool(self.fetcher),
                "storage": bool(self.storage),
                "indicators": bool(self.indicators),
                "score": bool(self.score),
                "backtest": bool(self.backtest),
                "ai_train": bool(self.ai),
                "ai_inference": bool(ai_inference),
            }
        }

# Expone una instancia global conveniente
_orch_singleton: Optional[Orchestrator] = None

def get_orchestrator(singleton: bool = True) -> Orchestrator:
    global _orch_singleton
    if singleton and _orch_singleton:
        return _orch_singleton
    _orch_singleton = Orchestrator()
    return _orch_singleton


# Si se ejecuta directamente, un pequeño demo local (no tocar producción)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orch = get_orchestrator(False)
    print("Orchestrator demo. Health:", orch.health_status())
