# core/orchestrator.py
"""
Orchestrator limpio y autocontenido para coordinar Fetcher, StorageAdapter, Indicators, Score, Backtest y AI.
Sobrescribe y unifica llamadas que el dashboard y scripts deben usar:
  - backfill_symbol
  - compute_indicators_for
  - compute_scores
  - run_backtest_for
  - train_ai
  - run_full_backfill
  - status / jobs (background)

Diseñado para degradar suavemente si alguno de los módulos (indicators, score, ai, backtest)
no está presente: informa en health_status y devuelve errores controlados en vez de romper.

API principal (métodos públicos):
  Orchestrator(storage=None, fetcher=None, indicators=None, score=None, ai=None, backtest=None, config=None)
    storage: instancia que provea upsert_candles, load_candles, get_ohlcv, list_assets, etc.
    fetcher: instancia con get_candles(symbol, timeframe, limit, force_network) y run_full_backfill (opcional)
    indicators: módulo con apply_indicators(df) (opcional)
    score: módulo con compute_score_safe / compute_score_timeseries / compute_and_persist_scores (opcional)
    ai: módulo/objeto con train_model / predict (opcional)
    backtest: módulo con run_backtest_for (opcional)

Siempre devuelve estructuras sencillas (dict/list/df) y registra errores.
"""

from typing import Optional, List, Dict, Any, Callable
import threading
import time
import uuid
import logging

logger = logging.getLogger("core.orchestrator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# Intentar importar las implementaciones por defecto si existen en el repo
try:
    from .storage_adapter import StorageAdapter  # preferida
except Exception:
    StorageAdapter = None

try:
    from .fetch import Fetcher
except Exception:
    Fetcher = None

try:
    import core.indicators as indicators_module  # type: ignore
except Exception:
    indicators_module = None

try:
    import core.score as score_module  # type: ignore
except Exception:
    score_module = None

try:
    import core.ai_train as ai_module  # type: ignore
except Exception:
    ai_module = None

try:
    import core.backtest as backtest_module  # type: ignore
except Exception:
    backtest_module = None


class Orchestrator:
    """
    Clase Orchestrator: coordina storage, fetcher y módulos de dominio.
    Mantiene un registro de jobs en memoria (self.jobs) para UI/monitorización.
    """

    def __init__(
        self,
        storage: Optional[Any] = None,
        fetcher: Optional[Any] = None,
        indicators: Optional[Any] = None,
        score: Optional[Any] = None,
        ai: Optional[Any] = None,
        backtest: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # Instanciar por defecto si se detectan implementaciones
        if storage is None and StorageAdapter is not None:
            try:
                storage = StorageAdapter()
            except Exception as e:
                logger.debug("No se pudo instanciar StorageAdapter por defecto: %s", e)
                storage = None

        if fetcher is None and Fetcher is not None:
            try:
                fetcher = Fetcher()
            except Exception as e:
                logger.debug("No se pudo instanciar Fetcher por defecto: %s", e)
                fetcher = None

        self.storage = storage
        self.fetcher = fetcher
        self.indicators = indicators or indicators_module
        self.score = score or score_module
        self.ai = ai or ai_module
        self.backtest = backtest or backtest_module
        self.config = config or {}

        # jobs: dict job_id -> {"type": str, "status": "running"|"done"|"error", "result": Any, "started_at":, "ended_at":, "error": str}
        self.jobs: Dict[str, Dict[str, Any]] = {}

        logger.info("Orchestrator inicializado. storage=%s fetcher=%s indicators=%s score=%s ai=%s backtest=%s",
                    bool(self.storage), bool(self.fetcher), bool(self.indicators), bool(self.score), bool(self.ai), bool(self.backtest))

    # -------------------------
    # Jobs / threading helpers
    # -------------------------
    def _start_job(self, job_type: str, target: Callable, *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "type": job_type,
            "status": "running",
            "result": None,
            "started_at": time.time(),
            "ended_at": None,
            "error": None
        }

        def _runner():
            try:
                logger.info("Job %s (%s) started", job_id, job_type)
                res = target(*args, **kwargs)
                self.jobs[job_id]["result"] = res
                self.jobs[job_id]["status"] = "done"
                logger.info("Job %s (%s) done", job_id, job_type)
            except Exception as e:
                self.jobs[job_id]["status"] = "error"
                self.jobs[job_id]["error"] = str(e)
                logger.exception("Job %s (%s) error: %s", job_id, job_type, e)
            finally:
                self.jobs[job_id]["ended_at"] = time.time()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        return self.jobs

    # -------------------------
    # Backfill
    # -------------------------
    def backfill_symbol(self, asset: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, persist: bool = True) -> Dict[str, Any]:
        """
        Backfill simple: pide candles al fetcher y los persiste con storage.upsert_candles.
        start_ms/end_ms se pasan al fetcher si soporta backfill_range, pero aquí intentamos
        una llamada simple universal: fetcher.get_candles(symbol, timeframe=interval, limit=...)
        Devuelve dict con conteo y errores.
        """
        if not self.fetcher:
            raise RuntimeError("No fetcher disponible para backfill_symbol")
        if not self.storage and persist:
            raise RuntimeError("No storage disponible para persistir candles")

        try:
            # try fetcher.backfill_range if exists (preferido)
            if hasattr(self.fetcher, "backfill_range") and callable(self.fetcher.backfill_range):
                try:
                    df = self.fetcher.backfill_range(asset, interval, start_ms, end_ms)
                except TypeError:
                    # signature mismatch: try fallback to get_candles
                    df = self.fetcher.get_candles(asset, timeframe=interval, limit=self.config.get("backfill_limit", 1000))
            else:
                # fallback: single fetch. Some providers ignore start/end and return recent.
                df = self.fetcher.get_candles(asset, timeframe=interval, limit=self.config.get("backfill_limit", 1000), force_network=True)

            # ensure DataFrame-like structure
            try:
                import pandas as pd
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
            except Exception:
                # keep as list/dict fallback
                pass

            # prepare rows
            if df is None:
                rows = []
            else:
                if hasattr(df, "to_dict"):
                    rows = df.to_dict("records")
                else:
                    # assume iterable of dicts
                    rows = list(df)

            # persist
            if persist and self.storage and rows:
                # storage.upsert_candles(asset, interval, rows)
                try:
                    written = self.storage.upsert_candles(asset, interval, rows)
                except Exception:
                    # try fallback to make_save_callback if present
                    if hasattr(self.storage, "make_save_callback"):
                        cb = self.storage.make_save_callback()
                        cb(asset, interval, rows)
                        written = len(rows)
                    else:
                        raise
            else:
                written = len(rows)
            return {"asset": asset, "interval": interval, "rows": written, "error": None}
        except Exception as e:
            logger.exception("backfill_symbol failed for %s %s", asset, interval)
            return {"asset": asset, "interval": interval, "rows": 0, "error": str(e)}

    def backfill_symbol_background(self, asset: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, persist: bool = True) -> str:
        """
        Lanza backfill_symbol en background y devuelve job_id.
        """
        return self._start_job("backfill", self.backfill_symbol, asset, interval, start_ms, end_ms, persist)

    def run_full_backfill(self, symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Orquestador de backfill completo. Si fetcher tiene run_full_backfill lo delega.
        Si storage está presente, intentará persistir.
        """
        results = {}
        try:
            if hasattr(self.fetcher, "run_full_backfill") and callable(self.fetcher.run_full_backfill):
                # prefer fetcher implementation which may know about paging
                try:
                    results = self.fetcher.run_full_backfill(symbols=symbols, per_symbol_limit=per_symbol_limit, timeframe=timeframe, storage_module=self.storage)
                    return results
                except Exception:
                    logger.exception("fetcher.run_full_backfill raised, falling back to orchestrator loop")

            # fallback simple loop
            targets = symbols or (self.storage.list_assets() if self.storage and hasattr(self.storage, "list_assets") else [])
            for sym in targets:
                r = self.backfill_symbol(sym, timeframe or "1m", None, None, persist=bool(self.storage))
                results[sym] = r
            return results
        except Exception as e:
            logger.exception("run_full_backfill failed: %s", e)
            return {"error": str(e)}

    def run_full_backfill_background(self, symbols: Optional[List[str]] = None, per_symbol_limit: int = 1000, timeframe: Optional[str] = None) -> str:
        return self._start_job("full_backfill", self.run_full_backfill, symbols, per_symbol_limit, timeframe)

    # -------------------------
    # Indicators
    # -------------------------
    def compute_indicators_for(self, asset: str, interval: str, lookback: Optional[int] = None) -> Any:
        """
        Carga candles desde storage y aplica indicadores (indicators.apply_indicators).
        Devuelve DataFrame con indicadores si indicators módulo disponible.
        """
        if not self.storage:
            raise RuntimeError("No storage disponible para compute_indicators_for")
        # load candles (try get_ohlcv first)
        df = None
        try:
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
        except Exception:
            logger.exception("Error cargando candles desde storage for %s %s", asset, interval)
            df = None

        if df is None or (hasattr(df, "empty") and df.empty):
            logger.info("No hay candles para %s %s", asset, interval)
            return df  # empty DF

        # apply indicators if available
        if self.indicators and hasattr(self.indicators, "apply_indicators"):
            try:
                df_ind = self.indicators.apply_indicators(df.copy())
                return df_ind
            except Exception:
                logger.exception("apply_indicators falló, devolviendo df base")
                return df
        else:
            # fallback: attempt minimal indicator set (EMA/RSI) inline to avoid breaking UI
            try:
                import pandas as pd
                df_local = df.copy()
                if "close" in df_local.columns:
                    # create ema_9 and ema_40 if missing
                    if "ema_9" not in df_local.columns:
                        df_local["ema_9"] = df_local["close"].ewm(span=9, adjust=False).mean()
                    if "ema_40" not in df_local.columns:
                        df_local["ema_40"] = df_local["close"].ewm(span=40, adjust=False).mean()
                return df_local
            except Exception:
                logger.exception("Fallback minimal indicators failed")
                return df

    def compute_indicators_for_background(self, asset: str, interval: str, lookback: Optional[int] = None) -> str:
        return self._start_job("compute_indicators", self.compute_indicators_for, asset, interval, lookback)

    # -------------------------
    # Scoring
    # -------------------------
    def compute_scores(self, asset: str, interval: str, weights: Optional[Dict[str, float]] = None, persist: bool = False) -> Any:
        """
        Compute scores for an asset/interval. Uses indicators if needed.
        If persist=True and storage has methods to save scores, persist.
        Returns the score timeseries (DataFrame or list of dicts).
        """
        df_ind = self.compute_indicators_for(asset, interval)
        if df_ind is None or (hasattr(df_ind, "empty") and df_ind.empty):
            return []

        try:
            # Prefer score.compute_score_safe (returns list) if available
            if self.score and hasattr(self.score, "compute_score_safe"):
                records = self.score.compute_score_safe(df_ind, None, weights)
                if persist and self.storage:
                    try:
                        if hasattr(self.score, "compute_and_persist_scores"):
                            # let score module handle persistence if implemented
                            self.score.compute_and_persist_scores(df_ind, None, weights, storage=self.storage, asset=asset, interval=interval)
                        else:
                            # fallback persistence: try storage.save_scores or make_save_callback
                            if hasattr(self.storage, "save_scores"):
                                self.storage.save_scores(asset, interval, records)
                            elif hasattr(self.storage, "make_save_callback"):
                                cb = self.storage.make_save_callback()
                                cb(asset, interval, records)
                    except Exception:
                        logger.exception("Error persisting scores for %s", asset)
                return records
            # else use compute_score_timeseries to get DataFrame
            if self.score and hasattr(self.score, "compute_score_timeseries"):
                df_scores = self.score.compute_score_timeseries(df_ind, None, weights)
                if persist and self.storage:
                    try:
                        rows = df_scores.to_dict("records")
                        if hasattr(self.storage, "save_scores"):
                            self.storage.save_scores(asset, interval, rows)
                        elif hasattr(self.storage, "make_save_callback"):
                            cb = self.storage.make_save_callback()
                            cb(asset, interval, rows)
                    except Exception:
                        logger.exception("Error persisting df_scores for %s", asset)
                return df_scores
            # fallback: attempt to call a simple compute in score module
            if self.score and hasattr(self.score, "compute_score_safe"):
                return self.score.compute_score_safe(df_ind, None, weights)
            # last fallback: return empty
            logger.info("No score module available, returning empty for %s %s", asset, interval)
            return []
        except Exception as e:
            logger.exception("compute_scores failed for %s %s", asset, interval)
            return {"error": str(e)}

    def compute_scores_background(self, asset: str, interval: str, weights: Optional[Dict[str, float]] = None, persist: bool = False) -> str:
        return self._start_job("compute_scores", self.compute_scores, asset, interval, weights, persist)

    # -------------------------
    # Backtesting
    # -------------------------
    def run_backtest_for(self, asset: str, interval: str, strategy_params: Optional[Dict[str, Any]] = None, persist: bool = False) -> Dict[str, Any]:
        """
        Ejecuta backtest si backtest module está disponible. Persiste resultado si storage ofrece método.
        """
        if not self.backtest:
            raise RuntimeError("No backtest module available")
        try:
            # prefer backtest.run_backtest_for signature
            if hasattr(self.backtest, "run_backtest_for"):
                results = self.backtest.run_backtest_for(asset, interval, strategy_params or {})
            elif hasattr(self.backtest, "run_backtest"):
                results = self.backtest.run_backtest(asset, interval, strategy_params or {})
            else:
                raise RuntimeError("backtest module no expone run_backtest_for/run_backtest")
            # persist if requested and storage has suitable method
            if persist and self.storage:
                try:
                    if hasattr(self.storage, "save_backtest"):
                        self.storage.save_backtest(asset, interval, results)
                    elif hasattr(self.storage, "make_save_callback"):
                        cb = self.storage.make_save_callback()
                        cb(f"backtest:{asset}", interval, [results])
                except Exception:
                    logger.exception("Error persisting backtest for %s", asset)
            return results
        except Exception as e:
            logger.exception("run_backtest_for failed for %s %s", asset, interval)
            return {"error": str(e)}

    def run_backtest_for_background(self, asset: str, interval: str, strategy_params: Optional[Dict[str, Any]] = None, persist: bool = False) -> str:
        return self._start_job("backtest", self.run_backtest_for, asset, interval, strategy_params, persist)

    # -------------------------
    # AI / Training
    # -------------------------
    def train_ai(self, asset: str, interval: str, train_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Entrena modelo IA si ai module expuesto. Debe devolver metadata del modelo/metricas.
        """
        if not self.ai:
            raise RuntimeError("No AI module available")
        try:
            # prefer AITrainer class if present
            if hasattr(self.ai, "AITrainer"):
                trainer_cls = getattr(self.ai, "AITrainer")
                trainer = trainer_cls(storage=self.storage, config=train_params or {})
                if hasattr(trainer, "train_model"):
                    res = trainer.train_model_from_storage(asset, interval) if hasattr(trainer, "train_model_from_storage") else trainer.train_model()
                    return res
                raise RuntimeError("AITrainer no expone train_model")
            elif hasattr(self.ai, "train_model"):
                res = self.ai.train_model(self.storage, asset, interval, train_params or {})
                return res
            else:
                raise RuntimeError("ai module no expone AITrainer ni train_model")
        except Exception as e:
            logger.exception("train_ai failed for %s %s", asset, interval)
            return {"error": str(e)}

    def train_ai_background(self, asset: str, interval: str, train_params: Optional[Dict[str, Any]] = None) -> str:
        return self._start_job("train_ai", self.train_ai, asset, interval, train_params)

    # -------------------------
    # Utilities / health
    # -------------------------
    def health_status(self) -> Dict[str, Any]:
        status = {
            "storage": bool(self.storage),
            "fetcher": bool(self.fetcher),
            "indicators": bool(self.indicators),
            "score": bool(self.score),
            "ai": bool(self.ai),
            "backtest": bool(self.backtest),
            "jobs_running": sum(1 for j in self.jobs.values() if j["status"] == "running"),
        }
        # if storage has DB URL or engine, try to give more info
        try:
            if self.storage and hasattr(self.storage, "database_url"):
                status["database_url"] = getattr(self.storage, "database_url")
        except Exception:
            pass
        return status

    def list_assets(self) -> List[str]:
        if self.storage and hasattr(self.storage, "list_assets"):
            try:
                return self.storage.list_assets()
            except Exception:
                logger.exception("list_assets failed")
                return []
        return []

    # -------------------------
    # Convenience compatibility shims
    # -------------------------
    # alias esperado por algunos tests/historico
    # (si se desea, exportar TradingOrchestrator = Orchestrator en módulo)
    pass  # class ends here


# Expose alias for backward compatibility
TradingOrchestrator = Orchestrator


def run_full_pipeline(*args, **kwargs):
    """
    Compat shim: crea una instancia de Orchestrator con los parámetros pasados (si alguno)
    y devuelve la instancia lista para usarse. No lanza jobs por sí mismo.
    """
    orch = Orchestrator(*args, **kwargs)
    return orch
