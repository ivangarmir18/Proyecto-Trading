# core/orchestrator.py
"""
Orchestrator: coordina fetch, indicadores, scoring, entrenamiento y backtests.

Provee métodos públicos que la UI y los scripts esperan:
  - fetch_data(assets, interval, start_ms=None, end_ms=None)
  - compute_indicators_for(asset, interval, lookback=1000) -> pd.DataFrame (indicators/features)
  - compute_and_store_indicators(asset, interval, lookback=1000) -> stores indicators if storage supports it
  - train_model(asset, interval, model_name, model_type='auto', model_params=None)
  - compute_scores(asset, interval, model_name=None, lookback=1000) -> calls score.infer_and_persist if available
  - run_backfill_for(asset, interval, start_ms, end_ms) -> uses Fetcher.backfill_range
  - run_backtest_for(asset, interval, start_ts=None, end_ts=None) -> delegates to core.backtest.run_backtest_for if present

Robusto frente a falta de módulos: comprueba existencia de funciones y lanza errores informativos.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Intentar importar componentes opcionales (fallar con advertencias si no están)
try:
    from core.storage_postgres import make_storage_from_env, PostgresStorage
except Exception:
    PostgresStorage = None
    make_storage_from_env = None

try:
    from core.fetch import Fetcher
except Exception:
    Fetcher = None

try:
    import core.score as score_module
except Exception:
    score_module = None

try:
    import core.backtest as backtest_module
except Exception:
    backtest_module = None


class Orchestrator:
    """
    Orchestrador central.

    storage: instancia que implemente la API usada (PostgresStorage)
    fetcher: instancia de Fetcher (si no se provee, se intentará crear una)
    config: dict opcional pasado hacia Fetcher / comportamiento
    """

    def __init__(self, storage: Optional[Any] = None, fetcher: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        # Storage
        if storage is None:
            if make_storage_from_env is None:
                raise RuntimeError("No se proporcionó storage y make_storage_from_env no está disponible.")
            storage = make_storage_from_env()
        self.storage = storage

        # Fetcher
        if fetcher is None:
            if Fetcher is not None:
                self.fetcher = Fetcher(storage=self.storage, config=config or {})
            else:
                self.fetcher = None
        else:
            self.fetcher = fetcher

        # keep config
        self.config = config or {}

    # -----------------------
    # Fetch helpers
    # -----------------------
    def fetch_data(self, assets: List[str], interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple assets. Returns dict asset -> DataFrame.
        No hace persistencia automática; solo retorna los DataFrames.
        """
        results: Dict[str, pd.DataFrame] = {}
        if not self.fetcher:
            raise RuntimeError("Fetcher no disponible en Orchestrator.")
        for asset in assets:
            try:
                df = self.fetcher.fetch_ohlcv(asset, interval, start_ms=start_ms, end_ms=end_ms)
                # Normalizar: asegurar columnas y tipos
                if df is None:
                    df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
                else:
                    # force numeric types and ts int
                    if "ts" in df.columns:
                        df["ts"] = df["ts"].astype("int64")
                    for c in ["open", "high", "low", "close", "volume"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                results[asset] = df
            except Exception:
                logger.exception("fetch_data failed for asset %s", asset)
                results[asset] = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        return results

    # -----------------------
    # Indicators / features
    # -----------------------
    def compute_indicators_for(self, asset: str, interval: str, lookback: int = 1000) -> pd.DataFrame:
        """
        Calcula features/indicadores para asset+interval.
        Flujo:
          1) intenta cargar velas desde storage.load_candles(limit=lookback)
          2) si no hay velas, intenta fetcher.fetch_ohlcv para rellenar (no persistir)
          3) si core.score.features_from_candles está disponible, la usa; sino aplica indicadores simples
        Devuelve DataFrame de features indexado por datetime, y con columna 'ts' (ms).
        """
        # 1) obtener candles
        df = None
        try:
            if hasattr(self.storage, "load_candles"):
                df = self.storage.load_candles(asset, interval, limit=lookback)
        except Exception:
            logger.exception("storage.load_candles falló, intentaremos fetch directo")

        # fallback fetch if no candles
        if (df is None) or df.empty:
            if self.fetcher:
                try:
                    df = self.fetcher.fetch_ohlcv(asset, interval, start_ms=None, end_ms=None)
                except Exception:
                    logger.exception("fetcher.fetch_ohlcv también falló")
                    df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            else:
                df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        # compute features
        if score_module and hasattr(score_module, "features_from_candles"):
            try:
                feats = score_module.features_from_candles(df)
                return feats
            except Exception:
                logger.exception("score.features_from_candles lanzó excepción; aplicando fallback")
        # fallback simple: devolver columnas passthrough y returns
        try:
            df2 = df.copy()
            if "ts" in df2.columns:
                df2["ts"] = df2["ts"].astype("int64")
                df2["ts_dt"] = pd.to_datetime(df2["ts"], unit="ms", utc=True)
                df2 = df2.set_index("ts_dt", drop=False)
            df2["ret_1"] = np.log(pd.to_numeric(df2["close"], errors="coerce")) - np.log(pd.to_numeric(df2["close"], errors="coerce")).shift(1)
            # fill basic moving averages
            df2["ma_5"] = pd.to_numeric(df2["close"], errors="coerce").rolling(5, min_periods=1).mean()
            return df2[["ts", "close", "ret_1", "ma_5"]].copy()
        except Exception:
            logger.exception("Fallback indicators failed; devolviendo dataframe vacío")
            return pd.DataFrame()

    def compute_and_store_indicators(self, asset: str, interval: str, lookback: int = 1000) -> Optional[pd.DataFrame]:
        """
        Calcula indicadores y, si storage expone save_indicators, intenta guardarlos.
        Retorna DataFrame calculado o None si fallo.
        """
        feats = self.compute_indicators_for(asset, interval, lookback=lookback)
        if feats is None or feats.empty:
            logger.info("No indicators computed for %s %s", asset, interval)
            return feats
        # intentar persistir
        try:
            if hasattr(self.storage, "save_indicators"):
                self.storage.save_indicators(feats, asset, interval)
            else:
                logger.debug("storage.save_indicators no implementado; se omite persistencia de indicadores")
        except Exception:
            logger.exception("save_indicators falló")
        return feats

    # -----------------------
    # Scoring / train / infer
    # -----------------------
    def train_model(self, asset: str, interval: str, model_name: str, model_type: str = "auto", model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Construye dataset (features + target) y entrena un modelo usando score.train_and_persist.
        Requisitos: core.score.train_and_persist disponible y storage con load_candles.
        Retorna metadata dict (o lanza error informativo).
        """
        if score_module is None or not hasattr(score_module, "train_and_persist"):
            raise RuntimeError("Module core.score.train_and_persist no disponible.")

        # cargar velas
        try:
            df = self.storage.load_candles(asset, interval, limit=5000)
        except Exception as e:
            logger.exception("load_candles falló en train_model")
            raise RuntimeError("No se pudieron cargar velas para entrenamiento.") from e

        if df is None or df.empty:
            raise RuntimeError("No hay velas para entrenar")

        # features y target
        feats = score_module.features_from_candles(df)
        target = score_module.make_target(df, horizon=1)
        # alinear
        target = target.reindex(feats.index).dropna()
        X = feats.reindex(target.index).drop(columns=["ts"], errors="ignore").fillna(0)

        # delegar entrenamiento
        res = score_module.train_and_persist(self.storage, asset, interval, model_name=model_name, X=X, y=target, model_type=model_type, model_params=model_params)
        return res

    def compute_scores(self, asset: str, interval: str, model_name: Optional[str] = None, lookback: int = 1000) -> pd.DataFrame:
        """
        Ejecuta inferencia y persiste scores. Usa score.infer_and_persist si está disponible.
        Retorna el DataFrame de scores que infer_and_persist devuelve.
        """
        if score_module is None or not hasattr(score_module, "infer_and_persist"):
            raise RuntimeError("core.score.infer_and_persist no disponible.")

        df_scores = score_module.infer_and_persist(self.storage, asset, interval, model_name=model_name, lookback=lookback)
        return df_scores

    # -----------------------
    # Backfill / worker helpers
    # -----------------------
    def run_backfill_for(self, asset: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, batch_window_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta backfill para un activo/interval entre start_ms y end_ms.
        - Si fetcher está disponible, llama a fetcher.backfill_range y usa storage.save_candles.
        - Si start_ms/end_ms faltan, intenta leer backfill_status desde storage o hacer un backfill reciente.
        Retorna resumen dict con conteos y tiempos.
        """
        if not self.fetcher:
            raise RuntimeError("Fetcher no disponible para backfill")

        # intentar deducir rango si no viene
        if start_ms is None or end_ms is None:
            # intentar leer last_ts de backfill_status
            try:
                stat = None
                if hasattr(self.storage, "get_backfill_status"):
                    stat = self.storage.get_backfill_status(asset)
                if stat and stat.get("last_ts"):
                    start_ms = int(stat.get("last_ts")) + 1
                    end_ms = self.fetcher.now_ms()
                else:
                    # falta info, backfill últimos 7 días
                    end_ms = self.fetcher.now_ms()
                    start_ms = end_ms - 7 * 24 * 3600 * 1000
            except Exception:
                logger.exception("No se pudo deducir rango backfill; usando últimos 7 días")
                end_ms = self.fetcher.now_ms()
                start_ms = end_ms - 7 * 24 * 3600 * 1000

        summary = {"asset": asset, "interval": interval, "start_ms": start_ms, "end_ms": end_ms, "saved_batches": 0}
        start_time = time.time()

        def _callback_save(df_batch: pd.DataFrame):
            # persistir lote
            if df_batch is None or df_batch.empty:
                return
            try:
                # intentar save_candles
                if hasattr(self.storage, "save_candles"):
                    self.storage.save_candles(df_batch, asset, interval)
                    summary["saved_batches"] += 1
                else:
                    logger.warning("storage.save_candles no implementado; lote descartado")
            except Exception:
                logger.exception("Error guardando batch en storage")

        try:
            self.fetcher.backfill_range(asset, interval, int(start_ms), int(end_ms), batch_window_ms=batch_window_ms, callback=_callback_save)
            # actualizar backfill_status si storage soporta update_backfill_status
            try:
                if hasattr(self.storage, "update_backfill_status"):
                    last_ts = end_ms
                    self.storage.update_backfill_status(asset, interval=interval, last_ts=last_ts)
            except Exception:
                logger.exception("update_backfill_status fallo")
        except Exception:
            logger.exception("run_backfill_for failed for %s %s", asset, interval)
            raise

        summary["duration_s"] = time.time() - start_time
        return summary

    # -----------------------
    # Backtest wrapper
    # -----------------------
    def run_backtest_for(self, asset: str, interval: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta backtest delegando a core.backtest.run_backtest_for si existe.
        Devuelve lo que entregue el módulo backtest (métricas, equity_curve, etc.).
        """
        if backtest_module is None or not hasattr(backtest_module, "run_backtest_for"):
            raise RuntimeError("core.backtest.run_backtest_for no disponible.")

        try:
            res = backtest_module.run_backtest_for(self.storage, asset, interval, start_ts=start_ts, end_ts=end_ts)
            return res
        except Exception:
            logger.exception("run_backtest_for raised")
            raise

    # -----------------------
    # Utilities
    # -----------------------
    def list_assets(self) -> List[str]:
        if hasattr(self.storage, "list_assets"):
            try:
                return self.storage.list_assets()
            except Exception:
                logger.exception("list_assets failed")
                return []
        return []

    def list_watchlist(self) -> List[Dict[str, Any]]:
        if hasattr(self.storage, "list_watchlist"):
            try:
                return self.storage.list_watchlist()
            except Exception:
                logger.exception("list_watchlist failed")
                return []
        return []
# --- Inicio parche orchestrator.py: safe_backfill + helper ---
import logging
logger = logging.getLogger(__name__)

def safe_backfill(self, asset, interval, start_ms, end_ms, batch_window_ms=6*3600*1000):
    """
    Envuelve la llamada a fetcher.backfill_range con comprobaciones y actualiza backfill_status si es posible.
    Uso: reemplazar llamadas directas a self.fetcher.backfill_range(...) por self.safe_backfill(...)
    """
    if not hasattr(self, "fetcher"):
        raise RuntimeError("Orchestrator no tiene fetcher configurado")
    fetcher = self.fetcher
    # si fetcher tiene backfill_range lo usamos
    if hasattr(fetcher, "backfill_range"):
        def _on_batch(batch):
            # guardar batch en storage si existe
            try:
                if hasattr(self, "storage") and hasattr(self.storage, "save_candles"):
                    # intentar extraer asset/interval desde batch si es DataFrame
                    try:
                        self.storage.save_candles(asset, interval, batch)
                    except Exception:
                        logger.exception("Fallo al guardar batch en storage")
            except Exception:
                logger.exception("Error en on_batch handler")

        fetcher.backfill_range(asset, interval, start_ms, end_ms, batch_window_ms, callback=_on_batch)
    else:
        raise RuntimeError("Fetcher no implementa backfill_range; actualizar fetcher o adaptador")

    # actualizar backfill_status si storage lo soporta (acepta ms o s; normalizamos a ms)
    try:
        last_ts = int(end_ms)
        if hasattr(self, "storage") and hasattr(self.storage, "update_backfill_status"):
            try:
                self.storage.update_backfill_status(asset, interval, last_ts)
            except Exception:
                # si la storage espera segundos en vez de ms, intentar convertir
                try:
                    self.storage.update_backfill_status(asset, interval, int(last_ts / 1000))
                except Exception:
                    logger.exception("No se pudo actualizar backfill_status")
    except Exception:
        logger.exception("Error al intentar actualizar backfill_status")

# adjuntar helper al objeto Orchestrator si existe
try:
    Orchestrator  # type: ignore
except Exception:
    Orchestrator = None

if Orchestrator is not None and not hasattr(Orchestrator, "safe_backfill"):
    setattr(Orchestrator, "safe_backfill", safe_backfill)

# --- Fin parche orchestrator.py ---

