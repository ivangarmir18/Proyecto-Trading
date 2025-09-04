"""
core/orchestrator.py

Coordinador central: cálculo de indicadores, scoring, entrenamiento ligero (metadata),
persistencia de indicadores y scores, y ejecución de backtests.

Diseñado para:
- Integrar PostgresStorage (guardar indicadores/scores/models)
- Proveer API para la UI y scripts (compute_indicators_for, compute_and_store, compute_scores, train_model, run_backtest_for)
- Ser extensible: si hay un módulo indicators.* con funciones avanzadas los usa; si no, cae a fallbacks.
"""
import logging
from typing import Optional, Callable, Dict, Any, List

import pandas as pd
import numpy as np

from core.storage_postgres import PostgresStorage

logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)

# Optional imports from project's indicators/score modules
try:
    from indicators import fibonacci as fib_module  # type: ignore
except Exception:
    fib_module = None

try:
    from core import score as core_score  # type: ignore
except Exception:
    core_score = None


class Orchestrator:
    def __init__(self, storage: Optional[PostgresStorage] = None):
        self.storage = storage or PostgresStorage()
        self.storage.init_db()

    # ---------------------
    # Indicators
    # ---------------------
    def compute_indicators_for(self, asset: str, interval: str, lookback: int = 1000) -> Dict[str, Any]:
        """
        Compute a set of indicators for asset/interval using stored candles.
        Returns a dict with computed indicators (ema, rsi, fibonacci if available).
        """
        df = self.storage.load_candles(asset, interval, limit=lookback, ascending=True)
        if df.empty:
            logger.warning("No candles for %s %s", asset, interval)
            return {}
        res: Dict[str, Any] = {}
        try:
            # EMAs
            res["ema_12"] = float(df["close"].ewm(span=12, adjust=False).mean().iloc[-1])
            res["ema_26"] = float(df["close"].ewm(span=26, adjust=False).mean().iloc[-1])
            # SMA 50/200
            res["sma_50"] = float(df["close"].rolling(window=50, min_periods=1).mean().iloc[-1])
            res["sma_200"] = float(df["close"].rolling(window=200, min_periods=1).mean().iloc[-1])
            # RSI
            res["rsi_14"] = float(self._rsi(df["close"], 14).iloc[-1])
        except Exception:
            logger.exception("Basic indicators computation failed")
        # Fibonacci / custom indicators
        if fib_module and hasattr(fib_module, "compute_fibonacci_levels"):
            try:
                res["fibonacci"] = fib_module.compute_fibonacci_levels(df)
            except Exception:
                logger.exception("Fibonacci computation failed")
        # Return indicators; caller can persist with save_indicators_to_db
        return res

    def save_indicators_to_db(self, asset: str, interval: str, ts: int, indicators: Dict[str, Any]) -> int:
        """
        Save indicators as a JSONB entry in indicators table (with ts).
        """
        try:
            df = pd.DataFrame([{"ts": int(ts), "asset": asset, "interval": interval, "indicators": indicators}])
            # Insert manually using storage's models? Storage currently doesn't have save_indicators method,
            # so we use 'indicators' table via save_scores-like insertion for simplicity.
            # We'll implement insertion here using storage._get_conn for integrity.
            conn = self.storage._get_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO indicators (ts, asset, interval, indicators) VALUES (%s,%s,%s,%s);",
                (int(ts), asset, interval, JsonSafe(indicators))
            )
            conn.commit()
            cur.close()
            self.storage._put_conn(conn)
            return 1
        except Exception:
            logger.exception("save_indicators_to_db failed")
            try:
                self.storage._put_conn(conn)
            except Exception:
                pass
            raise

    # ---------------------
    # Scoring
    # ---------------------
    def compute_scores_for(self, asset: str, interval: str, persist: bool = False) -> Dict[str, Any]:
        """
        Compute a score for the current latest state of an asset.
        - Uses core.score.score_from_indicators if available, else a fallback deterministic logic.
        - If persist=True, writes a row to scores table.
        """
        indicators = self.compute_indicators_for(asset, interval, lookback=500)
        if not indicators:
            return {}
        score_obj: Dict[str, Any]
        if core_score and hasattr(core_score, "score_from_indicators"):
            try:
                score_obj = core_score.score_from_indicators(indicators)
            except Exception:
                logger.exception("Project score function failed — using fallback")
                score_obj = self._fallback_score(indicators)
        else:
            score_obj = self._fallback_score(indicators)
        # Add metadata
        score_obj = {"score": score_obj.get("score"), "meta": score_obj.get("meta", indicators)}
        if persist:
            try:
                now_ts = int(pd.Timestamp.utcnow().timestamp())
                df = pd.DataFrame([{"ts": now_ts, "asset": asset, "interval": interval, "score": score_obj}])
                self.storage.save_scores(df)
            except Exception:
                logger.exception("Failed to persist score for %s", asset)
        return score_obj

    # ---------------------
    # Backtest orchestration
    # ---------------------
    def run_backtest_for(self,
                         asset: str,
                         interval: str,
                         strategy_fn: Callable,
                         lookback: int = 2000,
                         initial_capital: float = 1000.0,
                         fee: float = 0.0005,
                         save_scores: bool = False) -> Dict[str, Any]:
        """
        Load history and run backtest using core.backtest.run_backtest.
        Returns the backtest result dict and optionally persists summary to scores.
        """
        from core.backtest import run_backtest  # local import to avoid cycles
        df = self.storage.load_candles(asset, interval, limit=lookback, ascending=True)
        if df.empty:
            return {"error": "no_data"}
        result = run_backtest(df, lambda r, i, d, ctx=None: strategy_fn(r, i, d), initial_capital=initial_capital, fee=fee)
        if save_scores:
            # Persist a summary row into scores table
            now_ts = int(pd.Timestamp.utcnow().timestamp())
            summary = {"final_value": result.get("final_value"), "returns": result.get("returns"), "n_trades": result.get("n_trades")}
            dfscore = pd.DataFrame([{"ts": now_ts, "asset": asset, "interval": interval, "score": summary}])
            self.storage.save_scores(dfscore)
        return result

    # ---------------------
    # Training placeholder (IA)
    # ---------------------
    def train_model(self, asset: str, interval: str, model_name: str, training_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder for training. Implement your own training pipeline and call storage.save_model_record with metadata.
        Here we just persist a metadata record so UI can show model entries.
        """
        try:
            mid = self.storage.save_model_record(model_name, asset, interval, training_meta)
            return {"id": mid, "status": "saved_metadata"}
        except Exception:
            logger.exception("train_model failed")
            raise

    # ---------------------
    # Utilities
    # ---------------------
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _fallback_score(indicators: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        ema12 = indicators.get("ema_12")
        ema26 = indicators.get("ema_26")
        rsi = indicators.get("rsi_14")
        if ema12 is not None and ema26 is not None:
            score += 1 if ema12 > ema26 else -1
        if rsi is not None:
            if rsi > 70:
                score += 1
            elif rsi < 30:
                score -= 1
        return {"score": score, "meta": indicators}


# JSON adapter for insertion (fallback)
def JsonSafe(obj: Any) -> Any:
    """
    Ensure object is JSON-serializable by converting numpy types.
    """
    if isinstance(obj, dict):
        return {k: JsonSafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [JsonSafe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
