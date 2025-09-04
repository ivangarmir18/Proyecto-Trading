# core/score.py
"""
Módulo de Scoring / IA para Proyecto-Trading

Objetivo:
- Calcular indicadores (EMA, ATR, RSI, Fibonacci, momentum, vol)
- Generar features para ML
- Entrenar modelos (sklearn / LightGBM opcional) para predecir retornos futuros o señales
- Hacer inferencia (scores) en tiempo real / batch y persistir en DB
- Guardar metadatos del modelo en la tabla `models` (via PostgresStorage.save_model_record)
- Evaluación y backtest helpers

Diseño:
- Clase principal: ScoreEngine(storage)
- Métodos relevantes:
    - compute_indicators(df)
    - features_from_candles(df, lookback,...)
    - make_target(df, horizon, method='future_return')
    - train_model(asset, interval, df, features, target, params)
    - predict_scores(df, model)
    - save_model_to_storage(...)
    - evaluate_model(...) (cross-val / walk-forward)
    - backtest_signals(...) (usa core.backtest runner)

Dependencias sugeridas:
- pandas, numpy, scikit-learn, joblib, lightgbm (opcional), shap (opcional para explainability)

Ejemplo rápido de uso:
    from core.score import ScoreEngine
    s = ScoreEngine()
    s.train_and_persist(asset='BTCUSDT', interval='1m', horizon=60, model_type='lgbm')

"""

import logging
import os
import time
import math
from typing import Dict, Any, Optional, List, Tuple, Sequence

import numpy as np
import pandas as pd

# try optional deps
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import joblib
except Exception:
    joblib = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# Storage integration
from core.storage_postgres import PostgresStorage

logger = logging.getLogger("core.score")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)


# --------------------
# Utilities
# --------------------
def _ensure_numeric_series(s) -> pd.Series:
    s = s.copy()
    s = pd.to_numeric(s, errors="coerce")
    return s


def _safe_div(a, b):
    try:
        return a / b
    except Exception:
        return np.nan


# --------------------
# Indicators
# --------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # df must have high, low, close
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def momentum(series: pd.Series, window: int = 10) -> pd.Series:
    return series.pct_change(periods=window)


def volatility(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window=window, min_periods=1).std()


def fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Compute simple Fibonacci levels (0%, 23.6, 38.2, 50, 61.8, 100%) on last lookback bars.
    Returns a dict of levels based on min/max in window.
    """
    if df is None or df.empty:
        return {}
    window = df.tail(lookback)
    high = float(window["high"].max())
    low = float(window["low"].min())
    diff = high - low
    if diff == 0:
        return {"high": high, "low": low}
    levels = {
        "high": high,
        "low": low,
        "0.0": low,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "1.0": high,
    }
    return levels


# --------------------
# Feature engineering
# --------------------
def features_from_candles(df: pd.DataFrame, feature_cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Given df with ts,timestamp,open,high,low,close,volume, generate features:
      - ema_{12,26,50}
      - sma_{50,200}
      - rsi_14
      - atr_14
      - momentum_{5,10}
      - vol_{20}
      - returns_{1,5,10}
      - fib levels (as distances to current close)
    Returns a DataFrame with index aligned to input df (same length).
    """
    cfg = feature_cfg or {}
    df = df.copy().reset_index(drop=True)
    close = _ensure_numeric_series(df["close"])
    high = _ensure_numeric_series(df["high"])
    low = _ensure_numeric_series(df["low"])
    vol = _ensure_numeric_series(df["volume"]) if "volume" in df.columns else pd.Series(np.nan, index=df.index)

    features = pd.DataFrame(index=df.index)
    # EMAs
    for s in (12, 26, 50):
        features[f"ema_{s}"] = ema(close, s)
    # SMAs
    for w in (50, 200):
        features[f"sma_{w}"] = sma(close, w)
    # RSI
    features["rsi_14"] = rsi(close, 14)
    # ATR
    features["atr_14"] = atr(df, 14)
    # momentum
    for w in (5, 10):
        features[f"mom_{w}"] = momentum(close, w)
    # volatility
    features["vol_20"] = volatility(close, 20)
    # returns
    features["ret_1"] = close.pct_change(1)
    features["ret_5"] = close.pct_change(5)
    features["ret_10"] = close.pct_change(10)
    # price ratios
    features["close_over_ema12"] = _safe_div(close, features["ema_12"])
    features["close_over_sma50"] = _safe_div(close, features["sma_50"])
    # fib levels relative distances
    try:
        fib = fibonacci_levels(df, lookback=cfg.get("fib_lookback", 100))
        for k, v in fib.items():
            # distance to level normalized by price
            features[f"dist_to_fib_{k}"] = (v - close) / close
    except Exception:
        logger.exception("fibonacci feature generation failed")
    # time features if timestamp exists
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        features["hour"] = ts.dt.hour
        features["dayofweek"] = ts.dt.dayofweek
    # fill inf / nan sensibly
    features = features.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)
    return features


# --------------------
# Target creation
# --------------------
def make_target(df: pd.DataFrame, horizon: int = 60, method: str = "future_return") -> pd.Series:
    """
    Create a regression target:
      - future_return: (close_{t+h} / close_t) - 1  (float)
      - future_sign: sign(future_return) (-1/0/1) if method == 'sign'
    horizon: number of bars ahead (depending on your interval).
    """
    close = _ensure_numeric_series(df["close"])
    future = close.shift(-horizon)
    fut_ret = (future / close) - 1
    if method == "future_return":
        return fut_ret
    elif method == "sign":
        return np.sign(fut_ret).fillna(0).astype(int)
    else:
        raise ValueError("Unknown method for make_target")


# --------------------
# Model helpers
# --------------------
def _train_sklearn_regressor(X: pd.DataFrame, y: pd.Series, random_state: int = 42, **kwargs):
    """
    Train a RandomForestRegressor by default (fast to run locally). Returns fitted model.
    """
    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1, **kwargs)
    model.fit(X, y)
    return model


def _train_lightgbm(X: pd.DataFrame, y: pd.Series, params: Optional[dict] = None):
    if not HAS_LGB:
        raise RuntimeError("lightgbm not available")
    dtrain = lgb.Dataset(X, label=y)
    p = params or {"objective": "regression", "metric": "rmse", "verbose": -1}
    model = lgb.train(p, dtrain, num_boost_round=200)
    return model


def _save_model_file(model, path: str):
    if joblib is None:
        raise RuntimeError("joblib required to save model files (pip install joblib)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def _load_model_file(path: str):
    if joblib is None:
        raise RuntimeError("joblib required to load model files")
    return joblib.load(path)


# --------------------
# ScoreEngine
# --------------------
class ScoreEngine:
    def __init__(self, storage: Optional[PostgresStorage] = None, model_dir: str = "models"):
        self.storage = storage or PostgresStorage()
        self.model_dir = model_dir
        self.storage.init_db()
        os.makedirs(self.model_dir, exist_ok=True)

    # Indicator / features helpers
    def compute_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns a dict of scalar indicators for last bar (useful for UI).
        """
        res = {}
        try:
            last_window = df.tail(200)
            res["ema_12"] = float(ema(last_window["close"], 12).iloc[-1])
            res["ema_26"] = float(ema(last_window["close"], 26).iloc[-1])
            res["sma_50"] = float(sma(last_window["close"], 50).iloc[-1])
            res["sma_200"] = float(sma(last_window["close"], 200).iloc[-1])
            res["rsi_14"] = float(rsi(last_window["close"], 14).iloc[-1])
            res["atr_14"] = float(atr(last_window, 14).iloc[-1])
            res["fib"] = fibonacci_levels(last_window, lookback=100)
        except Exception:
            logger.exception("compute_indicators failed")
        return res

    def features_from_candles(self, df: pd.DataFrame, feature_cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        return features_from_candles(df, feature_cfg)

    # Training / eval
    def train_model(self,
                    asset: str,
                    interval: str,
                    df_candles: pd.DataFrame,
                    horizon: int = 60,
                    model_type: str = "lgbm",
                    feature_cfg: Optional[Dict[str, Any]] = None,
                    save_to_storage: bool = True,
                    model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        End-to-end train:
        - compute features from df_candles
        - create target (future_return) with horizon
        - split by time (TimeSeriesSplit) and train model
        - save model file and register metadata in storage.models table
        Returns dict with metrics and model path & id.
        """
        if df_candles is None or df_candles.empty:
            raise ValueError("df_candles empty")
        df_candles = df_candles.sort_values("ts").reset_index(drop=True)
        features = self.features_from_candles(df_candles, feature_cfg)
        target = make_target(df_candles, horizon=horizon, method="future_return")
        # align
        X = features.iloc[:-horizon].reset_index(drop=True)
        y = target.iloc[:-horizon].reset_index(drop=True).fillna(0.0)
        # drop any NaNs
        X = X.fillna(0.0)
        # simple time-series split CV for metrics
        tscv = TimeSeriesSplit(n_splits=3)
        metrics = {"folds": []}
        fold = 0
        models = []
        for train_idx, test_idx in tscv.split(X):
            fold += 1
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            # train
            if model_type == "lgbm" and HAS_LGB:
                model = _train_lightgbm(X_tr, y_tr)
            else:
                model = _train_sklearn_regressor(X_tr, y_tr)
            # predict
            if model_type == "lgbm" and HAS_LGB:
                y_pred = model.predict(X_te)
            else:
                y_pred = model.predict(X_te)
            rmse = float(math.sqrt(mean_squared_error(y_te, y_pred)))
            r2 = float(r2_score(y_te, y_pred))
            metrics["folds"].append({"fold": fold, "rmse": rmse, "r2": r2})
            models.append(model)
        # retrain on full data
        if model_type == "lgbm" and HAS_LGB:
            final_model = _train_lightgbm(X, y)
        else:
            final_model = _train_sklearn_regressor(X, y)
        # save model to disk
        timestamp = int(time.time())
        model_name = model_name or f"{asset}_{interval}_{model_type}_{timestamp}.pkl"
        model_path = os.path.join(self.model_dir, model_name)
        if joblib is None:
            logger.warning("joblib not installed - model won't be saved to disk")
            model_saved = None
        else:
            _save_model_file(final_model, model_path)
            model_saved = model_path
        # persist model metadata in DB
        model_id = None
        if save_to_storage:
            try:
                metadata = {
                    "asset": asset, "interval": interval, "model_type": model_type,
                    "model_path": model_saved, "trained_at": timestamp,
                    "metrics": metrics
                }
                model_id = self.storage.save_model_record(model_name, asset, interval, metadata)
            except Exception:
                logger.exception("Failed to save model record in DB")
        return {"model_name": model_name, "model_path": model_saved, "model_id": model_id, "metrics": metrics}

    def load_model(self, model_path: str):
        if joblib is None:
            raise RuntimeError("joblib required to load models")
        return _load_model_file(model_path)

    def infer(self,
              asset: str,
              interval: str,
              df_candles: pd.DataFrame,
              model: Any,
              horizon: int = 60,
              feature_cfg: Optional[Dict[str, Any]] = None,
              output_raw_scores: bool = True) -> pd.DataFrame:
        """
        Given a candles DF and a fitted model, compute features and produce a DataFrame with:
          ts, timestamp, score (predicted future_return), meta fields.
        The predicted score aligns with current bar predicting future horizon.
        """
        if df_candles is None or df_candles.empty:
            return pd.DataFrame()
        df = df_candles.sort_values("ts").reset_index(drop=True)
        feats = self.features_from_candles(df, feature_cfg)
        feats = feats.fillna(0.0)
        # remove last horizon rows that cannot have target if we had target
        X = feats
        # predict - if lightgbm model object predict same way as sklearn
        try:
            preds = model.predict(X)
        except Exception:
            # handle lgb Booster where predict expects np.array
            try:
                preds = model.predict(X.values)
            except Exception:
                logger.exception("Model prediction failed")
                preds = np.zeros(len(X))
        out = pd.DataFrame({
            "ts": df["ts"].astype(int),
            "timestamp": df["timestamp"],
            "score_raw": preds
        })
        # normalize/scale raw score to a bounded [-3,3] or probability-like score
        out["score"] = _normalize_score(out["score_raw"])
        # attach indicators snapshot for UI convenience
        try:
            ind = self.compute_indicators(df)
            # broadcast
            for k, v in ind.items():
                out[f"ind_{k}"] = str(v)  # store as string to avoid heavy JSON here
        except Exception:
            logger.exception("attach indicators failed")
        # Optionally persist scores to DB
        return out

    def predict_and_persist(self, asset: str, interval: str, df_candles: pd.DataFrame, model_path: Optional[str] = None, model_obj: Optional[Any] = None, persist: bool = True, horizon: int = 60):
        """
        Convenience: load model (if path given), run infer and persist scores in DB via storage.save_scores
        """
        if model_obj is None and model_path is None:
            # get latest model for this asset/interval from storage
            rec = self.storage.get_latest_model_record(asset, interval)
            if rec and rec.get("metadata") and rec["metadata"].get("model_path"):
                model_path = rec["metadata"]["model_path"]
        if model_obj is None:
            if model_path is None:
                raise ValueError("No model path or model object provided")
            model_obj = self.load_model(model_path)
        df_scores = self.infer(asset, interval, df_candles, model_obj, horizon=horizon)
        if df_scores is None or df_scores.empty:
            return None
        # prepare for DB: keep last N rows (e.g., all)
        df_db = df_scores[["ts", "timestamp", "score"]].copy()
        # storage.save_scores expects column 'score' to be serializable; wrap in dict for traceability
        df_db["asset"] = asset
        df_db["interval"] = interval
        df_db["score"] = df_db["score"].apply(lambda s: {"score": float(s)})
        # cast ts to int
        df_db["ts"] = df_db["ts"].astype(int)
        # persist
        if persist:
            try:
                # save in batches via storage.save_scores
                self.storage.save_scores(df_db)
            except Exception:
                logger.exception("Failed to persist scores to DB")
        return df_scores

    # --------------------
    # Evaluation & backtest
    # --------------------
    def evaluate_model(self, asset: str, interval: str, df_candles: pd.DataFrame, model_type: str = "lgbm", horizon: int = 60, n_splits: int = 3) -> Dict[str, Any]:
        """
        Walk-forward evaluation: trains model on each fold and returns aggregated metrics.
        Uses TimeSeriesSplit.
        """
        df_candles = df_candles.sort_values("ts").reset_index(drop=True)
        features = self.features_from_candles(df_candles)
        target = make_target(df_candles, horizon=horizon, method="future_return")
        X_full = features.iloc[:-horizon].fillna(0.0)
        y_full = target.iloc[:-horizon].fillna(0.0)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []
        for train_idx, test_idx in tscv.split(X_full):
            X_tr, X_te = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_tr, y_te = y_full.iloc[train_idx], y_full.iloc[test_idx]
            if model_type == "lgbm" and HAS_LGB:
                model = _train_lightgbm(X_tr, y_tr)
                y_pred = model.predict(X_te)
            else:
                model = _train_sklearn_regressor(X_tr, y_tr)
                y_pred = model.predict(X_te)
            rmse = float(math.sqrt(mean_squared_error(y_te, y_pred)))
            r2 = float(r2_score(y_te, y_pred))
            metrics.append({"rmse": rmse, "r2": r2})
        # aggregate
        avg_rmse = float(np.mean([m["rmse"] for m in metrics])) if metrics else None
        avg_r2 = float(np.mean([m["r2"] for m in metrics])) if metrics else None
        return {"folds": metrics, "avg_rmse": avg_rmse, "avg_r2": avg_r2}

    def backtest_signals(self, asset: str, interval: str, df_candles: pd.DataFrame, model_obj: Any, horizon: int = 60, threshold: float = 0.0, **backtest_kwargs) -> Dict[str, Any]:
        """
        Convert model predictions into simple buy/sell signals and run backtest using core.backtest.run_backtest.
        Strategy used:
          - buy when predicted future_return > threshold
          - sell when predicted future_return <= threshold (or stronger logic)
        Returns backtest result dict.
        """
        from core.backtest import run_backtest
        # infer scores
        df_scores = self.infer(asset, interval, df_candles, model_obj, horizon=horizon)
        if df_scores is None or df_scores.empty:
            return {"error": "no_scores"}
        # strategy function
        def strategy_fn(row, idx, df_all, state=None):
            # row is a row from df_candles in backtest loop; align by ts
            ts = int(row["ts"])
            # find corresponding score row (exact ts)
            sr = df_scores[df_scores["ts"] == ts]
            if sr.empty:
                return None
            score_val = float(sr.iloc[0]["score_raw"])
            # simple: buy if predicted return > threshold
            if score_val > threshold:
                return "buy"
            elif score_val < -threshold:
                return "sell"
            return None
        # pass df_candles to backtest.run_backtest
        result = run_backtest(df_candles, strategy_fn, **backtest_kwargs)
        return result


# --------------------
# Helpers
# --------------------
def _normalize_score(arr: Sequence[float], method: str = "tanh_scale") -> np.ndarray:
    """
    Normalize raw predictions into bounded scores.
    Default: tanh scaling to [-1,1], then multiply by 3 to map to [-3,3] (like previous behavior).
    """
    arr = np.asarray(arr, dtype=float)
    if method == "tanh_scale":
        # guard against big outliers
        scaled = np.tanh(arr / (np.std(arr) + 1e-9))
        return (scaled * 3.0)
    else:
        # simple clipping
        mx = np.nanmax(np.abs(arr)) if len(arr) > 0 else 1.0
        mx = mx if mx != 0 else 1.0
        return np.clip(arr / mx * 3.0, -3.0, 3.0)


# --------------------
# convenience top-level functions
# --------------------
def train_and_persist(storage: Optional[PostgresStorage], asset: str, interval: str, df_candles: pd.DataFrame, horizon: int = 60, model_type: str = "lgbm", save_to_storage: bool = True):
    se = ScoreEngine(storage=storage)
    return se.train_model(asset, interval, df_candles, horizon=horizon, model_type=model_type, save_to_storage=save_to_storage)


def infer_and_persist(storage: Optional[PostgresStorage], asset: str, interval: str, df_candles: pd.DataFrame, model_path: Optional[str] = None, horizon: int = 60):
    se = ScoreEngine(storage=storage)
    return se.predict_and_persist(asset, interval, df_candles, model_path=model_path, horizon=horizon)

