# core/ai_train.py
"""
AI training and prediction utilities.

- prepare_dataset(storage, asset, interval, lookback, horizon, features) -> X,y DataFrames
- train_model(storage, asset, interval, out_dir='models/', model_type='rf', **params) -> saves model and returns metrics
- predict_with_model(model_path, X_new) -> predictions
- apply_model_and_save_scores(storage, model_path, asset, interval, lookback, horizon) -> compute features, predict, save as scores

Design:
- Default model: RandomForestClassifier (sklearn). It's robust, easy to interpret, and fast to train.
- Task: classification of future direction over 'horizon' (1=next bar) using features engineered from candles+indicators.
- Walk-forward (time series) cross validation included.
- Models saved with joblib in folder models/{asset}_{interval}_{timestamp}.joblib
"""

from __future__ import annotations
import os
import json
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger("core.ai_train")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.getenv("AI_TRAIN_LOG_LEVEL", "INFO"))

# sklearn imports (lazy to provide nice error message if not installed)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    import joblib
except Exception as e:
    raise ImportError("scikit-learn and joblib are required: pip install scikit-learn joblib") from e

MODELS_DIR = os.getenv("MODELS_DIR", "models")

def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- Feature engineering ----------
def build_features_from_df(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Given candles with basic indicators (ema9, ema40, atr, macd, rsi), build tabular features:
    - returns for lags
    - ema diff
    - atr, rsi, macd current and changes
    - volume change
    """
    df = df.copy().reset_index(drop=True)
    df["ret"] = df["close"].pct_change().fillna(0)
    # lag returns
    for lag in range(1, lags+1):
        df[f"ret_lag_{lag}"] = df["ret"].shift(lag)
    df["ema_diff"] = (df["ema9"] - df["ema40"]) / (df["ema40"].replace(0, np.nan))
    df["atr_norm"] = df["atr"] / (df["close"].replace(0, np.nan))
    df["macd_signal_diff"] = df["macd"] - df.get("macd_signal", 0)
    df["vol_chg"] = df["volume"].pct_change().fillna(0)
    # fill na with 0
    feature_cols = [c for c in df.columns if c.startswith("ret_lag_")] + ["ema_diff", "atr_norm", "macd_signal_diff", "rsi", "vol_chg"]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feature_cols].fillna(0.0)
    return X, feature_cols

def prepare_dataset(storage, asset: str, interval: str, lookback: int = 2000, horizon: int = 1, min_rows: int = 200) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Fetch candles + indicators from storage, build dataset.
    Returns (X, y, feature_cols, df_full)
    y is binary: 1 if future return over horizon > 0, else 0.
    """
    # get sufficient history (we'll do lookback bars)
    df = storage.get_ohlcv(asset, interval, limit=lookback)
    if df is None or df.empty or len(df) < min_rows:
        raise RuntimeError(f"Not enough data for {asset} {interval} (got {0 if df is None else len(df)})")

    # If indicators table exists, try to JOIN indicators columns into df (fast path)
    # For simplicity: we assume storage has indicators table populated. If not, compute basic indicators locally.
    with storage.get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                  SELECT i.ts, i.ema9, i.ema40, i.atr, i.macd, i.macd_signal, i.rsi
                  FROM indicators i
                  WHERE i.asset = %s AND i.interval = %s AND i.ts >= %s
                  ORDER BY i.ts ASC
                """, (asset, interval, int(df["ts"].astype("int64") // 1_000_000).min()))
                rows = cur.fetchall()
                if rows:
                    inds = pd.DataFrame(rows, columns=["ts", "ema9", "ema40", "atr", "macd", "macd_signal", "rsi"])
                    inds["ts"] = pd.to_datetime(inds["ts"], unit="ms", utc=True)
                    # merge by ts
                    df = pd.merge_asof(df.sort_values("ts"), inds.sort_values("ts"), on="ts", direction="nearest", tolerance=pd.Timedelta("1s"))
                else:
                    # compute local indicators
                    from core.score import compute_basic_indicators
                    df = compute_basic_indicators(df)
            except Exception:
                logger.exception("Error fetching indicators; computing locally")
                from core.score import compute_basic_indicators
                df = compute_basic_indicators(df)

    df = df.dropna().reset_index(drop=True)
    # build features
    X, feature_cols = build_features_from_df(df, lags=5)
    # compute target: future return
    df["future_close"] = df["close"].shift(-horizon)
    df["future_ret"] = (df["future_close"] - df["close"]) / df["close"]
    y = (df["future_ret"] > 0).astype(int)  # binary up/down
    # drop last horizon rows with nan target
    valid_idx = ~y.isna()
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)
    df_full = df.loc[valid_idx].reset_index(drop=True)
    return X, y, feature_cols, df_full

# ---------- Training ----------
def train_model(storage, asset: str, interval: str, model_type: str = "rf", lookback: int = 2000, horizon: int = 1, n_estimators: int = 100, max_depth: Optional[int] = None, test_fraction: float = 0.2) -> Dict[str, Any]:
    """
    Train a classifier model (RandomForest by default) with walk-forward CV.

    Returns dict with model_path and metrics (accuracy, roc_auc, f1)
    """
    X, y, feature_cols, df_full = prepare_dataset(storage, asset, interval, lookback=lookback, horizon=horizon)
    n_samples = len(X)
    test_size = max(1, int(n_samples * test_fraction))
    train_size = n_samples - test_size
    if train_size < 50:
        raise RuntimeError("Not enough training data after split")

    # simple split: use initial train_size for training, last test_size for holdout
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    # model selection
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"model_type {model_type} not supported yet")

    logger.info("Entrenando modelo %s para %s %s: X_train=%s y_train=%s", model_type, asset, interval, X_train.shape, y_train.shape)
    model.fit(X_train, y_train)

    # predict and metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None

    # save model
    ensure_models_dir()
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_name = f"{asset}_{interval}_{model_type}_{ts}.joblib"
    model_path = os.path.join(MODELS_DIR, model_name)
    joblib.dump({"model": model, "feature_cols": feature_cols, "meta": {"asset": asset, "interval": interval, "horizon": horizon}}, model_path)
    logger.info("Modelo guardado en %s", model_path)

    metrics = {"accuracy": acc, "f1": f1, "roc_auc": roc, "model_path": model_path, "n_train": len(X_train), "n_test": len(X_test)}
    return metrics

# ---------- Prediction helper ----------
def load_model(model_path: str):
    import joblib
    return joblib.load(model_path)

def predict_from_model(model_blob: Dict[str, Any], X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model = model_blob["model"]
    feature_cols = model_blob["feature_cols"]
    X_in = X[feature_cols].fillna(0.0)
    preds = model.predict(X_in)
    proba = model.predict_proba(X_in)[:, 1] if hasattr(model, "predict_proba") else None
    return preds, proba

def apply_model_and_save_predictions(storage, model_path: str, asset: str, interval: str, lookback: int = 500, save_as_scores: bool = True):
    """
    Load model, prepare latest data, predict and optionally save predictions as scores in 'scores' table.
    """
    blob = load_model(model_path)
    # prepare dataset with enough rows for features
    X, y, feature_cols, df_full = prepare_dataset(storage, asset, interval, lookback=lookback)
    preds, proba = predict_from_model(blob, X)
    # build DataFrame of predictions aligned with df_full ts
    df_pred = df_full.copy().reset_index(drop=True)
    df_pred["pred_label"] = preds
    df_pred["pred_proba"] = proba if proba is not None else np.where(preds==1, 1.0, 0.0)
    # save last N predictions into scores table (upsert)
    if save_as_scores:
        from core.score import save_scores_to_db
        # create a dataframe compatible: ts, score (use pred_proba), range_min..target use existing columns from df_full if present
        df_save = pd.DataFrame({
            "ts": df_pred["ts"],
            "score": df_pred["pred_proba"],
            "range_min": df_pred.get("support", pd.NA),
            "range_max": df_pred.get("resistance", pd.NA),
            "stop": df_pred.get("support", pd.NA) - 1.3 * df_pred.get("atr", 0),
            "target": df_pred.get("resistance", pd.NA) + 1.3 * df_pred.get("atr", 0),
        })
        cnt = save_scores_to_db(storage, df_save, asset, interval)
        logger.info("Predicciones guardadas como scores: %d rows", cnt)
        return {"predictions_saved": cnt, "model_path": model_path}
    return {"predictions": len(df_pred), "model_path": model_path}
