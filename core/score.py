# core/score.py
"""
Módulo de scoring / features / entrenamiento / inferencia.

Funciones principales exportadas:
 - features_from_candles(df) -> DataFrame (features, indexed by ts)
 - make_target(df, horizon=1) -> Series (target aligned with features)
 - train_and_persist(storage, asset, interval, model_name, X, y, model_type='rf', model_params=None)
       -> dict (metadata)
 - infer_and_persist(storage, asset, interval, model_name=None, model_path=None, lookback=None)
       -> DataFrame (predictions persisted)

Diseño:
 - Las funciones son defensivas: validan inputs y lanzan excepciones claras.
 - El módulo intenta usar LightGBM si está instalado; si no, usa RandomForestRegressor de sklearn.
 - Modelos se guardan con joblib en data/models/{asset}_{interval}_{timestamp}_{model_name}.pkl
 - Registro de modelos: llama storage.save_model_record(asset, interval, model_name, metadata, path)
 - Inferencia: carga el último modelo si model_path no dado (storage.get_latest_model_record) y guarda resultados
   con storage.save_scores(df_scores, asset, interval). El DataFrame pasado a save_scores debe tener columnas:
   'ts' (ms int) y 'score' (dict JSON-serializable) -- storage se encargará de la persistencia.
"""

from __future__ import annotations

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# ML libs
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
except Exception:
    # si sklearn no está instalado, fallamos temprano en funciones de entrenamiento
    RandomForestRegressor = None
    LinearRegression = None
    train_test_split = None
    mean_squared_error = None
    r2_score = None
    joblib = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

MODELS_DIR = os.getenv("MODELS_DIR", "data/models")


# ---------------------------
# Utilities / features
# ---------------------------
def _ensure_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que df tenga columna 'ts' en ms y la convierte a índice datetime UTC."""
    if "ts" not in df.columns:
        raise ValueError("DataFrame de candles debe contener la columna 'ts' (ms desde epoch).")
    df2 = df.copy()
    # normalizar ts a int ms
    if np.issubdtype(df2["ts"].dtype, np.datetime64):
        df2["ts"] = (df2["ts"].astype("datetime64[ms]").astype("int64"))
    else:
        df2["ts"] = df2["ts"].astype("int64")
    df2 = df2.sort_values("ts")
    df2["ts_dt"] = pd.to_datetime(df2["ts"], unit="ms", utc=True)
    df2 = df2.set_index("ts_dt", drop=False)
    return df2


def _rolling_apply(df: pd.Series, window: int, fn):
    """Helper seguro para rolling apply que maneja ventanas cortas."""
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(df) < window:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df.rolling(window=window, min_periods=1).apply(fn, raw=False)


def features_from_candles(df: pd.DataFrame, include: Optional[list] = None) -> pd.DataFrame:
    """
    Calcula un set de features estándar a partir de candles OHLCV.
    Input: df con columnas ts (ms), open, high, low, close, volume
    Output: DataFrame indexado por ts_dt con columnas de features; mantiene columna 'ts' (ms int).
    Features incluidas (por defecto):
      - close, open, high, low, volume (passthrough)
      - returns_1: log return 1 bar
      - ma_5, ma_10, ma_20
      - ema_10, ema_20
      - vol_20: std dev of returns (volatility)
      - momentum_10: close / ma_10 - 1
      - rsi_14 (implementación simple)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df2 = _ensure_ts_index(df)

    close = pd.to_numeric(df2["close"], errors="coerce")
    open_ = pd.to_numeric(df2["open"], errors="coerce")
    high = pd.to_numeric(df2["high"], errors="coerce")
    low = pd.to_numeric(df2["low"], errors="coerce")
    volume = pd.to_numeric(df2.get("volume", pd.Series([np.nan] * len(df2))), errors="coerce")

    out = pd.DataFrame(index=df2.index)
    out["ts"] = df2["ts"].astype("int64")
    out["close"] = close
    out["open"] = open_
    out["high"] = high
    out["low"] = low
    out["volume"] = volume

    # returns
    out["ret_1"] = np.log(close) - np.log(close.shift(1))

    # moving averages
    out["ma_5"] = close.rolling(5, min_periods=1).mean()
    out["ma_10"] = close.rolling(10, min_periods=1).mean()
    out["ma_20"] = close.rolling(20, min_periods=1).mean()

    # ema
    out["ema_10"] = close.ewm(span=10, adjust=False).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()

    # volatility (std of returns)
    out["vol_20"] = out["ret_1"].rolling(20, min_periods=1).std()

    # momentum
    out["mom_10"] = close / out["ma_10"] - 1.0

    # rsi 14 (simple)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # fill small NaNs (but keep large NaNs)
    out = out.replace([np.inf, -np.inf], np.nan)

    # select included features if requested
    if include:
        cols = [c for c in include if c in out.columns]
        if not cols:
            raise ValueError("include no contiene columnas válidas")
        out = out[["ts"] + cols]

    return out


def make_target(df: pd.DataFrame, horizon: int = 1, target_type: str = "forward_return") -> pd.Series:
    """
    Construye objetivo a predecir a partir de velas.
    - horizon: número de barras hacia adelante
    - target_type: 'forward_return' -> (close_{t+h} / close_t) - 1
                  'forward_logret' -> log return
    Retorna Series indexada por ts_dt (misma indexación que features_from_candles).
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df2 = _ensure_ts_index(df)
    close = pd.to_numeric(df2["close"], errors="coerce")
    if target_type == "forward_return":
        target = (close.shift(-horizon) / close) - 1.0
    elif target_type == "forward_logret":
        target = np.log(close.shift(-horizon)) - np.log(close)
    else:
        raise ValueError("Unsupported target_type")

    # align index and drop last horizon rows (they will be NaN)
    target.name = "target"
    return target


# ---------------------------
# Training helpers
# ---------------------------
def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def _train_sklearn_regressor(X: pd.DataFrame, y: pd.Series, model_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
    """Entrena RandomForestRegressor (fallback) o LinearRegression si sample muy pequeño."""
    if RandomForestRegressor is None:
        raise RuntimeError("scikit-learn no está disponible en el entorno.")

    X2 = X.fillna(0).astype(float)
    y2 = y.astype(float)

    # train/test split
    if len(X2) < 50:
        # dataset pequeño -> usar regresión simple
        model = LinearRegression()
    else:
        params = model_params or {"n_estimators": 100, "max_depth": 8, "random_state": 42}
        model = RandomForestRegressor(**params)

    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds))) if mean_squared_error else None
    r2 = float(r2_score(y_test, preds)) if r2_score else None
    metrics = {"rmse": rmse, "r2": r2, "train_samples": len(X_train), "test_samples": len(X_test)}
    return model, metrics


def _train_lightgbm(X: pd.DataFrame, y: pd.Series, model_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
    """Entrena un LightGBM regresor si está disponible."""
    if lgb is None:
        raise RuntimeError("lightgbm no está instalado")

    X2 = X.fillna(0).astype(float)
    y2 = y.astype(float)

    params = model_params or {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}
    lgb_train = lgb.Dataset(X2, label=y2)
    booster = lgb.train(params, lgb_train, num_boost_round=100)
    # No hay split por defecto aquí — métricas mínimas
    preds = booster.predict(X2)
    rmse = float(np.sqrt(((preds - y2) ** 2).mean()))
    metrics = {"rmse": rmse, "train_samples": len(X2)}
    return booster, metrics


# ---------------------------
# Persistence: train_and_persist
# ---------------------------
def train_and_persist(storage: Any, asset: str, interval: str, model_name: str, X: pd.DataFrame, y: pd.Series,
                      model_type: str = "auto", model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Entrena un modelo y lo persiste en disco + storage.save_model_record.

    Parámetros:
      - storage: instancia con método save_model_record(asset, interval, model_name, metadata, path)
      - asset, interval, model_name: strings
      - X: DataFrame features (index ts_dt, has 'ts' column)
      - y: Series target (aligned index)
      - model_type: 'auto'|'lgb'|'sklearn'
      - model_params: dict con hyperparams

    Retorna metadata dict con métricas y ruta de archivo.
    """
    if X is None or X.empty:
        raise ValueError("X vacío")
    if y is None or y.empty:
        raise ValueError("y vacío")

    _ensure_models_dir()
    # ensure index alignment
    Xc = X.copy().drop(columns=["ts"], errors="ignore").fillna(0)
    # align y with X
    y_aligned = y.reindex(X.index).dropna()
    Xc = Xc.loc[y_aligned.index]

    # choose model
    chosen = model_type
    if model_type == "auto":
        chosen = "lgb" if lgb is not None else "sklearn"

    if chosen == "lgb":
        model, metrics = _train_lightgbm(Xc, y_aligned, model_params=model_params)
    elif chosen == "sklearn":
        model, metrics = _train_sklearn_regressor(Xc, y_aligned, model_params=model_params)
    else:
        raise ValueError("model_type desconocido")

    # persist model
    ts = int(time.time() * 1000)
    safe_name = f"{asset}_{interval}_{ts}_{model_name}".replace("/", "_")
    path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")

    # LightGBM booster no serializable con joblib? joblib works; otherwise dump booster with .save_model
    try:
        if joblib is None:
            raise RuntimeError("joblib no disponible para persistir modelos")
        joblib.dump(model, path)
    except Exception:
        # fallback para lightgbm: usar save_model
        if lgb is not None and isinstance(model, lgb.Booster):
            path = os.path.join(MODELS_DIR, f"{safe_name}.txt")
            model.save_model(path)
        else:
            raise

    # metadata
    metadata = {
        "asset": asset,
        "interval": interval,
        "model_name": model_name,
        "model_type": chosen,
        "model_params": model_params or {},
        "metrics": metrics,
        "saved_at": int(time.time() * 1000),
    }

    # save model record in storage if API available
    if storage and hasattr(storage, "save_model_record"):
        try:
            storage.save_model_record(asset, interval, model_name, metadata, path)
        except Exception:
            logger.exception("save_model_record falló — continuar de todos modos")

    return {"path": path, "metadata": metadata}


# ---------------------------
# Infer & persist
# ---------------------------
def infer_and_persist(storage: Any, asset: str, interval: str, model_name: Optional[str] = None,
                      model_path: Optional[str] = None, lookback: Optional[int] = None) -> pd.DataFrame:
    """
    Ejecuta inferencia con el último modelo (o model_path si se da) sobre velas recientes y persiste scores.

    Output:
      - DataFrame con columnas ts (ms) y score (dict)
    """
    if storage is None:
        raise ValueError("storage requerido para infer_and_persist")

    # obtener modelo: preferir model_path, luego storage.get_latest_model_record
    model = None
    used_path = model_path
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model_path no encontrado: {model_path}")
        if joblib is None:
            raise RuntimeError("joblib requerido para cargar modelos")
        try:
            model = joblib.load(model_path)
        except Exception:
            # try LightGBM load
            if lgb is not None:
                try:
                    model = lgb.Booster(model_file=model_path)
                except Exception:
                    raise
            else:
                raise
    else:
        rec = None
        if hasattr(storage, "get_latest_model_record"):
            rec = storage.get_latest_model_record(asset, interval)
        if rec is None:
            raise RuntimeError("No se encontró modelo en storage; proporciona model_path o entrena primero.")
        used_path = rec.get("path")
        if not used_path or not os.path.exists(used_path):
            raise FileNotFoundError(f"Ruta de modelo no válida en storage: {used_path}")
        if joblib is None:
            raise RuntimeError("joblib requerido para cargar modelos")
        try:
            model = joblib.load(used_path)
        except Exception:
            if lgb is not None:
                try:
                    model = lgb.Booster(model_file=used_path)
                except Exception:
                    raise
            else:
                raise

    # cargar velas recientes
    # lookback defines how many candles to fetch; try to use 1000 default
    lookback = int(lookback) if lookback else 1000
    try:
        df_candles = storage.load_candles(asset, interval, limit=lookback)
    except Exception as e:
        logger.exception("load_candles falló en infer_and_persist: %s", e)
        raise

    if df_candles is None or df_candles.empty:
        raise RuntimeError("No hay velas para inferir")

    feats = features_from_candles(df_candles)
    X = feats.drop(columns=["ts"], errors="ignore").fillna(0)
    X_idx = X.index
    # predict
    preds = None
    try:
        if lgb is not None and isinstance(model, lgb.Booster):
            preds = model.predict(X)
        else:
            preds = model.predict(X.values)
    except Exception as e:
        logger.exception("Error al predecir: %s", e)
        # intentar predict en bloques
        preds = []
        for i in range(0, len(X), 1000):
            block = X.iloc[i : i + 1000]
            try:
                if lgb is not None and isinstance(model, lgb.Booster):
                    preds_block = model.predict(block)
                else:
                    preds_block = model.predict(block.values)
                preds.extend(list(preds_block))
            except Exception:
                preds.extend([None] * len(block))
        preds = np.array(preds)

    # construir DataFrame para persistir
    df_scores = pd.DataFrame({"ts": feats["ts"].astype("int64"), "score": [None] * len(feats)}, index=feats.index)
    for i, p in enumerate(preds):
        # asume regresor -> valor numérico
        val = float(p) if p is not None and not np.isnan(p) else None
        df_scores.iat[i, df_scores.columns.get_loc("score")] = {"pred": val}

    # Persistir usando storage.save_scores (espera df_scores, asset, interval)
    try:
        storage.save_scores(df_scores, asset, interval)
    except Exception:
        logger.exception("storage.save_scores falló — intentar persistir por lotes")
        # intento de persistencia manual (si save_scores no existe)
        if hasattr(storage, "save_scores"):
            raise
        else:
            # fallback: intentar insertar fila por fila con save_scores/otro método no implementado -> raise
            raise RuntimeError("storage.save_scores no implementado y no se pudo persistir scores")

    return df_scores


# Export API
__all__ = [
    "features_from_candles",
    "make_target",
    "train_and_persist",
    "infer_and_persist",
]
