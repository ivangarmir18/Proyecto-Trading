# core/ai.py
"""
IA: entrenamiento y predicción para señales/score alternativas.

Características:
 - Prepara features desde tabla `indicators` (pd.json_normalize sobre `value`).
 - Entrena un modelo supervisado (RandomForestClassifier por defecto) para predecir probabilidad de "movimiento favorable".
 - Guarda/Carga modelos con joblib.
 - Funciones robustas: validación de shapes, manejo de NaNs, persistencia.
 - Si scikit-learn no está instalado, informa claramente (pip install scikit-learn joblib).
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import os
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    import joblib
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


# -------------------- feature engineering -------------------- #
def prepare_features_from_indicators(indicators_df: pd.DataFrame, include_price: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    indicators_df: DataFrame with columns ['ts', 'value'] where value is dict or normalized fields.
    Returns:
      - X: DataFrame features (one row per ts)
      - meta: pd.Series with ts aligned (index preserved)
    Notes:
      - If 'value' column is dicts, it will call pd.json_normalize.
      - It does not produce target y; label creation is separate.
    """
    if indicators_df is None or indicators_df.empty:
        raise ValueError("indicators_df vacío")
    df = indicators_df.copy().reset_index(drop=True)
    if 'value' in df.columns:
        # expand
        values = pd.json_normalize(df['value'])
        # flatten nested dictionaries (support.*, fibonacci.*)
        # pandas.json_normalize already flattens using dot notation
        X = values
    else:
        # assume already normalized
        X = df.copy()
    # include close price if present in df original (useful for atr normalization)
    if include_price and 'close' in df.columns:
        X['close'] = df['close'].values
    # ts meta
    meta_ts = pd.Series(df['ts'].values, name='ts')
    # drop columns that are non-numeric if any remain (like nested objects)
    for c in X.columns:
        if X[c].dtype == 'O':
            # try convert to numeric or drop
            try:
                X[c] = pd.to_numeric(X[c], errors='coerce')
            except Exception:
                X = X.drop(columns=[c])
    # fillna with median
    X = X.fillna(X.median(numeric_only=True))
    return X, meta_ts


def label_next_return(candles_df: pd.DataFrame, forward: int = 3, threshold: float = 0.0) -> pd.Series:
    """
    Crea etiqueta binaria basada en el retorno forward N velas.
    - forward: número de barras adelante para calcular return (close_t+forward / close_t - 1)
    - threshold: mínimo return para etiquetar como positivo (ej. 0.01 -> 1%).
    Devuelve pd.Series index-aligned con candles_df index and values in {0,1}.
    """
    if 'close' not in candles_df.columns:
        raise ValueError("candles_df debe contener 'close'")
    close = candles_df['close'].astype(float)
    fwd = close.shift(-forward)
    returns = (fwd - close) / close
    labels = (returns > threshold).astype(int)
    # last forward values will be NaN -> drop or set 0
    labels = labels.fillna(0).astype(int)
    return labels


# -------------------- model training / persistence -------------------- #
def _ensure_sklearn():
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn y joblib no están instalados. Instálalos con `pip install scikit-learn joblib`.")


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: Optional[str] = None,
    test_size: float = 0.2,
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Entrena un modelo (RandomForestClassifier por defecto) y devuelve métricas + ruta guardada.
    - X: features (DataFrame)
    - y: labels (Series)
    - model_path: path para guardar el modelo (joblib)
    - model_params: parámetros pasados al RandomForestClassifier
    """
    _ensure_sklearn()
    model_params = model_params or {"n_estimators": 200, "max_depth": 6, "n_jobs": -1, "random_state": random_state}
    if X.shape[0] < 10:
        raise ValueError("Muy pocos ejemplos para entrenar")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()>1 else None)
    clf = RandomForestClassifier(**model_params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    metrics = {"accuracy": float(accuracy_score(y_test, preds))}
    if proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics["roc_auc"] = None
    # save
    model_file = model_path or os.getenv("AI_MODEL_PATH", "models/watchlist_model.joblib")
    os.makedirs(os.path.dirname(model_file) or ".", exist_ok=True)
    joblib.dump(clf, model_file)
    log.info("Modelo guardado en %s", model_file)
    return {"model_path": model_file, "metrics": metrics, "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0])}


def load_model(model_path: Optional[str] = None):
    _ensure_sklearn()
    model_file = model_path or os.getenv("AI_MODEL_PATH", "models/watchlist_model.joblib")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Modelo no encontrado en {model_file}")
    return joblib.load(model_file)


def predict_proba(model, X: pd.DataFrame) -> pd.Series:
    _ensure_sklearn()
    if X is None or X.empty:
        return pd.Series([], dtype=float)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    return pd.Series(proba, index=X.index)


# -------------------- convenience end-to-end helpers -------------------- #
def train_and_evaluate_from_storage(storage, asset: str, interval: str, forward: int = 3, threshold: float = 0.0, model_path: Optional[str] = None, model_params: Optional[Dict[str, Any]] = None):
    """
    End-to-end: carga indicadores y candles desde storage, prepara features/labels, entrena modelo y lo guarda.
    storage: debe ofrecer export_last_candles(asset,interval,limit) y/o poder leer indicators table.
    """
    # 1) get candles
    candles = pd.DataFrame(storage.export_last_candles(asset, interval, limit=5000))
    if candles.empty:
        raise RuntimeError("No hay velas para entrenar")
    # 2) read indicators table if available (prefer wide)
    impl = getattr(storage, "impl", None)
    indicators_df = None
    if impl and hasattr(impl, "engine"):
        try:
            conn = impl.engine.connect()
            q = "select ts, value from indicators where asset = :asset and interval = :interval order by ts asc"
            indicators_df = pd.read_sql(q, conn, params={"asset": asset, "interval": interval})
            conn.close()
        except Exception:
            log.exception("No se pudieron leer indicadores desde DB, intentaremos recalcular")
            indicators_df = None
    if indicators_df is None or indicators_df.empty:
        # try to recompute indicators locally using core.indicators if present
        try:
            from .indicators import apply_indicators
            ind_df = apply_indicators(candles, asset=asset, interval=interval)
            indicators_df = ind_df[['ts','value']].copy()
        except Exception:
            raise RuntimeError("No hay indicadores disponibles y no se pudieron recomputar")
    # 3) prepare features
    X, meta_ts = prepare_features_from_indicators(indicators_df)
    # align candles and X by ts
    merged = pd.merge(pd.DataFrame({'ts': meta_ts.values}), candles[['ts','close']], left_on='ts', right_on='ts', how='left')
    # create labels using candles aligned by ts (matching indexes)
    # find positions where ts aligns in candles
    # build label series based on candles frame
    labels = label_next_return(candles, forward=forward, threshold=threshold)
    # align labels to X by ts
    labels_df = pd.DataFrame({'ts': candles['ts'].values, 'label': labels.values})
    X_with_ts = X.copy()
    X_with_ts['ts'] = meta_ts.values
    merged2 = pd.merge(X_with_ts, labels_df, on='ts', how='left')
    merged2['label'] = merged2['label'].fillna(0).astype(int)
    y = merged2['label']
    X_final = merged2.drop(columns=['ts','label'])
    # 4) train
    result = train_model(X_final, y, model_path=model_path, model_params=model_params)
    return result


# -------------------- example usage -------------------- #
if __name__ == "__main__":
    print("IA module: prueba rápida (necesita sklearn).")
    if not _SKLEARN_AVAILABLE:
        print("scikit-learn no disponible. Instalar con `pip install scikit-learn joblib`.")
    else:
        print("sklearn disponible.")
