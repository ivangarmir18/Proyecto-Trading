# core/ai_train.py
"""
AI training pipeline (LightGBM) para Watchlist.

Características:
- Usa core.storage si está; fallback a sqlite vía --db.
- Funciones exportadas: load_data, build_label_and_features, train_model_for_asset, train_multi_assets.
- CLI para entrenar single o multi.
- Compatible con Windows: usa callbacks para early stopping en LGBMClassifier.fit(..., callbacks=[...]).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb

# try to import storage module (core.storage or core.storage_postgres)
try:
    from core import storage as storage_mod  # type: ignore
    _HAS_STORAGE = True
except Exception:
    try:
        import core.storage_postgres as storage_mod  # type: ignore
        _HAS_STORAGE = True
    except Exception:
        storage_mod = None
        _HAS_STORAGE = False

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(asset: str, interval: str, db_path: Optional[str] = None):
    """
    Devuelve (candles, scores, indicators) como DataFrames.
    Prioriza core.storage (Postgres), si no usa sqlite `db_path`.
    """
    if _HAS_STORAGE and storage_mod is not None:
        try:
            candles = storage_mod.load_candles(asset, interval)
            scores = storage_mod.load_scores(asset, interval)
            try:
                indicators = storage_mod.load_indicators(asset, interval)
            except Exception:
                indicators = pd.DataFrame()
            return (candles or pd.DataFrame(),
                    scores or pd.DataFrame(),
                    indicators or pd.DataFrame())
        except Exception:
            # fallback to sqlite below
            pass

    # fallback: sqlite path required
    if not db_path:
        raise RuntimeError("No core.storage disponible y no se proporcionó --db para fallback sqlite.")
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        candles = pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE asset=? AND interval=? ORDER BY ts",
            con, params=(asset, interval)
        )
        scores = pd.read_sql_query(
            "SELECT ts, score, range_min, range_max, stop, target, multiplier FROM scores WHERE asset=? AND interval=? ORDER BY ts",
            con, params=(asset, interval)
        )
        try:
            ind_q = ("SELECT c.ts as ts, i.ema9, i.ema40, i.atr, i.macd, i.macd_signal, i.rsi, i.support, i.resistance "
                     "FROM candles c JOIN indicators i ON c.id=i.candle_id WHERE c.asset=? AND c.interval=? ORDER BY c.ts")
            indicators = pd.read_sql_query(ind_q, con, params=(asset, interval))
        except Exception:
            indicators = pd.DataFrame()
    finally:
        con.close()

    # coerce types
    for df in (candles, scores, indicators):
        if df is None:
            df = pd.DataFrame()
        if not df.empty and df['ts'].dtype == object:
            try:
                df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype('Int64')
            except Exception:
                pass

    return candles, scores, indicators


def build_label_and_features(candles: pd.DataFrame, scores: pd.DataFrame, indicators: pd.DataFrame,
                             horizon: int = 24) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Construye X (DataFrame), y (np.ndarray) y lista de feature columns.
    Label: 1 si target alcanzado antes que stop dentro del horizon.
    Features: ['score_f','ema_rel','ema_diff_norm','rsi','atr_rel','ret_1','ret_5','vol20']
    """
    if candles is None or candles.empty:
        return pd.DataFrame(), np.array([]), []

    dfc = candles.sort_values('ts').reset_index(drop=True).copy()
    dfc['ts_dt'] = pd.to_datetime(dfc['ts'], unit='s')

    dfs = (scores.sort_values('ts').reset_index(drop=True).copy() if scores is not None else pd.DataFrame())
    if not dfs.empty:
        dfs['ts_dt'] = pd.to_datetime(dfs['ts'], unit='s')
        merged = pd.merge_asof(dfc, dfs, left_on='ts_dt', right_on='ts_dt',
                               direction='backward', tolerance=pd.Timedelta('1h'))
    else:
        merged = dfc.copy()
        merged['score'] = np.nan
        merged['range_min'] = np.nan
        merged['range_max'] = np.nan
        merged['stop'] = np.nan
        merged['target'] = np.nan

    # merge indicators if present
    if indicators is not None and not indicators.empty:
        inds = indicators.sort_values('ts').reset_index(drop=True)
        inds['ts_dt'] = pd.to_datetime(inds['ts'], unit='s')
        merged = pd.merge_asof(merged, inds, left_on='ts_dt', right_on='ts_dt',
                               direction='backward', tolerance=pd.Timedelta('1h'))

    merged = merged.sort_values('ts').reset_index(drop=True)

    # fallback indicators if missing
    if 'ema9' not in merged.columns:
        merged['ema9'] = merged['close'].ewm(span=9, adjust=False).mean()
    if 'ema40' not in merged.columns:
        merged['ema40'] = merged['close'].ewm(span=40, adjust=False).mean()
    if 'rsi' not in merged.columns:
        delta = merged['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        merged['rsi'] = 100 - (100 / (1 + rs))
        merged['rsi'] = merged['rsi'].fillna(50)
    if 'atr' not in merged.columns:
        high_low = merged['high'] - merged['low']
        high_close = (merged['high'] - merged['close'].shift()).abs()
        low_close = (merged['low'] - merged['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        merged['atr'] = tr.rolling(window=14, min_periods=1).mean().fillna(0.0)

    # features
    merged['score_f'] = merged['score'].fillna(0.0).astype(float)
    merged['ema_rel'] = merged['close'] / (merged['ema40'] + 1e-9)
    merged['ema_diff_norm'] = (merged['ema9'] - merged['ema40']) / (merged['ema40'] + 1e-9)
    merged['atr_rel'] = merged['atr'] / (merged['close'] + 1e-9)
    merged['ret_1'] = merged['close'].pct_change(1).fillna(0.0)
    merged['ret_5'] = merged['close'].pct_change(5).fillna(0.0)
    merged['vol20'] = merged['close'].pct_change().rolling(20).std().fillna(0.0)

    feature_cols = ['score_f', 'ema_rel', 'ema_diff_norm', 'rsi', 'atr_rel', 'ret_1', 'ret_5', 'vol20']

    # construct labels scanning forward
    labels = []
    for i, row in merged.iterrows():
        if pd.isna(row.get('stop')) or pd.isna(row.get('target')):
            labels.append(np.nan)
            continue
        entry_idx = i + 1
        if entry_idx >= len(merged):
            labels.append(np.nan)
            continue
        stop = float(row['stop'])
        target = float(row['target'])
        hit_any = False
        hit_target_first = False
        for look in range(entry_idx, min(entry_idx + horizon, len(merged))):
            high = float(merged.iloc[look]['high'])
            low = float(merged.iloc[look]['low'])
            if high >= target and low > stop:
                hit_target_first = True
                hit_any = True
                break
            if low <= stop and high < target:
                hit_target_first = False
                hit_any = True
                break
            if high >= target and low <= stop:
                open_price = float(merged.iloc[look]['open'])
                if abs(target - open_price) <= abs(open_price - stop):
                    hit_target_first = True
                else:
                    hit_target_first = False
                hit_any = True
                break
        if not hit_any:
            labels.append(0)
        else:
            labels.append(1 if hit_target_first else 0)

    merged['label'] = labels
    df_model = merged.dropna(subset=['label']).reset_index(drop=True)
    if df_model.empty:
        return pd.DataFrame(), np.array([]), feature_cols

    X = df_model[feature_cols].fillna(0.0)
    y = df_model['label'].astype(int).values
    # return X (DataFrame), y (np.ndarray), feature_cols
    return X, y, feature_cols


def train_model_for_asset(asset: str, interval: str, db_path: Optional[str] = None,
                          horizon: int = 24, out_prefix: Optional[str] = None,
                          test_size: float = 0.2, random_state: int = 42):
    """
    Entrena un modelo LightGBM para un asset+interval y guarda artefactos.
    Retorna (model_path, feat_path, metrics_dict)
    """
    candles, scores, indicators = load_data(asset, interval, db_path=db_path)
    X, y, feature_cols = build_label_and_features(candles, scores, indicators, horizon=horizon)
    if X is None or X.empty or len(y) == 0:
        raise RuntimeError("No hay suficientes ejemplos etiquetados para entrenar")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 500
    }
    clf = lgb.LGBMClassifier(**params)

    # Use callbacks for early stopping & disable verbose logging
    try:
        callbacks = [lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
    except Exception:
        # fallback to empty callbacks if something odd in installed version
        callbacks = []

    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', callbacks=callbacks)

    preds = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, preds))
    acc = float(accuracy_score(y_test, (preds > 0.5).astype(int)))

    prefix = out_prefix or f"{asset}_{interval}"
    model_path = MODELS_DIR / f"{prefix}_ai_score_model.pkl"
    feat_path = MODELS_DIR / f"{prefix}_ai_feature_names.json"
    metrics_path = REPORTS_DIR / f"{prefix}_metrics.json"

    joblib.dump(clf, model_path)
    with feat_path.open("w", encoding="utf-8") as fh:
        json.dump(feature_cols, fh, indent=2)
    metrics = {"asset": asset, "interval": interval, "auc": auc, "accuracy": acc,
               "n_train": int(len(y_train)), "n_test": int(len(y_test))}
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return model_path, feat_path, metrics


def train_multi_assets(assets: List[str], interval: str, db_path: Optional[str] = None,
                       horizon: int = 24, out_prefix: str = "global"):
    """
    Entrena un modelo global sobre múltiples assets (concatena ejemplos).
    """
    frames_X = []
    frames_y = []
    for asset in assets:
        try:
            candles, scores, indicators = load_data(asset, interval, db_path=db_path)
            X, y, feature_cols = build_label_and_features(candles, scores, indicators, horizon=horizon)
            if X is None or X.empty:
                continue
            frames_X.append(X)
            frames_y.append(pd.Series(y))
        except Exception as e:
            print(f"[train_multi] skipping {asset}: {e}")
            continue

    if not frames_X:
        raise RuntimeError("No hay datos para entrenar en los assets indicados")

    X_all = pd.concat(frames_X, ignore_index=True)
    y_all = pd.concat(frames_y, ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 500
    }
    clf = lgb.LGBMClassifier(**params)
    try:
        callbacks = [lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
    except Exception:
        callbacks = []

    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', callbacks=callbacks)

    preds = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, preds))
    acc = float(accuracy_score(y_test, (preds > 0.5).astype(int)))

    model_path = MODELS_DIR / f"{out_prefix}_ai_score_model.pkl"
    feat_path = MODELS_DIR / f"{out_prefix}_ai_feature_names.json"
    metrics_path = REPORTS_DIR / f"{out_prefix}_metrics.json"

    joblib.dump(clf, model_path)
    with feat_path.open("w", encoding="utf-8") as fh:
        json.dump(list(X_all.columns), fh, indent=2)
    metrics = {"assets": assets, "interval": interval, "auc": auc, "accuracy": acc,
               "n_train": int(len(y_train)), "n_test": int(len(y_test))}
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return model_path, feat_path, metrics


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--asset', help='Símbolo (ej: BTCUSDT)')
    p.add_argument('--assets', help='Lista coma-separada de símbolos para modo multi')
    p.add_argument('--assets-file', help='CSV con columna symbol')
    p.add_argument('--interval', default='1h')
    p.add_argument('--db', default=None, help='Ruta sqlite si no hay core.storage disponible')
    p.add_argument('--horizon', type=int, default=24)
    p.add_argument('--out-prefix', default=None)
    p.add_argument('--mode', choices=['single','multi'], default='single')
    args = p.parse_args()

    if args.mode == 'single':
        if not args.asset:
            raise SystemExit("En modo single debes pasar --asset")
        model_path, feat_path, metrics = train_model_for_asset(args.asset, args.interval, db_path=args.db,
                                                               horizon=args.horizon, out_prefix=args.out_prefix)
        print("Modelo guardado en:", model_path)
        print("Features guardadas en:", feat_path)
        print("Metrics:", metrics)
    else:
        assets = []
        if args.assets:
            assets += [s.strip() for s in args.assets.split(",") if s.strip()]
        if args.assets_file:
            df = pd.read_csv(args.assets_file)
            cols_low = [c.lower() for c in df.columns]
            if "symbol" in cols_low:
                key = df.columns[cols_low.index("symbol")]
                assets += df[key].dropna().astype(str).tolist()
            else:
                assets += df.iloc[:, 0].dropna().astype(str).tolist()
        if not assets:
            raise SystemExit("No assets para modo multi")
        model_path, feat_path, metrics = train_multi_assets(assets, args.interval, db_path=args.db,
                                                           horizon=args.horizon, out_prefix=args.out_prefix or "global")
        print("Modelo guardado en:", model_path)
        print("Features guardadas en:", feat_path)
        print("Metrics:", metrics)


if __name__ == "__main__":
    cli()
