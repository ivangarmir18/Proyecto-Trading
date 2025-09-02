# core/ai_train.py
"""
Genera dataset para IA y entrena un LightGBM classifier que predice la probabilidad
de que un trade alcance target antes que stop (dentro de un horizonte).

Requisitos:
- core.storage tiene funciones load_candles(asset, interval) -> DataFrame
  y load_scores(asset, interval) -> DataFrame. Si no, use --db para sqlite fallback.
- Guarda:
  - models/ai_score_model.pkl  (joblib)
  - models/ai_feature_names.json (lista de features en orden)
Usage example:
python -m core.ai_train --asset BTCUSDT --interval 1h --db data/db/cryptos.db --out models/ai_score_model.pkl
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# try to import core.storage
try:
    from core import storage
    _HAS_STORAGE = True
except Exception:
    storage = None
    _HAS_STORAGE = False


def load_data(asset: str, interval: str, db_path: str = None):
    if _HAS_STORAGE:
        candles = storage.load_candles(asset, interval)
        scores = storage.load_scores(asset, interval)
        try:
            indicators = storage.load_indicators(asset, interval)
        except Exception:
            indicators = pd.DataFrame()
    else:
        if not db_path:
            raise RuntimeError("No core.storage and no --db provided")
        # fallback sqlite read
        import sqlite3
        con = sqlite3.connect(db_path)
        candles = pd.read_sql("SELECT ts, open, high, low, close, volume FROM candles WHERE asset=? AND interval=? ORDER BY ts",
                              con, params=(asset, interval))
        scores = pd.read_sql("SELECT ts, score, range_min, range_max, stop, target, multiplier FROM scores WHERE asset=? AND interval=? ORDER BY ts",
                             con, params=(asset, interval))
        try:
            ind_q = "SELECT c.ts as ts, i.ema9, i.ema40, i.atr, i.macd, i.macd_signal, i.rsi, i.support, i.resistance FROM candles c JOIN indicators i ON c.id=i.candle_id WHERE c.asset=? AND c.interval=? ORDER BY c.ts"
            indicators = pd.read_sql(ind_q, con, params=(asset, interval))
        except Exception:
            indicators = pd.DataFrame()
        con.close()

    # coerce types
    for df in (candles, scores, indicators):
        if df is None:
            df = pd.DataFrame()
        if not df.empty and df['ts'].dtype == object:
            try:
                df['ts'] = pd.to_datetime(df['ts']).astype(int) // 10**9
            except Exception:
                df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype('Int64')

    return candles, scores, indicators


def build_label_and_features(candles: pd.DataFrame, scores: pd.DataFrame, indicators: pd.DataFrame,
                             horizon: int = 24) -> Tuple[pd.DataFrame, List[str]]:
    """
    For each score row, produce label = 1 if target is reached before stop within next `horizon` candles.
    Features: score, ema_rel, ema_diff_norm, rsi, atr_rel, ret_1, ret_5, vol20
    """

    # merge scores onto candles using asof (score at or before candle.ts)
    dfc = candles.sort_values('ts').reset_index(drop=True).copy()
    dfs = scores.sort_values('ts').reset_index(drop=True).copy()
    dfc['ts_dt'] = pd.to_datetime(dfc['ts'], unit='s')
    dfs['ts_dt'] = pd.to_datetime(dfs['ts'], unit='s')
    merged = pd.merge_asof(dfc, dfs, on='ts_dt', direction='backward', tolerance=pd.Timedelta('1H'))
    merged = merged.sort_values('ts').reset_index(drop=True)

    # join indicators by ts if available
    if not indicators.empty:
        inds = indicators.sort_values('ts').reset_index(drop=True)
        inds['ts_dt'] = pd.to_datetime(inds['ts'], unit='s')
        merged = pd.merge_asof(merged, inds, on='ts_dt', direction='backward', tolerance=pd.Timedelta('1H'))
    merged = merged.drop(columns=[c for c in ['ts_dt'] if c in merged.columns], errors='ignore')

    # compute fallback indicators if missing
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
        rs = avg_gain / avg_loss
        merged['rsi'] = 100 - (100 / (1 + rs))
        merged['rsi'] = merged['rsi'].fillna(50)
    if 'atr' not in merged.columns:
        high_low = merged['high'] - merged['low']
        high_close = (merged['high'] - merged['close'].shift()).abs()
        low_close = (merged['low'] - merged['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        merged['atr'] = tr.rolling(window=14, min_periods=1).mean().fillna(0.0)

    # basic features
    merged['score_f'] = merged['score'].fillna(0.0).astype(float)
    merged['ema_rel'] = merged['close'] / (merged['ema40'] + 1e-9)
    merged['ema_diff_norm'] = (merged['ema9'] - merged['ema40']) / (merged['ema40'] + 1e-9)
    merged['atr_rel'] = merged['atr'] / (merged['close'] + 1e-9)
    merged['ret_1'] = merged['close'].pct_change(1).fillna(0.0)
    merged['ret_5'] = merged['close'].pct_change(5).fillna(0.0)
    merged['vol20'] = merged['close'].pct_change().rolling(20).std().fillna(0.0)

    feature_cols = ['score_f', 'ema_rel', 'ema_diff_norm', 'rsi', 'atr_rel', 'ret_1', 'ret_5', 'vol20']

    # construct labels by scanning forward within horizon for each row
    labels = []
    rows = []
    ts_to_index = {int(r['ts']): i for i, r in merged.iterrows()}

    for i, row in merged.iterrows():
        # require that row has stop & target
        if pd.isna(row.get('stop')) or pd.isna(row.get('target')):
            labels.append(np.nan)
            continue
        entry_idx = i + 1  # we assume entry at next candle open
        if entry_idx >= len(merged):
            labels.append(np.nan)
            continue
        stop = float(row['stop'])
        target = float(row['target'])
        hit_target_first = False
        hit_any = False

        for look in range(entry_idx, min(entry_idx + horizon, len(merged))):
            high = float(merged.iloc[look]['high'])
            low = float(merged.iloc[look]['low'])
            # if target reached and stop not reached prior
            if high >= target and low > stop:
                hit_target_first = True
                hit_any = True
                break
            if low <= stop and high < target:
                hit_target_first = False
                hit_any = True
                break
            if high >= target and low <= stop:
                # both in same candle -> choose by proximity to open
                open_price = float(merged.iloc[look]['open'])
                if abs(target - open_price) <= abs(open_price - stop):
                    hit_target_first = True
                else:
                    hit_target_first = False
                hit_any = True
                break
        if not hit_any:
            labels.append(0)  # treat as negative (target not reached within horizon)
        else:
            labels.append(1 if hit_target_first else 0)

    merged['label'] = labels
    # drop rows without label
    df_model = merged.dropna(subset=['label']).reset_index(drop=True)
    # keep only useful cols
    X = df_model[feature_cols].fillna(0.0)
    y = df_model['label'].astype(int).values

    return X, y, feature_cols, df_model


def train_model(X: pd.DataFrame, y: np.ndarray, params: dict = None, test_size: float = 0.2, random_state: int = 42):
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 200
        }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    lgbm = lgb.LGBMClassifier(**params)
    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=20, verbose=False)
    preds = lgbm.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return lgbm, auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset', required=True)
    p.add_argument('--interval', default='1h')
    p.add_argument('--db', default=None, help='sqlite fallback')
    p.add_argument('--horizon', type=int, default=24)
    p.add_argument('--out', default='models/ai_score_model.pkl')
    p.add_argument('--feat_out', default='models/ai_feature_names.json')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--random_state', type=int, default=42)
    args = p.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    candles, scores, indicators = load_data(args.asset, args.interval, db_path=args.db)
    X, y, feature_cols, df_model = build_label_and_features(candles, scores, indicators, horizon=args.horizon)
    if X.empty or len(y) == 0:
        raise RuntimeError("Not enough labeled data to train")

    model, auc = train_model(X, y, test_size=args.test_size, random_state=args.random_state)
    joblib.dump(model, args.out)
    with open(args.feat_out, 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, indent=2)

    print(f\"Model trained. AUC on holdout: {auc:.4f}\")\n```

---

### core/ai_inference.py (mejorada)
Reemplaza tu fichero con esta versión — ahora carga también el archivo `models/ai_feature_names.json` para asegurar orden correcto.
```python
# core/ai_inference.py
import joblib
import json
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ai_score_model.pkl"
FEATURES_PATH = Path(__file__).resolve().parents[1] / "models" / "ai_feature_names.json"

_model = None
_model_features = None

def load_model():
    global _model, _model_features
    if _model is None:
        if MODEL_PATH.exists():
            _model = joblib.load(MODEL_PATH)
        else:
            return None
    if _model_features is None:
        if FEATURES_PATH.exists():
            with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
                _model_features = json.load(f)
        else:
            _model_features = None
    return _model

def predict_prob(feature_dict: dict):
    \"\"\"Return probability in [0,1] or None if model missing.

    feature_dict: mapping feature_name -> value
    The function will order the features according to the saved feature list.
    \"\"\"
    mdl = load_model()
    if mdl is None:
        return None
    if _model_features is None:
        raise RuntimeError('Feature names file not found (models/ai_feature_names.json). Train model first.')

    X = np.array([[feature_dict.get(fn, 0.0) for fn in _model_features]])
    try:
        if hasattr(mdl, 'predict_proba'):
            return float(mdl.predict_proba(X)[0, 1])
        # fallback: if model returns raw score, map with sigmoid
        preds = mdl.predict(X)
        return float(1.0 / (1.0 + np.exp(-preds[0])))
    except Exception as e:
        # if prediction fails, return None to indicate no AI contribution
        return None
