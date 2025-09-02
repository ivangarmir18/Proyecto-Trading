# tests/test_ai_train.py
import os
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pytest

# import functions under test
from core import ai_train

# --- Fixtures: small synthetic dataset ---
@pytest.fixture
def small_candles():
    # 30 candles, 1-hour spaced
    ts_base = 1609459200  # 2021-01-01 00:00:00 UTC
    rows = []
    price = 100.0
    for i in range(30):
        open_p = price + np.random.uniform(-0.5, 0.5)
        high = open_p + np.random.uniform(0, 1.0)
        low = open_p - np.random.uniform(0, 1.0)
        close = low + np.random.uniform(0, high - low + 1e-6)
        vol = np.random.uniform(10, 100)
        rows.append({"ts": ts_base + i * 3600, "open": open_p, "high": high, "low": low, "close": close, "volume": vol})
        price = close
    return pd.DataFrame(rows)

@pytest.fixture
def small_scores(small_candles):
    # put a score every candle with stop/target around price +/- ATR-ish
    df = small_candles.copy()
    df['ts'] = df['ts']
    df['score'] = np.linspace(0.2, 0.9, len(df))
    # simple ATR proxy
    df['atr'] = (df['high'] - df['low']).rolling(3, min_periods=1).mean().fillna(1.0)
    df['stop'] = df['close'] - 1.2 * df['atr']
    df['target'] = df['close'] + 1.2 * df['atr']
    return df[['ts', 'score', 'stop', 'target']]

@pytest.fixture(autouse=True)
def patch_storage(monkeypatch, small_candles, small_scores):
    """
    Monkeypatch storage_mod functions so load_data uses these fixtures.
    """
    class DummyStorage:
        @staticmethod
        def load_candles(asset, interval):
            return small_candles.copy()
        @staticmethod
        def load_scores(asset, interval):
            return small_scores.copy()
        @staticmethod
        def load_indicators(asset, interval):
            return pd.DataFrame()
    monkeypatch.setattr(ai_train, "storage_mod", DummyStorage)
    # and tell ai_train _HAS_STORAGE True so load_data uses storage_mod
    monkeypatch.setattr(ai_train, "_HAS_STORAGE", True)
    yield

def test_build_label_and_features_basic(small_candles, small_scores):
    X, y, feat_cols = ai_train.build_label_and_features(small_candles, small_scores, pd.DataFrame(), horizon=6)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert len(feat_cols) > 0
    # at least one example labeled
    assert len(y) > 0
    # features present
    for c in feat_cols:
        assert c in X.columns

def test_train_model_for_asset_saves(tmp_path):
    # train a tiny model and check files saved
    asset = "TEST"
    interval = "1h"
    out_prefix = f"test_{np.random.randint(0,9999)}"
    # override global dirs to tmp_path
    ai_train.MODELS_DIR = Path(tmp_path) / "models"
    ai_train.REPORTS_DIR = Path(tmp_path) / "reports"
    ai_train.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ai_train.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path, feat_path, metrics = ai_train.train_model_for_asset(asset, interval, db_path=None, horizon=6, out_prefix=out_prefix)
    assert model_path.exists()
    assert feat_path.exists()
    assert metrics and "auc" in metrics
    # model can be loaded
    mdl = joblib.load(model_path)
    assert hasattr(mdl, "predict_proba")

    # feature list valid json
    with feat_path.open("r", encoding="utf-8") as fh:
        feats = json.load(fh)
    assert isinstance(feats, list)
