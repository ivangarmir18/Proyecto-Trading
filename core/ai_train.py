# core/ai_train.py
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from pathlib import Path
from core.storage import load_candles  # o función que te devuelva df timeline

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def make_training_examples(symbol, interval="1h", horizon=24, threshold_target=1.3):
    """
    Recorre candles y crea features + label.
    label=1 si en next `horizon` candles price reaches target (e.g. close + threshold_target*ATR)
    and before stop (close - threshold_stop*ATR). Adapt according to your target rule.
    """
    df = load_candles(symbol, interval=interval)
    if df.empty:
        return pd.DataFrame()
    # compute indicators (reuse indicator functions)
    df["EMA9"] = ema(df["close"], 9)
    df["EMA40"] = ema(df["close"], 40)
    df["RSI"] = rsi(df["close"], 14)
    df["ATR"] = atr(df, 14)
    macd_line, sig, hist = macd(df["close"])
    df["MACD_hist"] = hist
    # example features:
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["vol_over_ma5"] = df["volume"] / (df["volume"].rolling(5).mean() + 1e-9)
    # label construction:
    rows = []
    for i in range(len(df)-horizon):
        price = df.loc[i, "close"]
        atr = df.loc[i, "ATR"]
        target_level = price + threshold_target * atr
        stop_level = price - threshold_target * atr
        future = df.iloc[i+1:i+1+horizon]
        hit_target = (future["high"] >= target_level).any()
        hit_stop = (future["low"] <= stop_level).any()
        label = 1 if (hit_target and not hit_stop) else 0
        feat = {
            "EMA9_EMA40_gap": df.loc[i,"EMA9"] - df.loc[i,"EMA40"],
            "RSI": df.loc[i,"RSI"],
            "ATR_rel": (atr / price) if price else 0,
            "MACD_hist": df.loc[i,"MACD_hist"],
            "vol_over_ma5": df.loc[i,"vol_over_ma5"],
            "ret_1": df.loc[i,"ret_1"],
            "ret_3": df.loc[i,"ret_3"],
            "label": label,
            "ts": df.loc[i,"ts"]
        }
        rows.append(feat)
    return pd.DataFrame(rows)

def train_model(X, y, model_path: Path):
    # simple LightGBM
    train_data = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "metric":"auc", "verbosity": -1}
    bst = lgb.train(params, train_data, num_boost_round=200)
    joblib.dump(bst, model_path)
    return bst

def build_and_save(symbols, interval="1h"):
    # aggregate examples for many symbols
    frames = []
    for s in symbols:
        df_ex = make_training_examples(s, interval=interval, horizon=24)
        if not df_ex.empty:
            frames.append(df_ex)
    D = pd.concat(frames, ignore_index=True)
    # dropna, split, train
    D = D.dropna()
    X = D.drop(columns=["label","ts"])
    y = D["label"]
    model_path = MODEL_DIR / "ai_score_model.pkl"
    model = train_model(X, y, model_path)
    print("Model saved to", model_path)
    return model_path
