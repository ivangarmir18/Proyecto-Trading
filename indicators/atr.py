#ATR
# indicators/atr.py
import pandas as pd

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    df must contain columns: high, low, close
    Returns ATR as a pandas Series (same index as df)
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    return atr_series.fillna(0.0)
