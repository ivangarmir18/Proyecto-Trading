#EMA
# indicators/ema.py
import pandas as pd

def ema(series: pd.Series, period: int = 9) -> pd.Series:
    """
    Exponential moving average
    """
    return series.ewm(span=period, adjust=False).mean()
