# core/indicators.py
"""
Gestión centralizada de indicadores técnicos.

Este módulo unifica el cálculo de indicadores desde los scripts en `indicators/`
y proporciona una API uniforme para aplicarlos a activos. Además, permite
guardar los resultados en la base de datos si se pasa un PostgresStorage.

Ejemplo de uso:
    from core.indicators import apply_indicators
    df = fetcher.fetch_asset("BTCUSDT", "5m", "2024-01-01", "2024-01-10")
    df_ind = apply_indicators("BTCUSDT", df, {
        "ema": {"period": 14},
        "rsi": {"period": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
    }, storage=storage, interval="5m")
"""

from __future__ import annotations
import logging
import pandas as pd
from typing import Dict, Any, Optional

# Importamos los módulos de indicadores existentes
from indicators.ema import ema
from indicators.rsi import rsi
from indicators.macd import macd
# Si tienes ATR, Bollinger u otros, los importas aquí
# from indicators.atr import atr

logger = logging.getLogger("indicators")

# Registro de funciones disponibles
_INDICATOR_FUNCS = {
    "ema": ema,
    "rsi": rsi,
    "macd": macd,
    # "atr": atr,
}


def list_available_indicators() -> list[str]:
    """
    Devuelve una lista con los nombres de indicadores soportados.
    """
    return list(_INDICATOR_FUNCS.keys())


def calculate_indicator(name: str, df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calcula un indicador sobre un DataFrame.

    Args:
        name: nombre del indicador ("ema", "rsi", "macd", etc.)
        df: DataFrame OHLCV (con columnas ['ts','open','high','low','close','volume'])
        params: parámetros específicos del indicador (ej: period, fast, slow, signal)

    Returns:
        DataFrame con nuevas columnas para el indicador calculado.
    """
    name = name.lower()
    if name not in _INDICATOR_FUNCS:
        raise ValueError(f"Indicador no soportado: {name}")
    func = _INDICATOR_FUNCS[name]
    try:
        df_out = func(df.copy(), **params)
        return df_out
    except Exception as e:
        logger.exception("Error calculando indicador %s: %s", name, e)
        raise


def apply_indicators(
    asset: str,
    df: pd.DataFrame,
    indicators_cfg: Dict[str, Dict[str, Any]],
    storage: Optional[Any] = None,
    interval: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aplica múltiples indicadores a un DataFrame y opcionalmente guarda resultados en BD.

    Args:
        asset: símbolo del activo (ej: "BTCUSDT" o "AAPL")
        df: DataFrame OHLCV
        indicators_cfg: diccionario con configuraciones, ej:
            {
                "ema": {"period": 14},
                "rsi": {"period": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9}
            }
        storage: instancia de PostgresStorage (opcional)
        interval: timeframe (ej: "5m", "1h") si se quiere guardar en BD

    Returns:
        DataFrame con todas las columnas de indicadores añadidas.
    """
    out_df = df.copy()
    for ind_name, params in indicators_cfg.items():
        logger.info("Calculando indicador %s para %s con params=%s", ind_name, asset, params)
        out_df = calculate_indicator(ind_name, out_df, **params)

    if storage and interval:
        try:
            storage.upsert_indicators(asset, interval, out_df, indicators_cfg)
            logger.info("Indicadores guardados en BD para %s [%s]", asset, interval)
        except Exception as e:
            logger.exception("Error guardando indicadores en BD: %s", e)

    return out_df

