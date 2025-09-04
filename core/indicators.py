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

# --- Inicio parche core/indicators.py: registrar ATR y wrapper robusto ---
import inspect
import pandas as pd
from indicators.atr import atr as _atr  # registrar ATR si existe

# registrar atr si no estaba ya
try:
    if "atr" not in _INDICATOR_FUNCS:
        _INDICATOR_FUNCS["atr"] = _atr
except Exception:
    # si por alguna razón _INDICATOR_FUNCS no existe, no romper
    pass

def _name_for_scalar_indicator(name: str, params: dict) -> str:
    """
    Genera nombre de columna para indicadores que devuelven una sola serie.
    Ej: ema period=14 -> 'ema_14'
    """
    if not params:
        return name
    # prefer 'period' param, si existe
    p = params.get("period") or params.get("window") or params.get("length")
    if p:
        return f"{name}_{p}"
    return name

def _call_indicator_func(func, name, df: pd.DataFrame, **params):
    """
    Intenta varias firmas de llamada y normaliza la salida a DataFrame con columnas ya nombradas.
    - intentos de llamada: func(series, **params) -> Series
                       func(df, **params) -> Series/DataFrame/tuple
                       func(close=series, **params)
    - normaliza a DataFrame con nombres razonables.
    """
    # prefer close series for single-series indicators
    close = None
    if isinstance(df, pd.DataFrame) and "close" in df.columns:
        close = df["close"]

    # intentos de llamada ordenados
    call_attempts = []
    if close is not None:
        call_attempts.append(lambda: func(close, **params))
    call_attempts.append(lambda: func(df, **params))
    call_attempts.append(lambda: func(**{"series": close, **params}) if close is not None else func(**params))
    call_attempts.append(lambda: func(**params))

    last_exc = None
    out = None
    for attempt in call_attempts:
        try:
            out = attempt()
            break
        except TypeError as e:
            last_exc = e
            continue
        except Exception as e:
            # si hay otro error, lo elevamos (indicador probablemente falló)
            raise

    if out is None:
        raise RuntimeError(f"No se pudo ejecutar indicador {name}; último error: {last_exc}")

    # normalizar salida
    # 1) tuple (macd_line, signal_line, hist)
    if isinstance(out, tuple) or isinstance(out, list):
        # caso típico: macd -> (macd_line, signal, hist)
        cols = {}
        if len(out) >= 1:
            cols[f"{name}_line"] = pd.Series(out[0], index=close.index if close is not None else None) if isinstance(out[0], (pd.Series, list)) else pd.Series(out[0])
        if len(out) >= 2:
            cols[f"{name}_signal"] = pd.Series(out[1], index=close.index if close is not None else None) if isinstance(out[1], (pd.Series, list)) else pd.Series(out[1])
        if len(out) >= 3:
            cols[f"{name}_hist"] = pd.Series(out[2], index=close.index if close is not None else None) if isinstance(out[2], (pd.Series, list)) else pd.Series(out[2])
        return pd.DataFrame(cols)

    # 2) pandas Series
    if isinstance(out, pd.Series):
        col_name = _name_for_scalar_indicator(name, params)
        ser = out
        ser.index = df.index if isinstance(df, pd.DataFrame) else ser.index
        return pd.DataFrame({col_name: ser})

    # 3) pandas DataFrame
    if isinstance(out, pd.DataFrame):
        # prefix columns with name if ambiguous
        return out

    # 4) numeric scalar (rare), broadcast to index
    if isinstance(out, (int, float)):
        col_name = _name_for_scalar_indicator(name, params)
        idx = df.index if isinstance(df, pd.DataFrame) else None
        return pd.DataFrame({col_name: pd.Series([out]*len(idx), index=idx)})

    # fallback: intentar convertir a Series
    try:
        ser = pd.Series(out)
        col_name = _name_for_scalar_indicator(name, params)
        return pd.DataFrame({col_name: ser})
    except Exception:
        raise RuntimeError(f"Tipo de retorno no soportado por indicador {name}: {type(out)}")


# Reemplazar la función calculate_indicator con una versión robusta si no está ya correctamente implementada
try:
    # solo reemplazar si la versión existente es corta o frágil; en cualquier caso adjuntamos la versión robusta
    def calculate_indicator_safe(name: str, df: pd.DataFrame, **params) -> pd.DataFrame:
        name = name.lower()
        if name not in _INDICATOR_FUNCS:
            raise ValueError(f"Indicador no soportado: {name}")
        func = _INDICATOR_FUNCS[name]
        # llamar y normalizar
        out_df = _call_indicator_func(func, name, df, **params)
        # asegurar que los índices coinciden con df
        if isinstance(df, pd.DataFrame) and not out_df.empty:
            out_df = out_df.reset_index(drop=True) if not out_df.index.equals(df.index) else out_df
        return out_df

    # override
    calculate_indicator = calculate_indicator_safe
except Exception:
    # si falla, no rompemos la importación
    pass

# --- Fin parche core/indicators.py ---
