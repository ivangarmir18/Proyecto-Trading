# indicators/fibonacci.py
import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 144, ratios=None) -> Dict[str, float]:
    """
    Compute fibonacci levels from the last `lookback` rows in df.
    df must have columns: high, low
    Returns dict {str(ratio): level}  (same behaviour as antes)
    """
    if ratios is None:
        ratios = DEFAULT_RATIOS
    recent = df.tail(lookback)
    if recent.empty:
        return {}
    high = float(recent["high"].max())
    low = float(recent["low"].min())
    span = high - low
    if span <= 0:
        # degenerate case: all prices equal
        return {str(r): high for r in ratios}
    levels = {str(r): float(low + span * r) for r in ratios}
    return levels


_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")  # encuentra número en una cadena


def _parse_ratio_key(k: str) -> Optional[float]:
    """
    Extrae y devuelve el número que aparece en la clave `k`.
    Acepta:
      - '0.236', '0.5'
      - 'm_0.236', 'h_0.5', 'm_1.0', etc.
      - 'level_0.236', etc.
    Si no encuentra número válido devuelve None.
    """
    if k is None:
        return None
    # tratar claves que ya sean números
    try:
        return float(k)
    except Exception:
        pass
    # buscar primer número en la cadena
    m = _num_re.search(str(k))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def nearest_level(price: float, levels: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (ratio, level_value) of the nearest level to price.
    Si `levels` está vacío o no tiene claves numéricas válidas -> (None, None)

    Nota: robusto frente a claves con prefijos ('m_0.236', 'h_0.5', ...)
    Devuelve el ratio como float (ej. 0.236) y el valor del nivel (precio).
    """
    if not levels:
        return None, None

    items = []
    for k, v in levels.items():
        parsed = _parse_ratio_key(k)
        if parsed is None:
            # ignorar claves que no contienen número reconocible
            continue
        # garantizar que v sea float
        try:
            lv = float(v)
        except Exception:
            continue
        items.append((parsed, lv))

    if not items:
        return None, None

    ratio, level = min(items, key=lambda kv: abs(price - kv[1]))
    return ratio, level


def proximity_score(price: float, level: float, atr: float) -> float:
    """
    Map distance to a 0..1 score using ATR as scale:
    score = max(0, 1 - (dist / (3 * ATR)))
    If ATR is zero or None, returns 0.5 as neutral.
    """
    if atr is None or atr == 0:
        return 0.5
    dist = abs(price - level)
    val = 1.0 - (dist / (3.0 * atr))
    return max(0.0, min(1.0, val))


def save_levels_cache(symbol: str, levels: Dict[str, float]):
    """
    Save the latest fibonacci levels for quick inspection in data/cache/{symbol}_fib.json
    (misma salida que antes, sin cambios)
    """
    out = CACHE_DIR / f"{symbol}_fib.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(levels, f, indent=2)
