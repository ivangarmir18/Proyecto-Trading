# core/backtest.py
"""
Backtester robusto y compatible con la versión original del proyecto.

Características principales
- Simulación bar-by-bar (ejecución sobre close por defecto, con slippage y fee)
- Soporta long y short (simplificado), sizing ('allin' o fraction)
- Stops / Take-profit intrabar (usando high/low de la vela)
- Registro detallado de trades y equity curve
- Métricas: final_value, returns, sharpe, max_drawdown, win_rate, avg_win/loss
- Hooks para IA:
    - on_trade(trade_dict) -> llamado cada vez que se registra un trade
    - trades_to_labels(trades) -> convierte trades a etiquetas (realized returns)
    - generate_training_examples_from_trades(df_candles, trades, feature_fn) -> X,y
- API estable: run_backtest(...) devuelve dict con keys documentadas.

Estructura de trade dict:
    {"ts": int, "action": "buy"|"sell"|"short"|"cover", "price": float, "qty": float, "pnl": float|None, "entry_ts": int|None}

Notas:
- Si necesitas matching exacto de comportamiento con un exchange (fills parciales, maker/taker distinct), dímelo y adapto.
"""
from typing import Callable, Dict, Any, List, Optional, Tuple
import math
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("core.backtest")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# -------------------------
# Utilities / metrics
# -------------------------
def compute_max_drawdown(equity: List[float]) -> float:
    ser = pd.Series(equity)
    if ser.empty:
        return 0.0
    roll_max = ser.cummax()
    dd = (roll_max - ser) / roll_max
    return float(dd.max())


def compute_sharpe_from_returns(returns: pd.Series, annual_factor: float = 252.0) -> Optional[float]:
    if returns.empty:
        return None
    mean = returns.mean()
    std = returns.std()
    if std == 0 or math.isnan(std):
        return None
    return float((mean / std) * math.sqrt(annual_factor))


def _record_trade(log: List[Dict[str, Any]], ts: int, action: str, price: float, qty: float, pnl: Optional[float] = None, entry_ts: Optional[int] = None):
    entry = {"ts": int(ts), "action": action, "price": float(price), "qty": float(qty)}
    if pnl is not None:
        entry["pnl"] = float(pnl)
    if entry_ts is not None:
        entry["entry_ts"] = int(entry_ts)
    log.append(entry)
    return entry


# -------------------------
# Core backtest
# -------------------------
def run_backtest(df: pd.DataFrame,
                 strategy_fn: Callable[[pd.Series, int, pd.DataFrame, dict], Optional[str]],
                 initial_capital: float = 1000.0,
                 fee: float = 0.0005,
                 slippage: float = 0.0,
                 risk_per_trade: float = 1.0,
                 allow_short: bool = False,
                 position_sizing: str = "allin",
                 stop_loss_pct: Optional[float] = None,
                 take_profit_pct: Optional[float] = None,
                 compute_metrics_periods_per_year: int = 252,
                 on_trade: Optional[Callable[[Dict[str, Any]], None]] = None,
                 **kwargs) -> Dict[str, Any]:
    """
    Ejecuta la simulación.

    - df: DataFrame con columnas mínimas ['ts','timestamp','open','high','low','close','volume'] (ts epoch seconds)
    - strategy_fn(row, idx, df, state) -> 'buy'|'sell'|'short'|'cover'|None
    - stop_loss_pct / take_profit_pct: valores positivos (ej. 0.02 para 2%). Se aplican intrabar usando high/low:
        * para una posición long: TP si high >= entry_price*(1+tp), SL si low <= entry_price*(1-sl)
        * para short: invertido
    - on_trade: callback recibirá trade dict cada vez que se haga un trade
    - devuelve dict con: initial_capital, final_value, returns, n_trades, win_rate, avg_win, avg_loss,
      sharpe, max_drawdown, trades, equity_curve
    """
    if df is None or df.empty:
        return {"error": "no_data"}

    # copy & normalize
    df = df.copy().sort_values("ts").reset_index(drop=True)
    required_cols = {"ts", "close", "high", "low"}
    if not required_cols.issubset(set(df.columns)):
        logger.warning("DataFrame missing some required columns: required=%s, got=%s", required_cols, set(df.columns))
    # initialize state
    cash = float(initial_capital)
    position_qty = 0.0
    position_entry_price = None
    position_side: Optional[str] = None  # 'long' or 'short'
    entry_ts_of_position: Optional[int] = None

    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []

    engine_state: Dict[str, Any] = {}  # state passed to strategy_fn

    for idx, row in df.iterrows():
        ts = int(row["ts"])
        price = float(row["close"])
        high = float(row.get("high", price))
        low = float(row.get("low", price))

        # 1) first check if an intrabar stop/tp would trigger (only if we have a position)
        closed_by_stop_or_tp = False
        if position_qty > 0 and position_entry_price is not None and position_side == "long":
            # check take profit
            if take_profit_pct is not None:
                tp_price = position_entry_price * (1.0 + take_profit_pct)
                if high >= tp_price:
                    exec_price = tp_price * (1.0 - slippage)  # sell at tp adjusted by slippage
                    proceeds = position_qty * exec_price * (1 - fee)
                    pnl = proceeds - (position_qty * position_entry_price)
                    _record_trade(trades, ts, "sell", exec_price, position_qty, pnl, entry_ts_of_position)
                    if on_trade:
                        on_trade({"ts": ts, "action": "sell", "price": exec_price, "qty": position_qty, "pnl": pnl, "entry_ts": entry_ts_of_position})
                    cash += proceeds
                    position_qty = 0.0
                    position_entry_price = None
                    position_side = None
                    entry_ts_of_position = None
                    closed_by_stop_or_tp = True
            # check stop loss (if not already closed by tp)
            if (not closed_by_stop_or_tp) and (stop_loss_pct is not None):
                sl_price = position_entry_price * (1.0 - stop_loss_pct)
                if low <= sl_price:
                    exec_price = sl_price * (1.0 + slippage)
                    proceeds = position_qty * exec_price * (1 - fee)
                    pnl = proceeds - (position_qty * position_entry_price)
                    _record_trade(trades, ts, "sell", exec_price, position_qty, pnl, entry_ts_of_position)
                    if on_trade:
                        on_trade({"ts": ts, "action": "sell", "price": exec_price, "qty": position_qty, "pnl": pnl, "entry_ts": entry_ts_of_position})
                    cash += proceeds
                    position_qty = 0.0
                    position_entry_price = None
                    position_side = None
                    entry_ts_of_position = None
                    closed_by_stop_or_tp = True

        if position_qty > 0 and position_side == "short" and position_entry_price is not None:
            # for shorts: TP if low <= entry*(1-tp), SL if high >= entry*(1+sl)
            if take_profit_pct is not None:
                tp_price = position_entry_price * (1.0 - take_profit_pct)
                if low <= tp_price:
                    exec_price = tp_price * (1.0 + slippage)
                    pnl = (position_entry_price - exec_price) * position_qty - (position_entry_price * position_qty * fee)
                    # simplificado: add pnl and free collateral
                    cash += (position_entry_price * position_qty) + pnl
                    _record_trade(trades, ts, "cover", exec_price, position_qty, pnl, entry_ts_of_position)
                    if on_trade:
                        on_trade({"ts": ts, "action": "cover", "price": exec_price, "qty": position_qty, "pnl": pnl, "entry_ts": entry_ts_of_position})
                    position_qty = 0.0
                    position_entry_price = None
                    position_side = None
                    entry_ts_of_position = None
                    closed_by_stop_or_tp = True
            if (not closed_by_stop_or_tp) and (stop_loss_pct is not None):
                sl_price = position_entry_price * (1.0 + stop_loss_pct)
                if high >= sl_price:
                    exec_price = sl_price * (1.0 - slippage)
                    pnl = (position_entry_price - exec_price) * position_qty - (position_entry_price * position_qty * fee)
                    cash += (position_entry_price * position_qty) + pnl
                    _record_trade(trades, ts, "cover", exec_price, position_qty, pnl, entry_ts_of_position)
                    if on_trade:
                        on_trade({"ts": ts, "action": "cover", "price": exec_price, "qty": position_qty, "pnl": pnl, "entry_ts": entry_ts_of_position})
                    position_qty = 0.0
                    position_entry_price = None
                    position_side = None
                    entry_ts_of_position = None
                    closed_by_stop_or_tp = True

        # 2) if a stop/tp closed the position, continue to next bar
        if closed_by_stop_or_tp:
            # record equity for this bar and continue
            if position_qty > 0:
                mark = position_qty * price if position_side == "long" else cash + (position_entry_price - price) * position_qty
            else:
                mark = cash
            equity_curve.append({"ts": ts, "equity": float(mark)})
            continue

        # 3) otherwise, call strategy for a signal on this bar
        action = strategy_fn(row, idx, df, engine_state)

        # buy logic
        if action == "buy":
            # if currently short, first cover
            if position_qty > 0 and position_side == "short":
                # close short
                exec_price = price * (1.0 - slippage)
                pnl = (position_entry_price - exec_price) * position_qty - (position_entry_price * position_qty * fee)
                cash += (position_entry_price * position_qty) + pnl
                _record_trade(trades, ts, "cover", exec_price, position_qty, pnl, entry_ts_of_position)
                if on_trade:
                    on_trade({"ts": ts, "action": "cover", "price": exec_price, "qty": position_qty, "pnl": pnl, "entry_ts": entry_ts_of_position})
                position_qty = 0.0
                position_entry_price = None
                position_side = None
                entry_ts_of_position = None

            # then open long (if not already long)
            if position_side != "long":
                if position_sizing == "allin":
                    spend = cash * float(risk_per_trade)
                else:
                    spend = cash * float(risk_per_trade)
                if spend > 0:
                    exec_price = price * (1.0 + slippage)
                    qty = (spend * (1 - fee)) / exec_price
                    if qty > 0:
                        position_qty = qty
                        position_entry_price = exec_price
                        position_side = "long"
                        entry_ts_of_position = ts
                        cash -= (qty * exec_price) + (qty * exec_price * fee)
                        t = _record_trade(trades, ts, "buy", exec_price, qty, None, entry_ts_of_position)
                        if on_trade:
                            on_trade({**t, "entry_ts": entry_ts_of_position})

        # sell logic
        elif action == "sell":
            # if have a long -> close it
            if position_qty > 0 and position_side == "long":
                exec_price = price * (1.0 - slippage)
                proceeds = position_qty * exec_price * (1 - fee)
                pnl = proceeds - (position_qty * position_entry_price)
                _record_trade(trades, ts, "sell", exec_price, position_qty, pnl, entry_ts_of_position)
                if on_trade:
                    on_trade({"ts": ts, "action": "sell", "price": exec_price, "qty": position_qty, "pnl": pnl, "entry_ts": entry_ts_of_position})
                cash += proceeds
                position_qty = 0.0
                position_entry_price = None
                position_side = None
                entry_ts_of_position = None
            else:
                # optionally open short if allowed
                if allow_short and position_side != "short":
                    spend_collateral = cash * float(risk_per_trade)
                    if spend_collateral > 0:
                        exec_price = price * (1.0 + slippage)
                        qty = (spend_collateral * (1 - fee)) / exec_price
                        if qty > 0:
                            position_qty = qty
                            position_entry_price = exec_price
                            position_side = "short"
                            entry_ts_of_position = ts
                            cash -= (qty * exec_price * fee)  # simplified collateral handling
                            t = _record_trade(trades, ts, "short", exec_price, qty, None, entry_ts_of_position)
                            if on_trade:
                                on_trade({**t, "entry_ts": entry_ts_of_position})

        # other actions (None, hold) -> nothing to do

        # record equity at end of bar
        if position_qty > 0:
            if position_side == "long":
                mark = position_qty * price
            elif position_side == "short":
                mark = cash + (position_entry_price - price) * position_qty
            else:
                mark = cash
        else:
            mark = cash
        equity_curve.append({"ts": ts, "equity": float(mark)})

    # after loop, close any open position at last price (mark-to-market already used; but we won't force close)
    equity_series = pd.Series([e["equity"] for e in equity_curve]) if equity_curve else pd.Series([])
    if equity_series.empty:
        return {"error": "no_equity_curve"}
    final_value = float(equity_series.iloc[-1])
    returns = (final_value - initial_capital) / initial_capital
    per_returns = equity_series.pct_change().dropna()
    sharpe = compute_sharpe_from_returns(per_returns, annual_factor=compute_metrics_periods_per_year)
    max_dd = compute_max_drawdown(list(equity_series))
    pnl_list = [t.get("pnl") for t in trades if t.get("pnl") is not None]
    wins = [p for p in pnl_list if p > 0] if pnl_list else []
    losses = [p for p in pnl_list if p <= 0] if pnl_list else []
    win_rate = (len(wins) / len(pnl_list)) if pnl_list else 0.0
    avg_win = float(pd.Series(wins).mean()) if wins else 0.0
    avg_loss = float(pd.Series(losses).mean()) if losses else 0.0

    result = {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "returns": returns,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": trades,
        "equity_curve": equity_curve
    }
    return result


# -------------------------
# Helpers for IA
# -------------------------
def trades_to_labels(trades: List[Dict[str, Any]], only_entries: bool = True) -> Dict[int, float]:
    """
    Convierte una lista de trades (FIFO) a un mapping entry_ts -> realized_return.

    - only_entries: si True devuelve solo entradas tipo 'buy' (o 'short' si invertimos).
    - Realized return = (exit_price - entry_price) / entry_price for longs.
    - Para shorts: (entry_price - exit_price) / entry_price.
    - Empareja buys con sells por orden FIFO; para short empareja 'short'->'cover'.
    """
    labels: Dict[int, float] = {}
    buy_stack: List[Dict[str, Any]] = []
    short_stack: List[Dict[str, Any]] = []
    for t in trades:
        act = t.get("action")
        if act == "buy":
            buy_stack.append(t)
        elif act == "sell":
            if buy_stack:
                entry = buy_stack.pop(0)
                entry_price = entry.get("price")
                exit_price = t.get("price")
                qty = min(entry.get("qty", 0), t.get("qty", 0))
                if entry_price and qty:
                    realized = (exit_price - entry_price) / entry_price
                    labels[int(entry.get("ts"))] = float(realized)
        elif act == "short":
            short_stack.append(t)
        elif act in ("cover",):
            if short_stack:
                entry = short_stack.pop(0)
                entry_price = entry.get("price")
                exit_price = t.get("price")
                qty = min(entry.get("qty", 0), t.get("qty", 0))
                if entry_price and qty:
                    realized = (entry_price - exit_price) / entry_price
                    labels[int(entry.get("ts"))] = float(realized)
    if only_entries:
        return labels
    # else include also exits keyed by exit ts (not typical)
    return labels


def generate_training_examples_from_trades(df_candles: pd.DataFrame,
                                           trades: List[Dict[str, Any]],
                                           feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
                                           lookback_for_features: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Dado df_candles y trades, devuelve X (features) y y (targets) para entrenamiento.
    - feature_fn debe aceptar el df_candles completo y devolver features alineadas por índice (igual length).
    - Por convención, para una trade de entry en ts T, buscamos la fila en df_candles con ts==T y tomamos la fila features.loc[T_index].
    - Si no coincide ts exacto, hacemos búsqueda por igualdad entera.
    """
    if df_candles is None or df_candles.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    features = feature_fn(df_candles)
    labels_map = trades_to_labels(trades, only_entries=True)
    # map ts -> index
    ts_to_idx = {int(ts): idx for idx, ts in enumerate(df_candles["ts"].astype(int).tolist())}
    rows = []
    targets = []
    for entry_ts, realized in labels_map.items():
        idx = ts_to_idx.get(int(entry_ts))
        if idx is None:
            # try approximate match (nearest)
            diffs = (df_candles["ts"].astype(int) - int(entry_ts)).abs()
            if diffs.min() <= 1:
                idx = int(diffs.idxmin())
            else:
                continue
        row_feat = features.iloc[idx]
        rows.append(row_feat)
        targets.append(float(realized))
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)
    X = pd.DataFrame(rows).reset_index(drop=True)
    y = pd.Series(targets).reset_index(drop=True)
    return X, y


# -------------------------
# Quick unit-test helpers (used while debugging)
# -------------------------
def _sanity_check_basic():
    """Ejecuta un test rápido local para comprobar el comportamiento."""
    df = pd.DataFrame([
        {"ts": 1, "timestamp": "t1", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
        {"ts": 2, "timestamp": "t2", "open": 110, "high": 110, "low": 110, "close": 110, "volume": 1},
        {"ts": 3, "timestamp": "t3", "open": 120, "high": 120, "low": 120, "close": 120, "volume": 1},
    ])
    def strat(row, idx, df, state):
        if idx == 0: return "buy"
        if idx == 2: return "sell"
        return None
    res = run_backtest(df, strat, initial_capital=1000.0, fee=0.001, slippage=0.0)
    print("sanity:", res)
    return res


if __name__ == "__main__":
    # quick local check when run as script
    _sanity_check_basic()
