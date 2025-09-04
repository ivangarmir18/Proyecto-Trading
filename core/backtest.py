# core/backtest.py
"""
Backtesting profesional y reutilizable.

Diseño:
 - Input: candles DataFrame (ts, open, high, low, close, volume) ordenado asc por ts.
 - Signals: pandas.Series indexed like candles (same length) con valores in {-1,0,1} que representan posición objetivo.
            Alternativamente, puede pasarse una función strategy_fn(df) -> Series.
 - El backtester aplica cambios de posición sólo en la apertura de la vela siguiente (simula ejecución market @ open).
 - Parámetros: capital inicial, comisiones (pct), slippage (pct).
 - Output: dict con keys:
     - 'trades': list de trades {entry_ts, exit_ts, entry_price, exit_price, direction, pnl, return_pct}
     - 'metrics': dict con métricas (total_return, CAGR, max_drawdown, winrate, expectancy, num_trades)
     - 'equity': pandas.DataFrame with columns ts, equity, position
"""

from __future__ import annotations
from typing import Optional, Callable, List, Dict, Any, Union
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# -------------------- helpers -------------------- #
def _ensure_sorted_candles(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("candles_df is None")
    if 'ts' not in df.columns:
        raise ValueError("candles_df must contain 'ts' column (unix seconds)")
    df2 = df.sort_values('ts').reset_index(drop=True)
    return df2


def _signal_from_callable_or_series(signals, candles: pd.DataFrame) -> pd.Series:
    if callable(signals):
        s = signals(candles)
        if not isinstance(s, (pd.Series, pd.DataFrame)):
            raise ValueError("strategy_fn must return pandas.Series or DataFrame-like with single column")
        s = s.squeeze()
        s = s.reindex(candles.index).fillna(0).astype(int)
        return s
    if isinstance(signals, pd.Series):
        # align indices
        s = signals.reindex(candles.index).fillna(0).astype(int)
        return s
    # if None => neutral
    return pd.Series(0, index=candles.index)


def _annualize_return(total_return: float, days: float) -> float:
    if days <= 0:
        return float(total_return)
    try:
        years = max(days / 365.0, 1e-9)
        return (1 + total_return) ** (1.0 / years) - 1.0
    except Exception:
        return float(total_return)


def _max_drawdown_from_series(equity_vals: np.ndarray) -> float:
    # equity_vals is cumulative equity series
    peaks = np.maximum.accumulate(equity_vals)
    dd = (equity_vals - peaks) / peaks
    return float(min(dd))  # negative


# -------------------- built-in simple signal helpers -------------------- #
def signal_from_score_thresholds(scores_series: pd.Series, entry_threshold: float = 0.8, exit_threshold: float = 0.5) -> pd.Series:
    """
    genera una serie de señales 1/0 basada en un score:
     - entry when score >= entry_threshold -> position 1
     - exit (go flat) when score < exit_threshold -> position 0
    Mantiene posición hasta que condición de salida se cumple.
    """
    s = scores_series.fillna(0).astype(float).copy()
    pos = 0
    out = []
    for v in s.values:
        if pos == 0 and v >= entry_threshold:
            pos = 1
        elif pos == 1 and v < exit_threshold:
            pos = 0
        out.append(pos)
    return pd.Series(out, index=s.index)


# -------------------- core backtest runner -------------------- #
def run_backtest(
    candles_df: pd.DataFrame,
    signals: Union[pd.Series, Callable[[pd.DataFrame], pd.Series], None],
    capital: float = 10000.0,
    fee_pct: float = 0.0005,
    slippage: float = 0.0005,
    allow_short: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Ejecuta backtest.

    candles_df: must contain columns ['ts','open','high','low','close'] at least.
    signals: Series aligned to candles_df index with -1/0/1 or callable strategy(candles_df) -> Series
    returns dict with trades, equity DataFrame and metrics.
    """
    df = _ensure_sorted_candles(candles_df).reset_index(drop=True)
    idx = df.index
    # get signals aligned
    sig = _signal_from_callable_or_series(signals, df)
    # clamp for shorts
    if not allow_short:
        sig = sig.clip(lower=0)

    # We'll assume trades executed at next candle open (after signal appears)
    # Build position series: shift signals by 1 (changes take effect next bar open)
    pos = sig.shift(1).fillna(0).astype(int).copy()

    # keep only -1/0/1
    pos = pos.apply(lambda x: int(max(min(x, 1), -1)))

    initial_capital = float(capital)
    equity = float(capital)
    equity_series = []
    trades: List[Dict[str, Any]] = []
    current_trade = None  # dict storing entry info
    last_pos = 0

    for i in idx:
        row = df.loc[i]
        ts = int(row['ts'])
        open_price = float(row['open'])
        high = float(row['high'])
        low = float(row['low'])
        close = float(row['close'])

        position = int(pos.loc[i])  # position during this bar
        # position change?
        if last_pos == 0 and position != 0:
            # enter at open_price with commission+slippage
            entry_price = open_price * (1.0 + slippage if position > 0 else 1.0 - slippage)
            entry_comm = entry_price * fee_pct
            current_trade = {
                "entry_idx": int(i),
                "entry_ts": ts,
                "entry_price": entry_price,
                "direction": int(position),
                "entry_comm": entry_comm
            }
            if verbose: log.info("Entry %s @%s", current_trade['direction'], entry_price)
        elif last_pos != 0 and position == 0:
            # exit at open
            exit_price = open_price * (1.0 - slippage if last_pos > 0 else 1.0 + slippage)
            exit_comm = exit_price * fee_pct
            if current_trade is None:
                # defensive
                current_trade = {"entry_price": open_price, "entry_idx": None, "entry_ts": None, "direction": last_pos}
            entry_price = current_trade.get('entry_price')
            direction = current_trade.get('direction', last_pos)
            # PnL calculation
            if direction > 0:
                pnl = (exit_price - entry_price) - (current_trade.get('entry_comm', 0.0) + exit_comm)
            else:
                pnl = (entry_price - exit_price) - (current_trade.get('entry_comm', 0.0) + exit_comm)
            return_pct = pnl / (entry_price)  # approximate percent vs price
            trades.append({
                "entry_idx": current_trade.get("entry_idx"),
                "exit_idx": int(i),
                "entry_ts": current_trade.get("entry_ts"),
                "exit_ts": ts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": direction,
                "pnl": float(pnl),
                "return_pct": float(return_pct)
            })
            equity += pnl
            if verbose: log.info("Exit trade %s pnl=%.4f equity=%.2f", direction, pnl, equity)
            current_trade = None
        else:
            # no transaction this bar; mark unrealized PnL if you want (we compute equity series as cash + unrealized for position)
            pass

        # compute unrealized PnL for current open position at close price (mark-to-market)
        unreal = 0.0
        if current_trade:
            entry_price = current_trade['entry_price']
            direction = current_trade['direction']
            # mark-to-market using close
            if direction > 0:
                unreal = (close - entry_price)
            else:
                unreal = (entry_price - close)
            # subtract estimated fees on entry (already subtracted earlier) not on unreal PnL
        # equity snapshot = cash + unreal
        equity_snapshot = equity + unreal
        equity_series.append({"idx": int(i), "ts": ts, "equity": float(equity_snapshot), "position": int(position)})

        last_pos = position

    # if position still open at end, close at last close price
    if current_trade:
        last_row = df.iloc[-1]
        exit_price = float(last_row['close']) * (1.0 - slippage if current_trade['direction'] > 0 else 1.0 + slippage)
        exit_comm = exit_price * fee_pct
        entry_price = current_trade.get('entry_price')
        direction = current_trade.get('direction', 1)
        if direction > 0:
            pnl = (exit_price - entry_price) - (current_trade.get('entry_comm', 0.0) + exit_comm)
        else:
            pnl = (entry_price - exit_price) - (current_trade.get('entry_comm', 0.0) + exit_comm)
        return_pct = pnl / (entry_price)
        trades.append({
            "entry_idx": current_trade.get("entry_idx"),
            "exit_idx": int(df.index[-1]),
            "entry_ts": current_trade.get("entry_ts"),
            "exit_ts": int(last_row['ts']),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "pnl": float(pnl),
            "return_pct": float(return_pct)
        })
        equity += pnl
        equity_series[-1]['equity'] = float(equity)
        equity_series[-1]['position'] = 0

    eq_df = pd.DataFrame(equity_series)
    if eq_df.empty:
        # no trades executed; construct trivial equity series
        eq_df = pd.DataFrame([{"idx": int(i), "ts": int(df.loc[i,'ts']), "equity": initial_capital, "position": 0} for i in df.index])

    # metrics
    total_return = (eq_df['equity'].iloc[-1] / initial_capital) - 1.0
    days = (pd.to_datetime(int(df['ts'].iloc[-1]), unit='s') - pd.to_datetime(int(df['ts'].iloc[0]), unit='s')).days or 1
    cagr = _annualize_return(total_return, days)
    equity_vals = eq_df['equity'].values
    max_dd = _max_drawdown_from_series(equity_vals)

    num_trades = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    winrate = float(len(wins) / num_trades) if num_trades > 0 else 0.0
    avg_return = float(np.mean([t['return_pct'] for t in trades])) if trades else 0.0
    avg_win = float(np.mean([t['pnl'] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t['pnl'] for t in losses])) if losses else 0.0
    expectancy = ( (avg_win if avg_win else 0.0) * (len(wins)/num_trades if num_trades else 0) ) - ( (abs(avg_loss) if avg_loss else 0.0) * (len(losses)/num_trades if num_trades else 0) )

    metrics = {
        "initial_capital": float(initial_capital),
        "final_equity": float(eq_df['equity'].iloc[-1]),
        "total_return": float(total_return),
        "cagr_approx": float(cagr),
        "max_drawdown": float(max_dd),
        "num_trades": int(num_trades),
        "winrate": float(winrate),
        "avg_return_per_trade": float(avg_return),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy)
    }

    return {"trades": trades, "metrics": metrics, "equity": eq_df}


# -------------------- example usage -------------------- #
if __name__ == "__main__":
    # quick demo if executed directly (requires sample CSV with columns ts,open,high,low,close)
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="CSV de velas", required=True)
    parser.add_argument("--entry", type=float, default=0.9)
    parser.add_argument("--exit", type=float, default=0.6)
    args = parser.parse_args()
    df = pd.read_csv(args.file)
    # assume df has ts,open,high,low,close
    # make fake scores: normalized close
    scores = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
    sig = signal_from_score_thresholds(scores, entry_threshold=args.entry, exit_threshold=args.exit)
    res = run_backtest(df, sig, capital=10000)
    print("Metrics:", res['metrics'])
