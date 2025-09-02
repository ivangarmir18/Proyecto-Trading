# tools/backtest.py
"""
Backtester sencillo para evaluar señales (scores) con stop & target.
- Lee candles & scores desde core.storage o sqlite (--db)
- Abre posición LONG cuando score >= threshold (en siguiente vela open)
- Size fijo 1 unidad (o porcentaje configurable)
- Evalúa hasta horizon candles para ver si target/stop alcanzado
- Devuelve DataFrame con operaciones y métricas agregadas
"""
from __future__ import annotations
import argparse
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# intentar storage
try:
    from core import storage as storage_mod
except Exception:
    try:
        import core.storage_postgres as storage_mod
    except Exception:
        storage_mod = None

def load_series(asset: str, interval: str, db_path: str = None):
    if storage_mod is not None:
        candles = storage_mod.load_candles(asset, interval)
        scores = storage_mod.load_scores(asset, interval)
        return candles, scores
    # fallback sqlite
    if not db_path:
        raise RuntimeError("No storage y no db_path proporcionado")
    import sqlite3
    con = sqlite3.connect(db_path)
    candles = pd.read_sql_query("SELECT ts, open, high, low, close FROM candles WHERE asset=? AND interval=? ORDER BY ts",
                                con, params=(asset, interval))
    scores = pd.read_sql_query("SELECT ts, score, range_min, range_max, stop, target FROM scores WHERE asset=? AND interval=? ORDER BY ts",
                               con, params=(asset, interval))
    con.close()
    return candles, scores

def backtest_asset(asset: str, interval: str, threshold: float = 0.6, horizon: int = 24, db_path: str = None) -> pd.DataFrame:
    candles, scores = load_series(asset, interval, db_path=db_path)
    if candles is None or candles.empty or scores is None or scores.empty:
        raise RuntimeError("No hay datos para backtest")

    # merge by ts (scores ts are unix seconds)
    dfc = candles.sort_values('ts').reset_index(drop=True)
    dfs = scores.sort_values('ts').reset_index(drop=True)
    dfc['ts_dt'] = pd.to_datetime(dfc['ts'], unit='s')
    dfs['ts_dt'] = pd.to_datetime(dfs['ts'], unit='s')
    merged = pd.merge_asof(dfc, dfs, left_on='ts_dt', right_on='ts_dt', direction='backward', tolerance=pd.Timedelta('1H'))
    merged = merged.sort_values('ts').reset_index(drop=True)

    trades = []
    i = 0
    while i < len(merged)-1:
        row = merged.loc[i]
        score = row.get('score', 0.0) if 'score' in row else 0.0
        if pd.isna(score) or score < threshold:
            i += 1
            continue
        # open at next candle open (if exists)
        entry_idx = i + 1
        if entry_idx >= len(merged):
            break
        entry_open = float(merged.loc[entry_idx, 'open'])
        stop = row.get('stop')
        target = row.get('target')
        if pd.isna(stop) or pd.isna(target):
            i += 1
            continue
        stop = float(stop)
        target = float(target)
        exit_price = None
        exit_idx = None
        exit_type = None  # 'target', 'stop', 'timeout', 'reverse'
        for look in range(entry_idx, min(entry_idx + horizon, len(merged))):
            high = float(merged.loc[look, 'high'])
            low = float(merged.loc[look, 'low'])
            if high >= target and low > stop:
                exit_price = target
                exit_idx = look
                exit_type = 'target'
                break
            if low <= stop and high < target:
                exit_price = stop
                exit_idx = look
                exit_type = 'stop'
                break
            if high >= target and low <= stop:
                # both in same candle -> choose by proximity to open
                open_p = float(merged.loc[look, 'open'])
                if abs(target - open_p) <= abs(open_p - stop):
                    exit_price = target
                    exit_type = 'target'
                else:
                    exit_price = stop
                    exit_type = 'stop'
                exit_idx = look
                break
        if exit_price is None:
            # timeout: close at last candle close in horizon
            last_idx = min(entry_idx + horizon - 1, len(merged)-1)
            exit_price = float(merged.loc[last_idx, 'close'])
            exit_idx = last_idx
            exit_type = 'timeout'
        pnl = exit_price - entry_open  # assuming 1 unit long
        pnl_pct = pnl / entry_open
        trades.append({
            "asset": asset,
            "entry_ts": int(merged.loc[entry_idx, 'ts']),
            "entry_open": entry_open,
            "exit_ts": int(merged.loc[exit_idx, 'ts']),
            "exit_price": exit_price,
            "exit_type": exit_type,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "score": float(score)
        })
        # continue after exit_idx to avoid overlapping trades
        i = exit_idx + 1 if exit_idx is not None else i + 1

    trades_df = pd.DataFrame(trades)
    return trades_df

def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {}
    total = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(wins) / total if total else 0.0
    net_pnl = trades_df['pnl'].sum()
    net_pct = trades_df['pnl_pct'].sum()
    avg_rr = (wins['pnl'].abs().mean() / (losses['pnl'].abs().mean() + 1e-9)) if not losses.empty else float('inf')
    # simple drawdown: compute equity curve
    equity = trades_df['pnl'].cumsum()
    roll_max = equity.cummax()
    drawdown = (equity - roll_max).min() if not equity.empty else 0.0
    return {
        "n_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "net_pnl": float(net_pnl),
        "net_pct": float(net_pct),
        "avg_rr": float(avg_rr if np.isfinite(avg_rr) else 0.0),
        "max_drawdown": float(drawdown)
    }

def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--asset', required=True)
    p.add_argument('--interval', default='1h')
    p.add_argument('--threshold', type=float, default=0.6)
    p.add_argument('--horizon', type=int, default=24)
    p.add_argument('--db', default=None, help='Sqlite path fallback if no storage')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    trades = backtest_asset(args.asset, args.interval, threshold=args.threshold, horizon=args.horizon, db_path=args.db)
    metrics = compute_metrics(trades)
    print("Metrics:", metrics)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(args.out, index=False)
        print("Trades saved to:", args.out)
    else:
        print(trades.head())

if __name__ == "__main__":
    cli()
