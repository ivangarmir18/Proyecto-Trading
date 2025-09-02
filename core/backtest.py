"""
core/backtest.py

Backtester simple y autocontenido para el proyecto Watchlist.

- Usa core.storage.load_candles/load_scores/load_indicators si core.storage está disponible.
- Si no, hace fallback a SQLite usando --db.
- Entrada long cuando score >= threshold (entrada en la open del siguiente candle).
- Usa stop/target desde la fila de score; si faltan intentará inferirlos desde ATR en indicators.
- Cierra por stop/target (detecta intrabar por high/low) o por límite temporal (max_holding_candles).
- Exporta trades a CSV y muestra estadísticas básicas.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# try to use core.storage if available
try:
    from core import storage  # type: ignore
    _HAS_STORAGE = True
except Exception:
    storage = None
    _HAS_STORAGE = False


@dataclass
class Trade:
    asset: str
    interval: str
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    pnl: float
    return_pct: float
    reason: str


def load_data_from_sqlite(db_path: str, asset: str, interval: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leer candles, scores e indicadores (si existen) desde SQLite.
    Espera tablas: candles(asset, interval, ts, open, high, low, close, volume),
                   scores(asset, interval, ts, score, range_min, range_max, stop, target, multiplier),
                   indicators (joined by candle_id) opcional.
    """
    con = sqlite3.connect(db_path)
    q_candles = "SELECT ts, open, high, low, close, volume, asset, interval FROM candles WHERE asset=? AND interval=? ORDER BY ts"
    q_scores = "SELECT ts, score, range_min, range_max, stop, target, multiplier FROM scores WHERE asset=? AND interval=? ORDER BY ts"

    try:
        q_ind = ("SELECT c.ts as ts, i.ema9, i.ema40, i.atr, i.macd, i.macd_signal, i.rsi, i.support, i.resistance "
                 "FROM candles c JOIN indicators i ON c.id=i.candle_id WHERE c.asset=? AND c.interval=? ORDER BY c.ts")
        df_ind = pd.read_sql(q_ind, con, params=(asset, interval))
    except Exception:
        df_ind = pd.DataFrame()

    df_candles = pd.read_sql(q_candles, con, params=(asset, interval))
    df_scores = pd.read_sql(q_scores, con, params=(asset, interval))
    con.close()

    # normalize ts to integer seconds
    for df in (df_candles, df_scores, df_ind):
        if df is None or df.empty:
            continue
        if 'ts' in df.columns:
            if not np.issubdtype(df['ts'].dtype, np.number):
                try:
                    df['ts'] = pd.to_datetime(df['ts']).astype(int) // 10**9
                except Exception:
                    df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype('Int64')
    return df_candles, df_scores, df_ind


def align_data(candles: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Merge candles and scores using merge_asof so each candle has most recent prior score (if any).
    """
    if candles.empty:
        raise ValueError("No candles provided")
    dfc = candles.sort_values('ts').reset_index(drop=True).copy()
    dfs = scores.sort_values('ts').reset_index(drop=True).copy()

    dfc['ts_dt'] = pd.to_datetime(dfc['ts'], unit='s')
    if not dfs.empty:
        dfs['ts_dt'] = pd.to_datetime(dfs['ts'], unit='s')
        merged = pd.merge_asof(dfc, dfs, on='ts_dt', direction='backward', tolerance=pd.Timedelta('1H'))
    else:
        merged = dfc
        for col in ['score', 'range_min', 'range_max', 'stop', 'target', 'multiplier']:
            merged[col] = pd.NA

    merged = merged.drop(columns=['ts_dt'])
    return merged


def find_indicator_for_ts(indicators: pd.DataFrame, ts: int) -> Optional[dict]:
    if indicators is None or indicators.empty:
        return None
    row = indicators[indicators['ts'] == ts]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def asset_name_from_df(df: pd.DataFrame) -> str:
    if 'asset' in df.columns and not df['asset'].isna().all():
        v = df['asset'].dropna().unique()
        if len(v) > 0:
            return str(v[0])
    return 'UNKNOWN'


def simulate_trades(candles: pd.DataFrame, scores: pd.DataFrame, indicators: pd.DataFrame = pd.DataFrame(),
                    threshold: float = 0.6, max_holding_candles: int = 24, verbose: bool = False) -> Tuple[List[Trade], dict]:
    """
    Simulación principal:
      - Entrada: cuando score >= threshold (tomada en el siguiente candle.open)
      - Stop/Target: tomado desde la fila de score; si faltan se intenta inferir desde atr en 'indicators'
      - Salida: por stop/target (intrabar comprobando high/low) o por tiempo (max_holding_candles)
    """
    trades: List[Trade] = []
    df = align_data(candles, scores).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return trades, {'num_trades': 0}

    in_position = False
    entry_price = None
    entry_idx = None
    entry_ts = None
    entry_stop = None
    entry_target = None

    for idx in range(n - 1):  # we need next candle for entry price
        row = df.iloc[idx]
        score = float(row['score']) if 'score' in row and pd.notna(row['score']) else 0.0

        if not in_position and score >= threshold:
            next_idx = idx + 1
            if next_idx >= n:
                break
            next_candle = df.iloc[next_idx]
            entry_price = float(next_candle['open'])
            entry_idx = next_idx
            entry_ts = int(next_candle['ts'])

            # prefer stop/target from the score row
            stop = row.get('stop', None) if 'stop' in row else None
            target = row.get('target', None) if 'target' in row else None

            # fallback to indicators (using the score row ts)
            ind = find_indicator_for_ts(indicators, int(row['ts'])) if not indicators.empty else None
            if (stop is None or pd.isna(stop) or target is None or pd.isna(target)) and ind:
                atr = ind.get('atr', None)
                if atr and atr > 0:
                    stop = entry_price - 1.3 * float(atr)
                    target = entry_price + 1.3 * float(atr)

            if stop is None or target is None or pd.isna(stop) or pd.isna(target):
                if verbose:
                    print(f"Skipping entry at idx {idx} ts {row.get('ts')} - missing stop/target")
                in_position = False
                entry_price = None
                entry_idx = None
                entry_ts = None
                continue

            entry_stop = float(stop)
            entry_target = float(target)
            in_position = True
            if verbose:
                print(f"ENTER {asset_name_from_df(df)} idx={entry_idx} ts={entry_ts} price={entry_price:.6f} stop={entry_stop:.6f} target={entry_target:.6f}")
            continue

        if in_position:
            exit_idx = None
            exit_price = None
            exit_reason = None

            # scan subsequent candles up to horizon
            for look in range(entry_idx, min(entry_idx + max_holding_candles, n)):
                c = df.iloc[look]
                high = float(c['high'])
                low = float(c['low'])
                hit_target = high >= entry_target
                hit_stop = low <= entry_stop

                if hit_target and not hit_stop:
                    exit_idx = look
                    exit_price = entry_target
                    exit_reason = 'target'
                    break
                if hit_stop and not hit_target:
                    exit_idx = look
                    exit_price = entry_stop
                    exit_reason = 'stop'
                    break
                if hit_target and hit_stop:
                    # both in same candle: decide by proximity to open
                    open_price = float(c['open'])
                    dist_target = abs(entry_target - open_price)
                    dist_stop = abs(open_price - entry_stop)
                    if dist_target <= dist_stop:
                        exit_idx = look
                        exit_price = entry_target
                        exit_reason = 'target'
                    else:
                        exit_idx = look
                        exit_price = entry_stop
                        exit_reason = 'stop'
                    break

            if exit_idx is None:
                # time-based exit
                last_idx = min(entry_idx + max_holding_candles - 1, n - 1)
                c = df.iloc[last_idx]
                exit_idx = last_idx
                exit_price = float(c['close'])
                exit_reason = 'time'

            entry_p = entry_price
            exit_p = exit_price
            pnl = exit_p - entry_p
            return_pct = pnl / entry_p if entry_p != 0 else 0.0

            trade = Trade(
                asset=asset_name_from_df(df),
                interval=str(row.get('interval', 'unknown')),
                entry_ts=entry_ts,
                entry_price=entry_p,
                exit_ts=int(df.iloc[exit_idx]['ts']),
                exit_price=exit_p,
                pnl=pnl,
                return_pct=return_pct,
                reason=exit_reason
            )
            trades.append(trade)

            if verbose:
                print(f"CLOSE {trade.asset} entry={entry_p:.6f} exit={exit_p:.6f} reason={exit_reason} ret={return_pct*100:.2f}%")

            # reset position
            in_position = False
            entry_price = None
            entry_idx = None
            entry_ts = None
            entry_stop = None
            entry_target = None

    stats = compute_trade_stats(trades)
    return trades, stats


def compute_trade_stats(trades: List[Trade]) -> dict:
    if not trades:
        return {'num_trades': 0, 'total_return_pct': 0.0, 'winrate': 0.0, 'avg_return_pct': 0.0, 'max_drawdown_pct': 0.0}
    returns = np.array([t.return_pct for t in trades])
    equity = (1 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    total_return = float(equity[-1] - 1.0)
    num_trades = len(trades)
    wins = (returns > 0).sum()
    winrate = float(wins) / num_trades if num_trades > 0 else 0.0
    avg_return = float(np.mean(returns)) if num_trades > 0 else 0.0
    stats = {
        'num_trades': num_trades,
        'total_return_pct': total_return * 100.0,
        'winrate': winrate,
        'avg_return_pct': avg_return * 100.0,
        'max_drawdown_pct': max_dd * 100.0
    }
    return stats


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            'asset': t.asset,
            'interval': t.interval,
            'entry_ts': t.entry_ts,
            'entry_dt': datetime.utcfromtimestamp(t.entry_ts).isoformat() + 'Z',
            'entry_price': t.entry_price,
            'exit_ts': t.exit_ts,
            'exit_dt': datetime.utcfromtimestamp(t.exit_ts).isoformat() + 'Z',
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'return_pct': t.return_pct,
            'reason': t.reason
        })
    return pd.DataFrame(rows)


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument('--asset', required=True, help='Ticker, e.g. BTCUSDT or AAPL')
    p.add_argument('--interval', default='1h', help='Interval string stored in DB (5m,1h,4h,1d...)')
    p.add_argument('--db', default=None, help='Path to sqlite DB (fallback if core.storage not present)')
    p.add_argument('--threshold', type=float, default=0.6, help='Score threshold to enter')
    p.add_argument('--max_holding_candles', type=int, default=24, help='Max candles to hold a trade')
    p.add_argument('--out', default='backtest_trades.csv', help='CSV output path for trades')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    if _HAS_STORAGE:
        try:
            candles = storage.load_candles(args.asset, args.interval)
            scores = storage.load_scores(args.asset, args.interval)
            try:
                indicators = storage.load_indicators(args.asset, args.interval)
            except Exception:
                indicators = pd.DataFrame()
        except Exception as e:
            print("core.storage present but failed to load data:", e)
            if not args.db:
                raise
            candles, scores, indicators = load_data_from_sqlite(args.db, args.asset, args.interval)
    else:
        if not args.db:
            raise RuntimeError('core.storage not present; please provide --db sqlite path')
        candles, scores, indicators = load_data_from_sqlite(args.db, args.asset, args.interval)

    # basic checks
    if candles is None or candles.empty:
        raise RuntimeError('No candle data available for asset/interval')

    trades, stats = simulate_trades(candles, scores, indicators,
                                    threshold=args.threshold,
                                    max_holding_candles=args.max_holding_candles,
                                    verbose=args.verbose)

    df_trades = trades_to_dataframe(trades)
    df_trades.to_csv(args.out, index=False)

    print('\nBacktest summary:')
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"Trades saved to: {args.out}")


if __name__ == '__main__':
    main_cli()
