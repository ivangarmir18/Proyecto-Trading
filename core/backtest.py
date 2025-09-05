# core/backtest.py
"""
Backtest engine sencillo pero completo.

Función principal exportada:
    run_backtest_for(storage, asset, interval, start_ts=None, end_ts=None,
                     initial_capital=10000.0, fee=0.0005, slippage=0.0005,
                     fraction_of_capital=1.0, score_threshold=0.0, hold_period=1,
                     use_scores=True)

Retorna un dict con:
    {
      "asset": asset,
      "interval": interval,
      "start_ts": start_ts,
      "end_ts": end_ts,
      "initial_capital": ...,
      "final_capital": ...,
      "total_return": ...,
      "cagr": ...,
      "annual_volatility": ...,
      "sharpe": ...,
      "max_drawdown": ...,
      "max_drawdown_start": ...,
      "max_drawdown_end": ...,
      "trades": [ {trade dicts} ],
      "equity_curve": [ { "ts": ..., "equity": ... } ],
      "per_bar": pandas.DataFrame (index ts_dt) with columns: close, position, pnl, equity, returns
    }

Notas de diseño:
- El backtest usa precios de cierre para entradas/salidas (entrada al close de la barra cuando señal aparece).
- Si hay scores (storage.load_scores), alinea score por timestamp <= candle.ts y genera señales: long si score['pred'] > score_threshold.
- hold_period define cuántas barras mantener (1 = próxima barra exit allowed).
- fraction_of_capital define fracción del capital usada por trade (1.0 = todo el capital).
- fee y slippage aplicados en cada operación como costos relativos.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _bars_per_year_from_interval(interval: str) -> float:
    """
    Estimar número de barras por año según interval:
    - minutos/hours -> 365 * 24 * 60 / minutes
    - '1d' -> 252
    - '1w' -> 52
    - '1M' -> 12
    Fallback: 365
    """
    s = interval.strip().lower()
    try:
        if s.endswith("m"):
            mins = int(s[:-1]) if s[:-1].isdigit() else 1
            return 365.0 * 24.0 * 60.0 / max(1, mins)
        if s.endswith("h"):
            hours = int(s[:-1]) if s[:-1].isdigit() else 1
            return 365.0 * 24.0 / max(1, hours)
        if s.endswith("d"):
            # use trading days approximation
            return 252.0 / max(1, int(s[:-1]) if s[:-1].isdigit() else 1)
        if s.endswith("w"):
            return 52.0 / max(1, int(s[:-1]) if s[:-1].isdigit() else 1)
        if s.endswith("m"):  # 'M' months
            return 12.0
    except Exception:
        pass
    # fallback
    return 365.0


def _compute_drawdown(equity_series: pd.Series) -> Dict[str, Any]:
    """
    Recibe equity_series index datetime, returns max_dd value (positive fraction), start/end timestamps (ints ms),
    and drawdown series.
    """
    # equity_series must be numeric
    eq = equity_series.fillna(method="ffill").fillna(0.0)
    running_max = eq.cummax()
    drawdown = (running_max - eq) / running_max
    # handle case where running_max==0 -> drawdown NaN -> set 0
    drawdown = drawdown.fillna(0.0)
    if drawdown.empty:
        return {"max_drawdown": 0.0, "start": None, "end": None, "dd_series": drawdown}
    max_dd = float(drawdown.max())
    if max_dd <= 0:
        return {"max_drawdown": 0.0, "start": None, "end": None, "dd_series": drawdown}
    # find end index (first occurrence of max)
    end_idx = drawdown.idxmax()
    # find previous peak (running_max before end where running_max equals peak)
    peak_val = running_max.loc[: end_idx].max()
    # start is last index where running_max == peak_val before or at end_idx
    peaks = running_max.loc[: end_idx]
    # locate the last timestamp where peaks == peak_val
    starts = peaks[peaks == peak_val]
    start_idx = starts.index[-1] if not starts.empty else None
    # convert timestamps to ms
    start_ts = int(start_idx.value // 10 ** 6) if start_idx is not None else None
    end_ts = int(end_idx.value // 10 ** 6) if end_idx is not None else None
    return {"max_drawdown": max_dd, "start": start_ts, "end": end_ts, "dd_series": drawdown}


def _annualize_return(total_return: float, years: float) -> float:
    if years <= 0:
        return 0.0
    # (1 + total_return) ** (1/years) - 1
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def run_backtest_for(
    storage: Any,
    asset: str,
    interval: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    initial_capital: float = 10000.0,
    fee: float = 0.0005,
    slippage: float = 0.0005,
    fraction_of_capital: float = 1.0,
    score_threshold: float = 0.0,
    hold_period: int = 1,
    use_scores: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta backtest para asset/interval sobre datos en storage.
    - storage debe exponer load_candles(asset, interval, start_ts=None, end_ts=None, limit=None)
      y opcionalmente load_scores(asset, interval, start_ts=None, end_ts=None, limit=None)
      (en este proyecto PostgresStorage implementa esas funciones).
    - use_scores: si True, intenta usar storage.load_scores para generar señales, si no hay scores,
      cae al buy_and_hold.
    """
    if storage is None:
        raise ValueError("storage requerido para run_backtest_for")

    # 1) cargar velas
    try:
        # preferir cargar con rango si se proporciona
        df = storage.load_candles(asset, interval, start_ts=start_ts, end_ts=end_ts, limit=None)
    except Exception as e:
        logger.exception("storage.load_candles falló en backtest: %s", e)
        raise RuntimeError("No se pudieron cargar velas para backtest") from e

    if df is None or df.empty:
        raise RuntimeError("No hay velas para el backtest")

    # normalizar df: ts int -> datetime index
    df2 = df.copy()
    if "ts" not in df2.columns:
        raise RuntimeError("Las velas deben contener columna 'ts' (ms)")
    df2["ts"] = df2["ts"].astype("int64")
    df2["ts_dt"] = pd.to_datetime(df2["ts"], unit="ms", utc=True)
    df2 = df2.set_index("ts_dt", drop=False)
    df2 = df2.sort_index()
    close = pd.to_numeric(df2["close"], errors="coerce")

    # 2) load scores if requested and available
    signals = None
    if use_scores and hasattr(storage, "load_scores"):
        try:
            # load all scores in range to align
            scores_df = storage.load_scores(asset=asset, interval=interval, start_ts=start_ts, end_ts=end_ts, limit=None)
            if scores_df is not None and not scores_df.empty:
                # convert ts to datetime index
                scores_df = scores_df.copy()
                scores_df["ts"] = scores_df["ts"].astype("int64")
                scores_df["ts_dt"] = pd.to_datetime(scores_df["ts"], unit="ms", utc=True)
                scores_df = scores_df.set_index("ts_dt", drop=False).sort_index()
                # assume score JSON contains key 'pred' numeric (fallback to first numeric found)
                def extract_pred(row):
                    s = row.get("score") if isinstance(row, dict) else row
                    if isinstance(s, dict):
                        if "pred" in s:
                            return float(s["pred"])
                        # try first numeric value in dict
                        for v in s.values():
                            try:
                                return float(v)
                            except Exception:
                                continue
                    try:
                        return float(s)
                    except Exception:
                        return np.nan
                scores_df["_pred"] = scores_df["score"].apply(extract_pred)
                # align to candle index by forward-fill last known score <= candle.ts
                # create a series of preds indexed by ts_dt
                pred_series = scores_df["_pred"]
                # reindex to candles using asof (last valid index <= ts)
                preds_aligned = pd.Series(index=df2.index, dtype=float)
                # use pandas.merge_asof approach
                joined = pd.merge_asof(df2.reset_index().rename(columns={"ts_dt":"ts_dt_c"}).sort_values("ts_dt"),
                                       scores_df.reset_index().rename(columns={"ts_dt":"ts_dt_s"})[["ts_dt_s","_pred"]].sort_values("ts_dt_s"),
                                       left_on="ts_dt", right_on="ts_dt_s", direction="backward")
                preds_aligned = pd.Series(joined["_pred"].values, index=df2.index)
                # fill nans with 0
                preds_aligned = preds_aligned.fillna(0.0)
                signals = preds_aligned
        except Exception:
            logger.exception("Error cargando/alineando scores. Se usará buy-and-hold fallback.")
            signals = None

    # 3) generate trading signals
    # We'll produce 'signal' series: 1 -> long, 0 -> flat
    if signals is not None:
        signal = (signals > score_threshold).astype(int)
        # apply hold_period: when signal becomes 1, keep it for hold_period bars (simple implementation)
        if hold_period and hold_period > 1:
            sig = signal.copy()
            out = sig.copy() * 0
            n = len(sig)
            sig_values = sig.values
            for i in range(n):
                if sig_values[i] == 1:
                    # keep next hold_period bars including current
                    end = min(n, i + hold_period)
                    out.iloc[i:end] = 1
            signal = out
    else:
        # buy-and-hold: enter long at first available bar and hold to end
        signal = pd.Series(0, index=df2.index)
        signal.iloc[0] = 1
        signal = signal.cumsum().clip(upper=1)  # ensure 1 for all forward bars

    # 4) simulate PnL using close prices, simple discrete execution at close, position 0/1
    equity = []
    trades: List[Dict[str, Any]] = []
    cash = float(initial_capital)
    position = 0  # number of units (not quantity of assets) - we use fraction_of_capital to buy
    position_qty = 0.0
    entry_price = None
    last_equity = cash
    bar_index = 0

    per_bar_rows = []

    for ts, row in df2.iterrows():
        price = float(row["close"])
        desired = int(signal.iloc[bar_index])  # 0 or 1
        timestamp_ms = int(row["ts"])

        # if currently flat and desired=1 -> enter
        if position == 0 and desired == 1:
            # capital to allocate
            alloc = cash * float(fraction_of_capital)
            # compute quantity after fees/slippage: buy at price * (1 + slippage)
            buy_price = price * (1.0 + float(slippage))
            qty = alloc / buy_price if buy_price > 0 else 0.0
            # apply fee on trade value
            trade_cost = alloc * float(fee)
            cash = cash - alloc - trade_cost  # remaining cash
            position = 1
            position_qty = qty
            entry_price = buy_price
            trades.append({
                "type": "buy",
                "ts": timestamp_ms,
                "price": buy_price,
                "qty": qty,
                "cash_after": cash,
                "fee": trade_cost,
                "slippage": slippage
            })
        # if currently long and desired==0 -> exit
        elif position == 1 and desired == 0:
            # sell at price * (1 - slippage)
            sell_price = price * (1.0 - float(slippage))
            proceeds = position_qty * sell_price
            trade_cost = proceeds * float(fee)
            cash = cash + proceeds - trade_cost
            pnl = (sell_price - entry_price) * position_qty - trade_cost
            trades.append({
                "type": "sell",
                "ts": timestamp_ms,
                "price": sell_price,
                "qty": position_qty,
                "cash_after": cash,
                "fee": trade_cost,
                "slippage": slippage,
                "pnl": pnl
            })
            # reset position
            position = 0
            position_qty = 0.0
            entry_price = None

        # mark-to-market equity: cash + position_qty * price (mid price)
        mtm = cash + position_qty * price
        last_equity = float(mtm)
        equity.append({"ts": timestamp_ms, "equity": float(last_equity)})
        per_bar_rows.append({
            "ts": timestamp_ms,
            "close": price,
            "position": int(position),
            "qty": float(position_qty),
            "cash": float(cash),
            "equity": float(last_equity)
        })
        bar_index += 1

    # If still long at the end, close position at last close
    if position == 1 and position_qty > 0:
        final_price = float(df2["close"].iloc[-1]) * (1.0 - float(slippage))
        proceeds = position_qty * final_price
        trade_cost = proceeds * float(fee)
        cash = cash + proceeds - trade_cost
        pnl = (final_price - entry_price) * position_qty - trade_cost if entry_price is not None else 0.0
        timestamp_ms = int(df2["ts"].iloc[-1])
        trades.append({
            "type": "sell",
            "ts": timestamp_ms,
            "price": final_price,
            "qty": position_qty,
            "cash_after": cash,
            "fee": trade_cost,
            "slippage": slippage,
            "pnl": pnl
        })
        position = 0
        position_qty = 0.0
        entry_price = None
        last_equity = float(cash)
        equity.append({"ts": timestamp_ms, "equity": float(last_equity)})
        per_bar_rows.append({
            "ts": timestamp_ms,
            "close": float(df2["close"].iloc[-1]),
            "position": 0,
            "qty": 0.0,
            "cash": float(cash),
            "equity": float(last_equity)
        })

    # Build equity series
    equity_df = pd.DataFrame(equity)
    if equity_df.empty:
        raise RuntimeError("No se generó equity curve en backtest")
    equity_df = equity_df.drop_duplicates(subset=["ts"]).set_index(pd.to_datetime(equity_df["ts"], unit="ms", utc=True))
    equity_series = equity_df["equity"].astype(float)

    # Metrics
    start_equity = float(initial_capital)
    end_equity = float(equity_series.iloc[-1])
    total_return = (end_equity / start_equity) - 1.0 if start_equity != 0 else 0.0

    # period in years: compute from timestamps
    seconds = (equity_series.index[-1].to_datetime64() - equity_series.index[0].to_datetime64()) / np.timedelta64(1, "s")
    years = float(seconds) / (365.0 * 24.0 * 3600.0) if seconds > 0 else 0.0
    # fallback using bars_per_year
    bars_per_year = _bars_per_year_from_interval(interval)
    # per-bar returns for volatility: pct change of equity
    eq_returns = equity_series.pct_change().fillna(0.0)

    # annualized return
    if years > 0:
        cagr = _annualize_return(total_return, years)
    else:
        # fallback: use geometric mean with bars_per_year
        periods = len(eq_returns)
        if periods > 0:
            total_return_alt = (end_equity / start_equity) - 1.0
            years_alt = float(periods) / bars_per_year
            cagr = _annualize_return(total_return_alt, years_alt) if years_alt > 0 else 0.0
        else:
            cagr = 0.0

    # annualized vol
    try:
        ann_vol = float(eq_returns.std(ddof=0) * math.sqrt(bars_per_year))
    except Exception:
        ann_vol = float(0.0)

    # sharpe (risk-free = 0)
    sharpe = float((cagr) / ann_vol) if ann_vol > 0 else float("nan")

    # drawdown
    dd_info = _compute_drawdown(equity_series)
    max_dd = float(dd_info.get("max_drawdown", 0.0))
    dd_start = dd_info.get("start")
    dd_end = dd_info.get("end")

    # trades summary
    trades_summary = {
        "n_trades": len([t for t in trades if t.get("type") == "buy"]),
        "n_wins": len([t for t in trades if t.get("type") == "sell" and t.get("pnl", 0) > 0]),
        "n_losses": len([t for t in trades if t.get("type") == "sell" and t.get("pnl", 0) <= 0]),
        "net_pnl": float(end_equity - start_equity),
    }

    # per_bar DataFrame
    per_bar_df = pd.DataFrame(per_bar_rows)
    if not per_bar_df.empty:
        per_bar_df = per_bar_df.set_index(pd.to_datetime(per_bar_df["ts"], unit="ms", utc=True))
        per_bar_df["returns"] = per_bar_df["equity"].pct_change().fillna(0.0)

    result = {
        "asset": asset,
        "interval": interval,
        "start_ts": int(df2["ts"].iloc[0]),
        "end_ts": int(df2["ts"].iloc[-1]),
        "initial_capital": float(initial_capital),
        "final_capital": float(end_equity),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "max_drawdown": float(max_dd),
        "max_drawdown_start": dd_start,
        "max_drawdown_end": dd_end,
        "trades": trades,
        "trades_summary": trades_summary,
        "equity_curve": equity_df.reset_index().rename(columns={"index": "ts_dt"}).to_dict(orient="records"),
        "per_bar": per_bar_df,
    }

    return result


# Convenience alias (expected by Orchestrator)
def run_backtest_for_storage(*args, **kwargs):
    return run_backtest_for(*args, **kwargs)
