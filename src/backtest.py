"""
Simple-return buy-and-hold backtesting and performance comparison.

Used by both entry points: main.py applies it once over the held-out test
window, and src/walkforward.py applies it to each quarterly hold window before
stitching the results into one continuous series.

Two bugs from the previous version are fixed here.

1. UNITS. The old code fed LOG returns into ``returns @ weights`` and then
   compounded with ``(1 + r).cumprod()``. A weighted sum of log returns is not
   the log return of the weighted portfolio -- log is not linear, so that
   expression corresponds to no portfolio at all -- and compounding a log
   return as if it were a simple return is a second, opposite-signed error.
   Everything here is simple returns, end to end.

2. WEIGHT DRIFT. A buy-and-hold portfolio is not rebalanced daily. After
   inception the weights drift with performance: winners become a larger share
   of the book on their own. Taking a fixed weighted average of daily returns
   silently models a portfolio rebalanced back to target every single day, at
   no cost. Here the value path is tracked directly and the daily returns are
   derived from it.
"""

import numpy as np
import pandas as pd

from config import TRADING_DAYS


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series,
                      cost_bps: float = 0.0) -> pd.Series:
    """
    Daily simple returns of a buy-and-hold portfolio bought on day 0.

    Parameters
    ----------
    returns  : DataFrame (date x ticker) of SIMPLE daily returns.
    weights  : Series of target weights at inception, indexed by ticker.
    cost_bps : one-off entry cost in basis points, charged on gross exposure
               on day 0.

    The day-0 return is earned: the position is established at the previous
    close, so the base for the first day is ``weights.sum()`` rather than the
    day-0 portfolio value. Using the day-0 value as its own base would silently
    discard the first day's move.
    """
    if returns is None or len(returns) == 0:
        return pd.Series(dtype=float)

    weights = weights.reindex(returns.columns).fillna(0.0).astype(float)
    base = float(weights.sum())
    if not np.isfinite(base) or base == 0:
        raise ValueError("Portfolio weights sum to zero; nothing to backtest.")

    growth = (1 + returns.astype(float)).cumprod()      # per-asset value path
    value = growth @ weights.to_numpy()                 # portfolio value path
    prev = value.shift(1)
    prev.iloc[0] = base                                 # capital committed at t0
    port = value / prev - 1

    cost = (float(cost_bps) / 10_000.0) * float(weights.abs().sum())
    if cost:
        port.iloc[0] -= cost
    return port


def cumulative_returns(returns: pd.DataFrame, weights: pd.Series,
                       cost_bps: float = 0.0) -> pd.Series:
    """Growth of 1 unit of capital, buy-and-hold."""
    port = portfolio_returns(returns, weights, cost_bps=cost_bps)
    return (1 + port).cumprod()


def stats_from_returns(port_returns: pd.Series) -> dict:
    """CAGR, annualized volatility, Sharpe ratio (rf=0), and max drawdown."""
    port_returns = pd.Series(port_returns).dropna().astype(float)
    n_days = len(port_returns)
    if n_days == 0:
        return {"CAGR": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDrawdown": np.nan}

    cum = (1 + port_returns).cumprod()
    final = float(cum.iloc[-1])
    years = n_days / TRADING_DAYS
    if years <= 0:
        cagr = np.nan
    elif final <= 0:
        cagr = -1.0  # total loss of capital
    else:
        cagr = final ** (1 / years) - 1

    ann_vol = float(port_returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(port_returns.mean() * TRADING_DAYS / ann_vol) if ann_vol > 0 else np.nan

    # The starting value of 1.0 counts as a peak, so a loss on day one is a
    # drawdown rather than being measured from an already-depressed high.
    running_max = cum.cummax().clip(lower=1.0)
    max_dd = float(((cum - running_max) / running_max).min())

    return {"CAGR": cagr, "AnnVol": ann_vol, "Sharpe": sharpe, "MaxDrawdown": max_dd}


def performance_stats(returns: pd.DataFrame, weights: pd.Series,
                      cost_bps: float = 0.0) -> dict:
    """Convenience wrapper: build the return stream, then summarize it."""
    return stats_from_returns(portfolio_returns(returns, weights, cost_bps=cost_bps))


def compare_portfolios(streams: dict) -> pd.DataFrame:
    """
    Summarize several return streams side by side.

    Parameters
    ----------
    streams : {name: Series of daily simple portfolio returns}

    Takes ready-made return streams rather than (returns, weights) pairs
    because not every stream has weights: the NIFTY 50 benchmark is an index
    level series, and the old signature could not represent it.
    """
    rows = {name: stats_from_returns(stream) for name, stream in streams.items()}
    return pd.DataFrame(rows).T
