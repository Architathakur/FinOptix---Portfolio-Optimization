"""
Backtest and performance comparison: BL-optimized portfolio vs. an
equal-weight benchmark over the held-out test window.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cumulative_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weights = weights.reindex(returns.columns).fillna(0.0)
    port_returns = returns @ weights.values
    return (1 + port_returns).cumprod()


def performance_stats(returns: pd.DataFrame, weights: pd.Series) -> dict:
    """CAGR, annualized volatility, Sharpe ratio (rf=0), and max drawdown."""
    weights = weights.reindex(returns.columns).fillna(0.0)
    port_returns = returns @ weights.values
    cum = (1 + port_returns).cumprod()

    n_days = len(port_returns)
    if n_days == 0:
        return {"CAGR": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDrawdown": np.nan}

    total_return = cum.iloc[-1] - 1
    years = n_days / TRADING_DAYS
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    ann_vol = port_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (port_returns.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else np.nan

    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    return {"CAGR": cagr, "AnnVol": ann_vol, "Sharpe": sharpe, "MaxDrawdown": max_dd}


def compare_portfolios(returns: pd.DataFrame, portfolios: dict) -> pd.DataFrame:
    """
    portfolios: dict of {name: weights Series}
    Returns a DataFrame of performance_stats, one row per portfolio.
    """
    rows = {name: performance_stats(returns, w) for name, w in portfolios.items()}
    return pd.DataFrame(rows).T
