"""
Mean-variance optimization on top of the Black-Litterman posterior,
via PyPortfolioOpt.
"""

import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier


def max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """
    Optimize for maximum Sharpe ratio given expected returns `mu` and
    covariance `cov`. Returns a Series of weights indexed like `mu`,
    with tiny (<1%) positions cleaned to zero.
    """
    ef = EfficientFrontier(mu, cov)
    ef.max_sharpe()
    weights = ef.clean_weights()
    return pd.Series(weights).reindex(mu.index).fillna(0.0)
