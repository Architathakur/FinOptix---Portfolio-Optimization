import numpy as np
import pandas as pd
import pytest

from src.optimizer import max_sharpe_weights
from src.backtest import performance_stats, cumulative_returns, compare_portfolios

ASSETS = ["A", "B", "C"]


@pytest.fixture
def synthetic_mu_cov():
    mu = pd.Series([0.08, 0.05, 0.12], index=ASSETS)
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.02], [0.01, 0.03, 0.005], [0.02, 0.005, 0.06]],
        index=ASSETS, columns=ASSETS,
    )
    return mu, cov


def test_max_sharpe_weights_sum_to_one(synthetic_mu_cov):
    mu, cov = synthetic_mu_cov
    weights = max_sharpe_weights(mu, cov)
    assert set(weights.index) == set(ASSETS)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-9).all()  # long-only, no negative weights


@pytest.fixture
def synthetic_returns():
    np.random.seed(1)
    dates = pd.bdate_range("2024-01-01", periods=252)
    return pd.DataFrame(
        np.random.randn(252, 3) * 0.01, index=dates, columns=ASSETS
    )


def test_performance_stats_keys(synthetic_returns):
    weights = pd.Series([0.5, 0.3, 0.2], index=ASSETS)
    stats = performance_stats(synthetic_returns, weights)
    assert set(stats.keys()) == {"CAGR", "AnnVol", "Sharpe", "MaxDrawdown"}
    assert all(np.isfinite(v) for v in stats.values())


def test_cumulative_returns_starts_near_one(synthetic_returns):
    weights = pd.Series([1 / 3, 1 / 3, 1 / 3], index=ASSETS)
    cum = cumulative_returns(synthetic_returns, weights)
    assert cum.iloc[0] == pytest.approx(1 + synthetic_returns.iloc[0] @ weights.values, abs=1e-9)


def test_compare_portfolios_returns_one_row_per_portfolio(synthetic_returns):
    bl_weights = pd.Series([0.6, 0.2, 0.2], index=ASSETS)
    eq_weights = pd.Series([1 / 3, 1 / 3, 1 / 3], index=ASSETS)
    result = compare_portfolios(synthetic_returns, {"BL": bl_weights, "Equal": eq_weights})
    assert list(result.index) == ["BL", "Equal"]
    assert "Sharpe" in result.columns
