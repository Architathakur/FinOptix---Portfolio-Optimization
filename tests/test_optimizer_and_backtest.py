import numpy as np
import pandas as pd
import pytest

from src.optimizer import max_sharpe_weights
from src.backtest import (
    performance_stats,
    portfolio_returns,
    cumulative_returns,
    stats_from_returns,
    compare_portfolios,
)

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


def test_buy_and_hold_weights_drift_with_performance():
    """
    After inception the book is no longer at target weights: a winner is a
    bigger share of it. A fixed weighted average of daily returns would model
    a portfolio rebalanced every day, for free.
    """
    dates = pd.bdate_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"A": [1.0, 0.10], "B": [-0.5, 0.0]}, index=dates)
    weights = pd.Series({"A": 0.5, "B": 0.5})

    port = portfolio_returns(returns, weights)

    # Day 0: 0.5*2.0 + 0.5*0.5 = 1.25, so +25% on a base of 1.0.
    assert port.iloc[0] == pytest.approx(0.25)
    # Day 1: A is now 1.0 of a 1.25 book. Value 0.5*2.2 + 0.5*0.5 = 1.35,
    # i.e. +8%. Daily rebalancing would have produced 5%.
    assert port.iloc[1] == pytest.approx(0.08)
    assert port.iloc[1] != pytest.approx(0.05)


def test_entry_cost_reduces_returns_by_exactly_the_expected_amount(synthetic_returns):
    weights = pd.Series([0.6, 0.2, 0.2], index=ASSETS)
    cost_bps = 25.0
    expected_cost = cost_bps / 10_000 * weights.abs().sum()

    gross = portfolio_returns(synthetic_returns, weights, cost_bps=0.0)
    net = portfolio_returns(synthetic_returns, weights, cost_bps=cost_bps)

    # Charged once, on day 0, at exactly cost_bps of gross exposure.
    assert net.iloc[0] == pytest.approx(gross.iloc[0] - expected_cost, abs=1e-12)
    pd.testing.assert_series_equal(net.iloc[1:], gross.iloc[1:])

    # Every later day compounds off a book that is smaller by a constant
    # factor, so the two wealth paths stay in fixed proportion.
    ratio = cumulative_returns(synthetic_returns, weights, cost_bps=cost_bps) / \
        cumulative_returns(synthetic_returns, weights, cost_bps=0.0)
    expected_ratio = (1 + gross.iloc[0] - expected_cost) / (1 + gross.iloc[0])
    assert ratio.to_numpy() == pytest.approx(expected_ratio, abs=1e-12)


def test_compare_portfolios_accepts_return_streams(synthetic_returns):
    """
    compare_portfolios takes ready-made streams, so a benchmark with no
    weights -- an index series -- sits alongside the weighted portfolios.
    """
    bl = portfolio_returns(synthetic_returns, pd.Series([0.6, 0.2, 0.2], index=ASSETS))
    eq = portfolio_returns(synthetic_returns, pd.Series([1 / 3, 1 / 3, 1 / 3], index=ASSETS))
    benchmark = synthetic_returns["A"]  # index level series, no weights at all

    result = compare_portfolios({"BL": bl, "Equal": eq, "Benchmark": benchmark})

    assert list(result.index) == ["BL", "Equal", "Benchmark"]
    assert "Sharpe" in result.columns
    assert np.isfinite(result.to_numpy(dtype=float)).all()
    assert result.loc["BL"].to_dict() == pytest.approx(stats_from_returns(bl))
