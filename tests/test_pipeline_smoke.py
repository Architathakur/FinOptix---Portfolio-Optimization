"""
End-to-end pipeline run with no network access.

Both network entry points are replaced with synthetic generators, so this
exercises the real wiring -- panel construction, purged training, IC,
selection, Black-Litterman, optimization, backtest, bootstrap, plots -- on
data that never leaves the process.
"""

import numpy as np
import pandas as pd
import pytest

import config
import main as main_module
from src.features import TARGET_COLUMN, calculate_features
from src.ml_returns import purged_train_slice
from config import FEATURE_COLUMNS

SMOKE_TICKERS = ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS", "EEE.NS", "FFF.NS"]
PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _synthetic_close(n: int, seed: int) -> np.ndarray:
    """Mildly upward-drifting random walk, i.e. something equity-shaped."""
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))


def synthetic_prices(tickers, start, end, **kwargs) -> pd.DataFrame:
    """Stand-in for src.data.download_prices, same column layout."""
    tickers = list(tickers)
    dates = pd.bdate_range(start, end)
    columns = pd.MultiIndex.from_product([PRICE_FIELDS, tickers])
    data = pd.DataFrame(index=dates, columns=columns, dtype=float)

    for i, ticker in enumerate(tickers):
        close = _synthetic_close(len(dates), seed=1000 + i)
        rng = np.random.default_rng(2000 + i)
        data[("Close", ticker)] = close
        data[("Adj Close", ticker)] = close
        data[("Open", ticker)] = close * (1 + rng.normal(0, 0.002, len(dates)))
        data[("High", ticker)] = close * (1 + abs(rng.normal(0, 0.004, len(dates))))
        data[("Low", ticker)] = close * (1 - abs(rng.normal(0, 0.004, len(dates))))
        data[("Volume", ticker)] = rng.lognormal(13.0, 0.35, len(dates))

    return data.sort_index(axis=1)


def synthetic_fundamentals(tickers, **kwargs) -> pd.DataFrame:
    """Stand-in for src.data.download_fundamentals."""
    tickers = list(tickers)
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "PE": rng.uniform(12, 45, len(tickers)),
            "DE": rng.uniform(5, 120, len(tickers)),
            "MktCap": rng.uniform(2e11, 9e12, len(tickers)),
        },
        index=tickers,
    )


@pytest.fixture
def offline_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "download_prices", synthetic_prices)
    monkeypatch.setattr(main_module, "download_fundamentals", synthetic_fundamentals)
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("\n".join(SMOKE_TICKERS), encoding="utf-8")
    return tickers_file, tmp_path


def test_pipeline_runs_end_to_end_and_produces_finite_stats(offline_pipeline):
    tickers_file, output_dir = offline_pipeline

    stats = main_module.main(
        ["--tickers-file", str(tickers_file), "--top-n", "3", "--cost-bps", "20"]
    )

    assert isinstance(stats, pd.DataFrame)
    assert "Black-Litterman" in stats.index
    assert "Equal-Weight" in stats.index
    assert set(stats.columns) == {"CAGR", "AnnVol", "Sharpe", "MaxDrawdown"}
    assert np.isfinite(stats.to_numpy(dtype=float)).all(), f"non-finite stats:\n{stats}"

    for name in (
        "ml_model_metrics.csv",
        "information_coefficient.csv",
        "stock_scores.csv",
        "portfolio_weights.csv",
        "performance_stats.csv",
        "significance.csv",
        "cumulative_returns.png",
        "portfolio_weights.png",
    ):
        assert (output_dir / name).exists(), f"pipeline did not write {name}"

    ic = pd.read_csv(output_dir / "information_coefficient.csv", index_col="window")
    assert set(ic.index) == {"valid", "test"}
    assert (ic["n_periods"] > 0).all()


def test_training_slice_ends_before_validation_start_after_purge():
    """
    The purge is what stops a forward-looking label from straddling the split.
    The last training row must sit at least PURGE_DAYS observations before
    VALID_START, so its 21-day label closes before validation begins.
    """
    prices = synthetic_prices(["AAA.NS"], config.TRAIN_START, config.TEST_END)
    frame = calculate_features(
        pd.DataFrame(
            {"Close": prices[("Close", "AAA.NS")], "Volume": prices[("Volume", "AAA.NS")]}
        ),
        horizon=config.PRED_HORIZON,
    )

    train = purged_train_slice(
        frame, (config.TRAIN_START, config.TRAIN_END), purge=config.PURGE_DAYS
    )
    valid_start = pd.Timestamp(config.VALID_START)

    assert not train.empty
    assert train.index[-1] < valid_start

    usable = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    gap = usable[(usable.index > train.index[-1]) & (usable.index < valid_start)]
    assert len(gap) == config.PURGE_DAYS, (
        f"expected exactly {config.PURGE_DAYS} purged rows between the end of "
        f"training and VALID_START, found {len(gap)}"
    )
