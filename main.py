"""
Finoptix: ML-assisted Black-Litterman portfolio optimizer.

Run with:  python main.py

Pipeline:
  1. Download price history + fundamentals for the ticker universe
  2. Train a per-stock XGBoost model to predict daily returns
  3. Score stocks on ML expected return + fundamentals, select top N
  4. Compute Black-Litterman posterior returns, with views built directly
     from the ML predictions (the fix vs. the original notebook)
  5. Mean-variance optimize (max Sharpe) on the BL posterior
  6. Backtest the resulting portfolio against an equal-weight benchmark
  7. Save plots + a text report to outputs/
"""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data_cache") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.data import download_prices, download_fundamentals
from src.ml_returns import train_predict_all
from src.scoring import score_and_select
from src.black_litterman import (
    implied_equilibrium_returns,
    build_ml_views,
    omega_from_confidence,
    black_litterman_posterior,
    regularize_covariance,
)
from src.optimizer import max_sharpe_weights
from src.backtest import compare_portfolios, cumulative_returns

LOGGER = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Finoptix ML-assisted Black-Litterman pipeline."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=config.TOP_N_STOCKS,
        help=f"Number of stocks to select for the optimizer (default: {config.TOP_N_STOCKS}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.VIEW_CONFIDENCE,
        help=f"ML view confidence in (0, 1] (default: {config.VIEW_CONFIDENCE}).",
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        help="Optional text/CSV file of tickers, one per line or comma-separated.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached market data and re-download from yfinance.",
    )
    return parser.parse_args(argv)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_tickers(tickers_file: Path | None) -> list[str]:
    if tickers_file is None:
        return list(config.TICKERS)

    raw = tickers_file.read_text(encoding="utf-8")
    tickers = [
        item.strip()
        for line in raw.splitlines()
        for item in line.split(",")
        if item.strip() and not item.strip().startswith("#")
    ]
    if not tickers:
        raise ValueError(f"No tickers found in {tickers_file}")
    return tickers


def warn_on_model_quality(ml_metrics: pd.DataFrame):
    if ml_metrics.empty or "correlation" not in ml_metrics:
        LOGGER.warning("No ML model metrics were produced.")
        return

    weak = ml_metrics[
        ml_metrics["correlation"].isna() | (ml_metrics["correlation"] < -0.25)
    ]
    if not weak.empty:
        LOGGER.warning(
            "Tickers with NaN or strongly negative prediction correlation: %s",
            ", ".join(weak.index),
        )


def main(argv=None):
    args = parse_args(argv)
    configure_logging()

    if args.top_n < 2:
        raise ValueError("--top-n must be at least 2 for covariance estimation.")
    if not (0 < args.confidence <= 1):
        raise ValueError("--confidence must be in the interval (0, 1].")

    tickers = load_tickers(args.tickers_file)
    use_cache = not args.refresh_cache
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Data ----------------------------------------------------------
    LOGGER.info("Using %s tickers. top_n=%s confidence=%.3f", len(tickers), args.top_n, args.confidence)
    LOGGER.info("Loading training prices %s -> %s", config.TRAIN_START, config.TRAIN_END)
    train_prices = download_prices(
        tickers, config.TRAIN_START, config.TRAIN_END, use_cache=use_cache
    )
    LOGGER.info("Loading test prices %s -> %s", config.TEST_START, config.TEST_END)
    test_prices = download_prices(
        tickers, config.TEST_START, config.TEST_END, use_cache=use_cache
    )

    LOGGER.info("Loading fundamentals")
    fundamentals = download_fundamentals(tickers, use_cache=use_cache)

    # 2. ML expected returns -------------------------------------------
    LOGGER.info("Training per-ticker models and predicting held-out returns")
    actual_returns, expected_returns, ml_metrics = train_predict_all(
        train_prices, test_prices, tickers
    )
    ml_metrics.to_csv(f"{config.OUTPUT_DIR}/ml_model_metrics.csv")
    LOGGER.info("ML model metrics saved to %s/ml_model_metrics.csv", config.OUTPUT_DIR)
    LOGGER.info("ML metric summary:\n%s", ml_metrics.describe())
    warn_on_model_quality(ml_metrics)

    # 3. Score + select top N -------------------------------------------
    top_stocks, scores = score_and_select(expected_returns, fundamentals, top_n=args.top_n)
    scores.to_csv(f"{config.OUTPUT_DIR}/stock_scores.csv")
    LOGGER.info("Top %s stocks selected:\n%s", args.top_n, top_stocks)

    selected = list(top_stocks.index)

    # 4. Black-Litterman --------------------------------------------------
    LOGGER.info("Computing Black-Litterman posterior")
    close = test_prices["Close"].reindex(columns=selected).dropna(axis=1, how="all")
    close = close.ffill().dropna(axis=0, how="any")
    selected = list(close.columns)  # drop any that had missing data
    if len(selected) < 2:
        raise ValueError("Fewer than two selected stocks have usable test-window prices.")
    log_returns = np.log(close / close.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("No test-window returns available after cleaning selected prices.")

    cov_sample = regularize_covariance(log_returns.cov())
    w_mkt = np.ones(len(selected)) / len(selected)  # equal-weight proxy for "market"

    pi = implied_equilibrium_returns(cov_sample, w_mkt, config.RISK_AVERSION)

    P, Q = build_ml_views(expected_returns, selected)
    Omega = omega_from_confidence(P, cov_sample, config.TAU, args.confidence)
    mu_bl, cov_bl = black_litterman_posterior(cov_sample, pi, P, Q, Omega, tau=config.TAU)

    # 5. Optimize -----------------------------------------------------
    LOGGER.info("Optimizing max-Sharpe portfolio")
    bl_weights = max_sharpe_weights(mu_bl, cov_bl)
    equal_weights = pd.Series(w_mkt, index=selected)
    bl_weights.to_csv(f"{config.OUTPUT_DIR}/portfolio_weights.csv", header=["Weight"])

    LOGGER.info(
        "Black-Litterman optimized weights above 1%%:\n%s",
        bl_weights[bl_weights > 0.01].sort_values(ascending=False),
    )

    # 6. Backtest -------------------------------------------------------
    LOGGER.info("Backtesting Black-Litterman portfolio vs equal-weight benchmark")
    stats = compare_portfolios(
        log_returns, {"Black-Litterman": bl_weights, "Equal-Weight": equal_weights}
    )
    stats.to_csv(f"{config.OUTPUT_DIR}/performance_stats.csv")
    if not np.isfinite(stats.to_numpy()).all():
        LOGGER.warning("Performance stats contain NaN or infinite values:\n%s", stats)
    LOGGER.info("Performance comparison:\n%s", stats)

    # 7. Plots ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_returns(log_returns, bl_weights).plot(ax=ax, label="Black-Litterman")
    cumulative_returns(log_returns, equal_weights).plot(ax=ax, label="Equal-Weight", alpha=0.7)
    ax.set_title("Cumulative Returns: Black-Litterman vs. Equal-Weight")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of ₹1")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{config.OUTPUT_DIR}/cumulative_returns.png", dpi=150)
    LOGGER.info("Saved plot: %s/cumulative_returns.png", config.OUTPUT_DIR)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bl_weights[bl_weights > 0.01].sort_values(ascending=False).plot(
        kind="bar", ax=ax2, label="Black-Litterman"
    )
    ax2.set_title("Final Portfolio Weights")
    ax2.set_ylabel("Weight")
    fig2.tight_layout()
    fig2.savefig(f"{config.OUTPUT_DIR}/portfolio_weights.png", dpi=150)
    LOGGER.info("Saved plot: %s/portfolio_weights.png", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
