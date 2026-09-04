"""
FinOptix: ML-assisted Black-Litterman portfolio optimizer.

Run with:  python main.py

Pipeline:
  1. ONE download covering TRAIN_START..TEST_END, so every rolling feature
     warms up on a continuous price series
  2. Build the feature panel, train per-ticker models on TRAIN (purged), and
     predict forward returns over VALID and TEST
  3. Measure cross-sectional rank IC on both windows -- the metric that
     matches what the strategy actually does
  4. Select stocks, build Black-Litterman views and the covariance matrix
     using VALID ONLY
  5. Mean-variance optimize (max Sharpe) on the BL posterior
  6. Freeze the weights, then backtest on TEST against equal-weight and the
     NIFTY 50, with transaction costs
  7. Bootstrap-test whether the Sharpe differences are distinguishable from
     luck, and save plots + CSVs to outputs/

TEST is touched only in step 6, after the weights are final. Nothing in steps
1-5 is allowed to see it.
"""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data_cache") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")  # headless: the pipeline writes PNGs, it never shows them

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.data import download_prices, download_fundamentals
from src.ml_returns import build_panel, train_predict, window_slice
from src.scoring import score_and_select
from src.black_litterman import (
    implied_equilibrium_returns,
    build_ml_views,
    omega_from_confidence,
    black_litterman_posterior,
    regularize_covariance,
)
from src.optimizer import max_sharpe_weights
from src.evaluation import rank_ic, sharpe_difference_test
from src.backtest import compare_portfolios, portfolio_returns

LOGGER = logging.getLogger(__name__)

IC_SUMMARY_FIELDS = ["n_periods", "mean_ic", "ic_std", "ic_t_stat", "hit_rate"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the FinOptix ML-assisted Black-Litterman pipeline."
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
        "--cost-bps",
        type=float,
        default=config.COST_BPS,
        help=(
            "One-off entry cost in basis points of gross exposure, charged on "
            f"day 0 of the backtest (default: {config.COST_BPS})."
        ),
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
    """
    Light sanity check on the per-ticker models.

    Deliberately not the headline metric: a per-ticker time-series correlation
    says nothing about whether the model ranks the universe correctly, which is
    the only thing the strategy relies on. See the rank IC for that.
    """
    if ml_metrics.empty:
        LOGGER.warning("No ML model metrics were produced - no ticker trained.")
        return

    LOGGER.info("Trained %s per-ticker models.", len(ml_metrics))
    if "valid_corr" in ml_metrics:
        unusable = ml_metrics.index[ml_metrics["valid_corr"].isna()]
        if len(unusable):
            LOGGER.warning(
                "Tickers with no usable validation predictions: %s",
                ", ".join(map(str, unusable)),
            )


def _log_ic(name: str, result: dict):
    LOGGER.info(
        "Rank IC [%s]: mean=%.4f  std=%.4f  t=%.2f  hit_rate=%.2f  n_periods=%d",
        name,
        result["mean_ic"],
        result["ic_std"],
        result["ic_t_stat"] if np.isfinite(result["ic_t_stat"]) else float("nan"),
        result["hit_rate"],
        result["n_periods"],
    )


def _benchmark_stream(use_cache: bool, cost_bps: float) -> pd.Series | None:
    """
    NIFTY 50 total return stream over the test window.

    Nice to have, not load-bearing: if the download fails the pipeline carries
    on with the two internal benchmarks. Charged the same entry cost as the
    portfolios so the comparison is like for like. Its first day is unavailable
    (the download starts at TEST_START, so the first return needs two closes).
    """
    ticker = config.BENCHMARK_TICKER
    try:
        prices = download_prices(
            [ticker], config.TEST_START, config.TEST_END, use_cache=use_cache
        )
        close = prices["Close"][ticker].ffill().dropna()
        returns = close.pct_change().dropna().to_frame(name=ticker)
        if returns.empty:
            raise ValueError("no benchmark returns in the test window")
        return portfolio_returns(returns, pd.Series({ticker: 1.0}), cost_bps=cost_bps)
    except Exception as exc:  # network, bad symbol, empty window
        LOGGER.warning(
            "Benchmark %s unavailable (%s). Continuing without it.", ticker, exc
        )
        return None


def main(argv=None):
    args = parse_args(argv)
    configure_logging()

    if args.top_n < 2:
        raise ValueError("--top-n must be at least 2 for covariance estimation.")
    if not (0 < args.confidence <= 1):
        raise ValueError("--confidence must be in the interval (0, 1].")
    if args.cost_bps < 0:
        raise ValueError("--cost-bps must be non-negative.")

    tickers = load_tickers(args.tickers_file)
    use_cache = not args.refresh_cache
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Data ------------------------------------------------------------
    # One continuous download. Splitting the download and computing features
    # per-slice restarts every rolling window inside the later slice.
    LOGGER.info(
        "Using %s tickers. top_n=%s confidence=%.3f cost_bps=%.1f",
        len(tickers), args.top_n, args.confidence, args.cost_bps,
    )
    LOGGER.info(
        "TRAIN %s -> %s | VALID %s -> %s | TEST %s -> %s",
        config.TRAIN_START, config.TRAIN_END,
        config.VALID_START, config.VALID_END,
        config.TEST_START, config.TEST_END,
    )
    prices = download_prices(
        tickers, config.TRAIN_START, config.TEST_END, use_cache=use_cache
    )
    LOGGER.info("Loading fundamentals")
    fundamentals = download_fundamentals(tickers, use_cache=use_cache)

    # 2. Features + models -------------------------------------------------
    LOGGER.info("Building feature panel over the full price history")
    panel = build_panel(prices, tickers, horizon=config.PRED_HORIZON)
    if not panel:
        raise ValueError("No tickers produced a usable feature panel.")

    LOGGER.info(
        "Training on TRAIN (purged by %s rows) and predicting VALID + TEST",
        config.PURGE_DAYS,
    )
    predictions, actuals, ml_metrics = train_predict(
        panel,
        train_range=(config.TRAIN_START, config.TRAIN_END),
        predict_ranges={
            "valid": (config.VALID_START, config.VALID_END),
            "test": (config.TEST_START, config.TEST_END),
        },
        purge=config.PURGE_DAYS,
    )
    ml_metrics.to_csv(f"{config.OUTPUT_DIR}/ml_model_metrics.csv") #output#
    LOGGER.info("ML model metrics saved to %s/ml_model_metrics.csv", config.OUTPUT_DIR)
    warn_on_model_quality(ml_metrics)

    if predictions["valid"].empty:
        raise ValueError("No validation predictions were produced.")

    # 3. Information coefficient ------------------------------------------
    ic_results = {
        name: rank_ic(predictions[name], actuals[name], horizon=config.PRED_HORIZON)
        for name in ("valid", "test")
    }
    for name, result in ic_results.items():
        _log_ic(name, result)
    ic_table = pd.DataFrame(
        {name: {k: res[k] for k in IC_SUMMARY_FIELDS} for name, res in ic_results.items()}
    ).T
    ic_table.index.name = "window"
    ic_table.to_csv(f"{config.OUTPUT_DIR}/information_coefficient.csv") #output#
    LOGGER.info("Rank IC saved to %s/information_coefficient.csv", config.OUTPUT_DIR)

    # 4. Score + select top N ---------------------------------------------
    # VALID predictions only. Selecting on TEST predictions would pick the
    # stocks that already did well over the window we are about to backtest.
    top_stocks, scores = score_and_select(
        predictions["valid"], fundamentals, top_n=args.top_n
    )
    scores.to_csv(f"{config.OUTPUT_DIR}/stock_scores.csv") #output#
    LOGGER.info("Top %s stocks selected (on VALID):\n%s", args.top_n, top_stocks)
    selected = list(top_stocks.index)

    # 5. Black-Litterman, entirely from VALID ------------------------------
    LOGGER.info("Computing Black-Litterman posterior from validation-window data")
    close_all = prices["Close"].reindex(columns=selected).ffill()
    daily_returns = close_all.pct_change()

    valid_returns = window_slice(
        daily_returns, config.VALID_START, config.VALID_END
    ).dropna(axis=1, how="all").dropna(axis=0, how="any")
    dropped = sorted(set(selected) - set(valid_returns.columns))
    if dropped:
        LOGGER.warning("Dropping selected tickers with no validation prices: %s",
                       ", ".join(dropped))
    selected = list(valid_returns.columns)
    if len(selected) < 2:
        raise ValueError("Fewer than two selected stocks have usable validation prices.")

    # Annualized covariance, so it shares units with the annualized views below.
    cov_ann = regularize_covariance(valid_returns.cov() * config.TRADING_DAYS)
    w_mkt = np.ones(len(selected)) / len(selected)  # equal-weight proxy for "market"
    pi = implied_equilibrium_returns(cov_ann, w_mkt, config.RISK_AVERSION)

    # Q is a mean predicted PRED_HORIZON-day return; scale it to annual so the
    # views, the prior and the risk aversion all speak the same language. The
    # old code compared a daily-mean view against a daily covariance with an
    # implicitly annual risk aversion, and the units never lined up.
    P, Q = build_ml_views(predictions["valid"], selected)
    Q = Q * (config.TRADING_DAYS / config.PRED_HORIZON)
    Omega = omega_from_confidence(P, cov_ann, config.TAU, args.confidence)
    mu_bl, cov_bl = black_litterman_posterior(cov_ann, pi, P, Q, Omega, tau=config.TAU)

    # 6. Optimize -> weights are now FROZEN --------------------------------
    LOGGER.info("Optimizing max-Sharpe portfolio")
    bl_weights = max_sharpe_weights(mu_bl, cov_bl)
    equal_weights = pd.Series(1.0 / len(selected), index=selected)
    bl_weights.to_csv(f"{config.OUTPUT_DIR}/portfolio_weights.csv", header=["Weight"]) #output#
    LOGGER.info(
        "Black-Litterman optimized weights above 1%%:\n%s",
        bl_weights[bl_weights > 0.01].sort_values(ascending=False),
    )

    # 7. Backtest on TEST --------------------------------------------------
    # First time anything reads the test window.
    LOGGER.info("Backtesting on TEST %s -> %s", config.TEST_START, config.TEST_END)
    test_returns = window_slice(
        daily_returns, config.TEST_START, config.TEST_END
    ).dropna(axis=1, how="all").dropna(axis=0, how="any")
    if test_returns.empty:
        raise ValueError("No test-window returns available for the selected stocks.")
    missing_in_test = sorted(set(selected) - set(test_returns.columns))
    if missing_in_test:
        LOGGER.warning("Selected tickers missing from the test window: %s",
                       ", ".join(missing_in_test))

    streams = {
        "Black-Litterman": portfolio_returns(test_returns, bl_weights, cost_bps=args.cost_bps),
        "Equal-Weight": portfolio_returns(test_returns, equal_weights, cost_bps=args.cost_bps),
    }
    benchmark = _benchmark_stream(use_cache, args.cost_bps)
    if benchmark is not None:
        streams["NIFTY 50"] = benchmark

    stats = compare_portfolios(streams)
    stats.to_csv(f"{config.OUTPUT_DIR}/performance_stats.csv") #output#
    if not np.isfinite(stats.to_numpy(dtype=float)).all():
        LOGGER.warning("Performance stats contain NaN or infinite values:\n%s", stats)
    LOGGER.info("Performance comparison:\n%s", stats)

    # 8. Is the difference distinguishable from luck? ----------------------
    sig_rows = {}
    for name, stream in streams.items():
        if name == "Black-Litterman":
            continue
        sig_rows[f"Black-Litterman vs {name}"] = sharpe_difference_test(
            streams["Black-Litterman"], stream
        )
    significance = pd.DataFrame(sig_rows).T
    significance.index.name = "comparison"
    significance.to_csv(f"{config.OUTPUT_DIR}/significance.csv") #output#
    LOGGER.info("Sharpe difference bootstrap:\n%s", significance)

    # 9. Plots --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, stream in streams.items():
        (1 + stream).cumprod().plot(ax=ax, label=name, alpha=0.85)
    ax.set_title("Cumulative Returns Over the Held-Out Test Window")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 unit")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{config.OUTPUT_DIR}/cumulative_returns.png", dpi=150) #output#
    plt.close(fig)
    LOGGER.info("Saved plot: %s/cumulative_returns.png", config.OUTPUT_DIR)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bl_weights[bl_weights > 0.01].sort_values(ascending=False).plot(
        kind="bar", ax=ax2, label="Black-Litterman"
    )
    ax2.set_title("Final Portfolio Weights")
    ax2.set_ylabel("Weight")
    fig2.tight_layout()
    fig2.savefig(f"{config.OUTPUT_DIR}/portfolio_weights.png", dpi=150) #output#
    plt.close(fig2)
    LOGGER.info("Saved plot: %s/portfolio_weights.png", config.OUTPUT_DIR)

    # 10. The honest caveat -------------------------------------------------
    test_t = ic_results["test"]["ic_t_stat"]
    if not np.isfinite(test_t) or abs(test_t) <= 2:
        LOGGER.warning(
            "TEST rank IC t-stat is %.2f: the information coefficient is not "
            "distinguishable from zero. Any outperformance in the backtest "
            "above should be treated as noise, not as evidence of skill.",
            test_t if np.isfinite(test_t) else float("nan"),
        )

    return stats


if __name__ == "__main__":
    main()
