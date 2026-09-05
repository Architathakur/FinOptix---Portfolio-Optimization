"""
Walk-forward backtesting.

Run with:  python -m src.walkforward

The single-window backtest in main.py produces one allocation decision and
about eleven non-overlapping IC observations. That is not enough evidence to
separate a real effect from noise in either direction, which is exactly why
every bootstrap confidence interval it produced straddled zero. The experiment
was never powered to reach a conclusion.

This module re-runs the whole decision process quarterly on an expanding
training window, giving ~18 independent allocations and ~55 pooled IC
observations. It is a methodological upgrade, not an attempt to improve the
result -- the corrected numbers are expected to stay unimpressive.

The scheme, at each rebalance date t:

    TRAIN  WALKFORWARD_START -> (t - 1y - PURGE_DAYS)    expanding, fixed start
    VALID  (t - 1y)          -> (t - PURGE_DAYS)         makes every decision
    HOLD   t                 -> next rebalance date      strictly out of sample

PURGE_DAYS trading days are removed at BOTH inner boundaries. The label at
time t spans (t, t + PRED_HORIZON], so without the first gap the tail of TRAIN
carries labels reaching into VALID, and without the second the tail of VALID
carries labels reaching past t into the window being held. Both gaps are
enforced on the trading calendar, not on calendar days, so they survive
holidays. tests/test_walkforward.py asserts them for every rebalance.

Features are computed ONCE over the full price history and sliced per
rebalance. Recomputing them inside the loop would restart every rolling window
in each slice -- and with 18 rebalances it would also be 18x the work.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data_cache") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.data import download_prices, download_fundamentals
from src.ml_returns import MIN_TRAIN_ROWS, build_panel, train_predict, window_slice
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

BL = config.STREAM_BL
EW_SELECTED = config.STREAM_EW_SELECTED
EW_UNIVERSE = config.STREAM_EW_UNIVERSE
BENCHMARK = config.STREAM_BENCHMARK

# Chain of comparisons that decomposes the overall result. Each step isolates
# one layer, so a difference can be attributed rather than just observed.
COMPARISONS = [
    (EW_UNIVERSE, BENCHMARK, "equal-weighting vs cap-weighting the index"),
    (EW_SELECTED, EW_UNIVERSE, "selection effect: picks vs the whole universe"),
    (BL, EW_SELECTED, "weighting effect: BL weights vs equal weights, same names"),
    (BL, BENCHMARK, "overall: the strategy vs the index"),
]


@dataclass(frozen=True)
class Rebalance:
    """One walk-forward step. All windows are half-open, [start, end)."""

    date: pd.Timestamp
    train: tuple
    valid: tuple
    hold: tuple


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def build_schedule(calendar, start=None, end=None, valid_days=None, purge=None,
                   min_train_rows: int = MIN_TRAIN_ROWS, freq=None) -> list[Rebalance]:
    """
    Build the rebalance schedule on an actual trading calendar.

    Every boundary is an index into `calendar`, so the purge gaps are exact
    counts of trading days rather than calendar days that shrink across a
    holiday week.

    A candidate rebalance is dropped when its training window would hold fewer
    than `min_train_rows` rows; that is what sets the start of the backtest.
    HOLD windows chain to the next ACCEPTED rebalance, so dropping a candidate
    never leaves a gap in the stitched return series.
    """
    start = config.WALKFORWARD_START if start is None else start
    valid_days = config.WALKFORWARD_VALID_DAYS if valid_days is None else valid_days
    purge = config.PURGE_DAYS if purge is None else purge
    freq = config.WALKFORWARD_FREQ if freq is None else freq

    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    if len(cal) == 0:
        return []
    start_ts = pd.Timestamp(start)
    start_idx = int(cal.searchsorted(start_ts, side="left"))
    last_exclusive = (
        pd.Timestamp(end) if end is not None else cal[-1] + pd.Timedelta(days=1)
    )

    if start_idx < config.FEATURE_WARMUP_ROWS:
        LOGGER.warning(
            "Only %s trading days of history precede %s, but the features need "
            "%s to warm up. Early training rows will be dropped as NaN; "
            "increase WALKFORWARD_WARMUP_DAYS.",
            start_idx, start_ts.date(), config.FEATURE_WARMUP_ROWS,
        )

    accepted = []
    for quarter in pd.date_range(start=start_ts, end=cal[-1], freq=freq):
        pos = int(cal.searchsorted(quarter, side="left"))
        if pos >= len(cal) or pos in {a[0] for a in accepted}:
            continue
        t = cal[pos]

        valid_start_idx = int(
            cal.searchsorted(t - pd.Timedelta(days=valid_days), side="left")
        )
        valid_end_idx = pos - purge
        train_end_idx = valid_start_idx - purge

        if train_end_idx <= start_idx or valid_end_idx <= valid_start_idx:
            continue
        if (train_end_idx - start_idx) < min_train_rows:
            continue
        accepted.append((pos, t, train_end_idx, valid_start_idx, valid_end_idx))

    schedule = []
    for i, (_, t, train_end_idx, valid_start_idx, valid_end_idx) in enumerate(accepted):
        hold_end = accepted[i + 1][1] if i + 1 < len(accepted) else last_exclusive
        schedule.append(
            Rebalance(
                date=t,
                train=(start_ts, cal[train_end_idx]),
                valid=(cal[valid_start_idx], cal[valid_end_idx]),
                hold=(t, hold_end),
            )
        )
    return schedule


# ---------------------------------------------------------------------------
# Weights, drift and turnover
# ---------------------------------------------------------------------------
def drifted_weights(weights: pd.Series, hold_returns: pd.DataFrame) -> pd.Series:
    """
    Where the weights ended up after being held through a window.

    A portfolio is not rebalanced between rebalance dates, so by the time the
    next one arrives the winners are a larger share of the book. Charging
    turnover against the ORIGINAL target weights would overstate the trade;
    the position that actually has to be traded is the drifted one.
    """
    if weights is None or len(weights) == 0:
        return pd.Series(dtype=float)

    known = [c for c in weights.index if c in hold_returns.columns]
    growth = (1 + hold_returns[known]).prod() if known else pd.Series(dtype=float)
    value = weights.reindex(known).fillna(0.0) * growth

    # A name that vanished from the data keeps its last known value: we have no
    # return for it, and silently zeroing it would fabricate a costless exit.
    stranded = weights.index.difference(known)
    if len(stranded):
        value = pd.concat([value, weights.reindex(stranded).fillna(0.0)])

    total = float(value.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(0.0, index=value.index)
    return value / total


def turnover(new: pd.Series, held: pd.Series) -> float:
    """One-way turnover: sum |w_new - w_held|, counting entries and exits."""
    if new is None or len(new) == 0:
        return 0.0
    held = pd.Series(dtype=float) if held is None else held
    index = new.index.union(held.index)
    delta = new.reindex(index).fillna(0.0) - held.reindex(index).fillna(0.0)
    return float(delta.abs().sum())


def hold_window_stream(hold_returns: pd.DataFrame, weights: pd.Series,
                       cost: float) -> pd.Series:
    """
    Daily returns for one hold window, with the rebalance cost on day one.

    The buy-and-hold drift accounting is the same as the single-window path;
    only the cost differs, because it is charged on turnover rather than on
    gross exposure.
    """
    stream = portfolio_returns(hold_returns, weights, cost_bps=0.0)
    if len(stream) and cost:
        stream.iloc[0] -= cost
    return stream


def _clean_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop tickers with any gap in the window, then any residual all-NaN dates."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.dropna(axis=1, how="any").dropna(axis=0, how="any")


# ---------------------------------------------------------------------------
# The walk-forward loop
# ---------------------------------------------------------------------------
def run_walkforward(tickers, top_n: int, confidence: float, cost_bps: float,
                    use_cache: bool = True, use_fundamentals: bool | None = None) -> dict:
    """Fit, select, allocate and hold, quarterly. Returns a results dict."""
    if use_fundamentals is None:
        use_fundamentals = config.USE_FUNDAMENTALS_IN_SELECTION

    # 1. Data. The download reaches back before WALKFORWARD_START so the
    #    rolling features are warm on the first training day.
    download_start = (
        pd.Timestamp(config.WALKFORWARD_START)
        - pd.Timedelta(days=config.WALKFORWARD_WARMUP_DAYS)
    ).date().isoformat()
    LOGGER.info(
        "Downloading %s -> %s (%s days of pre-history for feature warm-up)",
        download_start, config.TEST_END, config.WALKFORWARD_WARMUP_DAYS,
    )
    prices = download_prices(tickers, download_start, config.TEST_END, use_cache=use_cache)

    fundamentals = None
    if use_fundamentals:
        LOGGER.warning(
            "USE_FUNDAMENTALS_IN_SELECTION is on. Selection is contaminated by a "
            "current fundamentals snapshot; these results are not a measurement."
        )
        fundamentals = download_fundamentals(tickers, use_cache=use_cache)

    # 2. Features once, over the whole history.
    LOGGER.info("Building feature panel over the full price history")
    panel = build_panel(prices, tickers, horizon=config.PRED_HORIZON)
    if not panel:
        raise ValueError("No tickers produced a usable feature panel.")
    universe = sorted(panel)
    LOGGER.info("Tradeable universe: %s tickers", len(universe))

    close_all = prices["Close"].reindex(columns=universe).ffill()
    daily_returns = close_all.pct_change()
    calendar = prices.index

    schedule = build_schedule(calendar)
    if not schedule:
        raise ValueError(
            "No rebalance date has enough training history. Check "
            "WALKFORWARD_START and WALKFORWARD_WARMUP_DAYS."
        )
    LOGGER.info(
        "%s quarterly rebalances, %s -> %s",
        len(schedule), schedule[0].date.date(), schedule[-1].date.date(),
    )

    benchmark_returns = _benchmark_returns(schedule, use_cache)

    strategies = [BL, EW_SELECTED, EW_UNIVERSE]
    if benchmark_returns is not None:
        strategies.append(BENCHMARK)

    held = {name: pd.Series(dtype=float) for name in strategies}
    streams = {name: [] for name in strategies}
    pooled_predictions, pooled_actuals = [], []
    period_rows = []
    optimizer_fallbacks = 0

    for step, rb in enumerate(schedule, start=1):
        LOGGER.info(
            "[%s/%s] rebalance %s | TRAIN %s->%s | VALID %s->%s | HOLD %s->%s",
            step, len(schedule), rb.date.date(),
            pd.Timestamp(rb.train[0]).date(), pd.Timestamp(rb.train[1]).date(),
            pd.Timestamp(rb.valid[0]).date(), pd.Timestamp(rb.valid[1]).date(),
            pd.Timestamp(rb.hold[0]).date(), pd.Timestamp(rb.hold[1]).date(),
        )

        # --- fit and predict. purge=0 because the schedule already put the
        #     purge gap at the window boundary; purging again would drop a
        #     second PURGE_DAYS of perfectly usable training rows.
        predictions, actuals, _ = train_predict(
            panel,
            train_range=rb.train,
            predict_ranges={"valid": rb.valid, "hold": rb.hold},
            purge=0,
            progress=False,
        )
        valid_predictions = predictions["valid"]
        if valid_predictions.empty:
            LOGGER.warning("No validation predictions at %s; skipping.", rb.date.date())
            continue

        # --- select, on VALID only
        try:
            top_stocks, _ = score_and_select(
                valid_predictions, fundamentals, top_n=top_n,
                use_fundamentals=use_fundamentals,
            )
        except ValueError as exc:
            LOGGER.warning("Selection failed at %s (%s); skipping.", rb.date.date(), exc)
            continue
        selected = list(top_stocks.index)

        # --- Black-Litterman, on VALID only
        valid_returns = _clean_returns(
            window_slice(daily_returns.reindex(columns=selected), *rb.valid)
        )
        selected = list(valid_returns.columns)
        if len(selected) < 2:
            LOGGER.warning("Fewer than two usable names at %s; skipping.", rb.date.date())
            continue

        cov_ann = regularize_covariance(valid_returns.cov() * config.TRADING_DAYS)
        w_mkt = np.ones(len(selected)) / len(selected)
        pi = implied_equilibrium_returns(cov_ann, w_mkt, config.RISK_AVERSION)
        P, Q = build_ml_views(valid_predictions, selected)
        Q = Q * (config.TRADING_DAYS / config.PRED_HORIZON)
        Omega = omega_from_confidence(P, cov_ann, config.TAU, confidence)
        mu_bl, cov_bl = black_litterman_posterior(
            cov_ann, pi, P, Q, Omega, tau=config.TAU
        )

        try:
            bl_weights = max_sharpe_weights(mu_bl, cov_bl)
        except Exception as exc:
            # Happens when no posterior return clears the risk-free rate. Fall
            # back to equal weight rather than skipping the quarter, so the
            # stitched series stays continuous. Logged and counted.
            optimizer_fallbacks += 1
            LOGGER.warning(
                "Max-Sharpe failed at %s (%s); falling back to equal weight.",
                rb.date.date(), exc,
            )
            bl_weights = pd.Series(1.0 / len(selected), index=selected)

        # === WEIGHTS ARE NOW FROZEN. Everything below reads the hold window. ==
        hold_all = window_slice(daily_returns, *rb.hold)
        hold_selected = _clean_returns(hold_all.reindex(columns=selected))
        hold_universe = _clean_returns(hold_all)
        if hold_selected.empty or hold_universe.empty:
            LOGGER.warning("No hold-window returns at %s; skipping.", rb.date.date())
            continue

        target = {
            BL: (bl_weights.reindex(hold_selected.columns).fillna(0.0), hold_selected),
            EW_SELECTED: (
                pd.Series(1.0 / len(hold_selected.columns), index=hold_selected.columns),
                hold_selected,
            ),
            EW_UNIVERSE: (
                pd.Series(1.0 / len(hold_universe.columns), index=hold_universe.columns),
                hold_universe,
            ),
        }
        if benchmark_returns is not None:
            hold_bench = _clean_returns(window_slice(benchmark_returns, *rb.hold))
            if not hold_bench.empty:
                target[BENCHMARK] = (
                    pd.Series(1.0, index=hold_bench.columns), hold_bench,
                )

        row = {
            "rebalance_date": rb.date.date().isoformat(),
            "n_selected": len(hold_selected.columns),
            "n_universe": len(hold_universe.columns),
            "hold_days": len(hold_selected),
        }
        for name in strategies:
            if name not in target:
                continue
            weights, window_returns = target[name]
            traded = turnover(weights, held[name])
            cost = (cost_bps / 10_000.0) * traded
            stream = hold_window_stream(window_returns, weights, cost)
            streams[name].append(stream)
            held[name] = drifted_weights(weights, window_returns)

            row[f"turnover_{name}"] = traded
            row[f"cost_{name}"] = cost
            row[f"return_{name}"] = float((1 + stream).prod() - 1)

        # --- IC on the hold window: out of sample, and used only to measure.
        #     These predictions never touched the weights above.
        hold_pred, hold_act = predictions["hold"], actuals["hold"]
        if not hold_pred.empty:
            pooled_predictions.append(hold_pred)
            pooled_actuals.append(hold_act)
            window_ic = rank_ic(hold_pred, hold_act, horizon=config.PRED_HORIZON)
            row["ic_mean"] = window_ic["mean_ic"]
            row["ic_n"] = window_ic["n_periods"]
        period_rows.append(row)

    if not period_rows:
        raise ValueError("No rebalance completed; nothing to report.")

    stitched = {
        name: pd.concat(parts).sort_index()
        for name, parts in streams.items() if parts
    }
    # Pool the hold-window predictions into one continuous frame and sample it
    # ONCE, so the stride lands every PRED_HORIZON days across the whole
    # backtest. Running rank_ic per window and averaging would restart the
    # stride at each rebalance and let forward windows overlap at the seams.
    pooled_ic = rank_ic(
        pd.concat(pooled_predictions).sort_index() if pooled_predictions else pd.DataFrame(),
        pd.concat(pooled_actuals).sort_index() if pooled_actuals else pd.DataFrame(),
        horizon=config.PRED_HORIZON,
    )

    by_period = pd.DataFrame(period_rows).set_index("rebalance_date")
    summary = compare_portfolios(stitched)
    for name in summary.index:
        col = f"turnover_{name}"
        if col in by_period:
            summary.loc[name, "TotalTurnover"] = by_period[col].sum()
            summary.loc[name, "AvgTurnover"] = by_period[col].mean()
            summary.loc[name, "TotalCostPct"] = by_period[f"cost_{name}"].sum() * 100

    significance = {}
    for a, b, note in COMPARISONS:
        if a in stitched and b in stitched:
            result = sharpe_difference_test(stitched[a], stitched[b])
            result["isolates"] = note
            significance[f"{a} vs {b}"] = result
    significance = pd.DataFrame(significance).T
    significance.index.name = "comparison"

    return {
        "schedule": schedule,
        "stitched": stitched,
        "summary": summary,
        "by_period": by_period,
        "pooled_ic": pooled_ic,
        "significance": significance,
        "universe": universe,
        "optimizer_fallbacks": optimizer_fallbacks,
        "use_fundamentals": use_fundamentals,
    }


def _benchmark_returns(schedule, use_cache: bool) -> pd.DataFrame | None:
    """NIFTY 50 daily returns over the walk-forward span. Nice to have."""
    ticker = config.BENCHMARK_TICKER
    start = (schedule[0].date - pd.Timedelta(days=10)).date().isoformat()
    try:
        prices = download_prices([ticker], start, config.TEST_END, use_cache=use_cache)
        close = prices["Close"][ticker].ffill().dropna()
        returns = close.pct_change().dropna().to_frame(name=ticker)
        if returns.empty:
            raise ValueError("no benchmark returns over the walk-forward span")
        return returns
    except Exception as exc:
        LOGGER.warning(
            "Benchmark %s unavailable (%s). Continuing without it.", ticker, exc
        )
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def save_results(results: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    results["summary"].to_csv(f"{output_dir}/walkforward_summary.csv") #output#
    results["by_period"].to_csv(f"{output_dir}/walkforward_by_period.csv") #output#
    results["significance"].to_csv(f"{output_dir}/walkforward_significance.csv") #output#

    ic = results["pooled_ic"]
    series = ic["series"].rename("ic")
    series.index.name = "date"
    series.to_csv(f"{output_dir}/walkforward_ic.csv") #output#
    pd.DataFrame(
        [{k: ic[k] for k in ("n_periods", "mean_ic", "ic_std", "ic_t_stat", "hit_rate")}]
    ).to_csv(f"{output_dir}/walkforward_ic_summary.csv", index=False) #output#

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, stream in results["stitched"].items():
        (1 + stream).cumprod().plot(ax=ax, label=name, alpha=0.85)
    for rb in results["schedule"][1:]:
        ax.axvline(rb.date, color="grey", linewidth=0.4, alpha=0.35)
    ax.set_title(
        f"Walk-Forward Cumulative Returns "
        f"({len(results['schedule'])} quarterly rebalances, out of sample)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 unit")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/walkforward_cumulative_returns.png", dpi=150) #output#
    plt.close(fig)
    LOGGER.info("Saved walk-forward artifacts to %s/", output_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the FinOptix walk-forward backtest (quarterly rebalancing)."
    )
    parser.add_argument("--top-n", type=int, default=config.TOP_N_STOCKS)
    parser.add_argument("--confidence", type=float, default=config.VIEW_CONFIDENCE)
    parser.add_argument(
        "--cost-bps", type=float, default=config.COST_BPS,
        help="Basis points charged on turnover at each rebalance "
             f"(default: {config.COST_BPS}).",
    )
    parser.add_argument("--tickers-file", type=Path)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--use-fundamentals", action="store_true",
        help="Blend current fundamentals into selection. This is look-ahead "
             "bias in a backtest; off by default. See config.py.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    from main import configure_logging, load_tickers  # reuse, don't duplicate

    args = parse_args(argv)
    configure_logging()

    if args.top_n < 2:
        raise ValueError("--top-n must be at least 2 for covariance estimation.")
    if not (0 < args.confidence <= 1):
        raise ValueError("--confidence must be in the interval (0, 1].")
    if args.cost_bps < 0:
        raise ValueError("--cost-bps must be non-negative.")

    tickers = load_tickers(args.tickers_file)
    results = run_walkforward(
        tickers,
        top_n=args.top_n,
        confidence=args.confidence,
        cost_bps=args.cost_bps,
        use_cache=not args.refresh_cache,
        use_fundamentals=True if args.use_fundamentals else None,
    )
    save_results(results, config.OUTPUT_DIR)

    ic = results["pooled_ic"]
    LOGGER.info(
        "Pooled rank IC over %s non-overlapping windows: mean=%.4f std=%.4f "
        "t=%.2f hit_rate=%.2f",
        ic["n_periods"], ic["mean_ic"], ic["ic_std"],
        ic["ic_t_stat"] if np.isfinite(ic["ic_t_stat"]) else float("nan"),
        ic["hit_rate"],
    )
    LOGGER.info("Walk-forward performance:\n%s", results["summary"])
    LOGGER.info("Sharpe differences:\n%s", results["significance"])
    if results["optimizer_fallbacks"]:
        LOGGER.warning(
            "%s rebalance(s) fell back to equal weight because max-Sharpe had no "
            "asset above the risk-free rate.", results["optimizer_fallbacks"],
        )

    t_stat = ic["ic_t_stat"]
    if not np.isfinite(t_stat) or abs(t_stat) <= 2:
        LOGGER.warning(
            "Pooled rank IC t-stat is %.2f: even with %s observations the IC is "
            "not distinguishable from zero. Treat any outperformance above as "
            "noise, not skill.",
            t_stat if np.isfinite(t_stat) else float("nan"), ic["n_periods"],
        )
    else:
        LOGGER.warning(
            "Pooled rank IC t-stat is %.2f, which is unusually strong for daily "
            "technical features on cross-sectional equity returns. Treat this as "
            "a suspected leak in the walk-forward wiring until it is ruled out, "
            "not as a discovery.", t_stat,
        )
    return results


if __name__ == "__main__":
    main()
