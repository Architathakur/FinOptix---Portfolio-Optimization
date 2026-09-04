"""
Evaluation metrics that match what the strategy actually does.

Two things live here: the information coefficient (does the model rank stocks
correctly?) and a bootstrap test for whether a Sharpe ratio difference between
two portfolios is distinguishable from luck.
"""

import numpy as np
import pandas as pd

from config import PRED_HORIZON, TRADING_DAYS

MIN_NAMES_PER_DATE = 5


def rank_ic(predictions: pd.DataFrame, actuals: pd.DataFrame,
            horizon: int = PRED_HORIZON, min_names: int = MIN_NAMES_PER_DATE) -> dict:
    """
    Cross-sectional rank information coefficient.

    For each sampled date, the Spearman correlation between predicted and
    realized forward returns ACROSS TICKERS; averaged over dates.

    Why this metric rather than per-ticker time-series correlation: a
    per-ticker correlation answers a question the strategy never asks. The
    strategy does not trade one stock against its own history -- it ranks the
    universe against itself on a given day and buys the top of that ranking.
    The IC is the metric that matches that decision, and it is what the field
    reports, which makes the number comparable to published results.

    Dates are sampled every `horizon` rows so the forward windows do not
    overlap. Overlapping windows share most of their price path, which
    autocorrelates the IC series; averaging still works but the standard error
    collapses and the t-stat inflates by roughly sqrt(horizon). A t-stat built
    from overlapping daily observations of a 21-day return is not a t-stat.

    Parameters
    ----------
    predictions, actuals : DataFrames (date x ticker). Realized returns are
        expected to be forward returns over the same `horizon`.
    horizon   : sampling stride, in rows, matching the label horizon.
    min_names : dates with fewer paired tickers than this are dropped; a
        cross-sectional correlation over 2-3 names is noise.

    Returns
    -------
    dict with n_periods, mean_ic, ic_std, ic_t_stat, hit_rate, series
    """
    empty = {
        "n_periods": 0, "mean_ic": np.nan, "ic_std": np.nan,
        "ic_t_stat": np.nan, "hit_rate": np.nan,
        "series": pd.Series(dtype=float),
    }
    if predictions is None or actuals is None or predictions.empty or actuals.empty:
        return empty

    common_cols = predictions.columns.intersection(actuals.columns)
    common_idx = predictions.index.intersection(actuals.index).sort_values()
    if len(common_cols) < min_names or len(common_idx) == 0:
        return empty

    preds = predictions.loc[common_idx, common_cols]
    real = actuals.loc[common_idx, common_cols]

    stride = max(int(horizon), 1)
    sampled = common_idx[::stride]

    ic_values = {}
    for date in sampled:
        pair = pd.concat(
            [preds.loc[date].rename("pred"), real.loc[date].rename("actual")],
            axis=1,
        ).dropna()
        if len(pair) < min_names:
            continue
        if pair["pred"].nunique() < 2 or pair["actual"].nunique() < 2:
            continue
        ic = pair["pred"].corr(pair["actual"], method="spearman")
        if np.isfinite(ic):
            ic_values[date] = float(ic)

    series = pd.Series(ic_values, dtype=float).sort_index()
    n = len(series)
    if n == 0:
        return empty

    mean_ic = float(series.mean())
    ic_std = float(series.std(ddof=1)) if n > 1 else np.nan
    t_stat = mean_ic / ic_std * np.sqrt(n) if n > 1 and ic_std and ic_std > 0 else np.nan

    return {
        "n_periods": n,
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "hit_rate": float((series > 0).mean()),
        "series": series,
    }


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio of a simple-return series, rf = 0."""
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    sd = returns.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def sharpe_difference_test(returns_a: pd.Series, returns_b: pd.Series,
                           n_boot: int = 5000, seed: int = 42,
                           periods_per_year: int = TRADING_DAYS) -> dict:
    """
    Bootstrap test for Sharpe(a) - Sharpe(b).

    Dates are resampled JOINTLY -- the same row indices are drawn from both
    series -- so the contemporaneous correlation between the two portfolios is
    preserved. Resampling them independently would destroy that correlation and
    inflate the variance of the difference, which is the whole quantity under
    test: two portfolios holding overlapping names move together, and only the
    part of the difference that survives that shared movement is evidence.

    Returns
    -------
    dict with observed_diff, ci_low, ci_high (2.5/97.5 percentiles), p_value
    (two-sided, from the mean-centred bootstrap distribution), n_obs, n_boot.
    """
    joined = pd.concat(
        [pd.Series(returns_a).rename("a"), pd.Series(returns_b).rename("b")],
        axis=1, join="inner",
    ).dropna()

    result = {
        "observed_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "p_value": np.nan, "n_obs": len(joined), "n_boot": int(n_boot),
    }
    if len(joined) < 3:
        return result

    a = joined["a"].to_numpy(dtype=float)
    b = joined["b"].to_numpy(dtype=float)
    scale = np.sqrt(periods_per_year)

    def _sharpe(x: np.ndarray, axis=-1):
        sd = x.std(axis=axis, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(sd > 0, x.mean(axis=axis) / sd * scale, np.nan)
        return sr

    observed = float(_sharpe(a) - _sharpe(b))
    result["observed_diff"] = observed
    if not np.isfinite(observed):
        return result

    rng = np.random.default_rng(seed)
    n = len(joined)
    idx = rng.integers(0, n, size=(int(n_boot), n))  # same draws for a and b
    diffs = _sharpe(a[idx], axis=1) - _sharpe(b[idx], axis=1)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return result

    result["ci_low"] = float(np.percentile(diffs, 2.5))
    result["ci_high"] = float(np.percentile(diffs, 97.5))
    # Centre the bootstrap distribution on zero to get a null distribution for
    # "no difference", then ask how often it is at least as extreme as observed.
    centred = diffs - diffs.mean()
    result["p_value"] = float(np.mean(np.abs(centred) >= abs(observed)))
    return result
