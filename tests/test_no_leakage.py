"""
Regression guard against look-ahead bias.

The premise: feed the feature pipeline a pure random walk with iid normal
returns and a fixed seed. On such a series the true predictability of the
forward return is exactly zero -- not small, zero, by construction. So any
score meaningfully above zero cannot be signal. It can only be the model
reading the future through a feature that contains it.

This is the test the previous feature set fails, and that failure is asserted
here rather than described: `old_calculate_features` below is a working
reconstruction of the original buggy pipeline, kept as a positive control.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xg

from config import FEATURE_COLUMNS, PRED_HORIZON, PURGE_DAYS, XGB_PARAMS
from src.features import TARGET_COLUMN, calculate_features

# Long enough that the 21-day forward windows give a few hundred effectively
# independent observations.
RANDOM_WALK_LEN = 8000
SEED = 42
SEEDS = (42, 7, 1234, 20240115, 99)
LEAK_THRESHOLD = 0.15
MAX_LEAK_RATIO = 0.25       # new feature set must score below 25% of the leaky one
CONTROL_MIN_SCORE = 0.20    # ...and the leaky control must actually leak

# ---------------------------------------------------------------------------
# Positive control: the ORIGINAL, BUGGY feature pipeline.
#
# Do not "fix" any of this. It is a frozen copy of the construction the current
# code replaced, and its entire purpose is to score badly. If it ever stops
# leaking, the guard below silently loses its reference point -- which is why
# the test asserts that it still leaks before comparing anything to it.
# ---------------------------------------------------------------------------
OLD_TARGET_COLUMN = "returns"
OLD_FEATURE_COLUMNS = [
    "volatility_20", "ma_10", "ma_50", "momentum_10", "momentum_50",
    "upper_band", "lower_band", "returns_20", "corr_close_vol_20",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
]
OLD_XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


def old_calculate_features(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """
    The original src/features.py, reconstructed verbatim.

    The target is the return ENDING at t, while ma_10, ma_50, volatility_20
    and both Bollinger bands are rolling windows that CONTAIN Close[t]. The
    model is therefore shown today's close and asked for today's move.
    """
    close = df["Close"]
    feat = pd.DataFrame(index=df.index)
    feat["returns"] = close.pct_change()                       # <- same-day target
    feat["returns_20"] = close.rolling(20).mean().pct_change()
    feat["volatility_20"] = close.rolling(20).std()            # <- contains Close[t]
    feat["ma_10"] = close.rolling(10).mean()                   # <- contains Close[t]
    feat["ma_50"] = close.rolling(50).mean()                   # <- contains Close[t]
    feat["momentum_10"] = close.rolling(10).mean().pct_change(horizon)
    feat["momentum_50"] = close.rolling(50).mean().pct_change(horizon)
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    feat["upper_band"] = ma_20 + 2 * std_20                    # <- contains Close[t]
    feat["lower_band"] = ma_20 - 2 * std_20                    # <- contains Close[t]
    feat["corr_close_vol_20"] = close.rolling(20).corr(df["Volume"])
    for lag in (1, 2, 3, 5):
        feat[f"return_lag_{lag}"] = feat["returns"].shift(lag)
    return feat


def random_walk_frame(n: int = RANDOM_WALK_LEN, seed: int = SEED) -> pd.DataFrame:
    """A price/volume series with zero predictable structure."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
    volume = rng.lognormal(13.0, 0.35, n)
    return pd.DataFrame(
        {"Close": close, "Volume": volume},
        index=pd.bdate_range("2000-01-03", periods=n),
    )


def _corr(a, b) -> float:
    """Pearson correlation, treating a degenerate (constant) series as zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _score(frame, features, target, params, purge) -> float:
    """|corr| between predictions and target on a purged 70/30 split."""
    frame = frame.dropna(subset=list(features) + [target])
    split = int(len(frame) * 0.7)
    train = frame.iloc[: split - purge] if purge else frame.iloc[:split]
    test = frame.iloc[split:]
    model = xg.XGBRegressor(**params)
    model.fit(train[features], train[target], verbose=False)
    return abs(_corr(model.predict(test[features]), test[target]))


@pytest.fixture(scope="module")
def random_walk_features() -> pd.DataFrame:
    frame = calculate_features(random_walk_frame())
    return frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])


def test_no_feature_is_correlated_with_the_future(random_walk_features):
    """No individual feature may know where an unpredictable series is going."""
    offenders = {
        column: _corr(random_walk_features[column], random_walk_features[TARGET_COLUMN])
        for column in FEATURE_COLUMNS
    }
    leaking = {c: round(v, 4) for c, v in offenders.items() if abs(v) >= LEAK_THRESHOLD}
    assert not leaking, (
        f"Features correlated with the forward return on a random walk "
        f"(|corr| >= {LEAK_THRESHOLD}): {leaking}. A feature at time t is "
        f"reading data from after t."
    )


def test_target_is_the_forward_return():
    """target[t] == close[t + PRED_HORIZON] / close[t] - 1, exactly."""
    frame = random_walk_frame(n=400)
    feat = calculate_features(frame)
    close = frame["Close"].to_numpy(dtype=float)

    expected = close[PRED_HORIZON:] / close[:-PRED_HORIZON] - 1
    np.testing.assert_allclose(
        feat[TARGET_COLUMN].to_numpy(dtype=float)[:-PRED_HORIZON],
        expected,
        rtol=1e-12,
        atol=0,
    )
    # The last horizon rows cannot have a label: the window has not closed.
    assert feat[TARGET_COLUMN].iloc[-PRED_HORIZON:].isna().all()
    # And the daily accounting return is genuinely one day forward.
    np.testing.assert_allclose(
        feat["ret_fwd_1"].to_numpy(dtype=float)[:-1],
        close[1:] / close[:-1] - 1,
        rtol=1e-12,
        atol=0,
    )


def test_new_feature_set_scores_far_below_the_leaky_original():
    """
    End to end: the current pipeline must score far below the original leaky
    one, measured on the same random walks in the same process.

    Why a ratio against a live control rather than an absolute threshold.
    An absolute bound like "|corr| < 0.15" is only valid at one series length.
    The new set's score is a noise floor, and a noise floor grows as the sample
    shrinks: at 8000 days it sits near 0.04, but at 1650 days one seed reaches
    0.196 -- still pure noise, still nothing to fix, yet it would trip a fixed
    0.15 bound. A guard that fails when you shorten its input is a guard that
    gets deleted rather than debugged.

    The leaky control does not have that problem. It scores 0.25-0.44 at every
    length tested, because it is measuring a real dependence rather than
    sampling noise. Dividing by it gives a quantity that stays stable where the
    raw number does not, and it makes the assertion say what we actually mean:
    not "this number is small" but "this construction is nothing like the one
    that was broken". It also survives dependency drift -- if a pandas or
    xgboost change altered rolling semantics, both sides move together and the
    comparison still holds, where a hardcoded constant would quietly become the
    wrong constant.

    Averaging across five seeds is the second half of it. A single seed's noise
    floor is itself noisy (0.008 to 0.123 across these five), so the mean is
    what makes the ratio comfortable rather than marginal -- without having to
    derive a standard error from the effective sample size, which is the
    fragility being removed in the first place.
    """
    old_original, old_regularized, new = [], [], []
    for seed in SEEDS:
        walk = random_walk_frame(seed=seed)
        old_frame = old_calculate_features(walk)
        new_frame = calculate_features(walk)

        old_original.append(
            _score(old_frame, OLD_FEATURE_COLUMNS, OLD_TARGET_COLUMN, OLD_XGB_PARAMS, purge=0)
        )
        old_regularized.append(
            _score(old_frame, OLD_FEATURE_COLUMNS, OLD_TARGET_COLUMN, XGB_PARAMS, purge=0)
        )
        new.append(
            _score(new_frame, FEATURE_COLUMNS, TARGET_COLUMN, XGB_PARAMS, purge=PURGE_DAYS)
        )

    old_mean = float(np.mean(old_original))
    old_reg_mean = float(np.mean(old_regularized))
    new_mean = float(np.mean(new))

    # The control has to actually leak, or the ratio below means nothing.
    assert old_mean > CONTROL_MIN_SCORE, (
        f"The leaky control scored only {old_mean:.4f} on a random walk. It is "
        f"supposed to leak; if it no longer does, this guard has lost its "
        f"reference point and the comparison below is vacuous."
    )
    # ...and it has to leak for the reason we claim: the features, not the
    # hyperparameters. Re-running it under the current regularized params must
    # not rescue it.
    assert old_reg_mean > CONTROL_MIN_SCORE, (
        f"The leaky feature set scored {old_reg_mean:.4f} under the current "
        f"regularized hyperparameters. If regularization alone fixes it, the "
        f"diagnosis recorded in README.md is wrong."
    )

    assert new_mean < old_mean * MAX_LEAK_RATIO, (
        f"Current feature set scored {new_mean:.4f} on random walks, against "
        f"{old_mean:.4f} for the known-leaky original -- a ratio of "
        f"{new_mean / old_mean:.3f}, above the {MAX_LEAK_RATIO} limit. The "
        f"current pipeline is behaving too much like the one with look-ahead "
        f"bias.\n"
        f"  per-seed new: {[round(v, 4) for v in new]}\n"
        f"  per-seed old: {[round(v, 4) for v in old_original]}"
    )
