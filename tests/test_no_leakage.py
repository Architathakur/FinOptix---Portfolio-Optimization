"""
Regression guard against look-ahead bias.

The premise: feed the feature pipeline a pure random walk with iid normal
returns and a fixed seed. On such a series the true predictability of the
forward return is exactly zero -- not small, zero, by construction. So any
score meaningfully above zero cannot be signal. It can only be the model
reading the future through a feature that contains it.

This is the test the previous feature set fails. Its target was the return
ENDING at t while ma_10, ma_50, volatility_20 and the Bollinger bands all
included Close[t] in their windows, so the model was shown today's close and
asked for today's move. On this same random walk that setup scores about 0.32.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xg

from config import FEATURE_COLUMNS, PRED_HORIZON, PURGE_DAYS, XGB_PARAMS
from src.features import TARGET_COLUMN, calculate_features

# Long enough that the 21-day forward windows give a few hundred effectively
# independent observations; the standard error of a correlation over ~230
# non-overlapping blocks is roughly 0.065, so the 0.15 threshold sits at more
# than two standard errors and the test does not flap.
RANDOM_WALK_LEN = 8000
SEED = 42
LEAK_THRESHOLD = 0.15


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


def test_model_cannot_predict_a_random_walk(random_walk_features):
    """
    End to end: the configured model, trained on a purged 70/30 split of
    random-walk data, must not be able to predict the held-out forward return.
    """
    frame = random_walk_features
    split = int(len(frame) * 0.7)
    train = frame.iloc[: split - PURGE_DAYS]   # purge: labels overlap the split
    test = frame.iloc[split:]

    model = xg.XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN], verbose=False)
    predictions = model.predict(test[FEATURE_COLUMNS])

    correlation = _corr(predictions, test[TARGET_COLUMN])
    assert abs(correlation) < LEAK_THRESHOLD, (
        f"Model scored |corr| = {abs(correlation):.4f} on a random walk, where "
        f"the achievable correlation is zero. Something in the feature set or "
        f"the split is leaking the future."
    )
