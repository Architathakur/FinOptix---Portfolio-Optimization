"""
Walk-forward split integrity.

These are the assertions that make the whole scheme trustworthy. A walk-forward
backtest looks rigorous by construction, which is exactly why a quiet
off-by-one in the window arithmetic is so dangerous: the result still looks
plausible, and nothing else in the pipeline would notice.

So the checks here are deliberately strict. They materialize the actual date
sets through the same `window_slice` the pipeline uses -- not just the schedule
arithmetic -- and assert that the purge gap is present at BOTH inner
boundaries, that the forward-looking LABEL of the last row of each window
closes before the next window opens, and that no hold-window date appears
anywhere in that rebalance's training or validation data.
"""

import numpy as np
import pandas as pd
import pytest

import config
from config import PRED_HORIZON, PURGE_DAYS
from src.ml_returns import window_slice
from src.walkforward import (
    Rebalance,
    build_schedule,
    drifted_weights,
    hold_window_stream,
    turnover,
)

# Mirrors a real download: WALKFORWARD_WARMUP_DAYS of pre-history before
# WALKFORWARD_START, then through to today.
CALENDAR = pd.bdate_range("2018-07-02", "2026-09-04")


@pytest.fixture(scope="module")
def schedule() -> list[Rebalance]:
    built = build_schedule(CALENDAR)
    assert built, "no rebalances were scheduled"
    return built


@pytest.fixture(scope="module")
def calendar_frame() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(len(CALENDAR))}, index=CALENDAR)


def _gap(after, before) -> pd.DatetimeIndex:
    """Trading days strictly between two dates."""
    return CALENDAR[(CALENDAR > after) & (CALENDAR < before)]


def test_schedule_is_quarterly_and_long_enough(schedule):
    assert len(schedule) >= 15, f"only {len(schedule)} rebalances"
    dates = [rb.date for rb in schedule]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)
    # Quarterly: consecutive rebalances are roughly 3 months apart.
    spacing = pd.Series(dates).diff().dropna().dt.days
    assert spacing.between(80, 100).all(), f"non-quarterly spacing: {sorted(set(spacing))}"


def test_every_rebalance_is_strictly_purged(schedule, calendar_frame):
    """TRAIN.end < VALID.start < t, with an exact purge gap at both boundaries."""
    for rb in schedule:
        train = window_slice(calendar_frame, *rb.train).index
        valid = window_slice(calendar_frame, *rb.valid).index
        hold = window_slice(calendar_frame, *rb.hold).index

        assert len(train) and len(valid) and len(hold), f"empty window at {rb.date}"

        # --- ordering
        assert train[-1] < valid[0], f"TRAIN overruns VALID at {rb.date}"
        assert valid[-1] < rb.date, f"VALID overruns the rebalance date at {rb.date}"
        assert hold[0] >= rb.date, f"HOLD starts before the rebalance date at {rb.date}"

        # --- the purge gap is present, and is exactly PURGE_DAYS trading days
        assert len(_gap(train[-1], valid[0])) == PURGE_DAYS, (
            f"TRAIN/VALID purge gap at {rb.date} is "
            f"{len(_gap(train[-1], valid[0]))} trading days, expected {PURGE_DAYS}"
        )
        assert len(_gap(valid[-1], rb.date)) == PURGE_DAYS, (
            f"VALID/HOLD purge gap at {rb.date} is "
            f"{len(_gap(valid[-1], rb.date))} trading days, expected {PURGE_DAYS}"
        )

        # --- the reason the gap exists: the last label in each window must
        #     close before the next window opens.
        train_label_close = CALENDAR[CALENDAR.get_loc(train[-1]) + PRED_HORIZON]
        assert train_label_close < valid[0], (
            f"the last TRAIN label at {rb.date} closes on {train_label_close}, "
            f"inside VALID (opens {valid[0]})"
        )
        valid_label_close = CALENDAR[CALENDAR.get_loc(valid[-1]) + PRED_HORIZON]
        assert valid_label_close < rb.date, (
            f"the last VALID label at {rb.date} closes on {valid_label_close}, "
            f"at or after the rebalance date"
        )


def test_no_hold_date_appears_in_train_or_valid(schedule, calendar_frame):
    """The held window is untouched by anything that decided the weights."""
    for rb in schedule:
        train = set(window_slice(calendar_frame, *rb.train).index)
        valid = set(window_slice(calendar_frame, *rb.valid).index)
        hold = set(window_slice(calendar_frame, *rb.hold).index)

        assert hold.isdisjoint(train), (
            f"{len(hold & train)} hold dates leaked into TRAIN at {rb.date}"
        )
        assert hold.isdisjoint(valid), (
            f"{len(hold & valid)} hold dates leaked into VALID at {rb.date}"
        )
        assert min(hold) > max(train)
        assert min(hold) > max(valid)


def test_train_window_expands_from_a_fixed_start(schedule):
    starts = {pd.Timestamp(rb.train[0]) for rb in schedule}
    assert starts == {pd.Timestamp(config.WALKFORWARD_START)}, "TRAIN start moved"
    ends = [pd.Timestamp(rb.train[1]) for rb in schedule]
    assert ends == sorted(ends) and len(set(ends)) == len(ends), "TRAIN did not expand"


def test_hold_windows_tile_the_period_exactly_once(schedule, calendar_frame):
    """
    Stitching the hold windows must reproduce the calendar with no date used
    twice and none skipped -- otherwise the stitched return series either
    double-counts a day or silently drops one.
    """
    stitched = pd.DatetimeIndex(
        np.concatenate([window_slice(calendar_frame, *rb.hold).index.values
                        for rb in schedule])
    )
    assert stitched.is_unique, "a trading day is held by two rebalances"
    assert stitched.is_monotonic_increasing
    expected = CALENDAR[CALENDAR >= schedule[0].date]
    pd.testing.assert_index_equal(stitched, expected)


def test_training_slice_meets_the_minimum(schedule, calendar_frame):
    for rb in schedule:
        train = window_slice(calendar_frame, *rb.train).index
        assert len(train) >= 250, f"only {len(train)} training rows at {rb.date}"


# ---------------------------------------------------------------------------
# Turnover and drift
# ---------------------------------------------------------------------------
def test_turnover_is_one_when_entering_from_cash():
    new = pd.Series({"A": 0.6, "B": 0.4})
    assert turnover(new, pd.Series(dtype=float)) == pytest.approx(1.0)


def test_turnover_counts_both_sides_of_a_switch():
    held = pd.Series({"A": 1.0})
    new = pd.Series({"B": 1.0})
    # Sell all of A, buy all of B: one-way turnover of 2.0.
    assert turnover(new, held) == pytest.approx(2.0)


def test_turnover_is_zero_when_holding_the_drifted_book():
    """Charging against target rather than drifted weights would invent a trade."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [-0.10, 0.0, 0.0]}, index=dates)
    weights = pd.Series({"A": 0.5, "B": 0.5})

    drifted = drifted_weights(weights, returns)
    assert drifted["A"] > 0.5 > drifted["B"]          # the winner grew
    assert drifted.sum() == pytest.approx(1.0)
    assert turnover(drifted, drifted) == pytest.approx(0.0)


def test_rebalance_cost_is_charged_once_on_the_first_day():
    dates = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"A": [0.01] * 4, "B": [0.01] * 4}, index=dates)
    weights = pd.Series({"A": 0.5, "B": 0.5})

    gross = hold_window_stream(returns, weights, cost=0.0)
    net = hold_window_stream(returns, weights, cost=0.002)

    assert net.iloc[0] == pytest.approx(gross.iloc[0] - 0.002)
    pd.testing.assert_series_equal(net.iloc[1:], gross.iloc[1:])
