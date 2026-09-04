"""
Per-stock expected-return model.

Structured in two stages, and the separation matters:

``build_panel`` computes features ONCE over the full, continuous price
history. The previous version built train features from one download and test
features from a second, shorter download. Every rolling window then warmed up
again inside the test slice -- and with a 252-day trailing-high feature on a
one-year test window, that leaves nothing usable at all. Features must be
computed on the unbroken series and sliced afterwards, never the reverse.

``train_predict`` fits one model per ticker on the training slice and predicts
over one or more named forward windows. The training slice is PURGED: its last
`purge` rows are dropped, because a label at time t covers (t, t + horizon] and
would otherwise overlap the window being predicted. Without the purge the model
trains on the first month of validation data.
"""

import logging

import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from config import FEATURE_COLUMNS, PRED_HORIZON, PURGE_DAYS, XGB_PARAMS
from src.features import TARGET_COLUMN, calculate_features

LOGGER = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 250


def _ticker_frame(price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Slice one ticker's price/volume columns out of a multi-ticker frame."""
    return pd.DataFrame(
        {
            "Close": price_data["Close"][ticker],
            "Volume": price_data["Volume"][ticker],
        }
    )


def _as_bound(value, index: pd.DatetimeIndex):
    """Coerce a date-ish bound to a Timestamp comparable with `index`."""
    if value is None:
        return None
    bound = pd.Timestamp(value)
    tz = getattr(index, "tz", None)
    if tz is not None and bound.tzinfo is None:
        bound = bound.tz_localize(tz)
    elif tz is None and bound.tzinfo is not None:
        bound = bound.tz_localize(None)
    return bound


def window_slice(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    """
    Half-open [start, end) slice by date.

    Half-open, not pandas' inclusive `.loc[start:end]`, so that adjacent
    windows sharing a boundary date (TRAIN_END == VALID_START) do not both
    claim it.
    """
    index = frame.index
    mask = pd.Series(True, index=index)
    start = _as_bound(start, index)
    end = _as_bound(end, index)
    if start is not None:
        mask &= index >= start
    if end is not None:
        mask &= index < end
    return frame.loc[mask.to_numpy()]


def build_panel(price_data: pd.DataFrame, tickers, horizon: int = PRED_HORIZON) -> dict:
    """
    Compute features once per ticker over the whole price history.

    Parameters
    ----------
    price_data : wide OHLCV frame with MultiIndex (field, ticker) columns,
                 covering the full TRAIN..TEST span in one continuous series.
    tickers    : tickers to build; those absent from `price_data` are skipped.
    horizon    : forward return horizon passed to calculate_features.

    Returns
    -------
    dict of {ticker: features DataFrame}
    """
    panel = {}
    for ticker in tickers:
        try:
            frame = _ticker_frame(price_data, ticker)
        except KeyError:
            # Ticker missing from the download (delisted / bad symbol)
            LOGGER.warning("No price columns for %s; skipping.", ticker)
            continue
        frame = frame.sort_index()
        if frame["Close"].notna().sum() < 2:
            LOGGER.warning("Insufficient close prices for %s; skipping.", ticker)
            continue
        panel[ticker] = calculate_features(frame, horizon=horizon)
    return panel


def purged_train_slice(frame: pd.DataFrame, train_range, purge: int = PURGE_DAYS,
                       columns=None) -> pd.DataFrame:
    """
    The training rows for one ticker: complete features AND label, restricted
    to `train_range`, with the final `purge` rows removed.

    The purge is the whole point. The label at the last training date spans the
    following `horizon` trading days, which lie inside the next window. Keeping
    those rows leaks the start of validation into training.
    """
    start, end = train_range
    columns = list(columns) if columns is not None else FEATURE_COLUMNS
    usable = frame.dropna(subset=columns + [TARGET_COLUMN])
    train = window_slice(usable, start, end)
    if purge > 0:
        train = train.iloc[:-purge] if len(train) > purge else train.iloc[0:0]
    return train


def train_predict(panel: dict, train_range, predict_ranges: dict,
                  purge: int = PURGE_DAYS, params: dict | None = None,
                  min_train_rows: int = MIN_TRAIN_ROWS):
    """
    Fit one model per ticker on `train_range`, predict over each named window.

    Parameters
    ----------
    panel          : {ticker: features DataFrame} from build_panel
    train_range    : (start, end) half-open training window
    predict_ranges : {name: (start, end)}, e.g.
                     {"valid": (VALID_START, VALID_END), "test": (...)}
    purge          : rows dropped from the end of the training slice
    min_train_rows : tickers with fewer usable training rows are skipped

    Returns
    -------
    predictions : {window name: DataFrame (date x ticker)} of predicted
                  `horizon`-day forward returns
    actuals     : {window name: DataFrame (date x ticker)} of realized
                  `horizon`-day forward returns, aligned to `predictions`
                  (NaN in the last `horizon` rows, where the forward window
                  has not closed yet)
    metrics     : DataFrame indexed by ticker
    """
    params = dict(params or XGB_PARAMS)
    predictions = {name: {} for name in predict_ranges}
    actuals = {name: {} for name in predict_ranges}
    metric_rows = {}
    skipped = []

    for ticker in tqdm(sorted(panel), desc="Training per-ticker XGBoost models"):
        frame = panel[ticker]
        train_df = purged_train_slice(frame, train_range, purge=purge)

        if len(train_df) < min_train_rows:
            LOGGER.warning(
                "Skipping %s: only %s usable training rows after purge (need %s).",
                ticker, len(train_df), min_train_rows,
            )
            skipped.append(ticker)
            continue

        model = xg.XGBRegressor(**params)
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], verbose=False)

        row = {"n_train": len(train_df), "train_end": train_df.index[-1]}
        for name, (start, end) in predict_ranges.items():
            # Features must be complete; the label may not be (the tail of the
            # final window has no closed forward return yet).
            window = window_slice(frame.dropna(subset=FEATURE_COLUMNS), start, end)
            if window.empty:
                LOGGER.warning("No usable %s rows for %s.", name, ticker)
                row[f"{name}_corr"] = np.nan
                row[f"{name}_rmse"] = np.nan
                continue

            preds = pd.Series(model.predict(window[FEATURE_COLUMNS]), index=window.index)
            realized = window[TARGET_COLUMN]
            predictions[name][ticker] = preds
            actuals[name][ticker] = realized

            both = pd.concat([preds, realized], axis=1).dropna()
            if len(both) > 1 and both.iloc[:, 0].std() > 0 and both.iloc[:, 1].std() > 0:
                row[f"{name}_corr"] = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
                row[f"{name}_rmse"] = float(
                    np.sqrt(mean_squared_error(both.iloc[:, 1], both.iloc[:, 0]))
                )
            else:
                row[f"{name}_corr"] = np.nan
                row[f"{name}_rmse"] = np.nan

        metric_rows[ticker] = row

    if skipped:
        LOGGER.warning("Skipped %s ticker(s) for insufficient history: %s",
                       len(skipped), ", ".join(skipped))

    predictions = {name: pd.DataFrame(cols) for name, cols in predictions.items()}
    actuals = {name: pd.DataFrame(cols) for name, cols in actuals.items()}
    metrics = pd.DataFrame(metric_rows).T
    return predictions, actuals, metrics
