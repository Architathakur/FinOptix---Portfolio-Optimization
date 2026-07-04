"""
Per-stock expected-return model.

Trains one XGBRegressor per ticker on the training window, then predicts
daily returns over the held-out test window. This avoids the original
notebook's bug of re-downloading test data inside the training loop -
here train and test prices are fetched once, up front.
"""

import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from config import FEATURE_COLUMNS, XGB_PARAMS
from src.features import calculate_features


def _ticker_frame(price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Slice one ticker's OHLCV columns out of a multi-ticker price frame."""
    return pd.DataFrame(
        {
            "Close": price_data["Close"][ticker],
            "High": price_data["High"][ticker],
            "Low": price_data["Low"][ticker],
            "Volume": price_data["Volume"][ticker],
        }
    )


def train_predict_all(train_prices, test_prices, tickers):
    """
    Fit one model per ticker on train_prices, predict returns on test_prices.

    Returns
    -------
    actual_returns, expected_returns : DataFrames (index=date, columns=tickers)
    metrics : DataFrame with correlation and RMSE per ticker
    """
    actual_returns = pd.DataFrame()
    expected_returns = pd.DataFrame()
    metric_rows = {}

    for ticker in tqdm(tickers, desc="Training per-ticker XGBoost models"):
        try:
            train_df = calculate_features(_ticker_frame(train_prices, ticker)).dropna()
            test_df = calculate_features(_ticker_frame(test_prices, ticker)).dropna()
        except KeyError:
            # Ticker missing from the download (e.g. delisted / bad symbol)
            continue

        if len(train_df) < 60 or len(test_df) < 20:
            # Not enough history to train/evaluate reliably - skip
            continue

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df["returns"]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df["returns"]

        model = xg.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        preds = pd.Series(model.predict(X_test), index=test_df.index)

        actual_returns[ticker] = y_test
        expected_returns[ticker] = preds

        aligned_actual, aligned_pred = y_test.align(preds, join="inner")
        if len(aligned_actual) > 1:
            corr = np.corrcoef(aligned_actual, aligned_pred)[0, 1]
            rmse = np.sqrt(mean_squared_error(aligned_actual, aligned_pred))
        else:
            corr, rmse = np.nan, np.nan
        metric_rows[ticker] = {"correlation": corr, "rmse": rmse}

    metrics = pd.DataFrame(metric_rows).T
    return actual_returns, expected_returns, metrics
