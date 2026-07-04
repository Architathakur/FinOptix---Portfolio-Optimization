"""
Feature engineering for the per-stock return prediction model.
"""

import pandas as pd

from config import FEATURE_COLUMNS


def calculate_features(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """
    Build technical features from a single ticker's OHLCV frame.

    Parameters
    ----------
    df : DataFrame with columns ['Close', 'High', 'Low', 'Volume']
    horizon : lookback window (in trading days) used for momentum features

    Returns
    -------
    DataFrame indexed like `df`, with a 'returns' target column plus all
    columns listed in config.FEATURE_COLUMNS.
    """
    feat = pd.DataFrame(index=df.index)
    feat["Close"] = df["Close"]

    feat["returns"] = df["Close"].pct_change()
    feat["returns_20"] = df["Close"].rolling(20).mean().pct_change()
    feat["volatility_20"] = df["Close"].rolling(20).std()
    feat["ma_10"] = df["Close"].rolling(10).mean()
    feat["ma_50"] = df["Close"].rolling(50).mean()
    feat["momentum_10"] = df["Close"].rolling(10).mean().pct_change(horizon)
    feat["momentum_50"] = df["Close"].rolling(50).mean().pct_change(horizon)
    ma_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    feat["upper_band"] = ma_20 + 2 * std_20
    feat["lower_band"] = ma_20 - 2 * std_20
    feat["corr_close_vol_20"] = df["Close"].rolling(20).corr(df["Volume"])

    for lag in (1, 2, 3, 5):
        feat[f"return_lag_{lag}"] = feat["returns"].shift(lag)

    missing = [c for c in FEATURE_COLUMNS if c not in feat.columns]
    assert not missing, f"calculate_features is missing expected columns: {missing}"

    return feat
