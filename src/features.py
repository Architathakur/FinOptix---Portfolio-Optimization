"""
Feature engineering for the per-stock return prediction model.

Two rules govern this module, and violating either one silently invalidates
every number the pipeline downstream produces:

1. A feature evaluated at time t may use price and volume data at or before
   t, and nothing else. No centred windows, no forward shifts, no rolling
   statistic whose window straddles t.
2. The label is strictly forward-looking: the return over (t, t + horizon].

The previous version broke both at once. It set the target to
``Close.pct_change()`` -- the return ENDING at t -- while ma_10, ma_50,
volatility_20 and the Bollinger bands all contained Close[t] in their rolling
windows. The model was handed today's close and asked to predict today's move,
which is why it reported prediction correlations of 0.21-0.52 where a real
daily-return model scores about 0.02-0.05. See tests/test_no_leakage.py for
the regression guard: on a pure random walk, where predictability is exactly
zero by construction, the old setup scored ~0.32 and this one scores ~0.00.

Every feature here is also scale-free. Raw price levels (ma_10, upper_band,
volatility_20 in rupees) are unusable for trees, which cannot extrapolate
beyond the splits they saw in training: a stock trading above its entire
training range falls into the topmost leaf and stays there.
"""

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, PRED_HORIZON

TARGET_COLUMN = "target"
FORWARD_1D_COLUMN = "ret_fwd_1"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, mapping a zero (or non-finite) denominator to NaN rather than inf."""
    denominator = denominator.where(np.isfinite(denominator) & (denominator != 0))
    return numerator / denominator


def calculate_features(df: pd.DataFrame, horizon: int = PRED_HORIZON) -> pd.DataFrame:
    """
    Build technical features from a single ticker's price/volume frame.

    Parameters
    ----------
    df : DataFrame with (at least) columns ['Close', 'Volume'], indexed by date
         in ascending order.
    horizon : forward return horizon in trading days, used for the target.

    Returns
    -------
    DataFrame indexed like `df`, containing:
      - 'Close'          : carried through for convenience / debugging
      - every column in config.FEATURE_COLUMNS, all backward-looking
      - 'target'         : close[t + horizon] / close[t] - 1
      - 'ret_fwd_1'      : close[t + 1] / close[t] - 1, for daily backtest
                           accounting
    Leading rows are NaN while the rolling windows warm up (the 252-day
    trailing high is the binding constraint), and the trailing `horizon` rows
    have no target. Callers drop them.
    """
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    volume = pd.to_numeric(df["Volume"], errors="coerce").astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["Close"] = close

    # --- trailing returns over several speeds --------------------------
    ret_1 = close.pct_change()
    feat["ret_1"] = ret_1
    feat["ret_5"] = close.pct_change(5)
    feat["ret_21"] = close.pct_change(21)
    feat["ret_63"] = close.pct_change(63)

    # --- price relative to its own moving averages ---------------------
    # Ratios, not levels: "5% above the 10-day mean" means the same thing at
    # any price, which is exactly what a tree split needs.
    ma10 = close.rolling(10).mean()
    ma50 = close.rolling(50).mean()
    feat["close_over_ma10"] = _safe_divide(close, ma10) - 1
    feat["close_over_ma50"] = _safe_divide(close, ma50) - 1
    feat["ma10_over_ma50"] = _safe_divide(ma10, ma50) - 1

    # --- realized volatility and its term structure --------------------
    vol_21 = ret_1.rolling(21).std()
    vol_63 = ret_1.rolling(63).std()
    feat["vol_21"] = vol_21
    feat["vol_ratio"] = _safe_divide(vol_21, vol_63) - 1

    # --- position inside the Bollinger channel -------------------------
    # 0 = on the lower band, 1 = on the upper band. Dividing by the channel
    # width is what makes this comparable across stocks; a flat price series
    # gives width 0, hence the guard.
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    lower_band = ma20 - 2 * sd20
    feat["band_pos"] = _safe_divide(close - lower_band, 4 * sd20)

    # --- Wilder's RSI --------------------------------------------------
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    feat["rsi_14"] = 100 - 100 / (1 + rs)

    # --- drawdown from the trailing 52-week high -----------------------
    feat["dist_52w_high"] = _safe_divide(close, close.rolling(252).max()) - 1

    # --- volume surprise and price/volume co-movement ------------------
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat["volume_z"] = _safe_divide(volume - vol_mean_20, vol_std_20)
    feat["corr_close_vol_20"] = close.rolling(20).corr(volume)

    # --- labels (the only forward-looking columns) ---------------------
    # pct_change(h).shift(-h) at t is close[t + h] / close[t] - 1.
    feat[TARGET_COLUMN] = close.pct_change(horizon).shift(-horizon)
    feat[FORWARD_1D_COLUMN] = close.pct_change().shift(-1)

    feat = feat.replace([np.inf, -np.inf], np.nan)

    missing = [c for c in FEATURE_COLUMNS if c not in feat.columns]
    assert not missing, f"calculate_features is missing expected columns: {missing}"

    return feat
