"""
Central configuration for the Finoptix pipeline.
Change values here rather than editing pipeline code.
"""

from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
TICKERS = [
    "BHARTIARTL.NS", "LTIM.NS", "HDFCLIFE.NS", "NTPC.NS", "MARUTI.NS",
    "NESTLEIND.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "TATASTEEL.NS",
    "ONGC.NS", "BAJAJ-AUTO.NS", "LT.NS", "ITC.NS", "TCS.NS", "BRITANNIA.NS",
    "ADANIENT.NS", "CIPLA.NS", "WIPRO.NS", "INDUSINDBK.NS",
    "ULTRACEMCO.NS", "TATACONSUM.NS", "BAJAJFINSV.NS", "RELIANCE.NS",
    "HEROMOTOCO.NS", "COALINDIA.NS", "TITAN.NS", "HINDALCO.NS",
    "APOLLOHOSP.NS", "TECHM.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "BPCL.NS", "SBILIFE.NS", "GRASIM.NS", "JSWSTEEL.NS",
    "ASIANPAINT.NS", "POWERGRID.NS", "ADANIPORTS.NS", "M&M.NS",
    "TATAMOTORS.NS", "SUNPHARMA.NS", "AXISBANK.NS", "HCLTECH.NS",
    "HINDUNILVR.NS", "INFY.NS", "SBIN.NS", "ICICIBANK.NS", "HDFCBANK.NS",
]

# ---------------------------------------------------------------------------
# Date ranges
# Train window feeds the XGBoost model. Test window is held out, used both
# to evaluate the model (actual vs predicted) and as the return history for
# the Black-Litterman covariance / equilibrium step.
# ---------------------------------------------------------------------------
TODAY = date.today()
TRAIN_START = "2020-01-01"
TRAIN_END = (TODAY - timedelta(days=365)).isoformat()
TEST_START = (TODAY - timedelta(days=365)).isoformat()
TEST_END = TODAY.isoformat()

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "volatility_20", "ma_10", "ma_50", "momentum_10", "momentum_50",
    "upper_band", "lower_band", "returns_20", "corr_close_vol_20",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
]

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# ---------------------------------------------------------------------------
# Fundamental scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = dict(
    return_score=0.40,
    pe_score=0.20,
    de_score=0.20,
    mktcap_score=0.20,
)
TOP_N_STOCKS = 10

# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------
RISK_AVERSION = 2.5   # delta
TAU = 0.025            # scales prior uncertainty
VIEW_CONFIDENCE = 0.5  # 0 = no confidence (ignore ML views), 1 = full confidence

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
