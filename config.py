"""
Central configuration for the FinOptix pipeline.
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
# Date ranges: a strict three-way split.
#
#   TRAIN  fits the per-ticker models. Nothing else.
#   VALID  is the only window used to make decisions: it produces the
#          predictions that drive stock selection, the Black-Litterman views
#          (Q), and the covariance matrix. Every knob that gets turned is
#          turned here.
#   TEST   is backtest-only. Nothing reads it until the portfolio weights are
#          frozen. The moment a parameter is chosen by looking at TEST, TEST
#          stops being held out and the reported performance becomes fiction.
#
# The windows are half-open, [start, end), so a boundary date belongs to
# exactly one window. TRAIN is additionally purged (see PURGE_DAYS) because a
# forward-looking label near the end of TRAIN would otherwise peek into VALID.
# ---------------------------------------------------------------------------
TODAY = date.today()
TRAIN_START = "2020-01-01"
TRAIN_END = (TODAY - timedelta(days=730)).isoformat()
VALID_START = TRAIN_END
VALID_END = (TODAY - timedelta(days=365)).isoformat()
TEST_START = VALID_END
TEST_END = TODAY.isoformat()

# ---------------------------------------------------------------------------
# Prediction horizon
#
# The model predicts the return over the NEXT PRED_HORIZON trading days. The
# label at time t therefore spans (t, t + PRED_HORIZON], so the last
# PURGE_DAYS rows of any training slice carry labels that overlap the window
# that follows it. Those rows are dropped.
# ---------------------------------------------------------------------------
PRED_HORIZON = 21
PURGE_DAYS = PRED_HORIZON

# ---------------------------------------------------------------------------
# Feature engineering
#
# Every feature is scale-free (a ratio, a z-score or a correlation) and uses
# only information available at or before time t. Raw rupee price levels are
# deliberately absent: gradient-boosted trees cannot extrapolate past their
# training splits, so a stock trading outside its training price range lands
# in a boundary leaf and the model quietly stops working.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_21", "ret_63",
    "close_over_ma10", "close_over_ma50", "ma10_over_ma50",
    "vol_21", "vol_ratio", "band_pos", "rsi_14", "dist_52w_high",
    "volume_z", "corr_close_vol_20",
]

# ---------------------------------------------------------------------------
# XGBoost
#
# Heavily regularized on purpose. Daily cross-sectional equity returns carry
# very little signal; a depth-6, 500-tree model on this feature set memorizes
# noise and reports a training fit that does not survive contact with a
# held-out window.
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=3,
    learning_rate=0.02,
    min_child_weight=20,
    reg_lambda=5.0,
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
# Backtest
# ---------------------------------------------------------------------------
BENCHMARK_TICKER = "^NSEI"   # NIFTY 50
COST_BPS = 20.0              # one-off entry cost, basis points of gross exposure
TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
